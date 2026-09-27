//! What the daemon last put on a checkout beyond its base commit, and how to
//! tell cheaply that it is still there.
//!
//! [`materialize`](super::materialize) records, for every path a
//! conversation changed, the content it wrote and the [`FileStamp`] the file
//! had afterwards; [`capture`](super::capture) re-records each path it reads
//! back. A later pass asks [`Ledger::verified`]: when the file's stamp is the
//! recorded one and not racy, the recorded content is what the file holds and
//! the file is never opened. That is what lets a pass over a checkout touch
//! only the files that actually changed — the property a build cache over the
//! checkout depends on.
//!
//! Every path the ledger does not name holds its base commit's content, or is
//! absent when the base has none: the ledger is exactly the checkout's
//! deviation from its base.

use std::collections::BTreeMap;

use base64::engine::general_purpose::STANDARD as BASE64;
use base64::Engine as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use super::stamp::FileStamp;
use crate::file_delta::now_ns;

/// One path the checkout holds differently from its base.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Entry {
    /// The file's bytes, or `None` when the path is absent. Base64 on the
    /// wire.
    #[serde(serialize_with = "to_base64", deserialize_with = "from_base64")]
    pub content: Option<Vec<u8>>,
    /// The file's stamp when `content` was recorded, `None` for an absent
    /// path.
    pub stamp: Option<FileStamp>,
}

/// A checkout's deviation from its base commit, stamped.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ledger {
    /// The commit the checkout's tracked files were put at.
    base: String,
    entries: BTreeMap<String, Entry>,
    /// When the stamps were last taken, nanoseconds since the Unix epoch — the
    /// moment a stamp's times are judged racy against.
    stamped_at_ns: i64,
}

impl Ledger {
    /// A ledger for a checkout at `base` with nothing on it beyond the base.
    pub fn new(base: impl Into<String>) -> Self {
        Self {
            base: base.into(),
            entries: BTreeMap::new(),
            stamped_at_ns: now_ns(),
        }
    }

    /// The commit the checkout's tracked files are at.
    pub fn base(&self) -> &str {
        &self.base
    }

    pub fn entry(&self, path: &str) -> Option<&Entry> {
        self.entries.get(path)
    }

    /// Every path the checkout holds differently from its base, in order.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// What `path` holds, known without reading it: the recorded content when
    /// the file's current stamp is the recorded one and not racy. `None` when
    /// the ledger cannot vouch — the path is not in it, the stamp differs, or
    /// it was taken too close to a write to tell — and the file must be read.
    pub fn verified(&self, path: &str, current: Option<FileStamp>) -> Option<Option<&[u8]>> {
        let entry = self.entries.get(path)?;
        match (entry.stamp, current) {
            (None, None) => Some(None),
            (Some(recorded), Some(now))
                if recorded == now && !recorded.is_racy(self.stamped_at_ns) =>
            {
                Some(entry.content.as_deref())
            }
            _ => None,
        }
    }

    /// Record `path` as holding `content` with `stamp`.
    pub fn record(
        &mut self,
        path: impl Into<String>,
        content: Option<Vec<u8>>,
        stamp: Option<FileStamp>,
    ) {
        self.entries.insert(path.into(), Entry { content, stamp });
    }

    /// Forget `path` — it holds its base's content again.
    pub fn forget(&mut self, path: &str) {
        self.entries.remove(path);
    }

    /// Mark the stamps as taken now. Called once a pass has recorded every
    /// path it touched, so the racy window is measured from after the last
    /// write it made.
    pub fn seal(&mut self) {
        self.stamped_at_ns = now_ns();
    }

    /// Shift the moment the stamps count as taken — for tests that need a
    /// stamp old enough to be trusted without sleeping out the racy window.
    #[cfg(test)]
    pub(crate) fn seal_at(&mut self, ns: i64) {
        self.stamped_at_ns = ns;
    }
}

fn to_base64<S: Serializer>(bytes: &Option<Vec<u8>>, s: S) -> Result<S::Ok, S::Error> {
    match bytes {
        Some(b) => s.serialize_some(&BASE64.encode(b)),
        None => s.serialize_none(),
    }
}

fn from_base64<'de, D: Deserializer<'de>>(d: D) -> Result<Option<Vec<u8>>, D::Error> {
    Option::<String>::deserialize(d)?
        .map(|text| BASE64.decode(text).map_err(serde::de::Error::custom))
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkout::stamp::RACY_WINDOW_NS;

    fn stamp(t: i64) -> FileStamp {
        FileStamp {
            size: 3,
            mtime_ns: t,
        }
    }

    /// **A matching, settled stamp vouches for the content; anything else
    /// sends the caller to read the file.**
    #[test]
    fn verification_needs_a_matching_settled_stamp() {
        let mut l = Ledger::new("abc");
        l.record("a.txt", Some(b"abc".to_vec()), Some(stamp(1_000)));
        l.record("gone.txt", None, None);
        l.seal_at(1_000 + RACY_WINDOW_NS * 10);

        assert_eq!(
            l.verified("a.txt", Some(stamp(1_000))),
            Some(Some(&b"abc"[..]))
        );
        assert_eq!(
            l.verified("a.txt", Some(stamp(1_001))),
            None,
            "a different stamp"
        );
        assert_eq!(l.verified("a.txt", None), None, "the file is gone");
        assert_eq!(l.verified("gone.txt", None), Some(None), "still absent");
        assert_eq!(l.verified("gone.txt", Some(stamp(5))), None, "it came back");
        assert_eq!(
            l.verified("other.txt", Some(stamp(1_000))),
            None,
            "not recorded"
        );

        // Taken too close to the write, the same stamp vouches for nothing.
        l.seal_at(1_000 + RACY_WINDOW_NS / 2);
        assert_eq!(l.verified("a.txt", Some(stamp(1_000))), None);
    }

    #[test]
    fn entries_are_recorded_forgotten_and_listed_in_order() {
        let mut l = Ledger::new("abc");
        assert!(l.is_empty());
        l.record("b", Some(vec![1]), Some(stamp(1)));
        l.record("a", None, None);
        assert_eq!(l.paths().collect::<Vec<_>>(), ["a", "b"]);
        l.forget("a");
        assert_eq!(l.len(), 1);
        assert_eq!(l.base(), "abc");
        assert_eq!(l.entry("b").unwrap().content.as_deref(), Some(&[1u8][..]));
    }

    /// The wire form round-trips, content as base64 — a ledger can outlive
    /// the process that wrote it.
    #[test]
    fn the_wire_form_round_trips() {
        let mut l = Ledger::new("abc");
        l.record("bin", Some(vec![0, 255]), Some(stamp(1)));
        l.record("gone", None, None);
        l.seal_at(42);
        let wire = serde_json::to_string(&l).unwrap();
        assert_eq!(
            wire,
            r#"{"base":"abc","entries":{"bin":{"content":"AP8=","stamp":{"size":3,"mtime_ns":1}},"gone":{"content":null,"stamp":null}},"stamped_at_ns":42}"#
        );
        assert_eq!(serde_json::from_str::<Ledger>(&wire).unwrap(), l);
    }
}
