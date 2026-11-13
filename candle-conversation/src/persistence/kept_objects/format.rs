//! The export's lines (`docs/hit_driven_projection.md` §13 step 11).
//!
//! ```text
//! {"format":"substrate-objects","version":1,"daemon":"zend","taken_at_ms":…}
//! {"object":{…CustomObjectPayload…},"stamp":[gen,epoch,hlc,origin]}
//! …
//! {"blob":"<sha256 hex>","bytes_b64":"…"}
//! …
//! ```
//!
//! An object line carries its record's original stamp: the union of several
//! members' exports takes the greatest per key, and a tombstone competes on it
//! like any write.

use serde::{Deserialize, Serialize};

use crate::persistence::identity::VersionStamp;
use crate::persistence::record::CustomObjectPayload;

/// The header's `format`.
pub const FORMAT: &str = "substrate-objects";

/// The header's `version`.
pub const VERSION: u32 = 1;

/// The export's first line.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Header {
    pub format: String,
    pub version: u32,
    /// `zend` or `npcd`.
    pub daemon: String,
    pub taken_at_ms: u64,
}

impl Header {
    pub fn new(daemon: &str, taken_at_ms: u64) -> Self {
        Self {
            format: FORMAT.into(),
            version: VERSION,
            daemon: daemon.into(),
            taken_at_ms,
        }
    }
}

/// A record's stamp as `[gen, epoch, hlc, origin]` — compared as an array in
/// that order, it orders as [`VersionStamp`] does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct Stamp(pub [u64; 4]);

impl From<VersionStamp> for Stamp {
    fn from(s: VersionStamp) -> Self {
        Stamp([s.gen, s.epoch, s.hlc, s.origin])
    }
}

/// Every line after the header.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Line {
    Object {
        object: CustomObjectPayload,
        stamp: Stamp,
    },
    Blob {
        blob: String,
        bytes_b64: String,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The lines read as the format states them, and a stamp orders as its
    /// version stamp does.
    #[test]
    fn the_lines_are_the_formats() {
        assert_eq!(
            serde_json::to_string(&Header::new("zend", 7)).unwrap(),
            r#"{"format":"substrate-objects","version":1,"daemon":"zend","taken_at_ms":7}"#
        );
        let obj = Line::Object {
            object: CustomObjectPayload::new("a/b", json!({"x": 1})),
            stamp: Stamp([1, 2, 3, 4]),
        };
        assert_eq!(
            serde_json::to_string(&obj).unwrap(),
            r#"{"object":{"key":"a/b","blob":{"x":1}},"stamp":[1,2,3,4]}"#
        );
        let blob: Line = serde_json::from_str(r#"{"blob":"ab","bytes_b64":"AQI="}"#).unwrap();
        assert_eq!(
            blob,
            Line::Blob {
                blob: "ab".into(),
                bytes_b64: "AQI=".into()
            }
        );
        let s = |gen, epoch, hlc, origin| {
            Stamp::from(VersionStamp {
                gen,
                epoch,
                hlc,
                origin,
            })
        };
        assert!(s(1, 0, 0, 0) > s(0, 9, 9, 9));
        assert!(s(1, 1, 5, 0) > s(1, 1, 4, 9));
    }
}
