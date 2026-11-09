//! Values derived from a site's files, kept per version of the file: a
//! prepared answer (W8), a parsed post (W10).
//!
//! An entry is keyed by where the file is — its real path on disk, or the
//! address of the compiled-in file — and what was derived from it, and is
//! valid while the file's stamp (modification time and length) is what it was
//! read at. It is never keyed by the bytes' address: the linker may share one
//! copy between byte-identical files, and two files would then answer as one.
//!
//! What is kept is bounded by cost: an insert past [`CAP`] clears the map, so
//! a deleted file's entry lives at most until then. Compiled-in files cost
//! nothing — they are in the binary already.

use std::collections::HashMap;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::path::PathBuf;
use std::sync::RwLock;
use std::time::SystemTime;

/// The most kept values may cost, in bytes.
pub const CAP: usize = 64 << 20;

/// Where a file is.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Origin {
    /// Its canonical path on disk.
    Disk(PathBuf),
    /// The address of its compiled-in `include_dir::File`.
    Embedded(usize),
}

/// A disk file's version: its modification time and length.
pub type Stamp = (SystemTime, u64);

/// An entry's key: the file, and what was derived from it.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Key {
    pub origin: Origin,
    pub kind: String,
}

/// Kept values, each with the stamp it was derived at.
pub struct Stamped<V> {
    inner: RwLock<Inner<V>>,
}

struct Inner<V> {
    map: HashMap<Key, (Option<Stamp>, V, usize)>,
    cost: usize,
}

impl<V> Default for Stamped<V> {
    fn default() -> Self {
        Stamped {
            inner: RwLock::new(Inner {
                map: HashMap::new(),
                cost: 0,
            }),
        }
    }
}

impl<V> Debug for Stamped<V> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let inner = self.inner.read().unwrap_or_else(|e| e.into_inner());
        f.debug_struct("Stamped")
            .field("entries", &inner.map.len())
            .field("cost", &inner.cost)
            .finish()
    }
}

impl<V: Clone> Stamped<V> {
    /// The value kept for `key`, when it was derived at `stamp`.
    pub fn get(&self, key: &Key, stamp: Option<Stamp>) -> Option<V> {
        let inner = self.inner.read().unwrap_or_else(|e| e.into_inner());
        let (at, value, _) = inner.map.get(key)?;
        (*at == stamp).then(|| value.clone())
    }

    /// Keep `value` for `key`, derived at `stamp`, costing `cost` bytes.
    pub fn put(&self, key: Key, stamp: Option<Stamp>, value: V, cost: usize) {
        let mut inner = self.inner.write().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, was)) = inner.map.remove(&key) {
            inner.cost -= was;
        }
        if inner.cost + cost > CAP {
            inner.map.clear();
            inner.cost = 0;
        }
        inner.cost += cost;
        inner.map.insert(key, (stamp, value, cost));
    }

    /// How many values are kept.
    pub fn len(&self) -> usize {
        self.inner
            .read()
            .unwrap_or_else(|e| e.into_inner())
            .map
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn key(path: &str) -> Key {
        Key {
            origin: Origin::Disk(PathBuf::from(path)),
            kind: "text/css".into(),
        }
    }

    fn at(secs: u64, len: u64) -> Option<Stamp> {
        Some((SystemTime::UNIX_EPOCH + Duration::from_secs(secs), len))
    }

    /// A value is answered only at the stamp it was kept at.
    #[test]
    fn a_value_answers_only_at_its_stamp() {
        let s: Stamped<u32> = Stamped::default();
        s.put(key("/a"), at(1, 10), 7, 10);
        assert_eq!(s.get(&key("/a"), at(1, 10)), Some(7));
        assert_eq!(s.get(&key("/a"), at(2, 10)), None, "a later mtime");
        assert_eq!(s.get(&key("/a"), at(1, 11)), None, "another length");
        assert_eq!(s.get(&key("/b"), at(1, 10)), None);
    }

    /// **An insert past the cap clears what was kept**, and a replaced entry
    /// gives its cost back.
    #[test]
    fn an_insert_past_the_cap_clears_the_map() {
        let s: Stamped<u32> = Stamped::default();
        s.put(key("/a"), at(1, 1), 1, CAP / 2);
        s.put(key("/a"), at(2, 1), 2, CAP / 2);
        s.put(key("/b"), at(1, 1), 3, CAP / 2);
        assert_eq!(s.len(), 2, "the replaced entry's cost was given back");
        s.put(key("/c"), at(1, 1), 4, 1);
        assert_eq!(s.len(), 1);
        assert_eq!(s.get(&key("/c"), at(1, 1)), Some(4));
    }
}
