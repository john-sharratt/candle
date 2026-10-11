//! A key → location map that keeps, per key, the record with the greatest
//! version stamp (`docs/replicated_substrate.md` §6.5).
//!
//! The persistence layer's location maps — where each stream's current
//! metadata record, each snapshot tail and each character live — decide what
//! maintenance carries and what it lets go. Records reach the log in the order
//! they arrive, not the order they were written, so "last appended" is not
//! "current": a record ranked below the one already held is placed in the log,
//! and dead the moment it lands.
//!
//! A key can hold a **withdrawn** value — a snapshot retraction, which is a
//! record and outranks what it withdraws, but names no location. It reads as
//! absent, and still refuses any older record offered after it.

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;

use super::identity::{RecordIdentity, Stamped};
use super::manifest::RecordLoc;

/// See the module docs.
#[derive(Clone, Debug)]
pub struct StampedLocs<K> {
    map: HashMap<K, Stamped<Option<RecordLoc>>>,
}

impl<K> Default for StampedLocs<K> {
    fn default() -> Self {
        StampedLocs {
            map: HashMap::new(),
        }
    }
}

impl<K: Hash + Eq + Copy> StampedLocs<K> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Offer the record `id` at `loc` (`None` for a withdrawal) for `key`.
    /// It is taken when it supersedes what the key holds; answers whether it
    /// was.
    pub fn offer(&mut self, key: K, loc: Option<RecordLoc>, id: RecordIdentity) -> bool {
        match self.map.entry(key) {
            Entry::Occupied(mut e) => {
                if !id.supersedes(&e.get().id) {
                    return false;
                }
                e.insert(Stamped { value: loc, id });
            }
            Entry::Vacant(e) => {
                e.insert(Stamped { value: loc, id });
            }
        }
        true
    }

    /// The current location for `key`; `None` when absent or withdrawn.
    pub fn get(&self, key: &K) -> Option<&RecordLoc> {
        self.map.get(key).and_then(|s| s.value.as_ref())
    }

    /// The identity of the record `key` holds, withdrawn or not.
    pub fn identity(&self, key: &K) -> Option<RecordIdentity> {
        self.map.get(key).map(|s| s.id)
    }

    /// Whether `key` has a current location.
    pub fn contains_key(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    /// Forget `key` entirely — what a dominating fact (a timeline tombstone)
    /// does to every record it names.
    pub fn remove(&mut self, key: &K) {
        self.map.remove(key);
    }

    /// Every key with a current location.
    pub fn iter(&self) -> impl Iterator<Item = (&K, &RecordLoc)> + '_ {
        self.map
            .iter()
            .filter_map(|(k, s)| s.value.as_ref().map(|loc| (k, loc)))
    }

    /// Every current location.
    pub fn values(&self) -> impl Iterator<Item = &RecordLoc> + '_ {
        self.map.values().filter_map(|s| s.value.as_ref())
    }

    /// Keep only the located keys `keep` accepts — what dropping a segment does
    /// to the entries naming it. A withdrawal names no location and stays.
    pub fn retain(&mut self, mut keep: impl FnMut(&K, &RecordLoc) -> bool) {
        self.map
            .retain(|k, s| s.value.as_ref().is_none_or(|loc| keep(k, loc)));
    }

    pub fn clear(&mut self) {
        self.map.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::segment::SegmentId;

    fn loc(offset: u64) -> RecordLoc {
        RecordLoc {
            segment: SegmentId(1),
            offset,
            payload_len: 10,
            record_size: 4096,
        }
    }

    fn id(hlc: u64) -> RecordIdentity {
        RecordIdentity {
            origin: 7,
            oseq: hlc,
            gen: 0,
            epoch: 0,
            hlc,
        }
    }

    /// The greater stamp holds the key whichever arrives first.
    #[test]
    fn the_greater_stamp_wins_in_either_order() {
        for order in [[1u64, 2], [2, 1]] {
            let mut m = StampedLocs::new();
            for hlc in order {
                m.offer(9u64, Some(loc(hlc * 4096)), id(hlc));
            }
            assert_eq!(m.get(&9), Some(&loc(8192)), "order {order:?}");
            assert_eq!(m.identity(&9), Some(id(2)));
        }
    }

    /// A relocated copy keeps its identity and moves the key to its new home.
    #[test]
    fn a_relocated_copy_moves_the_key() {
        let mut m = StampedLocs::new();
        assert!(m.offer(9u64, Some(loc(4096)), id(3)));
        assert!(m.offer(9u64, Some(loc(81920)), id(3)));
        assert_eq!(m.get(&9), Some(&loc(81920)));
    }

    /// A withdrawal reads as absent and still refuses an older record.
    #[test]
    fn a_withdrawal_refuses_what_it_outranks() {
        let mut m = StampedLocs::new();
        m.offer(9u64, Some(loc(4096)), id(1));
        assert!(m.offer(9u64, None, id(5)));
        assert_eq!(m.get(&9), None);
        assert!(!m.contains_key(&9));
        assert!(!m.offer(9u64, Some(loc(8192)), id(4)));
        assert_eq!(m.get(&9), None);
        assert!(m.offer(9u64, Some(loc(12288)), id(6)));
        assert_eq!(m.get(&9), Some(&loc(12288)));
        assert_eq!(m.iter().count(), 1);
    }
}
