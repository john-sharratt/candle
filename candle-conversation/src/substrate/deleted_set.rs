//! The deleted set (`docs/replicated_substrate.md` §6.8): every timeline whose
//! tombstone expired, kept for good as a 24-byte entry, and every keyed value
//! whose tombstone expired, as a 48-byte one; and the tombstones of each kind
//! a complete compaction has already run past.
//!
//! An entry dominates exactly as the tombstone it replaced did. A timeline is
//! tombstoned for as long as the substrate lives, so its records are dead at
//! read and every rewrite drops them, however late they arrive. A keyed value
//! keeps its tombstone's stamp: a write below it is dropped, one above it
//! stands, as against the tombstone.

use std::collections::{BTreeMap, BTreeSet};

use crate::persistence::deleted::{DeletedKey, DeletedPayload, DeletedTimeline};

/// See the module documentation.
#[derive(Debug, Default)]
pub struct DeletedSet {
    entries: BTreeMap<u64, DeletedTimeline>,
    swept: BTreeSet<u64>,
    keys: BTreeMap<u64, DeletedKey>,
    swept_keys: BTreeSet<u64>,
}

impl DeletedSet {
    /// Take in a `Deleted` record: its entries and its swept tombstones join
    /// what is held. Both only grow — of two entries for one key, the one with
    /// the greater stamp is kept — so the order records arrive in does not
    /// matter.
    pub fn apply(&mut self, payload: &DeletedPayload) {
        for entry in &payload.timelines {
            self.entries.insert(entry.timeline, *entry);
        }
        self.swept.extend(payload.swept.iter().copied());
        for entry in &payload.keys {
            let held = self.keys.entry(entry.key).or_insert(*entry);
            if entry.stamp() > held.stamp() {
                *held = *entry;
            }
        }
        self.swept_keys.extend(payload.swept_keys.iter().copied());
    }

    /// The entry for `timeline`, if its tombstone expired.
    pub fn entry(&self, timeline: u64) -> Option<&DeletedTimeline> {
        self.entries.get(&timeline)
    }

    /// Whether a complete compaction has run past `timeline`'s tombstone.
    pub fn is_swept(&self, timeline: u64) -> bool {
        self.swept.contains(&timeline)
    }

    /// The entry for the keyed value hashed `key`, if its tombstone expired.
    pub fn key_entry(&self, key: u64) -> Option<&DeletedKey> {
        self.keys.get(&key)
    }

    /// Whether a complete compaction has run past the tombstone on `key`.
    pub fn is_key_swept(&self, key: u64) -> bool {
        self.swept_keys.contains(&key)
    }

    /// The set as a `Deleted` record holds it, or `None` when it is empty.
    pub fn payload(&self) -> Option<DeletedPayload> {
        if self.entries.is_empty()
            && self.swept.is_empty()
            && self.keys.is_empty()
            && self.swept_keys.is_empty()
        {
            return None;
        }
        Some(DeletedPayload {
            timelines: self.entries.values().copied().collect(),
            swept: self.swept.iter().copied().collect(),
            keys: self.keys.values().copied().collect(),
            swept_keys: self.swept_keys.iter().copied().collect(),
        })
    }

    /// Every timeline entry, in timeline order.
    pub fn entries(&self) -> impl Iterator<Item = DeletedTimeline> + '_ {
        self.entries.values().copied()
    }

    /// Every keyed entry, in key order.
    pub fn key_entries(&self) -> impl Iterator<Item = DeletedKey> + '_ {
        self.keys.values().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(timeline: u64) -> DeletedTimeline {
        DeletedTimeline {
            timeline,
            origin: 0xA,
            oseq: timeline + 100,
        }
    }

    fn key(key: u64, hlc: u64) -> DeletedKey {
        DeletedKey {
            key,
            gen: 0,
            epoch: 0,
            hlc,
            origin: 0xA,
            oseq: hlc,
        }
    }

    #[test]
    fn records_join_in_any_order() {
        let first = DeletedPayload {
            timelines: vec![entry(5)],
            swept: vec![7],
            keys: vec![key(1, 50)],
            swept_keys: vec![2],
        };
        let second = DeletedPayload {
            timelines: vec![entry(3)],
            swept: vec![9],
            keys: vec![key(1, 40), key(4, 10)],
            swept_keys: vec![3],
        };
        let mut a = DeletedSet::default();
        a.apply(&first);
        a.apply(&second);
        let mut b = DeletedSet::default();
        b.apply(&second);
        b.apply(&first);
        assert_eq!(a.payload(), b.payload());
        assert_eq!(
            a.payload().unwrap(),
            DeletedPayload {
                timelines: vec![entry(3), entry(5)],
                swept: vec![7, 9],
                keys: vec![key(1, 50), key(4, 10)],
                swept_keys: vec![2, 3],
            }
        );
        assert_eq!(a.entry(5), Some(&entry(5)));
        assert!(a.is_swept(9) && !a.is_swept(5));
        assert_eq!(
            a.key_entry(1),
            Some(&key(1, 50)),
            "the greater stamp is kept"
        );
        assert!(a.is_key_swept(3) && !a.is_key_swept(1));
        assert_eq!(DeletedSet::default().payload(), None);
    }
}
