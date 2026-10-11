//! The deleted set (`docs/replicated_substrate.md` §6.8): every timeline whose
//! tombstone expired, kept for good as a 24-byte entry, and the timeline
//! tombstones a complete compaction has already run past.
//!
//! An entry dominates exactly as the tombstone it replaced did: the timeline
//! is tombstoned for as long as the substrate lives, so its records are dead
//! at read and every rewrite drops them, however late they arrive.

use std::collections::{BTreeMap, BTreeSet};

use crate::persistence::deleted::{DeletedPayload, DeletedTimeline};

/// See the module documentation.
#[derive(Debug, Default)]
pub struct DeletedSet {
    entries: BTreeMap<u64, DeletedTimeline>,
    swept: BTreeSet<u64>,
}

impl DeletedSet {
    /// Take in a `Deleted` record: its entries and its swept tombstones join
    /// what is held. Both only grow, so the order records arrive in does not
    /// matter.
    pub fn apply(&mut self, payload: &DeletedPayload) {
        for entry in &payload.timelines {
            self.entries.insert(entry.timeline, *entry);
        }
        self.swept.extend(payload.swept.iter().copied());
    }

    /// The entry for `timeline`, if its tombstone expired.
    pub fn entry(&self, timeline: u64) -> Option<&DeletedTimeline> {
        self.entries.get(&timeline)
    }

    /// Whether a complete compaction has run past `timeline`'s tombstone.
    pub fn is_swept(&self, timeline: u64) -> bool {
        self.swept.contains(&timeline)
    }

    /// The set as a `Deleted` record holds it, or `None` when it is empty.
    pub fn payload(&self) -> Option<DeletedPayload> {
        if self.entries.is_empty() && self.swept.is_empty() {
            return None;
        }
        Some(DeletedPayload {
            timelines: self.entries.values().copied().collect(),
            swept: self.swept.iter().copied().collect(),
        })
    }

    /// Every entry, in timeline order.
    pub fn entries(&self) -> impl Iterator<Item = DeletedTimeline> + '_ {
        self.entries.values().copied()
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

    #[test]
    fn records_join_in_any_order() {
        let first = DeletedPayload {
            timelines: vec![entry(5)],
            swept: vec![7],
        };
        let second = DeletedPayload {
            timelines: vec![entry(3)],
            swept: vec![9],
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
            }
        );
        assert_eq!(a.entry(5), Some(&entry(5)));
        assert!(a.is_swept(9) && !a.is_swept(5));
        assert_eq!(DeletedSet::default().payload(), None);
    }
}
