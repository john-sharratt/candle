//! The swept set (`docs/replicated_substrate.md` §6.8): the tombstones of each
//! kind a complete compaction has already run past, the first of the two
//! conditions for a tombstone to expire. An expired tombstone leaves nothing
//! behind.

use crate::persistence::deleted::DeletedPayload;

use super::pcoll::POrdSet;

/// See the module documentation. Persistent: a clone is O(1).
#[derive(Debug, Default, Clone)]
pub struct DeletedSet {
    swept: POrdSet<u64>,
    swept_keys: POrdSet<u64>,
}

impl DeletedSet {
    /// Take in a `Deleted` record: its swept tombstones join what is held. The
    /// sets only grow, so the order records arrive in does not matter.
    pub fn apply(&mut self, payload: &DeletedPayload) {
        for &t in &payload.swept {
            self.swept.insert(t);
        }
        for &k in &payload.swept_keys {
            self.swept_keys.insert(k);
        }
    }

    /// Whether a complete compaction has run past `timeline`'s tombstone.
    pub fn is_swept(&self, timeline: u64) -> bool {
        self.swept.contains(&timeline)
    }

    /// Whether a complete compaction has run past the tombstone on `key`.
    pub fn is_key_swept(&self, key: u64) -> bool {
        self.swept_keys.contains(&key)
    }

    /// The set as a `Deleted` record holds it, or `None` when it is empty.
    pub fn payload(&self) -> Option<DeletedPayload> {
        if self.swept.is_empty() && self.swept_keys.is_empty() {
            return None;
        }
        Some(DeletedPayload {
            swept: self.swept.iter().copied().collect(),
            swept_keys: self.swept_keys.iter().copied().collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn records_join_in_any_order() {
        let first = DeletedPayload {
            swept: vec![7],
            swept_keys: vec![2],
        };
        let second = DeletedPayload {
            swept: vec![9],
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
                swept: vec![7, 9],
                swept_keys: vec![2, 3],
            }
        );
        assert!(a.is_swept(9) && !a.is_swept(5));
        assert!(a.is_key_swept(3) && !a.is_key_swept(1));
        assert_eq!(DeletedSet::default().payload(), None);
    }
}
