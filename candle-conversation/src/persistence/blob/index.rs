//! Where every live blob is.
//!
//! Locations only — no blob's bytes are held — keyed by the first 128 bits of
//! each blob's hash, which its record header carries. Of the records of one
//! hash — the same bytes, written again or by another member — the one with
//! the greatest version stamp is held, as the accounting ranks them
//! (`docs/replicated_substrate.md` §6.5), so the two agree on which is live.
//! A blob its tombstone outranks is forgotten (`super::forget_if_retired`).
//! Built by the open walk, kept current by every append and every remote
//! record applied, rebuilt after compaction; compaction and maintenance read
//! it to know what to carry, and a read follows it to the bytes.

use std::collections::BTreeMap;

use super::payload::BlobKey;
use crate::persistence::identity::{RecordIdentity, Stamped};
use crate::persistence::manifest::RecordLoc;

/// Every live blob's location, and the identity of the record there. See the
/// module.
#[derive(Debug, Default)]
pub struct BlobIndex {
    blobs: BTreeMap<BlobKey, Stamped<RecordLoc>>,
}

impl BlobIndex {
    pub fn new() -> Self {
        Self::default()
    }

    /// Blob `key`'s record `id` is at `loc`. It is held when the key holds
    /// nothing or it outranks what the key holds — a relocated copy keeps its
    /// identity and so moves the key; answers whether it was.
    pub fn record(&mut self, key: BlobKey, loc: RecordLoc, id: RecordIdentity) -> bool {
        match self.blobs.get(&key) {
            Some(held) if !id.supersedes(&held.id) => false,
            _ => {
                self.blobs.insert(key, Stamped { value: loc, id });
                true
            }
        }
    }

    pub fn get(&self, key: BlobKey) -> Option<RecordLoc> {
        self.blobs.get(&key).map(|s| s.value)
    }

    /// The identity of the record blob `key` is held by.
    pub fn identity(&self, key: BlobKey) -> Option<RecordIdentity> {
        self.blobs.get(&key).map(|s| s.id)
    }

    /// Every blob, in key order — the order a rewrite carries them in.
    pub fn blobs(&self) -> impl Iterator<Item = (BlobKey, RecordLoc)> + '_ {
        self.blobs.iter().map(|(k, s)| (*k, s.value))
    }

    pub fn len(&self) -> usize {
        self.blobs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.blobs.is_empty()
    }

    /// Blob `key`, at `from`, now lives at `to`. A blob no longer at `from` is
    /// left where the index has it.
    pub fn repoint(&mut self, key: BlobKey, from: RecordLoc, to: RecordLoc) {
        if let Some(at) = self.blobs.get_mut(&key) {
            if at.value == from {
                at.value = to;
            }
        }
    }

    /// Forget blob `key`: its tombstone outranks the record held.
    pub fn forget(&mut self, key: BlobKey) {
        self.blobs.remove(&key);
    }

    pub fn clear(&mut self) {
        self.blobs.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::persistence::segment::SegmentId;

    const KEY: BlobKey = BlobKey { hi: 1, lo: 2 };

    fn loc(offset: u64) -> RecordLoc {
        RecordLoc {
            segment: SegmentId(1),
            offset,
            payload_len: 40,
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

    /// **Of two records of one hash, the greater stamp is held whichever
    /// arrives first**, as the accounting counts it live.
    #[test]
    fn the_greater_stamp_is_held_in_either_order() {
        for order in [[1u64, 2], [2, 1]] {
            let mut index = BlobIndex::new();
            for hlc in order {
                index.record(KEY, loc(hlc * 4096), id(hlc));
            }
            assert_eq!(index.get(KEY), Some(loc(8192)), "order {order:?}");
            assert_eq!(index.identity(KEY), Some(id(2)));
            assert_eq!(index.len(), 1);
        }
    }

    /// A relocated copy keeps its identity and moves the key; a forgotten key
    /// takes the next record of its hash as new.
    #[test]
    fn a_copy_moves_the_key_and_a_forgotten_one_starts_again() {
        let mut index = BlobIndex::new();
        assert!(index.record(KEY, loc(4096), id(3)));
        assert!(index.record(KEY, loc(81920), id(3)));
        assert_eq!(index.get(KEY), Some(loc(81920)));
        assert!(!index.record(KEY, loc(4096), id(2)));
        index.forget(KEY);
        assert!(index.is_empty());
        assert!(index.record(KEY, loc(4096), id(2)));
        assert_eq!(index.blobs().collect::<Vec<_>>(), vec![(KEY, loc(4096))]);
    }
}
