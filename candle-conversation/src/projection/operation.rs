//! What operating a cluster member asks of its substrate
//! (`docs/replicated_substrate.md` §7.7, §12): whether a write has failed for
//! lack of space and whether one succeeds now, how large the live log is, the
//! record types it holds, the records held back for a type this build does not
//! read, the tombstones waiting to expire, and the read-only liveness audit.

use std::collections::BTreeMap;

use super::resolver::Conversation;
use crate::persistence::liveness_audit::LivenessAudit;
use crate::persistence::Result as PersistenceResult;

/// A substrate's timeline tombstones, by how far each is from expiring (§6.8).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TombstoneCounts {
    /// Timeline tombstones held as records.
    pub held: usize,
    /// Of those, the ones a complete compaction has run past: waiting only for
    /// their age.
    pub swept: usize,
}

impl Conversation {
    /// Whether a write has failed for lack of space since the last
    /// [`Self::test_write`] succeeded.
    pub fn out_of_space(&self) -> bool {
        self.persistence.facts().out_of_space()
    }

    /// A test append and commit, as leaving fault mode needs (§7.7).
    pub fn test_write(&self) -> PersistenceResult<()> {
        self.persistence.call(|p| p.test_write())
    }

    /// The bytes of live records in the log — what a full compaction copies,
    /// and so needs free beside it.
    pub fn live_bytes(&self) -> u64 {
        let view = self.read();
        self.persistence
            .call(|p| p.segment_liveness(&view).values().sum())
    }

    /// Every record type the substrate holds, by tag: what a build replacing
    /// this one must decode (§3.9).
    pub fn held_record_tags(&self) -> Vec<u8> {
        self.persistence.facts().held_record_tags()
    }

    /// Records held back because this build does not decode their type, by
    /// type and origin, with how many of each (§6.5).
    pub fn deferred_records(&self) -> BTreeMap<(String, u64), usize> {
        self.persistence.call(|p| {
            let mut counts = BTreeMap::new();
            for d in &p.replica().deferred {
                *counts.entry((d.record_type.clone(), d.origin)).or_insert(0) += 1;
            }
            counts
        })
    }

    /// The timeline tombstones held, and how many of them a complete
    /// compaction has run past.
    pub fn tombstone_counts(&self) -> TombstoneCounts {
        let view = self.read();
        let held = view.tombstoned_timelines();
        TombstoneCounts {
            held: held.len(),
            swept: held.iter().filter(|t| view.is_tombstone_swept(**t)).count(),
        }
    }

    /// The read-only liveness audit: what maintenance counts as live against
    /// what it carries. Changes nothing.
    ///
    /// Only the plan runs on the log's owner, behind everything else it has to
    /// do. The walk reads every segment and takes as long as the disk does
    /// (fifteen minutes over BlackWorld's 346 GiB), so it runs here: on the
    /// owner it stopped every write, read and replicated record for as long.
    pub fn liveness_audit(&self) -> PersistenceResult<LivenessAudit> {
        let view = self.read();
        self.persistence
            .call_behind(|p| p.liveness_audit_plan(&view))
            .walk()
    }
}
