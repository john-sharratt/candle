//! A calibrated conversation's calibration signatures, held as one unit.
//!
//! Every belief scan of a calibrated group needs every calibration question of
//! every file, and the set hardly ever changes: a question is written once, at
//! commit. Fetching them turn by turn cost a stream lookup and a header parse per
//! question on every reprojection, 58–79 ms of a 120–150 ms scan over 59,963
//! questions in 2,757 files (BlackWorld, 2026-10-30). One record per conversation
//! makes that a check per conversation.
//!
//! A record stands for exactly what the substrate holds while all of these hold,
//! and is built again the first time one does not:
//!
//! - **the same calibrations at the same lengths** ([`CalRecord::stamp`]), which
//!   covers a calibration added, tombstoned or grown;
//! - **every turn it walked had a signature** ([`CalRecord::complete`]): a turn
//!   whose signature has not landed is read again, so a first write needs no
//!   notice;
//! - **no signature was replaced, and no turn or timeline tombstoned, since**
//!   ([`SigGeneration`]). Both are rare and neither says which conversation it
//!   touched, so either rebuilds every record.
//!
//! The records are shared by every version of the substrate, as the residences
//! are, and reading them takes no lock: a scan loads one snapshot of the table,
//! looks each conversation up in it, and publishes whatever it had to build in
//! one swap at the end. A reader on an older version that finds a newer record,
//! or the reverse, sees a stamp or generation that does not match and builds its
//! own.

use std::collections::HashMap;
use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::persistence::streams::StreamId;
use crate::projection::TimelineId;
use crate::provenance::SigWindow;

/// What a record was built against besides its stamp: rewrites of a stored
/// signature, and tombstones.
pub(crate) type SigGeneration = (u64, u64);

/// One conversation's calibration signatures.
#[derive(Debug)]
pub struct CalRecord {
    /// Each live calibration and its turn count, in `calibrations_of` order.
    stamp: Vec<(TimelineId, u32)>,
    generation: SigGeneration,
    /// Every turn walked held a signature.
    complete: bool,
    /// The stream of each signature in [`Self::sigs`].
    pub sids: Vec<StreamId>,
    /// Every calibration turn's signature window, in calibration and turn
    /// order.
    pub sigs: Vec<SigWindow>,
}

impl CalRecord {
    pub(crate) fn new(
        stamp: Vec<(TimelineId, u32)>,
        generation: SigGeneration,
        complete: bool,
        sids: Vec<StreamId>,
        sigs: Vec<SigWindow>,
    ) -> Self {
        Self {
            stamp,
            generation,
            complete,
            sids,
            sigs,
        }
    }

    /// Whether this record stands for calibrations `live` at `generation`.
    fn stands_for(
        &self,
        live: impl Iterator<Item = (TimelineId, u32)>,
        generation: SigGeneration,
    ) -> bool {
        self.complete && self.generation == generation && self.stamp.iter().copied().eq(live)
    }
}

/// One moment of the table, read without a lock for as long as a scan holds it.
pub type CalSnapshot = Arc<HashMap<TimelineId, Arc<CalRecord>>>;

/// `unit`'s record in `snapshot`, if one stands for calibrations `live` at
/// `generation`.
pub(crate) fn standing(
    snapshot: &CalSnapshot,
    unit: TimelineId,
    live: impl Iterator<Item = (TimelineId, u32)>,
    generation: SigGeneration,
) -> Option<Arc<CalRecord>> {
    snapshot
        .get(&unit)
        .filter(|r| r.stands_for(live, generation))
        .cloned()
}

/// Every conversation's record, shared by every version of the substrate.
#[derive(Debug, Default)]
pub struct CalRecords(ArcSwap<HashMap<TimelineId, Arc<CalRecord>>>);

impl CalRecords {
    /// The table as it stands.
    pub(crate) fn snapshot(&self) -> CalSnapshot {
        self.0.load_full()
    }

    /// Keep `fresh`, one swap however many a scan built.
    pub(crate) fn publish(&self, fresh: Vec<(TimelineId, Arc<CalRecord>)>) {
        if fresh.is_empty() {
            return;
        }
        self.0.rcu(|table| {
            let mut next = HashMap::clone(table);
            next.extend(fresh.iter().cloned());
            next
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(n: u64) -> TimelineId {
        TimelineId::for_test(n)
    }

    fn record(
        stamp: Vec<(TimelineId, u32)>,
        generation: SigGeneration,
        complete: bool,
    ) -> CalRecord {
        CalRecord::new(stamp, generation, complete, Vec::new(), Vec::new())
    }

    #[test]
    fn a_record_stands_for_what_it_was_built_from() {
        let r = record(vec![(tl(1), 24)], (0, 0), true);
        assert!(r.stands_for([(tl(1), 24)].into_iter(), (0, 0)));
    }

    #[test]
    fn a_calibration_grown_added_or_gone_refuses_it() {
        let r = record(vec![(tl(1), 24)], (0, 0), true);
        assert!(!r.stands_for([(tl(1), 25)].into_iter(), (0, 0)));
        assert!(!r.stands_for([(tl(1), 24), (tl(2), 3)].into_iter(), (0, 0)));
        assert!(!r.stands_for([].into_iter(), (0, 0)));
    }

    #[test]
    fn a_rewrite_or_a_tombstone_refuses_it() {
        let r = record(vec![(tl(1), 24)], (3, 7), true);
        assert!(!r.stands_for([(tl(1), 24)].into_iter(), (4, 7)));
        assert!(!r.stands_for([(tl(1), 24)].into_iter(), (3, 8)));
    }

    /// A turn whose signature had not landed is read again, however long the
    /// stamp stays the same.
    #[test]
    fn an_incomplete_record_never_stands() {
        let r = record(vec![(tl(1), 24)], (0, 0), false);
        assert!(!r.stands_for([(tl(1), 24)].into_iter(), (0, 0)));
    }

    #[test]
    fn the_table_hands_back_only_a_standing_record() {
        let records = CalRecords::default();
        records.publish(vec![(
            tl(9),
            Arc::new(record(vec![(tl(1), 2)], (0, 0), true)),
        )]);
        let snap = records.snapshot();
        assert!(standing(&snap, tl(9), [(tl(1), 2)].into_iter(), (0, 0)).is_some());
        assert!(standing(&snap, tl(9), [(tl(1), 3)].into_iter(), (0, 0)).is_none());
        assert!(standing(&snap, tl(8), [(tl(1), 2)].into_iter(), (0, 0)).is_none());
    }

    /// A snapshot is one moment: what is published after it is not in it, and
    /// a publish keeps what the table already held.
    #[test]
    fn a_publish_adds_to_the_table_and_leaves_a_snapshot_alone() {
        let records = CalRecords::default();
        let one = Arc::new(record(vec![(tl(1), 2)], (0, 0), true));
        records.publish(vec![(tl(9), one.clone())]);
        let before = records.snapshot();
        records.publish(vec![(tl(8), one)]);
        assert_eq!(before.len(), 1);
        assert_eq!(records.snapshot().len(), 2);
    }
}
