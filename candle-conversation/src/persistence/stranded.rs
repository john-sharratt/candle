//! What a maintenance op's apply left pointing into the segments it is about
//! to drop.
//!
//! An op relocates a target segment's live records, repoints the substrate
//! index at the copies, then unlinks the target. The relocation filters each
//! planned tail against the persistence-side live-tail maps, so a tail those
//! maps disagree on is never copied, and the apply never repoints it. If the
//! substrate index still names that record, unlinking the target leaves the
//! index pointing at a file that is gone: the record is indexed but
//! unreadable, on this member, until it restarts.
//!
//! [`Stranded`] names those records after the apply, and
//! `SubstratePersistence::finish_maintenance` keeps every segment holding one
//! linked. The op's relocated copies stand, the kept segment is planned again
//! by a later pass, and each record's two locations are logged.

use std::collections::BTreeSet;

use super::segment::SegmentId;

/// Which index entry a stranded record is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StrandedKind {
    Chunk,
    Tokens,
    IndexPage,
    Snapshot,
    BranchCheckpoint,
}

/// One record the substrate index still names inside a target segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StrandedRecord {
    pub kind: StrandedKind,
    pub stream_id: u64,
    pub segment: SegmentId,
    pub offset: u64,
}

/// Every record an op's apply left in a target segment.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Stranded {
    records: BTreeSet<StrandedRecord>,
}

impl Stranded {
    /// Note that `record` is still where the plan found it.
    pub fn push(&mut self, record: StrandedRecord) {
        self.records.insert(record);
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The records, in order.
    pub fn records(&self) -> impl Iterator<Item = &StrandedRecord> {
        self.records.iter()
    }

    /// Whether `segment` holds a stranded record, and so must stay linked.
    pub fn holds(&self, segment: SegmentId) -> bool {
        self.records.iter().any(|r| r.segment == segment)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(kind: StrandedKind, stream_id: u64, segment: u64) -> StrandedRecord {
        StrandedRecord {
            kind,
            stream_id,
            segment: SegmentId(segment),
            offset: 0,
        }
    }

    /// **A segment holding a stranded record is held; the others are not.**
    #[test]
    fn a_segment_holding_a_stranded_record_is_held() {
        let mut s = Stranded::default();
        assert!(s.is_empty());
        s.push(record(StrandedKind::Snapshot, 7, 16628));
        s.push(record(StrandedKind::Snapshot, 7, 16628));
        assert!(s.holds(SegmentId(16628)));
        assert!(!s.holds(SegmentId(16629)));
        assert_eq!(s.records().count(), 1, "one record, noted twice");
    }
}
