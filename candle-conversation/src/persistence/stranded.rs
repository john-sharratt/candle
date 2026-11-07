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
//! linked and logs each record's two locations. The persistence side keeps
//! the set, and choosing the next op leaves out a segment while the index
//! still names one of its records there ([`Stranded::held_in`]): picked again,
//! a mostly-dead segment would rank first, be kept again, and starve every
//! other segment. It comes back by itself the moment the index moves off — a
//! newer seal, a tombstone, a restart.

use std::collections::BTreeSet;

use super::segment::SegmentId;
use super::streams::StreamId;
use crate::substrate::{IndexPage, StreamKey, Substrate};

/// Which index entry a stranded record is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StrandedKind {
    Chunk,
    Tokens,
    IndexPage,
    Snapshot,
    BranchCheckpoint,
}

/// One record the substrate index names inside a segment an op was about to
/// drop.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct StrandedRecord {
    pub kind: StrandedKind,
    pub stream_id: u64,
    /// The stream's writer, for a per-writer entry (chunk, tokens, index
    /// page); 0 for a snapshot or branch checkpoint, which have one.
    pub writer: u64,
    /// The chunk's index, for a chunk; 0 otherwise.
    pub chunk_index: u64,
    pub segment: SegmentId,
    pub offset: u64,
}

impl StrandedRecord {
    /// Whether `substrate`'s index names this record where it was found.
    pub fn named_in(&self, substrate: &Substrate) -> bool {
        let key = StreamKey {
            stream_id: StreamId(self.stream_id),
            writer: self.writer,
        };
        let at = |segment: SegmentId, offset: u64| segment == self.segment && offset == self.offset;
        match self.kind {
            StrandedKind::Chunk => substrate
                .stream_at(key)
                .and_then(|s| s.chunks.get(&self.chunk_index).copied())
                .is_some_and(|l| at(l.segment, l.offset)),
            StrandedKind::Tokens => substrate
                .stream_at(key)
                .and_then(|s| s.tokens)
                .is_some_and(|l| at(l.segment, l.offset)),
            StrandedKind::IndexPage => substrate
                .stream_at(key)
                .and_then(|s| s.index_page.as_ref().and_then(IndexPage::stored))
                .is_some_and(|l| at(l.segment, l.offset)),
            StrandedKind::Snapshot => substrate
                .recurrent_snapshot_loc(key.stream_id)
                .is_some_and(|l| at(l.segment, l.offset)),
            StrandedKind::BranchCheckpoint => substrate
                .branch_checkpoint_loc(key.stream_id)
                .is_some_and(|l| at(l.segment, l.offset)),
        }
    }
}

/// Records the index names inside segments an op was about to drop.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Stranded {
    records: BTreeSet<StrandedRecord>,
}

impl Stranded {
    /// Note `record`.
    pub fn push(&mut self, record: StrandedRecord) {
        self.records.insert(record);
    }

    /// Take in every record of `other`.
    pub fn extend(&mut self, other: &Stranded) {
        self.records.extend(other.records.iter().copied());
    }

    /// Keep the records whose segment `keep` allows.
    pub fn retain_segments(&mut self, keep: impl Fn(SegmentId) -> bool) {
        self.records.retain(|r| keep(r.segment));
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// The records, in order.
    pub fn records(&self) -> impl Iterator<Item = &StrandedRecord> {
        self.records.iter()
    }

    /// Whether `segment` holds a record of the set, and so must stay linked.
    pub fn holds(&self, segment: SegmentId) -> bool {
        self.records.iter().any(|r| r.segment == segment)
    }

    /// The segments holding a record `substrate`'s index still names there:
    /// the ones the next op must leave alone.
    pub fn held_in(&self, substrate: &Substrate) -> BTreeSet<SegmentId> {
        self.records
            .iter()
            .filter(|r| r.named_in(substrate))
            .map(|r| r.segment)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(kind: StrandedKind, stream_id: u64, segment: u64) -> StrandedRecord {
        StrandedRecord {
            kind,
            stream_id,
            writer: 0,
            chunk_index: 0,
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

    /// A record the index does not name holds nothing back.
    #[test]
    fn a_record_the_index_does_not_name_holds_nothing_back() {
        let mut s = Stranded::default();
        s.push(record(StrandedKind::Snapshot, 7, 3));
        assert!(s.held_in(&Substrate::new()).is_empty());
    }

    /// The set keeps only the records of segments still there.
    #[test]
    fn the_set_keeps_only_segments_still_there() {
        let mut s = Stranded::default();
        s.push(record(StrandedKind::Snapshot, 7, 3));
        s.push(record(StrandedKind::Tokens, 8, 4));
        s.retain_segments(|seg| seg == SegmentId(4));
        assert!(!s.holds(SegmentId(3)) && s.holds(SegmentId(4)));
    }
}
