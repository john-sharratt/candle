//! Segment identity for the segmented redo log.
//!
//! The redo log is a set of ~4 GB **segment** files in `.substrate/`
//! (`docs/segmented_substrate_log.md`). Each record's in-RAM location is
//! addressed by `(SegmentId, offset)` rather than a bare file offset, so a
//! read routes to the segment file that physically holds it.
//!
//! `SegmentId` is the append-order rank of a segment: monotonically increasing,
//! never reused, and encoded in the filename (`seg-<id>.log` / `.active`). It is
//! also the **recency order** — because relocation (compact / combine) always
//! appends into the active (highest-id) segment, a key's live record is always
//! its highest-id occurrence, so id order is a valid last-writer-wins total
//! order (no manifest / LSN needed).
//!
//! During the single-segment phase every location carries [`FIRST_SEGMENT`];
//! nothing else changes until sealing/rotation lands.

/// Append-order identity of one log segment. Monotonic, never reused; higher id
/// = newer = wins under last-writer-wins.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct SegmentId(pub u64);

/// The first segment minted for a fresh store. Every location in the
/// single-segment phase carries this id.
pub const FIRST_SEGMENT: SegmentId = SegmentId(1);

impl SegmentId {
    /// The raw append-order rank.
    pub fn raw(self) -> u64 {
        self.0
    }

    /// The next segment id in append order.
    pub fn next(self) -> SegmentId {
        SegmentId(self.0 + 1)
    }
}

impl std::fmt::Display for SegmentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}
