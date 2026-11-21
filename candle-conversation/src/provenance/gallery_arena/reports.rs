//! Defines the views the gallery reports on itself: a resident turn's footprint,
//! why an index was missed, and the addresses of a turn made resident.
//!
//! A miss names its reason, because a scan with no index, one whose turn has
//! moved, and one scanning too few groups each call for a different response. The
//! covering groups let the caller build an index that serves the next reader as
//! well. An ensured turn carries the run id its addresses belong to, so anything
//! cached over those addresses can be checked against it.

use super::*;

/// One resident turn as [`GalleryArena::resident_runs`] reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentRun {
    pub sid: StreamId,
    /// Bytes of the pages it holds.
    pub bytes: u64,
    /// Time since a scan last used it.
    pub idle: Duration,
    pub holder: Holder,
}

/// Why a scan could not reuse a cached index.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(in super::super) enum IndexMiss {
    /// No index was cached for the segment set — never built, evicted from
    /// the cache, or too large to cache.
    Absent,
    /// The cached index named a turn since evicted, moved or re-uploaded.
    Stale,
    /// The cached index scans fewer fold groups than a segment weighs.
    /// `covering` is each segment's groups widened by the cached index's:
    /// built over those, the index serves this reader and the one before it.
    Narrower { covering: Vec<GroupSet> },
}

impl IndexMiss {
    pub(in super::super) fn reason(&self) -> &'static str {
        match self {
            IndexMiss::Absent => "no index cached for this segment set",
            IndexMiss::Stale => "a turn it holds was evicted, moved or re-uploaded",
            IndexMiss::Narrower { .. } => "the cached index scans fewer fold groups",
        }
    }
}

/// A turn made resident: its pages' device addresses (page order), the run id
/// they belong to ([`ResidentTurn::run_id`]), the fold groups the pages hold,
/// which of its tokens they hold, and whether they were uploaded now rather
/// than found resident.
pub(in super::super) struct Ensured {
    pub addrs: Vec<u64>,
    pub run_id: u64,
    pub groups: GroupSet,
    pub mask: Arc<KeepMask>,
    pub uploaded: bool,
}
