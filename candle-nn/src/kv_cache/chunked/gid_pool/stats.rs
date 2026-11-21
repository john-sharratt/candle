//! Occupancy counts for arenas and size classes, for diagnostics and for
//! out-of-regions reports.
//!
//! A class's live bytes are its slot count times its stride, whatever formats
//! occupy it, and a record arena is never counted as a size class. The empty
//! census names, for each empty arena, whether a creation window or a protection
//! keeps the sweep from releasing it, which is what an out-of-regions report
//! needs.

use super::*;

/// One key's arenas and how full they are.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyOccupancy {
    pub key: ArenaKey,
    /// Arenas registered for the key.
    pub arenas: usize,
    /// Slots holding a chunk, across those arenas.
    pub live: usize,
    /// Slots in one arena of the key.
    pub arena_chunks: usize,
}

/// The arenas of a pool that hold no chunk, and why the empty sweep would keep any
/// of them. An arena can be counted under both reasons.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmptyCensus {
    /// Arenas with no live chunk.
    pub empty: usize,
    /// Of those, arenas still inside their creation window.
    pub in_creation: usize,
    /// Of those, arenas protected from the sweep.
    pub protected: usize,
}

/// Occupancy of one size class's GPU arenas.
///
/// `reserved` counts whole ~16 MiB slabs ([`TARGET_ARENA_BYTES`]); `live`
/// counts occupied chunk slots at this class's stride. Both are GidPool
/// arena-slab quantities, distinct from the CUDA stream-ordered pool's
/// `reserved`/`used` (which include segment slack the GidPool never sees) —
/// report them on their own line, not as a partition of the CUDA-pool gap.
#[derive(Clone, Copy, Debug, Default)]
pub struct ClassOccupancy {
    /// Slot stride for this class, in bytes — its identity in a report.
    pub slot_bytes: usize,
    pub arenas: usize,
    pub reserved_bytes: usize,
    pub live_bytes: usize,
}

/// GPU arena occupancy per size class — the diagnostic the compress-to-free
/// relief rung is judged by.
///
/// This replaces the old float-vs-quant split, which cannot be computed any
/// more and was never quite the right question: an arena has no format, and
/// what the rung actually moves is occupancy *down the ladder* as bands
/// compress into smaller classes. A working rung shows the large classes'
/// live bytes falling while the small classes' rise by less.
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuArenaClassStats {
    /// One entry per rung of the ladder, in ascending stride order.
    pub classes: [ClassOccupancy; SizeClass::COUNT],
}

impl GpuArenaClassStats {
    /// Total GPU arenas across every class.
    pub fn total_arenas(&self) -> usize {
        self.classes.iter().map(|c| c.arenas).sum()
    }

    /// Total slab bytes reserved across every class.
    pub fn total_reserved_bytes(&self) -> usize {
        self.classes.iter().map(|c| c.reserved_bytes).sum()
    }

    /// Total bytes in occupied slots across every class.
    pub fn total_live_bytes(&self) -> usize {
        self.classes.iter().map(|c| c.live_bytes).sum()
    }
}
