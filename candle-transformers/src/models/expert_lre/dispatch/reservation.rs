//! The counters and the lock that let the forward thread and the pipeline thread
//! share one pass of invocations.
//!
//! A pass is a run of invocations with strictly increasing rows. Each invocation
//! counts itself into a shared reservation under one lock before it begins. A
//! boundary move runs only once every begun invocation has been served, and holds
//! the lock while it does, so no invocation can start underneath it. A reservation
//! is consumed by `record` exactly once: the host threads have been told to expect
//! it, and dropping it would leave them waiting on a summary that never arrives.

use super::*;

/// The forward thread's pass, shared with the pipeline thread under one lock.
///
/// A **pass** is a run of invocations with strictly increasing row: every trunk
/// forward, every sub-forward of a split prefill, every MTP draft step. Every
/// invocation begins by taking this lock and counting itself in `reserved`. A
/// boundary move (§9) runs only with every begun invocation served by the
/// pipeline thread, and holds this lock while it does, so no invocation can
/// begin under it.
pub(crate) struct PassState {
    pub(crate) pass: u64,
    pub(crate) reserved: u64,
}

/// The forward thread's own counters.
pub(super) struct ForwardSide {
    pub(super) seq: u64,
    pub(super) last_row: Option<usize>,
    /// The invocations recorded into the current graph segment — what decides
    /// where [`Dispatch::after`] launches it.
    pub(super) fill: SegmentFill,
}

/// The two pinned host ranges a remote entry lies in: the warm tier's pinned
/// part, and the pad.
pub(crate) type PinnedRanges = [(u64, u64); 2];

/// One invocation's host-side reservation, made by [`Dispatch::before`] and
/// consumed by [`Dispatch::record`]: its sequence number, the summary-ring slot
/// its bucketize writes, and its MoE row. The stager and the pipeline thread
/// have been told to expect it, so it must be recorded — dropping it leaves
/// both waiting on a summary word that never comes. Not `Copy`: `record` takes
/// it by value, so it is recorded once.
#[derive(Debug)]
#[must_use = "a reserved invocation must be recorded: the pipeline thread and the stager wait on its summary word"]
pub(crate) struct Reserved {
    pub(super) seq: u64,
    pub(super) slot: usize,
    pub(super) row: usize,
}

/// Bucketize's owner check (`moe_bucketize.cu`, OWNER CHECK) and, with
/// `tensor-assert`, the slot tags it reads — held here, beside the launches
/// that read them, so they outlive every launch. Without the feature the check
/// is all zero and bucketize checks nothing.
pub(crate) struct OwnerTags {
    pub(super) check: OwnerCheck,
    #[cfg(feature = "tensor-assert")]
    pub(super) _owners: Arc<SlotOwners>,
}

impl OwnerTags {
    /// The check over `owners`, whose slot `s` ends `s · slot_bytes` below
    /// `zone_end`.
    #[cfg(feature = "tensor-assert")]
    pub(crate) fn new(owners: Arc<SlotOwners>, zone_end: u64, slot_bytes: usize) -> Self {
        Self {
            check: owners.check(zone_end, slot_bytes),
            _owners: owners,
        }
    }

    /// No tags: bucketize checks nothing.
    #[cfg(not(feature = "tensor-assert"))]
    pub(crate) fn none() -> Self {
        Self {
            check: OwnerCheck::default(),
        }
    }
}
