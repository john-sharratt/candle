//! Caps how many prompt branch checkpoints survive a compaction.
//!
//! A checkpoint is a cache: it is a pure function of a prompt that is still on
//! disk, so dropping one costs a single prefill on next use. Each is large, and
//! nothing else reclaims them, because they are keyed by content and a prompt
//! edit orphans the old one. `partition_branch_checkpoints` keeps the newest by
//! write position, the nearest thing to recently used that the index records. A
//! checkpoint that is read but not rewritten does not refresh, so an active
//! branch can age out, at the cost of one prefill.

use super::*;

/// How many prompt branch checkpoints survive a compaction.
///
/// A bound rather than a policy: they are a cache, they are ~63 MiB each, and
/// nothing else reclaims them. Sized so a workspace that switches between a
/// handful of selector assignments never loses one it is actually using, while
/// a schema edited many times cannot grow the log without limit.
pub(in super::super) const MAX_BRANCH_CHECKPOINTS: usize = 8;

/// The branch checkpoints a compaction carries forward — the newest
/// [`MAX_BRANCH_CHECKPOINTS`] by write position, and the ids of those dropped.
///
/// "Newest by write position" is the closest thing to recently-used available
/// from the index, which holds locations and no timestamps. A checkpoint that is
/// re-read but not rewritten does not refresh, so an actively-used branch can in
/// principle age out — it costs one prefill and is written back, which is the
/// whole reason a cache can be evicted on a rough signal.
pub(in super::super) fn partition_branch_checkpoints(
    substrate: &Substrate,
) -> (Vec<(StreamId, RecordLoc)>, Vec<StreamId>) {
    let mut all: Vec<(StreamId, RecordLoc)> = substrate.branch_checkpoint_entries().collect();
    // Newest last in write order, so sort ascending and keep the tail.
    all.sort_by_key(|(_, loc)| (loc.segment, loc.offset));
    let dropped_n = all.len().saturating_sub(MAX_BRANCH_CHECKPOINTS);
    let dropped: Vec<StreamId> = all[..dropped_n].iter().map(|(sid, _)| *sid).collect();
    (all.split_off(dropped_n), dropped)
}

pub(super) fn branch_checkpoints_to_keep(substrate: &Substrate) -> Vec<(StreamId, RecordLoc)> {
    partition_branch_checkpoints(substrate).0
}
