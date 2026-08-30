//! The weight zone's live geometry, published lock-free for readers that do not
//! own the cache.
//!
//! ## Why this exists
//!
//! `GpuDispatchTables` caches one raw device address per expert, captured once
//! at load on the reasoning that an all-resident cache's weights never move.
//! They do move: `WeightZone::retract_to` concedes slots at the frontier to the
//! KV side under pressure, the KV arena allocates that ground and writes to it,
//! and a cached address stops naming an expert weight. The GEMM then reads KV
//! bytes as weights — finite, plausibly shaped, and wrong — which surfaces as a
//! NaN several layers downstream rather than as a fault.
//!
//! The check for that (`GpuDispatchTables::zone_moved`) needs three numbers:
//! current capacity, current frontier address, and a monotonic count of
//! concessions. All three live in `ExpertCacheInner`, behind the pipeline's
//! mutex — and `inner` is **moved into the pipeline thread's state** when the
//! cache is built, so the handle that dispatches the forward cannot reach it.
//!
//! That is not a hypothetical gap. The check was originally written as
//! `if let PipelineMode::Inline { inner, .. } = &self.mode`, on the stated
//! reasoning that "an all-resident cache is `Inline` by construction". It is
//! not: `ExpertCache::new` builds the dispatch tables when `all_resident` and
//! then returns `PipelineMode::Threaded`, so the branch never matched and the
//! guard was dead code in precisely the configuration it was written for.
//!
//! ## Why atomics rather than another lock
//!
//! The reader is the forward, once per layer per wave, on the hot path. It must
//! not queue behind the pipeline thread's mutex — and it does not need to: the
//! three values are written together but read independently, and a reader that
//! catches a torn pair sees *some* disagreement with what it captured, which is
//! the answer it was going to act on anyway. The failure mode of a stale read is
//! one extra layer on the host path, not a missed concession, because
//! `concede_epoch` only ever increases and the reader compares for inequality.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

/// Live weight-zone geometry, shared between the cache's interior and its handle.
#[derive(Debug)]
pub struct ZoneGeometry {
    capacity: AtomicUsize,
    /// Address of the lowest slot — the boundary the KV side pushes against.
    frontier: AtomicU64,
    /// Monotonic count of concessions. **The load-bearing one.**
    ///
    /// Capacity and frontier both come back: the tier buys ground, the weight
    /// side concedes slots at the frontier, the tier stands on them and writes
    /// activations there, and the zone then grows back. Afterwards capacity and
    /// frontier read exactly as they did at load while the conceded slots hold
    /// tier leftovers. A count that only increases cannot be undone by a regrow,
    /// which is the entire reason it is here.
    concede_epoch: AtomicU64,
}

impl ZoneGeometry {
    /// Publish the geometry a freshly built zone has.
    pub fn new(capacity: usize, frontier: u64) -> Self {
        Self {
            capacity: AtomicUsize::new(capacity),
            frontier: AtomicU64::new(frontier),
            concede_epoch: AtomicU64::new(0),
        }
    }

    /// Republish capacity and frontier after the zone has changed shape.
    pub fn publish(&self, capacity: usize, frontier: u64) {
        self.capacity.store(capacity, Ordering::Release);
        self.frontier.store(frontier, Ordering::Release);
    }

    /// Record that ground has left the weight side.
    ///
    /// **Call this after [`publish`](Self::publish), never before.** A reader
    /// that sees the bumped epoch must not then read a pre-concession frontier
    /// and conclude the zone is where it left it. The ordering is the caller's
    /// to get right — the two values are separate atomics, so nothing here can
    /// enforce it.
    pub fn concede(&self) {
        self.concede_epoch.fetch_add(1, Ordering::AcqRel);
    }

    pub fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Acquire)
    }

    pub fn frontier(&self) -> u64 {
        self.frontier.load(Ordering::Acquire)
    }

    pub fn concede_epoch(&self) -> u64 {
        self.concede_epoch.load(Ordering::Acquire)
    }
}

#[cfg(test)]
mod tests {
    use super::ZoneGeometry;
    use std::sync::Arc;

    #[test]
    fn a_concession_is_visible_even_after_the_zone_grows_back() {
        // The regression this whole type exists for. Concede, then restore the
        // exact capacity and frontier the tables were built against: the two
        // geometry comparisons go quiet and only the epoch still disagrees.
        let g = ZoneGeometry::new(10_496, 0x505b800000);
        let (built_cap, built_frontier, built_epoch) =
            (g.capacity(), g.frontier(), g.concede_epoch());

        g.publish(10_000, 0x505c000000);
        g.concede();
        g.publish(built_cap, built_frontier);

        assert_eq!(g.capacity(), built_cap, "capacity came back");
        assert_eq!(g.frontier(), built_frontier, "frontier came back");
        assert_ne!(
            g.concede_epoch(),
            built_epoch,
            "the concession must remain visible after the regrow"
        );
    }

    #[test]
    fn a_clone_kept_by_the_handle_sees_the_owners_updates() {
        // The whole point of the type. The cache's interior is moved into the
        // pipeline thread's state, so the handle that dispatches the forward
        // holds only this clone — if it did not track, the staleness check
        // would read load-time numbers for ever and never refuse anything,
        // which is precisely the bug it replaced.
        let owner = Arc::new(ZoneGeometry::new(10_496, 0x505b800000));
        let handle = owner.clone();
        let moved: Arc<ZoneGeometry> = owner; // as if into the pipeline thread

        moved.publish(8_088, 0x51969c0000);
        moved.concede();

        assert_eq!(handle.capacity(), 8_088);
        assert_eq!(handle.frontier(), 0x51969c0000);
        assert_eq!(handle.concede_epoch(), 1);
    }

    #[test]
    fn the_epoch_only_ever_increases() {
        let g = ZoneGeometry::new(8, 0x1000);
        let mut last = g.concede_epoch();
        for _ in 0..5 {
            g.concede();
            let now = g.concede_epoch();
            assert!(now > last, "epoch must be monotonic: {now} after {last}");
            last = now;
        }
    }
}
