//! A claimed region, returned to the free list when its handle is dropped.
//!
//! Holding a handle keeps its region out of circulation, so an arena owns its
//! handle and the free list stays correct whichever path drops the arena.
//! Dropping a handle does not mean the GPU has finished with the bytes, which is
//! why the next tenant waits for a quiesce. The drop stamps the quiesce epoch so
//! a claim can tell whether that wait is still owed. A region released twice
//! would be issued twice, so the drop panics rather than queue a duplicate or
//! quietly leak the region.

use super::*;

/// A claimed region, released back to the free list when dropped.
///
/// Holding one is what keeps a region out of circulation, so an [`Arena`] can
/// simply own its handle and the free list stays correct through every path
/// that drops an arena — release, truncate, or the backing going away.
///
/// **Dropping this does not mean the GPU is finished with the region.** It means
/// no host-side gid names it any more; kernels launched earlier can still be
/// reading those bytes. The wait belongs to whoever re-tenants the region, and
/// [`claim_region`] is where it lives — do not add a path that hands a released
/// region to a new owner without going through it.
///
/// [`Arena`]: super::arena::Arena
#[derive(Debug)]
pub(crate) struct RegionHandle {
    pub(super) ordinal: usize,
    pub(super) index: usize,
    pub(super) base: u64,
}

impl RegionHandle {
    /// Device address of the region's first byte.
    pub(crate) fn base(&self) -> u64 {
        self.base
    }

    /// Position of the region in the span. Rises left to right, so the highest
    /// index is the cheapest to evacuate (§3.10, problem 1).
    pub(crate) fn index(&self) -> usize {
        self.index
    }
}

impl Drop for RegionHandle {
    fn drop(&mut self) {
        let mut map = pools().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(pool) = map.get_mut(&self.ordinal) {
            // Stamp the quiesce epoch current at release. `claim_region` reads it
            // to decide whether this region still needs a wait, and its being
            // non-zero is what tells the next tenant to clean it at all.
            // **A region released twice is what puts one index on the free list
            // twice, and that is a double-issue waiting to happen.** The counts
            // cannot catch it: the duplicate pops twice, so `live` rises twice
            // and falls twice and `region_stats` still balances. Only the
            // per-index answer shows it, and it has to be checked *before* the
            // push, because after the push the damage is queued.
            //
            // **Panics.** `Drop` has nowhere to return an error to, and the two
            // quiet alternatives are both worse than stopping. Queuing the index
            // anyway hands the same ground to a second tenant, which writes over
            // live data and does not fault. Skipping the push instead keeps memory
            // safe but leaks a region and lets the process continue with its
            // bookkeeping known to be wrong — so the next occurrence is silent
            // too, and the run's results are no longer evidence of anything.
            //
            // There is no correct recovery here because the pool has already lost
            // track of an owner; whatever is wrong happened before this line. The
            // panic is the report.
            if !pool.owned[self.index] {
                panic!(
                    "reservation: region {} released while not held — the pool \
                     already had it free, so an owner has been lost. Releasing it \
                     again would queue one index twice and hand the same ground to \
                     two tenants, which writes over live data without faulting. \
                     {} live of {}, {} free.",
                    self.index,
                    pool.live,
                    pool.total,
                    pool.free.len(),
                );
            }
            pool.owned[self.index] = false;
            pool.dirt[self.index] = Dirt::any(pool.quiesce_epoch);
            pool.free.push(Reverse(self.index));
            pool.live -= 1;
            log::debug!("region {} released, {} live", self.index, pool.live);
        }
    }
}

/// [`Dirt::on`] for bytes any stream may still be touching.
pub(super) const ANY_STREAM: usize = 0;

/// What a region's previous tenant left behind: when, and who may still be at
/// it.
///
/// **The stream is what lets a claim skip the device-wide wait.** A released
/// KV arena may still be read by the persistence thread's copy stream, so only
/// a context-wide synchronise retires its readers. The wave transient tier is
/// written and read on the compute stream alone — every forward re-places it
/// over the same ground with no wait at all — so ground it leaves is retired,
/// for a claim on that stream, by stream order: the claim's clean is queued
/// behind the forward. Bytes left by tenants of both kinds within one epoch
/// count as any stream's.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Dirt {
    /// The quiesce epoch when its bytes last belonged to someone else; zero
    /// for never.
    pub(super) epoch: u64,
    /// The one stream whose work alone may still touch them (its raw handle),
    /// or [`ANY_STREAM`].
    pub(super) on: usize,
}

impl Dirt {
    /// Left by a tenant any stream may be reading, at `epoch`.
    pub(super) fn any(epoch: u64) -> Self {
        Self {
            epoch,
            on: ANY_STREAM,
        }
    }

    /// The same region left again, at `epoch`, by a tenant only `on` touches
    /// (or any stream, for [`ANY_STREAM`]): bytes still pending from an earlier
    /// tenant this epoch keep their own readers.
    pub(super) fn left(self, epoch: u64, on: usize) -> Self {
        let on = if self.epoch == epoch && self.on != on {
            ANY_STREAM
        } else {
            on
        };
        Self { epoch, on }
    }

    /// Whether a claim on `stream` at `epoch` must quiesce the device first.
    pub(super) fn needs_sync(self, epoch: u64, stream: usize) -> bool {
        self.epoch == epoch && (self.on == ANY_STREAM || self.on != stream)
    }
}

/// A stream's identity for [`Dirt::on`].
pub(super) fn stream_id(stream: &CudaStream) -> usize {
    stream.cu_stream() as usize
}
