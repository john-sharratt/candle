//! Offers the KV side's spare ground to the weight side and sweeps empty arenas.
//!
//! The weight side grows only from regions the KV side is not using now, less a
//! slack margin. It does not wait for a claim to fail before sweeping arenas
//! that have gone empty, since an unswept arena's regions would otherwise count
//! as demand. Each growth offer is bounded by the caller's own allocation unit,
//! and a negotiation never takes the whole offer. `SPARE_TALLY` records which
//! condition produced each zero, and whether the KV side's purchases succeed.

use super::*;

/// Arenas released by [`reclaim_empty_arenas`] since boot, and the calls that
/// found nothing. The ratio is what says whether a per-wave cadence is earning
/// its keep or just walking the registry.
pub(super) static SWEPT_ARENAS: AtomicU64 = AtomicU64::new(0);

pub(super) static SWEEP_CALLS: AtomicU64 = AtomicU64::new(0);

/// Why `spare_regions` answered as it did, plus whether the KV side's own
/// purchases succeed:
///
/// `[observing, pressure, occupancy_bound, regions_granted, buy_conceded,
///   buy_refused, fragmented]`
///
/// `observing`, `pressure`, `occupancy_bound` and `fragmented` attribute a zero
/// to one of the four things that can produce it, which is the difference
/// between "the mechanism is inert", "the ground is genuinely spoken for" and
/// "the ground is free but in holes the floor cannot reach". The two `buy_`
/// slots say whether a claim that waited would ever have anything to wait for.
///
/// There were four attributions while the weight side also measured against a
/// windowed history of KV demand; that term is gone (see
/// [`RegionPool::spare_regions`]) and its slot with it, rather than being left
/// to report a permanent zero under a name nothing can produce.
pub(super) static SPARE_TALLY: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// See [`SPARE_TALLY`].
pub fn spare_tally() -> [u64; 7] {
    std::array::from_fn(|i| SPARE_TALLY[i].load(Ordering::Relaxed))
}

/// Return every fully-empty arena's regions to the pool, now.
///
/// # Why this must not wait for a claim to fail
///
/// `live` in the region ledger counts regions **held by an arena**, empty or
/// not, and the elastic boundary's growth negotiation is built directly on it:
/// an unswept arena's regions are neither `live` demand the KV side is really
/// making nor free ground the weight side can be offered. An arena that went
/// chunk-empty three waves ago is therefore indistinguishable, to the one
/// mechanism that decides whether the weight side may take ground, from an arena
/// under active use.
///
/// The reactive sweeps do not cover this. `claim_region` sweeps only when the
/// free list is **empty**, and `place_transient` only when a placement has
/// already come up short — both are "we ran out" paths, and a workload with
/// spare regions never reaches either. It runs with `free 18` on the 3.6-35B
/// gate, so the reactive sweep never fires and empty arenas accumulate for a
/// whole config, inflating `live` and suppressing growth for the 60-second
/// window that reads it.
///
/// Cheap enough for a per-wave cadence by construction: each backing's pool
/// short-circuits on an atomic, so a registry with nothing to reclaim costs one
/// load per backing.
///
/// Returns arenas freed.
#[cfg(feature = "cuda")]
pub fn reclaim_empty_arenas() -> usize {
    let freed = super::super::backing::global_release_empty_arenas();
    SWEPT_ARENAS.fetch_add(freed as u64, Ordering::Relaxed);
    SWEEP_CALLS.fetch_add(1, Ordering::Relaxed);
    freed
}

/// `(arenas released, sweeps run)` since boot — see [`reclaim_empty_arenas`].
pub fn empty_sweep_stats() -> (u64, u64) {
    (
        SWEPT_ARENAS.load(Ordering::Relaxed),
        SWEEP_CALLS.load(Ordering::Relaxed),
    )
}

/// **The negotiation, from the KV side — and only in the growing direction.**
///
/// Regions the KV side is holding free beyond `slack` that the weight side could
/// take back, bounded by [`KV_GROW_STEP`].
///
/// It used to answer `(wanted, spare)`, where `wanted` drained an accumulated
/// count of refused claims. That half is gone, and its removal is the point.
/// **A counter records events; the weight side spent them as regions.** One
/// failed section-quantize drain walked the size-class ladder and left 4,436
/// units of demand behind it against a KV side that was twenty-eight regions
/// short — and the boundary paid all of it, evicting 1,598 experts and taking the
/// zone to a capacity below its own pinned working set, from which every
/// subsequent forward failed. Now a claim buys exactly the ground it needs at the
/// moment it needs it ([`set_ground_broker`]), so there is nothing to accumulate
/// and nothing to convert: the allocation *is* the measurement.
///
/// `min_grant` is the caller's own allocation unit in regions — the smallest
/// grant it can actually spend. A grant below it is not conservative, it is
/// wasted: see [`kv_grow_step`].
/// State the free regions `ordinal`'s KV side keeps in hand — its scheduler's
/// relief target — so the weight side's growth leaves them standing (see
/// [`FreeRegionTarget`]). Called once, as the scheduler starts. A device with no
/// reservation has nothing to divide, and is left alone.
pub fn set_kv_free_target(ordinal: usize, target: FreeRegionTarget) {
    let mut map = pools().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(pool) = map.get_mut(&ordinal) {
        pool.kv_target = target;
    }
}

/// State whether prompts are waiting to be prefilled on `ordinal`'s card, so the
/// weight side's growth leaves the next prefill's tier standing between prefills
/// (see [`Occupancy::waiting_tier_bytes`]). Called by the scheduler once per pass,
/// before the pass issues any forward. When none wait, the tier their prefills
/// stood is forgotten and the weight side may grow back into it.
pub fn set_prompts_waiting(ordinal: usize, waiting: bool) {
    let mut map = pools().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(pool) = map.get_mut(&ordinal) {
        pool.prompts_waiting = waiting;
        if !waiting {
            pool.waiting_tier_bytes = 0;
        }
    }
}

pub fn kv_spare_regions(
    stream: &std::sync::Arc<CudaStream>,
    slack: usize,
    min_grant: usize,
) -> Result<usize> {
    with_pool(stream, |pool| {
        let spare = pool.spare_regions(slack);
        Ok(kv_grow_step(spare, min_grant))
    })
}
