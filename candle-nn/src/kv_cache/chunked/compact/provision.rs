//! Creates the fresh low arenas that a pass moves frontier chunks into.
//!
//! One fresh arena goes to each pool that holds the frontier, and only for the KV
//! arenas above the first region another tenant holds, so each one emptied lowers
//! the frontier. An arena the free list places at or above the lowest region it
//! was meant to empty receives nothing and is released. `frontier_regions` gives
//! the frontier itself, one past the highest live region.

use super::*;

impl super::super::backing::ChunkedKvBacking {
    /// Give the pools holding the frontier fresh arenas as low in the span as the free
    /// list can place them, **the pool highest on the frontier first**, and add each to
    /// its pool's census so the walk plans into it.
    ///
    /// **One per KV arena holding the frontier, and only those.** The regions worth
    /// emptying are the KV arenas from the frontier down to the first region another
    /// tenant holds, paired with the holes below them (`region_pool::frontier_relocations`)
    /// — every pool among them, because packing is per pool and a pool gets no lower
    /// destination than its own lowest arena otherwise (every pool gapless, 131 arenas
    /// live and a frontier of 330 is a state a per-pool pack alone settles into). An
    /// arena standing under another tenant's region gets none: emptied, it would leave
    /// a hole under a frontier that tenant holds, which the next pass refills from the
    /// next arena down. Sized by "every pool, whatever lies above it", passes moved every
    /// chunk above any hole and the frontier stayed put — Qwen3-30B-A3B, 16 million chunk
    /// moves in one probe. `MAX_FRESH` bounds the claims one pass makes.
    ///
    /// **Why this order places them right.** `census_by_pool` is already sorted by each
    /// pool's highest arena, highest first, and the region free list is lowest-index
    /// first — so claiming in this order hands the lowest hole to the pool standing
    /// highest, the next hole to the next, and so on.
    ///
    /// A claim that lands at or above the lowest region it was meant to empty — another
    /// claim took the hole — ends the provisioning: that arena receives nothing, and the
    /// pass releases it as soon as the claims are in ([`FreshArenas::release_unused`]),
    /// or when [`FreshArenas`] drops on a pass that stops earlier.
    pub(super) fn provision_low_arenas(
        &self,
        census_by_pool: &mut [(ArenaKey, Vec<ArenaSlots>)],
        fresh: &mut FreshArenas<'_>,
    ) {
        /// Fresh arenas one pass may claim across every pool.
        const MAX_FRESH: usize = 64;
        // **Only for the arenas holding the frontier.** The KV arenas from the frontier
        // down to the first region another tenant holds, one per hole below them: each
        // emptied lowers the frontier. One below another tenant's region, emptied, is a
        // hole under a frontier that tenant still holds, which the next pass refills from
        // the next arena down — every chunk above any hole moved each pass, and the
        // frontier did not (Qwen3-30B-A3B: 16 million chunk moves in one probe).
        let held: HashSet<usize> = census_by_pool
            .iter()
            .flat_map(|(_, census)| census.iter().map(|a| a.rank))
            .collect();
        let relocate = self.regions_to_relocate(&held);
        let budget = relocate.len().min(MAX_FRESH);
        if budget == 0 {
            return;
        }
        // A fresh arena must land below the lowest region it is meant to empty; one the
        // free list placed at or above it receives nothing, and goes back unused.
        let cutoff = relocate[budget - 1];
        let to_empty: HashSet<usize> = relocate[..budget].iter().copied().collect();
        // Per pool, its arenas among those — one fresh arena each.
        let mut wanted: Vec<usize> = census_by_pool
            .iter()
            .map(|(_, census)| census.iter().filter(|a| to_empty.contains(&a.rank)).count())
            .collect();
        let mut made = 0usize;
        loop {
            let mut claimed_any = false;
            for (i, (key, census)) in census_by_pool.iter_mut().enumerate() {
                if census.is_empty() || wanted[i] == 0 {
                    continue;
                }
                if made >= budget {
                    return;
                }
                let arena_idx = match fresh.claim(&self.inner, *key) {
                    Ok(idx) => idx,
                    // No region for this pool is no region for any pool below it either.
                    Err(e) => {
                        tracing::debug!(
                            target: "candle_nn::kv_cache::compact",
                            ?key,
                            "no fresh low arena, packing the remaining pools without one: {e}",
                        );
                        return;
                    }
                };
                made += 1;
                claimed_any = true;
                let Some(slots) = self.fresh_arena_slots(*key, arena_idx) else {
                    wanted[i] = 0;
                    continue;
                };
                if slots.rank >= cutoff {
                    return;
                }
                wanted[i] -= 1;
                census.push(slots);
            }
            if !claimed_any {
                return;
            }
        }
    }

    /// One past the highest live region on this device — the frontier, in regions.
    pub fn frontier_regions(&self) -> Option<usize> {
        let candle::DeviceLocation::Cuda { gpu_id } = self.device().location() else {
            return None;
        };
        #[cfg(feature = "cuda")]
        return super::super::region_pool::region_stats(gpu_id).map(|s| s.live_watermark);
        #[cfg(not(feature = "cuda"))]
        {
            let _ = gpu_id;
            None
        }
    }
}
