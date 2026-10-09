//! Settling the device once the work on it has gone.
//!
//! Between forwards the wave loop sweeps empty arenas, packs the KV pools when a
//! pass would pay, and lets the weight side take ground the KV side has stopped
//! using — a little per iteration, because every one of those sits between two
//! forwards. With nothing in flight there is nothing to pace against: a turn that
//! has left the device leaves its arenas empty and the pools full of holes, and
//! the weight side can only grow back over a frontier that packing has lowered.
//! [`Scheduler::settle_device`] runs the three to completion, in that order.
//!
//! It runs when the scheduler falls idle after doing work — and only then. A
//! request for it ([`super::SchedulerRequest::Settle`]), which is how a caller
//! that has just evicted a timeline learns the card is ready for the next
//! request, is answered by that same idle settle: none of the three is budgeted,
//! so run from the request drain beside live decodes it would hold every one of
//! them for as long as the passes took.

use candle::Device;
use candle_nn::kv_cache::{end_wave_transient, forget_last_tier};

use super::compaction_stall::FrontierPass;
use super::interleave::reseed_achievable_weight;
use super::Scheduler;

/// The partition as [`Scheduler::settle_device`] left it.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SettleReport {
    /// Regions any tenant holds — KV and record arenas and span tenants.
    pub live_regions: usize,
    /// One past the highest live region: where the weight side's ground begins
    /// to be reachable.
    pub frontier: usize,
    /// The weight zone's extent, bytes.
    pub weight_bytes: usize,
    /// Arenas handed back — swept empties and what packing emptied.
    pub arenas_released: usize,
    /// Negotiations the weight side was offered before it stopped taking ground.
    pub negotiations: usize,
}

/// The most compaction passes one settle runs. Each lowers the frontier by the
/// regions one tenant can give up, and a pass that cannot closes the gate behind
/// it, so this bounds only a frontier held alternately by tenants that each give
/// up a little.
const MAX_FRONTIER_PASSES: usize = 16;

/// The most negotiations one settle offers. Each grant is half of what is spare
/// (`growth_policy::kv_grow_step`), so the zone converges in about
/// `log2(spare / min_grant)` of them — a dozen for a 128K turn's ground — and
/// this only bounds a policy that never stops granting.
const MAX_NEGOTIATIONS: usize = 64;

/// Unchanged readings in a row that end the offer. Two, not one: the first
/// negotiation after a purchase, or after demand fell, is refused by the
/// policy's own one-negotiation guard (`GrowthPolicy::spare`), so a single flat
/// reading is that guard and not yet the end of what is spare.
const FLAT_NEGOTIATIONS: usize = 2;

/// Offer the weight side ground until it stops taking it. `negotiate` runs one
/// negotiation and answers the zone's extent after it; `start` is the extent
/// before the first. Answers how many negotiations ran.
fn grow_until_flat(mut negotiate: impl FnMut() -> usize, start: usize) -> usize {
    let mut last = start;
    let mut flat = 0;
    for n in 1..=MAX_NEGOTIATIONS {
        let now = negotiate();
        if now == last {
            flat += 1;
            if flat == FLAT_NEGOTIATIONS {
                return n;
            }
        } else {
            flat = 0;
            last = now;
        }
    }
    MAX_NEGOTIATIONS
}

impl Scheduler {
    /// Sweep the empty arenas, pack the pools and the span tenants, then let the
    /// weight side grow back into what that freed — with nothing in flight, so
    /// each runs to completion rather than to a per-iteration budget.
    ///
    /// **Only with nothing in flight.** Every step it takes is one the wave loop
    /// takes in its own gap, run to completion rather than to a budget, so it is
    /// called from the idle branch alone.
    ///
    /// **The last forward's tier is forgotten first.** The growth negotiation keeps
    /// that much ground free for the next forward of a prefill; with the work over
    /// there is none, and a turn that ended on a deep prefill chunk would otherwise
    /// hold the weight side back by its whole tier while the engine sat idle.
    pub(super) fn settle_device(&mut self) -> SettleReport {
        let _g = super::profile::span("loop:settle");
        if let Device::Cuda(d) = self.session.device() {
            let stream = d.cuda_stream();
            end_wave_transient(&stream);
            forget_last_tier(&stream);
        }
        let mut arenas_released = self.session.release_empty_arenas().unwrap_or(0);
        // Lower the frontier as far as the tenants standing at it allow: each pass is
        // whichever tenant holds the frontier now, until none could move it — the
        // gate's own answer, which a pass that made no progress closes behind it.
        for _ in 0..MAX_FRONTIER_PASSES {
            match self.frontier_pass() {
                None => break,
                Some((shape, FrontierPass::SpanTenants)) => self.pack_span_tenants_judged(shape),
                Some((_, FrontierPass::Kv)) => arenas_released += self.pack_until_settled(),
            }
        }
        let weight_now = |s: &Self| s.session.kv_region_stats().map_or(0, |r| r.weight_bytes);
        let start = weight_now(self);
        let negotiations = grow_until_flat(
            || {
                self.model.reclaim_spare_ground();
                weight_now(self)
            },
            start,
        );
        reseed_achievable_weight();
        let stats = self.session.kv_region_stats();
        let report = SettleReport {
            live_regions: stats.map_or(0, |r| r.live),
            frontier: stats.map_or(0, |r| r.live_watermark),
            weight_bytes: stats.map_or(0, |r| r.weight_bytes),
            arenas_released,
            negotiations,
        };
        tracing::info!(
            target: "candle_conversation::scheduler::vram_relief",
            live_regions = report.live_regions,
            frontier = report.frontier,
            weight_mib = report.weight_bytes >> 20,
            grown_mib = report.weight_bytes.saturating_sub(start) >> 20,
            arenas_released = report.arenas_released,
            negotiations = report.negotiations,
            "settled the device: empties swept, pools packed, weight side grown back",
        );
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The offer ends on the second flat reading in a row, never the first: the
    /// first refusal after a purchase is the policy's guard, and the halving
    /// grants behind it still have to be taken.
    #[test]
    fn the_offer_runs_until_two_negotiations_take_nothing() {
        // A zone with nothing spare: the guard, then a genuine refusal.
        let mut zone = [100usize, 100, 150].into_iter();
        assert_eq!(grow_until_flat(|| zone.next().unwrap(), 100), 2);

        // The guard, then halving grants: 100 (flat), 150, 175, 187, 187 (flat),
        // 187 (flat) — six negotiations, the guard's flat reading not counted
        // against the end.
        let mut zone = [100usize, 150, 175, 187, 187, 187, 190].into_iter();
        assert_eq!(grow_until_flat(|| zone.next().unwrap(), 100), 6);
    }

    /// A policy that never stops granting is bounded.
    #[test]
    fn a_policy_that_always_grants_is_bounded() {
        let mut zone = 0usize;
        let n = grow_until_flat(
            || {
                zone += 1;
                zone
            },
            0,
        );
        assert_eq!(n, MAX_NEGOTIATIONS);
    }
}
