//! The capacity arithmetic the startup partition is sized from.
//!
//! Both permanent claims a model load makes — the expert cache and the KV
//! reservation — come from [`VramGovernor::usable`], so the accounting lives
//! once and the two cannot disagree about how much of `C` is left.
//!
//! This file also held the managed-allocation path: `reserve` for permanent
//! class-tagged allocations, and `allocate`, which retried an out-of-memory
//! allocation while escalating one relief rung per round up to `Critical`. Both
//! are gone with the ladder — nothing called them, because the allocation that
//! actually needed to survive transient exhaustion was KV, and KV no longer
//! allocates at all.

use super::VramGovernor;
use crate::{Error, Result};

/// True if `err` looks like a device out-of-memory (raw driver OOM, or the KV
/// subsystem's budget-exceeded marker). Kept in candle-core so the governor
/// needs no upward dependency; the substrings cover both paths.
pub fn is_oom(err: &Error) -> bool {
    let s = err.to_string();
    s.contains("out of memory")
        || s.contains("OUT_OF_MEMORY")
        || s.contains("CUDA_ERROR_OUT_OF_MEMORY")
        || s.contains("VRAM budget exceeded")
        || s.contains("kv-cache GPU VRAM budget exceeded")
}

impl VramGovernor {
    /// The bytes available to hold MoE experts resident, computed *now* from the
    /// live measurement (mandatory weights already loaded), leaving the KV floor
    /// and the scratch cushion free. The expert loader divides this by
    /// `max_expert_size` to pick how many slots to keep resident (§11).
    ///
    /// Bounded by [`Self::usable`] — the balloon-measured capacity `C`, not live
    /// headroom alone. Sizing against headroom overshoots: measured on a 16 GiB
    /// card, `C` was 13488 MiB while headroom at expert load was ~15000 MiB, so
    /// the cache took 8888 MiB (3065 slots) where the capacity allowed 6493
    /// (2187), and every later allocation ran into a pool whose `used` sat above
    /// `C` with the driver still reporting free memory. Startup never finished:
    /// section prefill and calibration together need ~4.4 GiB of KV, and the
    /// overshoot left them under 1 GiB. The expert cache is permanent — nothing
    /// reclaims it, no relief rung can shed a slot — so an overshoot here is not
    /// transient pressure, it is a card that never fits its own workload again.
    pub fn expert_budget(&self) -> Result<u64> {
        let usable = self.usable()?;
        let budget = usable
            .saturating_sub(self.kv_floor())
            .saturating_sub(self.scratch_margin());
        // A zero budget is not a small budget: the loader keeps no expert
        // resident and every forward streams all of them over PCIe. That is a
        // configuration failure (floor + cushion exceed what is left of `C`),
        // and it is otherwise indistinguishable from a card that is merely
        // tight, so say it rather than let the model come up silently crippled.
        if budget == 0 {
            tracing::warn!(
                target: "candle_core::vram",
                usable_mib = usable / (1024 * 1024),
                kv_floor_mib = self.kv_floor() / (1024 * 1024),
                scratch_margin_mib = self.scratch_margin() / (1024 * 1024),
                capacity_mib = self.capacity() / (1024 * 1024),
                "expert budget is zero — no experts will stay resident and every \
                 forward will stream them over PCIe"
            );
        }
        Ok(budget)
    }

    /// What is left of the measured capacity `C` right now: live headroom,
    /// bounded by `C` less what we have already spent of it.
    ///
    /// Both permanent claims a model load makes — the expert cache and the KV
    /// reservation — are sized from here, so the accounting lives once.
    ///
    /// What we have spent of `C` is the **drop in headroom since `C` was
    /// measured**, not `total - headroom`. DXGI reports
    /// `headroom = Budget - CurrentUsage`, so `total - headroom` is
    /// `(total - Budget) + CurrentUsage` — and the first term is the OS reserve,
    /// which the balloon already discovered and excluded from `C`. Subtracting
    /// it again double-books it and costs ~1 GiB on a 16 GiB card. Differencing
    /// two headroom readings cancels the reserve: it is present in both.
    ///
    /// The `Weights` tally can't serve as the spend either — the dense weights
    /// finish loading *after* the expert cache is sized, so it reads zero there.
    ///
    /// Falls back to headroom alone when `C` was never measured, or when the
    /// baseline is missing/stale (headroom above the baseline means memory came
    /// back, so nothing of `C` is spent).
    pub fn usable(&self) -> Result<u64> {
        let reading = self.probe.read()?;
        let capacity = self.capacity();
        let baseline = self.headroom_at_capacity();
        Ok(if capacity == 0 || baseline == 0 {
            reading.headroom
        } else {
            let spent_by_us = baseline.saturating_sub(reading.headroom);
            reading.headroom.min(capacity.saturating_sub(spent_by_us))
        })
    }
}
