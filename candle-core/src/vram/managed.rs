//! The capacity arithmetic the startup partition is sized from.
//!
//! There is one claim now, not two. [`VramGovernor::usable`] is what the device
//! reservation is sized from, and the reservation holds the KV side, the
//! transient tier and the expert cache between them — so there is no second
//! budget to keep in step with the first.
//!
//! `expert_budget()` was that second budget: `usable − kv_floor −
//! scratch_margin`, divided by `max_expert_size` to pick a resident-expert
//! count. It went with `kv_floor`. The count is now the weight zone's capacity,
//! `(span − MIN_ELASTIC_RESERVE) / slot_bytes`, computed once against a span
//! whose extent is a fact (`docs/elastic_vram_partition.md` §6).
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
