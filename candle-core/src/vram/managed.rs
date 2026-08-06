//! Managed allocation, the pressure-retry path, and the concurrency forecast.
//!
//! The gate is always the live measurement, never a running tally. `reserve` is
//! for permanent classes (weights, and the fixed part of scratch/experts) that
//! just need a class tag; `allocate` is for the variable/expert path that must
//! survive transient exhaustion by relieving and retrying (see
//! `docs/vram_governor_design.md` §9).

use super::{AllocClass, Criticality, VramGovernor};
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
    /// Record a permanent allocation of `bytes` under `class` and run `alloc`.
    /// No prediction, no gate — the class tag drives evictability/reporting only;
    /// the bytes update the loose per-class tally. If `alloc` fails it is
    /// surfaced verbatim (a permanent allocation that won't fit is a
    /// configuration error, not a runtime pressure event).
    pub fn reserve<T>(
        &self,
        class: AllocClass,
        bytes: u64,
        alloc: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        let v = alloc()?;
        self.credit_class(class, bytes);
        Ok(v)
    }

    /// Allocate `bytes` under `class`, retrying through the relief ladder on
    /// out-of-memory. On success credits the per-class tally. On repeated OOM it
    /// escalates one rung per round up to `Critical`, then surfaces a typed OOM
    /// (the circuit breaker — it never spins). Non-OOM errors propagate at once.
    pub fn allocate<T>(
        &self,
        class: AllocClass,
        bytes: u64,
        mut alloc: impl FnMut() -> Result<T>,
    ) -> Result<T> {
        // First attempt.
        match alloc() {
            Ok(v) => {
                self.credit_class(class, bytes);
                return Ok(v);
            }
            Err(e) if !is_oom(&e) => return Err(e),
            Err(_) => {}
        }
        // Escalate: relieve one rung deeper each round, retry after each.
        for tier in Criticality::ALL {
            self.run_tier_with_sync(class, tier, bytes);
            match alloc() {
                Ok(v) => {
                    self.credit_class(class, bytes);
                    return Ok(v);
                }
                Err(e) if !is_oom(&e) => return Err(e),
                Err(_) => {}
            }
        }
        Err(Error::Msg(format!(
            "vram governor: out of memory allocating {bytes} B for {class:?} after full relief ladder"
        )))
    }

    /// How many concurrent units of `per_unit_kv_bytes` fit right now, counting
    /// live headroom **plus** the KV that can be reversibly evicted (up to
    /// `Moderate` — never lossy/critical, so it never plans on damaging the
    /// cache). The scheduler uses this as the ceiling for its admission window.
    pub fn forecast_units(&self, per_unit_kv_bytes: u64) -> usize {
        let headroom = self.probe.read().map(|r| r.headroom).unwrap_or(0);
        let evictable = self.evictable_estimate(Criticality::Moderate);
        (headroom.saturating_add(evictable) / per_unit_kv_bytes.max(1)) as usize
    }

    /// The bytes available to hold MoE experts resident, computed *now* from the
    /// live measurement (mandatory weights already loaded), leaving the KV floor
    /// and the scratch cushion free. The expert loader divides this by
    /// `max_expert_size` to pick how many slots to keep resident (§11).
    ///
    /// Bounded by the balloon-measured capacity `C`, not by live headroom alone.
    /// `headroom` is what the driver reports free, and on a WDDM card that is
    /// materially more than what can actually be held resident — the balloon
    /// exists precisely to find that difference. Sizing against headroom spends
    /// it: measured on a 16 GiB card, `C` was 13488 MiB while headroom at expert
    /// load was ~15000 MiB, so the cache took 8888 MiB (3065 slots) where the
    /// capacity allowed 6493 (2187), and every later allocation ran into a pool
    /// whose `used` sat above `C` with the driver still reporting free memory.
    /// Startup never finished: section prefill and calibration together need
    /// ~4.4 GiB of KV, and the overshoot left them under 1 GiB. The expert cache is
    /// permanent — nothing reclaims it, no relief rung can shed a slot — so an
    /// overshoot here is not transient pressure, it is a card that never fits its
    /// own workload again.
    ///
    /// What we have already spent of `C` is the **drop in headroom since `C` was
    /// measured**, not `total - headroom`. DXGI reports
    /// `headroom = Budget - CurrentUsage`, so `total - headroom` is
    /// `(total - Budget) + CurrentUsage` — and the first term is the OS reserve,
    /// which the balloon already discovered and excluded from `C`. Subtracting it
    /// again double-books it and hands the expert cache ~1 GiB less than the card
    /// allows. Differencing two headroom readings cancels the reserve: it is
    /// present in both.
    ///
    /// The `Weights` tally can't serve as the spend either — the dense weights
    /// finish loading *after* the expert cache is sized, so it reads zero here.
    ///
    /// Falls back to headroom alone when `C` was never measured, or when the
    /// baseline is missing/stale (headroom above the baseline means memory came
    /// back, so nothing of `C` is spent).
    pub fn expert_budget(&self) -> Result<u64> {
        let reading = self.probe.read()?;
        let capacity = self.capacity();
        let baseline = self.headroom_at_capacity();
        let usable = if capacity == 0 || baseline == 0 {
            reading.headroom
        } else {
            let spent_by_us = baseline.saturating_sub(reading.headroom);
            reading.headroom.min(capacity.saturating_sub(spent_by_us))
        };
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
                capacity_mib = capacity / (1024 * 1024),
                "expert budget is zero — no experts will stay resident and every \
                 forward will stream them over PCIe"
            );
        }
        Ok(budget)
    }
}
