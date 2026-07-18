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
    pub fn expert_budget(&self) -> Result<u64> {
        let headroom = self.probe.read()?.headroom;
        Ok(headroom
            .saturating_sub(self.kv_floor())
            .saturating_sub(self.scratch_margin()))
    }
}
