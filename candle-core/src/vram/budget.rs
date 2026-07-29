//! The evolving budget: configuration, the KV floor (absolute + percentage), and
//! the relief-ladder thresholds (also absolute + percentage).
//!
//! Nothing here is a predicted footprint. `kv_floor` and the ladder thresholds
//! are derived from the balloon-measured capacity `C` and the resident weight
//! bytes — the only two things known before inference — and everything else is
//! observed from the live measurement (see `docs/vram_governor_design.md` §7–§8).

use super::{AllocClass, Criticality, VramGovernor};

const GIB: u64 = 1024 * 1024 * 1024;

/// One rung of the relief ladder: the headroom margin *above the KV floor* at
/// which this tier engages, expressed as an absolute term plus a fraction of the
/// total capacity `C`. The hybrid keeps margins sane on a 16 GiB card (the
/// absolute term dominates) and scaling on a 73 GiB card (the percentage term
/// dominates).
#[derive(Clone, Copy, Debug)]
pub struct LadderTier {
    pub abs: u64,
    pub pct: f64,
}

impl LadderTier {
    pub const fn new(abs: u64, pct: f64) -> Self {
        Self { abs, pct }
    }
    /// Bytes of margin above the floor for capacity `c`.
    pub fn margin(&self, c: u64) -> u64 {
        self.abs.saturating_add((self.pct * c as f64) as u64)
    }
}

/// Static configuration for a governor. Defaults encode the reviewed decisions
/// (`docs/vram_governor_design.md` §15); every field is overridable for tests
/// and per-card-class tuning.
#[derive(Clone, Debug)]
pub struct GovernorConfig {
    /// KV floor absolute term (default 3 GiB).
    pub kv_floor_abs: u64,
    /// KV floor fraction of `(C − Weights)` (default 0.15).
    pub kv_floor_pct: f64,
    /// Cushion left above the KV floor when computing the expert budget so the
    /// first forward's scratch lands before any KV eviction (default 1 GiB).
    pub scratch_margin: u64,
    /// The five ladder rungs, indexed by [`Criticality`]. `Critical` is always
    /// `{0, 0.0}` — it engages exactly at the floor.
    pub ladder: [LadderTier; 5],
    /// Fraction of total VRAM the balloon tries to claim (default 0.95).
    pub balloon_target_frac: f64,
    /// Absolute headroom (bytes) the balloon always leaves below `total`,
    /// combined with [`Self::balloon_target_frac`] as
    /// `C = min(frac × total, total − headroom_abs)` (default 2.5 GiB). The
    /// transient prefill/reprojection scratch peak a forward needs above the
    /// resident set is driven by the model's activation footprint, NOT the card
    /// size, so expressing it purely as a fraction is scale-wrong: 5% of a 16 GiB
    /// card is only ~0.8 GiB (too little — the scratch peak pages under WDDM),
    /// while lowering the fraction to fix that would waste ~10 GiB on a 72 GiB
    /// card. The absolute term binds on small cards (16 GiB, the minimum we
    /// support) and the fraction binds on large cards (5% > 2.5 GiB past ~50 GiB),
    /// so neither case is penalised.
    pub balloon_headroom_abs: u64,
    /// Headroom floor at which the balloon stops growing (default 512 MiB).
    pub balloon_floor: u64,
    /// Balloon growth granularity in bytes (default 256 MiB).
    pub balloon_chunk: u64,
    /// Minimum wall-clock between two `Critical`-tier syncs, in milliseconds
    /// (default 250). Tests set 0 so every `Critical` proceeds deterministically.
    pub critical_min_interval_ms: u64,
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            kv_floor_abs: env_bytes_mb("CANDLE_VRAM_KV_FLOOR_MB", 3 * 1024),
            kv_floor_pct: env_f64("CANDLE_VRAM_KV_FLOOR_PCT", 0.15),
            scratch_margin: env_bytes_mb("CANDLE_VRAM_SCRATCH_MARGIN_MB", 1024),
            // Trivial/Cheap engage early at zero hit-rate cost; Moderate/Costly
            // (KV eviction) are withheld until near the floor; Critical == floor.
            ladder: [
                LadderTier::new(2 * GIB, 0.040),     // Trivial
                LadderTier::new(3 * GIB / 2, 0.030), // Cheap
                LadderTier::new(GIB, 0.015),         // Moderate
                LadderTier::new(GIB / 2, 0.005),     // Costly
                LadderTier::new(0, 0.0),             // Critical
            ],
            balloon_target_frac: env_f64("CANDLE_VRAM_BALLOON_FRAC", 0.95),
            balloon_headroom_abs: env_bytes_mb("CANDLE_VRAM_BALLOON_HEADROOM_MB", 2560),
            balloon_floor: env_bytes_mb("CANDLE_VRAM_BALLOON_FLOOR_MB", 512),
            balloon_chunk: env_bytes_mb("CANDLE_VRAM_BALLOON_CHUNK_MB", 256),
            critical_min_interval_ms: 250,
        }
    }
}

fn env_bytes_mb(key: &str, default_mb: u64) -> u64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<u64>().ok())
        .unwrap_or(default_mb)
        .saturating_mul(1024 * 1024)
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|s| s.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite() && *v >= 0.0)
        .unwrap_or(default)
}

impl VramGovernor {
    /// The measured resident capacity from the balloon (`C`). `0` until the
    /// balloon has run — callers treat `0` as "unknown, budget not yet live".
    pub fn capacity(&self) -> u64 {
        self.capacity_c.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Loose per-class reserved tally — for the budget table and forecast, never
    /// an availability gate (that is always the live measurement).
    pub fn class_reserved(&self, class: AllocClass) -> u64 {
        self.class_reserved[class.idx()].load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Record `bytes` reserved under `class` (an allocation that already
    /// happened elsewhere, e.g. model weights or expert slots). Reporting +
    /// `kv_floor` base only — the availability gate is always the live measurement.
    pub fn credit_class(&self, class: AllocClass, bytes: u64) {
        self.class_reserved[class.idx()].fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    /// Decrement a class's loose reserved tally when a tracked allocation is
    /// freed (e.g. a KV arena released, an expert slot evicted). Reporting only —
    /// the availability gate is always the live measurement.
    pub fn debit_class(&self, class: AllocClass, bytes: u64) {
        let cell = &self.class_reserved[class.idx()];
        let mut cur = cell.load(std::sync::atomic::Ordering::Relaxed);
        loop {
            let next = cur.saturating_sub(bytes);
            match cell.compare_exchange_weak(
                cur,
                next,
                std::sync::atomic::Ordering::Relaxed,
                std::sync::atomic::Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(observed) => cur = observed,
            }
        }
    }

    /// The KV minimum working set we never evict below, and the reserve the
    /// expert budget must leave free: `3 GiB + 15% × (C − Weights)`.
    pub fn kv_floor(&self) -> u64 {
        let c = self.capacity();
        let weights = self.class_reserved(AllocClass::Weights);
        let base = c.saturating_sub(weights);
        self.config
            .kv_floor_abs
            .saturating_add((self.config.kv_floor_pct * base as f64) as u64)
    }

    /// The headroom trip point for `tier`: `kv_floor + abs + pct × C`. When live
    /// headroom drops to/below this, the tier engages.
    pub fn tier_threshold(&self, tier: Criticality) -> u64 {
        let c = self.capacity();
        self.kv_floor()
            .saturating_add(self.config.ladder[tier.idx()].margin(c))
    }

    /// The scratch cushion held above the floor when sizing experts (§11).
    pub fn scratch_margin(&self) -> u64 {
        self.config.scratch_margin
    }

    /// Config accessor (diagnostics / tests).
    pub fn config(&self) -> &GovernorConfig {
        &self.config
    }
}
