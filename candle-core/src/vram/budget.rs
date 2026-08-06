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
    ///
    /// Sized for the *cold-start* peak, not steady state: section prefill plus
    /// calibration hold ~4.4 GiB of KV at once, while a warm daemon sits at
    /// ~1 GiB reserved / 0.66 GiB live. Every byte here is a byte the expert
    /// cache cannot have.
    ///
    /// 2.5 GiB was tried on the 16 GiB card and backed out. It did move 512 MiB
    /// into the expert budget (2509 → 2684 resident experts), and cold start did
    /// complete — but calibration raised **five** `budget exceeded` wave
    /// failures where the conservative setting raised none, so it operates in the
    /// region where wave steps fail and retry. The apparent calibration speedup
    /// that accompanied it (1654 s → 1077 s) is not attributable: model load in
    /// the same run improved 39 %, which this term cannot influence, so the two
    /// runs differed in machine state. Re-deriving it needs a back-to-back A/B on
    /// an otherwise idle machine.
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
    /// `C = min(frac × total, total − headroom_abs)` (default 1 GiB).
    ///
    /// This is a cap on what the balloon may *try*, not a reserve for the
    /// running engine. The balloon claims and touches real pages, so it stops on
    /// its own when the driver refuses — that refusal is the honest ceiling, and
    /// it is the whole point of ballooning: touching forces residency, which
    /// evicts other tenants' cold allocations on WDDM. A generous cap costs
    /// nothing when the card is smaller than it; a tight one stops the balloon
    /// before it ever reaches the ceiling, and the difference is simply never
    /// used by anybody.
    ///
    /// It was 2.5 GiB, on the reasoning that a forward's transient scratch peak
    /// must live above the resident set. That is true, and it is already
    /// reserved — [`Self::scratch_margin`] is subtracted in `expert_budget` and
    /// sits below every relief rung. Reserving it here as well booked the same
    /// bytes twice. Measured on the 16 GiB card: the cap bound at 13815 MiB and
    /// `C` came out 13488, while the real ceiling the balloon finds when allowed
    /// to look is **14592** — 1104 MiB that no allocator was permitted to touch,
    /// on a card where startup was failing for want of ~1 GiB.
    ///
    /// Note what now bounds the claim: with the cap out of the way the balloon
    /// grows until [`Self::balloon_floor`] (512 MiB) or the driver refuses,
    /// where the old 2.5 GiB cap stopped it with ~2.5 GiB still free. On a
    /// machine whose display is driven by this GPU, the measurement window is
    /// correspondingly tighter. That window is brief and the balloon frees
    /// everything it claimed, but `balloon_floor` — not this term — is the knob
    /// that governs it.
    pub balloon_headroom_abs: u64,
    /// Headroom floor at which the balloon stops growing (default 512 MiB).
    pub balloon_floor: u64,
    /// Balloon growth granularity in bytes (default 256 MiB).
    pub balloon_chunk: u64,
    /// Minimum wall-clock between two `Critical`-tier syncs, in milliseconds
    /// (default 250). Tests set 0 so every `Critical` proceeds deterministically.
    pub critical_min_interval_ms: u64,
}

/// The shipped defaults, in the units their `CANDLE_VRAM_*` overrides use, so
/// [`GovernorConfig::default`] and [`GovernorConfig::defaults_ignoring_env`]
/// cannot drift apart.
const DEFAULT_KV_FLOOR_MB: u64 = 3 * 1024;
const DEFAULT_KV_FLOOR_PCT: f64 = 0.15;
const DEFAULT_SCRATCH_MARGIN_MB: u64 = 1024;
const DEFAULT_BALLOON_FRAC: f64 = 0.95;
const DEFAULT_BALLOON_HEADROOM_MB: u64 = 1024;
const DEFAULT_BALLOON_FLOOR_MB: u64 = 512;
const DEFAULT_BALLOON_CHUNK_MB: u64 = 256;
const DEFAULT_CRITICAL_MIN_INTERVAL_MS: u64 = 250;

/// Trivial/Cheap engage early at zero hit-rate cost; Moderate/Costly (KV
/// eviction) are withheld until near the floor; Critical == floor.
const DEFAULT_LADDER: [LadderTier; 5] = [
    LadderTier::new(2 * GIB, 0.040),     // Trivial
    LadderTier::new(3 * GIB / 2, 0.030), // Cheap
    LadderTier::new(GIB, 0.015),         // Moderate
    LadderTier::new(GIB / 2, 0.005),     // Costly
    LadderTier::new(0, 0.0),             // Critical
];

impl GovernorConfig {
    /// The shipped defaults with **no** `CANDLE_VRAM_*` override applied.
    ///
    /// [`Default`] resolves the environment on top of these. A test that pins a
    /// shipped value must use this: the knobs are used routinely here, and a
    /// developer with one exported would otherwise see a defaults assertion fail
    /// for a reason unrelated to the code under test.
    pub fn defaults_ignoring_env() -> Self {
        Self {
            kv_floor_abs: DEFAULT_KV_FLOOR_MB * 1024 * 1024,
            kv_floor_pct: DEFAULT_KV_FLOOR_PCT,
            scratch_margin: DEFAULT_SCRATCH_MARGIN_MB * 1024 * 1024,
            ladder: DEFAULT_LADDER,
            balloon_target_frac: DEFAULT_BALLOON_FRAC,
            balloon_headroom_abs: DEFAULT_BALLOON_HEADROOM_MB * 1024 * 1024,
            balloon_floor: DEFAULT_BALLOON_FLOOR_MB * 1024 * 1024,
            balloon_chunk: DEFAULT_BALLOON_CHUNK_MB * 1024 * 1024,
            critical_min_interval_ms: DEFAULT_CRITICAL_MIN_INTERVAL_MS,
        }
    }
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            kv_floor_abs: env_bytes_mb("CANDLE_VRAM_KV_FLOOR_MB", DEFAULT_KV_FLOOR_MB),
            kv_floor_pct: env_f64("CANDLE_VRAM_KV_FLOOR_PCT", DEFAULT_KV_FLOOR_PCT),
            scratch_margin: env_bytes_mb("CANDLE_VRAM_SCRATCH_MARGIN_MB", DEFAULT_SCRATCH_MARGIN_MB),
            ladder: DEFAULT_LADDER,
            balloon_target_frac: env_f64("CANDLE_VRAM_BALLOON_FRAC", DEFAULT_BALLOON_FRAC),
            balloon_headroom_abs: env_bytes_mb(
                "CANDLE_VRAM_BALLOON_HEADROOM_MB",
                DEFAULT_BALLOON_HEADROOM_MB,
            ),
            balloon_floor: env_bytes_mb("CANDLE_VRAM_BALLOON_FLOOR_MB", DEFAULT_BALLOON_FLOOR_MB),
            balloon_chunk: env_bytes_mb("CANDLE_VRAM_BALLOON_CHUNK_MB", DEFAULT_BALLOON_CHUNK_MB),
            critical_min_interval_ms: DEFAULT_CRITICAL_MIN_INTERVAL_MS,
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

    /// Set a class's reserved tally outright, for footprints that are
    /// **established once and replaced wholesale** rather than accumulated:
    /// the dense weights and the expert cache's slot capacity, both fixed by a
    /// model load.
    ///
    /// `credit_class` is wrong for those. It adds, so loading a second model
    /// into the same process tallies both — and for `Weights` that is not just
    /// a cosmetic over-count: `kv_floor` is `abs + pct × (C − weights)`, so a
    /// doubled tally drives `C − weights` to zero and collapses the floor to
    /// `kv_floor_abs`, silently removing the KV reserve on the card where it
    /// matters most.
    pub fn set_class(&self, class: AllocClass, bytes: u64) {
        self.class_reserved[class.idx()].store(bytes, std::sync::atomic::Ordering::Relaxed);
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
