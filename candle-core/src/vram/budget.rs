//! The startup budget: configuration, and the KV floor as an absolute plus a
//! percentage of what is left after weights.
//!
//! Nothing here is a predicted footprint. `kv_floor` is derived from the
//! balloon-measured capacity `C` and the resident weight bytes — the only two
//! things known before inference — and everything else is observed from the live
//! measurement (see `docs/vram_governor_design.md` §7).
//!
//! This also held the relief ladder's trip points: five `LadderTier` rungs of
//! `kv_floor + abs + pct × C`, each the headroom at which one rung engaged. They
//! went with the ladder (`docs/archived/arena_unification.md` §5).

use super::{AllocClass, VramGovernor};

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
    /// Cushion left outside the KV reservation, for the allocations that still
    /// come from the CUDA pool: the gallery arena's slabs, the grow-only
    /// sampler / provenance / MoE-routing scratches, and the threaded expert
    /// pipeline's per-layer combine target (default 1 GiB).
    ///
    /// It was "the cushion the first forward's scratch lands in". That is no
    /// longer what it holds — a forward's activations come from the
    /// reservation's transient tier now — and while it was still described that
    /// way it got subtracted a second time when sizing the KV span, against a
    /// transient tier that was then added back on top. Same bytes, two places,
    /// opposite signs.
    ///
    /// **Now re-derived against the pool, as that fix asked for.** Measured on
    /// the daemon across a full cold boot and six concurrent conversations: the
    /// CUDA pool reserves **once** during load (30 MiB → 7,232 MiB, three
    /// distinct values in 60 samples) and **never grows again** — not through
    /// the cold-ingest peak, not under concurrency. Its `used` swings ~370 MiB
    /// *inside* what it already holds, so serving draws nothing from this
    /// cushion.
    ///
    /// At 512 MiB the card peaks at 14,309 MiB of 16,376 under six concurrent
    /// conversations, which puts the process ~512 MiB below `capacity_c`
    /// (14,592) — that is this cushion, **untouched at peak**. It is not sized
    /// against serving, which needs none of it; it covers pool growth between
    /// `expert_budget` being computed and load finishing, and that window is
    /// what the clean cold boot at this value evidences.
    ///
    /// Every MiB beyond it is a MiB neither the expert cache nor the KV side can
    /// use. (The further ~1.8 GiB the card shows free is not claimable: it is
    /// where the driver refused the balloon, which is what makes `capacity_c`
    /// 14,592 rather than 16,376.)
    pub scratch_margin: u64,
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
}

/// The shipped defaults, in the units their `CANDLE_VRAM_*` overrides use, so
/// [`GovernorConfig::default`] and [`GovernorConfig::defaults_ignoring_env`]
/// cannot drift apart.
/// Sized against the **cold-ingest high-water mark**, not the steady state.
///
/// Measured on the 16 GiB card against the real `mind` corpus: a cold boot
/// climbs to **284 live regions (4,544 MiB)** while the system prompt's
/// collections prefill, then falls to **69 regions (1,104 MiB)** and stays
/// there. Steady state needs a sixth of what the boot transient does, and it is
/// the transient that decides whether the daemon comes up at all — at
/// `3 GiB + 15 %` the span is 218 regions and boot wedges with every region
/// occupied, relief freeing nothing because the content is pinned and not yet
/// warm-backed.
///
/// Measured span against this term, cold-booting the same `mind` corpus, with
/// decode probed at matched context depth (kv/fwd 550–736):
///
/// | `kv_floor_abs` | KV span | expert slots | residency | decode | boot |
/// |---|---|---|---|---|---|
/// | 3 GiB | 218 regions | 2618 | 42.6 % | — | dies: retry storm, no forwards |
/// | 4 GiB | 274 regions | 2267 | 36.9 % | — | ready |
/// | 5 GiB | 328 regions | 1917 | 31.2 % | 67 ms/fwd | clean |
/// | 6 GiB | 384 regions | 1566 | 25.5 % | 80 ms/fwd | clean, 100 regions unused |
///
/// **Every MiB here is a MiB the expert cache does not get, and the expert
/// cache is what pays for decode**: 1024 MiB of this term buys 56 KV regions and
/// costs 351 expert slots, worth ~16 % of decode throughput per step of the
/// table. So this is sized to what KV *needs*, never to what is spare.
///
/// Steady-state KV is **70 regions (1,120 MiB)**. What forces the term above
/// that is the cold-boot transient: the system prompt's collections must stay
/// resident until the last section that prefills over them is built, so the hot
/// set peaks at **284 regions** and only then collapses. That peak is structural
/// — see `docs/archived/arena_unification_results.md` for why it cannot be offloaded away
/// — and it is what this term has to clear.
///
/// The transient tier gave back 512 MiB (a shelf backing no allocator, and a
/// 512 MiB migration staging span against a 29,696 B peak), so this term buys
/// 32 more regions than it used to for the same expert budget.
///
/// **This term moved with `scratch_margin`, not against the expert cache.**
/// `expert_budget` is `usable − kv_floor − scratch_margin`, so raising this by
/// exactly what `scratch_margin` gave up (512 MiB) leaves the expert budget
/// identical while the KV span — `usable − scratch_margin − transient` — gains
/// the whole amount. 32 regions for nothing, which is what pays for clearing
/// the 284-region cold-boot peak.
///
/// Spending it there rather than on more experts is what the decode curve says
/// to do:
///
/// | expert slots | 1566 | 1917 | 2267 | 2355 | 2443 |
/// |---|---|---|---|---|---|
/// | decode median | 80 ms | 67 ms | 57 ms | 63 ms | 57 ms |
///
/// The first three are a real trend. The last three are **one flat band inside
/// run-to-run noise** — 2355 measuring slower than 2267 is what says so, since
/// residency cannot make decode worse. So the expert cache stops paying
/// somewhere near 2,300 slots on this model and slots past that are free to
/// spend, while a span clearing the 284-region peak removes a reproducible
/// first-boot casualty (the post-priming quantize drain, which needs a slot to
/// compress a section into before it can release the native one).
///
/// The percentage term tracks the card; the absolute term tracks the corpus,
/// and only the card is knowable here. A workspace with one very large
/// collection needs `CANDLE_VRAM_KV_FLOOR_MB` raised — the failure names the
/// exhausted reservation, so the symptom points at this knob.
const DEFAULT_KV_FLOOR_MB: u64 = 4352;
const DEFAULT_KV_FLOOR_PCT: f64 = 0.15;
const DEFAULT_SCRATCH_MARGIN_MB: u64 = 512;
const DEFAULT_BALLOON_FRAC: f64 = 0.95;
/// Tracks [`DEFAULT_SCRATCH_MARGIN_MB`], which is the invariant
/// `the_balloon_reserve_does_not_double_book_the_scratch_cushion` pins: this is
/// a cap on what the balloon may *try*, not a second reserve, so anything above
/// the cushion books the same bytes twice.
///
/// It does not bind on this card in any case — `C = min(frac × total, total −
/// this)` targets 15,864 MiB of the 16,376 MiB card, while the balloon actually
/// stops at 14,592 MiB where the driver refuses. The refusal is the honest
/// ceiling; this only stops the balloon reaching for something absurd.
const DEFAULT_BALLOON_HEADROOM_MB: u64 = 512;
const DEFAULT_BALLOON_FLOOR_MB: u64 = 512;
const DEFAULT_BALLOON_CHUNK_MB: u64 = 256;

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
            balloon_target_frac: DEFAULT_BALLOON_FRAC,
            balloon_headroom_abs: DEFAULT_BALLOON_HEADROOM_MB * 1024 * 1024,
            balloon_floor: DEFAULT_BALLOON_FLOOR_MB * 1024 * 1024,
            balloon_chunk: DEFAULT_BALLOON_CHUNK_MB * 1024 * 1024,
        }
    }
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            kv_floor_abs: env_bytes_mb("CANDLE_VRAM_KV_FLOOR_MB", DEFAULT_KV_FLOOR_MB),
            kv_floor_pct: env_f64("CANDLE_VRAM_KV_FLOOR_PCT", DEFAULT_KV_FLOOR_PCT),
            scratch_margin: env_bytes_mb(
                "CANDLE_VRAM_SCRATCH_MARGIN_MB",
                DEFAULT_SCRATCH_MARGIN_MB,
            ),
            balloon_target_frac: env_f64("CANDLE_VRAM_BALLOON_FRAC", DEFAULT_BALLOON_FRAC),
            balloon_headroom_abs: env_bytes_mb(
                "CANDLE_VRAM_BALLOON_HEADROOM_MB",
                DEFAULT_BALLOON_HEADROOM_MB,
            ),
            balloon_floor: env_bytes_mb("CANDLE_VRAM_BALLOON_FLOOR_MB", DEFAULT_BALLOON_FLOOR_MB),
            balloon_chunk: env_bytes_mb("CANDLE_VRAM_BALLOON_CHUNK_MB", DEFAULT_BALLOON_CHUNK_MB),
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

    /// **The VRAM the KV subsystem owns**, transient tier included, and so the
    /// reserve the expert budget must leave free: `3 GiB + 15% × (C − Weights)`.
    ///
    /// The expert loader takes `usable − kv_floor − scratch_margin`, so what
    /// survives to the first KV cache is `kv_floor + scratch_margin`; the
    /// reservation then claims exactly `kv_floor` across its two sides and
    /// leaves the cushion on the pool. `region_pool`'s
    /// `the_reservation_claims_exactly_the_kv_floor` pins that identity.
    ///
    /// It is no longer a *floor* in the evict-no-further sense — nothing evicts
    /// against a watermark any more (`docs/archived/arena_unification.md` §5). It is the
    /// partition knob, and step 7 tunes it.
    pub fn kv_floor(&self) -> u64 {
        let c = self.capacity();
        let weights = self.class_reserved(AllocClass::Weights);
        let base = c.saturating_sub(weights);
        self.config
            .kv_floor_abs
            .saturating_add((self.config.kv_floor_pct * base as f64) as u64)
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
