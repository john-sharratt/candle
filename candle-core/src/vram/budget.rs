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
    /// **The one number the card holds back**, in bytes (default 512 MiB).
    ///
    /// `C = min(headroom, total − this)`, on the balloon's growth loop and on
    /// the fast path that skips it alike ([`super::balloon::capacity_target`]).
    /// Everything the engine owns is sized from `C`, so this is the working
    /// margin left to the display driver and the OS — and it is left
    /// permanently, not just during the measurement.
    ///
    /// It replaces three terms that were trying to be this one:
    ///
    /// - `balloon_target_frac` (0.95). A fraction of the card is not a fact
    ///   about anything — "leave 5%" is 818 MiB on a 16 GiB card and 4.9 GiB on
    ///   a 96 GiB one, for no reason that scales with what the OS actually
    ///   needs. It was also applied *only* as the fast path's threshold, never
    ///   as a cap, so on an uncontended card no reserve was applied at all.
    /// - `balloon_headroom_abs` (512 MiB). The same quantity as this, but
    ///   documented as "a cap on what the balloon may try, not a reserve for the
    ///   running engine" — which is exactly the distinction that let the fast
    ///   path ignore it.
    /// - `balloon_floor` (512 MiB). A headroom floor for the growth loop. Two
    ///   numbers cannot both be "the amount we leave"; and stopping on live
    ///   headroom defeats the purpose of ballooning, which is to drive headroom
    ///   down so WDDM evicts other tenants' cold pages. The refusal is the
    ///   honest ceiling, and the balloon frees everything it claimed within
    ///   milliseconds.
    pub capacity_reserve: u64,
    /// Balloon growth granularity in bytes (default 256 MiB).
    pub balloon_chunk: u64,
    /// Smallest chunk the balloon refines down to on refusal (default 2 MiB).
    ///
    /// The reservation's granule size: below it a claim cannot be expressed in
    /// the allocator `C` is being measured *for*, so refining further would
    /// measure memory that could never be mapped. Refining from
    /// [`Self::balloon_chunk`] to here costs at most three extra failed
    /// allocations and removes an up-to-256 MiB under-measurement that would
    /// otherwise be permanent.
    pub balloon_min_chunk: u64,
}

/// The shipped defaults, in the units their `CANDLE_VRAM_*` overrides use, so
/// [`GovernorConfig::default`] and [`GovernorConfig::defaults_ignoring_env`]
/// cannot drift apart.
///
/// **`kv_floor_abs` and `kv_floor_pct` are gone.** They were the static
/// partition: `4352 MiB + 15% × (C − weights)` reserved for KV, with the expert
/// cache taking what was left. The measurement that justified them is worth
/// keeping, because it is also the argument for replacing them — this is what a
/// megabyte of the term cost, cold-booting the `mind` corpus on the 16 GiB card
/// with decode probed at matched context depth:
///
/// | `kv_floor_abs` | KV span | expert slots | decode | boot |
/// |---|---|---|---|---|
/// | 3 GiB | 218 regions | 2618 | — | dies: retry storm, no forwards |
/// | 4 GiB | 274 regions | 2267 | — | ready |
/// | 5 GiB | 328 regions | 1917 | 67 ms/fwd | clean |
/// | 6 GiB | 384 regions | 1566 | 80 ms/fwd | clean, 100 regions unused |
///
/// 1024 MiB bought 56 KV regions and cost 351 expert slots. The term had to
/// clear the **cold-boot transient** — 284 live regions (4,544 MiB) while the
/// system prompt's collections prefill — and the daemon then spent its life at
/// **70 regions (1,120 MiB)**. So roughly 3.4 GiB sat reserved for a peak that
/// happened once, on the card where the expert cache is what pays for decode.
///
/// A single number cannot be right for both moments, which is the whole reason
/// the boundary moves now (`docs/elastic_vram_partition.md` §1).
const DEFAULT_SCRATCH_MARGIN_MB: u64 = 512;
/// The working margin left to the display driver and the OS, permanently.
///
/// Distinct from [`DEFAULT_SCRATCH_MARGIN_MB`], which is memory *we* keep
/// outside the reservation for our own CUDA pool. This one is memory we never
/// claim at all. They are the same size by coincidence, not by derivation, and
/// the invariant that used to tie them together
/// (`the_balloon_reserve_does_not_double_book_the_scratch_cushion`) was pinning
/// a relationship that only existed because the old term was documented as "a
/// cap on what the balloon may try" rather than as a reserve.
///
/// This absolute reserve is NOT the WDDM residency margin — that is the OS's
/// per-process budget, which the balloon reads from the probe and applies as
/// a cap on every path (see `balloon::capacity_target`'s `headroom` cap). A
/// fixed number cannot stand in for the budget: 5 GiB was measured right on
/// the 73 GiB card and would take a third of a 16 GiB one.
const DEFAULT_CAPACITY_RESERVE_MB: u64 = 512;
const DEFAULT_BALLOON_CHUNK_MB: u64 = 256;
/// One VMM allocation granule on every device this runs on. `Reservation`
/// queries the real value at run time; the balloon cannot, because it runs
/// before any reservation exists, and 2 MiB is small enough that being wrong
/// about it costs at most 2 MiB of measurement precision.
const DEFAULT_BALLOON_MIN_CHUNK_MB: u64 = 2;

impl GovernorConfig {
    /// The shipped defaults with **no** `CANDLE_VRAM_*` override applied.
    ///
    /// [`Default`] resolves the environment on top of these. A test that pins a
    /// shipped value must use this: the knobs are used routinely here, and a
    /// developer with one exported would otherwise see a defaults assertion fail
    /// for a reason unrelated to the code under test.
    pub fn defaults_ignoring_env() -> Self {
        Self {
            scratch_margin: DEFAULT_SCRATCH_MARGIN_MB * 1024 * 1024,
            capacity_reserve: DEFAULT_CAPACITY_RESERVE_MB * 1024 * 1024,
            balloon_chunk: DEFAULT_BALLOON_CHUNK_MB * 1024 * 1024,
            balloon_min_chunk: DEFAULT_BALLOON_MIN_CHUNK_MB * 1024 * 1024,
        }
    }
}

impl Default for GovernorConfig {
    fn default() -> Self {
        Self {
            scratch_margin: env_bytes_mb(
                "CANDLE_VRAM_SCRATCH_MARGIN_MB",
                DEFAULT_SCRATCH_MARGIN_MB,
            ),
            capacity_reserve: env_bytes_mb(
                "CANDLE_VRAM_CAPACITY_RESERVE_MB",
                DEFAULT_CAPACITY_RESERVE_MB,
            ),
            balloon_chunk: env_bytes_mb("CANDLE_VRAM_BALLOON_CHUNK_MB", DEFAULT_BALLOON_CHUNK_MB),
            balloon_min_chunk: env_bytes_mb(
                "CANDLE_VRAM_BALLOON_MIN_CHUNK_MB",
                DEFAULT_BALLOON_MIN_CHUNK_MB,
            ),
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
    /// happened elsewhere, e.g. model weights or expert slots). **Reporting
    /// only** — no partition is derived from it, and the availability gate is
    /// always the live measurement.
    pub fn credit_class(&self, class: AllocClass, bytes: u64) {
        self.class_reserved[class.idx()].fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
    }

    /// Set a class's reserved tally outright, for footprints that are
    /// **established once and replaced wholesale** rather than accumulated:
    /// the dense weights and the expert cache's slot capacity, both fixed by a
    /// model load.
    ///
    /// `credit_class` is wrong for those. It adds, so loading a second model
    /// into the same process tallies both, and every report of the card's
    /// decomposition is then wrong by a whole model.
    ///
    /// The tally no longer *sizes* anything — `kv_floor` was the last reader and
    /// it is gone. In particular the span is **not** `usable − Weights`: the
    /// weights are already inside `usable`, and subtracting the tally on top
    /// would book them twice (`region_pool::span_from`).
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

    /// The cushion left **outside** the reservation, for the CUDA pool.
    ///
    /// The reservation takes `usable − this`. It is not held back from anything
    /// else — the KV side, the transient tier and the expert cache are all
    /// inside the span, and the boundary between them moves.
    pub fn pool_cushion(&self) -> u64 {
        self.config.scratch_margin
    }

    /// Config accessor (diagnostics / tests).
    pub fn config(&self) -> &GovernorConfig {
        &self.config
    }
}
