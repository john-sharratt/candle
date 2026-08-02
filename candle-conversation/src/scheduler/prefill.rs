use super::*;
use crate::persistence::thread::effective_turn_policy;
use crate::substrate::ConvCompression;
use crate::token_buffer::TokenBuffer;
use candle_transformers::models::batched_inference::PendingGlue;
use std::collections::{HashMap, HashSet};

/// Default pool-budget headroom we keep free by offloading hot KV — see
/// [`vram_budget_band`]. 2 GiB: sized **above** a wide ragged prefill forward's
/// transient allocation peak (per-sequence activations × batch width + MoE
/// expert gather), which on a memory-tight card is far larger than a lone
/// decode's. Relieving only near ~1 GiB let a 20-wide upload forward's peak tip
/// the card into WDDM host-memory spill — tens of seconds per forward — so we now
/// keep a wider margin and shed hot KV earlier to defend it.
const DEFAULT_VRAM_BUDGET_BAND_MB: usize = 2048;

/// Pool-budget headroom kept free (bytes), overridable at process start via
/// `CANDLE_VRAM_BUDGET_BAND_MB` so it can be tuned to a specific card/model without
/// a rebuild — the right value depends on the model's per-token activation
/// footprint and the prefill batch width, which vary per deployment. Cached on
/// first read. `0`/unparseable falls back to [`DEFAULT_VRAM_BUDGET_BAND_MB`].
fn vram_budget_band() -> usize {
    static BAND: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BAND.get_or_init(|| {
        let mb = std::env::var("CANDLE_VRAM_BUDGET_BAND_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&mb| mb > 0)
            .unwrap_or(DEFAULT_VRAM_BUDGET_BAND_MB);
        mb * 1024 * 1024
    })
}
/// Decode-phase pressure/relief reserve band (bytes), overridable via
/// `CANDLE_VRAM_DECODE_BAND_MB`. Default 1.5 GiB — thinner than the load band
/// because in decode the working set is stable (~1 chunk/token/step), so the
/// freed-float free-list (`reserved − used`) is genuinely spare and counts as
/// available; we keep only a small safety margin for the per-step forward +
/// MoE expert gather, letting the maximum KV stay resident (the whole point of
/// unbounded context). `0`/unparseable falls back to the default.
const DEFAULT_VRAM_DECODE_BAND_MB: usize = 1536;
fn vram_decode_band() -> usize {
    static BAND: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *BAND.get_or_init(|| {
        let mb = std::env::var("CANDLE_VRAM_DECODE_BAND_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&mb| mb > 0)
            .unwrap_or(DEFAULT_VRAM_DECODE_BAND_MB);
        mb * 1024 * 1024
    })
}

/// The phase a VRAM pressure/relief decision is made in. The freed-float
/// free-list (`pool_reserved − pool_used`) is genuinely reusable working space,
/// but *what it is available FOR* differs by phase, so the reserve band does too:
///
/// - [`Load`](VramPhase::Load) — bringing KV into VRAM *before* attention
///   (prefill upload, section/scope ingest, warm→hot elevation). The free-list
///   is the **destination** of the incoming KV — the forward will consume it —
///   and a wide ragged forward has a large transient activation peak. So the
///   free-list is not spare admission capacity here: we keep the wide band
///   (`max(capacity/10, 2 GiB)`) and relieve early, making real headroom before
///   the load competes for it. (We still *count* the free-list in `available`,
///   so the WDDM false-pressure fix stands — the phase only sets the band.)
/// - [`Decode`](VramPhase::Decode) — the working set is stable, so the free-list
///   is genuinely spare: a thin band ([`vram_decode_band`]) keeps KV maximally
///   resident.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum VramPhase {
    Load,
    Decode,
}

/// Pool reuse headroom (reserved-but-free pool bytes) below which the
/// stream-ordered pool can no longer absorb a new allocation without growing
/// our OS footprint — only then does a low driver `free` count as pressure
/// (mirrors `vram_has_room`'s `os_needed`). A fresh arena is small, so unlike
/// the budget band this stays tight; it is the OS-safety floor, not the
/// keep-forwards-fast headroom.
const VRAM_REUSE_BAND: usize = 512 * 1024 * 1024;
/// Bytes of hot KV to shed per pressure episode. The eviction overshoots the
/// trigger by this much, so the pool budget oscillates in
/// `[band, band + VRAM_EVICT_BAND]` (band = [`vram_budget_band`]) and we don't
/// re-trip on the very next wave.
const VRAM_EVICT_BAND: u64 = 1024 * 1024 * 1024;
/// Eviction hysteresis for the footprint reclaim, as **percentages of resident
/// capacity C** so they scale to any card (a 16 GB 4090 and a 72 GB workstation
/// alike — never an absolute GiB margin sized for one machine). Eviction of
/// resident KV starts only when `used` climbs within [`vram_evict_high_pct`] of C
/// (genuinely too much resident data — not the gap-inflated `reserved`), and then
/// evicts in ONE bulk pass down to [`vram_evict_low_pct`] below C, so it's a rare
/// decisive exit that coasts for many waves rather than per-wave nibbling (which
/// caused reload-churn stalls). Both env-tunable; `low` must exceed `high` or
/// eviction no-ops. The fragmented gap worth compacting is likewise a percentage
/// of C ([`VRAM_MIN_COMPACT_GAP_PCT`]).
const DEFAULT_VRAM_EVICT_HIGH_PCT: usize = 8; // evict when used > 92% of C
const VRAM_EVICT_HIGH_FLOOR_MB: usize = 1024;
const DEFAULT_VRAM_EVICT_LOW_PCT: usize = 20; // bulk-evict down to 80% of C
const VRAM_EVICT_LOW_FLOOR_MB: usize = 2048;
/// Below this fraction of C the reserved-but-free gap is just trim slack —
/// compaction (GPU chunk moves + a device sync) isn't worth it.
const VRAM_MIN_COMPACT_GAP_PCT: usize = 3;
const VRAM_MIN_COMPACT_GAP_FLOOR_MB: usize = 512;
/// Min interval between footprint-reclaim attempts. Without it, a `reserved`
/// pinned just over the compact-ceiling — a fragmented gap the engine keeps
/// reusing, which compaction can't lower — trips the pressure gate on every
/// scheduler-loop iteration and fires relief 5–7×/second with no effect. Relief
/// still runs when genuinely needed, just no faster than this.
const FOOTPRINT_RELIEF_COOLDOWN: std::time::Duration = std::time::Duration::from_millis(2000);
/// Hysteresis: the footprint gate ignores `reserved` overshooting the compact-
/// ceiling by less than this — a small overage is just the pool high-water
/// sitting harmlessly below the card, not worth relieving. Fixed (not % of C):
/// it's a "don't sweat a tiny overage" band, small on any card.
const FOOTPRINT_HYSTERESIS: usize = 256 * 1024 * 1024;

/// How long prefill throughput must be COMPLETELY silent (no forward
/// completing) under surviving VRAM pressure before the promote path halves
/// the admission window. Longer than any healthy forward (the widest
/// calibration forwards run ~7 s), so completions keep the width; a genuine
/// wedge still backs off, one halving per grace period. Device-OOM shrinks at
/// its own site instantly.
const PROMOTE_STALL_GRACE: std::time::Duration = std::time::Duration::from_secs(15);

fn env_pct(var: &str, default: usize, max: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&p| p >= 1 && p <= max)
        .unwrap_or(default)
}
fn vram_evict_high_pct() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        env_pct(
            "CANDLE_VRAM_EVICT_HIGH_PCT",
            DEFAULT_VRAM_EVICT_HIGH_PCT,
            50,
        )
    })
}
fn vram_evict_low_pct() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_pct("CANDLE_VRAM_EVICT_LOW_PCT", DEFAULT_VRAM_EVICT_LOW_PCT, 90))
}

/// A reclaim watermark margin (bytes) below resident capacity C: `max(pct% of C,
/// floor)`. The percentage scales to any card; the absolute floor keeps the
/// margin sane on a small (16 GB) card where a raw percentage would be too tight.
fn cap_margin(capacity: usize, pct: usize, floor_mb: usize) -> usize {
    (capacity / 100 * pct).max(floor_mb * 1024 * 1024)
}

/// Capacity fraction (%) at which cold **ingest** KV starts demoting to the warm
/// (RAM) tier — the gentle-early ladder rung, far below the near-cap eviction
/// watermarks (`vram_evict_high_pct` etc.). Ingest KV is zero-reload-cost (never
/// re-attended until query time; it re-elevates warm→hot on demand), so it is the
/// cheapest relief and sheds first. Env `CANDLE_INGEST_DEMOTE_PCT`, default 50.
fn ingest_demote_pct() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_pct("CANDLE_INGEST_DEMOTE_PCT", 50, 95))
}
/// Target hot→warm drain backlog, as a % of resident capacity, above which
/// ingest admission throttles down (and below half of which it reopens). This
/// is the *leading* backpressure signal — it keeps `used` off the warm-starved
/// climb before the lagging VRAM-pressure trip ever fires. Env
/// `CANDLE_INGEST_WARM_BACKLOG_PCT`, default 12 (≈ one-to-two passes of headroom
/// on a ~72 GiB card), clamped to 40.
fn ingest_warm_backlog_pct() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_pct("CANDLE_INGEST_WARM_BACKLOG_PCT", 12, 40))
}
/// Backlog (as a % of resident capacity) above which the wave loop blocks on a
/// device sync after its eviction callbacks — "heavy pressure". Draining the
/// primary stream lets the (now cross-layer-batched, short) hot→warm pass run
/// uncontended by ingest forwards, so it catches up instead of interleaving.
/// Above the throttle target ([`ingest_warm_backlog_pct`]) so the gentle AIMD
/// throttle acts first; the sync is the harder stop when that isn't enough. Env
/// `CANDLE_INGEST_SYNC_CEILING_PCT`, default 25, clamped to 80. Set to a high
/// value to effectively disable.
fn ingest_sync_ceiling_pct() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| env_pct("CANDLE_INGEST_SYNC_CEILING_PCT", 25, 80))
}
/// Free host RAM (as a % of total) below which ingest admission throttles to
/// protect the hot→warm migration. The warm (RAM) KV tier grows as sealed turns
/// migrate off the GPU; if it outpaces warm→cold demotion it exhausts host RAM
/// and the migration's contiguous staging buffer can't be allocated (the OOM
/// that aborted a full overnight load). Kept deliberately **below** the warm
/// purge's own free target (`max(2 GiB, 5% × total)`) so it fires only when the
/// purge has fallen behind, not in steady state. Floored at 2 GiB — several
/// 512 MiB migration batches of headroom. Env `CANDLE_INGEST_HOST_RAM_FLOOR_PCT`,
/// default 2, clamped to 20.
fn host_ram_floor_bytes(total_ram: u64) -> u64 {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let pct = *V.get_or_init(|| env_pct("CANDLE_INGEST_HOST_RAM_FLOOR_PCT", 2, 20)) as u64;
    std::cmp::max(2 * 1024 * 1024 * 1024, total_ram / 100 * pct)
}
/// Minimum spacing between OS memory probes for host-RAM backpressure —
/// `sysinfo` is a syscall, so the scheduler caches the reading between waves.
const HOST_RAM_PROBE_INTERVAL: std::time::Duration = std::time::Duration::from_millis(1000);
/// Sealed ingest turns kept hot per timeline (the rolling window) before the
/// gentle-early demote sheds the rest to RAM. Env `CANDLE_INGEST_HOT_WINDOW`,
/// default 8.
///
/// **Must cover the ingest projection's gather width.** With the tool-round-trip
/// ingest, each scope's summary decode projects the `scopes` group (`top_k` turns)
/// — i.e. an actively-ingesting conversation RE-ATTENDS its own recent turns every
/// scope. If this window is narrower than that gather, the demote sheds turns the
/// very next projection re-elevates: a warm↔hot churn that stalls the decode batch.
/// The scopes group is `top_k: 4`, so a scope's projected working set is ~4 turns
/// (2 coupled turns × ~2 scopes); 8 keeps a couple of scopes of margin resident so
/// the active working set never leaves hot.
fn ingest_hot_window() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("CANDLE_INGEST_HOT_WINDOW")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(8)
    })
}

/// Max float bytes the synchronous compress-to-free rung brings forward per relief
/// episode. Bounds the per-episode stall: a large accumulated backlog drains over
/// several episodes (plus the background persistence thread) instead of one
/// multi-second blocking compression of *everything* pending. This is a WORK/time
/// budget — compression cost scales with turns × chunks × layers (~model
/// dependent, not card capacity) — so it is an absolute MB, env-tunable. 1 GiB.
const DEFAULT_VRAM_COMPRESS_MAX_MB: usize = 1024;
/// The rung compresses `want × this` per episode (clamped to the max above), so
/// it overshoots the immediate shortfall a little and coasts rather than
/// re-tripping on the very next wave.
const VRAM_COMPRESS_HYSTERESIS: u64 = 4;
/// Base chunk-move budget for one bounded compaction pass. The relief ladder
/// escalates it as it climbs — a light defrag first (`reclaim_footprint`, Cheap
/// rung), a bigger one deeper (Costly rung) — so a large fragmented gap
/// consolidates over several bounded passes instead of one long blocking
/// compaction (a 20 s `compact_forced` was the symptom). Env-tunable; the
/// `candle_nn::kv_cache::compact` timing log shows the moves→ms ratio to tune it.
const DEFAULT_COMPACT_BASE_MOVES: usize = 4000;
fn compact_base_moves() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        std::env::var("CANDLE_VRAM_COMPACT_BASE_MOVES")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&n| n > 0)
            .unwrap_or(DEFAULT_COMPACT_BASE_MOVES)
    })
}
fn vram_compress_max() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        (std::env::var("CANDLE_VRAM_COMPRESS_MAX_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&mb| mb > 0)
            .unwrap_or(DEFAULT_VRAM_COMPRESS_MAX_MB) as u64)
            * 1024
            * 1024
    })
}
/// Safety cap on the synchronous substrate-offload flush under pressure. The
/// pass migrates hot→warm *before* its cold-disk writes, so the warm copies
/// the eviction needs exist well before this fires — a timeout only clips the
/// tail of the cold-write wait (turns are already evictable) and guards against
/// a wedged persistence thread; it is not the expected path.
const VRAM_OFFLOAD_FLUSH_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(5);

impl Scheduler {
    /// Promote up to `MAX_ACTIVE_PREFILLS - active_prefills.len()` newly
    /// submitted PrefillWorks from the FIFO queue into the in-flight
    /// `active_prefills` set. Emits the initial `Prefill` and
    /// `PrefillProgress(0, total)` events so callers see their submission
    /// was picked up.
    /// Under VRAM pressure, shed hot KV to the substrate to reopen budget, and
    /// report whether pressure **survived** the attempt. Three steps:
    ///
    ///  1. Give substrate offload complete priority: synchronously drain the
    ///     pending hot→warm migration so just-sealed turns gain a warm (RAM) copy
    ///     — only warm-backed turns can be evicted hot→warm.
    ///  2. Evict the least-recently-used hot turns across the resident
    ///     conversations (drop the hot copy, keep warm), returning their VRAM
    ///     chunks to the pool free-list.
    ///  3. Release the arenas the eviction emptied back to the pool, which is what
    ///     actually lowers `pool_used` and restores budget. The cheap release only
    ///     frees fully-empty arenas; if that leaves pressure unrelieved and a
    ///     forced defrag could reclaim a fragmented arena, fall back to the
    ///     chunk-moving compaction (guarded by `can_reclaim_arena`) rather than let
    ///     an admitted prefill spill to host memory and thrash to death.
    ///
    /// Returns `true` if VRAM is **still** under pressure afterward — the caller's
    /// signal to narrow the admission window and stop admitting this pass.
    /// `whence` tags the log line with the calling gate (`promote` / `pump`);
    /// `phase` sets the reserve band (see [`VramPhase`]).
    pub(super) fn relieve_vram_pressure(&mut self, whence: &str, phase: VramPhase) -> bool {
        let t = std::time::Instant::now();
        // Footprint reclaim FIRST: when the pool's RESERVED footprint nears the
        // card, the budget-based ladder below short-circuits as "already relieved"
        // (the reuse-inflated budget still reads healthy), so it would never act.
        // Defrag the fragmented gap / bulk-evict resident KV back under capacity
        // here, measured against physical capacity rather than the budget. No-op
        // when comfortably below both watermarks.
        self.reclaim_footprint();
        // The VRAM Governor owns the relief POLICY — measure the honest free VRAM,
        // run the ladder cheapest-rung-first, escalate only while below the restore
        // target, GPU-sync only at Critical. We execute each rung's MECHANISM via a
        // borrowed driver (`SchedulerReliefDriver`), so hot eviction runs on this
        // (the scheduler) thread with full working-set context — never off-thread
        // racing a live forward. See `docs/vram_governor_design.md` §8.
        let band = self.vram_band_for(phase) as u64;
        let (freed, deepest, flushed, evicted, compressed, released) =
            if let Some(gov) = self.session.vram_governor() {
                // Restore headroom to band + 50% — enough to clear pressure with
                // margin, but reachable in a single eviction pass (2× band asked for
                // ~9 GiB on a 73 GiB card, which one pass can't free, so the ladder
                // reported `relieved=false` and pointlessly climbed to Critical).
                let target = band.saturating_add(band / 2);
                let mut driver = SchedulerReliefDriver {
                    sched: self,
                    evicted: crate::substrate::EvictionReport { count: 0, bytes: 0 },
                    compressed: 0,
                    flushed: false,
                    released: 0,
                };
                let freed = gov
                    .relieve_with(target, &mut driver)
                    .map(|r| r.freed())
                    .unwrap_or(0);
                let deepest = gov.last_relief().map(|(tier, _)| tier);
                let SchedulerReliefDriver {
                    evicted,
                    compressed,
                    flushed,
                    released,
                    ..
                } = driver;
                (freed, deepest, flushed, evicted, compressed, released)
            } else {
                // Non-CUDA / no governor: direct cheapest-first ladder.
                let mut released = self.session.release_empty_arenas().unwrap_or(0);
                self.trim_kv_pool();
                if self.vram_under_pressure_for(phase) && self.session.can_reclaim_arena() {
                    let _ = self.session.defragment_bounded(compact_base_moves());
                    released += self.session.release_empty_arenas().unwrap_or(0);
                    self.trim_kv_pool();
                }
                let mut flushed = false;
                let mut evicted = crate::substrate::EvictionReport { count: 0, bytes: 0 };
                if self.vram_under_pressure_for(phase) {
                    flushed = super::timed_wait(|| {
                        self.persist_trigger
                            .flush_blocking(VRAM_OFFLOAD_FLUSH_TIMEOUT)
                    });
                    evicted = self.evict_cold_tail(VRAM_EVICT_BAND);
                    released += self.session.release_empty_arenas().unwrap_or(0);
                    self.trim_kv_pool();
                }
                // No governor ⇒ non-CUDA / no compress-to-free (quantize is CUDA-only).
                (released as u64, None, flushed, evicted, 0, released)
            };
        let still = self.vram_under_pressure_for(phase);
        // (Eviction volume is accounted at the `evict_cold_tail` chokepoint, not
        // here, so the governor driver's evictions aren't double-counted.)
        if let Some((free, total)) = self.session.vram_free_total() {
            let (pool_used, pool_reserved) = self.session.vram_pool_stats().unwrap_or((0, 0));
            let offload_ms = t.elapsed().as_millis() as u64;
            let evicted_mib = evicted.bytes / (1 << 20);
            let freed_mib = freed / (1 << 20);
            let rung = deepest
                .map(|r| format!("{r:?}"))
                .unwrap_or_else(|| "none".into());
            // Our footprint vs the OS-reserved high-water: this, not `vram_free`,
            // says what's actually consuming the card.
            let pool_used_mib = (pool_used / (1 << 20)) as u64;
            let pool_reserved_mib = (pool_reserved / (1 << 20)) as u64;
            let vram_free_mib = (free / (1 << 20)) as u64;
            let vram_total_mib = (total / (1 << 20)) as u64;
            let relieved = !still;
            // INFO only when the pass actually freed something — an eviction that
            // reclaimed hot turns or arenas is a real event worth surfacing. When it
            // was a no-op (nothing evictable, pressure persists) log at DEBUG: this
            // runs from both the promote and pump gates every scheduler loop, so
            // under a sustained upload burst an unconditional INFO floods the log.
            macro_rules! emit {
                ($lvl:ident) => {
                    tracing::$lvl!(
                        target: "candle_conversation::scheduler::timing",
                        whence,
                        deepest_rung = %rung,
                        freed_mib,
                        offload_ms,
                        warm_flushed = flushed,
                        turns_compressed = compressed,
                        turns_evicted = evicted.count,
                        evicted_mib,
                        arenas_released = released,
                        pool_used_mib,
                        pool_reserved_mib,
                        vram_free_mib,
                        vram_total_mib,
                        relieved,
                        "VRAM relief (governor ladder)"
                    )
                };
            }
            if evicted.count > 0 || compressed > 0 || released > 0 {
                emit!(info);
            } else {
                emit!(debug);
            }
        }
        still
    }

    /// Per-rung relief trace: emit a DEBUG line each time a ladder rung actually
    /// recovers VRAM, naming the rung + mechanism, the shortfall it was asked to
    /// cover (`want`), the bytes it recovered, and a rung-specific detail
    /// (arenas released, turns compressed/evicted, pool-used delta). The
    /// per-episode INFO summary in [`relieve_vram_pressure`] aggregates these; this
    /// is the granular breakdown for diagnosing which mechanism did the work.
    fn log_relief_event(
        &self,
        tier: &str,
        mechanism: &str,
        want: u64,
        freed: u64,
        dur_ms: u64,
        detail: String,
    ) {
        tracing::debug!(
            target: "candle_conversation::scheduler::vram_relief",
            tier,
            mechanism,
            want_mib = want / (1 << 20),
            freed_mib = freed / (1 << 20),
            dur_ms,
            %detail,
            "VRAM relief rung acted"
        );
    }

    pub(super) fn promote_new_prefills(&mut self) {
        // Ragged prefill forward width — how many in-flight prefills coalesce into
        // one forward: a burst of small parallel scopes (code_read's worker count),
        // a bulk collection ingest's per-section prefills (`insert_section_collection`
        // fires one prefill per section), or a batch of calibration cases. Capped by
        // the AIMD `admit_window`, which narrows the batch under VRAM pressure so the
        // forward's transient peak (which scales with width) can't OOM a busy card;
        // `MIN_PREFILL_WIDTH` keeps ≥1 in flight regardless. (Decode-side waves are
        // bounded separately, e.g. calibration's `CALIBRATION_BATCH`.)
        let cap = Self::MAX_PREFILL_WIDTH.min(self.admit_window.max(Self::MIN_PREFILL_WIDTH));
        while self.active_prefills.len() < cap {
            // VRAM-pressure backpressure (wave budgeting). Each admitted prefill
            // pins its conversation's KV in VRAM, so under pressure we shed hot KV
            // to the substrate rather than piling on more concurrent prefills; if
            // that doesn't clear it, narrow the window and leave the rest queued
            // this pass. We always keep ≥1 prefill in flight so the engine makes
            // progress — a single oversized turn is then bounded by the per-arena
            // VRAM budget gate (which compacts/fails fast rather than spilling).
            if !self.active_prefills.is_empty() && self.vram_under_pressure() {
                if self.relieve_vram_pressure("promote", VramPhase::Load) {
                    // Pressure survived eviction — stop piling on this pass
                    // (the `!is_empty` guard keeps ≥1 in flight). The window
                    // halves only on a genuine THROUGHPUT STALL, never on the
                    // mere presence of nominal pressure: multiplicative
                    // decrease is failure evidence, and a card whose steady
                    // state sits just under the pressure band would otherwise
                    // pin every bulk-prefill phase at the floor width.
                    //
                    // Stall detection is time-aware because this branch runs
                    // many times a second while `PREFILL_OK_TOKENS` advances
                    // only when a forward completes (seconds apart for wide
                    // forwards): a stall is real only when NO forward has
                    // completed for a full [`PROMOTE_STALL_GRACE`]. Each
                    // elapsed grace period backs off one halving and re-arms;
                    // a device-OOM still shrinks instantly at its own site.
                    let ok = super::PREFILL_OK_TOKENS.load(std::sync::atomic::Ordering::Relaxed);
                    if ok > self.promote_ok_tokens_seen {
                        self.promote_ok_tokens_seen = ok;
                        self.promote_last_progress = Some(std::time::Instant::now());
                    }
                    let stalled = self
                        .promote_last_progress
                        .is_some_and(|t| t.elapsed() >= PROMOTE_STALL_GRACE);
                    if stalled {
                        self.shrink_admit_window();
                        self.promote_last_progress = Some(std::time::Instant::now());
                    }
                    break;
                }
            }
            let work = match self.prefill_queue.pop_front() {
                Some(w) => w,
                None => break,
            };
            let total = work.tokens.len();
            let _ = work
                .event_tx
                .send(TurnEvent::Prefill(work.prefill_text.clone()));
            let _ = work.event_tx.send(TurnEvent::PrefillProgress {
                tokens_done: 0,
                tokens_total: total,
            });
            let error = if total == 0 {
                Some(ConversationError::Channel(
                    "prefill received zero tokens".into(),
                ))
            } else {
                None
            };
            self.active_prefills.push(ActivePrefill {
                work,
                offset: 0,
                next_projection: 0,
                final_logits: None,
                error,
                prefill_start: None,
            });
        }
    }

    /// True when VRAM is under pressure — the signal to offload hot KV to the
    /// substrate (and, failing that, to stop admitting more concurrent
    /// prefills).
    ///
    /// Two complementary gates, pressure if **either** trips:
    /// - **Pool budget low** — [`vram_budget_available`] (`init_free -
    ///   pool_used - reserve`) drops below [`vram_budget_band`]. Pool-aware, so
    ///   it doesn't false-fire when KV is freed back into our stream-ordered
    ///   pool (which the driver `free` can't see), robust to WDDM's polluted
    ///   driver free, and the gate hot-tier eviction can actually relieve
    ///   (dropping a hot copy lowers `pool_used`). The band sits above the
    ///   per-forward transient peak so we relieve *before* forwards stall.
    /// - **Driver free below the reserve floor *and* the pool can't absorb the
    ///   next allocation by reuse** — `free < max(10% total, 1 GiB)` while the
    ///   pool's reserved-but-free headroom (`pool_reserved - pool_used`) is
    ///   under [`VRAM_REUSE_BAND`]. The reuse-headroom qualifier mirrors
    ///   [`vram_has_room`]'s `os_needed` gate: a low driver free while the pool
    ///   still holds freed blocks to reuse is *not* pressure (reusing them
    ///   costs zero new OS memory, so `free` never moves) — without this
    ///   qualifier the floor false-fires on WDDM, where the pool's own
    ///   reservation pins driver free low, and needlessly throttles admission
    ///   while gigabytes of pool budget remain.
    ///
    /// `false` on non-CUDA / when the queries are unavailable.
    ///
    /// [`vram_budget_available`]: super::super::BatchedInferenceSession::vram_budget_available
    /// [`vram_has_room`]: candle_nn::kv_cache
    /// Phase-independent default (`Load`, the conservative wide band). Prefer
    /// [`vram_under_pressure_for`](Self::vram_under_pressure_for) at call sites
    /// that know their phase (the decode loop passes `Decode` for a thinner band).
    pub(super) fn vram_under_pressure(&self) -> bool {
        self.vram_under_pressure_for(VramPhase::Load)
    }

    /// Whether the futile-defrag latch holds for the CURRENT pool stats: a
    /// prior defrag pass moved chunks but shed ~nothing, and neither `used`
    /// nor `reserved` has moved by more than the hysteresis band since — the
    /// allocation landscape compaction already failed against. See
    /// [`Scheduler::defrag_futile_at`].
    fn defrag_futile(&self, used: usize, reserved: usize) -> bool {
        self.defrag_futile_at.is_some_and(|(r0, u0, streak)| {
            // The re-probe bar doubles with each consecutive futile pass (one
            // hysteresis band up to 8x), so ordinary KV drift does not re-run
            // a ~100 ms proven-futile compaction every 256 MiB forever; a
            // successful shed resets the latch entirely.
            let bar = FOOTPRINT_HYSTERESIS << streak.min(3);
            reserved.abs_diff(r0) < bar && used.abs_diff(u0) < bar
        })
    }

    /// The pressure/relief reserve band for `phase`. `Load` keeps the wide
    /// transient-peak reserve; `Decode` keeps only a thin safety margin so the
    /// freed-float free-list counts as available and KV stays maximally resident.
    /// See [`VramPhase`]. Without a governor, both fall back to the fixed load
    /// band (the capacity term needs the governor).
    fn vram_band_for(&self, phase: VramPhase) -> usize {
        let cap = self.session.vram_governor().map(|g| g.capacity() as usize);
        match (phase, cap) {
            (VramPhase::Load, Some(c)) => (c / 10).max(vram_budget_band()),
            (VramPhase::Decode, Some(c)) => (c / 20).max(vram_decode_band()),
            (_, None) => vram_budget_band(),
        }
    }

    /// Phase-aware VRAM pressure signal — see [`VramPhase`] for why the band
    /// differs by phase. The availability number is identical across phases
    /// (`headroom + reuse`); only the reserve band it is compared against changes.
    pub(super) fn vram_under_pressure_for(&self, phase: VramPhase) -> bool {
        // With a VRAM governor installed, `vram_budget_available` is the honest
        // real free VRAM (headroom + reusable pool free-list). Trip eviction while
        // there is still a phase-appropriate safety margin so a forward's
        // transient activations can't oversubscribe the card before relief fires.
        let band = self.vram_band_for(phase);
        let pool_low = self
            .session
            .vram_budget_available()
            .is_some_and(|avail| avail < band);
        let driver_below_floor = match (
            self.session.vram_free_total(),
            self.session.vram_pool_stats(),
        ) {
            (Some((free, total)), Some((used, reserved))) => {
                let reuse_headroom = reserved.saturating_sub(used);
                // Trip on low physical free when EITHER the reusable pool is small
                // (the original case — genuine pressure, nothing to reuse) OR the
                // pool has OVER-SUBSCRIBED the card (`reserved > total`). In the
                // over-subscription case the "reuse" gap lives in WDDM host memory,
                // so it's un-usable for a contiguous activation allocation — which
                // spills and degrades throughput (the 62→65 GiB grind) while the
                // budget/reuse accounting still reads GiBs "free". Eviction must fire
                // regardless of the gap's size, shedding whole arenas until reserved
                // drops back under the card. Without the `reserved > total` arm, a
                // large un-returnable gap silently suppresses all relief here.
                free < (total / 10).max(1usize << 30)
                    && (reuse_headroom < VRAM_REUSE_BAND || reserved > total)
            }
            _ => false,
        };
        // Footprint pressure, but only when it is *actionable*:
        // - **Defrag** — reserved over the ceiling by more than the hysteresis
        //   band AND a reclaimable arena exists. The reserved-but-free gap has two
        //   flavours: fragmented free *inside live arenas* (compaction consolidates
        //   it, `can_reclaim_arena` is true) vs. free that has already fallen to the
        //   CUDA pool's own free-list with no whole arena recoverable
        //   (`can_reclaim_arena` false; `trim` frees nothing, compaction has nothing
        //   to move). Only the first is actionable — firing on the second churns a
        //   no-op reclaim every wave (`shed=0`) while `reserved` sits high on
        //   un-returnable free memory. `used` staying far below C means that gap is
        //   free headroom, not a paging risk.
        // - **Evict** — resident `used` nears C. This is the real paging signal
        //   (the working set, not the gap-inflated footprint), independent of the
        //   un-returnable gap.
        //
        // [`reclaim_footprint`]: Self::reclaim_footprint
        let raw_footprint = match (self.session.vram_pool_stats(), self.resident_capacity()) {
            (Some((used, reserved)), Some(capacity)) => {
                let compact_ceiling = capacity
                    .saturating_sub(vram_budget_band())
                    .saturating_add(FOOTPRINT_HYSTERESIS);
                let evict_high = capacity.saturating_sub(cap_margin(
                    capacity,
                    vram_evict_high_pct(),
                    VRAM_EVICT_HIGH_FLOOR_MB,
                ));
                // The defrag arm is silenced while the futile latch holds: a
                // gap compaction has PROVEN it cannot lower must not read as
                // pressure every wave (that pinned the admission window at the
                // floor and throttled prefill to single-sequence forwards).
                (reserved > compact_ceiling
                    && self.session.can_reclaim_arena()
                    && !self.defrag_futile(used, reserved))
                    || used > evict_high
            }
            _ => false,
        };
        // Rate-limit: don't re-trip the footprint gate within the cooldown of the
        // last footprint reclaim. Reserved hugging the ceiling (a fragmented gap
        // the engine reuses, which compaction can't lower) would otherwise fire
        // relief every loop iteration; the budget gates above are unaffected.
        let footprint_pressure = raw_footprint
            && self
                .last_footprint_relief
                .map_or(true, |t| t.elapsed() >= FOOTPRINT_RELIEF_COOLDOWN);
        pool_low || driver_below_floor || footprint_pressure
    }

    /// The card's resident capacity C (bytes) — the balloon-measured limit below
    /// which the pool footprint stays resident (no WDDM paging). Falls back to the
    /// driver's physical total until the balloon has measured C (`capacity()` is 0
    /// then), so the reclaim's ceilings are never a spurious `0 − margin` at
    /// startup. `None` on non-CUDA / when unavailable.
    fn resident_capacity(&self) -> Option<usize> {
        self.session
            .vram_governor()
            .map(|g| g.capacity() as usize)
            .filter(|&c| c > 0)
            .or_else(|| self.session.vram_free_total().map(|(_, total)| total))
    }

    /// Reclaim VRAM footprint with two independent, hysteretic controllers so it
    /// neither oversubscribes the card nor thrashes. Runs on the scheduler thread
    /// between forwards; a cheap stats-read no-op when comfortably below both
    /// watermarks. Returns reserved-footprint bytes shed.
    ///
    /// - **Defrag (manages `reserved`):** when the reserved footprint nears
    ///   resident capacity C *and a whole arena is reclaimable*, the gap is
    ///   fragmented free *inside* live arenas that `trim` can't return — so
    ///   `reserved` sticks high even though `used` is far lower. Consolidate it with
    ///   a bounded compaction whose budget ramps with the overshoot (reclaiming that
    ///   space for KV, NOT evicting it), then release + trim. Fires only when the gap
    ///   is genuinely large (compaction is expensive) AND recoverable: a gap of free
    ///   that has already fallen to the CUDA pool's own free-list (no reclaimable
    ///   arena) is left alone — `trim` can't return it and compaction has nothing to
    ///   move, and with `used` far below C it is free headroom, not a paging risk.
    /// - **Evict (manages `used`):** only when the REAL resident data `used` —
    ///   not the gap-inflated `reserved` — climbs within [`vram_evict_high_pct`]
    ///   of C does it evict, and then in BULK down to [`vram_evict_low_pct`] below C
    ///   (hysteresis), a rare decisive exit that coasts rather than the per-wave
    ///   nibbling that caused reload-churn stalls. The working-set keep-list
    ///   protects what's being attended. At moderate `used` this evicts nothing.
    ///
    /// Emits `compact_ms`/`evict_ms` timing so a stall can be attributed to the
    /// reclaim's own GPU/lock work vs. downstream reprojection reloads.
    pub(super) fn reclaim_footprint(&mut self) -> u64 {
        let (Some(capacity), Some((used, reserved))) =
            (self.resident_capacity(), self.session.vram_pool_stats())
        else {
            return 0;
        };
        // Watermarks (see the constants for the % / floor scaling).
        let compact_ceiling = capacity.saturating_sub(vram_budget_band());
        let evict_high = capacity.saturating_sub(cap_margin(
            capacity,
            vram_evict_high_pct(),
            VRAM_EVICT_HIGH_FLOOR_MB,
        ));
        // Actionable pressure only. Defrag is actionable when `reserved` overshoots
        // the ceiling AND a whole arena is reclaimable (compaction has something to
        // consolidate). A reserved overshoot on un-returnable CUDA-pool free memory
        // (`can_reclaim_arena` false) is NOT: `trim` frees nothing and compaction
        // has nothing to move, so entering the path would stamp the cooldown and
        // log a `shed=0` no-op every wave while `reserved` sits high on free
        // headroom. Evict is actionable when resident `used` nears C. Neither → bail
        // without touching the cooldown so a genuine future trigger isn't
        // suppressed. (Empty-arena release still runs per-wave in the run loop.)
        let defrag_actionable = reserved > compact_ceiling.saturating_add(FOOTPRINT_HYSTERESIS)
            && self.session.can_reclaim_arena()
            && !self.defrag_futile(used, reserved);
        let evict_actionable = used > evict_high;
        if !defrag_actionable && !evict_actionable {
            return 0;
        }
        // Entering the pressure path: stamp the cooldown now (even if we can't
        // lower `reserved` this pass — e.g. the gap is fragmented) so the gate
        // doesn't re-fire this reclaim every loop iteration.
        self.last_footprint_relief = Some(std::time::Instant::now());
        let before = reserved;
        let mut compact_ms = 0u64;
        let mut compact_moves = 0usize;
        let mut evict_ms = 0u64;
        let mut evicted = crate::substrate::EvictionReport { count: 0, bytes: 0 };

        // ── Defrag controller ───────────────────────────────────────────────
        if reserved > compact_ceiling {
            let _ = self.session.release_empty_arenas();
            self.trim_kv_pool();
            let (u2, r2) = self.session.vram_pool_stats().unwrap_or((used, reserved));
            // Compact only when the release+trim above didn't clear it AND the gap
            // is genuinely large — the fragmented-free-inside-live-arenas state
            // that `trim` alone can't touch.
            // Gap threshold scales with the card (% of C) with a floor, never a
            // fixed GiB sized for one machine.
            let min_compact_gap = cap_margin(
                capacity,
                VRAM_MIN_COMPACT_GAP_PCT,
                VRAM_MIN_COMPACT_GAP_FLOOR_MB,
            ) as u64;
            if r2 > compact_ceiling
                && (r2.saturating_sub(u2) as u64) > min_compact_gap
                && self.session.can_reclaim_arena()
            {
                // Ramp the compaction budget with how far `reserved` overshoots the
                // ceiling: one base budget just past it, up to ~4× as `reserved`
                // nears C, so a fast-growing *compactable* gap is consolidated in
                // step instead of trailing a fixed per-wave base budget. Bounded at
                // the top so a pass never becomes the multi-second blocking
                // compaction the base budget replaced. Same ramp shape as the
                // eviction controller below.
                let over = r2.saturating_sub(compact_ceiling);
                let span = capacity.saturating_sub(compact_ceiling).max(1); // ceiling → C
                let mult10 = 10 + 30 * over.min(span) / span; // ×1 .. ×4
                let budget = compact_base_moves().saturating_mul(mult10) / 10;
                let t = std::time::Instant::now();
                compact_moves = self.session.defragment_bounded(budget).unwrap_or(0);
                let _ = self.session.release_empty_arenas();
                super::timed_synchronize(&self.device);
                self.trim_kv_pool();
                compact_ms = t.elapsed().as_millis() as u64;
            }
        }

        // ── Eviction controller (watermarks computed above) ─────────────────
        let used_now = self
            .session
            .vram_pool_stats()
            .map(|(u, _)| u)
            .unwrap_or(used);
        if used_now > evict_high {
            // Ramp the eviction depth with how far `used` is past the soft
            // threshold, so eviction is gradual and converges to steady state
            // rather than one bulk dump: shed ~1.5× the overage just past the
            // threshold, ramping to ~6× (capped at the deep low watermark) as
            // `used` nears capacity.
            let evict_low = capacity.saturating_sub(cap_margin(
                capacity,
                vram_evict_low_pct(),
                VRAM_EVICT_LOW_FLOOR_MB,
            ));
            let overage = used_now.saturating_sub(evict_high);
            let deep = used_now.saturating_sub(evict_low);
            let span = capacity.saturating_sub(evict_high).max(1);
            let mult10 = 15 + 45 * overage.min(span) / span; // ×1.5 .. ×6
            let target = overage.saturating_mul(mult10) / 10;
            let target = target.min(deep).max(1) as u64;
            let t = std::time::Instant::now();
            evicted = self.evict_cold_tail(target);
            if evicted.count > 0 {
                if self.session.can_reclaim_arena() {
                    let _ = self.session.defragment_bounded(compact_base_moves());
                }
                let _ = self.session.release_empty_arenas();
                super::timed_synchronize(&self.device);
                self.trim_kv_pool();
            }
            evict_ms = t.elapsed().as_millis() as u64;
        }

        let after = self
            .session
            .vram_pool_stats()
            .map(|(_, r)| r)
            .unwrap_or(before);
        let shed = before.saturating_sub(after) as u64;
        // Futility latch: a defrag that moved chunks yet shed ~nothing proves
        // the gap is unreclaimable in this allocation landscape (pinned chunks
        // hold every arena open) — stop re-running it, and stop reporting it as
        // pressure, until the landscape moves. Any real shed (or eviction)
        // re-arms the controller.
        if shed >= FOOTPRINT_HYSTERESIS as u64 || evicted.count > 0 {
            self.defrag_futile_at = None;
        } else if compact_moves > 0 {
            let used_after = self
                .session
                .vram_pool_stats()
                .map(|(u, _)| u)
                .unwrap_or(used);
            let streak = self
                .defrag_futile_at
                .map(|(_, _, n)| n.saturating_add(1))
                .unwrap_or(0u32);
            self.defrag_futile_at = Some((after, used_after, streak));
        }
        // Always log once we're in the pressure path (past the early-return),
        // including the case where we couldn't lower `reserved` (compact_ms=0,
        // shed=0) — that "attempted but stuck" state is exactly what the rapid-fire
        // symptom looked like, so the telemetry should show it.
        tracing::info!(
            target: "candle_conversation::scheduler::vram_relief",
            capacity_mib = capacity / (1 << 20),
            used_mib = used / (1 << 20),
            reserved_before_mib = before / (1 << 20),
            reserved_after_mib = after / (1 << 20),
            gap_mib = before.saturating_sub(used) / (1 << 20),
            turns_evicted = evicted.count,
            evicted_mib = evicted.bytes / (1 << 20),
            shed_mib = shed / (1 << 20),
            compact_ms,
            compact_moves,
            evict_ms,
            "footprint reclaim"
        );
        shed
    }

    /// Escalated recovery after a background compressor signalled VRAM starvation
    /// ([`VramGovernor::signal_starvation`]): it couldn't allocate its transient
    /// quant arena, so the compress-to-free that would relieve pressure is itself
    /// blocked by lack of VRAM. The starved turn is unharmed (still hot-float +
    /// consistent, retried next persistence pass); this frees the room that retry
    /// needs. First run the normal footprint reclaim (its defrag alone often frees
    /// the small arena the compressor needs); if `reserved` is still near capacity,
    /// force a bulk eviction **regardless of the `used` watermark** — starvation
    /// means we need room NOW, overriding the "only evict when used is high" gate.
    pub(super) fn relieve_compression_starvation(&mut self, count: u64) {
        let mut shed = self.reclaim_footprint();
        let still_tight = matches!(
            (self.resident_capacity(), self.session.vram_pool_stats()),
            (Some(cap), Some((_, reserved))) if reserved > cap.saturating_sub(vram_budget_band())
        );
        let mut evicted = crate::substrate::EvictionReport { count: 0, bytes: 0 };
        let mut evict_ms = 0u64;
        if still_tight {
            let before = self.session.vram_pool_stats().map(|(_, r)| r).unwrap_or(0);
            let t = std::time::Instant::now();
            evicted = self.evict_cold_tail(vram_budget_band().saturating_mul(2) as u64);
            if evicted.count > 0 {
                if self.session.can_reclaim_arena() {
                    // Starvation is acute — use the deeper (Costly-level) budget.
                    let _ = self
                        .session
                        .defragment_bounded(compact_base_moves().saturating_mul(3));
                }
                let _ = self.session.release_empty_arenas();
                super::timed_synchronize(&self.device);
                self.trim_kv_pool();
            }
            evict_ms = t.elapsed().as_millis() as u64;
            let after = self
                .session
                .vram_pool_stats()
                .map(|(_, r)| r)
                .unwrap_or(before);
            shed = shed.saturating_add(before.saturating_sub(after) as u64);
        }
        tracing::warn!(
            target: "candle_conversation::scheduler::vram_relief",
            starvation_events = count,
            turns_evicted = evicted.count,
            evict_ms,
            shed_mib = shed / (1 << 20),
            "escalated VRAM recovery after background-compression starvation"
        );
    }

    /// Shed least-recently-used hot turn KV to the warm (RAM) tier across the
    /// resident conversations, freeing up to `target_bytes` of pool VRAM.
    /// Oldest-first and reversible (a reselected turn reloads from RAM). Only
    /// turns that already hold a warm copy are evictable, so callers should
    /// first [`PersistenceTrigger::flush_blocking`] to make the just-sealed
    /// turns qualify. The `target_bytes` budget caps total bytes freed, so a
    /// conversation reached via several slots is naturally not over-evicted
    /// (and `evict_hot_to_free` is per-conversation scoped — it can never touch
    /// a parallel conversation's selected working set).
    fn evict_cold_tail(&mut self, target_bytes: u64) -> crate::substrate::EvictionReport {
        // Explicit protect-list: the union of every live slot's current
        // projection working set (the sealed turns/sections in-flight
        // prefills/decodes are attending over). Relief eviction must not drop the
        // hot copy of an in-scope turn — the block table still references its
        // chunks, so `hot = None` would free NO VRAM and only force a reload when
        // the turn is next reprojected. The reprojection path already protects its
        // incoming selection via the same keep-list; this extends that explicit
        // protection to the relief path. `evict_hot_to_free` resolves keys against
        // each conversation's own substrate, so passing the global union to every
        // conversation only ever protects that conversation's own attended turns
        // (a non-matching key is a no-op) — no per-conversation grouping needed.
        let mut keep_sections: Vec<SectionId> = Vec::new();
        let mut keep_turns: Vec<TurnKey> = Vec::new();
        for st in self.slot_projection_state.values() {
            keep_sections.extend(st.working_set.sections.iter().copied());
            keep_turns.extend(st.working_set.turns.iter().copied());
        }

        let t = std::time::Instant::now();
        let mut report = crate::substrate::EvictionReport { count: 0, bytes: 0 };
        let mut remaining = target_bytes;
        let convs: Vec<Conversation> = self.slot_conversations.values().cloned().collect();
        for conv in convs {
            if remaining == 0 {
                break;
            }
            let r = conv
                .write()
                .evict_hot_to_free(&keep_sections, &keep_turns, remaining);
            remaining = remaining.saturating_sub(r.bytes);
            report.count += r.count;
            report.bytes += r.bytes;
        }
        // Feed the GUI's phase timeline here — the single chokepoint every relief
        // path (governor driver, footprint reclaim, compression-starvation
        // recovery) funnels through, so each eviction is counted exactly once
        // regardless of caller.
        self.wave_stats.add_evict(
            report.bytes,
            report.count as u64,
            t.elapsed().as_millis() as u64,
        );
        report
    }

    /// Gentle-early ingest relief — the bottom rung of the pressure ladder.
    /// Once resident `used` crosses the ingest demote watermark
    /// ([`ingest_demote_pct`], ~50% of C — far below the near-cap footprint
    /// controllers), shed the sealed, warm-backed KV of append-only ingest
    /// timelines down to a small rolling hot window ([`ingest_hot_window`]).
    /// Zero reload cost: ingest KV is never re-attended until query time, when it
    /// re-elevates warm→hot on demand — so this is the cheapest relief and fires
    /// first, keeping a bulk repo ingest from pinning the whole corpus hot up to
    /// the aggressive eviction watermark. No-op when nothing is ingesting or
    /// `used` is below the watermark. Runs per-wave on the scheduler thread.
    pub(super) fn demote_cold_ingest_if_pressured(&mut self) {
        if self.ingest_timelines.is_empty() {
            return;
        }
        let (Some(capacity), Some((used, _))) =
            (self.resident_capacity(), self.session.vram_pool_stats())
        else {
            return;
        };
        let watermark = capacity / 100 * ingest_demote_pct();
        if used <= watermark {
            return;
        }
        let window = ingest_hot_window();
        // Relieve back to the watermark, no further: `target` bounds the LRU walk
        // so the demote sheds the least-recently-active ingest tail just enough to
        // clear the pressure, never the whole hot working set.
        let target_bytes = used.saturating_sub(watermark) as u64;
        // 1. Shed whatever is already warm-backed — free, no migration.
        let t_demote = std::time::Instant::now();
        let report = self.demote_ingest_once(window, target_bytes);
        // Feed the GUI's phase timeline: the gentle-rung ingest demotion.
        self.wave_stats.add_evict(
            report.bytes,
            report.count as u64,
            t_demote.elapsed().as_millis() as u64,
        );
        // 2. If `used` is still over the watermark, the demote is **warm-starved**:
        //    warm-copy production (the async persistence pass) lags the ingest seal
        //    rate, so the cold backlog is hot-without-warm and not yet demotable.
        //    NUDGE the persistence thread to run its hot→warm drain (non-blocking),
        //    and let the *next* wave's step 1 shed the freshly-warmed backlog. We
        //    deliberately do NOT `flush_blocking` here: this runs per-wave on the
        //    scheduler thread, and under sustained pressure the persist thread is
        //    already mid-pass — a blocking wait would stall the scheduler for the
        //    full timeout while draining nothing sooner. A `fire()` is a no-op when
        //    a pass is already queued, so it never adds latency.
        let nudged = if self
            .session
            .vram_pool_stats()
            .is_some_and(|(u, _)| u > watermark)
        {
            self.persist_trigger.fire();
            true
        } else {
            false
        };
        if report.count > 0 {
            // Freed hot arenas → release + trim so `reserved` can actually fall.
            let _ = self.session.release_empty_arenas();
            self.trim_kv_pool();
            tracing::debug!(
                target: "candle_conversation::scheduler::vram_relief",
                used_mib = used / (1 << 20),
                watermark_mib = watermark / (1 << 20),
                ingest_timelines = self.ingest_timelines.len(),
                turns = report.count,
                freed_mib = report.bytes / (1 << 20),
                window,
                nudged,
                "cold-ingest demote (gentle-early)"
            );
        }
    }

    /// Size the ingest admission window to the **hot→warm drain backlog** — the
    /// leading backpressure signal that keeps `used` off the warm-starved climb
    /// (see the pool-footprint dashboard). The persistence thread publishes its
    /// live backlog via [`PersistenceTrigger::pending_warm_bytes`]; when it
    /// exceeds the target the drain is behind the seal rate, so narrow the AIMD
    /// window (fewer concurrent scopes → lower seal rate → drain catches up);
    /// when it falls below half the target, reopen. `vram_under_pressure` stays
    /// the hard floor beneath this (its per-admission shrinks still fire on a
    /// true VRAM spike). No-op when nothing is ingesting — chat keeps the
    /// per-iteration AIMD recovery in the run loop. Runs at the ~2 s wave
    /// cadence, matching how often the backlog signal refreshes.
    /// Whether free host RAM has dropped below the ingest floor
    /// ([`host_ram_floor_bytes`]). Refreshes the cached `sysinfo` reading at most
    /// once per [`HOST_RAM_PROBE_INTERVAL`] — never a per-wave syscall. Returns
    /// `false` until the first probe, so a fresh scheduler never throttles blind.
    fn host_ram_pressured(&mut self) -> bool {
        let stale = self
            .host_ram_probe
            .map(|(t, _, _)| t.elapsed() >= HOST_RAM_PROBE_INTERVAL)
            .unwrap_or(true);
        if stale {
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            self.host_ram_probe = Some((
                std::time::Instant::now(),
                sys.available_memory(),
                sys.total_memory(),
            ));
        }
        match self.host_ram_probe {
            Some((_, available, total)) => available < host_ram_floor_bytes(total),
            None => false,
        }
    }

    pub(super) fn regulate_ingest_admission(&mut self) {
        if self.ingest_timelines.is_empty() {
            return;
        }
        // Host-tier backpressure: if the warm (RAM) tier has driven free host RAM
        // below the floor, throttle regardless of the VRAM backlog. The hot→warm
        // migration needs host memory for its staging buffer; starving it stalls
        // the whole drain. This is the leading signal the VRAM-only backlog check
        // below cannot see (VRAM can look fine while host RAM is nearly gone).
        if self.host_ram_pressured() {
            self.shrink_admit_window();
            return;
        }
        let Some(capacity) = self.resident_capacity() else {
            return;
        };
        let target = capacity / 100 * ingest_warm_backlog_pct();
        let backlog = self.persist_trigger.pending_warm_bytes() as usize;
        // Volume-floored progress: a tick certifies the current width, which a
        // trickle of tiny forwards cannot (see `EVIDENCE_MIN_PREFILL_TOKENS`).
        // Sub-floor volume accumulates — `admit_ok_tokens_seen` advances only
        // when the floor is cleared.
        let ok_tokens = super::PREFILL_OK_TOKENS.load(std::sync::atomic::Ordering::Relaxed);
        let progressed =
            ok_tokens >= self.admit_ok_tokens_seen + super::EVIDENCE_MIN_PREFILL_TOKENS;
        if progressed {
            self.admit_ok_tokens_seen = ok_tokens;
        }
        match super::backlog_admit_action(
            backlog,
            target,
            self.admit_window,
            Self::MAX_PREFILL_WIDTH,
            self.vram_under_pressure(),
        ) {
            // Drain falling behind the seal rate — throttle admission.
            super::BacklogAction::Shrink => self.shrink_admit_window(),
            // Drain caught up and VRAM is clear — reopen a notch.
            super::BacklogAction::Grow => {
                self.admit_grow_streak = 0;
                self.grow_admit_window();
            }
            // Deadband — or growth blocked only by the pressure bit. The
            // evidence path reopens a wedged window on proven OOM-free
            // throughput (see `evidence_admit_grow`); a real spike still
            // shrinks instantly and resets the streak.
            super::BacklogAction::Hold => {
                let (grow, streak) = super::evidence_admit_grow(
                    backlog,
                    target,
                    self.admit_window,
                    Self::MAX_PREFILL_WIDTH,
                    progressed,
                    self.admit_grow_streak,
                    super::INGEST_EVIDENCE_GROW_TICKS,
                );
                self.admit_grow_streak = streak;
                if grow {
                    let before = self.admit_window;
                    self.grow_admit_window();
                    tracing::info!(
                        target: "candle_conversation::scheduler::timing",
                        admit_window = self.admit_window,
                        was = before,
                        "admission window reopened on throughput evidence under nominal pressure"
                    );
                }
            }
        }
    }

    /// Under **heavy** hot→warm backlog, block the wave loop on a device sync so
    /// ingest stops racing ahead of the drain. This runs *after* the per-wave
    /// eviction callbacks, so it also drains the primary stream: the persist
    /// pass — now a handful of cross-layer-batched kernel launches — runs
    /// uncontended by ingest forwards instead of interleaving with them on the
    /// shared stream (the contention that inflates each pass on WDDM). A sync
    /// only *adds* ordering, so there is no KV-before-copy hazard. No-op unless
    /// ingesting and the backlog is over [`ingest_sync_ceiling_pct`].
    pub(super) fn sync_if_backlog_critical(&mut self) {
        if self.ingest_timelines.is_empty() {
            return;
        }
        let Some(capacity) = self.resident_capacity() else {
            return;
        };
        let ceiling = capacity / 100 * ingest_sync_ceiling_pct();
        let backlog = self.persist_trigger.pending_warm_bytes() as usize;
        if backlog <= ceiling {
            return;
        }
        let t = std::time::Instant::now();
        super::timed_synchronize(&self.device);
        tracing::debug!(
            target: "candle_conversation::scheduler::vram_relief",
            backlog_mib = backlog / (1 << 20),
            ceiling_mib = ceiling / (1 << 20),
            stall_ms = t.elapsed().as_millis() as u64,
            "heavy-backlog device sync (de-contend drain)"
        );
    }

    /// One pass of LRU-smart cold-ingest demotion across every live conversation,
    /// freeing at most `target_bytes` total (the `remaining` budget threads across
    /// conversations, so the walk stops the moment the watermark is cleared).
    /// `demote_cold_ingest` self-filters to the timelines each conversation owns (a
    /// non-matching id is a no-op), walks that conversation's `hot_lru` oldest-first
    /// so the least-recently-active tail sheds before an active window, and is
    /// idempotent (already-demoted turns have `hot = None` and are skipped). The
    /// global working-set protect-list is passed to every conversation but only
    /// ever matches that conversation's own attended turns — mirrors
    /// [`Self::evict_cold_tail`].
    fn demote_ingest_once(
        &mut self,
        window: usize,
        target_bytes: u64,
    ) -> crate::substrate::EvictionReport {
        // Protect the active working set of every live slot (what in-flight
        // prefills/decodes are attending) — the same union `evict_cold_tail`
        // builds, so an actively-ingesting conversation's gathered turns are never
        // demoted out from under the next projection.
        let mut keep_sections: Vec<SectionId> = Vec::new();
        let mut keep_turns: Vec<TurnKey> = Vec::new();
        for st in self.slot_projection_state.values() {
            keep_sections.extend(st.working_set.sections.iter().copied());
            keep_turns.extend(st.working_set.turns.iter().copied());
        }
        let mut report = crate::substrate::EvictionReport { count: 0, bytes: 0 };
        let mut remaining = target_bytes;
        let convs: Vec<Conversation> = self.slot_conversations.values().cloned().collect();
        for conv in convs {
            if remaining == 0 {
                break;
            }
            let r = conv.write().demote_cold_ingest(
                &self.ingest_timelines,
                &keep_turns,
                &keep_sections,
                window,
                remaining,
            );
            remaining = remaining.saturating_sub(r.bytes);
            report.count += r.count;
            report.bytes += r.bytes;
        }
        report
    }

    /// Compress-to-free: bring forward the quantization of completed, still-
    /// float turns under VRAM pressure. Mirrors the persistence thread's
    /// hot→warm quantize (same [`quantize_sealed_in_place`], same per-
    /// [`ConvCompression`] policy grouping) but installs **only** the quantized
    /// hot — it does not write the warm (RAM) copy.
    ///
    /// This is a deliberate division of labor: the pass runs on the scheduler
    /// thread to reclaim float VRAM *now* — for a turn NOT currently attended,
    /// the source float arenas free the instant the old hot `Arc`s drop under
    /// the write lock (the substrate held the only reference). For a turn the
    /// active decode IS attending over, the block-table GID clones keep the
    /// float chunks alive until the next reprojection rebuilds the table from the
    /// new quant `hot` — so its float reclaim lands one reproject later, still
    /// safe (a live forward never reads freed memory). Meanwhile the persistence
    /// thread still owns the warm/cold DtoH writes on its own tick (the
    /// compressed turns remain in `snapshot_pending_warm`, warm-absent, so it
    /// still picks them up and lands their bytes).
    ///
    /// A net shrink, not a move: the turn stays resident and attended-over, so
    /// there is no reload or hit-rate cost, and no *extra* quality loss — these
    /// turns get quantized on seal regardless; pressure only pulls it earlier.
    /// Turns whose hot is already quant (a prior pass, or persistence, beat us
    /// to them) are skipped via [`sealed_has_compressible_chunk`] so an undrained
    /// warm backlog doesn't re-walk finished turns. Returns the turns compressed.
    ///
    /// [`sealed_has_compressible_chunk`]: candle_nn::kv_cache::ChunkedKvBacking::sealed_has_compressible_chunk
    /// Bring forward the quantization of up to `budget_bytes` of completed float
    /// turns (estimated by their float footprint), oldest-conversation-first.
    /// **Bounded** so a large accumulated backlog is drained over several relief
    /// episodes — a few seconds each — rather than one multi-second blocking
    /// compression of *everything* pending (a 697-turn / 23 GiB / 66 s stall was
    /// the symptom). The background persistence thread drains the rest.
    fn compress_pending_turns(&mut self, budget_bytes: u64) -> usize {
        // Need an engine-wide turn policy to compress against; without one turns
        // stay native float (lossless capture) and there is nothing to bring
        // forward.
        let base = match self.session.compression_policy() {
            Some(p) => p,
            None => return 0,
        };
        let n_layers = self.session.num_layers();
        let device = self.session.device().clone();
        let copy_stream = match &device {
            Device::Cuda(d) => d.cuda_stream(),
            _ => return 0,
        };
        // Bound `backings`' immutable borrow of `self.session` to a disjoint
        // field from `self.elevate_pinned_scratch` (the `&mut` below), exactly
        // like `quantize_section_batch`.
        let backings = self.session.backings();

        let convs: Vec<Conversation> = self.slot_conversations.values().cloned().collect();
        let mut compressed = 0usize;
        // Estimated float bytes queued for compression so far — the bound.
        let mut collected: u64 = 0;
        for conv in convs {
            if collected >= budget_bytes {
                break; // Budget met — the rest drains next episode / in the background.
            }
            // Snapshot still-float turns (hot present, warm absent) grouped by
            // their per-conversation compression override — as the persistence
            // thread does — under a brief read lock, filtered to those whose hot
            // is still GPU-float so an undrained warm backlog can't make us
            // re-walk already-quant turns. Stop collecting once the byte budget is
            // reached so a big backlog doesn't compress all at once.
            let groups: HashMap<
                Option<ConvCompression>,
                Vec<(ResidenceIndex, Vec<SealedSequence>)>,
            > = {
                let view = conv.read();
                let mut g: HashMap<_, Vec<_>> = HashMap::new();
                for (idx, hot, cc) in view.snapshot_pending_warm() {
                    if hot.len() != n_layers {
                        continue;
                    }
                    // Layer 0 is representative: a turn's layers seal and
                    // compress together, so if layer 0 is still float, all are.
                    if !backings[0].sealed_has_compressible_chunk(&hot[0]) {
                        continue;
                    }
                    collected += sealed_total_bytes(&hot);
                    g.entry(cc).or_default().push((idx, hot));
                    if collected >= budget_bytes {
                        break;
                    }
                }
                g
            };

            for (cc, group) in groups {
                let policy = match effective_turn_policy(Some(&base), cc) {
                    Some(p) => p,
                    None => continue, // lossless capture: nothing to bring forward
                };
                // Per-residence quantized hot accumulator, one SealedSequence per
                // layer, filled positionally across the per-layer batched launches
                // (`quantize_sealed_in_place` returns one output per input in order).
                let mut q_per: Vec<Vec<SealedSequence>> = (0..group.len())
                    .map(|_| Vec::with_capacity(n_layers))
                    .collect();
                let mut ok = vec![true; group.len()];
                for layer in 0..n_layers {
                    let inputs: Vec<&SealedSequence> =
                        group.iter().map(|(_, hot)| &hot[layer]).collect();
                    match quantize_sealed_in_place(
                        &backings[layer],
                        &inputs,
                        &policy,
                        &device,
                        &copy_stream,
                        &mut self.elevate_pinned_scratch,
                    ) {
                        Ok(out) => {
                            for (slot, qi) in out.into_iter().enumerate() {
                                q_per[slot].push(qi);
                            }
                        }
                        Err(e) => {
                            tracing::warn!(
                                "compress_pending_turns: layer {layer} quantize failed: {e} (last CUDA kernel: {})",
                                candle::last_cuda_kernel_launch()
                            );
                            ok.fill(false);
                            break;
                        }
                    }
                }
                // Device-wide sync before the swap: the quantize kernels leave the
                // new Q-arenas' K/V writes in flight (including V work that can
                // retire on a stream a primary-stream-only sync misses — the
                // multi-turn V-duplication window), and the very next reproject on
                // THIS thread reads them. Mirrors the persistence thread's
                // post-batch `device.synchronize()`.
                if let Err(e) = device.synchronize() {
                    tracing::warn!(
                        "compress_pending_turns: device sync failed: {e:?} — skipping this group's installs"
                    );
                    continue;
                }
                // Atomic swap under one write lock: replace each residence's hot
                // with its quantized form. Dropping the old (float) hot `Vec`s
                // after the lock releases returns the source float chunks' arena
                // slots to the pool — the VRAM this rung exists to reclaim. Warm
                // stays untouched: the persistence thread still owes the DtoH.
                {
                    let mut view = conv.write();
                    for (i, (residence, _float)) in group.into_iter().enumerate() {
                        if !ok[i] || q_per[i].len() != n_layers {
                            continue;
                        }
                        view.replace_section_hot(residence, std::mem::take(&mut q_per[i]));
                        compressed += 1;
                    }
                }
            }
        }
        if compressed > 0 {
            // Wake the persistence thread so it lands the warm/cold copies of the
            // turns we just compressed without waiting for its 5 s tick.
            self.persist_trigger.fire();
        }
        compressed
    }

    /// Continuous-fair-wave prefill throttle: how many transformer layers a
    /// background prefill/glue cohort advances **per wave**
    /// (`docs/continuous_fair_waves.md`).
    ///
    /// `budget = ceil(N / R)`, where `R` is the decode-to-prefill airtime ratio
    /// of the interactive work to protect:
    /// - **No foreground decode active** → `R = 1` → `budget = N`: the prefill
    ///   clears every layer in one wave (nothing to shield → full speed).
    /// - **Decode active** → `R` = the max `decode_priority` ratio over the active
    ///   foreground decodes (default `High` when a layer can't be resolved) → the
    ///   prefill creeps `~N/R` layers per wave while decode keeps its experts hot.
    pub(super) fn wave_prefill_layer_budget(&self) -> usize {
        let n = self.model.num_layers().max(1);
        if self.foreground_decode_width() == 0 {
            return n;
        }
        let ratio = self
            .active_decodes
            .keys()
            .filter_map(|sid| self.decode_layer_priority(*sid))
            .map(|p| p.ratio())
            .max()
            .unwrap_or_else(|| crate::projection::DecodePriority::High.ratio());
        n.div_ceil(ratio.max(1) as usize).max(1)
    }

    /// Resolve the `decode_priority` of a decode slot's target layer, or `None`
    /// when the slot's target/timeline isn't resolvable (the caller then defaults
    /// to the protective `High`).
    pub(super) fn decode_layer_priority(
        &self,
        sid: SequenceId,
    ) -> Option<crate::projection::DecodePriority> {
        // A decode runs on a VIEW sequence, but the projection target (which
        // carries the layer's decode_priority) is pinned on the view's PARENT
        // slot. Resolve view → parent first, falling back to the sid itself for a
        // slot that decodes directly (no view).
        let slot = self
            .turn_views
            .get(&sid)
            .map(|v| v.parent_id)
            .unwrap_or(sid);
        let target = self.slot_targets.get(&slot)?;
        let builder = self.timeline_projections.get(&target.timeline)?;
        builder
            .schema()
            .layers
            .iter()
            .find(|l| l.id == target.layer)
            .map(|l| l.decode_priority)
    }

    /// Number of in-flight prefills that still have tokens left to process
    /// and have not errored.
    pub(super) fn prefill_width(&self) -> usize {
        self.active_prefills
            .iter()
            .filter(|p| p.error.is_none() && p.offset < p.work.tokens.len())
            .count()
    }

    /// Number of in-flight section ingests with tokens remaining (not errored).
    pub(super) fn section_ingest_width(&self) -> usize {
        self.active_section_ingests
            .iter()
            .filter(|s| s.error.is_none() && s.offset < s.tokens.len())
            .count()
    }

    /// Build one ragged section-ingest chunk: for each active section, its next
    /// `min(remaining, cap)` tokens, packed until the per-forward token budget.
    /// Returns `(seq_ids, inputs, group_idxs, advances)`, or `None` when nothing
    /// is pending. Shared by the standalone pass and the co-batched decode wave.
    ///
    /// Ragged batch: each section advances by its OWN min(remaining, cap). The
    /// varlen forward packs the heterogeneous lengths flat, so one near-finished
    /// section no longer collapses the whole wave to the batch minimum — the bug
    /// that dragged a 93-wide tool-catalog ingest down to ~1 token/seq/forward.
    ///
    /// Bound the TOTAL tokens to the same per-forward budget a normal prefill
    /// targets (`max_prefill_pass_tokens`). Without this the whole active set
    /// coalesces into one forward: the 93-section tool catalog (~21k tokens)
    /// packed into a single pass whose transient activation spiked VRAM to the
    /// card ceiling and paged. Sections beyond the budget ride the next chunk —
    /// the wave loop pumps until every section seals — so throughput is unchanged
    /// (each forward still fills to the expert-amortization target) while the peak
    /// stays bounded. At least one section is always admitted so the wave makes
    /// progress.
    #[allow(clippy::type_complexity)]
    pub(super) fn build_section_batch(
        &mut self,
    ) -> Option<(Vec<usize>, Vec<Tensor>, Vec<usize>, Vec<usize>)> {
        // Sections already creeping inside the wave group are excluded — their
        // offset isn't advanced until that group's head, so picking them here would
        // ingest the same chunk twice.
        let in_flight = self.wave_group_section_seqs();
        let active: Vec<usize> = (0..self.active_section_ingests.len())
            .filter(|&i| {
                let s = &self.active_section_ingests[i];
                s.error.is_none()
                    && s.offset < s.tokens.len()
                    && !in_flight.contains(&s.sequence_id.0)
            })
            .collect();
        if active.is_empty() {
            return None;
        }
        let cap = self.max_prefill_pass_tokens;
        let mut seq_ids: Vec<usize> = Vec::with_capacity(active.len());
        let mut inputs: Vec<Tensor> = Vec::with_capacity(active.len());
        let mut group_idxs: Vec<usize> = Vec::with_capacity(active.len());
        let mut advances: Vec<usize> = Vec::with_capacity(active.len());
        let mut batch_tokens = 0usize;
        for &i in &active {
            let s = &mut self.active_section_ingests[i];
            let off = s.offset;
            let advance = (s.tokens.len() - off).min(cap);
            // Stop packing once this forward has reached the per-forward budget
            // (but never emit an empty forward).
            if !seq_ids.is_empty() && batch_tokens + advance > cap {
                break;
            }
            let tokens = &s.tokens[off..off + advance];
            match Tensor::new(tokens, &self.device).and_then(|t| t.unsqueeze(0)) {
                Ok(t) => {
                    seq_ids.push(s.sequence_id.0);
                    inputs.push(t);
                    group_idxs.push(i);
                    advances.push(advance);
                    batch_tokens += advance;
                }
                Err(e) => {
                    s.error = Some(ConversationError::Model(e));
                }
            }
        }
        if seq_ids.is_empty() {
            return None;
        }
        Some((seq_ids, inputs, group_idxs, advances))
    }

    /// Commit one section-ingest chunk after its forward (standalone or
    /// co-batched): advance each section by its own `advance`, record its slot
    /// tokens, and bump its offset. Section logits are never used (no decode).
    pub(super) fn complete_section_chunk(&mut self, group_idxs: &[usize], advances: &[usize]) {
        for (&i, &advance) in group_idxs.iter().zip(advances.iter()) {
            let s = &mut self.active_section_ingests[i];
            if let Err(e) = self.session.advance_sequence(s.sequence_id.0, advance) {
                s.error = Some(ConversationError::Model(e));
                continue;
            }
            let seq_id = s.sequence_id;
            let off = s.offset;
            let chunk_tokens = s.tokens[off..off + advance].to_vec();
            super::Scheduler::record_slot_tokens(&mut self.slot_tokens, seq_id, &chunk_tokens);
            s.offset += advance;
        }
    }

    /// Drain completed or errored section ingest entries. Errored entries send
    /// `Err`; finished entries call `finalize_section_ingest` (seal + write)
    /// and send the `SealResult`.
    pub(super) fn finalize_done_section_ingests(&mut self) {
        let mut i = 0;
        while i < self.active_section_ingests.len() {
            let done = {
                let s = &self.active_section_ingests[i];
                s.error.is_some() || s.offset >= s.tokens.len()
            };
            if !done {
                i += 1;
                continue;
            }
            let s = self.active_section_ingests.swap_remove(i);
            if let Some(e) = s.error {
                let _ = s.response_tx.send(Err(e));
                continue;
            }
            let result = self.finalize_section_ingest(
                s.sequence_id,
                s.section_id,
                s.seal_block_from,
                std::sync::Arc::new(s.tokens.to_vec()),
                s.address,
                s.debug_name,
                s.in_collection,
            );
            let _ = s.response_tx.send(result);
            // swap_remove pulled the last element into i; don't increment.
        }
    }

    /// Clear the in-flight continuous-fair-wave prefill group (residual, cursor,
    /// members) so the next wave forms a fresh one.
    fn reset_wave_prefill(&mut self) {
        self.wave_prefill_residual = None;
        self.wave_prefill_cursor = 0;
        self.wave_prefill_members.clear();
    }

    /// Set of section-ingest `seq_id`s currently in flight in the wave group, so
    /// the standalone section pass and a fresh group formation don't double-admit
    /// a chunk that is already creeping (its offset isn't advanced until the head).
    pub(super) fn wave_group_section_seqs(&self) -> std::collections::HashSet<usize> {
        self.wave_prefill_members
            .iter()
            .filter_map(|m| match m {
                WaveMember::Section { seq_id, .. } => Some(*seq_id),
                WaveMember::Prefill { .. } => None,
            })
            .collect()
    }

    /// Form a FRESH wave group into `wave_prefill_members`: every ready dialogue
    /// prefill, plus — when `include_sections` and at least one prefill is present
    /// — section chunks bounded by the per-forward token cap. Section chunks join
    /// only alongside a cohort (so they co-batch a creep that is happening anyway);
    /// with no cohort the caller uses the faster full-sweep section path instead.
    /// Members are ordered prefills-then-sections and this order is then fixed for
    /// the group's life (the held residual depends on a stable input order).
    fn form_wave_group(&mut self, include_sections: bool) {
        let mut members: Vec<WaveMember> = (0..self.active_prefills.len())
            .filter(|&i| {
                let p = &self.active_prefills[i];
                p.error.is_none() && p.final_logits.is_none() && p.offset < p.work.tokens.len()
            })
            .map(|i| WaveMember::Prefill {
                seq_id: self.active_prefills[i].work.sequence_id.0,
            })
            .collect();
        if include_sections && !members.is_empty() {
            let cap = self.max_prefill_pass_tokens;
            let mut sec_tokens = 0usize;
            for i in 0..self.active_section_ingests.len() {
                let s = &self.active_section_ingests[i];
                if s.error.is_some() || s.offset >= s.tokens.len() {
                    continue;
                }
                let advance = (s.tokens.len() - s.offset).min(cap);
                // Bound the section contribution to the per-forward token budget
                // (at least one always admitted); the rest ride the next group.
                if sec_tokens > 0 && sec_tokens + advance > cap {
                    break;
                }
                members.push(WaveMember::Section {
                    seq_id: s.sequence_id.0,
                    advance,
                });
                sec_tokens += advance;
            }
        }
        self.wave_prefill_members = members;
    }

    /// Resume the held wave group: rebuild each member's `(seq_id, input tensor)`
    /// from its live backing (prefill = full token set; section = its stable
    /// `[offset, offset+advance)` chunk), dropping members that errored/completed.
    /// Returns the kept members (aligned with `seq_ids`/`inputs`) plus the
    /// `active_prefills` positions of the prefill members (for OOM/error routing).
    #[allow(clippy::type_complexity)]
    fn build_wave_group_inputs(
        &mut self,
    ) -> (Vec<WaveMember>, Vec<usize>, Vec<Tensor>, Vec<usize>) {
        let members = self.wave_prefill_members.clone();
        let mut kept: Vec<WaveMember> = Vec::with_capacity(members.len());
        let mut seq_ids: Vec<usize> = Vec::with_capacity(members.len());
        let mut inputs: Vec<Tensor> = Vec::with_capacity(members.len());
        let mut prefill_gidxs: Vec<usize> = Vec::new();
        for m in members {
            match m {
                WaveMember::Prefill { seq_id } => {
                    let Some(i) = self
                        .active_prefills
                        .iter()
                        .position(|p| p.work.sequence_id.0 == seq_id)
                    else {
                        continue;
                    };
                    if self.active_prefills[i].error.is_some()
                        || self.active_prefills[i].final_logits.is_some()
                    {
                        continue;
                    }
                    if self.active_prefills[i].prefill_start.is_none() {
                        self.active_prefills[i].prefill_start = Some(Instant::now());
                    }
                    let toks: Vec<u32> = self.active_prefills[i].work.tokens[..].to_vec();
                    match Tensor::new(toks.as_slice(), &self.device).and_then(|t| t.unsqueeze(0)) {
                        Ok(t) => {
                            kept.push(m);
                            seq_ids.push(seq_id);
                            inputs.push(t);
                            prefill_gidxs.push(i);
                        }
                        Err(e) => self.active_prefills[i].error = Some(ConversationError::Model(e)),
                    }
                }
                WaveMember::Section { seq_id, advance } => {
                    let Some(i) = self
                        .active_section_ingests
                        .iter()
                        .position(|s| s.sequence_id.0 == seq_id)
                    else {
                        continue;
                    };
                    if self.active_section_ingests[i].error.is_some() {
                        continue;
                    }
                    let off = self.active_section_ingests[i].offset;
                    let end = (off + advance).min(self.active_section_ingests[i].tokens.len());
                    let toks: Vec<u32> = self.active_section_ingests[i].tokens[off..end].to_vec();
                    match Tensor::new(toks.as_slice(), &self.device).and_then(|t| t.unsqueeze(0)) {
                        Ok(t) => {
                            kept.push(m);
                            seq_ids.push(seq_id);
                            inputs.push(t);
                        }
                        Err(e) => {
                            self.active_section_ingests[i].error = Some(ConversationError::Model(e))
                        }
                    }
                }
            }
        }
        (kept, seq_ids, inputs, prefill_gidxs)
    }

    /// Finish a wave group that reached the final layer: `members`/`member_logits`
    /// are aligned in caller order. Prefill members commit their offset, emit
    /// staged/progress events and record `final_logits` for promotion to decode;
    /// section members advance their chunk + record slot tokens (sealed later by
    /// `finalize_done_section_ingests`). Clears the group.
    fn complete_wave_group(&mut self, members: &[WaveMember], member_logits: &[Tensor]) {
        for (k, m) in members.iter().enumerate() {
            match *m {
                WaveMember::Prefill { seq_id: sid } => {
                    let Some(i) = self
                        .active_prefills
                        .iter()
                        .position(|p| p.work.sequence_id.0 == sid)
                    else {
                        continue;
                    };
                    let total = self.active_prefills[i].work.tokens.len();
                    let seq_id = self.active_prefills[i].work.sequence_id;
                    if let Err(e) = self.session.advance_sequence(seq_id.0, total) {
                        self.active_prefills[i].error = Some(ConversationError::Model(e));
                        continue;
                    }
                    let all_tokens: Vec<u32> = self.active_prefills[i].work.tokens[..].to_vec();
                    super::Scheduler::record_slot_tokens(
                        &mut self.slot_tokens,
                        seq_id,
                        &all_tokens,
                    );
                    self.active_prefills[i].offset = total;
                    // Staged calibration prefill: emit every segment's pinned
                    // projection here at completion (a wave processes the whole
                    // token set at once).
                    if let Some(comp) = self.active_prefills[i].work.staged_composition.clone() {
                        let gen_start = self.active_prefills[i].work.assistant_content_start;
                        let offs = self.active_prefills[i].work.projection_offsets.clone();
                        for seg in 0..offs.len() {
                            let prev_off = if seg == 0 { gen_start } else { offs[seg - 1] };
                            let mut ev = comp.clone();
                            ev.start_token = prev_off.saturating_sub(gen_start);
                            let _ = self.active_prefills[i]
                                .work
                                .event_tx
                                .send(TurnEvent::Projection(ev));
                        }
                        self.active_prefills[i].next_projection = offs.len();
                    }
                    let _ =
                        self.active_prefills[i]
                            .work
                            .event_tx
                            .send(TurnEvent::PrefillProgress {
                                tokens_done: total,
                                tokens_total: total,
                            });
                    if let Some(l) = member_logits.get(k) {
                        // DEEP-copy the final-logits row at capture. `Tensor::clone`
                        // is shallow (shared storage), and this tensor is HELD until
                        // the once-per-wave `promote_finished_prefills_to_decodes`
                        // samples the turn's FIRST token from it — up to a whole
                        // decode quantum later. The wave's forward path reuses its
                        // output buffers, so by promotion time the shared storage
                        // holds a LATER step's logits for some other slot: the first
                        // token gets sampled from a foreign distribution, and a
                        // greedy summary anchors on it and coherently continues in
                        // whatever language that row suggests (the stored CJK drift,
                        // 0.007%→0.135% at 42553ca3, amplified later by longer
                        // quanta). A real copy makes the captured row immutable —
                        // one ~vocab-sized row per completed prefill, negligible.
                        let owned = l.copy().unwrap_or_else(|_| l.clone());
                        self.active_prefills[i].final_logits = Some(owned);
                    }
                }
                WaveMember::Section {
                    seq_id: sid,
                    advance,
                } => {
                    let Some(i) = self
                        .active_section_ingests
                        .iter()
                        .position(|s| s.sequence_id.0 == sid)
                    else {
                        continue;
                    };
                    if let Err(e) = self.session.advance_sequence(sid, advance) {
                        self.active_section_ingests[i].error = Some(ConversationError::Model(e));
                        continue;
                    }
                    let seq_id = self.active_section_ingests[i].sequence_id;
                    let off = self.active_section_ingests[i].offset;
                    let end = (off + advance).min(self.active_section_ingests[i].tokens.len());
                    let chunk_tokens = self.active_section_ingests[i].tokens[off..end].to_vec();
                    super::Scheduler::record_slot_tokens(
                        &mut self.slot_tokens,
                        seq_id,
                        &chunk_tokens,
                    );
                    self.active_section_ingests[i].offset = end;
                }
            }
        }
        self.reset_wave_prefill();
    }

    /// Route a wave-group forward failure. On device-OOM, requeue the prefill
    /// members' scope prefills ([`Self::handle_prefill_oom`]) — section members and
    /// dialogue turns just retry next wave once the group is dropped. On any other
    /// error, surface it on each member's backing entry. Always resets the group.
    fn fail_wave_group(
        &mut self,
        members: &[WaveMember],
        prefill_gidxs: &[usize],
        err: &candle::Error,
    ) {
        if candle_nn::kv_cache::is_device_oom(err) {
            self.handle_prefill_oom(prefill_gidxs, err);
        } else {
            let msg = format!("wave group forward failed: {err}");
            for m in members {
                match *m {
                    WaveMember::Prefill { seq_id } => {
                        if let Some(i) = self
                            .active_prefills
                            .iter()
                            .position(|p| p.work.sequence_id.0 == seq_id)
                        {
                            self.active_prefills[i].error =
                                Some(ConversationError::Channel(msg.clone()));
                        }
                    }
                    WaveMember::Section { seq_id, .. } => {
                        if let Some(i) = self
                            .active_section_ingests
                            .iter()
                            .position(|s| s.sequence_id.0 == seq_id)
                        {
                            self.active_section_ingests[i].error =
                                Some(ConversationError::Channel(msg.clone()));
                        }
                    }
                }
            }
        }
        self.reset_wave_prefill();
    }

    /// Consume this wave's deferred gap-fill plans into a co-batchable glue group
    /// `(parent slot ids, glue-token input tensors, per-slot scatter descriptors)`.
    ///
    /// Deferred glue is ingest / compression gap-fill — a pure K/V scatter whose
    /// content prefills through a *separate* unit later (`apply_segments`), so it
    /// has no same-wave, same-slot consumer and can ride the wave as a full-sweep
    /// member alongside decode rather than a separate drain forward. `mem::take`
    /// consumes it once; later decode steps this wave see an empty queue. Returns
    /// `None` when nothing was deferred (or every plan was empty).
    fn take_wave_glue(&mut self) -> Option<(Vec<usize>, Vec<Tensor>, Vec<PendingGlue>)> {
        if self.deferred_glue_fires.is_empty() {
            return None;
        }
        let plans = std::mem::take(&mut self.deferred_glue_fires);
        let mut ids: Vec<usize> = Vec::with_capacity(plans.len());
        let mut inputs: Vec<Tensor> = Vec::with_capacity(plans.len());
        let mut pending: Vec<PendingGlue> = Vec::with_capacity(plans.len());
        for p in &plans {
            if p.glue_tokens.is_empty() {
                continue;
            }
            let input = match Tensor::new(p.glue_tokens.as_slice(), &self.device)
                .and_then(|t| t.unsqueeze(0))
            {
                Ok(t) => t,
                Err(e) => {
                    tracing::error!("wave glue input build failed: {e}");
                    continue;
                }
            };
            ids.push(p.parent_id.0);
            inputs.push(input);
            pending.push(PendingGlue {
                write_slice: p.glue_write_slice.clone(),
                write_in_blk: p.glue_write_in_blk.clone(),
                fwd_ahead: p.fwd_ahead.clone(),
            });
        }
        if ids.is_empty() {
            None
        } else {
            Some((ids, inputs, pending))
        }
    }

    /// Reconcile each slot's logical offset with its physical backing length —
    /// the wave-boundary invariant every member must satisfy: the varlen
    /// metadata (`cu_seqlens` / `kv_lens`, built from `session.offset`) and the
    /// slot headers (built from the live block table) describe the SAME slot,
    /// and the attention kernels resolve every `[0, kv_len)` position through
    /// the table. Any divergence sends the kernel past the slot's staged state
    /// into neighboring uploads (garbage slice indices → wild record pointers
    /// → CUDA_ERROR_ILLEGAL_ADDRESS, or silent cross-slot attention reads).
    ///
    /// Two producers, one per direction:
    /// - backing > offset: the co-batched glue scatter reserved gap chunk
    ///   space the unified wave didn't reflect in the slot's logical offset.
    ///   Left as-is, the NEXT prefill computes its write region from the
    ///   stale, shorter offset and clobbers the occupied `[offset, backing)`
    ///   span. Advance the offset up to the backing. (Previously a hard
    ///   assert that aborted the whole wave — the crash root at 42553ca3.)
    /// - offset > backing: a projection injected FEWER tokens than the
    ///   planner counted (`select-promote` drops sections it cannot lift to
    ///   hot under VRAM pressure), leaving the offset counting KV that never
    ///   landed. Clamp the offset down to the backing — positions are
    ///   slot-relative (slice ropes), so the clamped value is also the
    ///   correct RoPE base for the new tokens.
    fn reconcile_wave_offsets(&mut self, ids: &[usize]) -> candle::Result<()> {
        for &id in ids {
            let session_off = self.session.sequence_offset(id).unwrap_or(0);
            // Physical ground truth: the token count the live block table
            // actually covers (the same walk the slot-header build performs).
            // NOT `current_seq_len` — that is the write cursor and reads 0 for
            // freshly injected slots whose tables already hold sealed tokens.
            let backing_len = self
                .session
                .sequence_backing_tokens(id)
                .unwrap_or(session_off);
            if backing_len > session_off {
                self.session
                    .advance_sequence(id, backing_len - session_off)
                    .map_err(|e| candle::Error::Msg(format!("reconcile_wave_offsets: {e}")))?;
                tracing::debug!(
                    slot = id,
                    from = session_off,
                    to = backing_len,
                    "slot offset reconciled up to backing length"
                );
            } else if backing_len < session_off {
                self.session
                    .set_sequence_offset(id, backing_len)
                    .map_err(|e| candle::Error::Msg(format!("reconcile_wave_offsets: {e}")))?;
                tracing::warn!(
                    slot = id,
                    offset = session_off,
                    backing = backing_len,
                    "slot offset AHEAD of backing — clamped down (projection dropped \
                     sections it could not lift; kv metadata must describe the \
                     physical backing)"
                );
            }
        }
        Ok(())
    }

    /// Concatenate the present residual parts along the token dim (1) in the given
    /// caller order, skipping `None` parts. Returns `None` when all are absent.
    fn cat_caller_residual(parts: &[Option<&Tensor>]) -> candle::Result<Option<Tensor>> {
        let present: Vec<&Tensor> = parts.iter().filter_map(|p| *p).collect();
        match present.len() {
            0 => Ok(None),
            1 => Ok(Some(present[0].clone())),
            _ => Ok(Some(Tensor::cat(&present, 1)?)),
        }
    }

    /// The unified continuous-fair-wave step (`docs/continuous_fair_waves.md`): ONE
    /// forward folding every class of work through the shared grouped GEMM so one
    /// expert load per layer serves them all — the whole point on the streaming box.
    ///
    /// Two kinds of member co-batch here:
    /// - **Full-sweep** — decode (1 token/seq) and glue (deferred gap-fill scatter).
    ///   Both traverse all `N` layers every wave.
    /// - **Creep** — the wave group: dialogue prefills plus section-ingest chunks.
    ///   The group shares the GEMM only in `[cursor, win_end)`, its inter-layer
    ///   residual held across waves so the full-sweep members overtake it.
    ///
    /// So the sweep splits into up to THREE segments — `[0, cursor)` and
    /// `[win_end, N)` carry only the full-sweep members, `[cursor, win_end)` adds
    /// the creep. `forward_wave` returns the residual in CALLER order
    /// `[decode | creep | glue]`, so the segment boundaries split it by contiguous
    /// group: the creep is held WHOLE, the full-sweep members `[decode | glue]`
    /// continue. At the head, per-sequence logits are `[decode | creep]` (glue
    /// logits, if present, trail and are discarded): prefills promote, sections seal.
    ///
    /// With no creep group, all members are full-sweep: one `[0, N)` forward folding
    /// decode + a standalone section chunk + glue. The glue is a side effect only —
    /// its logits discarded and it must not advance its slot (asserted after).
    ///
    /// Called per decode step; the cohort/section/glue fold in on the first step
    /// (`wave_cohort_advanced` / `wave_section_advanced` guards, `take_wave_glue`
    /// drains once), the rest are plain decode.
    pub(super) fn decode_forward_cobatched(
        &mut self,
        decode_seqs: &[usize],
        decode_inputs: &[Tensor],
    ) -> candle::Result<Vec<Tensor>> {
        let n = self.model.num_layers().max(1);
        let n_dec = decode_seqs.len();
        let none_seqs: [usize; 0] = [];
        let none_inputs: [Tensor; 0] = [];
        // Wave-step wall-clock, shared across the co-batched classes so the prefill
        // and section throughput panels reflect the CONCURRENT rate (they ride
        // decode's sweep in one forward) rather than reading zero.
        let t_wave = Instant::now();

        // Fold this wave's deferred glue in as a full-sweep member co-batched with
        // decode (see `take_wave_glue`). A slot that decodes this wave is never
        // also a glue member — `take_active_decode_batch` excludes slots with a
        // pending deferred glue fire (they reproject this wave and resume decode
        // next), so the two groups are disjoint and the assembled context list
        // never lists a slot twice.
        let glue = self.take_wave_glue();
        let (glue_seqs, glue_inputs): (&[usize], &[Tensor]) = match &glue {
            Some((ids, ins, _)) => (ids.as_slice(), ins.as_slice()),
            None => (&none_seqs, &none_inputs),
        };
        let glue_pending: Option<&Vec<PendingGlue>> = glue.as_ref().map(|(_, _, p)| p);
        let has_glue = !glue_seqs.is_empty();
        let glue_tok: usize = glue_inputs
            .iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(0))
            .sum();
        // A "full-sweep" wave carries decode and/or glue across all N layers; it
        // drives segments 1 and 3. With neither, only the creep runs (seg 2).
        let has_fullsweep = n_dec > 0 || has_glue;

        let budget = self.wave_prefill_layer_budget();
        let cursor = self.wave_prefill_cursor;
        let win_end = (cursor + budget).min(n);

        // Form/resume the creep group (dialogue prefills + section chunks) unless it
        // was already advanced this wave. A fresh group folds section chunks in to
        // co-batch the creep (`form_wave_group(true)`), unless the standalone
        // section pass already ran this wave (no decode present).
        let (members, seq_ids, inputs, prefill_gidxs) = if !self.wave_cohort_advanced {
            if cursor == 0 && self.wave_prefill_residual.is_none() {
                self.form_wave_group(!self.wave_section_advanced);
            }
            self.build_wave_group_inputs()
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        // No creep group → one full-sweep [0, N) forward folding decode + a
        // standalone section chunk (if pending) + glue. All full-sweep, no residual
        // to hold; logits `[decode | section]` split at n_dec (glue logits, if any,
        // trail and are discarded).
        if seq_ids.is_empty() {
            let section = if !self.wave_cohort_advanced && !self.wave_section_advanced {
                if self.vram_under_pressure() {
                    self.relieve_vram_pressure("section", VramPhase::Load);
                }
                self.build_section_batch()
            } else {
                None
            };
            let (sec_seqs, sec_inputs, sec_gidx, sec_adv) = match section {
                Some((s, i, g, a)) => (s, i, g, a),
                None => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            };
            if sec_seqs.is_empty() && !has_fullsweep {
                // Nothing to run: no creep, no section, no decode, no glue.
                return Ok(Vec::new());
            }
            if !sec_seqs.is_empty() {
                self.wave_section_advanced = true;
            }
            if let Some(p) = glue_pending {
                self.session.set_pending_glue(p.clone());
            }
            let out = self.model.forward_wave(
                &mut self.session,
                decode_seqs,
                decode_inputs,
                &sec_seqs,
                &sec_inputs,
                glue_seqs,
                glue_inputs,
                0,
                n,
                None,
            )?;
            if has_glue {
                self.reconcile_wave_offsets(glue_seqs)?;
            }
            let logits = out.logits.unwrap_or_default();
            let d = n_dec.min(logits.len());
            let dec_logits = logits[..d].to_vec();
            if !sec_gidx.is_empty() {
                // Attended-KV summed before `complete_section_chunk` advances the
                // sequences. One record per co-batched section chunk.
                let sec_kv: usize = sec_seqs
                    .iter()
                    .map(|&sid| self.session.sequence_offset(sid).unwrap_or(0))
                    .sum();
                self.wave_stats.record_section(
                    sec_seqs.len(),
                    sec_adv.iter().sum(),
                    sec_kv,
                    t_wave.elapsed().as_millis() as u64,
                );
                super::PREFILL_OK_TOKENS.fetch_add(
                    sec_adv.iter().sum::<usize>() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                self.complete_section_chunk(&sec_gidx, &sec_adv);
            }
            return Ok(dec_logits);
        }

        // Creep group present. Full-sweep members (decode + glue) ride all N layers;
        // the creep rides only [cursor, win_end), its residual held WHOLE between
        // waves. The residual crosses `forward_wave` in caller order
        // `[decode | creep | glue]`, split by contiguous group at the boundaries.
        self.wave_cohort_advanced = true;
        let creep_tok: usize = inputs
            .iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(0))
            .sum();

        // Segment 1 — full-sweep members only over [0, cursor). Runs when there is
        // any full-sweep member (decode or glue) and cursor > 0; the creep resumes
        // from its held residual at `cursor`. Yields caller order `[decode | glue]`.
        let seg1_res: Option<Tensor> = if cursor > 0 && has_fullsweep {
            if let Some(p) = glue_pending {
                self.session.set_pending_glue(p.clone());
            }
            self.model
                .forward_wave(
                    &mut self.session,
                    decode_seqs,
                    decode_inputs,
                    &none_seqs,
                    &none_inputs,
                    glue_seqs,
                    glue_inputs,
                    0,
                    cursor,
                    None,
                )?
                .residual
        } else {
            None
        };
        // Split seg1's `[decode | glue]` so the creep residual inserts between them
        // for seg2's `[decode | creep | glue]` order.
        let (seg1_dec, seg1_glue): (Option<Tensor>, Option<Tensor>) = match &seg1_res {
            Some(r) => {
                let dec = if n_dec > 0 {
                    Some(r.narrow(1, 0, n_dec)?)
                } else {
                    None
                };
                let g = if glue_tok > 0 {
                    Some(r.narrow(1, n_dec, glue_tok)?)
                } else {
                    None
                };
                (dec, g)
            }
            None => (None, None),
        };

        // Segment 2 — full-sweep members + creep over [cursor, win_end). Input
        // residual caller order `[decode | creep | glue]`; at cursor 0 all embed
        // fresh (None).
        let pf_res = self.wave_prefill_residual.take();
        let seg2_in =
            Self::cat_caller_residual(&[seg1_dec.as_ref(), pf_res.as_ref(), seg1_glue.as_ref()])?;
        if let Some(p) = glue_pending {
            self.session.set_pending_glue(p.clone());
        }
        // Time seg2 alone (the co-batch the creep actually rides) — seg1 is
        // decode+glue over [0, cursor), which the creep did NOT ride, so charging
        // its wall-clock to the prefill channel would understate the prefill rate.
        let t_seg2 = Instant::now();
        let seg2 = match self.model.forward_wave(
            &mut self.session,
            decode_seqs,
            decode_inputs,
            &seq_ids,
            &inputs,
            glue_seqs,
            glue_inputs,
            cursor,
            win_end,
            seg2_in,
        ) {
            Ok(s) => s,
            Err(e) => {
                // Drop the creep group cleanly (requeue scope prefills on OOM) so a
                // fresh one forms next wave, then surface the error.
                self.fail_wave_group(&members, &prefill_gidxs, &e);
                return Err(e);
            }
        };

        // Record the co-batched creep throughput — prefill and section members are
        // tallied into their own channels, sharing seg2's wall-clock (the forward
        // they rode concurrently with decode) so the dashboard shows their CONCURRENT
        // rate instead of reading zero. One record per wave; KV is summed now, before
        // `complete_wave_group` advances the sequences at the head.
        {
            let ms = t_seg2.elapsed().as_millis() as u64;
            let (mut pf_seqs, mut pf_tok, mut pf_kv) = (0usize, 0usize, 0usize);
            let (mut sc_seqs, mut sc_tok, mut sc_kv) = (0usize, 0usize, 0usize);
            for (m, inp) in members.iter().zip(inputs.iter()) {
                let tok = inp.dims().get(1).copied().unwrap_or(0);
                match m {
                    WaveMember::Prefill { seq_id } => {
                        pf_seqs += 1;
                        pf_tok += tok;
                        pf_kv += self.session.sequence_offset(*seq_id).unwrap_or(0);
                    }
                    WaveMember::Section { seq_id, .. } => {
                        sc_seqs += 1;
                        sc_tok += tok;
                        sc_kv += self.session.sequence_offset(*seq_id).unwrap_or(0);
                    }
                }
            }
            if pf_seqs > 0 {
                self.wave_stats.record(true, pf_seqs, pf_tok, pf_kv, ms);
            }
            if sc_seqs > 0 {
                self.wave_stats.record_section(sc_seqs, sc_tok, sc_kv, ms);
            }
            // Every completed wave forward is OOM-free prefill throughput —
            // the progress signal the stall-grace gate and evidence reopen
            // read. Without this, pump-driven phases (scope ingest, section
            // creep) look stalled to the admission regulator even at full
            // throughput, because only the drain-path prefills tick it.
            if pf_tok + sc_tok > 0 {
                super::PREFILL_OK_TOKENS.fetch_add(
                    (pf_tok + sc_tok) as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
        }

        if win_end >= n {
            // Head reached: per-sequence logits, caller order `[decode | creep |
            // glue]`. Decode first; creep members next (promote/seal); glue logits,
            // if present, trail and are discarded.
            if has_glue {
                self.reconcile_wave_offsets(glue_seqs)?;
            }
            let logits = seg2.logits.unwrap_or_default();
            let d = n_dec.min(logits.len());
            let creep_end = (d + members.len()).min(logits.len());
            let dec_logits = logits[..d].to_vec();
            let member_logits = logits[d..creep_end].to_vec();
            self.complete_wave_group(&members, &member_logits);
            return Ok(dec_logits);
        }

        // Paused: split seg2's `[decode | creep | glue]` residual. Hold the creep
        // whole; continue the full-sweep members `[decode | glue]` into seg3.
        let res = seg2
            .residual
            .ok_or_else(|| candle::Error::Msg("co-batch wave: missing residual".into()))?;
        let dec_part = if n_dec > 0 {
            Some(res.narrow(1, 0, n_dec)?)
        } else {
            None
        };
        let creep_part = res.narrow(1, n_dec, creep_tok)?;
        let glue_part = if glue_tok > 0 {
            Some(res.narrow(1, n_dec + creep_tok, glue_tok)?)
        } else {
            None
        };
        self.wave_prefill_residual = Some(creep_part);
        self.wave_prefill_cursor = win_end;
        self.wave_prefill_members = members;

        // Segment 3 — full-sweep members only over [win_end, N). Input caller order
        // `[decode | glue]`. Skipped when there is no full-sweep member (the creep
        // paused at win_end, nothing else to sweep).
        if !has_fullsweep {
            return Ok(Vec::new());
        }
        let seg3_in = Self::cat_caller_residual(&[dec_part.as_ref(), glue_part.as_ref()])?;
        if let Some(p) = glue_pending {
            self.session.set_pending_glue(p.clone());
        }
        let seg3 = self.model.forward_wave(
            &mut self.session,
            decode_seqs,
            decode_inputs,
            &none_seqs,
            &none_inputs,
            glue_seqs,
            glue_inputs,
            win_end,
            n,
            seg3_in,
        )?;
        if has_glue {
            self.reconcile_wave_offsets(glue_seqs)?;
        }
        Ok(seg3.logits.unwrap_or_default())
    }

    /// Handle a device-OOM from the ragged prefill forward: the batch was too
    /// wide for the card. Narrow the admission window (so subsequent waves run
    /// smaller forwards) and surface the error on each in-batch prefill's caller
    /// channel.
    ///
    /// `group_idxs` are the `active_prefills` positions that were in this forward;
    /// they're still valid because nothing mutates `active_prefills` between the
    /// forward returning and this call.
    fn handle_prefill_oom(&mut self, group_idxs: &[usize], err: &candle::Error) {
        self.shrink_admit_window();
        let in_batch: HashSet<usize> = group_idxs.iter().copied().collect();
        let msg = format!("batched prefill forward failed: {err}");
        for (i, p) in self.active_prefills.iter_mut().enumerate() {
            if in_batch.contains(&i) {
                p.error = Some(ConversationError::Channel(msg.clone()));
            }
        }
    }

    /// Drain finished or errored entries from `active_prefills`. Errored
    /// entries emit `TurnEvent::Error`; finished entries are passed to
    /// `finalise_prefill` (which samples the first token and inserts into
    /// `active_decodes`).
    pub(super) fn promote_finished_prefills_to_decodes(&mut self) {
        // Use swap_remove for efficiency; iterate from the back.
        let mut i = 0;
        while i < self.active_prefills.len() {
            let done = {
                let p = &self.active_prefills[i];
                p.error.is_some() || (p.final_logits.is_some() && p.offset >= p.work.tokens.len())
            };
            if !done {
                i += 1;
                continue;
            }
            let p = self.active_prefills.swap_remove(i);
            let ActivePrefill {
                work,
                offset: _,
                next_projection: _,
                final_logits,
                error,
                prefill_start,
            } = p;
            // A compression-turn re-prefill carries no decode and reports to the
            // summariser, not a caller. Seal it directly off the wave (snapshot
            // the role-coherent K/V + record the turn) instead of running
            // `finalise_prefill`.
            if let SealAction::CompressionTurn { job_id } = &work.seal_action {
                let job_id = *job_id;
                let slot = work.sequence_id;
                match error {
                    Some(e) => {
                        if let Some(p) = self.pending_compression_seals.remove(&job_id) {
                            let _ = p
                                .response_tx
                                .send(Err(crate::summary_tree::ProbeError::Soft(format!(
                                    "SubmitSummaryProbe: reproject prefill: {e}"
                                ))));
                        }
                        self.free_summary_slot(slot);
                    }
                    None => self.complete_compression_turn(slot, job_id),
                }
                continue;
            }
            // A dialogue turn's reasoning-free re-prefill finished on the wave.
            // Seal the clean K/V + fire the deferred `Done` (no decode, reports to
            // the caller, not the summariser). On prefill error, surface it on the
            // caller channel and drop the slot's chunks.
            if let SealAction::TurnReprefill { pending_id } = &work.seal_action {
                let pending_id = *pending_id;
                match error {
                    Some(e) => {
                        if let Some(p) = self.pending_turn_seals.remove(&pending_id) {
                            let _ = p.event_tx.send(TurnEvent::Error(e));
                            let _ = self.session.truncate_sequence_to_blocks(p.parent_id.0, 0);
                        }
                    }
                    None => self.complete_turn_reprefill(pending_id),
                }
                continue;
            }
            if let Some(e) = error {
                let _ = work.event_tx.send(TurnEvent::Error(e));
                continue;
            }
            let logits = match final_logits {
                Some(l) => l,
                None => {
                    let _ = work
                        .event_tx
                        .send(TurnEvent::Error(ConversationError::Channel(
                            "prefill produced no final logits".into(),
                        )));
                    continue;
                }
            };
            let prefill_ms = prefill_start
                .map(|s| s.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let turn_start = work.submitted_at;
            let token_count = work.tokens.len();
            self.finalise_prefill(work, logits, prefill_ms, turn_start, token_count);
            // swap_remove pulled the last element into i; don't increment.
        }
    }

    /// Post-forward path shared by both single and batched prefill: sample
    /// the first token, emit it, and either transition to decode or close
    /// the turn out immediately on EOS / max_decode_tokens == 0.
    fn finalise_prefill(
        &mut self,
        work: PrefillWork,
        logits: Tensor,
        prefill_ms: f64,
        turn_start: Instant,
        token_count: usize,
    ) {
        // Total KV position after this prefill.
        let context_depth = self
            .session
            .sequence_offset(work.sequence_id.0)
            .unwrap_or(token_count);

        // Decode-start line: the effective sampling config this conversation turn
        // will decode under. Confirms empirically whether a turn is stochastic
        // (temp>0 + top_k/top_p) or greedy (temp≈0 → argmax), and at what context
        // depth. Enable with
        // `RUST_LOG=candle_conversation::scheduler::decode=debug`.
        tracing::trace!(
            target: "candle_conversation::scheduler::decode",
            seq = work.sequence_id.0,
            context_depth,
            prefill_tokens = token_count,
            max_decode_tokens = work.max_decode_tokens,
            temperature = work.sampling.temperature,
            top_k = work.sampling.top_k,
            top_p = work.sampling.top_p,
            repeat_penalty = work.sampling.repeat_penalty,
            segment_temp_boost = work.sampling.segment_temp_boost,
            dry = work.sampling.dry.is_some(),
            greedy = work.sampling.temperature <= 0.01,
            seed = work.sampling.seed,
            "conversation decode start",
        );

        let mut sampling_state = self
            .sampling_states
            .remove(&work.sequence_id)
            .expect("sampling state must exist for active sequence");
        sampling_state.end_turn();
        sampling_state.record_context_tokens(&work.tokens, self.sampler.max_recent_len());

        // Send prefill progress: complete (single-prefill path needs this;
        // batched path already streams progress per-chunk, but a final
        // tokens_done==tokens_total event is always benign).
        let _ = work.event_tx.send(TurnEvent::PrefillProgress {
            tokens_done: token_count,
            tokens_total: token_count,
        });

        let first_token = match self.sample_single(&logits, &work.sampling, &mut sampling_state) {
            Ok(t) => t,
            Err(e) => {
                self.sampling_states
                    .insert(work.sequence_id, sampling_state);
                let _ = work.event_tx.send(TurnEvent::Error(e));
                return;
            }
        };

        // Detect think-mode entry: the model opens its OWN `<think>` as the first
        // decoded token (we never prefill one). The `work.tokens` check covers a
        // caller-supplied assistant prefill that itself opens a think block.
        let initial_inside_think_block = {
            let tid = work.sampling.segment_open_token_id;
            if tid >= 0 {
                let tok = tid as u32;
                let prefill_has_think = work.tokens.iter().rev().take(5).any(|&t| t == tok);
                // The block opens either way: the common case is the model
                // sampling its OWN `<think>` as the first token; the rarer case is
                // a caller-supplied assistant prefill that already opens one.  In
                // BOTH cases the sampler's `in_segment` must flip — it gates the
                // reflection-marker suppression, the thinking temperature boost,
                // and the `</think>` EOT ramp (all keyed off `segment_len`, which
                // only advances while `in_segment`).  (DRY is no longer gated
                // here — it has its own `dry_span_len`/`dry_suppressed` scope,
                // reset at `<think>`/`</think>` via `enter_segment`/`exit_segment`.)
                // Flipping it only for the prefilled case left the sampler's flag
                // stuck false for a model-opened block, silently disabling every
                // one of those controls for its whole duration even though the
                // health flag (`inside_think_block`) correctly tracked it.
                let opens_think = prefill_has_think || first_token == tok;
                if opens_think && !sampling_state.in_segment {
                    sampling_state.enter_segment();
                }
                opens_think
            } else {
                false
            }
        };

        self.sampling_states
            .insert(work.sequence_id, sampling_state);

        // Per-token trace for the prefill-emitted first token.  Enable
        // with `RUST_LOG=candle_conversation::scheduler::sampling=trace`.
        // This is the canonical "what did the model say first?" diag —
        // an early-EOS bug very often shows up as the first sampled
        // token already being EOS, meaning the model's K/V context is
        // pushing logits onto the EOS column straight out of prefill.
        if tracing::enabled!(
            target: "candle_conversation::scheduler::sampling",
            tracing::Level::TRACE,
        ) {
            let decoded = self
                .tokenizer
                .decode(&[first_token], false)
                .unwrap_or_else(|_| "<?>".to_string());
            let first_token_is_eos = self.is_eos(first_token);
            tracing::trace!(
                target: "candle_conversation::scheduler::sampling",
                seq_id = work.sequence_id.0,
                step = 0,
                token_id = first_token,
                is_eos = first_token_is_eos,
                decoded = %decoded,
                "sampled token (prefill first)",
            );
            if first_token_is_eos {
                tracing::debug!(
                    target: "candle_conversation::scheduler::sampling",
                    seq_id = work.sequence_id.0,
                    token_id = first_token,
                    "EOS fired on the very first sampled token — model is \
                     producing EOS immediately after prefill; check K/V \
                     context coherence",
                );
            }
        }

        let sampling_temperature = work.sampling.temperature;

        if self.is_eos(first_token) || work.max_decode_tokens == 0 {
            // View sequences (SubmitTurn path): the prefill already wrote KV
            // blocks that must be finalized onto the parent and sealed into
            // the substrate.  Insert as a finished DecodeState so
            // cleanup_finished runs finalize_view + perform_seal_and_write.
            //
            // Non-view sequences (raw RULER / summarisation): no parent to
            // finalize and seal=None is correct — use the fast path.
            if self.turn_views.contains_key(&work.sequence_id) {
                self.active_decodes.insert(
                    work.sequence_id,
                    DecodeState {
                        event_tx: work.event_tx,
                        generated_tokens: TokenBuffer::from(vec![first_token]),
                        max_tokens: work.max_decode_tokens,
                        sampling_config: work.sampling,
                        seal_action: work.seal_action,
                        post_decode_tokens: work.post_decode_tokens,
                        belief: work.belief,
                        prefill_tokens: work.tokens,
                        user_text: work.user_text,
                        tags: work.tags,
                        user_content_start: work.user_content_start,
                        user_content_end: work.user_content_end,
                        assistant_content_start: work.assistant_content_start,
                        no_think: work.no_think,
                        prefill_assistant_text: work.prefill_assistant_text,
                        finished: true,
                        decode_start: Instant::now(),
                        prefill_ms,
                        prefill_token_count: context_depth,
                        turn_start,
                        health: {
                            let mut hs = crate::decode_health::DecodeHealthState::new(
                                self.health_config.repetition_window,
                                self.health_config.health_log_capacity,
                            );
                            hs.apply_baseline_config(
                                self.health_config.entropy_baseline_window,
                                self.health_config.entropy_trend_relative_factor,
                                self.health_config.entropy_trend_absolute_min_nats,
                            );
                            hs.inside_think_block = initial_inside_think_block;
                            hs.skip_entropy_checks = sampling_temperature <= 0.01;
                            hs
                        },
                        reprojection: work.reprojection,
                        non_punct_since_reproject: 0,
                        last_projection_end: 0,
                        in_tool_call: false,
                        triggers: work.triggers,
                        stencil: None,
                        pending_mask: None,
                    },
                );
            } else {
                self.finish_immediately(
                    work.sequence_id,
                    first_token,
                    &work.event_tx,
                    prefill_ms,
                    turn_start,
                    context_depth,
                );
            }
            return;
        }

        let _ = work.event_tx.send(TurnEvent::Token(first_token));

        // The first sampled token can itself be a stencil trigger — e.g. the
        // model emits `<tool_call>` as its very first response token, the common
        // case under /no_think (the think block is prefilled, so the model goes
        // straight to the call). The decode-loop trigger check runs only on
        // tokens sampled in `batch_decode_step`, never this one, so check it here
        // too — otherwise steering silently never engages for those calls.
        let stencil = work.triggers.driver_for(first_token);
        if let Some(d) = &stencil {
            tracing::debug!(
                target: "candle_conversation::stencil",
                seq_id = work.sequence_id.0,
                tree = d.tree().label(),
                trigger = first_token,
                "stencil steering started (trigger on the first decoded token)",
            );
        }
        // A first-token `<tool_call>` trigger enters the call immediately, so the
        // in-call state must be set HERE — the decode loop's `is_tool_open` scan
        // (which normally sets it) only sees tokens sampled in `batch_decode_step`,
        // never this one. Without it the in-call reprojection freeze never engages
        // for these turns and cadence/punctuation triggers re-orient the selection
        // mid-call. The early first-reprojection push below still fires once — it
        // is this turn's lock-in reprojection, exactly like the one `is_tool_open`
        // fires before freezing.
        let first_token_opens_call = stencil
            .as_ref()
            .is_some_and(|d| d.tree().label() == super::TOOL_CALL_TREE_LABEL);
        // Captured before `work.reprojection` moves into the DecodeState: the
        // early first-reprojection below fires only for turns whose target
        // layer runs belief-driven selection — a plain-prompt layer (the
        // titler's single-section schema) gains nothing from the extra swap.
        let wants_early_reprojection = work
            .reprojection
            .as_ref()
            .is_some_and(|p| p.has_belief_collections());

        self.active_decodes.insert(
            work.sequence_id,
            DecodeState {
                event_tx: work.event_tx,
                generated_tokens: TokenBuffer::from(vec![first_token]),
                max_tokens: work.max_decode_tokens,
                sampling_config: work.sampling,
                seal_action: work.seal_action,
                post_decode_tokens: work.post_decode_tokens,
                belief: work.belief,
                prefill_tokens: work.tokens,
                user_text: work.user_text,
                tags: work.tags,
                user_content_start: work.user_content_start,
                user_content_end: work.user_content_end,
                assistant_content_start: work.assistant_content_start,
                no_think: work.no_think,
                prefill_assistant_text: work.prefill_assistant_text,
                finished: false,
                decode_start: Instant::now(),
                prefill_ms,
                prefill_token_count: context_depth,
                turn_start,
                health: {
                    let mut hs = crate::decode_health::DecodeHealthState::new(
                        self.health_config.repetition_window,
                        self.health_config.health_log_capacity,
                    );
                    hs.apply_baseline_config(
                        self.health_config.entropy_baseline_window,
                        self.health_config.entropy_trend_relative_factor,
                        self.health_config.entropy_trend_absolute_min_nats,
                    );
                    hs.inside_think_block = initial_inside_think_block;
                    hs.skip_entropy_checks = sampling_temperature <= 0.01;
                    hs
                },
                reprojection: work.reprojection,
                non_punct_since_reproject: 0,
                last_projection_end: 0,
                in_tool_call: first_token_opens_call,
                triggers: work.triggers,
                stencil,
                pending_mask: None,
            },
        );
        // Fire the turn's FIRST reprojection immediately (drained right after
        // the next decode step, ~token 1). The prefill just wrote the user
        // query's wide-Q into R16, so the belief scan can score it and
        // materialize the right sections BEFORE the model's plan forms in the
        // early <think> tokens — waiting for the 64-token cadence lets a
        // wrong-tool prefix anchor the reasoning first (the submit-time
        // projection only carries the PREVIOUS turn's belief; it cannot see
        // this turn's query). For a first-token tool call this is the turn's
        // lock-in reprojection: `in_tool_call` is already set above, so the
        // call body stays frozen afterwards.
        if wants_early_reprojection {
            Self::queue_reprojection(&mut self.pending_reprojections, work.sequence_id);
        }
    }

    pub(super) fn run_prefill(
        &mut self,
        sequence_id: SequenceId,
        tokens: &[u32],
    ) -> Result<Tensor, ConversationError> {
        // Chunked prefill: split large prompts into bounded chunks to keep
        // intermediate activation buffers from growing unboundedly.
        let logits = if tokens.len() > self.max_prefill_pass_tokens {
            let mut last_logits: Option<Tensor> = None;
            for chunk in tokens.chunks(self.max_prefill_pass_tokens) {
                let input = Tensor::new(chunk, &self.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(ConversationError::Model)?;
                let nl = self.model.num_layers();
                let logits_vec = self
                    .model
                    .forward_wave(
                        &mut self.session,
                        &[],
                        &[],
                        &[sequence_id.0],
                        &[input],
                        &[],
                        &[],
                        0,
                        nl,
                        None,
                    )
                    .map(|s| s.logits.unwrap_or_default())
                    .map_err(ConversationError::Model)?;
                self.session
                    .advance_sequence(sequence_id.0, chunk.len())
                    .map_err(ConversationError::Model)?;
                super::Scheduler::record_slot_tokens(&mut self.slot_tokens, sequence_id, chunk);
                last_logits = logits_vec.into_iter().next();
            }
            last_logits.ok_or_else(|| {
                ConversationError::Channel("no logits returned from chunked prefill".into())
            })?
        } else {
            let input = Tensor::new(tokens, &self.device)
                .and_then(|t| t.unsqueeze(0))
                .map_err(ConversationError::Model)?;

            let logits_vec = self
                .model
                .forward_wave(
                    &mut self.session,
                    &[],
                    &[],
                    &[sequence_id.0],
                    &[input],
                    &[],
                    &[],
                    0,
                    self.model.num_layers().max(1),
                    None,
                )
                .map(|s| s.logits.unwrap_or_default())
                .map_err(ConversationError::Model)?;

            self.session
                .advance_sequence(sequence_id.0, tokens.len())
                .map_err(ConversationError::Model)?;

            // Mirror these tokens into the slot's diagnostic log so the
            // turn-complete dump can reconstruct the exact context the
            // kernel saw (compiled out without the `context-dump` feature).
            super::Scheduler::record_slot_tokens(&mut self.slot_tokens, sequence_id, tokens);

            logits_vec.into_iter().next().ok_or_else(|| {
                ConversationError::Channel("no logits returned from prefill".into())
            })?
        };

        // Single exit for every prefill path: the forward wrote KV without the
        // decode kernel's self-increment, so refresh the cached decode
        // slot-state's writer slice with the advanced tail length. Without
        // this, a mid-decode injection (a stencil static run, a think-steer
        // continuation) is INVISIBLE to the following decode steps — the
        // kernel attends the tail chunk at its stale pre-prefill length and
        // the model decodes as if the injected tokens were never written.
        // No-op for slots that haven't decoded yet.
        self.session
            .refresh_decode_slot_state(sequence_id.0)
            .map_err(ConversationError::Model)?;

        Ok(logits)
    }
}

/// Executes the VRAM Governor's relief ladder on the scheduler thread, borrowing
/// `&mut Scheduler` so each rung's mechanism (release / compact / hot→warm evict)
/// runs with full working-set context — never off-thread racing a live forward.
/// The governor calls [`relieve`](candle::vram::KvReliefDriver::relieve)
/// cheapest-rung-first and re-measures reality between rungs
/// (`docs/vram_governor_design.md` §8).
struct SchedulerReliefDriver<'a> {
    sched: &'a mut Scheduler,
    evicted: crate::substrate::EvictionReport,
    compressed: usize,
    flushed: bool,
    released: usize,
}

impl candle::vram::KvReliefDriver for SchedulerReliefDriver<'_> {
    fn relieve(&mut self, tier: candle::vram::Criticality, want: u64) -> u64 {
        use candle::vram::Criticality;
        // Nominal arena size for the freed-bytes estimate; the governor re-measures
        // the real headroom between rungs, so this only feeds logging/escalation.
        const ARENA_BYTES: u64 = 16 * 1024 * 1024;
        match tier {
            // Trivial: return empty arenas + reserved-but-free pool fragmentation
            // to the OS. No data movement, no hit-rate cost.
            Criticality::Trivial => {
                let t = std::time::Instant::now();
                let arenas = self.sched.session.release_empty_arenas().unwrap_or(0);
                self.released += arenas;
                self.sched.trim_kv_pool();
                let freed = arenas as u64 * ARENA_BYTES;
                if arenas > 0 {
                    self.sched.log_relief_event(
                        "Trivial",
                        "release-empty+trim",
                        want,
                        freed,
                        t.elapsed().as_millis() as u64,
                        format!("arenas_released={arenas}"),
                    );
                }
                freed
            }
            // Cheap: lossless compaction reclaims fragmented free space.
            Criticality::Cheap => {
                if self.sched.session.can_reclaim_arena() {
                    // Bounded defrag (one base budget); the ladder escalates the
                    // budget at Costly below. Consolidates the emptiest arenas
                    // first, then releases them.
                    let t = std::time::Instant::now();
                    let moves = self
                        .sched
                        .session
                        .defragment_bounded(compact_base_moves())
                        .unwrap_or(0);
                    let arenas = self.sched.session.release_empty_arenas().unwrap_or(0);
                    self.released += arenas;
                    self.sched.trim_kv_pool();
                    let freed = arenas as u64 * ARENA_BYTES;
                    if arenas > 0 || moves > 0 {
                        self.sched.log_relief_event(
                            "Cheap",
                            "compact",
                            want,
                            freed,
                            t.elapsed().as_millis() as u64,
                            format!("moves={moves} arenas_compacted={arenas}"),
                        );
                    }
                    freed
                } else {
                    0
                }
            }
            // Moderate: compress-to-free. Bring forward the quantization of
            // completed still-float turns — a NET SHRINK the persistence thread
            // would do anyway (float→quant in place), so it is cheaper than
            // eviction (which is a move to RAM, reloaded if re-attended). The
            // turn stays resident and attended-over; only the reusable float
            // working set shrinks. Ordered before `Costly` (evict) for exactly
            // that reason. The freed bytes are the real pool-used drop across the
            // pass + a sync (the float arenas free via `cuMemFreeAsync` as the old
            // hot `Arc`s drop; retire those frees so `pool_used` — and the reuse
            // term (`reserved − used`) the governor measures — reflects it).
            Criticality::Moderate => {
                let t = std::time::Instant::now();
                let used_before = self
                    .sched
                    .session
                    .vram_pool_stats()
                    .map(|(u, _)| u)
                    .unwrap_or(0);
                // Bound the batch: compress ~`want` (×hysteresis to coast) of
                // float per episode, capped, so a big backlog drains over several
                // episodes rather than one 66 s blocking pass of everything.
                let budget = want
                    .saturating_mul(VRAM_COMPRESS_HYSTERESIS)
                    .min(vram_compress_max());
                let compress_t = std::time::Instant::now();
                let turns = self.sched.compress_pending_turns(budget);
                let compress_ms = compress_t.elapsed().as_millis() as u64;
                self.compressed += turns;
                let arenas = self.sched.session.release_empty_arenas().unwrap_or(0);
                self.released += arenas;
                super::timed_synchronize(&self.sched.device);
                self.sched.trim_kv_pool();
                let used_after = self
                    .sched
                    .session
                    .vram_pool_stats()
                    .map(|(u, _)| u)
                    .unwrap_or(used_before);
                let freed = used_before.saturating_sub(used_after) as u64;
                if turns > 0 || arenas > 0 {
                    self.sched.log_relief_event(
                        "Moderate",
                        "compress-to-free",
                        want,
                        freed,
                        t.elapsed().as_millis() as u64,
                        format!(
                            "turns_compressed={turns} compress_ms={compress_ms} budget_mib={} arenas_released={arenas} pool_used {}->{}MiB",
                            budget / (1 << 20),
                            used_before / (1 << 20),
                            used_after / (1 << 20),
                        ),
                    );
                }
                freed
            }
            // Costly: drop the hot copies of the oldest warm-backed turns (a MOVE
            // to RAM, reloaded if re-attended — hence below compress), then
            // RECLAIM the freed VRAM.
            Criticality::Costly => {
                let t = std::time::Instant::now();
                // Evict already-warm turns first (cheap); only pay a short blocking
                // flush if that wasn't enough (under sustained pressure there are
                // usually plenty of warm turns, so we skip the multi-second wait).
                let evict_t = std::time::Instant::now();
                let mut rep = self.sched.evict_cold_tail(want);
                if rep.bytes < want {
                    self.flushed |= super::timed_wait(|| {
                        self.sched
                            .persist_trigger
                            .flush_blocking(std::time::Duration::from_secs(1))
                    });
                    let more = self.sched.evict_cold_tail(want.saturating_sub(rep.bytes));
                    rep.count += more.count;
                    rep.bytes += more.bytes;
                }
                let evict_ms = evict_t.elapsed().as_millis() as u64;
                self.evicted.count += rep.count;
                self.evicted.bytes += rep.bytes;
                // Reclaim. Eviction frees CHUNKS scattered across arenas, so the
                // arenas never go empty on their own and `cuMemFreeAsync` frees do
                // not retire without a sync — which is why `release`/`trim` were
                // reclaiming 0 and the ladder pointlessly escalated to Critical.
                // Consolidate the freed chunks into empty arenas (compact), retire
                // all the async frees + compaction copies with ONE sync, release
                // the now-empty arenas, and return the pool to the OS.
                // Deeper in the ladder ⇒ a bigger (but still bounded) defrag
                // budget than the Cheap rung — the "builds to more aggressive"
                // escalation.
                if self.sched.session.can_reclaim_arena() {
                    let _ = self
                        .sched
                        .session
                        .defragment_bounded(compact_base_moves().saturating_mul(3));
                }
                let arenas = self.sched.session.release_empty_arenas().unwrap_or(0);
                self.released += arenas;
                super::timed_synchronize(&self.sched.device);
                self.sched.trim_kv_pool();
                let freed = rep.bytes + arenas as u64 * ARENA_BYTES;
                if rep.count > 0 || arenas > 0 {
                    self.sched.log_relief_event(
                        "Costly",
                        "hot->warm evict",
                        want,
                        freed,
                        t.elapsed().as_millis() as u64,
                        format!(
                            "turns_evicted={} evicted_mib={} evict_ms={evict_ms} arenas_released={arenas} flush={}",
                            rep.count,
                            rep.bytes / (1 << 20),
                            self.flushed,
                        ),
                    );
                }
                freed
            }
            // Critical (warm→cold drop + expert-pool shrink) runs asynchronously
            // in the persistence thread the Costly flush kicked — no extra
            // synchronous work here.
            Criticality::Critical => 0,
        }
    }
}

#[cfg(test)]
mod host_ram_floor_tests {
    use super::host_ram_floor_bytes;

    /// The warm purge maintains `max(2 GiB, 5% x total)` free; the ingest
    /// throttle floor must sit at or below that so it fires only when the purge
    /// falls behind, never in steady state (which would throttle ingest forever).
    #[test]
    fn floor_stays_at_or_below_purge_target() {
        const GIB: u64 = 1024 * 1024 * 1024;
        for &total in &[16 * GIB, 64 * GIB, 189 * GIB, 512 * GIB] {
            let purge_target = std::cmp::max(2 * GIB, total / 20);
            let floor = host_ram_floor_bytes(total);
            assert!(
                floor <= purge_target,
                "floor {floor} must be <= purge target {purge_target} at total {total}"
            );
            assert!(
                floor >= 2 * GIB,
                "floor must keep at least 2 GiB of migration headroom"
            );
        }
    }
}
