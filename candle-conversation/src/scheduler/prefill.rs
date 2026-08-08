use super::admission::{
    admit_quantum, backlog_admit_action, budget_notches, decode_reserve_bytes, evidence_admit_grow,
    evidence_ticks_for, per_block_kv_bytes, plan_admission, prefill_cost_bytes, BacklogAction,
    BandParams, ThrottleReason,
};
use super::*;
use crate::persistence::thread::effective_turn_policy;
use crate::substrate::ConvCompression;
use crate::token_buffer::TokenBuffer;
use candle_transformers::models::batched_inference::PendingGlue;
use std::collections::{HashMap, HashSet};

/// Free KV regions kept in hand before [`Scheduler::vram_under_pressure_for`]
/// calls it pressure, as a divisor of the reservation's KV side plus an absolute
/// floor in regions. This is §3.8's setpoint.
///
/// It replaced a band of *bytes* derived from the driver — headroom held against
/// a wide forward's transient activation peak. That quantity is no longer the KV
/// side's business: transients come from the reservation's other end (§3.6), and
/// what a seal pass needs is simply somewhere to put its chunks. So the setpoint
/// asks the only question that remains, and asks it of an exact counter: are
/// there enough free regions to absorb the work already admitted?
///
/// Scaled to the span rather than fixed, so the same numbers hold on a 3.6 GiB
/// KV side and on the workstation's. Step 6 tunes both terms against the
/// observed claim rate; the floors are what keeps a small card from setting a
/// setpoint of two regions and stalling mid-seal.
const LOAD_SETPOINT_DIVISOR: usize = 8;
const LOAD_SETPOINT_FLOOR_REGIONS: usize = 24;
/// Decode's setpoint is half of load's: a decode step advances one token per
/// sequence, so KV grows by ~one chunk per sequence per 32 steps — orders of
/// magnitude slower than a prefill's upload, and the whole point of unbounded
/// context is to leave KV resident rather than evict it defensively.
const DECODE_SETPOINT_DIVISOR: usize = 16;
const DECODE_SETPOINT_FLOOR_REGIONS: usize = 8;

fn env_regions(var: &str) -> Option<usize> {
    std::env::var(var)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
}

/// Per-**sequence** transient-activation reserve for a LOAD-phase (prefill /
/// ingest) forward, in bytes. The reserve band grows by this coefficient for
/// each sequence co-batched into the imminent forward (see `vram_band_for`),
/// so a wide batch — which a large card admits — reserves in proportion to its
/// peak instead of a flat card fraction. Default 384 MiB: a prefill/ingest
/// sequence's activation buffers plus its share of the MoE expert gather.
/// Override with `CANDLE_VRAM_PER_SEQ_LOAD_MB` (the true value depends on the
/// model's per-token activation footprint and prefill width). Cached on first read.
const DEFAULT_VRAM_PER_SEQ_LOAD_MB: usize = 384;
fn per_seq_load_bytes() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let mb = std::env::var("CANDLE_VRAM_PER_SEQ_LOAD_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .filter(|&mb| mb > 0)
            .unwrap_or(DEFAULT_VRAM_PER_SEQ_LOAD_MB);
        mb * 1024 * 1024
    })
}

/// The region quantum in bytes.
fn region_bytes() -> u64 {
    candle_nn::kv_cache::REGION_BYTES as u64
}

/// The free-region setpoint for `phase`, in regions, given a KV side of
/// `total` regions. Pure — unit-tested in isolation.
fn setpoint_regions(phase: VramPhase, total: usize) -> usize {
    let (divisor, floor) = match phase {
        VramPhase::Load => (LOAD_SETPOINT_DIVISOR, LOAD_SETPOINT_FLOOR_REGIONS),
        VramPhase::Decode => (DECODE_SETPOINT_DIVISOR, DECODE_SETPOINT_FLOOR_REGIONS),
    };
    let floor = match phase {
        VramPhase::Load => env_regions("CANDLE_KV_FREE_REGIONS_LOAD").unwrap_or(floor),
        VramPhase::Decode => env_regions("CANDLE_KV_FREE_REGIONS_DECODE").unwrap_or(floor),
    };
    // Never ask for more than half the span: on a card too small to hold the
    // setpoint, demanding it would mean permanent pressure and an eviction pass
    // per wave that can never succeed.
    (total / divisor).max(floor).min(total / 2)
}

/// The phase a VRAM pressure decision is made in. Both phases read the same
/// exact counter — free regions — and differ only in how many they insist on:
///
/// - [`Load`](VramPhase::Load) — bringing KV into VRAM *before* attention
///   (prefill upload, section/scope ingest, warm→hot elevation). A wide ragged
///   forward claims regions fast, so the setpoint is wide enough that a seal
///   pass never finds the free list empty mid-wave.
/// - [`Decode`](VramPhase::Decode) — one token per sequence per step, so KV
///   grows slowly and predictably. A thin setpoint keeps the maximum KV
///   resident, which is the whole point of unbounded context.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum VramPhase {
    Load,
    Decode,
}

/// Regions the relief sequence frees past the setpoint, so a pass that just
/// clears pressure does not re-trip on the very next wave. Eviction is bulk and
/// coarse by nature — one turn's hot copy spans many chunks — so overshooting
/// deliberately is cheaper than nibbling every wave, which is what caused the
/// reload churn the old watermark ladder was built to damp.
const RELIEF_OVERSHOOT_REGIONS: usize = 8;

/// How long prefill throughput must be COMPLETELY silent (no forward
/// completing) under surviving VRAM pressure before the promote path halves
/// the admission window. Longer than any healthy forward (the widest
/// calibration forwards run ~7 s), so completions keep the width; a genuine
/// wedge still backs off, one halving per grace period. Device-OOM shrinks at
/// its own site instantly.
const PROMOTE_STALL_GRACE: std::time::Duration = std::time::Duration::from_secs(15);

/// Minimum wall-clock between "admitted nothing" throttle traces. The admission
/// pass runs many times a second, and a queue the budget won't take reproduces
/// the same line every iteration until the budget or the queue moves — without a
/// cooldown a single throttled ingest floods the log at the loop rate. Passes
/// that DID admit are never suppressed: their rate is bounded by real work.
const ADMIT_STARVED_LOG_INTERVAL: std::time::Duration = std::time::Duration::from_secs(2);

fn env_pct(var: &str, default: usize, max: usize) -> usize {
    std::env::var(var)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&p| p >= 1 && p <= max)
        .unwrap_or(default)
}
/// Capacity fraction (%) at which cold **ingest** KV starts demoting to the warm
/// (RAM) tier — gentle and early, well before the free-region setpoint is
/// approached at all. Ingest KV is zero-reload-cost (never
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
/// Slack the warm PIPELINE may hold above the standing budget: hot→warm output
/// that exists only while the drain moves it to cold. On a zero-budget machine
/// this is the only warm residency there ever is, and cutting admission for it
/// would recreate the ratchet-to-the-floor failure — the drain clears it in a
/// pass. The throttle fires only when `resident + pending` exceeds
/// `budget + slack`, i.e. when the drain is genuinely not keeping up. Default
/// 1 GiB; override with `CANDLE_WARM_PIPELINE_SLACK_MB`.
pub(super) fn warm_pipeline_slack_bytes() -> u64 {
    static V: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *V.get_or_init(|| {
        let mb = std::env::var("CANDLE_WARM_PIPELINE_SLACK_MB")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&mb| mb > 0)
            .unwrap_or(1024);
        mb * 1024 * 1024
    })
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
    /// Under VRAM pressure, shed until the free-region setpoint is met again,
    /// and report whether pressure **survived** the attempt.
    ///
    /// Cheapest first, each step run only if the one before it left pressure
    /// standing:
    ///
    ///  1. **Release empty arenas.** Under the reservation this is a free-list
    ///     push per region with no device work at all, so it is always worth
    ///     trying first — §3.8's "steal an empty region from any class".
    ///  2. **Evict resident galleries.** Belief-scan pages rebuild on demand
    ///     from the substrate blob, so dropping one costs only the rebuild.
    ///     They go before model KV for exactly that reason.
    ///  3. **Compress to free.** Bring forward the float→quant the persistence
    ///     thread would do anyway. A shrink in place rather than a move: the
    ///     turn stays resident and attended-over, and only its float working
    ///     set goes. Cheaper than eviction, which has to be reloaded if the
    ///     turn is re-attended.
    ///  4. **Evacuate.** Flush the pending hot→warm so just-sealed turns have a
    ///     warm copy — only warm-backed turns are evictable — then drop the hot
    ///     copies of the oldest ones. This is §3.8's evict-as-evacuation, and
    ///     it runs through the demotion path the tiering already owns; there is
    ///     no GPU→GPU compaction behind it any more.
    ///
    /// This ordering used to be the VRAM governor's relief ladder, each step a
    /// numbered `Criticality` rung with the governor re-measuring driver
    /// headroom between them to decide whether to climb. The rungs are gone:
    /// against an exact free-region count there is nothing to re-measure and
    /// nothing to arbitrate, so the priority is expressed as call order
    /// (`docs/archived/arena_unification.md` §5).
    ///
    /// Returns `true` if pressure is **still** on afterwards — the caller's
    /// signal to narrow the admission window, which is §3.8's third and last
    /// response. `whence` tags the log line with the calling gate.
    pub(super) fn relieve_vram_pressure(&mut self, whence: &str, phase: VramPhase) -> bool {
        let t = std::time::Instant::now();
        let Some(want) = self.relief_shortfall_bytes(phase) else {
            return false;
        };

        let mut released = self.session.release_empty_arenas().unwrap_or(0);
        let mut gallery_freed = 0u64;
        let mut compressed = 0usize;
        let mut flushed = false;
        let mut evicted = crate::substrate::EvictionReport { count: 0, bytes: 0 };

        // Gallery eviction — **this cannot clear the pressure below it**, and is
        // not here to.
        //
        // `evict_lru` drops `PageRun`s, returning pages to the gallery's own
        // `PagePool`. The VRAM behind them is `GalleryArena`'s `storage.slabs`,
        // which is only ever appended to (`add_slab`) and never shrunk, and
        // those slabs come from the CUDA pool rather than the KV reservation.
        // So `region_stats().free` is unchanged by this call and the next
        // `vram_under_pressure_for` is still true — `gallery_freed` counts bytes
        // returned to a free list, not to the card.
        //
        // Gallery growth is bounded by the arena itself now — it evicts to its
        // own ceiling at admission — so this no longer has to be the only limit,
        // and it must not fire merely because KV is tight. It used to: the test
        // was KV pressure alone, which this call cannot clear, so every episode
        // shed belief-scan residency that the next scan rebuilt from the
        // substrate. Now it only runs when the arena is *itself* over its
        // ceiling, which is the one case where evicting is the right answer and
        // the bytes are genuinely reclaimable.
        if self.vram_under_pressure_for(phase) {
            if let Some(arena) = self.gallery_arena.as_ref() {
                let cap = arena.cap_bytes();
                let resident = arena.resident_bytes();
                if resident > cap {
                    gallery_freed = arena.evict_lru((resident - cap).max(want));
                }
            }
        }

        if self.vram_under_pressure_for(phase) {
            // Bound the batch so a large backlog drains over several episodes
            // rather than one multi-second blocking pass over everything
            // pending; the persistence thread is working the same queue.
            let budget = want
                .saturating_mul(VRAM_COMPRESS_HYSTERESIS)
                .min(vram_compress_max());
            compressed = self.compress_pending_turns(budget);
            released += self.session.release_empty_arenas().unwrap_or(0);
        }

        if self.vram_under_pressure_for(phase) {
            evicted = self.evict_cold_tail(want);
            if evicted.bytes < want {
                // The blocking flush is only paid when the already-warm turns
                // were not enough: under sustained pressure there are usually
                // plenty of them, and this wait is measured in seconds.
                flushed = super::timed_wait(|| {
                    self.persist_trigger
                        .flush_blocking(VRAM_OFFLOAD_FLUSH_TIMEOUT)
                });
                let more = self.evict_cold_tail(want.saturating_sub(evicted.bytes));
                evicted.count += more.count;
                evicted.bytes += more.bytes;
            }
            released += self.session.release_empty_arenas().unwrap_or(0);
        }

        let still = self.vram_under_pressure_for(phase);
        let acted = released > 0 || gallery_freed > 0 || compressed > 0 || evicted.count > 0;
        if acted {
            relief_trace::note("sched", "relieve", want, evicted.bytes as u64);
        }
        let (free, setpoint) = self.kv_region_state(phase).unwrap_or((0, 0));
        // INFO when the pass actually shed something — that is a real event.
        // DEBUG otherwise: this runs from several gates every scheduler loop,
        // so an unconditional INFO floods the log under a sustained burst.
        macro_rules! emit {
            ($lvl:ident) => {
                tracing::$lvl!(
                    target: "candle_conversation::scheduler::timing",
                    whence,
                    want_mib = want / (1 << 20),
                    relief_ms = t.elapsed().as_millis() as u64,
                    warm_flushed = flushed,
                    gallery_freed_mib = gallery_freed / (1 << 20),
                    turns_compressed = compressed,
                    turns_evicted = evicted.count,
                    evicted_mib = evicted.bytes / (1 << 20),
                    arenas_released = released,
                    free_regions = free,
                    setpoint_regions = setpoint,
                    relieved = !still,
                    "KV region relief"
                )
            };
        }
        if acted {
            emit!(info);
        } else {
            emit!(debug);
        }
        still
    }

    /// Bytes one 32-token KV block costs across the whole model, in the formats
    /// a **live** sequence actually occupies — the unit every admission cost is
    /// quoted in. See [`per_block_kv_bytes`].
    ///
    /// ACTIVE formats, not the configured sealed ones: a block only reaches
    /// `k_format`/`v_format` once its turn seals and quantizes, and admission is
    /// deciding whether a sequence fits while it is running. Pricing the sealed
    /// pair understated the working set by ~3.7x (192 B/block active vs 52 B
    /// sealed), so admission cleared batches whose real KV was several GiB and
    /// the allocator then refused them one arena at a time. See
    /// [`candle_nn::kv_cache::active_kv_formats`].
    fn per_block_kv_bytes(&self) -> u64 {
        let (k, v) = self.session.active_kv_formats();
        per_block_kv_bytes(
            self.session.num_layers(),
            self.session.n_kv_head(),
            self.session.head_dim(),
            k,
            v,
        )
    }

    /// What the card can actually deliver to admission right now — the live
    /// ceiling [`Scheduler::admit_budget`] is clamped to on every read.
    ///
    /// Free reservation bytes plus reversibly-evictable KV, minus the hot KV the
    /// drain is skipping because it is pinned. The pinned discount is what keeps
    /// the forecast from reading its most optimistic exactly when the hot→warm
    /// drain has stalled: those bytes are counted as evictable but cannot be
    /// reclaimed at any price.
    ///
    /// The first term used to be a contest between three driver-derived
    /// estimates — governor headroom, the pool's reserved-but-free gap, and the
    /// allocator's own `init_free − pool_used − reserve` — clamped to whichever
    /// looked smallest, because each was wrong in a different regime. The worst
    /// was the reuse gap: admission once read 3045 MiB of it while `vram_free`
    /// was 0 and the pool held 15168 of 16375 MiB, admitted six prefills onto
    /// memory WDDM had already spilled, and the run aborted at ~3 tok/s. None of
    /// that survives the reservation. KV comes from regions that were claimed at
    /// startup, so what admission can spend is a count of the free ones, and no
    /// driver reading enters into it.
    ///
    /// Two corrections went with those estimates. One added what registered
    /// relievers claimed they could reversibly free; the other subtracted hot KV
    /// the drain was skipping because it was pinned, which the first had counted
    /// and could not actually reclaim. Both existed because the base number
    /// described *the card*. A free-region count describes what this process has
    /// claimed and not yet spent, so pinned KV is excluded by construction — it
    /// holds live regions — and evictable KV shows up as free regions the moment
    /// the relief pass ahead of admission actually evicts it. Measured, not
    /// forecast, which is why nothing has to be added back or discounted.
    pub(super) fn admit_budget_ceiling(&self) -> u64 {
        // No forward reserve is subtracted here: it is width-dependent, and
        // `plan_admission` holds it back at the width it is choosing. See
        // `admit_band_params`. The setpoint IS subtracted — those regions are
        // the relief pass's working room, not admission's to spend.
        let Some((free, setpoint)) = self.kv_region_state(VramPhase::Load) else {
            return 0;
        };
        (free.saturating_sub(setpoint) as u64).saturating_mul(region_bytes())
    }

    /// Bytes the work already in flight will still allocate this pass: every
    /// active prefill's *remaining* tokens, plus the amortised per-step growth of
    /// the live decodes. This is charged against the budget before anything new
    /// is admitted — committed work is never displaced by a fresh candidate.
    fn in_flight_cost_bytes(&self, per_block: u64) -> u64 {
        // KV only. The transient share of in-flight sequences is priced by the
        // reserve, which `plan_admission` evaluates at `live_width + n`.
        let prefill: u64 = self
            .active_prefills
            .iter()
            .filter(|p| p.error.is_none())
            .map(|p| prefill_cost_bytes(p.work.tokens.len().saturating_sub(p.offset), per_block))
            .sum();
        let sections: u64 = self
            .active_section_ingests
            .iter()
            .filter(|s| s.error.is_none())
            .map(|s| prefill_cost_bytes(s.tokens.len().saturating_sub(s.offset), per_block))
            .sum();
        prefill
            .saturating_add(sections)
            .saturating_add(decode_reserve_bytes(self.decode_width(), per_block))
    }

    /// Admit queued prefills against the VRAM byte budget.
    ///
    /// A burst of small parallel scopes (code_read's worker count), a bulk
    /// collection ingest's per-section prefills, or a batch of calibration cases
    /// all arrive here. What coalesces into one ragged forward is whatever fits
    /// the budget: the largest queued candidate that fits, then the rest of the
    /// queue in submission order (see [`plan_admission`]).
    ///
    /// [`Scheduler::MAX_PREFILL_WIDTH`] is a backstop above this, not the
    /// control; [`Scheduler::MIN_PREFILL_WIDTH`] keeps ≥1 in flight regardless,
    /// so an oversized lone turn still runs and is bounded by the per-arena VRAM
    /// gate (which compacts or fails fast rather than spilling to host memory).
    pub(super) fn promote_new_prefills(&mut self) {
        if self.prefill_queue.is_empty() {
            return;
        }
        let in_flight = self.active_prefills.len();
        if in_flight >= Self::MAX_PREFILL_WIDTH {
            return;
        }

        // VRAM-pressure backpressure. Each admitted prefill pins its
        // conversation's KV in VRAM, so under pressure we shed hot KV to the
        // substrate rather than piling on more concurrent prefills; if that
        // doesn't clear it, leave the rest queued this pass.
        if in_flight > 0 && self.vram_under_pressure() {
            if self.relieve_vram_pressure("promote", VramPhase::Load) {
                // Pressure survived eviction — stop piling on this pass (the
                // `in_flight > 0` guard keeps ≥1 in flight). The budget halves
                // only on a genuine THROUGHPUT STALL, never on the mere presence
                // of nominal pressure: multiplicative decrease is failure
                // evidence, and a card whose steady state sits just under the
                // pressure band would otherwise pin every bulk-prefill phase at
                // the floor.
                //
                // Stall detection is time-aware because this branch runs many
                // times a second while `PREFILL_OK_TOKENS` advances only when a
                // forward completes (seconds apart for wide forwards): a stall is
                // real only when NO forward has completed for a full
                // [`PROMOTE_STALL_GRACE`]. Each elapsed grace period backs off one
                // halving and re-arms; a device-OOM still cuts instantly at its
                // own site.
                let ok = super::PREFILL_OK_TOKENS.load(std::sync::atomic::Ordering::Relaxed);
                if ok > self.promote_ok_tokens_seen {
                    self.promote_ok_tokens_seen = ok;
                    self.promote_last_progress = Some(std::time::Instant::now());
                }
                let stalled = self
                    .promote_last_progress
                    .is_some_and(|t| t.elapsed() >= PROMOTE_STALL_GRACE);
                if stalled {
                    self.cut_admit_budget(ThrottleReason::ReliefSurvived);
                    self.promote_last_progress = Some(std::time::Instant::now());
                }
                return;
            }
        }

        let per_block = self.per_block_kv_bytes();
        // Two independent limits — see `plan_admission`. `available` is what the
        // card has and the forward reserve comes out of it; `setpoint` caps how
        // much KV admission may add. Do not pre-combine them.
        let available = self.admit_budget_ceiling();
        let setpoint = self.admit_budget;
        let live = self.in_flight_cost_bytes(per_block);
        let live_width = self.prefill_width() + self.section_ingest_width();
        let band = self.admit_band_params();
        let costs: Vec<u64> = self
            .prefill_queue
            .iter()
            .map(|w| prefill_cost_bytes(w.tokens.len(), per_block))
            .collect();
        let room = Self::MAX_PREFILL_WIDTH - in_flight;

        let mut plan = plan_admission(available, setpoint, live, live_width, &costs, room, &band);
        // Keep at least one prefill in flight even when nothing fits: an engine
        // that admits nothing makes no progress, and the alternative to a lone
        // oversized turn running is it never running at all. A turn forced
        // through here is still bounded by the per-arena VRAM gate, which
        // compacts or fails fast rather than spilling to host memory.
        //
        // The forced pick is the QUEUE HEAD, not the cheapest candidate.
        // Cheapest-first looks attractive — it fits the most work into a floored
        // budget — but under a budget that stays at the floor it becomes a
        // starvation loop: the expensive directories are never the cheapest, so
        // they are passed over on every pass while cheap ones keep arriving.
        // Measured on this workload with the budget pinned at 256 MiB and a
        // 384 MiB head: every forced admission took a 12-54 MiB candidate and no
        // large directory ever ran. FIFO bounds each item's wait by the queue
        // ahead of it.
        if plan.admitted.is_empty() && in_flight < Self::MIN_PREFILL_WIDTH && !costs.is_empty() {
            plan.spent = costs[0];
            plan.admitted.push(0);
            plan.skipped -= 1;
        }

        // Trace the pass whenever it admitted something — a real event, bounded in
        // rate by how fast work actually drains. A pass that admitted NOTHING is
        // the more interesting signal (queued work the budget won't take) but it
        // repeats every loop iteration until the budget or the queue moves, so it
        // is rate-limited to one line per [`ADMIT_STARVED_LOG_INTERVAL`].
        let starved = plan.admitted.is_empty();
        let due = self
            .last_admit_starved_log
            .is_none_or(|t| t.elapsed() >= ADMIT_STARVED_LOG_INTERVAL);
        if !starved || due {
            if starved {
                self.last_admit_starved_log = Some(std::time::Instant::now());
            }
            const MIB: u64 = 1 << 20;
            tracing::debug!(
                target: "candle_conversation::scheduler::throttle",
                available_mib = available / MIB,
                setpoint_mib = setpoint / MIB,
                live_mib = live / MIB,
                spent_mib = plan.spent / MIB,
                admitted = plan.admitted.len(),
                skipped = plan.skipped,
                in_flight,
                head_cost_mib = costs.iter().copied().max().unwrap_or(0) / MIB,
                reserve_mib = super::admission::reserve_for_width(
                    live_width + plan.admitted.len(),
                    &band,
                ) / MIB,
                "admission pass"
            );
        }

        // Remove by descending index so earlier positions stay valid as we take
        // them out of the queue.
        let mut take = plan.admitted;
        take.sort_unstable_by(|a, b| b.cmp(a));
        let mut admitted: Vec<PrefillWork> = take
            .into_iter()
            .filter_map(|i| self.prefill_queue.remove(i))
            .collect();
        // …then restore submission order among the admitted set.
        admitted.reverse();

        for work in admitted {
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

    /// The load-phase band's terms, for admission to evaluate at the width it is
    /// choosing rather than the width already in flight. Same `base`, `per_seq`
    /// and capacity clamp the pressure/relief gates use via
    /// [`Self::vram_band_for`] — one reserve law, two evaluation points.
    ///
    /// Splitting `per_seq` into a shared term (the MoE expert gather, which the
    /// whole batch pays once) plus a smaller marginal one is a measured dead end,
    /// however physical the argument sounds. At 512 MiB shared + 128 MiB
    /// marginal on the 16 GiB card it changed whole-phase calibration by 0.5%
    /// (981s → 976s, 472 → 475 tok/s) — inside run-to-run noise, because a wider
    /// batch holds more KV and `available` is measured after that KV lands, so
    /// width re-equilibrates at the same place. The same pass then lost 9 of 314
    /// directories to arena refusals where the unsplit law lost none.
    ///
    /// The trap that makes it look like a win: throughput decays as the substrate
    /// fills (the baseline's own halves run 595 then 329 tok/s), so a sample
    /// taken from the first minutes of calibration reads ~2x a sample taken from
    /// the middle. Compare whole phases, never windows.
    fn admit_band_params(&self) -> BandParams {
        BandParams {
            per_seq: per_seq_load_bytes() as u64,
            capacity: self
                .session
                .vram_governor()
                .map(|g| g.capacity())
                .unwrap_or(0),
        }
    }

    /// Free KV regions right now, and the setpoint for `phase` — the two
    /// numbers every pressure and admission decision is made from.
    ///
    /// `None` before the reservation exists, which the callers read as "no
    /// pressure, nothing to spend": there is no KV on the device yet to be
    /// under pressure about.
    fn kv_region_state(&self, phase: VramPhase) -> Option<(usize, usize)> {
        let stats = self.kv_regions()?;
        Some((stats.free, setpoint_regions(phase, stats.total)))
    }

    /// The KV side's region counters, or `None` before the reservation exists.
    fn kv_regions(&self) -> Option<candle_nn::kv_cache::RegionStats> {
        let candle::DeviceLocation::Cuda { gpu_id } = self.device.location() else {
            return None;
        };
        candle_nn::kv_cache::region_stats(gpu_id)
    }

    /// The card's resident capacity C (bytes) — the balloon-measured limit
    /// below which our footprint stays resident (no WDDM paging). Falls back to
    /// the driver's physical total until the balloon has measured C
    /// (`capacity()` is 0 then), so a threshold scaled by it is never a
    /// spurious zero at startup. `None` when unavailable.
    ///
    /// Only the host-side warm-tier thresholds still scale by C; the KV side's
    /// own pressure is counted in regions, not measured against the card.
    fn resident_capacity(&self) -> Option<usize> {
        self.session
            .vram_governor()
            .map(|g| g.capacity() as usize)
            .filter(|&c| c > 0)
            .or_else(|| self.session.vram_free_total().map(|(_, total)| total))
    }

    /// True when the KV side has fewer free regions than the setpoint — the
    /// signal to shed, and failing that to stop admitting.
    ///
    /// This used to be three gates in disjunction: a byte budget derived from
    /// `init_free − pool_used − reserve`, a driver-free floor qualified by how
    /// much the CUDA pool could still absorb by reuse, and a footprint gate on
    /// `pool_reserved` versus a compaction ceiling. Each existed because the
    /// other two were wrong in some regime, and the footprint gate needed a
    /// cooldown and a futility latch on top because a fragmented gap the engine
    /// kept reusing would otherwise report pressure on every scheduler loop.
    ///
    /// None of it survives the reservation. KV comes from regions claimed at
    /// startup, so the question "is there room?" has one exact answer that no
    /// driver reading enters into, and it cannot disagree with itself.
    ///
    /// Phase-independent default (`Load`, the wider setpoint). Prefer
    /// [`vram_under_pressure_for`](Self::vram_under_pressure_for) at call sites
    /// that know their phase.
    pub(super) fn vram_under_pressure(&self) -> bool {
        self.vram_under_pressure_for(VramPhase::Load)
    }

    /// Phase-aware pressure signal — see [`VramPhase`] for why the setpoint
    /// differs by phase.
    pub(super) fn vram_under_pressure_for(&self, phase: VramPhase) -> bool {
        self.kv_region_state(phase)
            .is_some_and(|(free, setpoint)| free < setpoint)
    }

    /// Bytes one relief pass should aim to free: enough to reach the setpoint
    /// plus [`RELIEF_OVERSHOOT_REGIONS`]. `None` when there is no pressure, so
    /// a relief call on a healthy cache costs one counter read.
    fn relief_shortfall_bytes(&self, phase: VramPhase) -> Option<u64> {
        let (free, setpoint) = self.kv_region_state(phase)?;
        if free >= setpoint {
            return None;
        }
        let target = setpoint.saturating_add(RELIEF_OVERSHOOT_REGIONS);
        Some((target.saturating_sub(free) as u64).saturating_mul(region_bytes()))
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

    /// Gentle-early ingest relief, run per-wave and long before the setpoint is
    /// approached. Once the KV side is more than [`ingest_demote_pct`] occupied
    /// (~50 % of its regions), shed the sealed, warm-backed KV of append-only
    /// ingest timelines down to a small rolling hot window
    /// ([`ingest_hot_window`]).
    ///
    /// Zero reload cost: ingest KV is never re-attended until query time, when
    /// it re-elevates warm→hot on demand. So it is the cheapest thing to shed
    /// and it sheds first, which is what keeps a bulk repo ingest from pinning
    /// a whole corpus hot until real pressure forces a much more expensive
    /// eviction of turns that are actually being attended.
    ///
    /// The watermark used to be `pool_used` against a fraction of the card.
    /// That reading no longer describes KV at all — the pool holds the model,
    /// the expert cache and a few scratches, so it sits at a high, flat
    /// fraction of C forever and the gate would fire on every wave regardless
    /// of how much ingest is resident. Occupancy of the KV span is the same
    /// question asked of the right counter.
    pub(super) fn demote_cold_ingest_if_pressured(&mut self) {
        if self.ingest_timelines.is_empty() {
            return;
        }
        let Some(stats) = self.kv_regions() else {
            return;
        };
        // Multiply before dividing. The same expression read `capacity / 100 *
        // pct` when `capacity` was bytes (~1.6e10), where the truncation was
        // invisible; `stats.total` is a region *count* in the hundreds, so
        // dividing first quantises the watermark to whole percent-of-100 steps
        // — and on any span below 100 regions it truncates to **zero**, which
        // the `live <= watermark` early-return below can never satisfy. That
        // turns the gentle-early rung into an unconditional full demote of the
        // ingest tail on every wave.
        let watermark = stats.total * ingest_demote_pct() / 100;
        if stats.live <= watermark {
            return;
        }
        let used = stats.live.saturating_mul(region_bytes() as usize);
        let watermark = watermark.saturating_mul(region_bytes() as usize);
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
        //    The test is whether step 1 *could* shed what it needed to, which is
        //    `report.bytes` against `target_bytes` — not the CUDA pool. This read
        //    the pool's `used`, which since KV moved to the reservation holds the
        //    model, the expert cache and the scratches: ~6.5 GiB against a
        //    region-derived watermark of ~2.4 GiB, so it was true on every wave
        //    and `nudged` recorded nothing. It is the same trap the doc comment
        //    above this function describes for the other gate.
        let nudged = if report.bytes < target_bytes {
            self.persist_trigger.fire();
            true
        } else {
            false
        };
        if report.count > 0 {
            // Freed hot arenas → release, so their regions return to the free
            // list where the pressure signal can see them.
            let _ = self.session.release_empty_arenas();
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
    /// Refresh the cached `sysinfo` reading at most once per
    /// [`HOST_RAM_PROBE_INTERVAL`] — never a per-wave syscall — and return the
    /// cached `(available, total)`. `(0, 0)` until the first probe.
    pub(super) fn host_ram_reading(&mut self) -> (u64, u64) {
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
        self.host_ram_probe
            .map(|(_, a, t)| (a, t))
            .unwrap_or((0, 0))
    }

    /// Whether the warm KV tier has outgrown its host-RAM budget PLUS the drain
    /// pipeline's slack — the condition under which slowing admission actually
    /// helps (less sealing → less hot→warm output). This replaced the absolute
    /// available-RAM floor, which our own resident weights held permanently
    /// true on any machine whose model fills RAM: an untestable condition that
    /// ratcheted the setpoint to the floor against structure, not pressure.
    pub(super) fn warm_over_budget(&mut self) -> bool {
        let (_, total) = self.host_ram_reading();
        if total == 0 {
            return false;
        }
        let budget = candle::vram::host_ram_budget(total);
        let usage = self
            .persist_trigger
            .warm_resident_bytes()
            .saturating_add(self.persist_trigger.pending_warm_bytes());
        usage
            > budget
                .kv_warm_budget_bytes
                .saturating_add(warm_pipeline_slack_bytes())
    }

    pub(super) fn regulate_ingest_admission(&mut self) {
        if self.ingest_timelines.is_empty() {
            return;
        }
        // Host-tier backpressure: throttle only when the warm KV tier has
        // outgrown its host-RAM budget plus the drain pipeline's slack — the one
        // host condition slowing admission can actually relieve. (The old
        // absolute available-RAM floor sat permanently tripped on any box whose
        // weights fill RAM, ratcheting the setpoint against structure.)
        if self.warm_over_budget() {
            self.cut_admit_budget_leveled(ThrottleReason::WarmOverBudget);
            return;
        }
        let Some(capacity) = self.resident_capacity() else {
            return;
        };
        let target = (capacity / 100 * ingest_warm_backlog_pct()) as u64;
        let backlog = self.persist_trigger.pending_warm_bytes();
        // "Is there room to reopen?" is asked against the STATIC bound, not the
        // live ceiling: this runs every wave, and the live ceiling costs a device
        // query plus a walk of the registered relievers. The live clamp still
        // happens where it matters — inside `raise_admit_budget`, and again at
        // admission time in `promote_new_prefills`.
        let ceiling = Self::max_admit_budget();
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
        match backlog_admit_action(
            backlog,
            target,
            self.admit_budget,
            ceiling,
            self.vram_under_pressure(),
        ) {
            // Drain falling behind the seal rate — throttle admission.
            BacklogAction::Shrink => self.cut_admit_budget_leveled(ThrottleReason::WarmBacklog),
            // Drain caught up and VRAM is clear — reopen a quantum.
            BacklogAction::Grow => {
                self.admit_grow_streak = 0;
                self.raise_admit_budget(ThrottleReason::DrainCaughtUp);
            }
            // Deadband — or growth blocked only by the pressure bit. The
            // evidence path reopens a wedged budget on proven OOM-free
            // throughput (see `evidence_admit_grow`); a real spike still cuts
            // instantly and resets the streak.
            BacklogAction::Hold => {
                let (grow, streak) = evidence_admit_grow(
                    backlog,
                    target,
                    self.admit_budget,
                    ceiling,
                    progressed,
                    self.admit_grow_streak,
                    // Cost scales with the budget already held, so the climb
                    // slows as it nears the budget that last collapsed instead
                    // of charging it at constant speed.
                    evidence_ticks_for(budget_notches(self.admit_budget, admit_quantum())),
                );
                self.admit_grow_streak = streak;
                if grow {
                    self.raise_admit_budget(ThrottleReason::Throughput);
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
    pub(super) fn reset_wave_prefill(&mut self) {
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
        // Not `let`: a residual/group mismatch below restarts the creep at layer 0
        // in place (see the check after `creep_tok`), which moves both.
        let mut cursor = self.wave_prefill_cursor;
        let mut win_end = (cursor + budget).min(n);

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

        // The held residual is a slice of a PREVIOUS wave's activations, sized by
        // that wave's creep membership. `build_wave_group_inputs` rebuilds the
        // group each wave and silently drops members that errored or completed, so
        // a mid-creep drop leaves a residual wider than the tokens it is about to
        // be paired with — the rows would then be attributed to the wrong members
        // for the remaining layers, and wrong activations are exactly what makes
        // the sampler emit token 0 forever.
        //
        // Recover rather than abort: the creep re-forms from layer 0 next wave.
        // That is idempotent — a prefill member re-feeds its whole token block
        // (`work.tokens[..]`, never a chunk) and its slot offset is only advanced
        // at completion, so re-running `[0, cursor)` rewrites the same KV at the
        // same positions. Deliberately NOT an assert: a hard assert on this path
        // is what aborted the whole wave at 42553ca3 (see `reconcile_wave_offsets`),
        // and a panic on the scheduler thread takes the daemon with it.
        if cursor > 0 {
            if let Some(res) = self.wave_prefill_residual.as_ref() {
                let held = res.dims().get(1).copied().unwrap_or(0);
                if held != creep_tok {
                    tracing::error!(
                        held_residual_tokens = held,
                        creep_tokens = creep_tok,
                        cursor,
                        members = members.len(),
                        "wave creep membership changed mid-sweep — the held residual \
                         no longer matches the group. Restarting the creep from \
                         layer 0; the affected prefill re-runs the layers it had \
                         already done.",
                    );
                    // Restart IN PLACE rather than re-entering: the wave's deferred
                    // glue was already drained by `take_wave_glue` above, so a
                    // recursive call would find an empty queue and silently drop it.
                    // Dropping the residual and rewinding the cursor gives the same
                    // fresh start — the group already rebuilt this wave is the
                    // consistent one, and seg1 is skipped once `cursor == 0`.
                    //
                    // Idempotent: a prefill member re-feeds its whole token block
                    // (`work.tokens[..]`, never a chunk) and its slot offset only
                    // advances at completion, so re-running `[0, cursor)` rewrites
                    // the same K/V at the same positions.
                    self.wave_prefill_residual = None;
                    self.wave_prefill_cursor = 0;
                    cursor = 0;
                    win_end = budget.min(n);
                }
            }
        }

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
    /// wide for the card. Cut the admission budget (so subsequent waves admit
    /// less) and surface the error on each in-batch prefill's caller channel.
    ///
    /// The hardest evidence the controller gets — a forward that actually failed
    /// — so it acts immediately here rather than waiting for the setpoint loop.
    ///
    /// `group_idxs` are the `active_prefills` positions that were in this forward;
    /// they're still valid because nothing mutates `active_prefills` between the
    /// forward returning and this call.
    fn handle_prefill_oom(&mut self, group_idxs: &[usize], err: &candle::Error) {
        let in_batch: HashSet<usize> = group_idxs.iter().copied().collect();
        self.cut_admit_budget(ThrottleReason::DeviceOom);
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
                        decode_busy_us: 0,
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
                decode_busy_us: 0,
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

#[cfg(test)]
mod setpoint_tests {
    use super::{setpoint_regions, VramPhase};

    /// The setpoint scales with the span so the same constants hold on this
    /// card's 226-region KV side and on the workstation's, and decode always
    /// insists on less than load — KV grows a chunk per sequence per 32 steps
    /// there, so evicting defensively would just cost reloads.
    #[test]
    fn the_setpoint_scales_with_the_span_and_decode_asks_for_less() {
        let load = setpoint_regions(VramPhase::Load, 800);
        let decode = setpoint_regions(VramPhase::Decode, 800);
        assert_eq!(load, 100, "load is span/8 once the span clears the floor");
        assert_eq!(decode, 50, "decode is span/16");
        assert!(decode < load);
    }

    /// On a span too small for the floors, the setpoint stops at half the span.
    /// Asking for more would mean permanent pressure: every wave would run a
    /// relief pass that cannot possibly reach a setpoint the card can't hold.
    #[test]
    fn a_small_span_clamps_to_half_rather_than_demanding_the_floor() {
        assert_eq!(setpoint_regions(VramPhase::Load, 32), 16);
        assert_eq!(setpoint_regions(VramPhase::Decode, 8), 4);
        assert_eq!(
            setpoint_regions(VramPhase::Load, 0),
            0,
            "no span, no demand"
        );
    }
}

#[cfg(test)]
mod warm_budget_tests {
    use super::warm_pipeline_slack_bytes;

    /// The slack exists so a zero-budget machine's transient drain traffic never
    /// reads as over-budget — tonight's healthy pipeline peaked ~0.7 GiB.
    #[test]
    fn default_slack_clears_a_healthy_drain_pipeline() {
        let slack = warm_pipeline_slack_bytes();
        assert!(slack >= 768 * 1024 * 1024, "slack {slack} too small");
    }
}
