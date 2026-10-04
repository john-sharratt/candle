use super::admission::{
    admit_quantum, budget_notches, evidence_admit_grow, evidence_ticks_for, per_block_kv_bytes,
    ThrottleReason,
};
use super::admit;
use super::admit_ground::AdmitPass;
use super::*;
use crate::persistence::thread::effective_turn_policy;
use crate::recorded_reply::replayed_step;
use crate::substrate::ConvCompression;
use crate::token_buffer::TokenBuffer;
use candle_nn::kv_cache::{end_wave_transient, is_device_oom};
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

/// What one [`Scheduler::compress_pending_turns`] pass achieved.
///
/// The two fields answer different questions and the caller needs both:
/// `compressed == 0` alone cannot distinguish "there was nothing pending" from
/// "the rung ran and the pool refused it ground", and those want opposite
/// responses — the first is a quiet pass, the second is the compress-to-free
/// rung failing at the moment compression is what would relieve the pressure.
#[derive(Default)]
pub(super) struct CompressPass {
    /// Turns whose hot copy was replaced by its quantized form.
    compressed: usize,
    /// The pass stopped early because a quantize destination could not be
    /// allocated, rather than because it ran out of work or hit its budget.
    refused: bool,
}

fn env_regions(var: &str) -> Option<usize> {
    std::env::var(var)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
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

/// What one KV compaction pass did: the arenas it emptied and handed back, and whether
/// its budget stopped it with work left.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct PackPass {
    pub(super) released: usize,
    pub(super) clipped: bool,
}

impl Scheduler {
    /// Pack the KV pools if they are fragmented enough to be worth a pass, then ask
    /// the weight side to take what packing released.
    ///
    /// **Gated on the gain, not run unconditionally.** A pass costs a device-wide
    /// sync and a copy per relocated chunk, so it is only worth making when there is
    /// real ground to recover: `MIN_FREEABLE_ARENAS` regions, which at 16 MiB each is
    /// the point where the expert cache can actually do something with the result.
    ///
    /// **And the second half is not optional.** Lowering the frontier only makes it
    /// *possible* for `weight_floor` to move left; `reclaim_spare_ground` is what
    /// moves it. Without that call a pass hands back regions nobody claims, decode is
    /// exactly as slow as before, and the work is invisible in every metric except
    /// the one that matters — which is why the two are one method and not two.
    ///
    /// **Considered every wave, run at most every `MIN_INTERVAL`.** Relief, which runs
    /// only under pressure and whose alternatives cost a turn or an expert, packs the KV
    /// pools directly and leaves the span tenants to this path.
    ///
    /// Answers the arenas the pass emptied and handed back to the region free list —
    /// zero when it did not run.
    pub(super) fn compact_kv_if_fragmented(&mut self) -> usize {
        /// Shortest interval between passes.
        ///
        /// The gate in `pack_kv_pools` is cheap, but the pass behind it is not: its
        /// census walks every arena's occupancy bitmap once per rung of the ladder.
        /// Running it on every wave-loop iteration would put that walk between every
        /// pair of forwards. This is what makes "every wave, cheap-signal gated" mean
        /// *considered* every wave rather than *run* every wave.
        const MIN_INTERVAL: std::time::Duration = std::time::Duration::from_millis(150);

        // **The interval before either counter.** This runs on every iteration of the
        // wave loop: the interval is two loads, and the gate behind it takes a read
        // lock per rung of the ladder — cheap next to a census, not cheap next to
        // nothing, and pointless on an iteration that has already decided not to run.
        //
        // Returns without standing down deliberately: the pass it is deferring is
        // still coming, so a refusal's hold on the persistence thread must survive it.
        if self
            .last_kv_compaction
            .is_some_and(|t| t.elapsed() < MIN_INTERVAL)
        {
            return 0;
        }
        if !self.kv_worth_packing() {
            return 0;
        }
        self.compact_span_tenants();
        self.pack_until_settled()
    }

    /// Whether the KV side is fragmented enough to be worth a pass: holes below the
    /// frontier, or arenas a pack would empty.
    ///
    /// **A cheap gate, because this is consulted every wave.** `kv_ground_lost` is not
    /// cheap — it sums `packed_arenas` across the ladder, which is the census — so it
    /// cannot be the thing that decides whether to census. The hole count is a handful
    /// of field reads behind one lock; the sparsity sum takes a read lock per rung of
    /// the ladder and walks that pool's arena map.
    ///
    /// Both halves of the loss, because either alone misses the other's regime. Holes —
    /// free regions stranded below the frontier — are self-correcting under allocation,
    /// so a steadily-loaded pool reads zero holes with tens of sparse arenas beneath it;
    /// gating on holes alone ran 16 passes over 110 s of churn and left the pools at
    /// 82%. Sparsity alone would miss the burst, where a mass eviction strands the
    /// frontier over ground that is genuinely free.
    ///
    /// **A closed gate stands down** the hold a previous refusal put on the persistence
    /// thread. A refused pass tells migrates to step aside for it; if the gate has since
    /// closed, that promise has to be withdrawn or hot→warm defers for a pass that is
    /// never coming.
    fn kv_worth_packing(&self) -> bool {
        /// Regions recoverable before a pass is worth its sync and its copies.
        const MIN_FREEABLE_ARENAS: usize = 8;
        let worth = self.session.kv_region_stats().is_some_and(|stats| {
            let holes = stats.live_watermark.saturating_sub(stats.live);
            holes >= MIN_FREEABLE_ARENAS
                || holes + self.session.kv_sparse_arenas() >= MIN_FREEABLE_ARENAS
        });
        if !worth {
            candle_nn::kv_cache::clear_compaction_waiting();
        }
        worth
    }

    /// Pack the span tenants that are not KV pools — the recurrent state store and the
    /// provenance gallery — in the same gap and for the same holes.
    ///
    /// **Once per wave-loop consideration, never in the follow-ups or relief.** Each is
    /// a batch of device copies of whole blocks (a recurrent state is ~3 MiB) bounded by
    /// a move count rather than the KV pass's clock, and repeating it does not converge
    /// faster. Run from every relief episode and every follow-up pass as well, it ran
    /// twice a second on Qwen3.8-Flash-Next, moving hundreds of states a time for a
    /// region or none, beside forwards already slowed by streamed experts — and the
    /// probe's efficiency gate fell from 98–99% to 76–87%.
    fn compact_span_tenants(&mut self) {
        /// Layer states one recurrent pass may move. Each is one device copy of a few
        /// MiB, so this bounds the pass at a few hundred MiB of copy — well under the
        /// KV pass's budget — while letting a badly scattered arena set converge in a
        /// handful of passes.
        const MAX_STATE_MOVES: usize = 256;

        /// Gallery pages a single pass may move.
        ///
        /// A page is 6 KiB against a recurrent state block's 3 MiB, so this is a
        /// budget in the same class of copy — ~24 MiB — while being enough moves
        /// that a badly scattered gallery converges in a handful of passes rather
        /// than hundreds. The gallery's own ceiling is 512 MiB, or ~87k pages, so
        /// this is deliberately a fraction of the worst case.
        const MAX_GALLERY_MOVES: usize = 4096;

        // **Recurrent state first.** Its arenas are regions of the span like the KV
        // side's, and on a hybrid stack they are most of what stands above the holes —
        // the KV pools can be packed to a region while the frontier stays pinned by a
        // state arena near the top. Its pass moves each state block with one device
        // copy and repoints the one store that holds it, so it needs no holder sweep
        // and no quiesce; a failure part-way leaves every store consistent (each move
        // is whole) and is reported loudly rather than retried silently.
        match self.model.compact_recurrent(MAX_STATE_MOVES) {
            Ok(r) if r.planned > 0 => {
                tracing::info!(
                    target: "candle_conversation::scheduler::vram_relief",
                    planned = r.planned,
                    moved = r.moved,
                    regions_before = r.regions_before,
                    regions_after = r.regions_after,
                    "recurrent state compaction packed the state arenas",
                );
                if r.regions_released() > 0 {
                    let _g = super::profile::span("compact:reclaim_weights");
                    self.model.reclaim_spare_ground();
                }
            }
            Ok(_) => {}
            Err(e) => tracing::error!(
                target: "candle_conversation::scheduler::vram_relief",
                "recurrent state compaction failed: {e}",
            ),
        }

        // **The gallery's pages fragment from churn, not from size.** Its runs are
        // variable length and are freed out of claim order — LRU eviction, and a
        // re-seal freeing a turn's old run mid-corpus — so a corpus that reads
        // nearly full while it is still growing ends up with live pages scattered
        // across arenas, a few of them pinning the frontier above everything below.
        // Pages a scan is reading are skipped, so this cannot move ground out from
        // under an in-flight launch.
        if let Some(arena) = self.gallery_arena.as_ref() {
            match arena.compact(MAX_GALLERY_MOVES) {
                Ok(r) if r.planned > 0 => {
                    tracing::info!(
                        target: "candle_conversation::scheduler::vram_relief",
                        planned = r.planned,
                        moved = r.moved,
                        regions_before = r.regions_before,
                        regions_after = r.regions_after,
                        "gallery compaction packed the page arenas",
                    );
                    if r.regions_released() > 0 {
                        let _g = super::profile::span("compact:reclaim_weights");
                        self.model.reclaim_spare_ground();
                    }
                }
                Ok(_) => {}
                Err(e) => tracing::error!(
                    target: "candle_conversation::scheduler::vram_relief",
                    "gallery compaction failed: {e}",
                ),
            }
        }
    }

    /// Pass after pass until one finishes inside its budget — the pass both the wave
    /// loop and relief run.
    ///
    /// **One pass is not enough after a burst.** Relief's compression rewrites a dozen
    /// or more float turns as quantized chunks in one go, and a mass retirement frees
    /// chunks across every arena, so 50–80 arenas of air arrive at once on the 30B probe
    /// and a single budgeted pass clips after 20–50 of them. The next chance to finish
    /// is the next wave-loop iteration, which spans a decode quantum and its
    /// housekeeping — 1–3 s there — and the probe measured the remainder standing that
    /// long, at 70%, while the forwards in between bought ground from the weight side
    /// against it. Back to back, the passes cost up to ~80 ms each, and only when a
    /// pass clipped; an iteration with nothing to pack still costs one gate.
    fn pack_until_settled(&mut self) -> usize {
        /// Passes one rung may run back to back.
        const MAX_PASSES: usize = 4;
        let mut released = 0;
        for _ in 0..MAX_PASSES {
            let Some(pass) = self.pack_kv_pools() else {
                break;
            };
            released += pass.released;
            if !pass.clipped {
                break;
            }
        }
        released
    }

    /// One KV compaction pass if the pools are fragmented enough to be worth it, then
    /// the weight side's reclaim — what [`Self::compact_kv_if_fragmented`] paces and
    /// relief runs directly. The span tenants are not packed here; see
    /// [`Self::compact_span_tenants`].
    ///
    /// Answers what the pass did — nothing when the pools were not worth a pass — or
    /// `None` when it was turned away by a holder of the partition or of chunk
    /// locations (a forward, a hot→warm migrate group, a cold-writer free), which is
    /// the one answer that changes by asking again shortly.
    pub(super) fn pack_kv_pools(&mut self) -> Option<PackPass> {
        /// Wall-clock a single pass may spend **planning and claiming**. A budget
        /// rather than a move cap: the ladder spans 320 B to 16 KiB, so the same
        /// "one move" is fifty times the bandwidth at one rung and a count cannot
        /// bound a duration.
        ///
        /// Generous, because the copies are one launch and the *walks* are what the
        /// clock bounds — the occupancy census over every arena of every rung, then
        /// the claims. At 8 ms the pass clipped on every run and handed back a single
        /// arena with the frontier where it started; at 20 ms the census alone spent
        /// the budget and 37 of 40 attempts claimed nothing. Both halves are now
        /// separately bounded (see `compact_backings`), and this is sized so each has
        /// room: the wave loop's own housekeeping already spends 40–66 ms on the
        /// promote pass at this point, so a pass of this order is in proportion to
        /// what the gap already costs.
        /// Measured at 40 ms: 31 of 42 passes clipped, and the pools held 94–99%
        /// except through the first mass eviction, where 24 conversations retiring at
        /// once left one 1.5 s sample at 71%. A clipped pass is not wasted — it packs a
        /// prefix and the next resumes closer — but through a burst the arrival rate is
        /// what has to be matched, and clipping every pass means it never is.
        const BUDGET: std::time::Duration = std::time::Duration::from_millis(80);

        if !self.kv_worth_packing() {
            return Some(PackPass::default());
        }

        // **Every registered conversation, not one.** Conversations in a workspace
        // share a substrate, so sweeping one sweeps all its residences — but the
        // scheduler can host conversations on more than one substrate, and a
        // workspace left out keeps residences naming vacated slots. Sweeping them all
        // is safe because a pass is idempotent under one `Sweep`: after a rewrite the
        // residence holds NEW gids and the map is keyed on the old ones, so a second
        // visit matches nothing.
        let substrates: Vec<_> = self.slot_conversations.values().cloned().collect();
        let mut swept = 0usize;
        // **The projection caches are holders too.** Taken out for the duration so
        // the closure can rewrite them while `self.session` is borrowed — two
        // disjoint fields of `self`, which the borrow checker cannot split across a
        // method call on one of them — and put back below whatever the pass returns.
        //
        // Without this they kept naming the slots a pass had vacated: a cached glue
        // island or a pending user part is Arc-injected into a slot by a later
        // projection instead of being recomputed, so the stale K/V is read as if it
        // were the span's own. That is the incomplete holder set the pass's
        // completeness check found — 129 of 6767 relocated slots named by nobody it
        // reached — and the reason a compaction could poison a sequence that was not
        // even resident when it ran.
        let mut projections = std::mem::take(&mut self.slot_projection_state);
        // The closure receives the pass's OWN `Sweep`, already carrying whatever the
        // backings rewrote. Sharing it is what makes an allocation held by both a
        // block table and a residence come back as one replacement.
        let outcome = self.session.compact_kv(BUDGET, &mut |sweep| {
            for s in &substrates {
                swept += s.rewrite_for_compaction(sweep)?;
            }
            for state in projections.values_mut() {
                swept += state.rewrite_for_compaction(sweep)?;
            }
            Ok(())
        });
        self.slot_projection_state = projections;
        // **The interval is stamped by a pass that ran, never by one that was
        // refused.** `MIN_INTERVAL` exists to bound the census, and a contended pass
        // pays no census — it refuses on a `try_` acquisition and returns. Stamping
        // ahead of the call made every refusal cost a quarter second of not trying
        // again, and against a persistence thread that holds its migrate for the
        // length of a hot→warm batch that meant 15 passes out of 52 attempts and the
        // pools left at 69%. Now the next wave picks up the gap the moment the
        // migrate lets go.
        if !matches!(
            outcome,
            Err(candle_nn::kv_cache::CompactionRefused::WaveInFlight)
                | Err(candle_nn::kv_cache::CompactionRefused::MigrateInFlight)
        ) {
            self.last_kv_compaction = Some(std::time::Instant::now());
        }
        // **The pass's own phase breakdown, filed where every other wave span lands.**
        // `candle-nn` sits below the profiler, so the pass measures itself and this
        // records it — see `CompactionReport::timings`. Recorded for a refusal too when
        // one got far enough to plan, because "the census cost 30 ms and then the claims
        // lost" is a diagnosis and "compaction was refused" is not.
        if let Ok(report) = &outcome {
            let t = report.timings;
            super::profile::record("compact:quiesce", t.quiesce);
            super::profile::record("compact:plan", t.plan);
            super::profile::record("compact:claim", t.claim);
            super::profile::record("compact:copy", t.copy);
            super::profile::record("compact:barrier", t.barrier);
            super::profile::record("compact:sweep", t.sweep);
            super::profile::record("compact:mint", t.mint);
            super::profile::record("compact:invalidate", t.invalidate);
            super::profile::record("compact:publish", t.publish);
        }
        let pass = match &outcome {
            Ok(r) => Some(PackPass {
                released: r.arenas_released,
                clipped: r.clipped,
            }),
            Err(
                candle_nn::kv_cache::CompactionRefused::WaveInFlight
                | candle_nn::kv_cache::CompactionRefused::MigrateInFlight,
            ) => None,
            Err(_) => Some(PackPass::default()),
        };
        match outcome {
            Ok(report) if !report.is_empty() => {
                // The weight side, immediately, while no wave generation is live.
                let _g = super::profile::span("compact:reclaim_weights");
                self.model.reclaim_spare_ground();
                tracing::info!(
                    target: "candle_conversation::scheduler::vram_relief",
                    moves = report.moves,
                    allocations = report.allocations_rewritten,
                    // Fresh records for the relocated chunks, one batched launch. Lower
                    // than `allocations` by however many were writer windows, which carry
                    // no record and are addressed from their gids.
                    minted = report.records_minted,
                    // Non-zero means the up-front reservation under-provisioned and those
                    // chunks' sources were not reclaimed. Correct, but degrading.
                    records_declined = report.records_declined,
                    record_arenas_reserved = report.record_arenas_reserved,
                    // Arenas created ahead of demand; the unused ones, which the pass
                    // hands back itself, are among `arenas_released`.
                    fresh_arenas = report.fresh_arenas,
                    // `KvHead` records copied lower (included in `moves`), and the
                    // copies no holder took — expected where the chunk was minted anew.
                    records_moved = report.records_moved,
                    records_unfollowed = report.records_unfollowed,
                    // The pass's downstream cost: each cleared buffer is a host
                    // re-serialisation and an upload on its slot's next sync.
                    buffers_cleared = report.decode_buffers_cleared,
                    arenas_released = report.arenas_released,
                    frontier_before = report.frontier_before,
                    frontier_after = report.frontier_after,
                    // What holds the frontier up once the pass is done — a pass that
                    // moved a lot and lowered nothing is explained here or nowhere.
                    top_arena = ?report.top_arena,
                    reclaimed_mib =
                        report.regions_reclaimed() * (candle_nn::kv_cache::REGION_BYTES >> 20),
                    substrate_sequences = swept,
                    clipped = report.clipped,
                    // **What the pass waited for, beside what it did.** The pre-plan
                    // drain is the one phase whose cost belongs to other work — it
                    // waits for whatever was in flight when the pass began — and it
                    // is outside the budget, so without it here a pass that spent
                    // 60 ms waiting and 20 ms working is indistinguishable from one
                    // that spent 80 ms working. Only visible through the profile
                    // spans otherwise, and those compile to nothing by default.
                    quiesce_us = report.timings.quiesce.as_micros(),
                    // The two halves the budget bounds: the census and plan, then the
                    // claim walk. A pass that clips after one batch spent its budget in
                    // the first.
                    plan_us = report.timings.plan.as_micros(),
                    claim_us = report.timings.claim.as_micros(),
                    // The host walk over every slot's decode-buffer pins, outside the
                    // budget like the quiesce. Named because it scales with slots ×
                    // chunks × layers, not with what the pass moved.
                    invalidate_us = report.timings.invalidate.as_micros(),
                    // **The figure that says whether the pass was complete.** A relocated
                    // slot no holder the sweep reached names: its claim is wasted, its
                    // source is not reclaimed, and because nothing names it the census
                    // will plan it again next pass. Not a correctness problem — every
                    // party naming a slot keeps it alive — but the holder list is
                    // maintained by hand and this is how a missing one shows up.
                    unwitnessed = report.unwitnessed,
                    // Read/write collisions declined before the launch. Non-zero is
                    // the guard working: a collision suffered corrupts a chunk
                    // silently, and this is the only line that says it was there.
                    source_collisions = report.source_collisions,
                    "kv compaction packed the pools and handed the ground back",
                );
            }
            Ok(_) => {}
            Err(candle_nn::kv_cache::CompactionRefused::WaveInFlight) => {
                // Ordinary contention. The next pass tries again.
            }
            // **A fault is reported, not declined.** Every other arm here is "not
            // now" and belongs at `debug`, which is why this one cannot share the
            // arm: a partition invariant that broke would have been a debug line
            // nobody reads, on a pass that runs every wave.
            //
            // Every active turn fails, not one. The pass runs in the wave loop
            // between forwards, so there is no turn whose call stack this is on —
            // and the fault is not one turn's anyway: the arenas are pooled across
            // sessions, so a plan that named ground no arena owns implicates
            // whatever any of them is holding. Failing the turns that exist is what
            // makes it visible at the only place a caller is watching.
            Err(candle_nn::kv_cache::CompactionRefused::Fault(msg)) => {
                let ids: Vec<_> = self.active_decodes.keys().copied().collect();
                tracing::error!(
                    target: "candle_conversation::scheduler::vram_relief",
                    turns = ids.len(),
                    "kv compaction found a broken partition invariant and abandoned \
                     the pass: {msg}",
                );
                self.fail_all_decodes(
                    &ids,
                    &format!("kv compaction found a broken partition invariant: {msg}"),
                );
            }
            Err(e) => {
                tracing::debug!(
                    target: "candle_conversation::scheduler::vram_relief",
                    "kv compaction declined: {e:?}",
                );
            }
        }
        pass
    }

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
    ///  4. **Pack the pools.** A KV compaction pass: arenas held by a few live
    ///     chunks are emptied into lower ones and handed back. One budgeted
    ///     device pass, and nothing resident is lost — which the two rungs after
    ///     it cannot say.
    ///  5. **Evacuate.** Flush the pending hot→warm so just-sealed turns have a
    ///     warm copy — only warm-backed turns are evictable — then drop the hot
    ///     copies of the oldest ones. This is §3.8's evict-as-evacuation, and
    ///     it runs through the demotion path the tiering already owns.
    ///  6. **Take ground from the weight side**, which costs expert residency.
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

        // **Hand back a finished forward's transient tier before recycling
        // anything.** The tier outlives the guards that used it — a forward's
        // outputs escape into its caller — so relief, which runs between
        // forwards, can find one still standing over ground its wave no longer
        // needs. Every rung below claims regions, so it goes back first.
        //
        // This is not tidiness. `region_ceiling` is `transient_base` while a tier
        // is placed: an *address*, fixed where the last forward put it. Move the
        // weight boundary and it does not follow. So **a placed tier makes the
        // ceiling deaf to the boundary** — the last rung concedes weight-side
        // ground and the rungs above it still cannot claim a region, because the
        // cap is pinned at wherever the tier was placed. That is the shape of the
        // section-prefill wedge: the weight side conceded down to its floor
        // across thousands of retries while the ceiling never moved off 293.
        //
        // Declines while a wave generation is live, which is the one case where
        // the tier is genuinely still in use.
        if let Device::Cuda(d) = self.session.device() {
            end_wave_transient(&d.cuda_stream());
        }

        let mut released = self.session.release_empty_arenas().unwrap_or(0);
        let mut gallery_freed = 0u64;
        let mut compressed = 0usize;
        let mut compress_refused = false;
        let mut flushed = false;
        let mut evicted = crate::substrate::EvictionReport { count: 0, bytes: 0 };

        // Gallery eviction — **this cannot clear the pressure below it**, and is
        // not here to.
        //
        // `evict_to_cap` drops `PageRun`s, returning page slots to the gallery's
        // own arenas. A region goes back to the span only when the last page in
        // its arena goes, so `gallery_freed` counts pages freed, and
        // `region_stats().free` moves only by the arenas that emptied — the next
        // `vram_under_pressure_for` can still be true.
        //
        // Gallery growth is bounded by the arena itself — it evicts to its own
        // ceiling at admission — so this is not the only limit, and it must not
        // fire merely because KV is tight: that would shed belief-scan residency
        // the next scan rebuilds from the substrate, every episode. It enforces
        // the same ceiling by the same rule the arena does (`evict_to_cap`): only
        // turns no scan has used recently go, so a working set above the ceiling
        // stays. A plain LRU here shed a live dialogue's working set on every
        // decode relief — 892 MiB a wave, `relieved=false` each time since
        // scattered pages return no region — and its next scan re-uploaded it,
        // a 1.6–2.0 s index rebuild per reprojection.
        if self.vram_under_pressure_for(phase) {
            if let Some(arena) = self.gallery_arena.as_ref() {
                gallery_freed = arena.evict_to_cap();
            }
        }

        if self.vram_under_pressure_for(phase) {
            // Bound the batch so a large backlog drains over several episodes
            // rather than one multi-second blocking pass over everything
            // pending; the persistence thread is working the same queue.
            let budget = want
                .saturating_mul(VRAM_COMPRESS_HYSTERESIS)
                .min(vram_compress_max());
            let pass = self.compress_pending_turns(budget);
            compressed = pass.compressed;
            compress_refused = pass.refused;
            released += self.session.release_empty_arenas().unwrap_or(0);
        }

        // **Pack the pools before taking anything from anyone.** Ground the KV side
        // holds as air — arenas kept alive by a handful of live chunks — is ground
        // a compaction hands back without evicting a turn or an expert, and both
        // rungs below cost one of those. Without this rung relief reached for them
        // with the pools a third air: measured on Qwen3-30B-A3B, the weight zone
        // conceded 193, 240 and 290 MiB in two seconds while 287 arenas held what
        // 185 would, and the efficiency gate read 51–64% through it. After
        // compression, because a compressed turn's float working set leaves exactly
        // that air behind — and **whenever compression ran, relieved or not**: the
        // air it leaves is the weight side's ground either way, and left for the
        // wave loop's next pass it stood for seconds (a 30B probe sample at 58%, 244
        // arenas holding what 142 would, with nothing retired).
        //
        // Unpaced: the interval bounds the census on iterations with nothing at stake,
        // and here the alternative is an eviction.
        let mut packed = 0usize;
        if compressed > 0 || self.vram_under_pressure_for(phase) {
            packed = self.pack_until_settled();
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

        // **Pack again before the weight side pays.** The evictions just above left air
        // of their own, and a pass the first rung could not get in for has had the
        // flush's wait as well. Measured on the 30B probe before this rung: 1,513 ms of
        // relief that flushed, evicted two turns and then conceded 160 MiB of expert
        // residency with `arenas_packed=0`. Whenever eviction ran, for the reason the
        // first pack runs whenever compression did.
        if evicted.count > 0 || self.vram_under_pressure_for(phase) {
            packed += self.pack_until_settled();
        }

        // **Last resort, and the only one that adds ground rather than
        // recycling it.** Everything above reclaims KV the engine already owns —
        // compress a turn, evict a cold tail, drop an empty arena — and all of
        // it is worth nothing against a workload with nothing reclaimable. A
        // base conversation's sections are permanent by design: they are not
        // turns, so there is no turn to compress and no tail to evict, and a
        // section prefill that outgrows its ground stalls with every relief
        // counter reading zero. That is exactly how it failed.
        //
        // The weight side is holding ground in that case, and the boundary is
        // meant to move. It could not: the give-back runs at the end of a
        // completed forward, and the wave that needs it never completes. Asking
        // here breaks that circle — this is between waves, which is where the
        // move is safe, and a refusal (a wave still open, or the zone already at
        // its floor) comes back as zero rather than as a wait.
        //
        // **`want` is the ask.** It is the shortfall this pass measured against
        // the setpoint, and passing it is the whole of the fix for the run that
        // died here: the boundary used to read an accumulated count of refused
        // claims instead, which said 4,436 regions on a pass whose own `want_mib`
        // was 448 — 28 regions. It conceded 5,752 MiB, evicted 1,598 experts, and
        // put the zone under its pinned working set, after which nothing ran.
        // The number was in this function the whole time; it just was not sent.
        let mut conceded = 0u64;
        if self.vram_under_pressure_for(phase) {
            conceded = self
                .model
                .request_kv_ground(want.div_ceil(region_bytes()) as usize);
        }

        let still = self.vram_under_pressure_for(phase);
        let acted = released > 0
            || gallery_freed > 0
            || compressed > 0
            || packed > 0
            || evicted.count > 0
            || conceded > 0;
        if acted {
            relief_trace::note("sched", "relieve", want, evicted.bytes);
        }
        let (free, setpoint) = self.kv_region_state(phase).unwrap_or((0, 0));
        // INFO when the pass actually shed something — that is a real event.
        // TRACE otherwise: this runs from several gates every scheduler loop,
        // so a no-op pass at any lower level floods the log under a sustained
        // burst.
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
                    compress_refused,
                    arenas_packed = packed,
                    turns_evicted = evicted.count,
                    evicted_mib = evicted.bytes / (1 << 20),
                    arenas_released = released,
                    conceded_mib = conceded / (1 << 20),
                    free_regions = free,
                    setpoint_regions = setpoint,
                    relieved = !still,
                    "KV region relief"
                )
            };
        }
        // A refused compression is not an action, but it *is* an event: the rung
        // that shrinks a resident turn in place was asked to run and could not
        // get the ground to run in. Left at DEBUG it reads as `turns_compressed=0`,
        // identical to a pass with nothing to compress — which is how the
        // feedback loop running backwards (compression is what relieves the
        // pressure that refuses it) stayed invisible through the whole wedge.
        if acted || compress_refused {
            emit!(info);
        } else {
            emit!(trace);
        }
        // **Relief that shed something is an admission opportunity too.** Unlike
        // a completion it can happen with nothing finishing at all, and a pass
        // gated only on completions would leave that ground unoffered until the
        // next slot ended — which on a stalled engine is never.
        if acted {
            self.settled_since_admit = true;
        }
        still
    }

    /// Rows the next wave already carries before admission adds anything: the
    /// held creep group, which rides that wave whole.
    ///
    /// **Charged to the wave before any offer is judged**, for two reasons. The
    /// offers behind it are then priced against the forward that will actually
    /// run rather than an empty one — a cohort already in flight is most of the
    /// copy those offers would otherwise look like they were amortising alone.
    /// And it denies the head waiver: the model carries its first offer
    /// unconditionally so a slot too large to ever fit cannot block the queue
    /// forever, and a wave already carrying a cohort is not that case.
    pub(super) fn standing_rows(&self) -> usize {
        self.wave_prefill_members
            .iter()
            .map(|m| match m {
                WaveMember::Prefill { advance, .. } | WaveMember::Section { advance, .. } => {
                    *advance
                }
            })
            .sum()
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
    ///
    /// # The seal's second copy is *not* charged here, and that was tried
    ///
    /// `docs/archived/elastic_vram_partition.md` §7 phase 1 asks admit to account for
    /// "persistence's quantize destinations", and the obvious reading — a block
    /// occupies its active slot *and* its sealed destination while the
    /// compressor copies between them, so charge both — was built and reverted.
    ///
    /// It is the wrong shape. The overlap lasts one copy; the charge lasts the
    /// block's whole life. Applying it here doubles the price of **every** block
    /// in every admission decision, in-flight accounting and decode reserve, so
    /// admission clears roughly half the work it should. On a live rebuild that
    /// showed as `(no forwards)` against a `64MiB` budget with a 14k-token
    /// backlog: nothing admitted, so nothing completed, so nothing freed, so the
    /// budget never recovered.
    ///
    /// A transient double-occupancy is a *reserve* — a fixed pool the compressor
    /// draws on — not a per-block tariff. That is what §7 means and it is still
    /// unbuilt.
    pub(super) fn per_block_kv_bytes(&self) -> u64 {
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
        // No forward reserve is subtracted here: admission holds the tier back
        // itself, at the width it is choosing (`Scheduler::admit_headroom`). The
        // setpoint IS subtracted — those regions are the relief pass's working
        // room, not admission's to spend.
        let Some((free, setpoint)) = self.kv_region_state(VramPhase::Load) else {
            return 0;
        };
        (free.saturating_sub(setpoint) as u64).saturating_mul(region_bytes())
    }

    /// Admit queued prefills against what they do to the wave's throughput.
    ///
    /// A burst of small parallel scopes (code_read's worker count), a bulk
    /// collection ingest's per-section prefills, or a batch of calibration cases
    /// all arrive here. What coalesces into one ragged forward is whatever makes
    /// the wave *faster* carrying it: offers are taken in priority order
    /// ([`admit::order`]), priced against the state the eviction pass just
    /// settled ([`admit::cost`]), and admitted while the rate improves
    /// ([`admit::rate`]) without crossing the residency the engine defends
    /// ([`admit::gate`]).
    ///
    /// [`Scheduler::MAX_PREFILL_WIDTH`] is a backstop above this, not the
    /// control. The keep-one-alive rule below is what guarantees progress: a
    /// pass that admits nothing with nothing in flight takes the queue head
    /// regardless, so an oversized lone turn still runs and is bounded by the
    /// per-arena VRAM gate rather than by a refusal it can never clear.
    pub(super) fn promote_new_prefills(&mut self) {
        if self.prefill_queue.is_empty() {
            return;
        }
        // Every gate below reads the priority pause; judge against the present.
        // `in_flight` counts only prefills that can advance — see
        // `running_prefills`.
        self.observe_priorities();
        let in_flight = self.running_prefills();
        if in_flight >= Self::MAX_PREFILL_WIDTH {
            return;
        }

        // VRAM-pressure backpressure. Each admitted prefill pins its
        // conversation's KV in VRAM, so under pressure we shed hot KV to the
        // substrate rather than piling on more concurrent prefills; if that
        // doesn't clear it, leave the rest queued this pass.
        if in_flight > 0
            && self.vram_under_pressure()
            && self.relieve_vram_pressure("promote", VramPhase::Load)
        {
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

        // **The planner is consulted once per settling, not once per pass.**
        //
        // [`admit::fill`] refuses an unsettled ground itself (`Ground::settled`),
        // but the machinery that *reaches* that refusal is not free:
        // [`Self::take_planner`] arms the planner, and `AdmitPass::new` prices the
        // standing tier, reads the region census and computes the headroom and the
        // budget — every bit of it discarded the moment `fill` answers `skipped`.
        // This pass runs from the top of the wave loop AND from every decode step
        // (`Scheduler::mid_wave_admission`), so on an unsettled engine that setup,
        // not the fill, is the whole cost.
        //
        // Nothing new can fit that did not fit before unless something freed
        // ground, so the gate belongs in front of the pricing rather than behind
        // it. What opens the next pass is a sequence finishing — a decode
        // completing, a prefill promoting to decode, a section sealing, a slot
        // terminally freed — plus the two events that can offer ground with
        // nothing finishing at all: a turn arriving at an idle engine, which has
        // never been offered and would otherwise wait for a completion that is
        // never coming, and a relief pass that actually shed.
        if !self.settled_since_admit {
            // Skipping the pass must not skip the deadlock-freedom rule.
            if in_flight == 0 {
                self.force_queue_head("admission is closed until something settles");
            }
            return;
        }

        // **The decision is a rate, not a fit.**
        //
        // Admission used to ask whether an offer's bytes fitted the ground
        // standing free above the weight floor. That question has no answer in
        // tokens a second: a wave of 250 rows and a wave of 2,000 pay the *same*
        // expert copy — a prefill forward needs every expert, resident or
        // streamed — so the narrow one is not cheaper, it is slower per row. A
        // gate that admits by fit stops widening the moment the bytes run out,
        // which on this card measured ~470 tok/s against a modelled 1,210 at the
        // same residency.
        //
        // The bytes have not gone away: they are what the offer *costs*, read by
        // the model as the residency it dislodges, which is what makes a wide
        // wave stop being worth it. They are one term in a throughput comparison
        // now rather than the whole question.
        let Some(mut rate) = self.take_planner() else {
            // No weight side to trade rows against — a dense stack, or an expert
            // cache that has not published its gauges yet (which is every engine
            // before its first classify). There is no rate question to ask, so
            // the queue head goes in and the width backstop is the only bound.
            // Without this an engine would never take the first prefill that
            // makes the cache describe itself.
            //
            // **And it says so, because this branch admits exactly one.** It used to
            // return in silence, which makes it indistinguishable from a healthy engine
            // in every log the scheduler emits: no admission pass line, no refusal, just
            // a wave one row wide, forever. Measured on the 30B-A3B — twenty conversations
            // queued, `decode seqs avg=1.0 max=1`, 32 t/s against a batched ceiling of
            // 518 — and the only way to find it was to notice that the pass which should
            // have logged never did. A path that quietly costs an order of magnitude is
            // the one path that must not be quiet.
            if let Some(work) = self.pop_unpaused_head() {
                tracing::debug!(
                    target: "candle_conversation::scheduler::throttle",
                    queued = self.prefill_queue.len(),
                    in_flight,
                    "no weight plan: admitting ONE prefill unjudged. The rate model cannot \
                     run without the expert cache's gauges, so the wave stays as narrow as \
                     this path makes it",
                );
                self.begin_prefill(work);
            }
            return;
        };
        let mut pass = AdmitPass::new(self);
        let filled = admit::fill(&mut pass, &mut rate);
        self.wave_rate = Some(rate);
        // **Keep one prefill in flight, whatever the model says.**
        //
        // The engine's deadlock-freedom rule, and the one guarantee no
        // throughput judgement may override: an engine that admits nothing makes
        // no progress, and nothing completes, so nothing frees the ground the
        // refusal was about. The wave then refuses the same turn on the same
        // grounds forever.
        //
        // `fill` has its own version of this — it carries its first offer
        // unconditionally — but that waiver is keyed on the *wave* being empty,
        // and a wave is not empty while a decode is still stepping. So a turn
        // arriving behind a decode is judged on its merits, which is right, and
        // if it is refused while nothing is prefilling there is nothing left to
        // create the conditions for it to be admitted later.
        //
        // This is the rule the byte-fit planner carried as `MIN_PREFILL_WIDTH`,
        // restored after its absence wedged a live daemon: two turns queued,
        // nothing in flight, `stopped_on_weights` on every pass, and the loop
        // spinning in relief that could not help because the ground it wanted
        // was not the ground being refused.
        //
        // FIFO, not the cheapest that fits: under a budget stuck at its floor,
        // cheapest-first starves the expensive work permanently, and the
        // expensive work is never the cheapest.
        if filled.prefills == 0 && self.running_prefills() == 0 {
            self.force_queue_head("admission refused every offer");
        }
        // The pass acted on the completions that opened it; the next one waits
        // for its own.
        self.settled_since_admit = false;
        if filled.prefills > 0
            || filled.stopped_on_weights
            || filled.stopped_on_rate
            || !self.prefill_queue.is_empty()
        {
            tracing::debug!(
                target: "candle_conversation::scheduler::throttle",
                admitted = filled.prefills,
                in_flight,
                queued = self.prefill_queue.len(),
                // **The one field that distinguishes "refused" from "never
                // asked".** A pass that skips admits nothing and refuses
                // nothing, which reads identically to a pass that judged every
                // offer and turned it down — and the two call for opposite
                // fixes.
                skipped = filled.skipped,
                // Which of the two ends a pass stopped at is the signal worth
                // having: weights means the device is the bound and the producer
                // should hear about it, rate means the engine is working well and
                // the queue simply rides the next wave.
                stopped_on_weights = filled.stopped_on_weights,
                stopped_on_rate = filled.stopped_on_rate,
                "admission pass"
            );
        }
    }

    /// Admit the queue head whatever the rate model would have said — the
    /// engine's deadlock-freedom rule, and the one guarantee no throughput
    /// judgement may override.
    ///
    /// An engine that admits nothing makes no progress, so nothing completes, so
    /// nothing frees the ground the refusal was about, and the same turn is
    /// refused on the same grounds forever. This is the rule the byte-fit planner
    /// carried as `MIN_PREFILL_WIDTH`, restored after its absence wedged a live
    /// daemon: two turns queued, nothing in flight, `stopped_on_weights` on every
    /// pass, and the loop spinning in relief that could not help because the
    /// ground it wanted was not the ground being refused.
    ///
    /// FIFO, not the cheapest that fits: under a budget stuck at its floor,
    /// cheapest-first starves the expensive work permanently, and the expensive
    /// work is never the cheapest.
    ///
    /// Callers apply it only with no prefill that can advance
    /// ([`Self::running_prefills`] zero). `why` names the path
    /// that forced it, because a head admitted this way was never judged and a
    /// reader has to be able to tell that from an admission that was.
    ///
    /// The head is the first entry **not paused** by priority
    /// ([`Self::priority_paused`]): a dialogue turn queued behind ingest is the
    /// one forced in, and paused ingest is never forced past a pause the
    /// running conversation holds. Nothing paused can deadlock the engine — the
    /// work pausing it is running.
    fn force_queue_head(&mut self, why: &'static str) {
        if let Some(work) = self.pop_unpaused_head() {
            tracing::debug!(
                target: "candle_conversation::scheduler::throttle",
                queued = self.prefill_queue.len() + 1,
                why,
                "nothing in flight; forcing the queue head so the engine makes progress",
            );
            self.begin_prefill(work);
        }
    }

    /// Remove and return the first queued prefill that is not paused by
    /// priority, in queue order.
    fn pop_unpaused_head(&mut self) -> Option<PrefillWork> {
        self.observe_priorities();
        let at = (0..self.prefill_queue.len())
            .find(|&i| !self.priority_paused(self.prefill_queue[i].sequence_id))?;
        self.prefill_queue.remove(at)
    }

    /// The planner, built on first use and taken for the duration of a pass.
    ///
    /// Taken rather than borrowed because the pass borrows the whole scheduler:
    /// [`admit::fill`] needs `&mut` on both the ground and the planner, and they
    /// live in the same struct. It goes back the moment the pass ends.
    ///
    /// `None` while the model cannot describe its weight side — see
    /// [`ManagedBatchedModel::weight_plan`].
    fn take_planner(&mut self) -> Option<admit::WaveRate> {
        if self.wave_rate.is_none() {
            // **The refusal names itself.** `WeightPlan::from_stats` declines a partial
            // gauge set outright — any of `moe_layers`, `total_experts`,
            // `expert_slot_bytes` or `zone_max_bytes` reading zero — and a bare `?` here
            // turns that into an absent planner with no record of which gauge was missing.
            // Since an absent planner collapses every wave to one row (see the caller),
            // the distinction between "no expert cache at all" and "the cache has not
            // described itself yet" is the difference between a dense model working as
            // designed and a routed model silently running an order of magnitude slow.
            // **A routed model with broken gauges is fatal, and a dense one is not.**
            //
            // These were one `None` until the conflation was measured: an absent plan
            // collapses admission to one prefill per pass (see the caller), which is
            // correct for a dense stack and an order of magnitude for a routed one. On the
            // 30B-A3B at 72 GiB it read `decode seqs avg=1.0 max=1` and 32 t/s against a
            // batched ceiling of 518, logged nothing, and inverted with card size — a card
            // too small to hold the checkpoint streamed experts, published gauges and
            // batched, while a card large enough held everything, published nothing, and
            // ran one row wide.
            //
            // So the broken case panics. It is not recoverable by degrading: the engine
            // would serve, slowly, with no signal distinguishable from a healthy narrow
            // workload, and the only way it was ever found was noticing that a log line
            // which should have appeared never did.
            let p = match self.model.weight_plan() {
                candle_transformers::models::expert_lre::WeightPlanning::Ready(p) => p,
                candle_transformers::models::expert_lre::WeightPlanning::Dense => {
                    tracing::debug!(
                        target: "candle_conversation::scheduler::throttle",
                        "dense stack: no expert residency to trade rows against, so the \
                         width backstop is the only bound",
                    );
                    return None;
                }
                candle_transformers::models::expert_lre::WeightPlanning::Incomplete { field } => {
                    let s = self.model.expert_stats();
                    panic!(
                        "the expert cache reports a routed model but its gauge set is \
                         incomplete: `{field}` is zero. The wave rate planner cannot be \
                         armed without it, and an unarmed planner silently narrows every \
                         wave to one row — so this refuses to run rather than serve at a \
                         fraction of the rate with no way to tell. Gauges: \
                         moe_layers={:?} total_experts={:?} slot_bytes={:?} \
                         zone_bytes={:?} zone_max_bytes={:?}",
                        s.as_ref().map(|s| s.moe_layers),
                        s.as_ref().map(|s| s.total_experts),
                        s.as_ref().map(|s| s.expert_slot_bytes),
                        s.as_ref().map(|s| s.zone_bytes),
                        s.as_ref().map(|s| s.zone_max_bytes),
                    );
                }
            };
            let geometry = admit::rate::ExpertGeometry {
                moe_layers: p.moe_layers,
                experts_per_layer: p.experts_per_layer,
                slot_bytes: p.slot_bytes,
            };
            // Measured on the device the experts actually stream over, so the
            // 4090 Mobile, the 3090 behind PCIe 3.0 and the Blackwell each seed
            // from their own bus rather than from a datasheet.
            let rate = match admit::WaveRate::measure(
                &self.device,
                geometry,
                admit::rate::RateModel::default(),
                admit::rate::DecodeModel::default(),
            ) {
                Ok(r) => r,
                Err(e) => {
                    tracing::warn!("wave rate planner: link probe failed: {e}");
                    return None;
                }
            };
            tracing::info!(
                target: "candle_conversation::scheduler::admission",
                moe_layers = p.moe_layers,
                experts_per_layer = p.experts_per_layer,
                slot_mib = p.slot_bytes >> 20,
                link_gbps = rate.link_bytes_per_s() / 1e9,
                "wave rate planner armed",
            );
            self.wave_rate = Some(rate);
        }
        self.wave_rate.take()
    }

    /// Put one admitted turn into flight.
    ///
    /// The tail of admission, and the whole of what admitting a prefill *does*:
    /// announce it to its caller and move it onto `active_prefills`. Separate
    /// from the pass that chooses it so the choosing can be replaced without
    /// touching what the choice means.
    pub(super) fn begin_prefill(&mut self, work: PrefillWork) {
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
        // No index cut here. Admission is not a unit boundary — it is the moment
        // work leaves the queue, which happens once per unit but says nothing
        // about where that unit's tokens start. The boundary was taken with the
        // unit's K/V anchor (`Scheduler::close_unit_boundary`), and a second cut
        // on this slot would close whatever the unit has already forwarded into
        // a page of its own.
        self.active_prefills.push(ActivePrefill {
            work,
            offset: 0,
            next_projection: 0,
            final_logits: None,
            error,
            prefill_start: None,
        });
    }

    /// Free KV regions right now, and the setpoint for `phase` — the two
    /// numbers every pressure and admission decision is made from.
    ///
    /// "Free" includes regions a standing transient tier has blocked
    /// (`stats.blocked`): every decision made from this pair concerns work
    /// scheduled for a *later* forward, and that forward's phase 0 releases the
    /// tier before any of its claims run. Counting only the tier-capped free
    /// count made every wave's own scratch read as KV pressure from the
    /// scheduler's seat, shedding sequences to relieve ground that was never
    /// occupied.
    ///
    /// `None` before the reservation exists, which the callers read as "no
    /// pressure, nothing to spend": there is no KV on the device yet to be
    /// under pressure about.
    pub(super) fn kv_region_state(&self, phase: VramPhase) -> Option<(usize, usize)> {
        let stats = self.kv_regions()?;
        Some((
            stats.free + stats.blocked,
            setpoint_regions(phase, stats.total),
        ))
    }

    /// The KV side's region counters, or `None` before the reservation exists.
    fn kv_regions(&self) -> Option<candle_nn::kv_cache::RegionStats> {
        let candle::DeviceLocation::Cuda { gpu_id } = self.device.location() else {
            return None;
        };
        candle_nn::kv_cache::region_stats(gpu_id)
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

    /// **Ingest is not paced. The engine's own admission governs it.**
    ///
    /// This used to run an AIMD controller against the hot→warm backlog: cut the
    /// admission budget when the drain fell behind the seal rate, reopen a quantum
    /// when it caught up. That is a throttle on the *ingest*, and it was the wrong
    /// place to put one. Whether a prefill belongs in the next wave is a question
    /// the rate model already answers per offer, in the currency that decides it —
    /// expert bytes over the bus against rows gained (`admit::rate`). The backlog
    /// controller sat in front of that, cutting the width the model was about to
    /// judge, on a signal (drain lag) that says nothing about whether the wave gets
    /// faster. Measured effect: prefill spent long stretches running
    /// single-sequence mini-forwards at a fraction of batched throughput while the
    /// model would have admitted more.
    ///
    /// What remains is the one condition that is **not** a pacing decision: the
    /// warm KV tier outgrowing its host-RAM budget. That is a host-OOM guard — it
    /// has already aborted one full overnight load — and it is a hard cut, not a
    /// controller. Slowing admission genuinely relieves it (less sealing → less
    /// hot→warm output), and nothing else in the engine defends host memory.
    ///
    /// The reopen stays evidence-based and is no longer gated on the backlog:
    /// forwards completing out-of-memory-free are proof the current width is
    /// sustainable, and that is the only proof this controller needs. A device OOM
    /// or an eviction survival still cuts instantly through
    /// [`Self::cut_admit_budget_leveled`], which resets the streak.
    pub(super) fn regulate_ingest_admission(&mut self) {
        if self.ingest_timelines.is_empty() {
            return;
        }
        if self.warm_over_budget() {
            self.cut_admit_budget_leveled(ThrottleReason::WarmOverBudget);
            return;
        }
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
        let (grow, streak) = evidence_admit_grow(
            self.admit_budget,
            ceiling,
            progressed,
            self.admit_grow_streak,
            // Cost scales with the budget already held, so the climb slows as it
            // nears the budget that last collapsed instead of charging it at
            // constant speed.
            evidence_ticks_for(budget_notches(self.admit_budget, admit_quantum())),
        );
        self.admit_grow_streak = streak;
        if grow {
            self.raise_admit_budget(ThrottleReason::Throughput);
        }
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
    /// warm backlog doesn't re-walk finished turns.
    ///
    /// [`sealed_has_compressible_chunk`]: candle_nn::kv_cache::ChunkedKvBacking::sealed_has_compressible_chunk
    /// Bring forward the quantization of up to `budget_bytes` of completed float
    /// turns (estimated by their float footprint), oldest-conversation-first.
    /// **Bounded** so a large accumulated backlog is drained over several relief
    /// episodes — a few seconds each — rather than one multi-second blocking
    /// compression of *everything* pending (a 697-turn / 23 GiB / 66 s stall was
    /// the symptom). The background persistence thread drains the rest.
    fn compress_pending_turns(&mut self, budget_bytes: u64) -> CompressPass {
        // Need an engine-wide turn policy to compress against; without one turns
        // stay native float (lossless capture) and there is nothing to bring
        // forward.
        let base = match self.session.compression_policy() {
            Some(p) => p,
            None => return CompressPass::default(),
        };
        let n_layers = self.session.num_layers();
        let device = self.session.device().clone();
        let copy_stream = match &device {
            Device::Cuda(d) => d.cuda_stream(),
            _ => return CompressPass::default(),
        };
        // Bound `backings`' immutable borrow of `self.session` to a disjoint
        // field from `self.elevate_pinned_scratch` (the `&mut` below), exactly
        // like `quantize_section_batch`.
        let backings = self.session.backings();

        let convs: Vec<Conversation> = self.slot_conversations.values().cloned().collect();
        let mut compressed = 0usize;
        let mut refused = false;
        // Estimated float bytes queued for compression so far — the bound.
        let mut collected: u64 = 0;
        'convs: for conv in convs {
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
                        Err(e) if is_device_oom(&e) => {
                            // **The pool refused a quantize destination.** This
                            // rung cannot fix that: the ground it needs comes
                            // from the rungs below (evict a cold tail) or from
                            // the boundary (`request_kv_ground`), and both of
                            // them run after this returns. Every remaining group
                            // would be refused for the same reason, so stop the
                            // pass rather than burn a kernel launch per group
                            // rediscovering it.
                            //
                            // Reported, not retried and not waited on. Waiting
                            // here would deadlock: this runs on the scheduler
                            // thread, and the scheduler thread is what would
                            // release the ground — both the rung below and the
                            // next wave's `end_wave_transient` are further down
                            // this same call stack's future.
                            tracing::debug!(
                                "compress_pending_turns: layer {layer} was refused a quantize \
                                 destination, stopping the pass: {e}"
                            );
                            refused = true;
                            ok.fill(false);
                            break;
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
                let sync_failed = if let Err(e) = device.synchronize() {
                    tracing::warn!(
                        "compress_pending_turns: device sync failed: {e:?} — skipping this group's installs"
                    );
                    true
                } else {
                    false
                };
                // Leaving **after** the sync, not at the refusal. The layers that
                // quantized before it left kernels in flight writing into
                // `q_per`'s destination arenas, and dropping those handles
                // returns their regions to the pool — so an early exit would
                // hand a region back while a kernel was still writing into it.
                // `ok` is all-false for this group, so the swap below is a no-op
                // for it either way.
                //
                // **Before the sync's own bail-out**, because that one only skips
                // a group: leaving the refusal check behind it means a failed sync
                // resumes the pass and launches quantizes for every remaining
                // group, each of which the pool refuses for the same reason the
                // first one was refused.
                if refused {
                    break 'convs;
                }
                if sync_failed {
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
        CompressPass {
            compressed,
            refused,
        }
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
    /// targets ([`Self::prefill_pass_budget`]). Without this the whole active set
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
        let cap = self.prefill_pass_budget();
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
            // A sealed section hands its scratch slot back — a completion like
            // any other, and an admission opportunity.
            self.settled_since_admit = true;
            if let Some(e) = s.error {
                let _ = s.response_tx.send(Err(e));
                continue;
            }
            let result = self.finalize_section_ingest(
                s.sequence_id,
                s.section_id,
                s.seal_block_from,
                Arc::new(s.tokens.to_vec()),
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

    /// Form a FRESH wave group into `wave_prefill_members`: ready dialogue
    /// prefills in queue order up to [`Self::prefill_pass_budget`], plus — when
    /// `include_sections` and at least one prefill is present — section chunks in
    /// whatever of that budget the prefills left. Section chunks join
    /// only alongside a cohort (so they co-batch a creep that is happening anyway);
    /// with no cohort the caller uses the faster full-sweep section path instead.
    /// Members are ordered prefills-then-sections and this order is then fixed for
    /// the group's life (the held residual depends on a stable input order).
    fn form_wave_group(&mut self, include_sections: bool) {
        // ── Dialogue prefills creep in bounded chunks ────────────────────────
        //
        // Each takes at most `max_prefill_pass_tokens` rows, packed under that
        // same per-forward budget (at least one always admitted), exactly like
        // the section chunks below; its offset advances at the head and the rest
        // rides the next group. A prefill used to enter whole: a 15.2k-token
        // Cline turn asked this wave for a 13.2 GiB transient tier — more than
        // all the ground below the weight floor on a 16 GB card — so it could
        // never be placed and failed identically on every retry, after first
        // stripping the expert cache to its floor trying.
        let cap = self.max_prefill_pass_tokens;
        // ── A prompt past the model's RoPE reach fails alone ─────────────────
        //
        // Its headers would be refused, and the refusal would fail every other
        // member of the forward with it. Refused here instead, before it joins.
        let reach = self.session.rope_reach();
        for p in self.active_prefills.iter_mut() {
            if p.error.is_some() || p.final_logits.is_some() {
                continue;
            }
            let at = self
                .session
                .sequence_offset(p.work.sequence_id.0)
                .unwrap_or(0);
            let end = at + p.work.tokens.len().saturating_sub(p.offset);
            if end > reach {
                p.error = Some(ConversationError::Other(format!(
                    "this turn's prompt reaches position {end}, past the {reach} the model's \
                     RoPE schedule supports"
                )));
            }
        }
        let mut members: Vec<WaveMember> = Vec::new();
        let mut prefill_tokens = 0usize;
        // A prefill paused behind higher-priority work sits this group out and
        // keeps its place — see `priority_pause`.
        self.observe_priorities();
        for p in &self.active_prefills {
            if p.error.is_some() || p.final_logits.is_some() || p.offset >= p.work.tokens.len() {
                continue;
            }
            if self.priority_paused(p.work.sequence_id) {
                continue;
            }
            let advance = (p.work.tokens.len() - p.offset).min(cap);
            if prefill_tokens > 0 && prefill_tokens + advance > cap {
                break;
            }
            members.push(WaveMember::Prefill {
                seq_id: p.work.sequence_id.0,
                advance,
            });
            prefill_tokens += advance;
        }
        // ── One adapter per wave ─────────────────────────────────────────────
        //
        // Same rule the decode cohort follows, applied where the prefill group
        // is formed: the projections run once over every row, so a group carries
        // one adapter or none. The first ready prefill sets it and the rest wait
        // for a group of their own — they are still active, so nothing is lost,
        // and successive groups drain each adapter's queue in turn.
        //
        // Sections are filtered by the same key below rather than after the
        // fact: a section chunk is an ordinary row of this forward, and an
        // unadapted ingest riding an adapted group would be prefilled through
        // the wrong projections and its KV written that way permanently.
        let group_adapter = members
            .first()
            .map(|m| self.session.sequence_adapter(m.seq_id()))
            .unwrap_or(None)
            .map(|s| s.to_owned());
        members.retain(|m| self.session.sequence_adapter(m.seq_id()) == group_adapter.as_deref());

        // ── Bounded by what one forward can carry ────────────────────────────
        //
        // A dialogue prefill rides the group whole — the wave takes a member's full
        // token set — so the group's rows are the sum of its members' turns, and the
        // forward prices its transient tier from that sum. Unbounded, a burst of
        // queued turns became one forward: npcd's world ingest on the routed
        // Qwen3.6-35B-A3B asked for a 3.3 GB tier against a 3.0 GB gap between the
        // KV frontier and the weight floor, and every turn in the wave failed with
        // it. Admitted in queue order up to the pass budget; the rest stay active
        // and form the next group.
        let budget = self.prefill_pass_budget();
        let lens: Vec<usize> = members
            .iter()
            .map(|m| {
                self.active_prefills
                    .iter()
                    .find(|p| p.work.sequence_id.0 == m.seq_id())
                    .map_or(0, |p| p.work.tokens.len())
            })
            .collect();
        let admitted = super::admission::admit_within(lens.iter().copied(), budget);
        members.truncate(admitted);
        let mut used: usize = lens[..admitted].iter().sum();

        if include_sections && !members.is_empty() {
            for i in 0..self.active_section_ingests.len() {
                let s = &self.active_section_ingests[i];
                if s.error.is_some() || s.offset >= s.tokens.len() {
                    continue;
                }
                if self.session.sequence_adapter(s.sequence_id.0) != group_adapter.as_deref() {
                    continue;
                }
                let advance = (s.tokens.len() - s.offset).min(budget);
                // Sections share the forward, so they draw on the budget the
                // prefills above have already spent and fill only what is left.
                // The cohort already guarantees this group makes progress, so no
                // section is forced in past it; the rest ride a later group or the
                // standalone section pass.
                if used + advance > budget {
                    break;
                }
                members.push(WaveMember::Section {
                    seq_id: s.sequence_id.0,
                    advance,
                });
                used += advance;
            }
        }
        self.wave_prefill_members = members;
    }

    /// Resume the held wave group: rebuild each member's `(seq_id, input tensor)`
    /// from its live backing — both kinds feed their stable
    /// `[offset, offset+advance)` chunk — dropping members that errored/completed.
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
                WaveMember::Prefill { seq_id, advance } => {
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
                    let off = self.active_prefills[i].offset;
                    let end = (off + advance).min(self.active_prefills[i].work.tokens.len());
                    let toks: Vec<u32> = self.active_prefills[i].work.tokens[off..end].to_vec();
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
    /// are aligned in caller order. Prefill members commit their chunk and emit a
    /// progress event; the FINAL chunk also emits the staged events and records
    /// `final_logits` for promotion to decode, while an earlier one leaves the
    /// prefill active for the next group. Section members advance their chunk +
    /// record slot tokens (sealed later by `finalize_done_section_ingests`).
    /// Clears the group.
    fn complete_wave_group(&mut self, members: &[WaveMember], member_logits: &[Tensor]) {
        for (k, m) in members.iter().enumerate() {
            match *m {
                WaveMember::Prefill {
                    seq_id: sid,
                    advance,
                } => {
                    let Some(i) = self
                        .active_prefills
                        .iter()
                        .position(|p| p.work.sequence_id.0 == sid)
                    else {
                        continue;
                    };
                    let total = self.active_prefills[i].work.tokens.len();
                    let seq_id = self.active_prefills[i].work.sequence_id;
                    let off = self.active_prefills[i].offset;
                    let end = (off + advance).min(total);
                    if let Err(e) = self.session.advance_sequence(seq_id.0, end - off) {
                        self.active_prefills[i].error = Some(ConversationError::Model(e));
                        continue;
                    }
                    let chunk_tokens: Vec<u32> =
                        self.active_prefills[i].work.tokens[off..end].to_vec();
                    super::Scheduler::record_slot_tokens(
                        &mut self.slot_tokens,
                        seq_id,
                        &chunk_tokens,
                    );
                    self.active_prefills[i].offset = end;
                    if end < total {
                        // More to prefill: report progress and ride the next group.
                        let _ = self.active_prefills[i].work.event_tx.send(
                            TurnEvent::PrefillProgress {
                                tokens_done: end,
                                tokens_total: total,
                            },
                        );
                        continue;
                    }
                    // Staged calibration prefill: emit every segment's pinned
                    // projection here, once the last chunk has landed.
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
                        //
                        // `to_owned_tensor`, not `copy`: the row is a view of the
                        // wave's whole logits block, and `copy` clones the
                        // storage it views — every row of the block — where this
                        // copies the row alone.
                        let owned = l.to_owned_tensor().unwrap_or_else(|_| l.clone());
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
                    WaveMember::Prefill { seq_id, .. } => {
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
                    // Dropped unfilled: the slot's assembly recorded these glue
                    // pieces as placed, and no later rebuild may keep them.
                    if let Some(state) = self.slot_projection_state.get_mut(&p.parent_id) {
                        state.placed_pieces.clear();
                    }
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
        verify_seqs: &[usize],
        verify_inputs: &[Tensor],
    ) -> candle::Result<Vec<Tensor>> {
        // Fold this wave's deferred glue in as a full-sweep member co-batched with
        // decode (see `take_wave_glue`). A slot that decodes this wave is never
        // also a glue member — `take_active_decode_batch` excludes slots with a
        // pending deferred glue fire (they reproject this wave and resume decode
        // next), so the two groups are disjoint and the assembled context list
        // never lists a slot twice.
        let glue = self.take_wave_glue();
        let glue_slots: Vec<usize> = glue
            .as_ref()
            .map(|(ids, _, _)| ids.clone())
            .unwrap_or_default();
        let out = self.wave_step(decode_seqs, decode_inputs, verify_seqs, verify_inputs, glue);
        if out.is_err() {
            // The glue was taken off the queue and its fill did not complete.
            // Each slot's assembly recorded those glue pieces as placed;
            // unfilled, they are zero chunks no later rebuild may keep.
            for slot in glue_slots {
                if let Some(state) = self.slot_projection_state.get_mut(&SequenceId(slot)) {
                    state.placed_pieces.clear();
                }
            }
        }
        out
    }

    /// One wave step of [`Self::decode_forward_cobatched`], with the wave's
    /// deferred glue already taken.
    fn wave_step(
        &mut self,
        decode_seqs: &[usize],
        decode_inputs: &[Tensor],
        verify_seqs: &[usize],
        verify_inputs: &[Tensor],
        glue: Option<(Vec<usize>, Vec<Tensor>, Vec<PendingGlue>)>,
    ) -> candle::Result<Vec<Tensor>> {
        let n = self.model.num_layers().max(1);
        let n_dec = decode_seqs.len();
        let none_seqs: [usize; 0] = [];
        let none_inputs: [Tensor; 0] = [];
        // A speculative step's verify blocks. They are multi-token, so they take
        // the PREFILL slot rather than the decode slot — but they are
        // **full-sweep** members like decode, not creep: their logits are read
        // by the accept walk in this same step, so a block held mid-sweep would
        // stall the decode it exists to accelerate. They therefore ride the
        // prefill slot in EVERY segment, and the creep joins them only inside
        // its window. Empty on an ordinary decode wave, which collapses all of
        // this back to what it was.
        let verify_tok: usize = verify_inputs
            .iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(0))
            .sum();
        // Rows ahead of the creep in caller order, and the logits prefix the
        // caller gets back: `[decode | verify]`.
        let head_rows = n_dec + verify_tok;
        // Wave-step wall-clock, shared across the co-batched classes so the prefill
        // and section throughput panels reflect the CONCURRENT rate (they ride
        // decode's sweep in one forward) rather than reading zero.
        let t_wave = Instant::now();

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
                let _g = super::profile::span("loop:wave:form_group");
                self.form_wave_group(!self.wave_section_advanced);
            }
            let _g = super::profile::span("loop:wave:build_inputs");
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
                let _g = super::profile::span("loop:wave:build_section");
                self.build_section_batch()
            } else {
                None
            };
            let (sec_seqs, sec_inputs, sec_gidx, sec_adv) = match section {
                Some((s, i, g, a)) => (s, i, g, a),
                None => (Vec::new(), Vec::new(), Vec::new(), Vec::new()),
            };
            if sec_seqs.is_empty() && !has_fullsweep && verify_seqs.is_empty() {
                // Nothing to run: no creep, no section, no decode, no glue, no
                // verify blocks.
                return Ok(Vec::new());
            }
            if !sec_seqs.is_empty() {
                self.wave_section_advanced = true;
            }
            if let Some(p) = glue_pending {
                self.session.set_pending_glue(p.clone());
            }
            // Verify blocks lead the prefill slot so `[decode | verify]` stays
            // the logits prefix regardless of what else joined.
            let pre_seqs: Vec<usize> = verify_seqs.iter().chain(&sec_seqs).copied().collect();
            let pre_inputs: Vec<Tensor> =
                verify_inputs.iter().chain(&sec_inputs).cloned().collect();
            // **The model's own time, separated from the scheduler's around it.**
            // Without this the whole wave step reads as one opaque block: on the
            // flagship `loop:prefill` was 336 ms/call against a 132 ms device
            // sweep, and nothing said whether the difference was the forward
            // waiting on the device or the assembly on either side of it.
            let out = {
                let _g = super::profile::span("loop:wave:forward");
                self.model.forward_wave(
                    &mut self.session,
                    decode_seqs,
                    decode_inputs,
                    &pre_seqs,
                    &pre_inputs,
                    glue_seqs,
                    glue_inputs,
                    0,
                    n,
                    None,
                )?
            };
            if has_glue {
                let _g = super::profile::span("loop:wave:reconcile_offsets");
                self.reconcile_wave_offsets(glue_seqs)?;
            }
            let logits = {
                let _g = super::profile::span("loop:wave:logits_owned");
                out.logits_owned()?
            };
            let d = head_rows.min(logits.len());
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
        let seg1_res: Option<Tensor> = if cursor > 0 && (has_fullsweep || !verify_seqs.is_empty()) {
            if let Some(p) = glue_pending {
                self.session.set_pending_glue(p.clone());
            }
            let _g = super::profile::span("loop:wave:forward");
            self.model
                .forward_wave(
                    &mut self.session,
                    decode_seqs,
                    decode_inputs,
                    verify_seqs,
                    verify_inputs,
                    glue_seqs,
                    glue_inputs,
                    0,
                    cursor,
                    None,
                )?
                .into_residual()
        } else {
            None
        };
        // Split seg1's `[decode | verify | glue]` so the creep residual inserts
        // between them for seg2's `[decode | verify | creep | glue]` order.
        let (seg1_dec, seg1_glue): (Option<Tensor>, Option<Tensor>) = match &seg1_res {
            Some(r) => {
                let dec = if head_rows > 0 {
                    Some(r.narrow(1, 0, head_rows)?)
                } else {
                    None
                };
                let g = if glue_tok > 0 {
                    Some(r.narrow(1, head_rows, glue_tok)?)
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
        // Verify blocks lead the prefill slot, the creep follows: `[decode |
        // verify]` then stays the contiguous head that every segment shares and
        // that the caller reads its logits from.
        let mid_seqs: Vec<usize> = verify_seqs.iter().chain(&seq_ids).copied().collect();
        let mid_inputs: Vec<Tensor> = verify_inputs.iter().chain(&inputs).cloned().collect();
        let _g_seg2 = super::profile::span("loop:wave:forward");
        let seg2 = match self.model.forward_wave(
            &mut self.session,
            decode_seqs,
            decode_inputs,
            &mid_seqs,
            &mid_inputs,
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
        // Closed here, not at end of scope. Left to drop naturally it would still be
        // live when seg3's span opens below under the SAME name, so a paused wave
        // (`win_end < n` with a full-sweep member) counted seg3's forward twice and
        // inflated the call count — and it would also charge this span with the
        // tallying, the residual narrowing and `complete_wave_group`, which is the
        // opposite of what it is for.
        _g_seg2.end();

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
                    WaveMember::Prefill { seq_id, .. } => {
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
            // **Teach the planner what this machine actually did.** A forward is
            // `T = X/bw + W·c` — non-resident expert bytes over the bus, plus
            // compute for the rows — so one observation is one equation in two
            // unknowns and a set of them at different widths and residencies
            // determines both. Without this the model runs on its seeds forever,
            // and the seeds are a measurement of one card with one checkpoint:
            // the decode one was found 26x optimistic, which is why the model
            // refuses to judge on it until it has samples.
            //
            // Prefill and section rows are both prefill rows to the model — the
            // copy is per forward, and they rode the same one.
            self.observe_prefill_forward(pf_tok + sc_tok, t_seg2.elapsed().as_micros() as u64);
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
                let _g = super::profile::span("loop:wave:reconcile_offsets");
                self.reconcile_wave_offsets(glue_seqs)?;
            }
            let logits = {
                let _g = super::profile::span("loop:wave:logits_owned");
                seg2.logits_owned()?
            };
            let d = head_rows.min(logits.len());
            let creep_end = (d + members.len()).min(logits.len());
            let dec_logits = logits[..d].to_vec();
            let member_logits = logits[d..creep_end].to_vec();
            self.complete_wave_group(&members, &member_logits);
            return Ok(dec_logits);
        }

        // Paused: split seg2's `[decode | creep | glue]` residual. Hold the creep
        // whole; continue the full-sweep members `[decode | glue]` into seg3.
        let res = seg2
            .into_residual()
            .ok_or_else(|| candle::Error::Msg("co-batch wave: missing residual".into()))?;
        let dec_part = if head_rows > 0 {
            Some(res.narrow(1, 0, head_rows)?)
        } else {
            None
        };
        let creep_part = res.narrow(1, head_rows, creep_tok)?;
        let glue_part = if glue_tok > 0 {
            Some(res.narrow(1, head_rows + creep_tok, glue_tok)?)
        } else {
            None
        };
        self.wave_prefill_residual = Some(creep_part);
        self.wave_prefill_cursor = win_end;
        self.wave_prefill_members = members;

        // Segment 3 — full-sweep members only over [win_end, N). Input caller order
        // `[decode | verify | glue]`. Skipped when there is no full-sweep member
        // (the creep paused at win_end, nothing else to sweep).
        if !has_fullsweep && verify_seqs.is_empty() {
            return Ok(Vec::new());
        }
        let seg3_in = Self::cat_caller_residual(&[dec_part.as_ref(), glue_part.as_ref()])?;
        if let Some(p) = glue_pending {
            self.session.set_pending_glue(p.clone());
        }
        let _g_seg3 = super::profile::span("loop:wave:forward");
        let seg3 = self.model.forward_wave(
            &mut self.session,
            decode_seqs,
            decode_inputs,
            verify_seqs,
            verify_inputs,
            glue_seqs,
            glue_inputs,
            win_end,
            n,
            seg3_in,
        )?;
        // Closed before the post-forward work, for the same reason as seg2's.
        _g_seg3.end();
        if has_glue {
            let _g = super::profile::span("loop:wave:reconcile_offsets");
            self.reconcile_wave_offsets(glue_seqs)?;
        }
        let mut logits = {
            let _g = super::profile::span("loop:wave:logits_owned");
            seg3.logits_owned()?
        };
        logits.truncate(head_rows);
        Ok(logits)
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
        // **A prefill leaving this list is an admission opportunity.**
        //
        // It is the most common completion an ingest workload has — many short
        // turns that prefill, decode a summary and end — and it frees a prefill
        // slot under the width backstop, which is the definition the settled
        // gate is written to: nothing new can fit that did not fit before
        // unless something freed ground, and this freed some.
        //
        // Without it the engine pins at one prefill in flight. The pass clears
        // the flag, the prefill finishes, nothing sets it again, so the next
        // pass takes `fill`'s fast path and the keep-one-alive guard forces a
        // single head — which is then in flight, so the pass after that admits
        // nothing at all until an unrelated event. The rate model never gets
        // consulted and the queue drains one turn at a time through the
        // deadlock guard, which is the opposite of filling a wave.
        if !self.active_prefills.is_empty() {
            self.settled_since_admit = true;
        }
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
                    None => {
                        let t = std::time::Instant::now();
                        self.complete_compression_turn(slot, job_id);
                        crate::scheduler::run::note_promote_split(
                            crate::scheduler::run::PromoteStep::Compression,
                            t.elapsed().as_micros() as u64,
                        );
                    }
                }
                continue;
            }
            // A turn that will never decode never finalizes its view, so the
            // view is released here or not at all.
            if let Some(e) = error {
                let _ = work.event_tx.send(TurnEvent::Error(e));
                // Reclaim the carved view, or the sequence wedges forever: the
                // view was registered in `turn_views` before the prefill ran and
                // never reached `active_decodes`, so a dangling one makes the
                // parent's next `SubmitTurn` wind-down refuse with `TurnInFlight`
                // for good. Every prefill-error path drains through here.
                self.discard_turn_view(work.sequence_id);
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
                    self.discard_turn_view(work.sequence_id);
                    continue;
                }
            };
            let prefill_ms = prefill_start
                .map(|s| s.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let turn_start = work.submitted_at;
            let token_count = work.tokens.len();
            let t_fin = std::time::Instant::now();
            self.finalise_prefill(work, logits, prefill_ms, turn_start, token_count);
            crate::scheduler::run::note_promote_split(
                crate::scheduler::run::PromoteStep::Finalise,
                t_fin.elapsed().as_micros() as u64,
            );
            // swap_remove pulled the last element into i; don't increment.
        }
    }

    /// Post-forward path shared by both single and batched prefill: sample
    /// the first token, emit it, and either transition to decode or close
    /// the turn out immediately on EOS / max_decode_tokens == 0.
    fn finalise_prefill(
        &mut self,
        mut work: PrefillWork,
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

        // **The turn decodes no further than the model's RoPE reaches.** Its
        // N generated tokens and the trailing structural tokens forwarded at
        // the seal all take positions past `context_depth`, and a header past
        // the schedule's last ceiling is refused — for the whole wave it rides
        // in. So a turn that would outgrow the reach ends where it runs out, as
        // one that spent its budget, and the rest of the wave never sees it.
        let room = self
            .session
            .rope_reach()
            .saturating_sub(context_depth + work.post_decode_tokens.len());
        if work.max_decode_tokens > room {
            tracing::warn!(
                seq_id = work.sequence_id.0,
                context_depth,
                budget = work.max_decode_tokens,
                room,
                reach = self.session.rope_reach(),
                "turn budget capped at the model's RoPE reach",
            );
            work.max_decode_tokens = room;
        }

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
        sampling_state.end_turn(work.sampling.cross_turn_window);
        sampling_state.record_context_tokens(&work.tokens, self.sampler.max_recent_len());

        // Send prefill progress: complete (single-prefill path needs this;
        // batched path already streams progress per-chunk, but a final
        // tokens_done==tokens_total event is always benign).
        let _ = work.event_tx.send(TurnEvent::PrefillProgress {
            tokens_done: token_count,
            tokens_total: token_count,
        });

        // ── a turn that BEGINS inside a grammar ──────────────────────────────
        //
        // `triggers` cannot express this. Both registry checks run on *sampled*
        // tokens — this function's first-token check and the decode loop's
        // per-token one — so a turn whose grammar is entered on a token the
        // caller prefilled would never arm at all: the prefill goes into K/V
        // without passing the sampler. That is not a missed optimisation, it is
        // a grammar that silently does not apply, and the decode then imitates
        // the shape it was seeded with while nothing enforces it.
        //
        // So the tree is armed here, before anything is sampled. Its opening
        // scaffold was already appended to `work.tokens` when the turn was
        // assembled (`Conversation::submit_turn`), which is why the walk starts
        // by replaying it: the driver has to sit at the same node the K/V does.
        // What is left is the first genuine choice the grammar leaves open, and
        // the first sampled token of the turn is taken under its mask.
        let mut turn_driver = work.turn_grammar.clone().map(StencilDriver::new);
        let mut sampling = work.sampling.clone();
        if let Some(driver) = turn_driver.as_mut() {
            let (scaffold, action) = driver.opening();
            match &action {
                StepMask::Branch(set) => {
                    sampling.stencil = set.tokens().iter().map(|&t| t as i32).collect();
                }
                // A tree whose opening is free text or empty constrains nothing
                // here; the decode loop picks it up from the next step.
                StepMask::Free { .. } | StepMask::Done | StepMask::Prefill(_) => {}
            }
            tracing::debug!(
                target: "candle_conversation::stencil",
                seq_id = work.sequence_id.0,
                tree = driver.tree().label(),
                scaffold = scaffold.len(),
                masked = matches!(action, StepMask::Branch(_)),
                "turn grammar armed at the prefill boundary",
            );
        }

        let sampled = match self.sample_single(&logits, &sampling, &mut sampling_state) {
            Ok(t) => t,
            Err(e) => {
                self.sampling_states
                    .insert(work.sequence_id, sampling_state);
                let _ = work.event_tx.send(TurnEvent::Error(e));
                return;
            }
        };
        // A replayed turn opens with its recording's first id, whatever the
        // prefill's logits chose.
        let first_token = match (work.recorded_reply.as_deref(), self.eos_tokens.first()) {
            (Some(reply), Some(&eos)) => replayed_step(reply, 0, eos),
            _ => sampled,
        };

        // Detect think-mode entry: the model opens its OWN `<think>` as the first
        // decoded token, or the assistant prefill leaves one open. "Leaves open"
        // is not "contains" — see [`prefill_leaves_think_open`].
        let initial_inside_think_block = {
            let tid = work.sampling.segment_open_token_id;
            if tid >= 0 {
                let tok = tid as u32;
                // **Only the ASSISTANT lead can leave a block open**, so the scan
                // starts at `assistant_content_start`. The markers are ordinary
                // vocabulary ids, and the tokenizer emits them for the literal
                // text too — so a `<tool_response>` carrying source that merely
                // MENTIONS `<think>` puts the open id in the USER half of the
                // grid. This repo's own `dialect.rs` does exactly that, and the
                // code-reading ingest feeds it back in. Scanning the whole grid
                // would arm `in_segment` off that quoted text before the turn had
                // decoded anything, and under `ThinkMode::Off` (hard cap of one)
                // the second decoded token would be rewritten to `</think>`.
                let assistant_lead = work
                    .tokens
                    .get(work.assistant_content_start as usize..)
                    .unwrap_or(&[]);
                let close = u32::try_from(work.sampling.segment_close_token_id).ok();
                let prefill_has_think =
                    prefill_leaves_think_open(assistant_lead.iter(), tok, close);
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
            // The first token ended the turn: an end-of-sequence, or a budget of
            // zero decoded tokens.
            let finish = if self.is_eos(first_token) {
                FinishReason::Stop
            } else {
                FinishReason::Length
            };
            // View sequences (SubmitTurn path): the prefill already wrote KV
            // blocks that must be finalized onto the parent and sealed into
            // the substrate.  Insert as a finished DecodeState so
            // cleanup_finished runs finalize_view + perform_seal_and_write.
            //
            // Non-view sequences (raw RULER / summarisation): no parent to
            // finalize and seal=None is correct — use the fast path.
            if self.turn_views.contains_key(&work.sequence_id) {
                // Through `push_generated` like every other token, so the
                // reasoning boundary is seen even on this no-decode path.
                let mut state = DecodeState {
                    event_tx: work.event_tx,
                    generated_tokens: TokenBuffer::default(),
                    think_close_at: None,
                    forwarded_generated: 0,
                    draft_depth: DraftDepth::default(),
                    pending_page_cut: false,
                    pending_page_cut_after: None,
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
                    finish,
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
                    free_tool_calls_from_penalties: work.free_tool_calls_from_penalties,
                    triggers: work.triggers,
                    stencil: None,
                    pending_mask: None,
                    recorded_reply: work.recorded_reply.map(Replay::new),
                };
                // The turn's first token opens a page at the prefill/decode
                // boundary, so the reasoning starts one of its own.
                state.push_committed(first_token, self.think_close, &self.page_break_tokens);
                // No speculative rewind can be in flight — this turn decodes
                // nothing — so the cut is taken at once.
                super::flush_page_cut(self.model.as_ref(), work.sequence_id, &mut state);
                self.active_decodes.insert(work.sequence_id, state);
            } else {
                self.finish_immediately(
                    work.sequence_id,
                    first_token,
                    &work.event_tx,
                    prefill_ms,
                    turn_start,
                    context_depth,
                    finish,
                );
            }
            return;
        }

        let _ = work.event_tx.send(TurnEvent::Token(first_token));

        // An armed turn grammar already owns this token — it was sampled under
        // the mask above, so it is fed back rather than tested against the
        // registry. A turn cannot be in both states: beginning inside a tree and
        // entering one on this token are the same slot.
        let stencil = match turn_driver {
            Some(mut driver) => {
                let bytes = self
                    .tokenizer
                    .decode(&[first_token], false)
                    .unwrap_or_default();
                driver.accept(first_token, bytes.as_bytes());
                Some(driver)
            }
            // The first sampled token can itself be a stencil trigger — e.g. the
            // model emits `<tool_call>` as its very first response token. The
            // decode-loop trigger check runs only on tokens sampled in
            // `batch_decode_step`, never this one, so check it here too —
            // otherwise steering silently never engages for those calls.
            None => {
                let d = work.triggers.driver_for(first_token);
                // A once-trigger (the think block) is spent by firing, so the
                // rest of the turn decodes that token as text.
                if let Some(rest) = work.triggers.after_firing(first_token) {
                    work.triggers = Arc::new(rest);
                }
                if let Some(d) = &d {
                    tracing::debug!(
                        target: "candle_conversation::stencil",
                        seq_id = work.sequence_id.0,
                        tree = d.tree().label(),
                        trigger = first_token,
                        "stencil steering started (trigger on the first decoded token)",
                    );
                }
                d
            }
        };
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

        // Through `push_generated` like every other token, so a `</think>` the
        // prefill's own logits produced still fixes the reasoning boundary.
        let mut state = DecodeState {
            event_tx: work.event_tx,
            generated_tokens: TokenBuffer::default(),
            think_close_at: None,
            forwarded_generated: 0,
            draft_depth: DraftDepth::default(),
            pending_page_cut: false,
            pending_page_cut_after: None,
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
            finish: FinishReason::Stop,
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
            free_tool_calls_from_penalties: work.free_tool_calls_from_penalties,
            triggers: work.triggers,
            stencil,
            pending_mask: None,
            recorded_reply: work.recorded_reply.map(Replay::new),
        };
        // The turn's first token opens a page at the prefill/decode boundary, so
        // the reasoning starts one of its own.
        state.push_committed(first_token, self.think_close, &self.page_break_tokens);
        // No speculative rewind can be in flight on this path — the turn has not
        // decoded yet — so the cut is taken at once.
        super::flush_page_cut(self.model.as_ref(), work.sequence_id, &mut state);
        self.active_decodes.insert(work.sequence_id, state);
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

    /// Forward `tokens` on `sequence_id`, splitting the pass at the turn's
    /// reasoning boundary if it falls inside them.
    ///
    /// **A forward must not carry tokens from both sides of `</think>`.** A
    /// span's index rows are pooled by the forward that carries it, so a block
    /// pooled across the boundary cannot be un-pooled at seal time and the
    /// reasoning would not occupy whole pages. Plain decode never straddles —
    /// one token per sequence per wave — but a static run does: a think-steer
    /// tree suppresses the model's own `</think>` and injects the closing tag as
    /// a run, which can carry tokens after it.
    ///
    /// Splitting here rather than at the two call sites is what makes the
    /// invariant structural: this is the one function that forwards an arbitrary
    /// token span for a single sequence, so a future third caller inherits it.
    pub(super) fn run_prefill(
        &mut self,
        sequence_id: SequenceId,
        tokens: &[u32],
    ) -> Result<Tensor, ConversationError> {
        // **Every break token in the span, not just the first.** A prefilled
        // assistant head carries `<think>` and `</think>` in one pass, and a
        // multi-turn prefill carries a turn closer as well — splitting once
        // would leave the later markers pooled across their own boundaries,
        // which is not correctable afterwards.
        let mut rest = tokens;
        let mut last_logits = None;
        while let Some(at) = self.reasoning_split(rest) {
            let (head, tail) = rest.split_at(at);
            last_logits = Some(self.run_prefill_span(sequence_id, head)?);
            match self.model.close_positional_page(sequence_id.0) {
                // The head's own token ids when it is short. A surplus page is
                // identified by what is IN it, and the pages that do not belong
                // to any turn are consistently 5 and 7 tokens wide — small
                // enough to name outright rather than infer from their width.
                Ok(closed) => tracing::info!(
                    target: "candle_conversation::scheduler::unit_boundary",
                    seq_id = sequence_id.0,
                    site = "prefill-break-token",
                    closed,
                    at,
                    span = rest.len(),
                    head = ?(head.len() <= 16).then_some(head),
                    break_token = ?head.last(),
                    "index: closed a page mid-prefill at a break token"
                ),
                Err(e) => tracing::warn!(
                    seq_id = sequence_id.0,
                    "closing the index page at a prefilled break token failed ({e}); the \
                     region it bounds will not occupy whole pages and cannot be windowed \
                     out of a later projection"
                ),
            }
            rest = tail;
        }
        if rest.is_empty() {
            // Every token was consumed by a split, so the last head's logits are
            // the span's — `reasoning_split` never returns a split at the end,
            // so this is only reachable for an empty input.
            return match last_logits {
                Some(l) => Ok(l),
                None => self.run_prefill_span(sequence_id, rest),
            };
        }
        self.run_prefill_span(sequence_id, rest)
    }

    /// Where to cut `tokens` so the reasoning boundary lands on a page edge:
    /// one past this turn's first `</think>`, or `None` when the span carries no
    /// boundary that needs one.
    ///
    /// `None` when the marker is absent, when it is the last token (nothing
    /// follows it in this pass, so the next forward is already the edge), or
    /// when the turn has recorded a close already — `think_close_at` holds the
    /// first only, and a later `</think>` in the answer body must not move a
    /// boundary that is fixed.
    /// **Reads the token stream, not the decode state.** The previous version
    /// asked `active_decodes` for the turn's `DecodeState` — which does not exist
    /// yet while the turn is prefilling, because it is built from the prefill's
    /// own logits afterwards. So for the case this exists to serve, a
    /// `<think>…</think>` block baked into the prompt, the lookup returned `None`
    /// and the pass was never split: 0 splits across 822 turns.
    ///
    /// One past the first break token, and `None` when that is the end of the
    /// span — there is nothing on the far side to separate.
    fn reasoning_split(&self, tokens: &[u32]) -> Option<usize> {
        let at = tokens
            .iter()
            .position(|t| self.page_break_tokens.contains(t))?
            + 1;
        (at < tokens.len()).then_some(at)
    }

    fn run_prefill_span(
        &mut self,
        sequence_id: SequenceId,
        tokens: &[u32],
    ) -> Result<Tensor, ConversationError> {
        // Chunked prefill: split large prompts into bounded chunks to keep
        // intermediate activation buffers from growing unboundedly.
        let pass = self.prefill_pass_budget();
        let logits = if tokens.len() > pass {
            let mut last_logits: Option<Tensor> = None;
            for chunk in tokens.chunks(pass) {
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
                    .and_then(|s| s.logits_owned())
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
                .and_then(|s| s.logits_owned())
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

        // Nothing to bring up to date here: the prefill's own commit
        // (`KvCache::commit_written_tokens`, the one place every write outside
        // the decode kernel goes through) marked the cached decode slot buffer,
        // and the next sync that reads it re-serialises its writer region.
        Ok(logits)
    }
}

/// Whether an assistant prefill leaves a think block **open** at the point
/// decode takes over.
///
/// **Containing `<think>` is not the question.** Qwen3.5 suppresses reasoning by
/// prefilling an already-closed block, `<think>\n\n</think>\n\n`, so the open
/// marker is in every suppressed turn's prefill by construction — and the
/// sampler's segment state only ever sees *sampled* tokens, so the prefilled
/// close never reaches it. Asked "is `<think>` among the last few tokens", the
/// check answered yes on exactly the turns where thinking had been turned off,
/// and the sampler then decoded the whole answer believing it was inside a
/// think block: the ban on `</think>` outside a block lifted, and any
/// `force_segment_close_after` fired a closer into the prose. Measured on an
/// unstencilled reflection turn: `The belt is</think>`, nine tokens, the answer
/// ended by a forced close of a block that had closed before it began.
///
/// The most recent marker decides, which is the rule that holds whatever the
/// prefill is: a closed block, a closed block followed by a tool-call opener, a
/// bare `<think>` for a turn that is meant to reason, or an earlier turn's
/// markers further back in the buffer. The scan stops at the first marker from
/// the end, so it costs a handful of comparisons on every real prefill.
/// **The caller decides what is scanned, and it hands over the ASSISTANT lead
/// only.** The markers are ordinary vocabulary ids and the tokenizer emits them
/// for literal text, so a `<tool_response>` carrying source that merely mentions
/// `<think>` puts the open id in the USER half of the grid — this repo's own
/// `dialect.rs` does exactly that, and the code-reading ingest feeds it back.
/// Scanning a whole prefill would arm `in_segment` off that quoted text before
/// the turn had decoded anything; under `ThinkMode::Off`, where the hard cap is
/// one token, the second decoded token then gets rewritten to `</think>`. The
/// slice at `assistant_content_start` is what prevents it, and
/// `only_the_assistant_lead_is_scanned` holds the boundary.
fn prefill_leaves_think_open<'a>(
    tokens: impl DoubleEndedIterator<Item = &'a u32>,
    open: u32,
    close: Option<u32>,
) -> bool {
    tokens
        .rev()
        .find(|&&t| t == open || Some(t) == close)
        .is_some_and(|&t| t == open)
}

/// The slice boundary the helper above relies on, which main's own module does
/// not exercise: a prefill whose USER half quotes `<think>` while its assistant
/// lead closes its block.
#[cfg(test)]
mod think_prefill_slice_tests {
    use super::prefill_leaves_think_open;

    const OPEN: u32 = 248068;
    const CLOSE: u32 = 248069;

    /// **User content that merely QUOTES `<think>` must not arm the flag.**
    #[test]
    fn only_the_assistant_lead_is_scanned() {
        // `[user … <think> … ] [assistant lead: closed block]`
        let grid = [9u32, OPEN, 9, 9, OPEN, 3, CLOSE, 4];
        let assistant_content_start = 4;
        assert!(
            !prefill_leaves_think_open(grid[assistant_content_start..].iter(), OPEN, Some(CLOSE)),
            "the assistant lead closed its block; quoted user text must not override that"
        );
        // What the slice prevents: handed the user half, the very same scan does
        // arm — so the boundary is doing the work, not the scan.
        assert!(prefill_leaves_think_open(
            grid[..2].iter(),
            OPEN,
            Some(CLOSE)
        ));
    }

    /// A deep opener in the assistant lead still arms. The fixed 5-token tail
    /// window this replaced would have missed it entirely.
    #[test]
    fn a_deep_opener_in_the_lead_still_arms() {
        let mut deep = vec![OPEN];
        deep.extend(std::iter::repeat_n(7u32, 40));
        assert!(prefill_leaves_think_open(deep.iter(), OPEN, Some(CLOSE)));
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

#[cfg(test)]
mod wave_chunk_tests {
    use std::sync::Arc;
    use std::time::Instant;

    use candle_transformers::models::rope_schedule::RungSelect;

    use super::super::tests::make_test_scheduler;
    use super::super::*;

    /// A dialogue prefill carrying `tokens` and nothing else.
    pub(super) fn dialogue_prefill(seq: SequenceId, tokens: Vec<u32>) -> ActivePrefill {
        let (event_tx, _event_rx) = flume::unbounded();
        ActivePrefill {
            work: PrefillWork {
                sequence_id: seq,
                tokens: TokenBuffer::from(tokens),
                prefill_text: String::new(),
                user_text: String::new(),
                tags: Vec::new(),
                user_content_start: 0,
                user_content_end: 0,
                assistant_content_start: 0,
                no_think: false,
                prefill_assistant_text: String::new(),
                event_tx,
                max_decode_tokens: 0,
                sampling: SamplingConfig::compression(),
                submitted_at: Instant::now(),
                reprojection: None,
                belief: PriorBelief::default(),
                seal_action: SealAction::None,
                post_decode_tokens: TokenBuffer::default(),
                projection_offsets: Vec::new(),
                staged_composition: None,
                triggers: Arc::new(TriggerRegistry::new()),
                turn_grammar: None,
                free_tool_calls_from_penalties: false,
                recorded_reply: None,
            },
            offset: 0,
            next_projection: 0,
            final_logits: None,
            error: None,
            prefill_start: None,
        }
    }

    /// **A dialogue prefill enters the wave group in pass-sized chunks.** It
    /// used to enter whole: a 15.2k-token Cline turn needed a 13.2 GiB transient
    /// tier on a 16 GB card, more than all the ground below the weight floor,
    /// and failed identically on every retry.
    #[test]
    fn a_long_dialogue_prefill_creeps_in_pass_sized_chunks() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let cap = scheduler.max_prefill_pass_tokens;
        let seq = SequenceId(scheduler.session.create_sequence().expect("create"));
        scheduler
            .active_prefills
            .push(dialogue_prefill(seq, (0..15_000u32).collect()));

        scheduler.form_wave_group(false);
        let (members, _, inputs, _) = scheduler.build_wave_group_inputs();
        assert_eq!(members.len(), 1);
        assert_eq!(
            inputs[0].dims(),
            &[1, cap],
            "the first chunk is one pass wide"
        );

        // A later group resumes at the committed offset, never from token 0.
        scheduler.reset_wave_prefill();
        scheduler.active_prefills[0].offset = 14_800;
        scheduler.form_wave_group(false);
        let (_, _, inputs, _) = scheduler.build_wave_group_inputs();
        let rows = inputs[0].to_vec2::<u32>().expect("u32 rows");
        assert_eq!(rows[0].len(), 200, "the tail chunk is what remains");
        assert_eq!(rows[0][0], 14_800, "the chunk starts at the offset");
    }

    /// Prefills share one pass budget: small ones pack together, and one that
    /// would carry the pass past the cap waits for the next group.
    #[test]
    fn dialogue_prefills_pack_under_one_pass_budget() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let cap = scheduler.max_prefill_pass_tokens;
        let a = SequenceId(scheduler.session.create_sequence().expect("create"));
        let b = SequenceId(scheduler.session.create_sequence().expect("create"));
        let c = SequenceId(scheduler.session.create_sequence().expect("create"));
        scheduler
            .active_prefills
            .push(dialogue_prefill(a, vec![1; 100]));
        scheduler
            .active_prefills
            .push(dialogue_prefill(b, vec![1; 100]));
        scheduler
            .active_prefills
            .push(dialogue_prefill(c, vec![1; cap]));

        scheduler.form_wave_group(false);
        let seqs: Vec<usize> = scheduler
            .wave_prefill_members
            .iter()
            .map(|m| m.seq_id())
            .collect();
        assert_eq!(
            seqs,
            vec![a.0, b.0],
            "the third would carry the pass past the cap"
        );
    }

    /// **A prompt past the model's RoPE reach fails alone.** Its headers would
    /// be refused and take the whole forward with them; instead it is errored
    /// before the group forms, and a prompt that fits rides as usual.
    #[test]
    fn a_prompt_past_the_rope_reach_fails_alone() {
        let (mut scheduler, _tx) = make_test_scheduler();
        scheduler
            .session
            .set_rope_select(RungSelect::new(vec![64], 0).expect("ceilings"));
        let fits = SequenceId(scheduler.session.create_sequence().expect("create"));
        let over = SequenceId(scheduler.session.create_sequence().expect("create"));
        scheduler
            .active_prefills
            .push(dialogue_prefill(fits, vec![1; 64]));
        scheduler
            .active_prefills
            .push(dialogue_prefill(over, vec![1; 65]));

        scheduler.form_wave_group(false);
        let seqs: Vec<usize> = scheduler
            .wave_prefill_members
            .iter()
            .map(|m| m.seq_id())
            .collect();
        assert_eq!(
            seqs,
            vec![fits.0],
            "only the prompt that fits joins the group"
        );
        assert!(scheduler.active_prefills[0].error.is_none());
        assert!(
            matches!(
                &scheduler.active_prefills[1].error,
                Some(ConversationError::Other(m)) if m.contains("65") && m.contains("64")
            ),
            "the prompt past the reach carries its own error"
        );
    }
}

#[cfg(test)]
mod turn_view_release_tests {
    use super::super::tests::{make_test_scheduler, register_turn_view};
    use super::super::*;
    use super::wave_chunk_tests::dialogue_prefill;

    /// **An errored turn prefill releases its view.** The error reached the
    /// caller and the view stayed registered with the parent's prefix borrowed
    /// — nothing but a completed decode or a reprojection ever released it, and
    /// an errored turn has neither.
    #[test]
    fn an_errored_turn_prefill_releases_its_view() {
        let (mut scheduler, _tx) = make_test_scheduler();
        let parent = SequenceId(scheduler.session.create_sequence().expect("create"));
        let view = register_turn_view(&mut scheduler, parent);
        let mut prefill = dialogue_prefill(view, vec![1; 8]);
        prefill.error = Some(ConversationError::Channel("the wave failed".into()));
        scheduler.active_prefills.push(prefill);

        scheduler.promote_finished_prefills_to_decodes();

        assert!(scheduler.turn_views.is_empty(), "the view is unregistered");
        assert!(
            scheduler.session.sequence_offset(view.0).is_none(),
            "the view's slot is released"
        );
        assert!(
            scheduler.session.sequence_offset(parent.0).is_some(),
            "the parent is untouched"
        );
    }
}

#[cfg(test)]
mod priority_admission_tests {
    use std::sync::Arc;

    use super::super::tests::make_test_scheduler;
    use super::super::*;
    use super::wave_chunk_tests::dialogue_prefill;

    const YAML: &str = r#"
system_prompt:
  sections:
    - id: frame
      content: "frame"
layers:
  - name: dialogue
    window: 8000
    decode_priority: high
    summary:
      turns:
        max_tokens: 256
        user: { system_prompt: s, user_prompt: u }
        assistant: { system_prompt: s, user_prompt: u }
    score_formula: max
    budget: { priority: 40 }
    groups:
      - id: chat
        selection: { kind: top_k, k: 2 }
  - name: ingest
    window: 8000
    decode_priority: low
    summary:
      turns:
        max_tokens: 256
        user: { system_prompt: s, user_prompt: u }
        assistant: { system_prompt: s, user_prompt: u }
    score_formula: max
    budget: { priority: 10 }
    groups:
      - id: files
        selection: { kind: top_k, k: 2 }
"#;

    /// Bind `seq` to `layer`'s group on its own timeline, so its priority
    /// resolves through the schema the way a live slot's does.
    fn bind(sched: &mut Scheduler, builder: &Arc<Builder>, seq: SequenceId, layer: &str, tl: u64) {
        let group = if layer == "dialogue" { "chat" } else { "files" };
        let timeline = TimelineId::from_raw(tl).expect("timeline id");
        sched.slot_targets.insert(
            seq,
            ProjectionTarget {
                layer: builder.id_for_layer(layer).unwrap(),
                group: builder.id_for_group(group).unwrap(),
                timeline,
            },
        );
        sched
            .timeline_projections
            .insert(timeline, Arc::clone(builder));
    }

    /// **A dialogue turn queued behind paused ingest is admitted.** The paused
    /// ingest prefill moves no row, so it is not "in flight" for the
    /// keep-one-alive rule: with admission closed until something settles, the
    /// turn pausing it is the head that gets forced in. Counting the paused
    /// prefill as in flight left the turn queued with nothing able to settle.
    #[test]
    fn a_dialogue_turn_is_admitted_past_paused_ingest() {
        let (mut sched, _tx) = make_test_scheduler();
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let ingest = SequenceId(sched.session.create_sequence().expect("create"));
        let dialogue = SequenceId(sched.session.create_sequence().expect("create"));
        bind(&mut sched, &builder, ingest, "ingest", 1);
        bind(&mut sched, &builder, dialogue, "dialogue", 2);
        sched
            .active_prefills
            .push(dialogue_prefill(ingest, vec![1; 8]));
        sched
            .prefill_queue
            .push_back(dialogue_prefill(dialogue, vec![1; 8]).work);
        sched.settled_since_admit = false;

        sched.promote_new_prefills();

        assert!(
            sched.priority_paused(ingest),
            "the queued turn pauses ingest"
        );
        assert_eq!(
            sched.running_prefills(),
            1,
            "only the dialogue turn can run"
        );
        assert!(
            sched
                .active_prefills
                .iter()
                .any(|p| p.work.sequence_id == dialogue),
            "the dialogue turn was left queued behind paused ingest"
        );
    }

    /// **A slot whose priority does not resolve pauses nobody.** It is gated as
    /// `High` for itself, but it is not evidence that a conversation is running,
    /// so ingest keeps its turn.
    #[test]
    fn an_unresolvable_slot_does_not_pause_ingest() {
        let (mut sched, _tx) = make_test_scheduler();
        let builder = Arc::new(Builder::from_yaml(YAML).unwrap());
        let ingest = SequenceId(sched.session.create_sequence().expect("create"));
        let probe = SequenceId(sched.session.create_sequence().expect("create"));
        bind(&mut sched, &builder, ingest, "ingest", 1);
        sched
            .active_prefills
            .push(dialogue_prefill(ingest, vec![1; 8]));
        sched
            .active_prefills
            .push(dialogue_prefill(probe, vec![1; 8]));

        sched.observe_priorities();

        assert!(
            !sched.priority_paused(probe),
            "never paused: it counts as High"
        );
        assert!(!sched.priority_paused(ingest), "and it pauses nothing");
    }
}

#[cfg(test)]
mod prefill_think_tests {
    use super::prefill_leaves_think_open;

    const OPEN: u32 = 89;
    const CLOSE: u32 = 90;
    const NL: u32 = 10;
    const TOOL_CALL: u32 = 91;
    const WORD: u32 = 5;

    fn open(tokens: &[u32]) -> bool {
        prefill_leaves_think_open(tokens.iter(), OPEN, Some(CLOSE))
    }

    /// **The suppression prefill is a closed block, and it must read as one.**
    ///
    /// The regression this exists for: Qwen3.5's `<think>\n\n</think>\n\n`
    /// contains the open marker, and the check that asked "does it contain one"
    /// put every think-suppressed turn's sampler inside a block it would never
    /// see close.
    #[test]
    fn a_prefilled_closed_block_leaves_nothing_open() {
        assert!(!open(&[WORD, OPEN, NL, CLOSE, NL]));
    }

    /// The acting turn's prefill: the closed block, then straight into the call.
    #[test]
    fn a_closed_block_followed_by_a_call_opener_leaves_nothing_open() {
        assert!(!open(&[WORD, OPEN, NL, CLOSE, NL, TOOL_CALL]));
    }

    /// A turn that is meant to reason prefills a bare `<think>`, and it must.
    #[test]
    fn a_bare_open_marker_leaves_the_block_open() {
        assert!(open(&[WORD, NL, OPEN]));
    }

    #[test]
    fn a_prefill_with_no_markers_leaves_nothing_open() {
        assert!(!open(&[WORD, NL, WORD]));
        assert!(!open(&[]));
    }

    /// The most recent marker decides, so an earlier turn's closed block
    /// further back in the buffer cannot mask a fresh open, and cannot fake one.
    #[test]
    fn only_the_most_recent_marker_counts() {
        assert!(open(&[OPEN, WORD, CLOSE, WORD, NL, OPEN]));
        assert!(!open(&[OPEN, WORD, CLOSE, WORD, OPEN, NL, CLOSE, NL]));
    }

    /// A vocabulary with no close token cannot close a block, so an open marker
    /// anywhere behind the boundary leaves it open.
    #[test]
    fn without_a_close_token_an_open_marker_stays_open() {
        assert!(prefill_leaves_think_open(
            [WORD, OPEN, NL, CLOSE].iter(),
            OPEN,
            None
        ));
    }
}
