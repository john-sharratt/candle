use super::prefill::VramPhase;
use super::*;
use std::time::{Duration, Instant};

/// Wall-clock ceiling for one decode quantum ("wave"). The quantum is CLIPPED to
/// this whether or not decode has finished — unfinished sequences persist in
/// `active_decodes` and resume in the next quantum, so clipping just passes the
/// remaining work forward. Time-slicing (rather than the old fixed 32-step
/// budget) guarantees the loop returns to the top every ~2 s to re-form the creep
/// cohort with work admitted mid-wave, run admission + relief, and aggregate
/// telemetry — regardless of how fast or slow the individual steps run.
const WAVE_SLICE: Duration = Duration::from_millis(2000);

/// Hard safety cap on decode steps within one quantum, independent of the wall
/// clock. At the ~74 ms/step WDDM launch floor a 2 s slice is ~27 steps, far
/// under this; the cap only engages if decode ever runs far faster (TCC mode /
/// CUDA graphs), bounding per-quantum KV growth so the loop-top relief can't be
/// outrun before the deadline check fires.
const MAX_DECODE_STEPS: usize = 256;

impl Scheduler {
    /// Number of currently-active decode sequences (including summary probes).
    /// Used as the "is there decode work to run" guard — and by the decode-less
    /// creep in `run_prefill_until_budget` to skip its standalone advance when the
    /// decode quantum will co-batch the cohort instead.
    pub(super) fn decode_width(&self) -> usize {
        self.active_decodes.values().filter(|s| !s.finished).count()
    }

    /// Active *foreground* decode sequences — excludes compression passes.
    ///
    /// Count the active foreground dialogue decodes that feed the
    /// prefill/decode flip heuristic. Compression half-passes ride
    /// `active_decodes` and the decode wave like any decode, but are excluded
    /// here so they never hold the loop in decode-first mode at the expense of
    /// dialogue prefills (they're off the critical path — background work that
    /// co-batches opportunistically).
    pub(super) fn foreground_decode_width(&self) -> usize {
        self.active_decodes
            .iter()
            .filter(|(_, s)| {
                !s.finished && !matches!(s.seal_action, super::SealAction::CompressionPass { .. })
            })
            .count()
    }

    /// Run decode steps until the `WAVE_SLICE` deadline or decode is empty.
    ///
    /// The quantum is clipped by wall-clock, not step count — an unfinished
    /// generation is passed forward via `active_decodes` to the next quantum. We
    /// deliberately do **not** yield mid-quantum on width comparisons (the outer
    /// dispatcher already chose this phase; the time slice is what guarantees the
    /// other phases get airtime), but we DO admit newly-queued work mid-quantum via
    /// `mid_wave_admission` so a long generation can't starve fresh conversations.
    fn run_decode_until_budget(&mut self) {
        // Phase-b VRAM relief: a sustained decode grows KV every step with no
        // admission gate of its own, so relieve once at the quantum boundary when
        // under pressure — the governor's cheapest-first ladder sheds cold turns,
        // and the reprojection drain below handles working-set turnover. One check
        // per quantum (not per step) keeps it cheap.
        if self.vram_under_pressure_for(VramPhase::Decode) {
            self.relieve_vram_pressure("decode", VramPhase::Decode);
        }
        // Fire any reprojection queued by the just-completed prefill quantum
        // BEFORE the first decode step of this quantum. The turn's first
        // reprojection is queued in `finalise_prefill`, right after the first
        // token is sampled — for a think turn that first token is `<think>`
        // itself, so this is the projection point immediately after `<think>`:
        // it scans the freshly-prefilled query and re-selects the tools before
        // the model decodes its first *reasoning* token. Draining only at the
        // end of `batch_decode_step` (below) would let that first reasoning
        // token be sampled against the query-blind opening projection first,
        // which is exactly where a wrong-tool / hallucinated answer anchors.
        let t_reproj0 = Instant::now();
        self.drain_pending_reprojections();
        self.wave_stats
            .add_phase(WavePhase::Reproject, t_reproj0.elapsed().as_millis() as u64);
        let deadline = Instant::now() + WAVE_SLICE;
        let mut steps = 0usize;
        loop {
            if self.decode_width() == 0 {
                // No live decode work, but there may be sequences inserted as
                // finished during the prefill phase (EOS on first token) that
                // the decode loop never had a chance to clean up.
                self.cleanup_finished();
                return;
            }
            // Inject any pending tool-call static runs (Layer 3) before the
            // decode forward, so a `Static` run costs one prefill rather than N
            // decode steps.  No-op when no sequence has an active stencil.
            self.inject_stencil_prefills();
            self.batch_decode_step();
            // Drain any continuous-re-projection swaps queued during the
            // batch.  Must run BEFORE cleanup_finished so a swap that
            // re-keys an active_decodes entry doesn't race with finalize.
            // Timed separately (a sub-slice of the decode quantum) because the
            // provenance scan + glue gap-fill here is a prime "grows over time" suspect.
            let t_reproj = Instant::now();
            self.drain_pending_reprojections();
            self.wave_stats
                .add_phase(WavePhase::Reproject, t_reproj.elapsed().as_millis() as u64);
            self.cleanup_finished();
            steps += 1;

            // Take on conversations that queued WHILE this wave was executing,
            // under the same entry criteria the top-of-loop admission uses. The
            // newly-projected prefills land in `active_prefills` now and join the
            // creep cohort the moment this quantum clips to the top and
            // `form_wave_group` re-forms — so admission latency is bounded by
            // WAVE_SLICE, not by the whole in-flight generation finishing.
            if !self.mid_wave_admission() {
                self.shutdown_requested = true;
                return;
            }

            // Clip to the time slice regardless of remaining decode work; the
            // remaining sequences persist in `active_decodes` and resume next
            // quantum. The step cap is only a backstop if steps ever run far
            // under the WDDM floor (see `MAX_DECODE_STEPS`).
            if Instant::now() >= deadline || steps >= MAX_DECODE_STEPS {
                return;
            }
        }
    }

    /// Admit work that arrived mid-wave and run the cheap per-wave KV relief, so a
    /// time-sliced decode quantum adapts to newly-queued conversations and keeps KV
    /// bounded across a long slice without waiting for the quantum to end. Mirrors
    /// the top-of-loop admission (`drain` → `promote` → `pump`) under the identical
    /// `admit_window`/VRAM cap, plus the per-wave ingest throttle + gentle demote.
    ///
    /// Returns `false` if the drain observed shutdown/disconnect: the request is
    /// already consumed here (so the top-of-loop drain can't re-read it), so the
    /// caller records the intent and the main loop breaks on it.
    ///
    /// Cheap on the common path: the rx peek skips the whole admission block when
    /// nothing queued, and the ingest relief self-gates on `ingest_timelines`
    /// (a no-op outside a workspace ingest).
    fn mid_wave_admission(&mut self) -> bool {
        if !self.rx.is_empty() {
            // Flag the drain so the assembler attributes its sub-timers to the
            // drain buckets and `apply_projection` DEFERS its gap-fills into the
            // unified wave step (`take_wave_glue`), exactly as the loop-top drain.
            let t_drain = Instant::now();
            IN_DRAIN.store(true, std::sync::atomic::Ordering::Relaxed);
            self.batch_drain_gap_fills = true;
            let cont = self.drain_submissions();
            self.batch_drain_gap_fills = false;
            IN_DRAIN.store(false, std::sync::atomic::Ordering::Relaxed);
            self.wave_stats
                .add_phase(WavePhase::Drain, t_drain.elapsed().as_millis() as u64);
            if !cont {
                return false;
            }
            let t_promote = Instant::now();
            self.promote_new_prefills();
            self.wave_stats
                .add_phase(WavePhase::Promote, t_promote.elapsed().as_millis() as u64);
        }
        // Bound KV production to the hot→warm drain rate across the slice, not just
        // once per quantum. Self-gated on an active ingest so a plain dialogue decode
        // pays nothing (the loop-top call still handles the ingest-finished reopen).
        if !self.ingest_timelines.is_empty() {
            self.regulate_ingest_admission();
            self.demote_cold_ingest_if_pressured();
        }
        true
    }

    /// Advance the continuous-fair-wave prefill cohort — ONLY via the co-batched
    /// wave step (`decode_forward_cobatched`), never a separate serial forward.
    ///
    /// - If the decode quantum already advanced the cohort this wave
    ///   (`wave_cohort_advanced`), nothing to do.
    /// - If ANY decode is active, the decode quantum folds the cohort into its
    ///   sweep (one shared expert load per layer) — so skip here even though the
    ///   wave loop may have run this pass first (width ordering). Running the
    ///   cohort standalone now would split decode and the cohort into two
    ///   sequential forwards, the exact thing continuous-fair-waves removes.
    /// - Only when there is NO decode to fold into do we advance the creep here,
    ///   by running the SAME wave step with an empty decode group — a decode-less
    ///   sweep through `decode_forward_cobatched`, not a distinct serial path. That
    ///   step folds every no-decode class of work: dialogue prefills, section
    ///   chunks (`build_section_batch`) and deferred glue (`take_wave_glue`), so it
    ///   fires whenever any of the three has pending work.
    fn run_prefill_until_budget(&mut self) {
        if self.wave_cohort_advanced || self.decode_width() > 0 {
            return;
        }
        if self.prefill_width() == 0
            && self.section_ingest_width() == 0
            && self.deferred_glue_fires.is_empty()
        {
            return;
        }
        if let Err(e) = self.decode_forward_cobatched(&[], &[]) {
            tracing::error!("decode-less wave step failed: {e}");
        }
    }

    /// Seal completed section ingests. Section chunks are advanced ONLY by the
    /// unified wave step (`decode_forward_cobatched`) — co-batched into decode's
    /// sweep when decode is active, else folded into the decode-less sweep by
    /// `run_prefill_until_budget`. This quantum only drains the sections that step
    /// *completed* so they seal + send their `SealResult`. Cheap when none are done.
    fn run_section_ingest_until_budget(&mut self) {
        self.finalize_done_section_ingests();
    }

    /// Main scheduler loop. Runs on the scheduler thread until shutdown.
    ///
    /// Each iteration runs the unified wave step: decode (clipped to `WAVE_SLICE`)
    /// co-batches the creeping prefill/section cohort and deferred glue into its
    /// sweep, and when no decode is active the same step advances the creep on its
    /// own. The wider phase is dispatched first so freshly-arrived prefills don't
    /// wait a whole decode slice when there's no decode work yet.
    pub fn run(&mut self) {
        tracing::info!("scheduler started");
        // One-time snapshot of the governor's budget partition (capacity C, KV
        // floor, ladder thresholds, per-class reserved, live headroom) so a run's
        // starting VRAM state is visible in the log before any waves.
        if let Some(gov) = self.session.vram_governor() {
            gov.log_budget("startup");
        }

        loop {
            // A mid-wave admission drain inside the decode quantum can consume the
            // shutdown request (so the top-of-loop drain below can't see it) — break
            // on the recorded intent before blocking on the idle recv.
            if self.shutdown_requested {
                break;
            }
            // 1. Drain pending submissions (non-blocking). This synchronously
            // handles SubmitTurn (projection + elevate + apply_segments gap-fill
            // + view create) on the scheduler thread — a prime suspect for the
            // wall-clock that is NOT a forward, so time it.
            let t_drain = Instant::now();
            // Flag the drain so the assembler's shared sub-timers (inject / prefill
            // / gap-fill, also used by reproject) attribute to the drain buckets,
            // and so `apply_projection` DEFERS its gap-fills into one batched forward.
            IN_DRAIN.store(true, std::sync::atomic::Ordering::Relaxed);
            self.batch_drain_gap_fills = true;
            let cont = self.drain_submissions();
            // The drain's deferred gap-fills are NOT fired here. They are a pure
            // K/V scatter whose ingest content prefills through a separate unit
            // later, so the unified wave step consumes them (`take_wave_glue`) and
            // co-batches the scatter into decode's sweep as a full-sweep member —
            // one forward instead of a separate drain launch. See
            // `decode_forward_cobatched`.
            self.batch_drain_gap_fills = false;
            IN_DRAIN.store(false, std::sync::atomic::Ordering::Relaxed);
            self.wave_stats
                .add_phase(WavePhase::Drain, t_drain.elapsed().as_millis() as u64);
            if !cont {
                break; // Shutdown requested or channel closed.
            }

            // 1b. Drain any background-compression VRAM-starvation signal. A
            // persistence hot→warm compress-to-free that couldn't allocate its
            // quant arena leaves its turn hot-float + consistent (retried next
            // pass) but signals the governor; escalate recovery here so that retry
            // has room, before we admit more prefills that would tighten VRAM
            // further. Cheap atomic swap in the common (no-starvation) case.
            let starved = self
                .session
                .vram_governor()
                .map(|g| g.take_starvation())
                .unwrap_or(0);
            if starved > 0 {
                self.relieve_compression_starvation(starved);
            }

            // 2. Promote queued PrefillWork → ActivePrefill (up to cap).
            let t_promote = Instant::now();
            self.promote_new_prefills();
            self.wave_stats
                .add_phase(WavePhase::Promote, t_promote.elapsed().as_millis() as u64);

            // 3. If idle, block waiting for work. Deferred glue counts as work: the
            // unified wave step scatters it (`take_wave_glue`), so don't block while
            // any is pending or it would never be consumed.
            if self.active_decodes.is_empty()
                && self.active_prefills.is_empty()
                && self.prefill_queue.is_empty()
                && self.active_section_ingests.is_empty()
                && self.deferred_glue_fires.is_empty()
            {
                // Time ONLY the recv block (not the request handling) — this is the
                // scheduler idle between requests, attributed to the Idle phase so it
                // isn't mislabeled as Blocked in the GUI.
                let t_idle = Instant::now();
                let req = match self.rx.recv() {
                    Ok(req) => req,
                    Err(_) => break, // Engine dropped.
                };
                self.wave_stats
                    .add_idle(t_idle.elapsed().as_millis() as u64);
                if !self.handle_request(req) {
                    break;
                }
                continue;
            }

            // 4. Always run all quanta each iteration; order by current width.
            // Summary decodes are excluded from the flip count so they never
            // hold the loop in decode-first mode at the expense of prefills.
            // Fresh per wave: the prefill cohort is advanced ONCE — folded into the
            // first co-batched decode step if decode runs and the window is at a
            // sweep boundary, else by the interleaved prefill pass. The guard is
            // reset here (before both quanta) so its meaning is per-wave regardless
            // of the width-ordered dispatch below.
            self.wave_cohort_advanced = false;
            self.wave_section_advanced = false;
            let dw = self.foreground_decode_width();
            let pw = self.prefill_width();
            let sw = self.section_ingest_width();
            if dw >= pw.max(sw) {
                self.timed_decode();
                self.timed_prefill();
                self.timed_section();
            } else if sw >= pw {
                self.timed_section();
                self.timed_prefill();
                self.timed_decode();
            } else {
                self.timed_prefill();
                self.timed_section();
                self.timed_decode();
            }

            // Drain prefills that reached the head this wave — whether they finished
            // inside the co-batched DECODE sweep (`decode_width() > 0`) or the
            // decode-less prefill sweep. This MUST run every wave, unconditionally:
            // a finished prefill has `offset == tokens.len()` so it drops out of
            // `prefill_width()`, and gating the drain behind the prefill quantum's
            // early-returns (decode active, or `prefill_width() == 0` once the last
            // in-flight prefill completes) strands finished prefills in
            // `active_prefills` — blocking new admissions and stalling ingest with
            // "no forwards". Idempotent and cheap when nothing finished.
            self.promote_finished_prefills_to_decodes();

            // Per-wave ingest backpressure + gentle demote. These are cheap (an
            // atomic backlog read + a bounded warm-backed LRU walk) and self-gate on
            // `ingest_timelines` (a no-op when not ingesting), so they run EVERY wave
            // — not on the 2 s telemetry cadence like the footprint defrag below.
            // CFW's co-batched wave folds a wide prefill cohort into every forward,
            // so KV grows far faster per wave than the serial passes it replaced; a
            // 2 s-cadence throttle lets `used` overshoot massively between ticks (the
            // leak-like climb) and lets ingest outrun the hot→warm drain into
            // warm-starvation, where the demote finds nothing warm-backed to shed.
            // Sizing the admission window to the drain backlog and shedding the
            // warm-backed tail every wave bounds KV production to the drain rate, so
            // `used` holds at the demote watermark instead of climbing.
            self.regulate_ingest_admission();
            self.demote_cold_ingest_if_pressured();

            // AIMD reopen (non-ingest): if a prior pressure episode narrowed the
            // admission window, probe it back open by one slot per loop once VRAM
            // is no longer under pressure — gradual recovery so we don't snap to
            // full width and re-trip on the next wide wave. Gated on the window
            // being closed so the steady state never pays the VRAM query. Ingest
            // is excluded here: its window is driven from the drain backlog at the
            // wave cadence below (`regulate_ingest_admission`), and a per-loop
            // reopen would fight that throttle.
            if self.admit_window < Self::MAX_PREFILL_WIDTH
                && self.ingest_timelines.is_empty()
                && !self.vram_under_pressure()
            {
                self.grow_admit_window();
            }

            // Flush the wave summary + phase breakdown if its 2 s window
            // elapsed — even when no forward ran this iteration, so stalls still
            // surface their phase split. (The expert-DMA delta and cumulative
            // op-profile dumps were removed once measurement ruled the expert
            // path out — dma_loads stays 0; the prefill cost is the attention
            // kernel, seen in the per-forward `code-read prefill` breakdown.)
            if self.wave_stats.due() {
                // Our eviction gate's own view of VRAM: the pool budget we
                // defend (vram_budget_available) and pool_used — queried only
                // on the wave we emit, not every iteration.
                let kv_vram = self
                    .session
                    .vram_budget_available()
                    .zip(self.session.vram_pool_stats())
                    .map(|(budget, (used, _reserved))| (budget, used));
                let backlog = self.pending_prefill_tokens();
                // Resident-arena format split for the arena panel. `mem_get_info`
                // returns bytes → convert to MiB here so the ring fields match
                // their `_mib` names (and the dashboard's GiB scale).
                let mib = |b: usize| (b >> 20) as u64;
                let fmt = self.session.kv_gpu_format_stats().map(|fs| {
                    (
                        fs.float_arenas as u32,
                        mib(fs.float_reserved_bytes),
                        mib(fs.float_live_bytes),
                        fs.quant_arenas as u32,
                        mib(fs.quant_reserved_bytes),
                        mib(fs.quant_live_bytes),
                    )
                });
                // Whole-card VRAM decomposition: KV-pool reserved footprint +
                // driver total/free (`cuMemGetInfo`). `0` when unavailable
                // (non-CUDA) — the panel then shows an empty decomposition.
                let reserved_mib = self
                    .session
                    .vram_pool_stats()
                    .map(|(_, r)| mib(r))
                    .unwrap_or(0);
                let (total_mib, free_mib) = self
                    .session
                    .vram_free_total()
                    .map(|(f, t)| (mib(t), mib(f)))
                    .unwrap_or((0, 0));
                // Live resident model-weight VRAM: fixed base weights + the
                // resident-expert footprint, which rises/falls as MoE experts page
                // VRAM↔pinned RAM. Sampled from the model each wave so the weights
                // band tracks the real (time-varying) footprint rather than a
                // static startup snapshot. `0` when the model can't report it.
                let weights_mib = self.model.resident_weight_bytes().map(mib).unwrap_or(0);
                // Active-work counts at this wave for the slots panel.
                let slots = (
                    self.slot_conversations.len() as u32,
                    self.decode_width() as u32,
                    self.prefill_width() as u32,
                    self.section_ingest_width() as u32,
                );
                self.wave_stats.flush(
                    kv_vram,
                    backlog,
                    fmt,
                    (reserved_mib, total_mib, free_mib, weights_mib),
                    slots,
                );
                // Return freed KV VRAM to the OS every wave. FIRST release
                // now-empty arenas: compression (hot→warm) and eviction leave
                // arenas fully free but still *reserved*, and the async pool never
                // reclaims them on its own — so without this they pile up (~14 GiB
                // / 871 arenas observed) and drive the driver's real free toward
                // zero. When real free craters, a wide prefill's CONTIGUOUS
                // transient activation peak — which the scattered pool free-list
                // (counted in `vram_budget_available`) can't satisfy — spills to
                // host memory, a multi-second stall the budget never saw coming.
                // THEN trim the pool so `pool_reserved` (the card's real
                // footprint) tracks `pool_used`. Sweeping every wave keeps real
                // OS-free healthy; cheap and KV-preserving (only fully-empty
                // arenas), i.e. the relief ladder's Trivial rung run ahead of the
                // pressure instead of reactively after a forward already stalled.
                let swept = self.session.release_empty_arenas().unwrap_or(0);
                if swept > 0 {
                    relief_trace::note("sched", "arena_sweep", swept as u64, 0);
                    tracing::debug!(
                        target: "candle_conversation::scheduler::vram_relief",
                        arenas_swept = swept,
                        "proactive empty-arena sweep (per-wave)"
                    );
                }
                self.trim_kv_pool();
                // NOTE: the gentle-early ingest demote + admission backpressure now
                // run PER-WAVE (above), not on this 2 s cadence — they're cheap and
                // must track CFW's fast per-wave KV growth to hold `used` at the
                // demote watermark. Only the expensive footprint defrag stays here.
                // Footprint reclaim: defrag the fragmented reserved gap when it
                // nears capacity (so a wide forward's transient peak can't push the
                // card into WDDM paging), and bulk-evict resident KV only when
                // `used` itself nears capacity. No-op (cheap stats read) when
                // comfortably below both watermarks.
                // (Any KV eviction inside reclaim is accounted at the
                // `evict_cold_tail` chokepoint; the defrag/trim bytes it also sheds
                // are pool-footprint reclaim, not eviction, so they aren't counted
                // as eviction volume here.)
                self.reclaim_footprint();
                // Last resort under heavy backlog: block the wave loop on a
                // device sync so ingest stops outrunning the drain and the
                // primary stream empties — letting the (short, batched) hot→warm
                // pass run uncontended. Fires only well above the throttle
                // target; no-op otherwise.
                self.sync_if_backlog_critical();
            }

            // Livelock guard. If this wave had NO runnable forward work of any class,
            // yet the idle `recv` above did not block (some queue is non-empty), then
            // work exists that this thread cannot clear by spinning. During ingest a
            // hot→warm backlog clamps `admit_window` to zero (`regulate_ingest_admission`),
            // so queued prefills/sections can't be admitted and every width reads 0.
            // Busy-spinning here burns a core AND continuously re-takes the conversation
            // read lock, starving the persistence thread's install-warm WRITE lock — so
            // the backlog can never drain, the gate never reopens, and the spin never
            // ends. That is the observed deadlock: wave stuck 100% "blocked", backlog
            // frozen, no hot→warm pass completing. A short sleep yields the uncontended
            // lock window the drain needs and re-checks admission on the next wave;
            // fresh submissions are still picked up by `drain_submissions` at the top.
            // (On win32 the 2ms rounds up to the ~15ms scheduler timer granularity —
            // ~60 Hz re-check, still far below a busy-spin, which is the point.)
            if self.decode_width() == 0
                && self.prefill_width() == 0
                && self.section_ingest_width() == 0
                && self.deferred_glue_fires.is_empty()
            {
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
        }

        tracing::info!("scheduler shut down");
    }

    /// Run the decode quantum, attributing its wall-clock to [`WavePhase::Decode`]
    /// (the reprojection drain inside it is separately attributed to
    /// [`WavePhase::Reproject`] as a sub-slice).
    fn timed_decode(&mut self) {
        let t = Instant::now();
        self.run_decode_until_budget();
        self.wave_stats
            .add_phase(WavePhase::Decode, t.elapsed().as_millis() as u64);
    }

    /// Run the prefill quantum, attributing its wall-clock to [`WavePhase::Prefill`].
    fn timed_prefill(&mut self) {
        let t = Instant::now();
        self.run_prefill_until_budget();
        self.wave_stats
            .add_phase(WavePhase::Prefill, t.elapsed().as_millis() as u64);
    }

    /// Run the section-ingest quantum, attributing its wall-clock to
    /// [`WavePhase::Section`].
    fn timed_section(&mut self) {
        let t = Instant::now();
        self.run_section_ingest_until_budget();
        self.wave_stats
            .add_phase(WavePhase::Section, t.elapsed().as_millis() as u64);
    }

    /// Trim the CUDA pool's reserved-but-free fragmentation back to the OS,
    /// keeping a slack floor of `pool_used + TRIM_SLACK` reserved so the next
    /// allocations reuse pool memory instead of re-hitting the OS.
    ///
    /// Keeps `pool_reserved` (the card's physical footprint) tracking
    /// `pool_used` (what [`vram_budget_available`] measures), so the budget
    /// never under-reads the true occupancy and lets KV oversubscribe VRAM.
    /// No-op on non-CUDA / when the pool allocator is unavailable.
    ///
    /// [`vram_budget_available`]: candle_nn::kv_cache::vram_budget_available
    pub(super) fn trim_kv_pool(&self) {
        let Some((used, reserved)) = self.session.vram_pool_stats() else {
            return;
        };
        let mib = |b: usize| b / (1024 * 1024);
        // Slack floor of ready blocks retained so a single wave's seal/realloc
        // churn reuses pool memory rather than re-allocating from the OS every
        // trim. Default 2 GiB; override with `CANDLE_KV_POOL_TRIM_SLACK_MB`.
        let slack = std::env::var("CANDLE_KV_POOL_TRIM_SLACK_MB")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(2048)
            .saturating_mul(1024 * 1024);
        let keep = used.saturating_add(slack);
        // Diagnostic: surface the pool's true physical footprint every wave so we
        // can watch `reserved` track (or diverge from) `used` — the number the
        // VRAM budget can't see. Once per ~2 s, so cheap.
        tracing::debug!(
            "kv-pool: used={}MiB reserved={}MiB gap={}MiB keep={}MiB",
            mib(used),
            mib(reserved),
            mib(reserved.saturating_sub(used)),
            mib(keep),
        );
        // Split the resident GPU arenas into float (the live decode/prefill
        // working set + not-yet-compressed completed turns — reusable, and what
        // the compress-to-free rung shrinks) vs quant (sealed attended-over
        // context, held). Its own line: these are GidPool arena-slab bytes, NOT a
        // partition of the CUDA-pool `gap` above (that also holds segment slack
        // the GidPool never sees). Watch `float` fall across a pressure episode
        // to confirm compress is bringing quantization forward.
        if let Some(fs) = self.session.kv_gpu_format_stats() {
            tracing::debug!(
                "kv-pool fmt: float={}arenas/{}MiB (live {}MiB) quant={}arenas/{}MiB (live {}MiB)",
                fs.float_arenas,
                mib(fs.float_reserved_bytes),
                mib(fs.float_live_bytes),
                fs.quant_arenas,
                mib(fs.quant_reserved_bytes),
                mib(fs.quant_live_bytes),
            );
        }
        // Nothing to reclaim if the pool is already within the slack floor.
        if reserved <= keep {
            return;
        }
        // Best-effort return of whole free segments to the OS. The pool usually
        // can't release much — freed chunks are scattered inside partially-used
        // segments — but that reserved-but-free memory is REUSABLE by new KV with
        // no new OS allocation, and `vram_budget_available` now counts it, so a
        // large `gap` is not a problem to chase. No sync here (it reclaimed 0
        // anyway; the memory is fragmented, not merely pending-free).
        if let Some((before, after)) = self.session.trim_kv_pool(keep) {
            let freed = before.saturating_sub(after);
            if freed > 0 {
                relief_trace::note("sched", "pool_trim", before as u64, after as u64);
            }
            tracing::debug!(
                "trimmed KV pool: reserved {}MiB -> {}MiB (freed {}MiB, kept used {}MiB + slack)",
                mib(before),
                mib(after),
                mib(freed),
                mib(used),
            );
        }
    }
}
