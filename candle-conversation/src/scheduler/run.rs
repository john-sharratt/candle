use super::admission::{admit_quantum, budget_notches, evidence_ticks_for, ThrottleReason};
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
    /// `admit_budget`/VRAM cap, plus the per-wave ingest throttle + gentle demote.
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
            // pass) but signals the governor; make room here so that retry has
            // some, before we admit more prefills that would tighten VRAM
            // further. Cheap atomic swap in the common (no-starvation) case.
            //
            // Starvation used to get its own escalated recovery path — a
            // footprint reclaim, then a bulk eviction that overrode the "only
            // evict when `used` is high" watermark. It needed the override
            // because the watermark was a guess about the card; the free-region
            // count is not, and a compressor that could not get an arena is
            // exactly the state the ordinary pressure signal reports. So it
            // takes the ordinary path, at the load setpoint.
            let starved = self
                .session
                .vram_governor()
                .map(|g| g.take_starvation())
                .unwrap_or(0);
            if starved > 0 {
                tracing::warn!(
                    target: "candle_conversation::scheduler::vram_relief",
                    starvation_events = starved,
                    "background compression starved of VRAM"
                );
                self.relieve_vram_pressure("starvation", VramPhase::Load);
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

            // AIMD reopen (non-ingest): if a prior pressure episode cut the
            // admission budget, probe it back open by one quantum per loop once
            // VRAM is no longer under pressure — gradual recovery so we don't
            // snap to full width and re-trip on the next wide wave. Gated on the
            // budget being below the live ceiling so the steady state never pays
            // the VRAM query twice. Ingest is excluded here: its budget is driven
            // from the drain backlog at the wave cadence below
            // (`regulate_ingest_admission`), and a per-loop reopen would fight
            // that throttle.
            //
            // Under CHRONIC nominal pressure (a card whose steady-state
            // availability sits just under the band — e.g. the expert-resident
            // budget leaves KV a couple hundred MiB short of it), the
            // pressure-clear path never fires and the budget wedges at the
            // floor, serializing e.g. section calibration into single-sequence
            // mini-forwards. The evidence path reopens it anyway: every check
            // that finds NEW prefill tokens forwarded OOM-free counts one
            // streak tick (idle loop iterations neither count nor reset — the
            // loop spins far faster than forwards complete), and a full streak
            // grows the budget one quantum. Any cut (real OOM, eviction
            // survival) resets the streak — multiplicative decrease still wins.
            if self.ingest_timelines.is_empty() && self.admit_budget < Self::max_admit_budget() {
                if !self.vram_under_pressure() {
                    self.admit_grow_streak = 0;
                    self.raise_admit_budget(ThrottleReason::Throughput);
                } else {
                    // A tick requires a real VOLUME of new tokens, not just
                    // any completion: three 25-token interactive turns are not
                    // evidence that a wider budget survives this pressure.
                    // Small forwards accumulate toward the floor rather than
                    // being discarded (`admit_ok_tokens_seen` advances only
                    // when a tick fires).
                    let ok = PREFILL_OK_TOKENS.load(std::sync::atomic::Ordering::Relaxed);
                    if ok >= self.admit_ok_tokens_seen + EVIDENCE_MIN_PREFILL_TOKENS {
                        self.admit_ok_tokens_seen = ok;
                        self.admit_grow_streak += 1;
                        let need =
                            evidence_ticks_for(budget_notches(self.admit_budget, admit_quantum()));
                        if self.admit_grow_streak >= need {
                            self.admit_grow_streak = 0;
                            self.raise_admit_budget(ThrottleReason::Throughput);
                        }
                    }
                }
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
                // Resident-arena occupancy for the arena panel, split small vs
                // large half of the size-class ladder. `mem_get_info` returns
                // bytes → convert to MiB here so the ring fields match their
                // `_mib` names (and the dashboard's GiB scale).
                //
                // The panel used to split float vs quant, which is not a
                // question an arena can answer any more — it holds whatever
                // fits its slots. The ladder split carries the same signal:
                // compression moves occupancy DOWN the ladder, so a working
                // compress-to-free rung shows the large half falling.
                let mib = |b: usize| (b >> 20) as u64;
                let fmt = self.session.kv_gpu_class_stats().map(|cs| {
                    let half = cs.classes.len() / 2;
                    let sum = |rows: &[candle_nn::kv_cache::ClassOccupancy]| {
                        rows.iter().fold((0usize, 0usize, 0usize), |acc, c| {
                            (
                                acc.0 + c.arenas,
                                acc.1 + c.reserved_bytes,
                                acc.2 + c.live_bytes,
                            )
                        })
                    };
                    let (l_arenas, l_res, l_live) = sum(&cs.classes[half..]);
                    let (s_arenas, s_res, s_live) = sum(&cs.classes[..half]);
                    (
                        l_arenas as u32,
                        mib(l_res),
                        mib(l_live),
                        s_arenas as u32,
                        mib(s_res),
                        mib(s_live),
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
                // Same cadence: publish the full memory report (global slot for
                // `GET /v1/memory` + one JSON debug line). See `memory_report`.
                self.publish_memory_report();
                // Return emptied regions to the free list. Compression
                // (hot→warm) and eviction free chunks scattered across arenas,
                // and an arena only gives its region back once its last chunk
                // goes — so nothing surfaces without a sweep looking for it.
                // Running it here, ahead of pressure, is what keeps the
                // free-region count honest: the setpoint is compared against
                // regions that are genuinely claimable, not against a count
                // that would only be right after the next relief pass.
                let swept = self.session.release_empty_arenas().unwrap_or(0);
                if swept > 0 {
                    relief_trace::note("sched", "arena_sweep", swept as u64, 0);
                    tracing::debug!(
                        target: "candle_conversation::scheduler::vram_relief",
                        arenas_swept = swept,
                        "proactive empty-arena sweep (per-wave)"
                    );
                }
                self.log_kv_memory();
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
            // hot→warm backlog cuts `admit_budget` to the floor (`regulate_ingest_admission`),
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

    /// Publish the KV memory picture at the wave loop's slow cadence: the CUDA
    /// pool, the reservation's regions, the transient domains, and the
    /// slot-state slabs.
    ///
    /// This used to also *trim* the pool — return its reserved-but-free
    /// fragmentation to the OS, keeping a `pool_used + slack` floor — and every
    /// relief rung called it after freeing anything. That mattered when KV was
    /// pool memory: freed arenas stayed reserved, `pool_reserved` climbed away
    /// from `pool_used`, and the driver's real free went to zero. None of it
    /// applies now. KV comes from the reservation, so releasing an arena moves a
    /// region between two lists and changes no pool accounting at all; what is
    /// left in the pool is the model, the expert cache and the few remaining
    /// grow-only scratches, which reach their size and stay there.
    /// `cuMemPoolTrimTo` on that is work with nothing to reclaim — and it
    /// synchronously unmaps, which is why it needed a guard against in-flight
    /// kernels holding captured pointers. Dropping the trim drops the guard.
    pub(super) fn log_kv_memory(&self) {
        let mib = |b: usize| b / (1024 * 1024);
        if let Some((used, reserved)) = self.session.vram_pool_stats() {
            // What is left of the pool once KV moved out. Flat is the healthy
            // shape: growth here means something outside the reservation is
            // still allocating per-wave.
            tracing::debug!(
                "kv-pool: used={}MiB reserved={}MiB gap={}MiB",
                mib(used),
                mib(reserved),
                mib(reserved.saturating_sub(used)),
            );
        }
        // Resident GPU arena occupancy, one line per occupied size class. Watch
        // occupancy move from the large classes to the small ones across a
        // pressure episode to confirm compress-to-free is bringing
        // quantization forward.
        if let Some(cs) = self.session.kv_gpu_class_stats() {
            let rows: Vec<String> = cs
                .classes
                .iter()
                .filter(|c| c.arenas > 0)
                .map(|c| {
                    format!(
                        "{}B={}a/{}MiB(live {}MiB)",
                        c.slot_bytes,
                        c.arenas,
                        mib(c.reserved_bytes),
                        mib(c.live_bytes)
                    )
                })
                .collect();
            tracing::debug!("kv-pool classes: {}", rows.join(" "));
        }
        // The reservation's KV side. `free` is the pressure signal admission
        // reads; `peak_live` against `total` says how close the startup
        // partition came to binding, which is what step 7 tunes.
        if let Some(r) = candle_nn::kv_cache::region_stats(0) {
            tracing::debug!(
                "kv-regions: live={} peak={} free={} of {} ({}MiB) | transient carved={}MiB of {}MiB",
                r.live,
                r.peak_live,
                r.free,
                r.total,
                mib(r.total * candle_nn::kv_cache::REGION_BYTES),
                mib(r.transient_carved),
                mib(r.transient_bytes),
            );
        }
        // Per-domain transient peaks, the terms the transient tier is sized
        // from: `S = 2*W_wave + W_persist + shelf`.
        if let Some((cursor, peak, cap)) = candle_nn::kv_cache::persistence_domain_stats(0) {
            if peak > 0 {
                tracing::debug!(
                    "kv-transient persist: cursor={}MiB peak={}MiB cap={}MiB",
                    mib(cursor),
                    mib(peak),
                    mib(cap),
                );
            }
        }
        // The `W_wave` term of the same equation. Both halves, because the span
        // has to hold the larger of the two.
        if let Some(halves) = candle_nn::kv_cache::wave_domain_stats(0) {
            let peak = halves.iter().map(|h| h.1).max().unwrap_or(0);
            if peak > 0 {
                tracing::debug!(
                    "kv-transient wave: peak={}MiB (a={}MiB b={}MiB) cap={}MiB each",
                    mib(peak),
                    mib(halves[0].1),
                    mib(halves[1].1),
                    mib(halves[0].2),
                );
            }
        }
        // Slot-state slabs: the region tier's answer to audit A13. `slabs`
        // settling and then staying flat while sequences deepen is the evidence
        // that the decode path has stopped calling the allocator — it used to
        // free and re-allocate this buffer every time a sequence crossed a
        // 32-token boundary, per layer.
        let (live, slabs, bytes) = candle_nn::kv_cache::slot_state_stats();
        if slabs > 0 {
            tracing::debug!(
                "kv-slotstate: live={live} slabs={slabs} reserved={}MiB promotions={}",
                mib(bytes),
                candle_nn::kv_cache::class_promotion_count(),
            );
        }
    }
}
