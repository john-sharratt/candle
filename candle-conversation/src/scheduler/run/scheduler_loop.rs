//! The scheduler's main loop: one wave per iteration, with admission, the
//! weight side's settling, and the idle wait.
//!
//! Each iteration runs the unified wave step, which decode drives under its
//! slice and which carries the prefill cohort and deferred glue in its sweep. The
//! iteration drains submissions, promotes queued prefills, and then either runs
//! the quanta or blocks on its channel, so work that is admissible is never
//! waited on and work that is not waits for the thing that could change it. When
//! the loop falls idle after work, the weight side settles so that what the work
//! freed is handed back before the loop parks. An expert pipeline that has exited
//! ends the process, because the supervisor restarts the engine whole, and no
//! forward can succeed without the pipeline.

use super::*;

impl Scheduler {
    /// Main scheduler loop. Runs on the scheduler thread until shutdown.
    ///
    /// Each iteration runs the unified wave step: decode (clipped to `WAVE_SLICE`)
    /// co-batches the creeping prefill/section cohort and deferred glue into its
    /// sweep, and when no decode is active the same step advances the creep on its
    /// own. The wider phase is dispatched first so freshly-arrived prefills don't
    /// wait a whole decode slice when there's no decode work yet.
    pub fn run(&mut self) {
        tracing::info!("scheduler started");
        // This thread owns the device's KV allocations and compacts them, so
        // hot KV let go of anywhere else is handed here to be freed
        // (`substrate::residence`).
        own_device_frees();
        // **A seed, deliberately not the measurement.**
        //
        // The figure is exact only with nothing in flight, and the idle branch
        // below is where that is true — every region live there is permanent
        // (the system prompt, the tool catalog, the resident corpus) and none of
        // it is the wave's to give back. Here at entry none of that has been
        // prefilled yet, so this reads a near-empty card and over-states what the
        // zone can reach; the hold taken from it then defends a residency the
        // catalog makes unreachable, and every offer refuses on the floor.
        //
        // It is still taken, because the alternative is worse: nothing else
        // writes the static, so an engine that never reaches the idle branch —
        // a sustained ingest, which never empties all five queues at once —
        // would read `achievable_weight_now() == None`, a hold of zero, and
        // admission defending nothing at all. A first pass that is too tight
        // self-corrects at the first idle; one that defends nothing does not.
        super::super::interleave::reseed_achievable_weight();
        // **The weight side leaves what relief frees to.** Relief defends
        // `LOAD_TARGET` free regions by conceding weight ground; a weight side that
        // grew back below it would take those regions again on the next forward,
        // and the two would trade them every step.
        if let DeviceLocation::Cuda { gpu_id } = self.session.device().location() {
            set_kv_free_target(gpu_id, LOAD_TARGET);
        }
        // One-time snapshot of the governor's budget partition (capacity C, KV
        // floor, ladder thresholds, per-class reserved, live headroom) so a run's
        // starting VRAM state is visible in the log before any waves.
        if let Some(gov) = self.session.vram_governor() {
            gov.log_budget("startup");
        }
        // Watch every allocation that reaches the driver instead of one of the
        // reservation's allocators, for as long as the engine serves. Armed once
        // and never dropped inside the loop: a window around the quanta alone
        // missed the housekeeping between them, request handling, and every
        // thread whose work did not happen to overlap it — the persistence
        // thread's sealing, above all. Each wave drains what accumulated since
        // the last into one report. A ZST guard without the feature.
        #[cfg(feature = "forbidden_allocations")]
        let _alloc_watch = candle::forbidden_alloc::armed();

        loop {
            // A mid-wave admission drain inside the decode quantum can consume the
            // shutdown request (so the top-of-loop drain below can't see it) — break
            // on the recorded intent before blocking on the idle recv.
            if self.shutdown_requested {
                break;
            }
            // **An expert pipeline that has exited ends the process.** Neither its
            // thread nor its stager is restarted, so every forward after it fails:
            // the daemon kept answering every request, NPC and chat alike, with
            // "expert stager died — channel closed" and served nothing until
            // someone restarted it by hand. Exiting hands the restart to the
            // supervisor (keep), which brings the engine back whole.
            if self.model.expert_pipeline_dead() {
                tracing::error!(
                    "the expert pipeline has exited and cannot be restarted in place; \
                     ending the process so its supervisor starts it again"
                );
                exit(ENGINE_DEAD_EXIT);
            }
            // The top of the pass: a wave's wall-clock runs from here.
            let t_pass = Instant::now();
            // 1. Drain pending submissions (non-blocking). This synchronously
            // handles SubmitTurn (projection + elevate + apply_segments gap-fill
            // + view create) on the scheduler thread — a prime suspect for the
            // wall-clock that is NOT a forward, so time it.
            let _g_drain = profile::span("loop:drain_submissions");
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
            let drain_elapsed = t_drain.elapsed();
            self.step_census
                .spend(super::super::step_census::Activity::Drain, drain_elapsed);
            self.wave_stats
                .add_phase(WavePhase::Drain, drain_elapsed.as_millis() as u64);
            // Non-forward wall clock, and therefore per-forward overhead the planner has
            // to know about — the drain runs once per wave whatever the wave carries.
            self.wave_overhead_us += drain_elapsed.as_micros() as u64;
            _g_drain.end();
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
                self.relieve_vram_pressure(
                    "starvation",
                    VramPhase::Load,
                    super::super::prefill::ReliefNeed::Paced,
                );
            }

            // 1c. **Serve any co-resident model waiting for the card.**
            //
            // Here, and only here. This is the one point in the loop where no
            // forward is in flight and none has been issued for this pass,
            // which is the window a span claim and a boundary move are legal
            // in — `claim_span_region` refuses anywhere inside the quanta,
            // correctly, because the transient tier stands flush against the KV
            // frontier while a wave is open and the ground above it belongs to
            // that wave.
            //
            // Normal inference is blocked for the whole drain. That is the
            // design rather than a cost: the guest stands in ground the KV side
            // was using, and a wave running against regions that now hold a
            // diffusion model's weights would read them as attention state —
            // which does not fault, because every address in the span is
            // mapped, and so surfaces as numbers rather than as an error.
            //
            // Costs one atomic load on the overwhelming majority of passes,
            // where nothing is queued.
            {
                let guests = Arc::clone(&self.guests);
                let t_guests = Instant::now();
                let served = self.drain_guests(&guests);
                self.step_census.spend(
                    super::super::step_census::Activity::Device,
                    t_guests.elapsed(),
                );
                if let Some(report) = served {
                    // The drain evicted, ran and handed its ground back. Skip
                    // straight to the top: the widths this pass measured are
                    // from before the eviction, and forming a wave against them
                    // would admit against a working set that is no longer there.
                    tracing::info!(
                        target: "candle_conversation::guest",
                        served = report.served,
                        failed = report.failed,
                        total_ms = report.total_ms,
                        "normal inference resuming"
                    );
                    continue;
                }
            }

            // 2. Promote queued PrefillWork → ActivePrefill (up to cap).
            {
                let _g = profile::span("loop:promote_new_prefills");
                let t_promote = Instant::now();
                self.promote_new_prefills();
                self.step_census.spend(
                    super::super::step_census::Activity::Promote,
                    t_promote.elapsed(),
                );
                self.wave_stats
                    .add_phase(WavePhase::Promote, t_promote.elapsed().as_millis() as u64);
            }
            // **While prompts wait, their prefill's tier is spoken for.** Told
            // before this pass issues a forward, so the growth negotiation after
            // it leaves that ground standing through the decodes between two
            // prefills, instead of handing it to the weight side and having the
            // next prefill buy it back (`set_prompts_waiting`).
            if let DeviceLocation::Cuda { gpu_id } = self.session.device().location() {
                set_prompts_waiting(gpu_id, self.pending_prefill_tokens() > 0);
            }

            // 2a. One background scan (see `norm_warm`): a warm-up's replayed
            // turn or ingest slice, or a group-vote scan — its own band.
            let t_warm = Instant::now();
            let ran = self.step_norm_warm();
            let took = t_warm.elapsed();
            self.step_census
                .spend(super::super::step_census::Activity::Warm, took);
            self.wave_stats.add_warm(took.as_micros() as u64);
            if let Some(kind) = ran {
                self.wave_stats.warm_split.add(kind, took);
            }

            // 2b. A turn whose prefill stands still is told what it waits
            // behind, so its caller can show it rather than look hung
            // (`prefill_wait`).
            self.timed_activity(
                super::super::step_census::Activity::Waits,
                Self::report_prefill_waits,
            );

            // 2c. An idle slot whose last seal is durable gives its recurrent
            // store back (`recurrent_park`).
            let t_park = Instant::now();
            self.park_durable_states();
            self.step_census
                .spend(super::super::step_census::Activity::Park, t_park.elapsed());

            // 3. If idle, block waiting for work. Deferred glue counts as work: the
            // unified wave step scatters it (`take_wave_glue`), so don't block while
            // any is pending or it would never be consumed.
            if self.active_decodes.is_empty()
                && self.active_prefills.is_empty()
                && self.prefill_queue.is_empty()
                && self.pending_submits.is_empty()
                && self.section_queue.is_empty()
                && self.active_section_ingests.is_empty()
                && self.deferred_glue_fires.is_empty()
            {
                // **Settle once on falling idle after work.** The wave loop packs
                // and grows the weight side a little per iteration, paced against
                // the forwards beside it; with none left, what the finished work
                // freed is handed back whole before the loop parks — otherwise it
                // waits, owed, inside the next request's first forwards.
                //
                // **And again whenever ground has freed since.** Conversations
                // release, stores park and hot copies demote on their own clocks
                // after the last forward; what they free stands above the frontier
                // a settle on falling idle already packed (`settle::resettle_owed`).
                // Read every wake of the idle wait — a request or a rates period.
                if let Some(stats) = self.session.kv_region_stats() {
                    let now = SettledGround {
                        live: stats.live,
                        frontier: stats.live_watermark,
                    };
                    if resettle_owed(self.settled_ground, now) {
                        self.settle_owed = true;
                    }
                }
                if std::mem::take(&mut self.settle_owed) {
                    let t_settle = Instant::now();
                    let report = self.settle_device();
                    self.step_census.spend(
                        super::super::step_census::Activity::Device,
                        t_settle.elapsed(),
                    );
                    for waiter in self.settle_waiters.drain(..) {
                        let _ = waiter.send(report);
                    }
                }
                // Nothing is in flight, so everything resident is permanent:
                // the one moment the achievable weight residency is exact.
                super::super::interleave::reseed_achievable_weight();
                tracing::trace!("idle; waiting for a request");
                // The wait is counted idle as it runs, not the request handling
                // after it — so it isn't mislabeled as Blocked in the GUI — and it
                // closes the wave window each time one falls due (`recv_idle`).
                //
                // A queued warm-up is work too: wait only as long as it allows,
                // then go round to score its next replayed turn. So is a guest
                // job waiting for room — the drain at the top of the loop is
                // what tries it again, and nothing arrives to wake it.
                let wait = match (self.norm_warm_wait(), self.guests.queue.retry_wait()) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
                let req = if let Some(wait) = wait {
                    match self.recv_idle(wait) {
                        Ok(req) => req,
                        Err(RecvTimeoutError::Timeout) => continue,
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                } else {
                    // **Priced while idle too.** Its reader takes the rates every
                    // period whether or not this member runs anything, and an idle
                    // member that only priced inside its quanta published what a
                    // loaded run last left it — WhiteWorld's 208 ms step for a whole
                    // run of 87 turns it was never placed (ed4d8c781, 2026-10-20).
                    // So the wait ends each period to price the state it is in.
                    let period = super::super::phase_ring::rates_read_period()
                        .unwrap_or(super::super::prefill::RATES_LOG_EVERY);
                    match self.recv_idle(period) {
                        Ok(req) => req,
                        Err(RecvTimeoutError::Timeout) => {
                            if self.wave_rate.is_none() {
                                self.wave_rate = self.take_planner();
                            }
                            // **The zone grows back while idle too.** Nothing
                            // forwards to ask for the ground the KV side is not
                            // using, so an idle card stood at what its last wave
                            // left it: WhiteWorld (71b3b203a, 2026-10-20) a third
                            // of its experts resident with nothing running, and
                            // one decode more priced at 177 ms. Between forwards
                            // is where the boundary may move, and this wake comes
                            // once a rates period at most — a request ends the
                            // wait without it — so the boundary is not asked to
                            // move faster than the rates are read.
                            self.model.reclaim_spare_ground();
                            // Priced as the idle card it is: the residency the
                            // zone holds now, nothing charged (`settle_idle`).
                            if let (Some(rate), Some(zone)) = (
                                self.wave_rate.as_mut(),
                                super::super::interleave::effective_weight_zone_bytes(0),
                            ) {
                                rate.settle_idle(zone);
                            }
                            self.publish_learned_rates();
                            continue;
                        }
                        Err(RecvTimeoutError::Disconnected) => break,
                    }
                };
                // The handling itself is Requests, not Idle — it is work, and
                // leaving it untimed put it in the Blocked remainder.
                let t_req = Instant::now();
                let keep_going = self.handle_request(req);
                self.step_census.spend(
                    super::super::step_census::Activity::Requests,
                    t_req.elapsed(),
                );
                self.wave_stats
                    .add_requests(t_req.elapsed().as_micros() as u64);
                if !keep_going {
                    break;
                }
                continue;
            }

            // 3b. **A wave that just failed is not rebuilt at once.** Nothing it
            // stands on has changed yet, so the same wave would fail the same
            // way. Wait out the backoff — a request ends it early and is handled
            // — then try again: a decode-less wave's (`wave_retry`), which is
            // only taken with no decodes running, and a decode step whose tier
            // was refused (`decode_retry`), which leaves its turns where they
            // stood for the next step.
            let decode_less = self
                .wave_retry
                .take_wait()
                .filter(|_| self.decode_width() == 0);
            let owed = decode_less.max(self.decode_retry.take_wait());
            if let Some(wait) = owed {
                let t_wait = Instant::now();
                match self.rx.recv_timeout(wait) {
                    Ok(req) => {
                        self.step_census
                            .spend(super::super::step_census::Activity::Idle, t_wait.elapsed());
                        self.wave_stats
                            .add_idle(t_wait.elapsed().as_millis() as u64);
                        let t_req = Instant::now();
                        let keep_going = self.handle_request(req);
                        self.step_census.spend(
                            super::super::step_census::Activity::Requests,
                            t_req.elapsed(),
                        );
                        self.wave_stats
                            .add_requests(t_req.elapsed().as_micros() as u64);
                        if !keep_going {
                            break;
                        }
                        continue;
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        self.step_census
                            .spend(super::super::step_census::Activity::Idle, t_wait.elapsed());
                        self.wave_stats
                            .add_idle(t_wait.elapsed().as_millis() as u64);
                    }
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }

            // 4. Always run all quanta each iteration; order by current width.
            // Summary decodes are excluded from the flip count so they never
            // hold the loop in decode-first mode at the expense of prefills.
            // Fresh for this pass's prefill quantum, which advances the cohort
            // when nothing decodes. The decode quantum sets the same guards per
            // step, from the airtime decode is owed (`creep_owed_after`).
            self.wave_cohort_advanced = false;
            self.wave_section_advanced = false;
            // A width the partition refused holds only until ground frees.
            self.settle_refused_width();
            // Why the pool was reached, scoped to THIS wave — always on, unlike
            // the site-naming report above. The lifetime totals cannot answer it:
            // they are dominated by declines that are correct (the residual
            // stream has no arena to inherit, model loading has no wave), so only
            // a window in which every allocation *should* inherit separates a
            // provenance break from ordinary work. Two atomic loads.
            let declines = DeclineSnapshot::now();
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
            // Everything from here to the bottom of the iteration is per-wave
            // housekeeping that runs OUTSIDE the five timed quanta: promotion of
            // finished prefills, ingest admission regulation, cold-ingest demote,
            // the AIMD budget walk, and the GPU span harvest. None of it was
            // timed, so all of it fell into the unattributed remainder and drew
            // as "blocked" — which is what made that band large and unexplained.
            let t_house = Instant::now();
            // The whole between-quanta block as one span, so the profile table can say
            // what share of a wave is not the forward at all. Its parts are spanned
            // individually below; this is the total they have to add up to.
            let house_span = super::super::profile::span("loop:housekeeping");
            let (no_ticket, closed) = declines.bytes_since();
            publish_wave_declines(no_ticket, closed);
            // Everything that reached the driver since the previous wave's
            // report, on any thread. Each allocation is memory the reservation
            // was never sized for, and together they are what grows the CUDA
            // pool's reserved footprint under load — the bytes WDDM demotes
            // first.
            #[cfg(feature = "forbidden_allocations")]
            {
                let report = candle::forbidden_alloc::take_report();
                if report.total_calls > 0 {
                    tracing::warn!(
                        target: "candle_conversation::scheduler::forbidden_alloc",
                        calls = report.total_calls,
                        mib = report.total_bytes >> 20,
                        "allocated outside the reservation since the last wave\n{report}"
                    );
                }
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
            {
                let _g = profile::span("loop:promote_finished");
                let t = Instant::now();
                self.promote_finished_prefills_to_decodes();
                HOUSE_PROMOTE_US.fetch_add(
                    t.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            // Prefills still advancing hand rows back to the forward as they go —
            // the standing rows an offer is sized against fall with every sweep —
            // so the pass re-reads them once a quantum though none has finished.
            if !self.active_prefills.is_empty() {
                self.settled_since_admit = true;
            }

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
            {
                let _g = profile::span("loop:ingest_admission");
                let t = Instant::now();
                self.regulate_ingest_admission();
                HOUSE_ADMIT_US.fetch_add(
                    t.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            {
                let _g = profile::span("loop:demote_cold_ingest");
                let t = Instant::now();
                self.demote_cold_ingest_if_pressured();
                HOUSE_DEMOTE_US.fetch_add(
                    t.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }

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

            // Harvest the GPU spans this wave's forwards enqueued. Non-blocking:
            // a pair whose work is still in flight stays pending for a later
            // pass, so this costs a few `cuEventQuery` calls and never stalls
            // the loop. It has to happen SOMEWHERE, though — the events are
            // recorded into the stream by the model and only become numbers at a
            // drain, so without one a daemon opens spans forever, the pool walks
            // up to its high-water mark, and the span that was meant to cost two
            // enqueues starts blocking instead. The bench harness drains at its
            // config boundary; the wave loop is this process's equivalent.
            let t_gpu = Instant::now();
            candle_transformers::models::profile::gpu_drain();
            HOUSE_GPUDRAIN_US.fetch_add(
                t_gpu.elapsed().as_micros() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
            // Free the hot KV other threads let go of since the last wave — an
            // eviction, a tombstone, a demote — here on the device's own thread,
            // where no compaction pass is mid-relocation.
            drain_reclaimed();
            // Closed BEFORE the flush block: the flush is the window boundary
            // itself, and its own cost is already accounted (eviction/sync carve
            // out of the remainder), so folding it in here would double-count.
            self.step_census.spend(
                super::super::step_census::Activity::House,
                t_house.elapsed(),
            );
            let house_us = t_house.elapsed().as_micros() as u64;
            self.wave_stats.add_housekeeping(house_us);
            self.wave_overhead_us += house_us;

            // **Tell the rate planner what this wave cost outside its forwards.**
            //
            // The planner prices width by what width amortises, and it could only see the
            // forward's own duration — so the loop's overhead, which is per forward and
            // does not scale with rows, was worth nothing to it. On a resident card that
            // left *nothing* for width to amortise and the wave stopped widening at 4 rows
            // where the engine wanted 20. See `RateModel::overhead_secs`.
            //
            // Divided by the forwards this wave actually ran, and skipped when it ran
            // none: a wave with no forwards has no per-forward cost, and reporting its
            // whole housekeeping as one forward's overhead would teach the model that a
            // forward costs an idle loop iteration.
            if self.wave_forwards > 0 {
                self.wave_stats
                    .add_wave(t_pass.elapsed().as_micros() as u64, self.wave_overhead_us);
                let per_forward = self.wave_overhead_us as f64 / 1e6 / self.wave_forwards as f64;
                if let Some(rate) = self.wave_rate.as_mut() {
                    rate.observe_overhead(per_forward);
                }
                self.settle_owed = true;
            }
            self.wave_forwards = 0;
            self.wave_overhead_us = 0;

            // **Create what a refused forward asked for — every gap, not every
            // telemetry window.**
            //
            // A forward that needs an arena it could not have pre-claimed is
            // refused outright: it runs on the thread that owns the partition,
            // so there is no "come back later" it can act on. The refusal
            // records the size class, and this is the only thing that acts on
            // that record. The format a seal picks comes from the data it just
            // wrote, so `admit_wave_kv` cannot claim it ahead of time and the
            // demand is not rare.
            //
            // It used to sit inside the 2 s summary block below, which made
            // recovery a side effect of logging: a wave refused an arena, and
            // the turns of the next two seconds were refused the same arena for
            // the same reason before the window came round. Nothing about the
            // creation wants that cadence — it is cheap when there is no demand
            // (one lock, an empty `Vec`), and the wave loop is the gap on every
            // iteration, not one in every few hundred.
            // **Pack the pools, then let the weight side take what packing released.**
            // Both halves, here, in this order, because either alone buys nothing: a
            // frontier that falls with nobody claiming the ground is invisible in
            // every metric except decode, and a weight side asked to grow over a
            // frontier nothing lowered has nothing to take.
            //
            // **Every wave, not once per telemetry window.** It sat inside the 2 s
            // summary block, which made compaction a side effect of logging: the
            // pools fragmented for two seconds between passes, and with the pass
            // itself time-budgeted it never caught up — measured at 48–60% efficiency
            // with every pass clipped. The gate in `compact_kv_if_fragmented` is a
            // cheap counter read and a minimum interval, so being considered here
            // costs the loop nothing on the iterations it declines.
            //
            // Legal here for the same reason `create_deferred_arenas` is: the quanta
            // are done, every forward's guards are dropped, and no wave generation is
            // live — which is the condition `set_weight_floor` itself checks.
            // `compact_kv` refuses rather than corrupts if that ever stops being true.
            //
            // **Arenas first, then compaction.** The arenas a refused pass is owed and
            // the reserves are made protected; compaction's empty sweep, run after,
            // cannot take back what was just made for a demand that has not filled it
            // (`docs/kv_drain_under_pressure.md` §2).
            // **The gap quantize's catch-up comes first** (`docs/kv_quantize_in_the_gap.md`
            // §3): the last gap's converted blocks go in place and its lease goes back
            // to the pools, so the arena creation and compaction below see every slot
            // the lease no longer needs.
            // **Every span below is timed into its band and its census activity**
            // — the gap's own calls into `quantize`, compaction into
            // `compaction` / `compact`, the arena creation and idle demote into
            // `housekeeping` / `compact` — so the phase chart and the census both
            // name them. Untimed, all of it fell into the window's remainder and
            // drew as `blocked`.
            let t_gap = Instant::now();
            match self.session.gap_catch_up() {
                Ok(c) => self.kv_gap.caught_up(&c),
                Err(e) => tracing::warn!(
                    target: "candle_conversation::scheduler::kv_gap",
                    "gap catch-up failed: {e}"
                ),
            }
            let caught_up = t_gap.elapsed();
            self.kv_gap.spent(caught_up);
            self.wave_stats.add_quantize(caught_up.as_micros() as u64);
            self.step_census
                .spend(super::super::step_census::Activity::Quantize, caught_up);
            let t_upkeep = Instant::now();
            match self.session.create_deferred_arenas() {
                Ok(0) => {}
                Ok(n) => tracing::debug!(
                    target: "candle_conversation::scheduler::vram_relief",
                    arenas_created = n,
                    "created arenas a refused pass asked for"
                ),
                Err(e) => tracing::warn!("deferred arena creation failed: {e}"),
            }
            let arenas = t_upkeep.elapsed();
            self.wave_stats.add_housekeeping(arenas.as_micros() as u64);
            let t_compact = Instant::now();
            {
                // The whole pass, including the gate that usually declines it — so the
                // table shows what being *considered* every wave costs, not only what a
                // pass costs when it runs. The phase breakdown is filed inside.
                let _g = super::super::profile::span("loop:compact_kv");
                self.compact_kv_if_fragmented();
            }
            let compacted = t_compact.elapsed();
            self.wave_stats.add_compaction(compacted.as_micros() as u64);
            self.step_census.spend(
                super::super::step_census::Activity::Compact,
                arenas + compacted,
            );
            // **Then this gap's quantize**, over every complete block the wave's
            // forwards left — after compaction, so its lease is allocated over the
            // packed pools, the lowest-placed arenas first. A wave that ran prefill or
            // ingest is a prefill gap: its K/V compresses differently from decode's,
            // and the two keep their shares apart.
            {
                let _g = super::super::profile::span("loop:kv_gap");
                let t_gap = Instant::now();
                let kind = if self.prefill_width() > 0 || self.section_ingest_width() > 0 {
                    GapKind::Prefill
                } else {
                    GapKind::Decode
                };
                match self.session.gap_quantize(kind) {
                    Ok(q) => self.kv_gap.quantized(&q),
                    Err(e) => tracing::warn!(
                        target: "candle_conversation::scheduler::kv_gap",
                        "gap quantize failed: {e}"
                    ),
                }
                self.kv_gap.spent(t_gap.elapsed());
                // A turn stored with native chunks — its seal gap short of
                // ground — is quantized here, one a gap.
                self.quantize_a_native_residence();
                let quantized = t_gap.elapsed();
                self.wave_stats.add_quantize(quantized.as_micros() as u64);
                self.step_census
                    .spend(super::super::step_census::Activity::Quantize, quantized);
                let t_demote = Instant::now();
                // **The counterpart to elevate** (`docs/vram_governor_design.md`
                // §8.1), here between forwards, where every free is ordered
                // behind the forward that last read the block: drop the hot
                // copy of residences no wave has attended for
                // `IDLE_DEMOTE_GRACE_EPOCHS`, and only where a warm copy holds
                // the same bytes. Without it the hot set only grows, and every
                // shortfall is met by conceding expert-weight ground.
                self.demote_idle_hot();
                let demoted = t_demote.elapsed();
                self.wave_stats.add_housekeeping(demoted.as_micros() as u64);
                self.step_census
                    .spend(super::super::step_census::Activity::Compact, demoted);
                self.kv_gap.flush_due(Instant::now());
            }

            // Flush the wave summary + phase breakdown if its 2 s window
            // elapsed — even when no forward ran this iteration, so stalls still
            // surface their phase split.
            if self.wave_stats.due() {
                let t_flush = Instant::now();
                self.flush_wave_telemetry();
                let flushed = t_flush.elapsed();
                self.wave_stats.add_telemetry(flushed.as_micros() as u64);
                self.step_census
                    .spend(super::super::step_census::Activity::Telemetry, flushed);
            }
            // Closes here, not at end of scope: the livelock guard below can sleep, and a
            // span that swallowed that sleep would report the wave's idle wait as
            // housekeeping cost.
            house_span.end();

            // **Nothing admissible: block until something that could change that
            // arrives.** This wave ran no forward of any class, yet the idle `recv`
            // above did not block — a queue holds work this thread cannot admit.
            // During ingest a hot→warm backlog cuts `admit_budget` to the floor
            // (`regulate_ingest_admission`), so every width reads 0 until the
            // persistence thread drains it. Going round again changes nothing, and
            // every pass re-takes the substrate read lock: spun, that starved the
            // persistence thread's install-warm WRITE lock, so the backlog never
            // drained and the gate never reopened — the wave stuck 100% "blocked".
            //
            // So the loop waits on its channel, which is where every change it is
            // waiting for arrives: a new submission, and the persistence thread's
            // `Wake` the moment its backlog falls (`scheduler_waker`). With no pass
            // running, nothing here holds the substrate lock, and the drain gets it.
            // The timeout re-checks on its own for anything that changes admission
            // without a message.
            if self.decode_width() == 0
                && self.prefill_width() == 0
                && self.section_ingest_width() == 0
                && self.deferred_glue_fires.is_empty()
            {
                let t_wait = Instant::now();
                match self.rx.recv_timeout(UNADMITTED_RECHECK) {
                    Ok(req) => {
                        self.step_census
                            .spend(super::super::step_census::Activity::Idle, t_wait.elapsed());
                        self.wave_stats
                            .add_idle(t_wait.elapsed().as_millis() as u64);
                        let t_req = Instant::now();
                        let keep_going = self.handle_request(req);
                        self.step_census.spend(
                            super::super::step_census::Activity::Requests,
                            t_req.elapsed(),
                        );
                        self.wave_stats
                            .add_requests(t_req.elapsed().as_micros() as u64);
                        if !keep_going {
                            break;
                        }
                    }
                    Err(RecvTimeoutError::Timeout) => {
                        self.step_census
                            .spend(super::super::step_census::Activity::Idle, t_wait.elapsed());
                        self.wave_stats
                            .add_idle(t_wait.elapsed().as_millis() as u64);
                    }
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            }
        }

        tracing::info!("scheduler shut down");
    }
}
