//! The decode quantum: the steps that advance every active decode under a
//! wall-clock slice, and the work that runs around each step.
//!
//! The quantum is clipped by wall clock, and an unfinished generation is carried
//! into the next quantum rather than stopped. A reprojection queued by a prefill
//! fires before the first decode step, so the model's first reasoning token is
//! sampled against the projection that answers the query. The creep and section
//! chunks ride a step only when decode has been paid what the last co-batched
//! step owed it, and a debt holds only against work of its own band or below, so
//! a higher-band prompt shares the device by its own ratio. A prefill that
//! finishes decodes from the next step, not the next quantum.

use super::*;

impl Scheduler {
    /// Number of currently-active decode sequences (including summary probes).
    /// Used as the "is there decode work to run" guard — and by the decode-less
    /// creep in `run_prefill_until_budget` to skip its standalone advance when the
    /// decode quantum will co-batch the cohort instead.
    pub(in super::super) fn decode_width(&self) -> usize {
        self.active_decodes.values().filter(|s| !s.finished).count()
    }

    /// The decodes [`Self::decode_width`] counts, as a tier is priced over them:
    /// each one's sequence, the position its next row stands at, and the block
    /// its next step runs — one row, or its committed token and the draft depth
    /// the step will give it, by the step's own rule (`DecodeState::drafts`,
    /// `DraftDepth::budget` under the model's width ceiling). A decode's
    /// selection carve follows its depth and its live index, and a verifying
    /// one's head scores every row of its block, so the count alone under-prices
    /// a step beside deep or drafting conversations. A decode holding a played
    /// static run writes that run instead (`stencil_wave`).
    pub(in super::super) fn decode_stage(&self) -> Vec<(usize, usize, usize)> {
        let ceiling = self.model.draft_budget(self.decode_width());
        let token_cost = self.model.draft_token_cost();
        self.active_decodes
            .iter()
            .filter(|(_, s)| !s.finished)
            .map(|(&id, s)| {
                let depth = if s.drafts() {
                    s.draft_depth.budget(ceiling, token_cost)
                } else {
                    0
                };
                (
                    id.0,
                    self.session.sequence_offset(id.0).unwrap_or(0),
                    self.stencil_row_len(id).unwrap_or(1 + depth),
                )
            })
            .collect()
    }

    /// Active *foreground* decode sequences — excludes compression passes.
    ///
    /// Count the active foreground dialogue decodes that feed the
    /// prefill/decode flip heuristic. Compression half-passes ride
    /// `active_decodes` and the decode wave like any decode, but are excluded
    /// here so they never hold the loop in decode-first mode at the expense of
    /// dialogue prefills (they're off the critical path — background work that
    /// co-batches opportunistically).
    pub(in super::super) fn foreground_decode_width(&self) -> usize {
        self.foreground_decodes().count()
    }

    /// The active foreground decodes themselves — the set
    /// [`Self::foreground_decode_width`] counts, and the decodes the prefill
    /// throttle protects (`airtime_ratio_over`).
    pub(in super::super) fn foreground_decodes(&self) -> impl Iterator<Item = SequenceId> + '_ {
        self.active_decodes
            .iter()
            .filter(|(_, s)| {
                !s.finished
                    && !matches!(
                        s.seal_action,
                        super::super::SealAction::CompressionPass { .. }
                    )
            })
            .map(|(&id, _)| id)
    }

    /// Run decode steps until the `WAVE_SLICE` deadline or decode is empty.
    ///
    /// The quantum is clipped by wall-clock, not step count — an unfinished
    /// generation is passed forward via `active_decodes` to the next quantum. We
    /// deliberately do **not** yield mid-quantum on width comparisons (the outer
    /// dispatcher already chose this phase; the time slice is what guarantees the
    /// other phases get airtime), but we DO admit newly-queued work mid-quantum via
    /// `mid_wave_admission` so a long generation can't starve fresh conversations.
    pub(super) fn run_decode_until_budget(&mut self) {
        // Phase-b VRAM relief: a sustained decode grows KV every step with no
        // admission gate of its own, so relieve once at the quantum boundary when
        // under pressure — the governor's cheapest-first ladder sheds cold turns,
        // and the reprojection drain below handles working-set turnover. One check
        // per quantum (not per step) keeps it cheap.
        {
            let _g = profile::span("loop:decode:relief_check");
            let t = Instant::now();
            if self.vram_under_pressure_for(VramPhase::Decode) {
                self.relieve_vram_pressure(
                    "decode",
                    VramPhase::Decode,
                    super::super::prefill::ReliefNeed::Paced,
                );
            }
            self.step_census
                .spend(super::super::step_census::Activity::Relief, t.elapsed());
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
        {
            let _g = profile::span("loop:decode:reproject_first");
            self.drain_pending_reprojections();
        }
        self.step_census.spend(
            super::super::step_census::Activity::Reproject,
            t_reproj0.elapsed(),
        );
        self.wave_stats
            .add_phase(WavePhase::Reproject, t_reproj0.elapsed().as_millis() as u64);
        let deadline = Instant::now() + WAVE_SLICE;
        let mut steps = 0usize;
        // When this quantum first saw a finished turn it has not yet cleaned up —
        // the clock `CLEANUP_DEFER` runs on.
        let mut first_finished: Option<Instant> = None;
        loop {
            if self.decode_width() == 0 {
                // No live decode work, but there may be sequences inserted as
                // finished during the prefill phase (EOS on first token) that
                // the decode loop never had a chance to clean up.
                //
                // Spanned like the in-loop call: this is the exit taken on the
                // quanta where nothing decodes, which is exactly where a seal is
                // most likely to be the whole cost, so leaving it unattributed
                // under-reports the span precisely when it matters.
                let _g = profile::span("loop:decode:cleanup");
                self.timed_cleanup();
                return;
            }
            // A turn its caller let go of costs nothing more: its prefill leaves
            // before the step, so the step does not sweep it.
            self.timed_activity(
                super::super::step_census::Activity::Withdraw,
                Self::withdraw_abandoned_prefills,
            );
            self.timed_rates_and_gallery();
            // Play any pending tool-call static runs (Layer 3) before the
            // decode forward, which writes them beside the decode rows, so a
            // `Static` run costs neither N decode steps nor a forward of its
            // own.  No-op when no sequence has an active stencil.
            {
                let _g = profile::span("loop:decode:stencil");
                let t = Instant::now();
                self.play_stencil_runs();
                self.step_census
                    .spend(super::super::step_census::Activity::Stencil, t.elapsed());
            }
            {
                let _g = profile::span("loop:decode:step");
                // Every step is a wave: the creep and a section chunk ride it
                // whenever decode has been paid what the last co-batched step
                // owed it (`creep_owed_after`), so work admitted mid-quantum
                // joins the next step rather than the next quantum.
                // **A debt holds only work of its own band or below.** It is
                // decode's protection from the work that ran it up; a prompt of
                // a higher band shares the device by its own ratio. Run up by a
                // `Low` ingest chunk at 64, one debt held four Makers' `High`
                // prompts at a fixed backlog for most of a minute (WhiteWorld,
                // 2026-10-26).
                // Queued or still a submission counts: admission offers a prompt
                // whatever the debt, but rows closed behind the debtor's own
                // members leave it unstarted, and read only among the running
                // the debt stood over it, with a `High` backlog fixed for
                // minutes (WhiteWorld, 2026-10-31).
                if !self.creep_owed.is_zero()
                    && (self.waiting_above(self.creep_owed_band, |_| false)
                        || self.a_queued_prompt_outranks(self.creep_owed_band))
                {
                    self.creep_owed = Duration::ZERO;
                }
                let owed = !self.creep_owed.is_zero();
                self.wave_cohort_advanced = owed;
                self.wave_section_advanced = owed;
                // Reset before the step so a step that runs no wave reads as
                // decode-only, then filed by the shape the wave turned out to be.
                self.wave_cobatched = false;
                self.wave_step_creep_layer_rows = 0;
                self.wave_step_creep_layers = 0;
                self.wave_step_creep_layer_pairs = 0.0;
                self.wave_step_short_sweep = false;
                // What the creep had to ride with, before the step moves it.
                let admitted = self.prefill_width() > 0 || self.section_ingest_width() > 0;
                let queued = !self.prefill_queue.is_empty() || !self.pending_submits.is_empty();
                let width = self.decode_width();
                let t_step = Instant::now();
                let decoding_before = self.decode_tokens_committed();
                self.batch_decode_step();
                // Tokens a decoding row commits a step: one without a draft,
                // its accepted drafts and one more with — what a reply's tokens
                // are divided by to price its steps (`LearnedRates::tokens_per_step`).
                //
                // **The same rows on both sides of the step.** Summed over the
                // rows still decoding after it, a turn that finished in the step
                // left with all its tokens and the step read as committing
                // fewer — none at all through WhiteWorld's minute of ending turns
                // (3cccc3fcb, 2026-10-21: 0.00 after 3.38).
                let committed: usize = decoding_before
                    .iter()
                    .map(|&(id, before)| {
                        self.active_decodes
                            .get(&id)
                            .map_or(0, |s| s.generated_tokens.len().saturating_sub(before))
                    })
                    .sum();
                // Per row the step carried: a row held out of it — behind an
                // `Urgent` decode, an adapter, a glue fire — committed nothing
                // because it did not ride, not because a step yields less.
                if self.step_rows > 0 {
                    self.tokens_per_step
                        .add_value(committed as f64 / self.step_rows as f64);
                }
                let step = t_step.elapsed();
                // The phase timeline's decode: the tokens the step committed, and
                // the step's wall-clock split with the creep that rode it.
                self.wave_stats.add_decode_committed(committed as u64);
                self.wave_stats.close_decode_step(
                    step.as_micros() as u64,
                    self.decode_only_step.map(|d| d.as_micros() as u64),
                );
                profile::record(
                    if self.wave_cobatched {
                        "decode:cobatched:step"
                    } else {
                        "decode:decode_only:step"
                    },
                    step,
                );
                // Glue rides every step it is due and is a decode row's own
                // reprojection, not the creep: only the creep or a section chunk
                // riding this step runs up the debt.
                let carried = !owed && (self.wave_cohort_advanced || self.wave_section_advanced);
                self.step_census.record(
                    super::super::step_census::StepShape::of_decode(
                        self.wave_cobatched,
                        carried,
                        owed,
                        admitted,
                        queued,
                    ),
                    step,
                );
                self.creep_owed = next_creep_owed(
                    self.creep_owed,
                    step,
                    carried,
                    self.decode_only_step,
                    self.wave_step_ratio,
                );
                if carried {
                    self.creep_owed_band = self.wave_step_band;
                }
                self.width_steps.record(width, step, Instant::now());
                if !self.wave_cobatched {
                    self.decode_only_step = Some(next_decode_only(self.decode_only_step, step));
                } else if let Some(decode_only) = self
                    .decode_only_step
                    .filter(|_| !self.wave_step_short_sweep)
                {
                    // What the creep that rode cost beside decode, so the next
                    // slice fits a decode step's worth of it.
                    self.creep_layer_row_cost = next_layer_row_cost(
                        self.creep_layer_row_cost,
                        step,
                        decode_only,
                        self.wave_step_creep_layer_rows,
                    );
                    self.creep_layer_cost = next_layer_row_cost(
                        self.creep_layer_cost,
                        step,
                        decode_only,
                        self.wave_step_creep_layers,
                    );
                }
            }
            // A prefill this step finished decodes from the next step, not the
            // next quantum: promoted here, ahead of the drain below, so the
            // reprojection its first token queued fires before its first decode
            // step — the order the quantum's opening drain keeps. Left to the
            // loop top, every finished prompt waited out the rest of the slice;
            // a calibration's twenty-four 64-token thoughts reached the decode
            // set a quantum at a time.
            {
                let _g = profile::span("loop:decode:promote_finished");
                let t = Instant::now();
                self.promote_finished_prefills_to_decodes();
                HOUSE_PROMOTE_US.fetch_add(
                    t.elapsed().as_micros() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            // Drain any continuous-re-projection swaps queued during the
            // batch.  Must run BEFORE cleanup_finished so a swap that
            // re-keys an active_decodes entry doesn't race with finalize.
            // Timed separately (a sub-slice of the decode quantum) because the
            // provenance scan + glue gap-fill here is a prime "grows over time" suspect.
            let t_reproj = Instant::now();
            {
                let _g = profile::span("loop:decode:reproject");
                self.drain_pending_reprojections();
            }
            self.step_census.spend(
                super::super::step_census::Activity::Reproject,
                t_reproj.elapsed(),
            );
            self.wave_stats
                .add_phase(WavePhase::Reproject, t_reproj.elapsed().as_millis() as u64);
            if first_finished.is_none() && self.active_decodes.values().any(|s| s.finished) {
                first_finished = Some(Instant::now());
            }
            let urgent = first_finished.is_some()
                && self
                    .active_decodes
                    .values()
                    .any(|s| s.finished && s.wrote_tool_call);
            if cleanup_due(first_finished, Instant::now(), self.decode_width(), urgent) {
                let _g = profile::span("loop:decode:cleanup");
                self.timed_cleanup();
                first_finished = None;
            }
            steps += 1;

            // Take on conversations that queued WHILE this wave was executing,
            // under the same entry criteria the top-of-loop admission uses. The
            // newly-projected prefills land in `active_prefills` now and join the
            // creep cohort the next time `form_wave_group` re-forms — at the
            // next step the creep rides with no cohort in flight.
            let admitted = {
                let _g = profile::span("loop:decode:mid_wave_admit");
                let t = Instant::now();
                let admitted = self.mid_wave_admission();
                self.step_census
                    .spend(super::super::step_census::Activity::MidAdmit, t.elapsed());
                admitted
            };
            if !admitted {
                self.shutdown_requested = true;
                if first_finished.is_some() {
                    let _g = profile::span("loop:decode:cleanup");
                    self.timed_cleanup();
                }
                return;
            }

            // Clip to the time slice regardless of remaining decode work; the
            // remaining sequences persist in `active_decodes` and resume next
            // quantum. The step cap is only a backstop if steps ever run far
            // under the WDDM floor (see `MAX_DECODE_STEPS`). A finished turn whose
            // cleanup was deferred is cleaned up on the way out, so the deferral
            // never outlives the quantum.
            if Instant::now() >= deadline || steps >= MAX_DECODE_STEPS {
                if first_finished.is_some() {
                    let _g = profile::span("loop:decode:cleanup");
                    self.timed_cleanup();
                }
                return;
            }
        }
    }

    /// Each running decode and the tokens it has committed so far.
    pub(super) fn decode_tokens_committed(&self) -> Vec<(SequenceId, usize)> {
        self.active_decodes
            .iter()
            .filter(|(_, s)| !s.finished)
            .map(|(&id, s)| (id, s.generated_tokens.len()))
            .collect()
    }
}
