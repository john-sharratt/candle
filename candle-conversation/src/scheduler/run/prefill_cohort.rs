//! Admission and the prefill cohort's advance, both folded into the co-batched
//! wave step rather than run as separate forwards.
//!
//! The prefill cohort advances only through the wave step. When decode is active
//! the cohort joins decode's sweep, so one expert load serves both. When nothing
//! decodes, the same step runs with an empty decode group, so dialogue prefills,
//! section chunks and deferred glue all still advance. Work that arrives during a
//! quantum is admitted mid-quantum, and admission reopens when something settles
//! as well as when work arrives. A refused tier is waited on and retried with a
//! bound, so a span that never frees ground ends its members rather than holding
//! them forever.

use super::*;

impl Scheduler {
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
    pub(super) fn mid_wave_admission(&mut self) -> bool {
        if !self.rx.is_empty() {
            // Flag the drain so the assembler attributes its sub-timers to the
            // drain buckets and `apply_projection` DEFERS its gap-fills into the
            // unified wave step (`take_wave_glue`), exactly as the loop-top drain.
            let t_drain = Instant::now();
            let drain_span = profile::span("mid:drain");
            IN_DRAIN.store(true, std::sync::atomic::Ordering::Relaxed);
            self.batch_drain_gap_fills = true;
            let cont = self.drain_submissions();
            self.batch_drain_gap_fills = false;
            IN_DRAIN.store(false, std::sync::atomic::Ordering::Relaxed);
            drain_span.end();
            self.wave_stats
                .add_phase(WavePhase::Drain, t_drain.elapsed().as_millis() as u64);
            if !cont {
                return false;
            }
        }
        // **Admission reopens on whatever settled, not only on an arrival.** A
        // turn finishing or a prefill leaving mid-quantum frees the ground a
        // refused offer was waiting on; gated on the channel alone, the queue
        // behind it waited out the slice — 88% of passes stopped on the rate
        // while their turns were re-offered two seconds apart. The settled flag
        // is what keeps this cheap: nothing changed, nothing is priced.
        if self.settled_since_admit {
            let t_promote = Instant::now();
            let _g = profile::span("mid:promote");
            self.promote_new_prefills();
            // Inside the decode quantum, so it is carved out of decode into the
            // allocation band, never added beside it.
            self.wave_stats
                .add_mid_promote(t_promote.elapsed().as_micros() as u64);
        }
        // Bound KV production to the hot→warm drain rate across the slice, not just
        // once per quantum. Self-gated on an active ingest so a plain dialogue decode
        // pays nothing (the loop-top call still handles the ingest-finished reopen).
        if !self.ingest_timelines.is_empty() {
            {
                let _g = profile::span("mid:regulate_ingest");
                self.regulate_ingest_admission();
            }
            let _g = profile::span("mid:demote_cold_ingest");
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
    pub(super) fn run_prefill_until_budget(&mut self) {
        if self.wave_cohort_advanced || self.decode_width() > 0 {
            return;
        }
        if self.prefill_width() == 0
            && self.section_ingest_width() == 0
            && self.deferred_glue_fires.is_empty()
        {
            return;
        }
        self.wave_step_creep_layer_rows = 0;
        self.wave_step_creep_layers = 0;
        self.wave_step_creep_layer_pairs = 0.0;
        let t_step = Instant::now();
        let out = self.decode_forward_cobatched(&[], &[], &[], &[], &[], &[]);
        self.step_census.record(
            match &out {
                Err(e) if is_tier_refusal(e) || is_span_full(e) => {
                    super::super::step_census::StepShape::Refused
                }
                _ => super::super::step_census::StepShape::CreepOnly,
            },
            t_step.elapsed(),
        );
        match out {
            Ok(_) => {
                if let Some(failures) = self.wave_retry.succeeded() {
                    tracing::info!(failures, "decode-less wave step ran after failing");
                }
                // What a layer-row of creep costs with nothing beside it, so
                // the next decode-less step fits `DECODE_LESS_SLICE`.
                self.creep_layer_row_cost = next_layer_row_cost(
                    self.creep_layer_row_cost,
                    t_step.elapsed(),
                    Duration::ZERO,
                    self.wave_step_creep_layer_rows,
                );
                self.creep_layer_cost = next_layer_row_cost(
                    self.creep_layer_cost,
                    t_step.elapsed(),
                    Duration::ZERO,
                    self.wave_step_creep_layers,
                );
            }
            // A refused tier is answered where it was refused — the next wave is
            // composed narrower, or one already at the least chunk waits for
            // ground like a full span — so it is not a fault. It is still waited
            // between attempts and bounded like one: retried at once it spun
            // (9,426 refusals in 27 s on the 3090), and a span that never gives
            // ground back must end its members rather than hold them for ever.
            Err(e) if is_tier_refusal(&e) || is_span_full(&e) => match self.wave_retry.failed() {
                Next::Retry { attempt, wait } => {
                    // As a mid-sweep window does: lower the frontier into the
                    // holes below it before waiting, so the retry finds the
                    // ground the tier was refused.
                    if let Some(short) = candle_nn::kv_cache::tier_refusal_shortfall(&e) {
                        self.lower_frontier_for(short);
                    }
                    if WaveRetry::logs(attempt) {
                        tracing::warn!(
                            attempt,
                            wait_us = wait.as_micros() as u64,
                            "decode-less wave step refused its tier; waiting for ground: {e}"
                        );
                    }
                }
                Next::GiveUp { attempts } => {
                    tracing::error!(
                        attempts,
                        "decode-less wave step refused its tier on every attempt; failing its \
                         members: {e}"
                    );
                    self.fail_decode_less_wave(&format!(
                        "the wave carrying this found no ground in {attempts} attempts: {e}"
                    ));
                }
            },
            Err(e) => match self.wave_retry.failed() {
                Next::Retry { wait, attempt } => {
                    if WaveRetry::logs(attempt) {
                        tracing::error!(
                            attempt,
                            wait_us = wait.as_micros() as u64,
                            "decode-less wave step failed: {e}"
                        );
                    }
                }
                Next::GiveUp { attempts } => {
                    tracing::error!(
                        attempts,
                        "decode-less wave step failed on every attempt; failing its members: {e}"
                    );
                    self.fail_decode_less_wave(&format!(
                        "the wave carrying this could not run in {attempts} attempts: {e}"
                    ));
                }
            },
        }
    }

    /// Fail everything a decode-less wave carries — every prefill and section
    /// ingest still owing tokens — so each caller hears the error and its
    /// ground is handed back. See `wave_retry` for when this is the right call.
    pub(in super::super) fn fail_decode_less_wave(&mut self, why: &str) {
        for p in &mut self.active_prefills {
            if p.error.is_none() && p.finished.is_none() && p.offset < p.work.tokens.len() {
                p.error = Some(ConversationError::Other(why.to_string()));
            }
        }
        for s in &mut self.active_section_ingests {
            if s.error.is_none() && s.offset < s.tokens.len() {
                s.error = Some(ConversationError::Other(why.to_string()));
            }
        }
        self.reset_wave_prefill();
    }

    /// Seal completed section ingests. Section chunks are advanced ONLY by the
    /// unified wave step (`decode_forward_cobatched`) — co-batched into decode's
    /// sweep when decode is active, else folded into the decode-less sweep by
    /// `run_prefill_until_budget`. This quantum only drains the sections that step
    /// *completed* so they seal + send their `SealResult`. Cheap when none are done.
    pub(super) fn run_section_ingest_until_budget(&mut self) {
        self.finalize_done_section_ingests();
    }
}
