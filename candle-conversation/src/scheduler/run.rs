use super::admission::{admit_quantum, budget_notches, evidence_ticks_for, ThrottleReason};
use super::creep_slice::next_layer_row_cost;
use super::prefill::{VramPhase, LOAD_TARGET};
use super::wave_retry::{Next, WaveRetry};
use super::*;
use candle::wave_provenance::{publish_wave_declines, DeclineSnapshot};
use candle::DeviceLocation;
use candle_nn::kv_cache::{
    is_span_full, is_tier_refusal, region_stats, set_kv_free_target, set_prompts_waiting,
    span_partition, vram_bands,
};
use std::process::exit;
use std::sync::atomic::AtomicU64;
use std::time::{Duration, Instant};

/// Per-sub-step housekeeping timers (microseconds), swapped out and logged at
/// each wave-window flush. The `housekeeping` phase is one band on the
/// dashboard; these decompose it so a regression can be pinned to the step that
/// caused it rather than to the aggregate.
static HOUSE_PROMOTE_US: AtomicU64 = AtomicU64::new(0);
static HOUSE_ADMIT_US: AtomicU64 = AtomicU64::new(0);
static HOUSE_DEMOTE_US: AtomicU64 = AtomicU64::new(0);
static HOUSE_GPUDRAIN_US: AtomicU64 = AtomicU64::new(0);

/// Which completion path inside `promote_finished_prefills_to_decodes` a span
/// belongs to — that step dominates housekeeping, so it is decomposed further.
pub(super) enum PromoteStep {
    Finalise,
    Compression,
}

static PROMOTE_FINALISE_US: AtomicU64 = AtomicU64::new(0);
static PROMOTE_COMPRESSION_US: AtomicU64 = AtomicU64::new(0);

/// Record one completion-path span (microseconds).
pub(super) fn note_promote_split(step: PromoteStep, us: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    match step {
        PromoteStep::Finalise => PROMOTE_FINALISE_US.fetch_add(us, Relaxed),
        PromoteStep::Compression => PROMOTE_COMPRESSION_US.fetch_add(us, Relaxed),
    };
}

/// Drain the promote-path sub-timers, in ms, as `(finalise, compression)`.
pub(super) fn take_promote_split() -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        PROMOTE_FINALISE_US.swap(0, Relaxed) / 1000,
        PROMOTE_COMPRESSION_US.swap(0, Relaxed) / 1000,
    )
}

/// Drain the housekeeping sub-timers, in milliseconds, as
/// `(promote, admit, demote, gpu_drain)`.
pub(super) fn take_housekeeping_split() -> (u64, u64, u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    (
        HOUSE_PROMOTE_US.swap(0, Relaxed) / 1000,
        HOUSE_ADMIT_US.swap(0, Relaxed) / 1000,
        HOUSE_DEMOTE_US.swap(0, Relaxed) / 1000,
        HOUSE_GPUDRAIN_US.swap(0, Relaxed) / 1000,
    )
}

/// Wall-clock ceiling for one decode quantum ("wave"). The quantum is CLIPPED to
/// this whether or not decode has finished — unfinished sequences persist in
/// `active_decodes` and resume in the next quantum, so clipping just passes the
/// remaining work forward. Time-slicing (rather than the old fixed 32-step
/// budget) guarantees the loop returns to the top every ~2 s to re-form the creep
/// cohort with work admitted mid-wave, run admission + relief, and aggregate
/// telemetry — regardless of how fast or slow the individual steps run.
const WAVE_SLICE: Duration = Duration::from_millis(2000);

/// How long a loop that could admit nothing waits on its channel before it
/// looks again on its own. What reopens admission — a submission, the
/// persistence drain's backlog falling — arrives on the channel and ends the
/// wait at once; this bounds only a change that sends nothing.
const UNADMITTED_RECHECK: Duration = Duration::from_millis(100);

/// Hard safety cap on decode steps within one quantum, independent of the wall
/// clock. At the ~74 ms/step WDDM launch floor a 2 s slice is ~27 steps, far
/// under this; the cap only engages if decode ever runs far faster (TCC mode /
/// CUDA graphs), bounding per-quantum KV growth so the loop-top relief can't be
/// outrun before the deadline check fires.
const MAX_DECODE_STEPS: usize = 256;

/// Longest a finished turn waits for its seal while other turns are still decoding.
///
/// A finished turn's cleanup — its seal and the closing tail it forwards — runs on
/// the loop thread, so run between decode steps it holds back every turn still
/// generating. Measured on the Qwen3.6-35B-A3B with eight turns decoding together:
/// two that accepted more draft tokens finished a step early, and their ~40 ms of
/// cleanup sat in front of the other six's last step. Deferred, the early finishers
/// seal with the rest when the cohort drains — sharing its tail forward — and their
/// tokens have all streamed already, so what waits is only the `Done`. The bound is
/// what keeps a short turn from waiting out a long one's whole generation.
///
/// A turn that wrote a tool call is not deferred at all: its `Done` is what its
/// caller acts on — it runs the tool and submits the result — so waiting there
/// stalls an agent loop rather than a status line.
const CLEANUP_DEFER: Duration = Duration::from_millis(100);

/// The process exit status when the engine can serve nothing again — the expert
/// pipeline has exited. `EX_SOFTWARE`: an internal fault, not a usage error.
const ENGINE_DEAD_EXIT: i32 = 70;

/// Whether finished turns are cleaned up after this decode step: at once when one
/// of them is `urgent` (it wrote a tool call), once nothing is left decoding, or
/// once the earliest of them has waited [`CLEANUP_DEFER`].
fn cleanup_due(
    first_finished: Option<Instant>,
    now: Instant,
    decoding: usize,
    urgent: bool,
) -> bool {
    match first_finished {
        None => false,
        Some(at) => urgent || decoding == 0 || now.saturating_duration_since(at) >= CLEANUP_DEFER,
    }
}

/// The decode-only airtime a step that also carried the prefill creep owes
/// before the creep may ride again: the creep's share of the step — its length
/// `step` less what a decode-only step takes now (`decode_only`) — times the
/// airtime ratio `R` the creep yielded to (the ratio of the decodes that
/// protect themselves from it — `prefill::throttle_ratio`). Decode gets `R`
/// times the creep's airtime.
///
/// **A co-batched step must not take decode's share of the device.** At `R = 1`
/// (no decode protects itself from the creep — an ingest beside its own turns)
/// a step carries every layer of an 8K-token
/// chunk: 4.2 s at depth. Measured in zend, with the creep riding every step it
/// could: 24 of 135 steps carried it and took 100 s of a 120 s window — eight
/// sessions decoding ~14% of the time where `R = 1` means equal airtime.
///
/// **And the debt is paid in decode steps, not in quanta.** Paying it by letting
/// the creep ride only a quantum's first step held a prefill to one advance per
/// [`WAVE_SLICE`] whenever anything decoded: an 8K chunk that runs in 120 ms
/// went once every 2 s, and a turn's tool result waited out the rest of another
/// conversation's slice before its prefill moved — 2 s of nothing between the
/// turns of a live dialogue, and an ingest of 30 workers finishing 0.7 folders a
/// minute. Owed by the step, the creep rides again as soon as decode has had its
/// share.
///
/// **The creep's share, not the step's length over `R`.** Owing `step / R`
/// assumed the creep's `N / R` layers were a sliver of a step; at depth they are
/// not. One layer of an eight-member ingest group at 300K positions cost ~40 ms
/// beside a ~30 ms chat decode, and at `R = 64` it owed 1 ms — so it rode every
/// step, and a `High` dialogue decoded 13–15 steps a second where it ran 33
/// alone (`.5`, 2026-10-17). The share times `R` is what `R` says: beside a
/// dialogue that ingest rides about once in 65 steps' worth of decode airtime,
/// and at full speed once the dialogue stops decoding. With no decode-only step
/// measured yet the whole step counts as the creep's, which errs toward decode.
fn creep_owed_after(step: Duration, decode_only: Option<Duration>, ratio: f64) -> Duration {
    let share = step.saturating_sub(decode_only.unwrap_or(Duration::ZERO));
    share.mul_f64(ratio.max(0.0))
}

/// The airtime decode is owed after a step of length `step`: what a step that
/// carried the creep or a section chunk owes (`creep_owed_after`), or what was
/// owed less a step that did not.
fn next_creep_owed(
    owed: Duration,
    step: Duration,
    carried: bool,
    decode_only: Option<Duration>,
    ratio: f64,
) -> Duration {
    if carried {
        creep_owed_after(step, decode_only, ratio)
    } else {
        owed.saturating_sub(step)
    }
}

/// What a decode-only step takes now, after one more of length `step`: an
/// exponential average weighting the newest by [`DECODE_ONLY_WEIGHT`], so it
/// follows the decode width and depth within a few steps.
fn next_decode_only(avg: Option<Duration>, step: Duration) -> Duration {
    match avg {
        None => step,
        Some(a) => a.mul_f64(1.0 - DECODE_ONLY_WEIGHT) + step.mul_f64(DECODE_ONLY_WEIGHT),
    }
}

/// The newest decode-only step's weight in [`next_decode_only`].
const DECODE_ONLY_WEIGHT: f64 = 0.25;

impl Scheduler {
    /// Number of currently-active decode sequences (including summary probes).
    /// Used as the "is there decode work to run" guard — and by the decode-less
    /// creep in `run_prefill_until_budget` to skip its standalone advance when the
    /// decode quantum will co-batch the cohort instead.
    pub(super) fn decode_width(&self) -> usize {
        self.active_decodes.values().filter(|s| !s.finished).count()
    }

    /// The decodes [`Self::decode_width`] counts, as a tier is priced over them:
    /// each one's sequence, the position its next row stands at, and the block
    /// its next step runs — one row, or its committed token and the draft depth
    /// the step will give it, by the step's own rule (`DecodeState::drafts`,
    /// `DraftDepth::budget` under the model's width ceiling). A decode's
    /// selection carve follows its depth and its live index, and a verifying
    /// one's head scores every row of its block, so the count alone under-prices
    /// a step beside deep or drafting conversations.
    pub(super) fn decode_stage(&self) -> Vec<(usize, usize, usize)> {
        let ceiling = self.model.draft_budget(self.decode_width());
        let token_cost = self.model.draft_token_cost();
        self.active_decodes
            .iter()
            .filter(|(_, s)| !s.finished)
            .map(|(id, s)| {
                let depth = if s.drafts() {
                    s.draft_depth.budget(ceiling, token_cost)
                } else {
                    0
                };
                (
                    id.0,
                    self.session.sequence_offset(id.0).unwrap_or(0),
                    1 + depth,
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
    pub(super) fn foreground_decode_width(&self) -> usize {
        self.foreground_decodes().count()
    }

    /// The active foreground decodes themselves — the set
    /// [`Self::foreground_decode_width`] counts, and the decodes the prefill
    /// throttle protects (`airtime_ratio_over`).
    pub(super) fn foreground_decodes(&self) -> impl Iterator<Item = SequenceId> + '_ {
        self.active_decodes
            .iter()
            .filter(|(_, s)| {
                !s.finished && !matches!(s.seal_action, super::SealAction::CompressionPass { .. })
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
    fn run_decode_until_budget(&mut self) {
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
                    super::prefill::ReliefNeed::Paced,
                );
            }
            self.step_census
                .spend(super::step_census::Activity::Relief, t.elapsed());
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
        self.step_census
            .spend(super::step_census::Activity::Reproject, t_reproj0.elapsed());
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
                super::step_census::Activity::Withdraw,
                Self::withdraw_abandoned_prefills,
            );
            self.timed_rates_and_gallery();
            // Inject any pending tool-call static runs (Layer 3) before the
            // decode forward, so a `Static` run costs one prefill rather than N
            // decode steps.  No-op when no sequence has an active stencil.
            {
                let _g = profile::span("loop:decode:stencil");
                let t = Instant::now();
                self.inject_stencil_prefills();
                self.step_census
                    .spend(super::step_census::Activity::Stencil, t.elapsed());
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
                if !self.creep_owed.is_zero() && self.waiting_above(self.creep_owed_band, |_| false)
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
                if !decoding_before.is_empty() {
                    let committed: usize = decoding_before
                        .iter()
                        .map(|&(id, before)| {
                            self.active_decodes
                                .get(&id)
                                .map_or(0, |s| s.generated_tokens.len().saturating_sub(before))
                        })
                        .sum();
                    self.tokens_per_step
                        .add_value(committed as f64 / decoding_before.len() as f64);
                }
                let step = t_step.elapsed();
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
                    super::step_census::StepShape::of_decode(
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
                } else if let Some(decode_only) = self.decode_only_step {
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
            self.step_census
                .spend(super::step_census::Activity::Reproject, t_reproj.elapsed());
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
                    .spend(super::step_census::Activity::MidAdmit, t.elapsed());
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
            self.wave_stats
                .add_phase(WavePhase::Promote, t_promote.elapsed().as_millis() as u64);
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
        self.wave_step_creep_layer_rows = 0;
        self.wave_step_creep_layers = 0;
        self.wave_step_creep_layer_pairs = 0.0;
        let t_step = Instant::now();
        let out = self.decode_forward_cobatched(&[], &[], &[], &[]);
        self.step_census.record(
            match &out {
                Err(e) if is_tier_refusal(e) || is_span_full(e) => {
                    super::step_census::StepShape::Refused
                }
                _ => super::step_census::StepShape::CreepOnly,
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
    pub(super) fn fail_decode_less_wave(&mut self, why: &str) {
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
        super::interleave::reseed_achievable_weight();
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
                .spend(super::step_census::Activity::Drain, drain_elapsed);
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
                    super::prefill::ReliefNeed::Paced,
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
                self.step_census
                    .spend(super::step_census::Activity::Device, t_guests.elapsed());
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
                self.step_census
                    .spend(super::step_census::Activity::Promote, t_promote.elapsed());
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

            // 2a. One turn of the normalization warm-up replay (see `norm_warm`).
            let t_warm = Instant::now();
            self.step_norm_warm();
            self.step_census
                .spend(super::step_census::Activity::Warm, t_warm.elapsed());

            // 2b. A turn whose prefill stands still is told what it waits
            // behind, so its caller can show it rather than look hung
            // (`prefill_wait`).
            self.timed_activity(
                super::step_census::Activity::Waits,
                Self::report_prefill_waits,
            );

            // 2c. An idle slot whose last seal is durable gives its recurrent
            // store back (`recurrent_park`).
            let t_park = Instant::now();
            self.park_durable_states();
            self.step_census
                .spend(super::step_census::Activity::Park, t_park.elapsed());

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
                if std::mem::take(&mut self.settle_owed) {
                    let t_settle = Instant::now();
                    let report = self.settle_device();
                    self.step_census
                        .spend(super::step_census::Activity::Device, t_settle.elapsed());
                    for waiter in self.settle_waiters.drain(..) {
                        let _ = waiter.send(report);
                    }
                }
                // Nothing is in flight, so everything resident is permanent:
                // the one moment the achievable weight residency is exact.
                super::interleave::reseed_achievable_weight();
                tracing::trace!("idle; waiting for a request");
                // Time ONLY the recv block (not the request handling) — this is the
                // scheduler idle between requests, attributed to the Idle phase so it
                // isn't mislabeled as Blocked in the GUI.
                let t_idle = Instant::now();
                // A queued warm-up is work too: wait only as long as it allows,
                // then go round to score its next replayed turn. So is a guest
                // job waiting for room — the drain at the top of the loop is
                // what tries it again, and nothing arrives to wake it.
                let wait = match (self.norm_warm_wait(), self.guests.queue.retry_wait()) {
                    (Some(a), Some(b)) => Some(a.min(b)),
                    (a, b) => a.or(b),
                };
                let req = if let Some(wait) = wait {
                    match self.rx.recv_timeout(wait) {
                        Ok(req) => req,
                        Err(flume::RecvTimeoutError::Timeout) => {
                            self.step_census
                                .spend(super::step_census::Activity::Idle, t_idle.elapsed());
                            self.wave_stats
                                .add_idle(t_idle.elapsed().as_millis() as u64);
                            continue;
                        }
                        Err(flume::RecvTimeoutError::Disconnected) => break,
                    }
                } else {
                    // **Priced while idle too.** Its reader takes the rates every
                    // period whether or not this member runs anything, and an idle
                    // member that only priced inside its quanta published what a
                    // loaded run last left it — WhiteWorld's 208 ms step for a whole
                    // run of 87 turns it was never placed (ed4d8c781, 2026-10-20).
                    // So the wait ends each period to price the state it is in.
                    let period = super::phase_ring::rates_read_period()
                        .unwrap_or(super::prefill::RATES_LOG_EVERY);
                    match self.rx.recv_timeout(period) {
                        Ok(req) => req,
                        Err(flume::RecvTimeoutError::Timeout) => {
                            self.step_census
                                .spend(super::step_census::Activity::Idle, t_idle.elapsed());
                            self.wave_stats
                                .add_idle(t_idle.elapsed().as_millis() as u64);
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
                                super::interleave::effective_weight_zone_bytes(0),
                            ) {
                                rate.settle_idle(zone);
                            }
                            self.publish_learned_rates();
                            continue;
                        }
                        Err(flume::RecvTimeoutError::Disconnected) => break,
                    }
                };
                self.step_census
                    .spend(super::step_census::Activity::Idle, t_idle.elapsed());
                self.wave_stats
                    .add_idle(t_idle.elapsed().as_millis() as u64);
                // The handling itself is Requests, not Idle — it is work, and
                // leaving it untimed put it in the Blocked remainder.
                let t_req = Instant::now();
                let keep_going = self.handle_request(req);
                self.step_census
                    .spend(super::step_census::Activity::Requests, t_req.elapsed());
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
                            .spend(super::step_census::Activity::Idle, t_wait.elapsed());
                        self.wave_stats
                            .add_idle(t_wait.elapsed().as_millis() as u64);
                        let t_req = Instant::now();
                        let keep_going = self.handle_request(req);
                        self.step_census
                            .spend(super::step_census::Activity::Requests, t_req.elapsed());
                        self.wave_stats
                            .add_requests(t_req.elapsed().as_micros() as u64);
                        if !keep_going {
                            break;
                        }
                        continue;
                    }
                    Err(flume::RecvTimeoutError::Timeout) => {
                        self.step_census
                            .spend(super::step_census::Activity::Idle, t_wait.elapsed());
                        self.wave_stats
                            .add_idle(t_wait.elapsed().as_millis() as u64);
                    }
                    Err(flume::RecvTimeoutError::Disconnected) => break,
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
            let house_span = super::profile::span("loop:housekeeping");
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
            // Closed BEFORE the flush block: the flush is the window boundary
            // itself, and its own cost is already accounted (eviction/sync carve
            // out of the remainder), so folding it in here would double-count.
            self.step_census
                .spend(super::step_census::Activity::House, t_house.elapsed());
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
            // Legal here for the same reason `create_deferred_arenas` below is: the
            // quanta are done, every forward's guards are dropped, and no wave
            // generation is live — which is the condition `set_weight_floor` itself
            // checks. `compact_kv` refuses rather than corrupts if that ever stops
            // being true.
            {
                // The whole pass, including the gate that usually declines it — so the
                // table shows what being *considered* every wave costs, not only what a
                // pass costs when it runs. The phase breakdown is filed inside.
                let _g = super::profile::span("loop:compact_kv");
                self.compact_kv_if_fragmented();
            }

            match self.session.create_deferred_arenas() {
                Ok(0) => {}
                Ok(n) => tracing::debug!(
                    target: "candle_conversation::scheduler::vram_relief",
                    arenas_created = n,
                    "created arenas a refused pass asked for"
                ),
                Err(e) => tracing::warn!("deferred arena creation failed: {e}"),
            }

            // Flush the wave summary + phase breakdown if its 2 s window
            // elapsed — even when no forward ran this iteration, so stalls still
            // surface their phase split.
            if self.wave_stats.due() {
                // The 2 s telemetry window: the memory report, the class census, the
                // fragmentation log. Cheap per wave only because it is rare, which is
                // exactly the claim a span is needed to check.
                let _g = super::profile::span("loop:telemetry");

                // **What the rate planner currently believes a forward costs.**
                //
                // Published because every one of these was invisible while being wrong.
                // The planner decides how wide a wave is, which is the whole of decode
                // throughput, and it does so from four learned numbers — none of which
                // appeared in any log. A zero `fixed`+`overhead` makes the projected rate
                // flat in width and narrows every wave to one row, and that state was
                // reached on a card whose only distinguishing feature was being large
                // enough to hold the checkpoint. It read as healthy throughout.
                if let Some(rate) = self.wave_rate.as_ref() {
                    tracing::trace!(
                        target: "candle_conversation::scheduler::throttle",
                        fixed_ms = rate.fixed_secs() * 1e3,
                        overhead_ms = rate.overhead_secs() * 1e3,
                        compute_us_per_token = rate.compute_secs_per_token() * 1e6,
                        link_gbps = rate.effective_bytes_per_s() / 1e9,
                        samples = rate.samples(),
                        "wave rate model",
                    );
                }
                // Our eviction gate's own view of VRAM: the pool budget we
                // defend (vram_budget_available) and pool_used — queried only
                // on the wave we emit, not every iteration.
                let kv_vram = self
                    .session
                    .vram_budget_available()
                    .zip(self.session.vram_pool_stats())
                    .map(|(budget, (used, _reserved))| (budget, used));
                let backlog = self.pending_prefill_tokens();
                // The whole card, cut by what holds each byte: the driver's total
                // and free, what was on the card before the span, the CUDA pool
                // outside it, the span's own partition and the resident experts
                // in its weight zone. The bands add up to the card's total, so
                // the panel stacking them can show where every byte went.
                let pool_reserved = self.session.vram_pool_stats().map_or(0, |(_, r)| r as u64);
                let (free, total) = self
                    .session
                    .vram_free_total()
                    .map_or((0, 0), |(f, t)| (f as u64, t as u64));
                let (context, span) = match self.device.location() {
                    DeviceLocation::Cuda { gpu_id } => (
                        region_stats(gpu_id).and_then(|s| s.pre_reservation_in_use_bytes),
                        span_partition(gpu_id),
                    ),
                    _ => (None, None),
                };
                let resident_experts = self
                    .model
                    .expert_stats()
                    .map_or(0, |s| s.resident_vram_bytes as u64);
                let vram = vram_bands(total, free, context, pool_reserved, span, resident_experts);
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
                    vram,
                    slots,
                    self.model.expert_stats().map(|s| ExpertCounters::of(&s)),
                    match &self.device {
                        Device::Cuda(d) => Some(d.capture_stats()),
                        _ => None,
                    },
                    self.wave_rate.as_mut().map(|r| r.take_trades()),
                    self.model.recurrent_census(),
                );
                self.report_waiting_submits();
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
                    tracing::trace!(
                        target: "candle_conversation::scheduler::vram_relief",
                        arenas_swept = swept,
                        "proactive empty-arena sweep (per-wave)"
                    );
                }
                self.log_kv_memory();
                // **How fragmented the KV pools are — measured where a compaction
                // would run, at the cadence it would run at.**
                //
                // Deliberately beside `release_empty_arenas` above, because the two
                // are the same job at different granularities and the contrast is
                // the point: the sweep returns an arena whose LAST chunk has gone,
                // and nothing moves a chunk between arenas, so an arena holding a
                // handful of live chunks keeps its whole 16 MiB region indefinitely.
                // `freeable` is the regions that are recoverable and that the sweep
                // structurally cannot recover.
                //
                // This is the figure that is NOT region holes. A region freed below
                // the arena frontier is taken by the next claim — the free list is
                // lowest-index-first — so holes are self-correcting and measured to
                // peak in the tens and settle at one. Sparsity does not
                // self-correct, which is why it is what gets logged and what a
                // compaction is judged by.
                // **The frontier first, because the frontier is what costs
                // weights.** The tier stands above the highest live arena and
                // `weight_floor` is measured from there, so expert residency is
                // set by that one index — not by how many arenas are live and not
                // by how full they are. `could_be` is where the frontier would sit
                // with everything packed, and the gap between them is the ground a
                // perfect compaction hands back, split into its two causes: the
                // holes that hold the frontier up, and the sparsity that fills it
                // with air.
                if let Some(g) = self.session.kv_ground_lost() {
                    if g.total() > 0 {
                        let mib = candle_nn::kv_cache::REGION_BYTES >> 20;
                        tracing::trace!(
                            target: "candle_conversation::scheduler::vram_relief",
                            frontier = g.watermark,
                            could_be = g.packed_arenas,
                            live_arenas = g.live_arenas,
                            arena_holes = g.arena_holes(),
                            sparsity = g.sparsity(),
                            denied_mib = g.total() * mib,
                            // What reclaiming it would cost. Logged beside the
                            // gain because the ratio is the only thing that says
                            // whether a pass is worth its bandwidth: four arenas
                            // for forty thousand copies is not.
                            planned_moves = self.session.kv_planned_moves(),
                            "kv ground denied to the weight side by fragmentation",
                        );
                    }
                }
                let frag = self.session.kv_fragmentation();
                let freeable: usize = frag.iter().map(|(_, f)| f.freeable_arenas()).sum();
                if freeable > 0 {
                    let rows: Vec<String> = frag
                        .iter()
                        .filter(|(_, f)| f.freeable_arenas() > 0)
                        .map(|(k, f)| {
                            format!(
                                "{}B={}a/{}pack({}%)",
                                k.class.bytes(),
                                f.arenas,
                                f.packed_arenas,
                                f.occupancy_pct(),
                            )
                        })
                        .collect();
                    tracing::trace!(
                        target: "candle_conversation::scheduler::vram_relief",
                        freeable_arenas = freeable,
                        freeable_mib = freeable * (candle_nn::kv_cache::REGION_BYTES >> 20),
                        "kv fragmentation (a perfect pack would return this): {}",
                        rows.join(" "),
                    );
                }
                // **Is the planner still running on its seeds?** Every figure
                // it decides with is learned, and the seeds are a measurement of
                // one card with one checkpoint — the decode one was found 26x
                // optimistic. A run that admits oddly is asked this first, and
                // without the line the answer is unobtainable after the fact.
                if let Some(r) = self.wave_rate.as_ref() {
                    tracing::trace!(
                        target: "candle_conversation::scheduler::admission",
                        link_gbps = r.link_bytes_per_s() / 1e9,
                        effective_gbps = r.effective_bytes_per_s() / 1e9,
                        link_fraction = r.link_fraction(),
                        compute_us_per_row = r.compute_secs_per_token() * 1e6,
                        layer_us = r.layer_secs() * 1e6,
                        hit_exponent = r.hit_exponent(),
                        // The two-cost fit only answers once the observations
                        // have enough spread to separate the copy from the
                        // compute; until then both are seeds.
                        fit_converged = r.cost_fit_converged(),
                        prefill_samples = r.samples(),
                        decode_samples = r.decode_samples(),
                        hit_samples = r.hit_samples(),
                        "wave rate planner",
                    );
                }
                // Wake the sealing pass now that its ground exists, rather
                // than leaving it to the 5 s tick. Guarded on there being work —
                // an atomic load — so an idle engine is not woken once per wave
                // to find nothing. The trigger coalesces on a one-slot channel,
                // so a pass already running or already queued absorbs this.
                if self.persist_triggers.pending_warm_bytes() > 0 {
                    self.persist_triggers.fire();
                }
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
                            .spend(super::step_census::Activity::Idle, t_wait.elapsed());
                        self.wave_stats
                            .add_idle(t_wait.elapsed().as_millis() as u64);
                        let t_req = Instant::now();
                        let keep_going = self.handle_request(req);
                        self.step_census
                            .spend(super::step_census::Activity::Requests, t_req.elapsed());
                        self.wave_stats
                            .add_requests(t_req.elapsed().as_micros() as u64);
                        if !keep_going {
                            break;
                        }
                    }
                    Err(flume::RecvTimeoutError::Timeout) => {
                        self.step_census
                            .spend(super::step_census::Activity::Idle, t_wait.elapsed());
                        self.wave_stats
                            .add_idle(t_wait.elapsed().as_millis() as u64);
                    }
                    Err(flume::RecvTimeoutError::Disconnected) => break,
                }
            }
        }

        tracing::info!("scheduler shut down");
    }

    /// Run the decode quantum, attributing its wall-clock to [`WavePhase::Decode`]
    /// (the reprojection drain inside it is separately attributed to
    /// [`WavePhase::Reproject`] as a sub-slice).
    fn timed_decode(&mut self) {
        let _g = profile::span("loop:decode");
        let t = Instant::now();
        self.wave_stats.in_decode_quantum = true;
        self.run_decode_until_budget();
        self.wave_stats.in_decode_quantum = false;
        self.wave_stats
            .add_phase(WavePhase::Decode, t.elapsed().as_millis() as u64);
    }

    /// Run `work`, filing its wall with the census under `activity`.
    fn timed_activity(&mut self, activity: super::step_census::Activity, work: fn(&mut Self)) {
        let t = Instant::now();
        work(self);
        self.step_census.spend(activity, t.elapsed());
    }

    /// Each running decode and the tokens it has committed so far.
    fn decode_tokens_committed(&self) -> Vec<(SequenceId, usize)> {
        self.active_decodes
            .iter()
            .filter(|(_, s)| !s.finished)
            .map(|(&id, s)| (id, s.generated_tokens.len()))
            .collect()
    }

    /// Clean up finished turns — their seals — filed with the census.
    fn timed_cleanup(&mut self) {
        let t = Instant::now();
        self.cleanup_finished();
        self.step_census
            .spend(super::step_census::Activity::Seal, t.elapsed());
    }

    /// Update the gallery's keep trade and publish the learned rates, each
    /// filed with the census (`step_census`).
    fn timed_rates_and_gallery(&mut self) {
        let t = Instant::now();
        self.update_gallery_keep();
        let gallery = t.elapsed();
        self.publish_learned_rates();
        self.step_census
            .spend(super::step_census::Activity::Gallery, gallery);
        self.step_census.spend(
            super::step_census::Activity::Rates,
            t.elapsed().saturating_sub(gallery),
        );
    }

    /// Run the prefill quantum, attributing its wall-clock to [`WavePhase::Prefill`].
    fn timed_prefill(&mut self) {
        let _g = profile::span("loop:prefill");
        let t = Instant::now();
        self.timed_activity(
            super::step_census::Activity::Withdraw,
            Self::withdraw_abandoned_prefills,
        );
        self.timed_rates_and_gallery();
        self.run_prefill_until_budget();
        self.wave_stats
            .add_phase(WavePhase::Prefill, t.elapsed().as_millis() as u64);
    }

    /// Run the section-ingest quantum, attributing its wall-clock to
    /// [`WavePhase::Section`].
    fn timed_section(&mut self) {
        let _g = profile::span("loop:section");
        let t = Instant::now();
        self.timed_activity(
            super::step_census::Activity::Seal,
            Self::run_section_ingest_until_budget,
        );
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
            tracing::trace!(
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
            tracing::trace!("kv-pool classes: {}", rows.join(" "));
        }
        // The reservation's KV side. `free` is the pressure signal admission
        // reads; `peak_live` against `total` says how close the startup
        // partition came to binding, which is what step 7 tunes.
        //
        // The last two must read zero. A region claim creates an arena, arena
        // creation waits for the gap between forwards, and the wave's tier is
        // placed against the arena frontier as it stands at that moment — so a
        // claim arriving with a tier standing is an arena created inside a wave.
        if let Some(r) = candle_nn::kv_cache::region_stats(0) {
            tracing::trace!(
                "kv-regions: live={} peak={} free={} of {} ({}MiB) | tier={}MiB \
                 (ceiling {} regions) | weights={}MiB | in-wave-arenas={} in-wave-refusals={}",
                r.live,
                r.peak_live,
                r.free,
                r.total,
                mib(r.total * candle_nn::kv_cache::REGION_BYTES),
                mib(r.transient_bytes),
                r.transient_ceiling,
                mib(r.weight_bytes),
                r.fresh_claims_during_wave,
                r.refusals_during_wave,
            );
        }
        // Per-domain transient peaks, the terms the transient tier is sized
        // from: `S = 2*W_wave + W_persist + shelf`.
        if let Some((cursor, peak, cap)) = candle_nn::kv_cache::persistence_domain_stats(0) {
            if peak > 0 {
                tracing::trace!(
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
                tracing::trace!(
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
            tracing::trace!(
                "kv-slotstate: live={live} slabs={slabs} reserved={}MiB promotions={}",
                mib(bytes),
                candle_nn::kv_cache::class_promotion_count(),
            );
        }
    }
}

#[cfg(test)]
mod cleanup_deferral_tests {
    use super::{cleanup_due, CLEANUP_DEFER};
    use std::time::{Duration, Instant};

    /// Nothing finished, nothing to clean up — however long the quantum has run.
    #[test]
    fn nothing_finished_is_never_due() {
        assert!(!cleanup_due(None, Instant::now(), 0, false));
        assert!(!cleanup_due(None, Instant::now(), 8, false));
        assert!(!cleanup_due(None, Instant::now(), 8, true));
    }

    /// The cohort drained: clean up at once, without waiting out the bound.
    #[test]
    fn a_drained_cohort_cleans_up_immediately() {
        let now = Instant::now();
        assert!(cleanup_due(Some(now), now, 0, false));
    }

    /// Others still decoding: the finished turn waits until the bound, and no longer.
    #[test]
    fn a_finished_turn_waits_for_the_others_up_to_the_bound() {
        let at = Instant::now();
        assert!(!cleanup_due(
            Some(at),
            at + Duration::from_millis(40),
            6,
            false
        ));
        let just_short = at + CLEANUP_DEFER - Duration::from_millis(1);
        assert!(!cleanup_due(Some(at), just_short, 6, false));
        assert!(cleanup_due(Some(at), at + CLEANUP_DEFER, 6, false));
    }

    /// A turn that wrote a tool call is cleaned up the step it finishes, with
    /// others still decoding: its caller is waiting on the `Done` to run the tool.
    #[test]
    fn a_tool_call_turn_is_cleaned_up_without_waiting() {
        let at = Instant::now();
        assert!(cleanup_due(Some(at), at, 6, true));
    }
}

#[cfg(test)]
mod decode_airtime_tests {
    use super::{creep_owed_after, next_creep_owed, next_decode_only};
    use std::time::Duration;

    fn ms(n: u64) -> Duration {
        Duration::from_millis(n)
    }

    /// **The creep rides again once decode is paid, a step or two later.** At
    /// `R = 1` a 120 ms co-batched step beside 40 ms decode-only steps owes its
    /// 80 ms share; two decode-only steps pay it and the third may carry the
    /// creep — where it used to wait for the next 2 s quantum.
    #[test]
    fn decode_only_steps_pay_the_debt_and_free_the_creep() {
        let mut owed = next_creep_owed(Duration::ZERO, ms(120), true, Some(ms(40)), 1.0);
        assert_eq!(owed, ms(80));
        owed = next_creep_owed(owed, ms(40), false, Some(ms(40)), 1.0);
        assert!(!owed.is_zero(), "still owed");
        owed = next_creep_owed(owed, ms(40), false, Some(ms(40)), 1.0);
        assert!(owed.is_zero(), "paid: the next step carries the creep");
    }

    /// **The measured case.** A ~30 ms dialogue decode carrying one layer of an
    /// ingest group took 70 ms (`.5`, 2026-10-17). Behind that `High` decode
    /// (`R = 64`) the 40 ms share owes 2.56 s of decode-only airtime: about 85
    /// decode steps before the ingest rides again, not the next one.
    #[test]
    fn a_throttled_creep_owes_its_share_times_the_ratio() {
        let owed = next_creep_owed(Duration::ZERO, ms(70), true, Some(ms(30)), 64.0);
        assert_eq!(owed, ms(2_560));
        let mut left = owed;
        let mut steps = 0;
        while !left.is_zero() {
            left = next_creep_owed(left, ms(30), false, Some(ms(30)), 64.0);
            steps += 1;
        }
        assert_eq!(steps, 86);
    }

    /// A step that carried the creep sets the debt to its own share rather than
    /// adding to what was owed: it only rode because the last debt was paid.
    #[test]
    fn a_carrying_step_sets_the_debt_to_its_own_share() {
        assert_eq!(
            next_creep_owed(ms(500), ms(300), true, Some(ms(100)), 1.0),
            ms(200)
        );
        assert_eq!(
            next_creep_owed(ms(500), ms(300), true, Some(ms(100)), 16.0),
            ms(3_200)
        );
    }

    /// The measured case: a 4.2 s co-batched step under a `Low` decode (R = 1)
    /// owes the decode its own share — equal airtime.
    #[test]
    fn an_unthrottled_creep_owes_the_decode_its_share() {
        assert_eq!(creep_owed_after(ms(4_200), Some(ms(200)), 1.0), ms(4_000));
    }

    /// Before any decode-only step has been measured the whole step counts as
    /// the creep's: the error is toward decode.
    #[test]
    fn with_no_decode_only_step_measured_the_whole_step_is_owed() {
        assert_eq!(creep_owed_after(ms(70), None, 4.0), ms(280));
    }

    /// A step no longer than a decode-only one owes nothing — the creep cost
    /// decode no airtime — and neither does a zero ratio: a split whose slice
    /// already carries decode's share (`turn_airtime::debt_ratio`). A band's
    /// ratio is at least 1 before it gets here.
    #[test]
    fn a_free_ride_owes_nothing_and_neither_does_a_zero_ratio() {
        assert!(creep_owed_after(ms(25), Some(ms(30)), 64.0).is_zero());
        assert!(creep_owed_after(ms(3_000), Some(ms(1_000)), 0.0).is_zero());
    }

    /// The decode-only average starts at the first step and moves a quarter of
    /// the way to each new one.
    #[test]
    fn the_decode_only_average_follows_the_steps() {
        let a = next_decode_only(None, ms(40));
        assert_eq!(a, ms(40));
        let a = next_decode_only(Some(a), ms(80));
        assert_eq!(a, ms(50));
    }
}
