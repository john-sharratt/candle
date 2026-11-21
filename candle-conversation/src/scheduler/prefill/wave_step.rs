//! Runs one wave step: the decode, verify, glue and creep members that share a
//! forward, with the creep split into segments by the layers it crosses.
//!
//! Full-sweep members, which are decodes, verify blocks and glue, cross every
//! layer. A creep joins only for its own window of layers and holds its residual
//! between steps in its arena. A creep yields to decodes that outrank it at their
//! ratio, and a prompt that outranks a creep in flight pauses it at the layer it
//! reached. A decode-less step is bounded to a slice, so a waiting prompt does not
//! wait out a whole sweep. A segmented step is bracketed so that a later segment's
//! failure rolls back the earlier ones, and a creep's failed window retries the
//! same layers rather than restarting them. The head scores each member's last
//! row, so logits come back in a fixed order.

use super::*;

impl Scheduler {
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
    /// the creep — run as one step (`docs/wave_step.md`): one tier, one forward
    /// span, the full-sweep members' residual handed between segments on it, and
    /// the creep's held between steps in its arena. At the head, per-sequence
    /// logits are `[decode | creep]` (glue logits, if present, trail and are
    /// discarded): prefills promote, sections seal.
    ///
    /// With no creep group, all members are full-sweep: one `[0, N)` forward folding
    /// decode + a standalone section chunk + glue. The glue is a side effect only —
    /// its logits discarded and it must not advance its slot (asserted after).
    ///
    /// Called per decode step; the cohort/section/glue fold in on the first step
    /// (`wave_cohort_advanced` / `wave_section_advanced` guards, `take_wave_glue`
    /// drains once), the rest are plain decode.
    ///
    /// `stencil_seqs`/`stencil_inputs` are the decodes' static runs
    /// (`stencil_wave`): full-sweep members of the prefill slot behind the verify
    /// blocks, each scored at its last row, so the logits come back
    /// `[decode | verify | stencil]`.
    pub(in super::super) fn decode_forward_cobatched(
        &mut self,
        decode_seqs: &[usize],
        decode_inputs: &[Tensor],
        verify_seqs: &[usize],
        verify_inputs: &[Tensor],
        stencil_seqs: &[usize],
        stencil_inputs: &[Tensor],
    ) -> candle::Result<WaveLogits> {
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
        let out = self.wave_step(
            decode_seqs,
            decode_inputs,
            verify_seqs,
            verify_inputs,
            stencil_seqs,
            stencil_inputs,
            glue,
        );
        // **A segmented step stands or falls whole for its full-sweep members.**
        // A refused later segment rolled back only its own layers; the ones
        // before it committed, and the retry would run their tokens in again —
        // a qwen4exp index a verify block ahead of its K/V per refusal, the
        // K/V a token long on those layers, the recurrence advanced twice.
        let out = match (self.wave_step_open.take(), out) {
            (None, out) => out,
            (Some(_), Ok(l)) => {
                if let Err(e) = self.model.end_step(&mut self.session, true, 0) {
                    tracing::warn!("closing a wave step that ran: {e}");
                }
                Ok(l)
            }
            (Some(committed), Err(e)) => {
                match self.model.end_step(&mut self.session, false, committed) {
                    Ok(()) => {
                        if committed > 0 {
                            tracing::info!(
                                committed,
                                "a wave step failed past its first segment; the layers its \
                                 earlier segments committed were taken back"
                            );
                        }
                        Err(e)
                    }
                    // Not the original error: a refusal is retried, and a retry
                    // on state the undo could not restore is what this exists
                    // to prevent. Worded without the refusal's marker so the
                    // turns fail instead.
                    Err(u) => {
                        tracing::error!(
                            committed,
                            error = %e,
                            "a wave step failed and its undo failed too; its decodes fail: {u}"
                        );
                        Err(candle::Error::Msg(format!(
                            "a wave step's segment failed and the step could not be undone \
                             ({u}) — the sequences' carried state is not trustworthy"
                        )))
                    }
                }
            }
        };
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
    #[allow(clippy::too_many_arguments)]
    pub(super) fn wave_step(
        &mut self,
        decode_seqs: &[usize],
        decode_inputs: &[Tensor],
        verify_seqs: &[usize],
        verify_inputs: &[Tensor],
        stencil_seqs: &[usize],
        stencil_inputs: &[Tensor],
        glue: Option<(Vec<usize>, Vec<Tensor>, Vec<PendingGlue>)>,
    ) -> candle::Result<WaveLogits> {
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
        // caller gets back: `[decode | verify | stencil]`.
        let head_rows = n_dec + verify_tok + stencil_seqs.len();
        // The static runs follow the verify blocks as full-sweep members of the
        // same slot for the same reason — the step samples the token after each
        // from its run's row — but a run is not a verify block: nothing armed it,
        // so the head scores its last row only. From here on the slot's
        // full-sweep members are both.
        let full_seqs: Vec<usize> = verify_seqs.iter().chain(stencil_seqs).copied().collect();
        let full_inputs: Vec<Tensor> = verify_inputs
            .iter()
            .chain(stencil_inputs)
            .cloned()
            .collect();
        let verify_seqs = full_seqs.as_slice();
        let verify_inputs = full_inputs.as_slice();
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
        // Glue rows are prompt rows too — a tool loop's results ride the decode
        // sweep as glue — so the `High` ones count toward the drain (`high_drain`)
        // once the forward that carried them has run.
        let high_glue: usize = glue_seqs
            .iter()
            .zip(glue_inputs)
            .filter(|(&id, _)| self.decode_priority_or_high(SequenceId(id)).is_live())
            .map(|(_, t)| t.dims().get(1).copied().unwrap_or(0))
            .sum();
        // A "full-sweep" wave carries decode and/or glue across all N layers; it
        // drives segments 1 and 3. With neither, only the creep runs (seg 2).
        let has_fullsweep = n_dec > 0 || has_glue;

        // Not `let`: pausing, resuming, or a creep whose members all left
        // mid-sweep moves it.
        let mut cursor = self.wave_prefill_cursor;
        // **A prompt that outranks the creep in flight pauses it.** A group
        // only re-forms at its sweep boundary, and a throttled group takes `R`
        // steps to reach it: an ingest group creeping one layer a step behind a
        // cast's decodes held a dialogue's tool result back for all of them. It
        // is set aside at the layer it reached ([`PausedCreep`]), the prompt's
        // band runs, and it resumes there. It is never restarted — its finished
        // layers hold its rows in their carried state (`reset_wave_prefill`) —
        // and with a creep already paused, the one in flight finishes its
        // remaining layers this step instead.
        let mut finish_sweep = false;

        // Form/resume the creep group (dialogue prefills + section chunks) unless it
        // was already advanced this wave. A fresh group folds section chunks in to
        // co-batch the creep (`form_wave_group(true)`), unless the standalone
        // section pass already ran this wave (no decode present).
        let (members, seq_ids, inputs, prefill_gidxs) = if !self.wave_cohort_advanced {
            if (cursor > 0 || self.wave_prefill_residual.is_some())
                && self.creep_outranked_by_a_waiting_prompt()
            {
                if self.paused_creep.is_none() {
                    self.pause_creep();
                    cursor = 0;
                } else {
                    finish_sweep = true;
                }
            }
            if cursor == 0 && self.wave_prefill_residual.is_none() {
                if self.resume_creep() {
                    cursor = self.wave_prefill_cursor;
                } else {
                    let _g = super::super::profile::span("loop:wave:form_group");
                    self.form_wave_group(!self.wave_section_advanced);
                }
            }
            let _g = super::super::profile::span("loop:wave:build_inputs");
            let built = self.build_wave_group_inputs();
            // **A member that left mid-sweep takes its rows with it.**
            // `build_wave_group_inputs` drops a member that errored or whose
            // slot went, and the held residual still carries its rows. Taken
            // out here, the survivors keep the layers they have run.
            if cursor > 0 && built.0.len() != self.wave_prefill_members.len() {
                let mut kept = built.0.iter().map(|m| m.seq_id()).peekable();
                let keep: Vec<bool> = self
                    .wave_prefill_members
                    .iter()
                    .map(|m| {
                        let here = kept.peek() == Some(&m.seq_id());
                        if here {
                            kept.next();
                        }
                        here
                    })
                    .collect();
                tracing::warn!(
                    held = self.wave_prefill_members.len(),
                    kept = built.0.len(),
                    cursor,
                    "creep members left mid-sweep; taking their rows out of the held residual"
                );
                self.drop_wave_members(&keep)?;
                if self.wave_prefill_members.is_empty() {
                    cursor = 0;
                }
            }
            // **The held residual must be exactly the survivors' rows.** Paired
            // with rows it does not hold, the remaining layers would attribute
            // activations to the wrong members — which is what makes a sampler
            // emit token 0 forever. Nothing can be replayed either (see
            // `reset_wave_prefill`), so the creep's members fail, loudly, and
            // this step runs without them. Not an assert: a panic on the
            // scheduler thread takes the daemon with it.
            let rows: usize = built
                .2
                .iter()
                .map(|t| t.dims().get(1).copied().unwrap_or(0))
                .sum();
            let held = self.wave_prefill_residual.as_ref().map(CreepResidual::rows);
            match held {
                Some(held) if cursor > 0 && held != rows => {
                    tracing::error!(
                        held_residual_tokens = held,
                        creep_tokens = rows,
                        cursor,
                        members = built.0.len(),
                        "the creep's held residual does not match its members mid-sweep; \
                         failing them rather than replaying layers they already ran"
                    );
                    self.fail_wave_members(
                        "the prefill's creep lost track of its rows mid-sweep; retry the turn",
                    );
                    cursor = 0;
                    (Vec::new(), Vec::new(), Vec::new(), Vec::new())
                }
                _ => built,
            }
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        self.wave_cobatched = has_glue || !seq_ids.is_empty();

        // The creep yields to the decodes that protect themselves from it
        // (`throttle_ratio`). Without a creep the step co-batches glue and a
        // section chunk, at the band of the sections it takes
        // (`section_batch_band`).
        //
        // **A dialogue's own section is not background.** Counted as `Low`,
        // a journal entry's chunk beside the cast's `High` decodes owed decode
        // 64 times its share of the step, and while that debt ran no creep
        // formed: on WhiteWorld (2026-10-26) four Makers' prompts sat at a
        // fixed backlog for 40–60 s at a time while one decode ran alone, and
        // prefilled at ~900 tok/s only in the gaps where nothing decoded.
        let work = if seq_ids.is_empty() {
            self.section_batch_band().unwrap_or(DecodePriority::Low)
        } else {
            self.creep_priority(&members)
        };
        self.wave_step_band = work;
        let band = self.airtime_ratio_over(work);
        // **A `High` prompt beside `High` decodes splits the device by the work
        // each side holds** (`turn_airtime`): the smaller backlog first, and a
        // prompt that has waited its own time at the front. The band's constant
        // stands until both sides are measured, and below `High` the bands decide
        // alone — background ingest yields to dialogue whatever the backlogs say.
        let derived = (work.is_live() && band == HIGH_BESIDE_HIGH_RATIO)
            .then(|| {
                super::super::turn_airtime::high_airtime_ratio(
                    self.segment_mean.tokens()?,
                    self.decode_only_step?,
                    self.high_backlog_rows,
                    self.creep_airtime.per_s()?,
                    &self.high_prompt_waits(),
                )
            })
            .flatten();
        if derived.is_some() {
            self.high_airtime = derived;
        }
        let ratio = derived.unwrap_or(band.max(1) as f64);
        let owes = owes_decode(
            work,
            self.foreground_decodes()
                .map(|sid| self.decode_priority_or_high(sid)),
        );
        self.wave_step_ratio = match derived {
            Some(r) => super::super::turn_airtime::debt_ratio(r),
            None if owes => ratio,
            None => 0.0,
        };
        self.wave_step_slice = super::super::turn_airtime::slice_steps(
            ratio,
            self.decode_only_step.unwrap_or(std::time::Duration::ZERO),
        );
        // A creep advances `ceil(N / R)` layers — and beside decode rows it
        // yields to, no more than a decode-only step's worth of them
        // (`creep_slice`), times what `R < 1` lets a slice carry, so the share
        // `R` sets is paid in steps, not seconds. Beside only its own band's
        // decodes it takes the arrival bound instead (`slice_target`).
        let creep_rows: usize = inputs
            .iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(0))
            .sum();
        // A model's draft head runs over every row of the step that completes
        // the trunk, so its blocks are layers of the sweep: budgeted, cut at by
        // `window_end`, and counted in what the step teaches the cost.
        let head = self.model.sweep_head_layers();
        let left = n - cursor.min(n) + head;
        let budget = if finish_sweep {
            left
        } else if head_rows > 0 {
            time_bounded_layers(
                creep_layer_budget(n, ratio.ceil() as u32).min(left),
                creep_rows,
                self.creep_layer_row_cost,
                self.creep_layer_cost,
                slice_target(
                    band > 1 || derived.is_some(),
                    self.decode_only_step,
                    self.wave_step_slice,
                    DECODE_LESS_SLICE,
                ),
            )
        } else {
            // **Decode-less, still bounded in time.** With nothing decoding
            // there is no decode cadence to protect, but there is a prompt that
            // may arrive: it can only take the device at a step boundary
            // (`creep_outranked_by_a_waiting_prompt` pauses the creep there), so
            // a whole-sweep step made it wait the sweep. Measured on `.5`
            // (2026-10-18): a passthrough chat's every next request found an
            // 8,190-row ingest cohort starting a 2.6 s, 48-layer decode-less
            // step in the gap between two of its turns, and waited it out.
            time_bounded_layers(
                creep_layer_budget(n, ratio.ceil() as u32).min(left),
                creep_rows,
                self.creep_layer_row_cost,
                self.creep_layer_cost,
                Some(DECODE_LESS_SLICE),
            )
        };
        let (win_end, head_run) = window_end(cursor.min(n), budget, n, head);
        // What the creep ran this step, for the next slice's cost estimate:
        // beside decode its excess over a decode-only step, alone its whole step.
        // The head's blocks count as layers the step ran.
        let swept = win_end - cursor.min(win_end) + head_run;
        self.wave_step_creep_layer_rows = if !seq_ids.is_empty() {
            creep_rows * swept
        } else {
            0
        };
        self.wave_step_creep_layers = if !seq_ids.is_empty() { swept } else { 0 };
        // And the query×key pairs those rows attend through the same layers —
        // the creep's cost at depth, which its rows alone do not say.
        self.wave_step_creep_layer_pairs = if !seq_ids.is_empty() {
            members
                .iter()
                .zip(inputs.iter())
                .map(|(m, inp)| {
                    let tok = inp.dims().get(1).copied().unwrap_or(0) as f64;
                    let at = self.session.sequence_offset(m.seq_id()).unwrap_or(0) as f64;
                    tok * (at + tok / 2.0)
                })
                .sum::<f64>()
                * swept as f64
        } else {
            0.0
        };

        // No creep group → one full-sweep [0, N) forward folding decode + a
        // standalone section chunk (if pending) + glue. All full-sweep, no residual
        // to hold; logits `[decode | section]` split at n_dec (glue logits, if any,
        // trail and are discarded).
        if seq_ids.is_empty() {
            let section = if !self.wave_cohort_advanced && !self.wave_section_advanced {
                // **Beside a person's turn, only for a section batch there is.**
                // While an `Urgent` decode runs — someone waiting on it token by
                // token — this relief is for the section batch alone, and runs
                // only when there is one to build (`section_batch_band`, the
                // predicate the build asks first). Run on every step regardless,
                // it packed for a batch that did not exist: on BlackWorld
                // (2026-11-09, one dialogue reprojecting over 200K of history,
                // `queued_sections=0` throughout) five of these in one reply
                // spent 300–1,050 ms each repacking the pools after the
                // projection's recalls, and every decode wave behind one ran at
                // 60–98 ms a forward against 34–39. With no one waiting it also
                // keeps the setpoint for the work to come.
                //
                // **Paced either way** (`relief_pacing`): at most a tenth of the
                // scheduler's time, each wait priced by the last relief's cost.
                let urgent = self
                    .foreground_decodes()
                    .any(|sid| self.decode_priority_or_high(sid) == DecodePriority::Urgent);
                let wanted = !urgent || self.section_batch_band().is_some();
                let now = std::time::Instant::now();
                if wanted && self.section_relief_pacing.ready(now) && self.vram_under_pressure() {
                    self.relieve_vram_pressure("section", VramPhase::Load, ReliefNeed::Paced);
                    let ended = std::time::Instant::now();
                    self.section_relief_pacing.ran(ended, ended - now);
                }
                let _g = super::super::profile::span("loop:wave:build_section");
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
                return Ok(WaveLogits::empty());
            }
            if !sec_seqs.is_empty() {
                self.wave_section_advanced = true;
                self.wave_cobatched = true;
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
                let _g = super::super::profile::span("loop:wave:forward");
                match self.model.forward_wave(
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
                ) {
                    Ok(out) => out,
                    Err(e) => {
                        self.fail_section_batch(&sec_gidx, &sec_adv, &e);
                        return Err(e);
                    }
                }
            };
            if has_glue {
                let _g = super::super::profile::span("loop:wave:reconcile_offsets");
                self.reconcile_wave_offsets(glue_seqs)?;
                self.high_drain.drained(high_glue);
            }
            // Read in place behind the forward's guard — see `WaveLogits`.
            let dec_logits = WaveLogits::held(out, head_rows);
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
                self.wave_stats.record_cobatched_split(
                    cobatched_decode_rows(decode_inputs, verify_inputs),
                    0,
                    sec_adv.iter().sum(),
                );
                super::super::PREFILL_OK_TOKENS.fetch_add(
                    sec_adv.iter().sum::<usize>() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                self.complete_section_chunk(&sec_gidx, &sec_adv);
            }
            return Ok(dec_logits);
        }

        // Creep group present: one step (`docs/wave_step.md`). The full-sweep
        // members — decode, verify blocks, glue — cross every layer; the creep
        // crosses `[cursor, win_end)`, resuming from its arena's committed rows
        // and, when the window stops short, leaving its rows there for the next
        // step. Verify blocks lead the prefill slot and the creep follows, so
        // `[decode | verify]` is the contiguous head every segment shares and the
        // caller reads its logits from.
        self.wave_cohort_advanced = true;
        let creep_tok = creep_rows;

        // A step of more than one segment carrying full-sweep members is opened
        // on the model, so a later segment's failure takes the earlier ones back
        // for them (closed in `decode_forward_cobatched`). The creep needs none:
        // it holds its residual and cursor and retries only the failed window.
        // Nor does glue: its K/V is reserved past its offset, and a failed step
        // clears its placed pieces so the slot rebuilds them anyway.
        let full_sweep: Vec<usize> = decode_seqs.iter().chain(verify_seqs).copied().collect();
        if !full_sweep.is_empty() && (cursor > 0 || win_end < n) {
            // Unbracketed, a later segment's failure is the bug the bracket
            // exists for, so the step does not run: the error carries no
            // refusal marker, and the decodes fail rather than retry.
            if let Err(e) = self.model.begin_step(&full_sweep) {
                return Err(candle::Error::Msg(format!(
                    "a segmented wave step could not open its step bracket: {e}"
                )));
            }
            self.wave_step_open = Some(0);
        }

        if let Some(p) = glue_pending {
            self.session.set_pending_glue(p.clone());
        }
        let stepped = {
            let _g = super::super::profile::span("loop:wave:forward");
            self.model.forward_step(
                &mut self.session,
                StepSpec {
                    decode: StepRows {
                        seqs: decode_seqs,
                        inputs: decode_inputs,
                    },
                    verify: StepRows {
                        seqs: verify_seqs,
                        inputs: verify_inputs,
                    },
                    glue: StepRows {
                        seqs: glue_seqs,
                        inputs: glue_inputs,
                    },
                    creep: Some(StepCreep {
                        rows: StepRows {
                            seqs: &seq_ids,
                            inputs: &inputs,
                        },
                        cursor,
                        win_end,
                        residual: &mut self.wave_prefill_residual,
                    }),
                },
            )
        };
        // The trunk layers the step's segments committed for the full-sweep
        // members — `[0, cursor)`, the creep's window, `[win_end, n)` in turn
        // (`drive_step`) — which a failed step takes back. A layer counted that
        // did not commit is truncated to the offset it already holds.
        if let Some(committed) = self.wave_step_open.as_mut() {
            *committed = match &stepped {
                Ok(s) if s.head.is_ok() => n,
                Ok(s) if s.creep_advanced => win_end,
                Ok(_) => 0,
                Err(_) => cursor,
            };
        }
        let stepped = match stepped {
            Ok(s) => {
                self.wave_sweep_failures = 0;
                s
            }
            Err(e) => {
                // **The creep's window failed, and a retryable failure retries
                // the same window.** The forward's failure bracket rolled back
                // what this window did, the arena's committed half is as the
                // window found it, and `[0, cursor)` stands — so the group keeps
                // its residual and its cursor rather than re-forming, which would
                // run those layers' rows into their carried state again
                // (`reset_wave_prefill`). Bounded: a window refused
                // `MID_SWEEP_RETRIES` times in a row fails its members instead.
                let retryable = is_device_oom(&e)
                    || tier_refusal_shortfall(&e).is_some()
                    || is_span_full(&e)
                    || is_expert_stall(&e)
                    || candle_nn::kv_cache::is_arena_exhausted(&e);
                if cursor > 0 && retryable && self.wave_sweep_failures < MID_SWEEP_RETRIES {
                    self.wave_sweep_failures += 1;
                    // The ground the window was refused is often standing below
                    // the frontier as holes: move the frontier down before the
                    // retry, or the retry finds the same ground and fails again.
                    if let Some(short) = tier_refusal_shortfall(&e) {
                        self.lower_frontier_for(short);
                    }
                    // The K/V pools as they stand, against the formation line's.
                    let (kv_arenas, kv_live_mib) =
                        self.session.kv_gpu_class_stats().map_or((0, 0), |s| {
                            (
                                s.total_arenas(),
                                s.classes.iter().map(|c| c.live_bytes).sum::<usize>() >> 20,
                            )
                        });
                    tracing::warn!(
                        cursor,
                        win_end,
                        rows = creep_tok,
                        attempt = self.wave_sweep_failures,
                        kv_arenas,
                        kv_live_mib,
                        error = %e,
                        "creep window failed mid-sweep; holding the sweep to retry it"
                    );
                    return Err(e);
                }
                self.wave_sweep_failures = 0;
                if cursor > 0 {
                    // **The width this window was refused is the next group's
                    // too, unless it is told.** The members fail — nothing
                    // mid-sweep may be retried from layer 0 — but the group
                    // that forms next is composed against the same cap, so
                    // without the refusal noted it formed at the same width and
                    // failed at the same window: on the RTX 3090 (2026-10-18)
                    // an 8,192-row cohort every five seconds, each failing its
                    // file ingests at the first full-attention layer.
                    if let Some(short) = tier_refusal_shortfall(&e) {
                        self.note_refused_width(creep_tok, short);
                    }
                    self.fail_wave_members(&format!("wave group forward failed mid-sweep: {e}"));
                } else {
                    // Drop the creep group cleanly (requeue scope prefills on OOM)
                    // so a fresh one forms next wave, then surface the error.
                    self.fail_wave_group(&members, &prefill_gidxs, creep_tok, &e);
                }
                return Err(e);
            }
        };
        // A segment that carried only the full-sweep rows failed before the
        // creep's window ran: nothing of the creep moved, and the step's error is
        // the caller's.
        if !stepped.creep_advanced {
            return match stepped.head {
                Err(e) => Err(e),
                Ok(_) => Err(candle::Error::Msg(
                    "co-batch wave: the step ran without its creep window".into(),
                )),
            };
        }
        // The work ahead of every waiting turn moved (`prefill_wait`).
        self.prefill_waits.window_ran();

        // Record the co-batched creep throughput — prefill and section members are
        // tallied into their own channels, sharing the wall-clock of the segment
        // the creep rode (concurrently with decode) so the dashboard shows their
        // CONCURRENT rate instead of reading zero. The full-sweep segments around
        // it, which the creep did not ride, are not charged to it: that would
        // understate the prefill rate. One record per wave; KV is summed now,
        // before `complete_wave_group` advances the sequences at the head.
        let creep_us = stepped.creep_us;
        {
            let ms = creep_us / 1000;
            let (mut pf_seqs, mut pf_tok, mut pf_kv) = (0usize, 0usize, 0usize);
            let (mut sc_seqs, mut sc_tok, mut sc_kv) = (0usize, 0usize, 0usize);
            let mut high_tok = 0usize;
            // Query×key pairs the members' rows attend over the sweep: each row
            // the context before it, for the rate model's depth term.
            let (mut pf_pairs, mut sc_pairs) = (0f64, 0f64);
            let pairs_of = |tok: usize, at: usize| tok as f64 * (at as f64 + tok as f64 / 2.0);
            for (m, inp) in members.iter().zip(inputs.iter()) {
                let tok = inp.dims().get(1).copied().unwrap_or(0);
                match m {
                    WaveMember::Prefill { seq_id, .. } => {
                        pf_seqs += 1;
                        pf_tok += tok;
                        let at = self.session.sequence_offset(*seq_id).unwrap_or(0);
                        pf_pairs += pairs_of(tok, at);
                        pf_kv += at;
                        if self.decode_priority_or_high(SequenceId(*seq_id)).is_live() {
                            high_tok += tok;
                        }
                    }
                    WaveMember::Section { seq_id, .. } => {
                        sc_seqs += 1;
                        sc_tok += tok;
                        let at = self.session.sequence_offset(*seq_id).unwrap_or(0);
                        sc_pairs += pairs_of(tok, at);
                        sc_kv += at;
                    }
                }
            }
            // The creep rode layers `[cursor, win_end)` only: what it prefilled
            // this wave is that window's share of its tokens, so a sweep split
            // over several waves counts its tokens once.
            let pf_swept = window_tokens(pf_tok, cursor, win_end, n);
            let sc_swept = window_tokens(sc_tok, cursor, win_end, n);
            // What the `High` backlog drained by, for the rate this card advertises.
            self.high_drain
                .drained(window_tokens(high_tok, cursor, win_end, n));
            // The creep's speed per second it held the device: its window's
            // time, less what the decode rows beside it would have taken over
            // those layers alone (`turn_airtime`).
            let decode_part = match (head_rows > 0, self.decode_only_step) {
                (true, Some(d)) => {
                    d.mul_f64((win_end - cursor.min(win_end)) as f64 / n.max(1) as f64)
                }
                _ => std::time::Duration::ZERO,
            };
            self.creep_airtime.add(
                pf_swept + sc_swept,
                std::time::Duration::from_micros(creep_us).saturating_sub(decode_part),
            );
            if pf_seqs > 0 {
                self.wave_stats.record(true, pf_seqs, pf_swept, pf_kv, ms);
            }
            if sc_seqs > 0 {
                self.wave_stats.record_section(sc_seqs, sc_swept, sc_kv, ms);
            }
            // The phase timeline's share of this sweep: the rows that rode it,
            // which the decode step it ran in is split by.
            let decode_rows = cobatched_decode_rows(decode_inputs, verify_inputs);
            self.wave_stats
                .record_cobatched_split(decode_rows, pf_tok, sc_tok);
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
            // copy is per forward, and they rode the same one. The model prices a
            // whole sweep, so a window's time is taken to the sweep it is part of.
            self.observe_prefill_forward(
                pf_tok + sc_tok,
                pf_pairs + sc_pairs,
                full_sweep_us(creep_us, cursor, win_end, n),
            );
            // Every completed wave forward is OOM-free prefill throughput —
            // the progress signal the stall-grace gate and evidence reopen
            // read. Without this, pump-driven phases (scope ingest, section
            // creep) look stalled to the admission regulator even at full
            // throughput, because only the drain-path prefills tick it.
            if pf_swept + sc_swept > 0 {
                super::super::PREFILL_OK_TOKENS.fetch_add(
                    (pf_swept + sc_swept) as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
        }

        if win_end >= n {
            // Head reached: per-sequence logits, caller order `[decode | creep |
            // glue]`. Decode first; creep members next (promote/seal); glue logits,
            // if present, trail and are discarded.
            if has_glue {
                let _g = super::super::profile::span("loop:wave:reconcile_offsets");
                self.reconcile_wave_offsets(glue_seqs)?;
                self.high_drain.drained(high_glue);
            }
            // Read in place behind the step's guard — see `WaveLogits`. The
            // creep's rows are consumed here; `complete_wave_group` copies the
            // one row a finished prefill keeps past this wave.
            let head = stepped.head?;
            let logits = head.logits_on_span();
            // **The head scores the step's own rows or the step names itself.**
            // Decode and every verify row first, then each creep member's last
            // row, then each glue sequence's: a head that returned fewer gave
            // some of them nobody's logits, and the verify readback fails every
            // turn in the wave. On BlackWorld (9d69e59c5, 2026-10-20) a step
            // carrying 1 plain decode, 7 verify blocks (33 rows) and a 16-member
            // creep scored 18 rows; this is what says which part was dropped.
            let scored_want = head_rows + members.len() + glue_seqs.len();
            if logits.len() != scored_want {
                let blocks: Vec<usize> = verify_inputs
                    .iter()
                    .map(|t| t.dims().get(1).copied().unwrap_or(0))
                    .collect();
                let creep_rows: Vec<usize> = inputs
                    .iter()
                    .map(|t| t.dims().get(1).copied().unwrap_or(0))
                    .collect();
                tracing::error!(
                    target: "candle_conversation::scheduler::prefill",
                    scored = logits.len(),
                    want = scored_want,
                    decode = n_dec,
                    verify_seqs = ?verify_seqs,
                    verify_blocks = ?blocks,
                    creep_members = members.len(),
                    creep_rows = ?creep_rows,
                    glue = glue_seqs.len(),
                    cursor,
                    win_end,
                    layers = n,
                    "the step's head scored a different number of rows than it carried"
                );
            }
            let d = head_rows.min(logits.len());
            let creep_end = (d + members.len()).min(logits.len());
            // Timed only when it crept: a sweep finished in one step placed one
            // tier, and nothing could take its ground between windows.
            if cursor > 0 {
                self.sweep_forecast.sweep_done(std::time::Instant::now());
            } else {
                self.sweep_forecast.sweep_abandoned();
            }
            let held = self.held_arena_bytes();
            self.sweep_forecast.held_ended(held);
            self.complete_wave_group(&members, &logits[d..creep_end]);
            return Ok(WaveLogits::held_rows(head, logits[..d].to_vec()));
        }

        // The window stopped short: the creep's rows are in its arena, and it
        // resumes at `win_end`. Each prefill member is told how far its chunk
        // has run, so a cohort crawling a layer a window beside a person's
        // decode reads to its caller as moving, not as never admitted.
        for m in &members {
            let WaveMember::Prefill { seq_id, .. } = *m else {
                continue;
            };
            if let Some(p) = self
                .active_prefills
                .iter()
                .find(|p| p.work.sequence_id.0 == seq_id)
            {
                let _ = p.work.event_tx.send(TurnEvent::PrefillLayers {
                    layers_done: win_end,
                    layers_total: n,
                });
            }
        }
        self.wave_prefill_cursor = win_end;
        self.wave_prefill_members = members;
        self.wave_prefill_rows = inputs
            .iter()
            .map(|t| t.dims().get(1).copied().unwrap_or(0))
            .collect();
        // The full-sweep members' last segment ran after the creep committed; its
        // failure is the step's, with the creep already advanced.
        let head = stepped.head?;
        if !has_fullsweep && verify_seqs.is_empty() {
            return Ok(WaveLogits::empty());
        }
        if has_glue {
            let _g = super::super::profile::span("loop:wave:reconcile_offsets");
            self.reconcile_wave_offsets(glue_seqs)?;
            self.high_drain.drained(high_glue);
        }
        Ok(WaveLogits::held(head, head_rows))
    }
}
