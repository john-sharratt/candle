//! Samples the first token of each finished prefill and moves the turn into
//! decode, or closes it out at once when it has no decode budget.
//!
//! A prefill leaving the prefill list is an admission opportunity, since the
//! ground it held is free now and the rate model can offer more. The first
//! tokens of every turn that finished in a wave are sampled together in one
//! dispatch, read where the head wrote their rows, and a row holding NaN fails
//! only its own turn. A turn with no decode budget, or whose first token is the
//! end of the sequence, has no reply, so that token is never streamed. The
//! first token also passes through the turn's grammar, stencil trigger and
//! think-block state, so those constraints apply from the first token onward.

use super::*;

impl Scheduler {
    /// Drain finished or errored entries from `active_prefills`. Errored
    /// entries emit `TurnEvent::Error`; finished entries are passed to
    /// `finalise_prefill` (which samples the first token and inserts into
    /// `active_decodes`).
    pub(in super::super) fn promote_finished_prefills_to_decodes(&mut self) {
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
        //
        // **Only a prefill that leaves.** This runs after every decode step, so
        // a flag raised whenever a prefill is in flight would run a whole
        // admission pass a step for nothing that finished; the loop top re-reads
        // prefills still advancing once a quantum (`Scheduler::run`).
        let in_flight = self.active_prefills.len();
        // Turns whose prefill finished cleanly, with the first token the wave
        // that finished them sampled (`sample_first_rows`).
        let mut ready: Vec<(FinishedPrefill, FirstSampled)> = Vec::new();
        // Use swap_remove for efficiency; iterate from the back.
        let mut i = 0;
        while i < self.active_prefills.len() {
            let done = {
                let p = &self.active_prefills[i];
                p.error.is_some() || (p.finished.is_some() && p.offset >= p.work.tokens.len())
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
                finished,
                error,
                prefill_start,
                stops_every: _,
                awaiting: _,
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
                // A first sample taken before the failure holds the sequence's
                // sampling state; it goes back where the next turn finds it.
                if let Some(PrefillEnd::Sampled(s)) = finished {
                    self.sampling_states
                        .insert(work.sequence_id, s.first.sampling_state);
                }
                let _ = work.event_tx.send(TurnEvent::Error(e));
                // Reclaim the carved view, or the sequence wedges forever: the
                // view was registered in `turn_views` before the prefill ran and
                // never reached `active_decodes`, so a dangling one makes the
                // parent's next `SubmitTurn` wind-down refuse with `TurnInFlight`
                // for good. Every prefill-error path drains through here.
                self.discard_turn_view(work.sequence_id);
                continue;
            }
            let sampled = match finished {
                Some(PrefillEnd::Sampled(s)) => s,
                Some(PrefillEnd::Sealed) | None => {
                    let _ = work
                        .event_tx
                        .send(TurnEvent::Error(ConversationError::Channel(
                            "prefill finished without sampling its first token".into(),
                        )));
                    self.discard_turn_view(work.sequence_id);
                    continue;
                }
            };
            let prefill_ms = prefill_start
                .map(|s| s.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            ready.push((
                FinishedPrefill {
                    turn_start: work.submitted_at,
                    work,
                    prefill_ms,
                },
                sampled,
            ));
            // swap_remove pulled the last element into i; don't increment.
        }
        if self.active_prefills.len() < in_flight {
            self.settled_since_admit = true;
        }
        if !ready.is_empty() {
            let t_fin = std::time::Instant::now();
            for (f, s) in ready {
                self.finish_first_sample(f, s.first, s.token);
            }
            crate::scheduler::run::note_promote_split(
                crate::scheduler::run::PromoteStep::Finalise,
                t_fin.elapsed().as_micros() as u64,
            );
        }
    }

    /// Sample the first token of every prefill whose last chunk this wave group
    /// delivered, from its row on the forward's span: prepare each, sample all
    /// in one dispatch, and record the result for promotion.
    ///
    /// **One sample for the cohort, not one per turn.** Each dispatch is a
    /// launch and a readback over the whole vocabulary, so eight turns finishing
    /// the same wave paid eight of them, one after another, before the last
    /// first token went out — 2.5 ms each on the Qwen3.6-35B-A3B. The sampler
    /// already samples a mixed batch row by row on each row's own config and
    /// state, which is what the decode step relies on, so the batch changes no
    /// row's draw. The rows are read where the head wrote them — see
    /// [`Self::sample_first_tokens`].
    ///
    /// A compression re-prefill decodes nothing: it is marked sealed and its row
    /// is not read.
    pub(super) fn sample_first_rows(&mut self, finishing: Vec<(usize, Tensor)>) {
        let mut rows: Vec<Tensor> = Vec::with_capacity(finishing.len());
        let mut prepared: Vec<(usize, FirstSample)> = Vec::with_capacity(finishing.len());
        for (i, row) in finishing {
            if matches!(
                self.active_prefills[i].work.seal_action,
                SealAction::CompressionTurn { .. }
            ) {
                self.active_prefills[i].finished = Some(PrefillEnd::Sealed);
                continue;
            }
            // `[1, vocab]`, a view of the head's row, so adjacent rows name
            // their block whole.
            let row = match row.flatten_all().and_then(|r| r.unsqueeze(0)) {
                Ok(r) => r,
                Err(e) => {
                    self.active_prefills[i].error = Some(ConversationError::Model(e));
                    continue;
                }
            };
            prepared.push((i, self.prepare_first_sample(i)));
            rows.push(row);
        }
        if prepared.is_empty() {
            return;
        }
        let tokens: Vec<Result<u32, ConversationError>> = {
            let _g = super::super::profile::span("finalise:sample");
            self.sample_first_tokens(&rows, &mut prepared)
        };
        for ((i, first), token) in prepared.into_iter().zip(tokens) {
            self.active_prefills[i].finished =
                Some(PrefillEnd::Sampled(FirstSampled { first, token }));
        }
    }

    /// Every prepared turn's first token from its row in `rows`, in order.
    ///
    /// **Sampled where the rows are.** Rows that sit together in the head's
    /// block are named as one `[n, vocab]` view and sampled in one dispatch;
    /// gathering scattered rows into a block of their own would be a
    /// vocabulary-wide carve per finishing turn. A wave whose finishing rows
    /// are one run — the common case, and always when one turn finishes — is a
    /// single dispatch.
    ///
    /// **A row with NaN fails its own turn, not the run's.** A dispatch that
    /// fails is read row by row: a row holding NaN is its own prefill's forward
    /// gone bad and fails, and each row that holds none is sampled again alone.
    /// Scoring leaves every state as it found it (`BatchedSampler::score_rows`)
    /// and the failure is raised before any row commits, so the second draw is
    /// the one the first would have made. Failed together, one character's bad
    /// forward took every turn beside it: on WhiteWorld (2026-11-11) Zen's mind
    /// failed every turn for twenty minutes with its own row clean, sampled
    /// beside another character's NaN.
    pub(super) fn sample_first_tokens(
        &self,
        rows: &[Tensor],
        prepared: &mut [(usize, FirstSample)],
    ) -> Vec<Result<u32, ConversationError>> {
        let mut out: Vec<Result<u32, ConversationError>> = Vec::with_capacity(prepared.len());
        for (start, end) in contiguous_runs(rows) {
            let n = end - start;
            let run = &mut prepared[start..end];
            // Owned, because the states below borrow the same entries mutably.
            let configs: Vec<SamplingConfig> =
                run.iter().map(|(_, s)| s.sampling.clone()).collect();
            let config_refs: Vec<&SamplingConfig> = configs.iter().collect();
            let mut states: Vec<&mut SequenceSamplingState> =
                run.iter_mut().map(|(_, s)| &mut s.sampling_state).collect();
            let block = Tensor::cat_view(&rows[start..end], 0)
                .expect("a run is named as one view by construction");
            match self.sampler.sample_batch(&block, &mut states, &config_refs) {
                Ok(tokens) if tokens.len() == n => out.extend(tokens.into_iter().map(Ok)),
                Ok(tokens) => out.extend((0..n).map(|_| {
                    Err(ConversationError::Channel(format!(
                        "first-token sample returned {} tokens for {n} turns",
                        tokens.len()
                    )))
                })),
                Err(e) => {
                    // Read row by row: a row with NaN fails its own turn, and
                    // a clean one beside it is sampled again alone.
                    let msg = e.to_string();
                    for k in 0..n {
                        let row = &rows[start + k];
                        let seq = self.active_prefills[run[k].0].work.sequence_id;
                        let clean = row_is_nan_free(row);
                        if n > 1 && clean == Some(true) {
                            let alone = Tensor::cat_view(slice::from_ref(row), 0)
                                .expect("one row is named as one view");
                            let mut state = [&mut run[k].1.sampling_state];
                            match self
                                .sampler
                                .sample_batch(&alone, &mut state, &[&configs[k]])
                            {
                                Ok(tokens) if tokens.len() == 1 => {
                                    tracing::warn!(
                                        seq = seq.0,
                                        "first-token sample failed for a block of {n} turn(s); \
                                         this turn's row holds no NaN and was sampled alone: {msg}"
                                    );
                                    out.push(Ok(tokens[0]));
                                    continue;
                                }
                                Ok(tokens) => out.push(Err(ConversationError::Channel(format!(
                                    "first-token sample returned {} tokens for 1 turn",
                                    tokens.len()
                                )))),
                                Err(alone) => out.push(Err(ConversationError::Model(
                                    candle::Error::Msg(format!(
                                        "{alone} (this turn's own row holds no NaN; sampled \
                                         alone after a block of {n} failed: {msg})"
                                    )),
                                ))),
                            }
                            continue;
                        }
                        tracing::warn!(
                            seq = seq.0,
                            row_nan_free = ?clean,
                            "first-token sample failed for a block of {n} turn(s): {msg}"
                        );
                        let why = match clean {
                            Some(false) => format!("{msg} (this turn's prefill row holds NaN)"),
                            Some(true) | None => msg.clone(),
                        };
                        out.push(Err(ConversationError::Model(candle::Error::Msg(why))));
                    }
                }
            }
        }
        out
    }

    /// Everything a turn's first sample needs, taken before the sample: the
    /// turn's budget capped at the RoPE reach, its sampling state brought to the
    /// new turn, and its sampling config with any turn grammar's opening mask.
    ///
    /// `i` indexes `active_prefills`: the turn is readied where it stands.
    pub(super) fn prepare_first_sample(&mut self, i: usize) -> FirstSample {
        let Self {
            session,
            sampling_states,
            sampler,
            active_prefills,
            ..
        } = self;
        let work = &mut active_prefills[i].work;
        let token_count = work.tokens.len();
        // Total KV position after this prefill.
        let context_depth = session
            .sequence_offset(work.sequence_id.0)
            .unwrap_or(token_count);

        // **The turn decodes no further than the model's RoPE reaches.** Its
        // N generated tokens and the trailing structural tokens forwarded at
        // the seal all take positions past `context_depth`, and a header past
        // the schedule's last ceiling is refused — for the whole wave it rides
        // in. So a turn that would outgrow the reach ends where it runs out, as
        // one that spent its budget, and the rest of the wave never sees it.
        let room = session
            .rope_reach()
            .saturating_sub(context_depth + work.post_decode_tokens.len());
        if work.max_decode_tokens > room {
            tracing::warn!(
                seq_id = work.sequence_id.0,
                context_depth,
                budget = work.max_decode_tokens,
                room,
                reach = session.rope_reach(),
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

        let mut sampling_state = sampling_states
            .remove(&work.sequence_id)
            .expect("sampling state must exist for active sequence");
        sampling_state.end_turn(work.sampling.cross_turn_window);
        sampling_state.record_context_tokens(&work.tokens, sampler.max_recent_len());

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
        let mut turn_driver = work.turn_grammar.as_ref().map(TurnGrammar::driver);
        let mut sampling = work.sampling.clone();
        if let Some(driver) = turn_driver.as_mut() {
            let (scaffold, action) = driver.opening();
            match &action {
                // By name, whatever its width: this wave priced no bitmap.
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

        FirstSample {
            context_depth,
            sampling_state,
            sampling,
            turn_driver,
        }
    }

    /// Emit a turn's first token and either move it to decode or close it out
    /// at once on EOS / `max_decode_tokens == 0` — the half of finalising a
    /// prefill that follows its sample.
    pub(super) fn finish_first_sample(
        &mut self,
        finished: FinishedPrefill,
        first: FirstSample,
        sampled: Result<u32, ConversationError>,
    ) {
        let FinishedPrefill {
            mut work,
            prefill_ms,
            turn_start,
        } = finished;
        // `sampling` was this turn's config for the sample just taken; the turn
        // decodes on `work.sampling` from here, as it did before the split.
        let FirstSample {
            context_depth,
            mut sampling_state,
            sampling: _,
            turn_driver,
        } = first;
        let sampled = match sampled {
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
        let first_token = match (work.recorded_reply.as_ref(), self.eos_tokens.first()) {
            (Some(reply), Some(&eos)) if !reply.free_after || !reply.ids.is_empty() => {
                replayed_step(&reply.ids, 0, eos)
            }
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
                // only advances while `in_segment`) — and it keeps DRY off for
                // the block, which applies only to a free-text value outside it.
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
            let first_is_eos = self.is_eos(first_token);
            let finish = if first_is_eos {
                FinishReason::Stop
            } else {
                FinishReason::Length
            };
            // **A turn with no decode budget has no reply.** The token sampled
            // after its prefill is never forwarded — it is the model's guess at
            // what would come next — so it is not streamed, not reported, and not
            // written into the turn's text. A turn WITH a budget whose first token
            // ended it is a one-token reply, and that token reaches the stream on
            // both paths below, which otherwise finish it without decoding.
            let replies = work.max_decode_tokens > 0;
            if replies && streams_committed(first_is_eos) {
                let _ = work.event_tx.send(TurnEvent::Token(first_token));
            }
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
                    faulted: false,
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
                    boundary: BoundaryState::default(),
                    last_projection_end: 0,
                    wrote_tool_call: false,
                    free_tool_calls_from_penalties: work.free_tool_calls_from_penalties,
                    triggers: work.triggers,
                    stencil: None,
                    pending_mask: None,
                    recorded_reply: work.recorded_reply.map(Replay::from),
                };
                if replies {
                    // The turn's first token opens a page at the prefill/decode
                    // boundary, so the reasoning starts one of its own.
                    state.push_committed(first_token, self.think_close, &self.page_break_tokens);
                    // No speculative rewind can be in flight — this turn decodes
                    // nothing more — so the cut is taken at once.
                    super::super::flush_page_cut(self.model.as_ref(), work.sequence_id, &mut state);
                }
                self.active_decodes.insert(work.sequence_id, state);
            } else {
                self.finish_immediately(
                    work.sequence_id,
                    replies.then_some(first_token),
                    &work.event_tx,
                    prefill_ms,
                    turn_start,
                    context_depth,
                    work.tokens.len(),
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
                // The reply's first token: nothing reasoned before it.
                let d = work.triggers.driver_after(first_token, 0);
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
        // mid-call. Such a call decodes on the slot its opening was prefilled
        // against: no boundary fires inside its body.
        let first_token_opens_call = stencil
            .as_ref()
            .is_some_and(|d| d.tree().label() == super::super::TOOL_CALL_TREE_LABEL);

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
            faulted: false,
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
            boundary: BoundaryState::starting(first_token_opens_call),
            last_projection_end: 0,
            wrote_tool_call: first_token_opens_call,
            free_tool_calls_from_penalties: work.free_tool_calls_from_penalties,
            triggers: work.triggers,
            stencil,
            pending_mask: None,
            recorded_reply: work.recorded_reply.map(Replay::from),
        };
        // The turn's first token opens a page at the prefill/decode boundary, so
        // the reasoning starts one of its own.
        state.push_committed(first_token, self.think_close, &self.page_break_tokens);
        // No speculative rewind can be in flight on this path — the turn has not
        // decoded yet — so the cut is taken at once.
        super::super::flush_page_cut(self.model.as_ref(), work.sequence_id, &mut state);
        // No boundary at the end of the opening: the reply decodes on the slot
        // its question was prefilled against, so the recurrent state and the
        // attended K/V agree. The question's hits are counted; the first
        // boundary to apply them is the rule's first in the reply
        // (`provenance::boundaries`).
        self.active_decodes.insert(work.sequence_id, state);
    }
}

/// A turn whose prefill finished cleanly, its first token already drawn.
pub(super) struct FinishedPrefill {
    pub(super) work: PrefillWork,
    pub(super) prefill_ms: f64,
    pub(super) turn_start: Instant,
}
