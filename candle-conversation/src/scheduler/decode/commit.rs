//! Commits each decoded token through the one path every token takes, and
//! forwards the prefixes of healed tokens once the KV is final.
//!
//! The stages run in a fixed order, because each guard depends on the one before
//! it: the stencil advance, healing, the think-block and penalty bookkeeping, the
//! health checks, the generated-token push, termination, and finally the
//! reprojection triggers. A suppressed close is dropped before any of that, so it
//! reaches neither the output nor the sequence. The same path serves a
//! single-token step and each position of a speculative block. Healed prefixes
//! are held until the rollback has settled, since until then the KV runs past the
//! healed position.

use super::*;
use crate::scheduler::think_gate::{reprojects, Thinking};

/// The leading tokens of a healed token, waiting to be forwarded.
///
/// A heal commits a sampled token as other text, and when that text is several
/// tokens long, all but the last have to be written into the sequence's KV —
/// the last rides the next decode like any committed token. They cannot be
/// written at commit time: inside a speculative step the wave has written the
/// whole verified block, so the KV runs past the healed position until the
/// rollback trims it. The commit records them; the step forwards them after the
/// rollback, where the KV ends at the last committed position.
pub(in super::super) struct HealPrefill {
    pub(super) seq: SequenceId,
    pub(super) tokens: Vec<u32>,
}

impl Scheduler {
    /// Sample and commit the token after each sequence's static run, from the
    /// row the wave that wrote the run left.
    ///
    /// **This is what keeps a static run inside one forward.** The wave already
    /// computed the row after the run's last token; forwarding that token again
    /// as the next decode step's row, only to read the same distribution, was a
    /// second pass. The token goes through [`Self::commit_decoded_tokens`], the
    /// path every decoded token takes, against the row that produced it. The
    /// sequence rode this step as its run rather than as a decode row: its
    /// stencil has advanced, and the constraint for the token after this one is
    /// the next play's to set.
    pub(in super::super) fn commit_after_injection(
        &mut self,
        mut rows: HashMap<SequenceId, Tensor>,
    ) {
        let mut ids: Vec<SequenceId> = rows
            .keys()
            .copied()
            .filter(|id| self.active_decodes.get(id).is_some_and(|s| !s.finished))
            .collect();
        ids.sort_by_key(|id| id.0);
        if ids.is_empty() {
            return;
        }
        // `[1, vocab]` each, as the decode step's rows are.
        let shaped: candle::Result<Vec<Tensor>> = ids
            .iter()
            .map(|id| {
                rows.remove(id)
                    .expect("ids are the rows' keys")
                    .flatten_all()
                    .and_then(|r| r.unsqueeze(0))
            })
            .collect();
        let (row_list, stacked) = match shaped.and_then(|r| hold_rows(&r).map(|s| (r, s))) {
            Ok(pair) => pair,
            Err(e) => {
                self.fail_all_decodes(&ids, &format!("stencil continuation rows: {e}"));
                return;
            }
        };
        let configs = self.step_configs(&ids);
        let (state_ids, mut states): (Vec<SequenceId>, Vec<SequenceSamplingState>) =
            self.take_sampling_states(&ids).into_iter().unzip();
        let sampled = {
            let config_refs: Vec<&SamplingConfig> = configs.iter().collect();
            let mut state_refs: Vec<&mut SequenceSamplingState> = states.iter_mut().collect();
            self.sampler
                .sample_batch(&stacked, &mut state_refs, &config_refs)
        };
        for (id, state) in state_ids.into_iter().zip(states) {
            // A plain step's forward has committed and folded every row it ran,
            // so the state is settled and can be described at once.
            if state.degenerate && self.fail_degenerate(id) {
                describe_recurrent_state(self.model.as_ref(), id);
            }
            self.sampling_states.insert(id, state);
        }
        let mut tokens = match sampled {
            Ok(t) if t.len() == ids.len() => t,
            Ok(t) => {
                self.fail_all_decodes(
                    &ids,
                    &format!(
                        "stencil continuation sampled {} tokens for {}",
                        t.len(),
                        ids.len()
                    ),
                );
                return;
            }
            Err(e) => {
                self.describe_nonfinite_step("stencil", &ids, &e);
                self.fail_all_decodes(&ids, &format!("stencil continuation sample failed: {e}"));
                return;
            }
        };
        let heals = self.commit_decoded_tokens(&ids, &mut tokens, &row_list);
        // Nothing to roll back: the KV ends at the run's last token, which is
        // exactly where a healed prefix is forwarded from.
        for heal in heals {
            self.forward_heal_prefill(heal);
        }
        // A run whose last token is a break leaves the cut to the token after it
        // (`push_committed` records it), and the KV already ends at the break —
        // the edge the cut belongs on — so it is taken now, as the decode step
        // takes its own once its rollback has settled.
        for &id in &ids {
            if let Some(state) = self.active_decodes.get_mut(&id) {
                super::super::flush_page_cut(self.model.as_ref(), id, state);
            }
        }
    }

    /// Commit one decoded token per sequence: everything that happens to a token
    /// between "the sampler chose it" and "the turn has moved on".
    ///
    /// Stencil advance and exit-token healing, the think-block and DRY span
    /// edges, every decode-health check, the generated-token push, EOS/budget
    /// termination, the stream event, and the reprojection triggers — in that
    /// order, because each stage's guards depend on the ones before it (a
    /// dropped `</think>` must not reach the EOS seal; a heal rewrites
    /// `next_tokens[i]` before the token is committed anywhere).
    ///
    /// **One call commits one token per listed sequence, and `seq_ids` is the
    /// cohort for THIS token, not for the wave.** Plain decode calls it once
    /// with every sequence in the forward. Speculative decode calls it once per
    /// accepted block position with the sequences still alive at that position,
    /// so a multi-token step runs each token through the identical path a
    /// single-token step would have — which is the whole reason this is a
    /// function and not the tail of [`Self::batch_decode_step`].
    ///
    /// `logits_vec[i]` must be the row that PRODUCED `next_tokens[i]`: the
    /// health checks read its distribution, so handing over a later position's
    /// row would judge a token against logits that did not choose it.
    ///
    /// Returns the prefixes of healed tokens, which the caller forwards with
    /// [`Self::forward_heal_prefill`] once each sequence's KV ends at its last
    /// committed position — see [`HealPrefill`].
    pub(in super::super) fn commit_decoded_tokens(
        &mut self,
        seq_ids: &[SequenceId],
        next_tokens: &mut [u32],
        logits_vec: &[Tensor],
    ) -> Vec<HealPrefill> {
        let mut heal_prefills = Vec::new();
        if decode_diag::armed() {
            self.diag_commit(seq_ids, next_tokens, logits_vec);
        }
        // A replayed turn commits its recording, whatever was sampled — here,
        // ahead of every stage below, so each one sees the token the turn takes.
        if let Some(&eos) = self.eos_tokens.first() {
            for (i, seq_id) in seq_ids.iter().enumerate() {
                let replay = self.active_decodes.get(seq_id).and_then(|s| {
                    let at = s.generated_tokens.len();
                    s.recorded_reply
                        .as_ref()
                        .filter(|replay| replay.forces(at))
                        .map(|replay| (replay.ids(), at))
                });
                if let Some((reply, at)) = replay {
                    next_tokens[i] = replayed_step(reply, at, eos);
                }
            }
        }
        // Advance each sequence's tool-call stencil with the token just sampled:
        // feed it into an active walk, or start a walk if it is a trigger token
        // (e.g. `<tool_call>`).  An empty trigger registry never starts a walk.
        // Only an *active* walk needs the token's decoded bytes (free-text
        // terminators read them); starting a walk needs only the token id, so the
        // tokenizer decode stays off the hot path when no tool call is running.
        // `(row, bytes to commit)` for rows whose sampled token is committed as
        // other bytes — a span that closed strictly inside it, or a repair of a
        // token that could not be committed as written — healed after this loop.
        let mut heals: Vec<(usize, Vec<u8>)> = Vec::new();
        // Rows whose suppressed `</think>` close was dropped (a deep/exhaustive
        // think-steer retry): the sampled close is not committed, so it never
        // lands in the output and `inside_think_block` stays set;
        // `play_stencil_runs` plays the continuation static before the next
        // wave (the session already advanced to it on `accept`).
        let mut dropped = vec![false; seq_ids.len()];
        for (i, &seq_id) in seq_ids.iter().enumerate() {
            let token = next_tokens[i];
            let active = self
                .active_decodes
                .get(&seq_id)
                .is_some_and(|s| s.stencil.is_some());
            let bytes = if active {
                self.tokenizer
                    .decode(&[token], false)
                    .map(String::into_bytes)
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            if let Some(state) = self.active_decodes.get_mut(&seq_id) {
                match state.stencil.as_mut() {
                    Some(driver) => {
                        match driver.accept(token, &bytes) {
                            Healed::Rewrite { bytes } => heals.push((i, bytes)),
                            // A suppressed close: drop the closing token (do not
                            // commit it).  This is the model's own `</think>` OR an
                            // intercepted EOS (a token-closed span now closes on
                            // either) — both land here as `Healed::Drop` and are thus
                            // skipped by the commit loop below, including the EOS-seal,
                            // so neither is written to the sequence; the steering's
                            // injected closing tag / continuation prefills in its place.
                            // A delimiter the grammar does not continue with after a
                            // value lands here too, and the grammar writes the right
                            // structure in its place. On a replayed turn the refusal is the steering's
                            // own close, which the recording holds — once. See
                            // `Replay::refuse`.
                            Healed::Drop => {
                                dropped[i] = true;
                                let at = state.generated_tokens.len();
                                if state
                                    .recorded_reply
                                    .as_mut()
                                    .is_some_and(|replay| replay.refuse(at))
                                {
                                    // `DecodeState::fail`, spelled out: the
                                    // stencil driver still borrows `state`.
                                    let _ = state.event_tx.send(TurnEvent::Error(
                                        ConversationError::Channel(format!(
                                            "the replay left its recording at reply id {at}: \
                                             the stencil refused the recorded token twice"
                                        )),
                                    ));
                                    state.faulted = true;
                                    state.finished = true;
                                }
                            }
                            Healed::No => {}
                        }
                        if driver.is_done() {
                            Self::log_stencil_finish(seq_id.0, driver, "completed");
                            state.stencil = None;
                        }
                        // Clear the pending mask so `play_stencil_runs` re-drives
                        // the driver into the continuation static this drop advanced to.
                        if dropped[i] {
                            state.pending_mask = None;
                        }
                    }
                    None => {
                        // An end token with a tail bound does not end the turn:
                        // it is dropped, and the tail writes in its place and
                        // ends the turn itself — once, and only after an
                        // answer: an end straight after the reasoning ends the
                        // turn as it stands. A replayed turn plays what its
                        // recording holds, the tail's text included.
                        let ends =
                            self.eos_tokens.contains(&token) && state.recorded_reply.is_none();
                        let tail = match ends {
                            true => state.triggers.tail_driver().filter(|_| {
                                let after =
                                    after_reasoning(&state.generated_tokens, state.think_close_at);
                                answered(after, |run| {
                                    self.tokenizer.decode(run, false).unwrap_or_default()
                                })
                            }),
                            false => None,
                        };
                        if let Some(driver) = tail {
                            state.triggers = Arc::new(state.triggers.without_tail());
                            tracing::debug!(
                                target: "candle_conversation::stencil",
                                seq_id = seq_id.0,
                                tree = driver.tree().label(),
                                "the answer's end handed over to its tail",
                            );
                            dropped[i] = true;
                            state.pending_mask = None;
                            state.stencil = Some(driver);
                            continue;
                        }
                        // Held as a turn that has reasoned for as long as this
                        // one's block ran: what the first `</think>` closed.
                        let reasoned = state.think_close_at.unwrap_or(0);
                        if let Some(mut driver) = state.triggers.driver_after(token, reasoned) {
                            // A trigger token (e.g. `<tool_call>`) opened a grammar:
                            // steer the rest of this call to the catalog's shape.
                            // A once-trigger (the think block) is spent by firing,
                            // so the rest of the turn decodes that token as text.
                            if let Some(rest) = state.triggers.after_firing(token) {
                                state.triggers = Arc::new(rest);
                            }
                            tracing::debug!(
                                target: "candle_conversation::stencil",
                                seq_id = seq_id.0,
                                tree = driver.tree().label(),
                                trigger = token,
                                "stencil steering started (trigger token decoded)",
                            );
                            // The call writes the tool its reasoning named.
                            if driver.tree().label() == TOOL_CALL_TREE_LABEL {
                                if let Some(name) = steer_to_named_tool(
                                    &self.tokenizer,
                                    &state.generated_tokens,
                                    &mut driver,
                                ) {
                                    tracing::debug!(
                                        target: "candle_conversation::stencil",
                                        seq_id = seq_id.0,
                                        tool = %name,
                                        "tool name steered to the tool the reasoning named",
                                    );
                                }
                            }
                            state.stencil = Some(driver);
                        }
                    }
                }
            }
        }

        // Restore `in_segment` and restart the per-span thinking budget for any
        // row whose close was suppressed.  When the model sampled `</think>`,
        // `sample_batch` already flipped `in_segment` off — but the steering
        // dropped that close and kept the block open, so the flag is stale.  Re-arm
        // it so the EOT boost/force ramp and the in-thinking temperature/DRY gates
        // stay live across the steered spans.  (The injected `</think>` clears it
        // for real in `play_stencil_runs`.)
        //
        // The structural think-close ban (`think_close_ban_active`) leans on this
        // restore too: it arms whenever `in_segment` is false, so without the
        // re-arm a steered block's first dropped close would ban every later
        // close attempt and pin the continuation spans to their forced caps.
        //
        // A suppressed close also ends one span and opens the next, so the EOT
        // ramp's clock (`segment_len`) restarts here.  This is the single
        // chokepoint every suppressed close passes through — a model `</think>`,
        // an intercepted EOS (both are `TokenClosedDrop`), or a force-injected
        // close.  `</think>` already zeroed `segment_len` via `exit_segment`, but
        // an EOS close did not (EOS isn't the eot token), so reset unconditionally:
        // each span gets its own budget regardless of how it closed.
        //
        // Only a dropped close of the segment re-arms it — see `segment_rearm`.
        for (i, &was_dropped) in dropped.iter().enumerate() {
            if was_dropped {
                let close = self
                    .active_decodes
                    .get(&seq_ids[i])
                    .map(|s| s.sampling_config.segment_close_token_id);
                if let Some(ss) = self.sampling_states.get_mut(&seq_ids[i]) {
                    if !drop_rearms_segment(next_tokens[i], close.unwrap_or(-1), ss.in_segment) {
                        continue;
                    }
                    if ss.segment_len > 0 {
                        tracing::debug!(
                            target: "candle_conversation::eot",
                            seq = seq_ids[i].0,
                            span_thinking_len = ss.segment_len,
                            "steered span closed — restarting per-span thinking budget",
                        );
                    }
                    ss.in_segment = true;
                    ss.segment_len = 0;
                }
            }
        }

        // Sync each sequence's penalty scope to its stencil as it stands after
        // this token. This is the single chokepoint every stencil open/close
        // passes through, regardless of which path changed it (trigger start,
        // completed, terminal, dropped, error).
        for &seq_id in seq_ids.iter() {
            // `in_tool_call` is the tool call specifically (by tree label) and
            // gates the repetition penalties — reasoning keeps full repetition
            // control, only tool-call arguments are freed to reproduce the
            // prompt's numbers/paths verbatim.
            let stencil = self
                .active_decodes
                .get(&seq_id)
                .and_then(|s| s.stencil.as_ref());
            let label: Option<&str> = stencil.map(|d| d.tree().label());
            // DRY's scope: a free-text value the grammar is writing. Each value
            // is its own window — `set_free_text` starts it at the value's first
            // token and empties it when the value closes — and the think block,
            // the grammar's structure and unconstrained prose are outside it.
            let mid_value = stencil.is_some_and(|d| d.mid_free_span());
            // **Only when the schema asked for it.** Lifting the penalties suits
            // a caller whose tool arguments are quotations — paths, identifiers,
            // numbers that are only correct if they repeat. It is the opposite
            // of what a caller wants when the argument *is* the prose: a cast
            // whose every utterance is a `say` decoded its dialogue with
            // presence, frequency, repeat and DRY all off, and repeated itself
            // word for word. Off unless named.
            let freed = self
                .active_decodes
                .get(&seq_id)
                .is_some_and(|s| s.free_tool_calls_from_penalties);
            let writing_call = label == Some(super::super::TOOL_CALL_TREE_LABEL);
            let in_tool_call = freed && writing_call;
            if let Some(ss) = self.sampling_states.get_mut(&seq_id) {
                ss.set_free_text(mid_value && !ss.in_segment);
                ss.in_tool_call = in_tool_call;
                // Unconditional, unlike the penalty lift: the length budget is
                // a prose answer's and never a call's, whoever the caller is.
                ss.writing_call = writing_call;
            }
        }

        // Heal rewritten tokens: commit the re-tokenized bytes the stencil gave
        // in place of the sampled token. Either the model closed a free-text
        // value with a token that also carries the next node's delimiter (e.g.
        // `",`) — the bytes are the valid prefix, and the successor re-emits the
        // delimiter — or the token could not be committed as written, and the
        // bytes are its repair (an escaped character, a completed value, the
        // text replacing an EOS inside a value). Common case (a single token) is
        // a plain swap; a multi-token rewrite records all-but-last as forwarded,
        // hands them back to be forwarded once the step's KV is final, and lets
        // the last ride the next decode.
        for (i, bytes) in heals {
            let seq_id = seq_ids[i];
            let text = String::from_utf8_lossy(&bytes);
            let healed: Vec<u32> = self
                .tokenizer
                .encode(text.as_ref(), false)
                .map(|e| e.get_ids().to_vec())
                .unwrap_or_default();
            let Some((&last, prefix)) = healed.split_last() else {
                continue; // nothing valid to commit (degenerate) — leave as-is
            };
            if !prefix.is_empty() {
                let think_close = self.think_close;
                let breaks = &self.page_break_tokens;
                if let Some(state) = self.active_decodes.get_mut(&seq_id) {
                    for &t in prefix {
                        state.push_forwarded(t, think_close, breaks);
                        let _ = state.event_tx.send(TurnEvent::Token(t));
                    }
                }
                heal_prefills.push(HealPrefill {
                    seq: seq_id,
                    tokens: prefix.to_vec(),
                });
            }
            next_tokens[i] = last;
        }

        // Process each sampled token
        for (i, &seq_id) in seq_ids.iter().enumerate() {
            let next_token = next_tokens[i];

            if let Some(state) = self.active_decodes.get_mut(&seq_id) {
                // ── Decode health checks ──────────────────────────────────────────────
                // Gated by a single runtime bool (false by default). When disabled,
                // this is one never-taken branch — near-zero overhead.  A suppressed
                // `</think>` (`dropped[i]`) is never emitted, so it carries no logits
                // signal to judge — skip health for it (and so don't advance the step
                // counter on a non-token).
                if self.health_config.enabled && !dropped[i] {
                    let step = state.health.step;
                    state.health.step += 1;

                    // GPU logit check: NaN / Inf / magnitude / entropy (interval-gated).
                    // Also fires one step before every page boundary (step % 32 == 31) so
                    // we have a probe point on both sides of each chunk boundary, which
                    // makes page-boundary corruption visible even if it doesn't persist.
                    let check_interval = self.health_config.logit_check_interval;
                    let on_page_pre_probe = step > 0 && step % 32 == 31;
                    // Also check every step once dense_mode is active (triggered when
                    // any prior check saw entropy below the trend threshold). This gives
                    // full-resolution data in the health log for the final approach.
                    if (check_interval > 0 && step % check_interval == 0)
                        || on_page_pre_probe
                        || state.health.dense_mode
                    {
                        match crate::decode_health::check_logits(
                            &logits_vec[i],
                            self.health_config.logit_magnitude_threshold,
                            step,
                        ) {
                            Ok(Some(evt)) => {
                                tracing::warn!(
                                    target: "candle_conversation::decode_health",
                                    seq_id = seq_id.0, %evt,
                                    "decode health abort: aborting sequence"
                                );
                                let _ = state
                                    .event_tx
                                    .send(TurnEvent::HealthWarning(evt.to_string()));
                                state.finished = true;
                                continue;
                            }
                            Err(e) => tracing::debug!(
                                target: "candle_conversation::decode_health",
                                "logit health check error (non-fatal): {e}"
                            ),
                            Ok(None) => {}
                        }

                        // Entropy check: H = −Σ p log p.
                        // Fires on hard-floor collapse (consecutive steps below threshold)
                        // or sustained trend collapse (rolling window all below soft threshold).
                        // `is_interval` gates whether this check advances the trend window;
                        // dense-mode extra steps do not advance it.
                        // Skipped entirely when temperature≤0.01: peaked distributions are
                        // expected by design and would cause immediate false-positive aborts.
                        let is_interval_check =
                            (check_interval > 0 && step % check_interval == 0) || on_page_pre_probe;
                        if !state.health.skip_entropy_checks {
                            match crate::decode_health::check_entropy(
                                &logits_vec[i],
                                &mut state.health,
                                self.health_config.entropy_hard_threshold_nats,
                                self.health_config.entropy_hard_min_consec,
                                self.health_config.entropy_trend_window,
                                self.health_config.entropy_trend_threshold_nats,
                                self.health_config.entropy_interval_floor_threshold_nats,
                                self.health_config.entropy_interval_floor_consec,
                                self.health_config.interval_argmax_dominance_window,
                                self.health_config.interval_argmax_dominance_fraction,
                                self.health_config.entropy_trend_recent_veto_window,
                                self.health_config.entropy_trend_recent_veto_factor,
                                &self.health_config.structural_token_ids,
                                step,
                                is_interval_check,
                            ) {
                                Ok(Some(evt)) => {
                                    tracing::warn!(
                                        target: "candle_conversation::decode_health",
                                        seq_id = seq_id.0, %evt,
                                        "decode health abort: aborting sequence"
                                    );
                                    let top_tokens = match &evt {
                                        crate::decode_health::HealthEvent::EntropyCollapse {
                                            ref top_tokens,
                                            ..
                                        } => top_tokens.as_slice(),
                                        crate::decode_health::HealthEvent::ArgmaxDominance {
                                            ref top_tokens,
                                            ..
                                        } => top_tokens.as_slice(),
                                        _ => &[],
                                    };
                                    let dump = crate::decode_health::render_health_dump(
                                        &state.health.health_log,
                                        step,
                                        self.health_config.entropy_hard_threshold_nats,
                                        state.health.entropy_effective_trend_threshold,
                                        self.health_config.logit_check_interval,
                                        state.prefill_token_count,
                                        state.sampling_config.temperature,
                                        state.sampling_config.top_k,
                                        state.sampling_config.top_p,
                                        state.sampling_config.repeat_penalty,
                                        &state.health.recent_tokens,
                                        top_tokens,
                                    );
                                    tracing::warn!(
                                        target: "candle_conversation::decode_health",
                                        "{}",
                                        dump
                                    );
                                    let _ = state
                                        .event_tx
                                        .send(TurnEvent::HealthWarning(evt.to_string()));
                                    state.finished = true;
                                    continue;
                                }
                                Err(e) => tracing::debug!(
                                    target: "candle_conversation::decode_health",
                                    "entropy health check error (non-fatal): {e}"
                                ),
                                Ok(None) => {}
                            }
                        } // if !skip_entropy_checks
                    }

                    // Update think-block tracking state based on the token just sampled.
                    // This takes effect on the *next* step's logit checks, which is correct:
                    // the logits that produced <think> are evaluated before entering the block,
                    // and the logits that produced </think> are evaluated before exiting it.
                    //
                    // A *suppressed* </think> (a deep/exhaustive steer retry, `dropped[i]`)
                    // is not a real close — the steered block continues — so it explicitly
                    // does NOT exit the block.  Stating that here keeps the invariant local
                    // rather than riding on the dropped token's commit being skipped below.
                    if state.sampling_config.segment_open_token_id >= 0 {
                        if next_token == state.sampling_config.segment_open_token_id as u32 {
                            state.health.inside_think_block = true;
                        } else if next_token == state.sampling_config.segment_close_token_id as u32
                            && !dropped[i]
                        {
                            state.health.inside_think_block = false;
                        }
                    }

                    // A suppressed </think> is dropped: it produces no output token, no
                    // stream event, and no repetition signal.  The think-block state above
                    // is the only per-token update a retry needs; everything below is for
                    // genuinely emitted tokens.
                    if dropped[i] {
                        continue;
                    }

                    // CPU repetition check: push token then test the window.
                    state
                        .health
                        .push_token(next_token, self.health_config.repetition_window);
                    if let Some(evt) = crate::decode_health::check_repetition(
                        &state.health,
                        self.health_config.repetition_threshold,
                    ) {
                        tracing::warn!(
                            target: "candle_conversation::decode_health",
                            seq_id = seq_id.0, %evt,
                            "decode health abort: aborting sequence"
                        );
                        let dump = crate::decode_health::render_health_dump(
                            &state.health.health_log,
                            step,
                            self.health_config.entropy_hard_threshold_nats,
                            self.health_config.entropy_trend_threshold_nats,
                            self.health_config.logit_check_interval,
                            state.prefill_token_count,
                            state.sampling_config.temperature,
                            state.sampling_config.top_k,
                            state.sampling_config.top_p,
                            state.sampling_config.repeat_penalty,
                            &state.health.recent_tokens,
                            &[],
                        );
                        tracing::warn!(
                            target: "candle_conversation::decode_health",
                            "{}",
                            dump
                        );
                        let _ = state
                            .event_tx
                            .send(TurnEvent::HealthWarning(evt.to_string()));
                        state.finished = true;
                        continue;
                    }

                    // CPU phrase-loop check: detect multi-token cyclic repetition (local minima).
                    let max_period = self.health_config.phrase_loop_max_period;
                    if max_period >= 2 {
                        if let Some(evt) = crate::decode_health::check_phrase_loop(
                            &state.health,
                            max_period,
                            self.health_config.phrase_loop_min_reps,
                            self.health_config.phrase_loop_min_total_tokens,
                        ) {
                            tracing::warn!(
                                target: "candle_conversation::decode_health",
                                seq_id = seq_id.0, %evt,
                                "decode health abort: aborting sequence"
                            );
                            let dump = crate::decode_health::render_health_dump(
                                &state.health.health_log,
                                step,
                                self.health_config.entropy_hard_threshold_nats,
                                self.health_config.entropy_trend_threshold_nats,
                                self.health_config.logit_check_interval,
                                state.prefill_token_count,
                                state.sampling_config.temperature,
                                state.sampling_config.top_k,
                                state.sampling_config.top_p,
                                state.sampling_config.repeat_penalty,
                                &state.health.recent_tokens,
                                &[],
                            );
                            tracing::warn!(
                                target: "candle_conversation::decode_health",
                                "{}",
                                dump
                            );
                            let _ = state
                                .event_tx
                                .send(TurnEvent::HealthWarning(evt.to_string()));
                            state.finished = true;
                            continue;
                        }
                    }

                    // ── Final-step entropy check ──────────────────────────────────────
                    // Runs unconditionally when this token is the last one (EOS or
                    // max_tokens), but only when the interval check did NOT already run
                    // this step (to avoid a double-check on aligned steps).
                    //
                    // Does NOT abort — generation is ending legitimately. Emits a
                    // tracing warning only so late-stage collapses are observable in
                    // traces without surfacing as a spurious abort to the caller.
                    let is_last_token = self.eos_tokens.contains(&next_token)
                        || state.generated_tokens.len() + 1 >= state.max_tokens;
                    let on_interval = (self.health_config.logit_check_interval > 0
                        && step % self.health_config.logit_check_interval == 0)
                        || (step > 0 && step % 32 == 31);
                    if is_last_token && !on_interval && !state.health.skip_entropy_checks {
                        match crate::decode_health::check_entropy(
                            &logits_vec[i],
                            &mut state.health,
                            self.health_config.entropy_hard_threshold_nats,
                            self.health_config.entropy_hard_min_consec,
                            self.health_config.entropy_trend_window,
                            self.health_config.entropy_trend_threshold_nats,
                            self.health_config.entropy_interval_floor_threshold_nats,
                            self.health_config.entropy_interval_floor_consec,
                            self.health_config.interval_argmax_dominance_window,
                            self.health_config.interval_argmax_dominance_fraction,
                            self.health_config.entropy_trend_recent_veto_window,
                            self.health_config.entropy_trend_recent_veto_factor,
                            &self.health_config.structural_token_ids,
                            step,
                            false, // final-step: observation only, do not advance trend window
                        ) {
                            Ok(Some(evt)) => {
                                let top_tokens = match &evt {
                                    crate::decode_health::HealthEvent::EntropyCollapse {
                                        ref top_tokens,
                                        ..
                                    } => top_tokens.as_slice(),
                                    crate::decode_health::HealthEvent::ArgmaxDominance {
                                        ref top_tokens,
                                        ..
                                    } => top_tokens.as_slice(),
                                    _ => &[],
                                };
                                let dump = crate::decode_health::render_health_dump(
                                    &state.health.health_log,
                                    step,
                                    self.health_config.entropy_hard_threshold_nats,
                                    state.health.entropy_effective_trend_threshold,
                                    self.health_config.logit_check_interval,
                                    state.prefill_token_count,
                                    state.sampling_config.temperature,
                                    state.sampling_config.top_k,
                                    state.sampling_config.top_p,
                                    state.sampling_config.repeat_penalty,
                                    &state.health.recent_tokens,
                                    top_tokens,
                                );
                                tracing::warn!(
                                    target: "candle_conversation::decode_health",
                                    seq_id = seq_id.0,
                                    "final-step entropy warning (generation complete): {evt}"
                                );
                                tracing::warn!(
                                    target: "candle_conversation::decode_health",
                                    "{}",
                                    dump
                                );
                            }
                            Err(e) => tracing::debug!(
                                target: "candle_conversation::decode_health",
                                "final-step entropy check error (non-fatal): {e}"
                            ),
                            Ok(None) => {}
                        }
                    }
                }
                // ── End health checks ─────────────────────────────────────────────────

                // A dropped token (a suppressed `</think>` OR an intercepted EOS of
                // a token-closed think span) is never committed: it carries no KV,
                // emits no stream event, and must not seal the turn.  The steering's
                // injected closing tag / continuation prefills in its place.  The
                // health block above already skips it, but that block is gated on
                // `health_config.enabled`; this guard makes the skip unconditional so
                // the EOS-seal below cannot fire on an intercepted EOS.
                if dropped[i] {
                    continue;
                }

                // Commit, and close the index page when this token opens a new
                // region of the turn. Closing HERE rather than at the forward is
                // the same instant as far as the cache is concerned — nothing
                // else appends for this sequence between a commit and the step
                // that forwards the committed token — and it is the one place
                // that knows the token's role.
                state.push_committed(next_token, self.think_close, &self.page_break_tokens);

                // Check termination.
                let is_eos = self.eos_tokens.contains(&next_token);

                // Per-token trace.  Off by default; enable with
                // `RUST_LOG=candle_conversation::scheduler::sampling=trace`.
                // Logs the token id, decoded text fragment, the step
                // index, and whether the model just fired EOS — the
                // canonical view of "what did the model say at each
                // decode step?" for diagnosing early-EOS, garbled
                // output, or stuck-on-one-token failures.
                if tracing::enabled!(
                    target: "candle_conversation::scheduler::sampling",
                    tracing::Level::TRACE,
                ) {
                    let decoded = self
                        .tokenizer
                        .decode(&[next_token], false)
                        .unwrap_or_else(|_| "<?>".to_string());
                    tracing::trace!(
                        target: "candle_conversation::scheduler::sampling",
                        seq_id = seq_id.0,
                        step = state.generated_tokens.len() - 1,
                        token_id = next_token,
                        is_eos,
                        decoded = %decoded,
                        "sampled token",
                    );
                    if is_eos {
                        tracing::debug!(
                            target: "candle_conversation::scheduler::sampling",
                            seq_id = seq_id.0,
                            step = state.generated_tokens.len() - 1,
                            token_id = next_token,
                            "EOS fired — generation terminating",
                        );
                    }
                }

                // Emit the raw token ID — the turn's last one included, unless it
                // is the EOS that ends it. If the caller dropped the handle, stop
                // generating.
                if streams_committed(is_eos)
                    && state.event_tx.send(TurnEvent::Token(next_token)).is_err()
                {
                    state.finished = true;
                }
                if let Some(finish) = FinishReason::after_token(
                    is_eos,
                    state.generated_tokens.len(),
                    state.max_tokens,
                ) {
                    state.finish = finish;
                    state.finished = true;
                }

                // ── Continuous re-projection triggers ─────────────────────────
                // The boundary rule (`provenance::boundaries`) — the one the
                // gallery cuts ingested turns by: a line feed past the content
                // gate and the cadence since the last boundary, none inside a
                // tool call. Skipped when the sequence just finished —
                // finalize fires via `cleanup_finished` instead.
                if !state.finished {
                    let count = state.generated_tokens.len();
                    let step = match state.reprojection.as_ref() {
                        Some(p) => state.boundary.step(&p.boundary, next_token),
                        None => Step::Quiet,
                    };
                    if step == Step::OpenedToolCall {
                        state.wrote_tool_call = true;
                    }
                    // A played-back turn reprojects where its recording did; a
                    // live one where the rule fires, as the think gate lets it.
                    let fires = match state.reprojection.as_ref() {
                        Some(p) => match p.projection.script() {
                            Some(script) => script.fires_at(count),
                            None => {
                                let thinking = Thinking::after(
                                    next_token,
                                    self.sampling_states
                                        .get(&seq_id)
                                        .is_some_and(|ss| ss.in_segment),
                                    state.sampling_config.segment_close_token_id,
                                );
                                let fires = reprojects(step.fires(), thinking, p.decode);
                                if fires && !step.fires() {
                                    state.boundary.restart();
                                }
                                fires
                            }
                        },
                        None => false,
                    };
                    if fires {
                        Self::queue_reprojection(&mut self.pending_reprojections, seq_id);
                    }
                    // Every step, the range's sections' hit rates: one placed
                    // that stayed under the bar leaves, one resident over the
                    // entry bar enters, and the next wave lays the range out
                    // again (`docs/placed_section_layout.md` §3.4).
                    if let Some(p) = state.reprojection.as_ref() {
                        let slot = self.turn_views.get(&seq_id).map_or(seq_id, |v| v.parent_id);
                        let changed = step_sections(
                            self.model.as_mut(),
                            self.slot_projection_state.get_mut(&slot),
                            slot,
                            p,
                        );
                        if changed {
                            Self::queue_reprojection(&mut self.pending_relayouts, seq_id);
                        }
                    }
                }
            }
        }
        heal_prefills
    }

    /// Forward a healed token's prefix into its sequence.
    ///
    /// `run_prefill` cuts the span at any break token it CONTAINS — but
    /// `reasoning_split` deliberately declines to split when the break token is
    /// the pass's LAST, on the understanding that the caller commits it and cuts
    /// at its own commit. That holds for a stencil's static run, whose final
    /// token rides the next decode step; here every token is already recorded
    /// as forwarded, so a prefix ending on a marker closes its page here, or the
    /// turn's reasoning would not occupy whole pages.
    ///
    /// The prefix is already in the turn's output, so a forward that fails
    /// leaves output with no KV behind it, and the sequence is failed rather
    /// than left to decode on from a cache that does not hold what it said.
    pub(in super::super) fn forward_heal_prefill(&mut self, heal: HealPrefill) {
        let HealPrefill { seq, tokens } = heal;
        if let Err(e) = self.run_prefill(seq, &tokens) {
            self.fail_all_decodes(
                &[seq],
                &format!("forwarding a healed token's prefix failed: {e}"),
            );
            return;
        }
        if !tokens
            .last()
            .is_some_and(|t| self.page_break_tokens.contains(t))
        {
            return;
        }
        match self.model.close_positional_page(seq.0) {
            Ok(closed) => tracing::info!(
                target: "candle_conversation::scheduler::unit_boundary",
                seq_id = seq.0,
                site = "heal-trailing-break-token",
                closed,
                "index: closed a page after a healed prefix ending on a break token"
            ),
            Err(e) => tracing::warn!(
                seq_id = seq.0,
                "closing the index page after a healed break token failed ({e}); the \
                 region it bounds will not occupy whole pages and cannot be windowed \
                 out of a later projection"
            ),
        }
    }
}
