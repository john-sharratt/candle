use super::*;

impl Scheduler {
    // ── Decode ─────────────────────────────────────────────────────────

    /// Run one decode step for all active (non-finished) sequences.
    ///
    /// Each sequence contributes exactly 1 token (its last generated token).
    /// One batched `forward_batched` call processes all sequences in parallel.
    pub(super) fn batch_decode_step(&mut self) {
        let seq_ids: Vec<SequenceId> = self
            .active_decodes
            .iter()
            .filter(|(_, s)| !s.finished)
            .map(|(&id, _)| id)
            .collect();

        if seq_ids.is_empty() {
            return;
        }

        let _t_step = super::PhaseTimer::new("decode_batch_step");

        tracing::debug!(
            target: "sched",
            "decode batch={} prefill_active={}",
            seq_ids.len(),
            self.active_prefills.len()
        );

        // Build input tensors: each sequence's last generated token, shape [1, 1].
        // Capture the per-slot input tokens so we can mirror them
        // into the diagnostic log alongside `advance_sequence`.
        let input_tokens: Vec<u32> = seq_ids
            .iter()
            .map(|&id| *self.active_decodes[&id].generated_tokens.last().unwrap())
            .collect();
        let inputs: Vec<Tensor> = match input_tokens
            .iter()
            .map(|&last_token| {
                Tensor::new(&[last_token], &self.device).and_then(|t| t.unsqueeze(0))
            })
            .collect::<candle::Result<Vec<_>>>()
        {
            Ok(t) => t,
            Err(e) => {
                self.fail_all_decodes(&seq_ids, &format!("failed to create decode inputs: {e}"));
                return;
            }
        };

        // Extract raw usize IDs for the forward_batched call into candle-transformers.
        let seq_ids_raw: Vec<usize> = seq_ids.iter().map(|id| id.0).collect();

        // Forward pass: all active sequences, 1 token each.
        let t_fwd = std::time::Instant::now();
        let logits_vec = match self
            .model
            .forward_batched(&mut self.session, &seq_ids_raw, &inputs)
        {
            Ok(l) => l,
            Err(e) => {
                self.fail_all_decodes(&seq_ids, &format!("decode forward failed: {e}"));
                return;
            }
        };
        let fwd_ms = t_fwd.elapsed().as_millis() as u64;
        super::record_phase(t_fwd, "decode_forward");

        // Advance offsets (1 token per sequence) and mirror the
        // input token into the slot's diagnostic log — this is the
        // moment the kernel commits `input_tokens[i]` to the slot's
        // KV cache.
        for (i, &id) in seq_ids.iter().enumerate() {
            if let Err(e) = self.session.advance_sequence(id.0, 1) {
                tracing::warn!("failed to advance sequence {}: {}", id, e);
            }
            super::Scheduler::record_slot_tokens(
                &mut self.slot_tokens,
                id,
                std::slice::from_ref(&input_tokens[i]),
            );
        }

        // Extract provenance Q-sigs after advance_sequence so the offset is
        // current.  The newly-completed block (if any) is guaranteed R16 —
        // it finished this step and the bg_quantizer cannot have touched it
        // yet.  Results accumulate in DecodeState::prov_sig_entries and are
        // passed to perform_seal_and_write, making seal-time extraction cover
        // only the final partial block (also always R16).
        self.extract_prov_after_step(&seq_ids);

        // Clone sampling configs before taking mutable references
        let configs: Vec<SamplingConfig> = seq_ids
            .iter()
            .map(|id| self.active_decodes[id].sampling_config.clone())
            .collect();
        let config_refs: Vec<&SamplingConfig> = configs.iter().collect();

        // Temporarily remove persistent sampling states to avoid borrow
        // conflict with self.sample_batch_from_logits().
        let mut removed_states: Vec<(SequenceId, SequenceSamplingState)> = seq_ids
            .iter()
            .map(|&id| {
                let state = self
                    .sampling_states
                    .remove(&id)
                    .expect("sampling state must exist for active sequence");
                (id, state)
            })
            .collect();

        let mut sampling_states: Vec<&mut SequenceSamplingState> =
            removed_states.iter_mut().map(|(_, state)| state).collect();

        // Sample next token for all sequences in a single batched call
        let t_sample = std::time::Instant::now();
        let next_tokens =
            match self.sample_batch_from_logits(&logits_vec, &mut sampling_states, &config_refs) {
                Ok(tokens) => tokens,
                Err(e) => {
                    // Reinsert states before failing
                    for (id, state) in removed_states {
                        self.sampling_states.insert(id, state);
                    }
                    let seq_ids: Vec<SequenceId> = self.active_decodes.keys().copied().collect();
                    self.fail_all_decodes(&seq_ids, &format!("sampling failed: {e}"));
                    return;
                }
            };

        // Reinsert persistent sampling states
        for (id, state) in removed_states {
            self.sampling_states.insert(id, state);
        }
        let sample_ms = t_sample.elapsed().as_millis() as u64;
        super::record_phase(t_sample, "decode_sample");

        // Pre-decode the sampled tokens into a readable string for the
        // timing trace.  Only built when the debug level is actually
        // active (gated by `tracing::enabled!`) so the tokenizer call
        // doesn't fire on the hot path under default logging.  Multi-
        // sequence batches join the per-sequence fragments with `|`.
        let token_str: String = if tracing::enabled!(
            target: "candle_conversation::scheduler::timing",
            tracing::Level::DEBUG,
        ) {
            let skip = !self.show_special_tokens;
            next_tokens
                .iter()
                .map(|&t| {
                    self.tokenizer
                        .decode(&[t], skip)
                        .unwrap_or_else(|_| "<?>".to_string())
                })
                .collect::<Vec<_>>()
                .join("|")
        } else {
            String::new()
        };

        tracing::debug!(
            target: "candle_conversation::scheduler::timing",
            batch = seq_ids.len(),
            fwd_ms,
            sample_ms,
            token_str = %token_str,
            "decode_step",
        );

        // Process each sampled token
        for (i, &seq_id) in seq_ids.iter().enumerate() {
            let next_token = next_tokens[i];

            if let Some(state) = self.active_decodes.get_mut(&seq_id) {
                // ── Decode health checks ──────────────────────────────────────────────
                // Gated by a single runtime bool (false by default). When disabled,
                // this is one never-taken branch — near-zero overhead.
                if self.health_config.enabled {
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
                    if state.sampling_config.think_start_token_id >= 0 {
                        if next_token == state.sampling_config.think_start_token_id as u32 {
                            state.health.inside_think_block = true;
                        } else if next_token == state.sampling_config.eot_token_id as u32 {
                            state.health.inside_think_block = false;
                        }
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

                state.generated_tokens.push(next_token);

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

                if is_eos || state.generated_tokens.len() >= state.max_tokens {
                    state.finished = true;
                } else {
                    // Emit the raw token ID. If the caller dropped the handle,
                    // stop generating.
                    if state.event_tx.send(TurnEvent::Token(next_token)).is_err() {
                        state.finished = true;
                    }
                }

                // ── Continuous re-projection triggers ─────────────────────────
                // Two trigger conditions, OR'd:
                //   (a) Cadence: decoded count crossed an `every_n_tokens`
                //       boundary.  Guarantees a re-project at predictable
                //       intervals even on long uninterrupted spans.
                //   (b) Punctuation: the just-sampled token is in the
                //       policy's `trigger_token_ids` set (linefeed, period,
                //       etc.).  Re-orients attention at semantic
                //       transitions — paragraph/sentence boundaries — which
                //       are usually the natural moments to reconsider what
                //       context the model needs next.
                // De-duped so a token that satisfies BOTH only queues once.
                // Skipped when the sequence just finished — finalize fires
                // via `cleanup_finished` instead.
                if !state.finished {
                    if let Some(p) = state.reprojection.as_ref() {
                        let cadence_fire = p.every_n_tokens > 0
                            && state.generated_tokens.len() % p.every_n_tokens == 0;
                        let punctuation_fire = !p.trigger_token_ids.is_empty()
                            && p.trigger_token_ids.contains(&next_token);
                        if (cadence_fire || punctuation_fire)
                            && !self.pending_reprojections.contains(&seq_id)
                        {
                            self.pending_reprojections.push(seq_id);
                        }
                    }
                }
            }
        }
    }

    /// Drain the queue of views whose decoded count crossed an
    /// `every_n_tokens` boundary in the last `batch_decode_step`.
    ///
    /// Each entry triggers a BDP scan + projection + view swap; on
    /// success the view's `SequenceId` changes (the new id is internal
    /// to the scheduler — never visible to the caller).  Failures mark
    /// the view as finished with an error event.
    pub(super) fn drain_pending_reprojections(&mut self) {
        if self.pending_reprojections.is_empty() {
            return;
        }
        let _t_drain = super::PhaseTimer::new("drain_reprojections");
        // Take the queue out so `reproject_view` can mutate `self.active_decodes`
        // without aliasing.
        let pending = std::mem::take(&mut self.pending_reprojections);
        for view_id in pending {
            // The view may have finished (EOS/health) between trigger
            // and drain — in that case the cleanup path handles finalize
            // and we have nothing to do.
            if !self
                .active_decodes
                .get(&view_id)
                .map_or(false, |s| !s.finished)
            {
                continue;
            }
            match self.reproject_view(view_id) {
                Ok(_new_id) => {} // re-key already done internally
                Err(e) => {
                    tracing::warn!(
                        target: "candle_conversation::scheduler::reproject",
                        view_id = view_id.0, err = %e,
                        "reprojection failed; aborting decode for this view"
                    );
                    if let Some(state) = self.active_decodes.get_mut(&view_id) {
                        let _ = state.event_tx.send(TurnEvent::Error(e));
                        state.finished = true;
                    }
                }
            }
        }
    }

    /// Send an error to all active decodes and mark them finished.
    fn fail_all_decodes(&mut self, seq_ids: &[SequenceId], msg: &str) {
        for &id in seq_ids {
            if let Some(state) = self.active_decodes.get_mut(&id) {
                let _ = state
                    .event_tx
                    .send(TurnEvent::Error(ConversationError::Channel(
                        msg.to_string(),
                    )));
                state.finished = true;
            }
        }
    }
}
