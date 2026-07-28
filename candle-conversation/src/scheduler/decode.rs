use super::*;

impl Scheduler {
    /// Emit the steering finish trace: a one-line summary of the path the
    /// stencil walk took (so a malformed tool call is diagnosable — e.g.
    /// `bailed=true`, or `free_tokens=0` where an argument value was expected).
    /// `reason` distinguishes a clean completion from a failsafe drop.
    fn log_stencil_finish(seq_id: usize, driver: &StencilDriver, reason: &str) {
        let s = driver.stats();
        tracing::debug!(
            target: "candle_conversation::stencil",
            seq_id,
            tree = driver.tree().label(),
            reason,
            prefills = s.prefills,
            prefill_tokens = s.prefill_tokens,
            branch_tokens = s.branch_tokens,
            free_tokens = s.free_tokens,
            heals = s.heals,
            bailed = s.bailed,
            "stencil steering finished",
        );
    }

    // ── Stencil static-run prefill (Layer 3) ───────────────────────────

    /// Inject any pending `Static` runs for active stencil sequences via the
    /// prefill path, then record the next decode action in `pending_mask`.  Runs
    /// once before `batch_decode_step` each decode iteration.
    ///
    /// A static run `R` following the sequence's pending token `Y` (the last
    /// generated, not-yet-forwarded token) is injected as `[Y] ++ R[..last]` in
    /// one prefill forward; the whole run is appended to `generated_tokens`, and
    /// `R.last()` rides the normal decode forward in `batch_decode_step` — that
    /// forward is exactly what produces the after-run logits for the next
    /// `Branch`/`Free` token.  So an N-token run costs one prefill instead of N
    /// decode steps, with no double-write of any token's KV.
    pub(super) fn inject_stencil_prefills(&mut self) {
        let ids: Vec<SequenceId> = self
            .active_decodes
            .iter()
            .filter(|(_, s)| !s.finished && s.stencil.is_some() && s.pending_mask.is_none())
            .map(|(&id, _)| id)
            .collect();
        let max_recent = self.sampler.max_recent_len();

        for id in ids {
            let mut guard = 0usize;
            loop {
                guard += 1;
                if guard > 100_000 {
                    tracing::error!(seq_id = id.0, "stencil prefill runaway — bailing");
                    if let Some(s) = self.active_decodes.get_mut(&id) {
                        if let Some(d) = &s.stencil {
                            Self::log_stencil_finish(id.0, d, "runaway");
                        }
                        s.stencil = None;
                    }
                    break;
                }

                // Advance the driver one step (brief mutable borrow).
                let action = match self
                    .active_decodes
                    .get_mut(&id)
                    .and_then(|s| s.stencil.as_mut())
                {
                    Some(driver) => driver.step(),
                    None => break,
                };

                let StepMask::Prefill(run) = action else {
                    // Branch / Free / Done — the action for the upcoming decode.
                    if let Some(s) = self.active_decodes.get_mut(&id) {
                        s.pending_mask = Some(action);
                    }
                    break;
                };

                // Prefill `[Y] ++ run[..last]`; `run.last()` rides the decode.
                let Some(y) = self
                    .active_decodes
                    .get(&id)
                    .and_then(|s| s.generated_tokens.last().copied())
                else {
                    tracing::warn!(
                        seq_id = id.0,
                        "stencil run with no pending token — dropping"
                    );
                    if let Some(s) = self.active_decodes.get_mut(&id) {
                        if let Some(d) = &s.stencil {
                            Self::log_stencil_finish(id.0, d, "no pending token");
                        }
                        s.stencil = None;
                    }
                    break;
                };
                // A close run ends with the assistant EOS (`}}\n</tool_call>` +
                // `<|im_end|>`): a tool call is the whole assistant turn, so the
                // EOS terminates it.  Detect it here so the turn is sealed instead
                // of the model free-decoding a hallucinated answer past the call.
                let ends_turn = run.last().is_some_and(|&t| self.eos_tokens.contains(&t));

                let mut input = Vec::with_capacity(run.len());
                input.push(y);
                // `run.last()` is never forwarded here: for a normal run it rides
                // the decode in `batch_decode_step`; for an EOS-terminated run it
                // is the turn terminator, whose KV is never written (exactly as a
                // model-sampled EOS).  Either way it is excluded from the prefill.
                input.extend_from_slice(&run[..run.len() - 1]);

                if let Err(e) = self.run_prefill(id, &input) {
                    tracing::warn!(
                        seq_id = id.0,
                        "stencil prefill forward failed: {e} — dropping"
                    );
                    if let Some(s) = self.active_decodes.get_mut(&id) {
                        if let Some(d) = &s.stencil {
                            Self::log_stencil_finish(id.0, d, "prefill failed");
                        }
                        s.stencil = None;
                    }
                    break;
                }

                // Append the run to the emitted output and stream it; `run.last()`
                // becomes `generated_tokens.last()`, which the decode forwards.
                // The trailing EOS of a close run is pushed to the buffer (it is
                // part of the sealed turn) but never streamed — matching the
                // normal decode path, which buffers EOS but does not emit it.
                let mut carries_eot = false;
                if let Some(s) = self.active_decodes.get_mut(&id) {
                    let last = run.len() - 1;
                    for (k, &t) in run.iter().enumerate() {
                        s.generated_tokens.push(t);
                        if !(ends_turn && k == last) {
                            let _ = s.event_tx.send(TurnEvent::Token(t));
                        }
                    }
                    // A think-steer tree always drops the model's own `</think>`
                    // (the close token is suppressed) and injects the closing tag
                    // as a static run instead, so the commit-loop flip that clears
                    // `inside_think_block` on a sampled `</think>` never fires for a
                    // think turn.  This injected closing tag is where the block
                    // actually ends, so clear the flag here when the prefilled run
                    // carries it — otherwise health stays relaxed forever after the
                    // first think block.
                    let eot = s.sampling_config.segment_close_token_id;
                    carries_eot = eot >= 0 && run.contains(&(eot as u32));
                    if carries_eot {
                        s.health.inside_think_block = false;
                    }
                }
                // Record the run in the repeat-penalty window so the model's
                // subsequent free decode sees the tool-call tokens as recent
                // context.  `record_context_tokens` touches only `recent_tokens`
                // — not `token_counts` or `current_len` — so it cannot skew
                // frequency/presence penalties or the EOS-length failsafe.
                if let Some(ss) = self.sampling_states.get_mut(&id) {
                    ss.record_context_tokens(&run, max_recent);
                    // The injected `</think>` is where a steered block actually ends;
                    // clear the sampler's `in_segment` here too (its commit-loop twin
                    // never sees the dropped close), so the EOT boost and the
                    // in-thinking temperature/DRY gates switch off for the answer.
                    if carries_eot {
                        ss.in_segment = false;
                    }
                }

                if ends_turn {
                    // The stencil emitted the tool call's closing EOS — the
                    // assistant turn is complete.  Mark it finished and stop
                    // steering; `cleanup_finished` finalizes and seals exactly as
                    // for a model-sampled EOS.
                    if let Some(s) = self.active_decodes.get_mut(&id) {
                        if let Some(d) = &s.stencil {
                            Self::log_stencil_finish(id.0, d, "completed");
                        }
                        s.stencil = None;
                        s.finished = true;
                    }
                    break;
                }
                // Loop: the next action is the node after this static run.
            }
        }
    }

    // ── Decode ─────────────────────────────────────────────────────────

    /// Run one decode step for all active (non-finished) sequences.
    ///
    /// Each sequence contributes exactly 1 token (its last generated token).
    /// One batched `forward_batched` call processes all sequences in parallel.
    pub(super) fn batch_decode_step(&mut self) {
        // A slot with a pending deferred glue fire reprojects THIS wave (the
        // co-batched glue member rewrites its `[sealed | glue]` prefix), so its
        // logical offset and block table are mid-rewrite. It must not also run
        // a decode step this wave: the wave would then list the same slot in
        // both the decode and glue groups, and the per-unique-id cache borrow
        // (`caches_for_sequences_mut`) collapses the assembled context list,
        // desyncing every later member's varlen metadata from its slot headers.
        // Skip it here; its decode resumes next wave against the reprojected
        // backing (the glue fire drains in `take_wave_glue` during this step).
        let glue_pending: std::collections::HashSet<usize> = self
            .deferred_glue_fires
            .iter()
            .map(|p| p.parent_id.0)
            .collect();
        let seq_ids: Vec<SequenceId> = self
            .active_decodes
            .iter()
            .filter(|(_, s)| !s.finished)
            .map(|(&id, _)| id)
            .filter(|id| !glue_pending.contains(&id.0))
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

        // Attended-KV length each decode step sweeps, summed over the batch
        // (captured before the forward advances the offsets). On the wave line
        // this is the per-step context the decode attends over.
        let kv_len: usize = seq_ids_raw
            .iter()
            .map(|&sid| self.session.sequence_offset(sid).unwrap_or(0))
            .sum();

        // Forward pass: all active sequences, 1 token each — co-batching the
        // in-flight prefill cohort into decode's sweep at its active layer window
        // (docs/continuous_fair_waves.md) so one expert load per layer serves both.
        let t_fwd = std::time::Instant::now();
        let logits_vec = match self.decode_forward_cobatched(&seq_ids_raw, &inputs) {
            Ok(l) => l,
            Err(e) => {
                self.fail_all_decodes(&seq_ids, &format!("decode forward failed: {e}"));
                return;
            }
        };
        let fwd_ms = t_fwd.elapsed().as_millis() as u64;
        super::record_phase(t_fwd, "decode_forward");
        // Decode batch = N sequences × 1 token each.
        self.wave_stats
            .record(false, seq_ids.len(), seq_ids.len(), kv_len, fwd_ms);

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

        // Clone sampling configs before taking mutable references
        let mut configs: Vec<SamplingConfig> = seq_ids
            .iter()
            .map(|id| self.active_decodes[id].sampling_config.clone())
            .collect();

        // Fold each active tool-call stencil's constraint for this token (set by
        // `inject_stencil_prefills`, which already injected any preceding static
        // runs) into this row's `stencil` allow-list: a branch is its frontier (a
        // tiny gather + sample), a free-text span clears the stencil (free decode
        // through the kernel), and `Done` ends the walk (free decode + drop).
        //
        // The sampler resolves these per row in `sample_batch`, so a mask set
        // here constrains only this sequence — other rows in the wave are
        // unaffected, and constrained rows never run the full-vocab kernel.
        for (i, &id) in seq_ids.iter().enumerate() {
            if let Some(state) = self.active_decodes.get_mut(&id) {
                if state.stencil.is_none() {
                    continue;
                }
                match state.pending_mask.take() {
                    Some(StepMask::Branch(set)) => {
                        configs[i].stencil = set.tokens().iter().map(|&t| t as i32).collect();
                    }
                    Some(StepMask::Free { .. }) => {
                        // A free-text span decodes normally — nothing is banned.  A
                        // think-steer span's `</think>` and EOS are both intercepted
                        // by the session's `observe` (the suppressed close drops the
                        // token and prefills a continuation; the final span injects
                        // the closing tag), and tool-call value spans close on a byte
                        // delimiter — so the sampler just runs free here.
                        configs[i].stencil.clear();
                    }
                    Some(StepMask::Done) => {
                        if let Some(d) = &state.stencil {
                            Self::log_stencil_finish(id.0, d, "completed");
                        }
                        state.stencil = None;
                        configs[i].stencil.clear();
                    }
                    // A `Prefill` is consumed by `inject_stencil_prefills`; `None`
                    // means the driver wasn't advanced — free-decode this step.
                    Some(StepMask::Prefill(_)) | None => configs[i].stencil.clear(),
                }
            }
        }

        let config_refs: Vec<&SamplingConfig> = configs.iter().collect();

        // Temporarily remove persistent sampling states to avoid borrow
        // conflict with self.sample_batch_from_logits().
        let mut removed_states: Vec<(SequenceId, SequenceSamplingState)> = seq_ids
            .iter()
            .map(|&id| {
                let mut state = self
                    .sampling_states
                    .remove(&id)
                    .expect("sampling state must exist for active sequence");
                // Sync the steering span's close semantics into the sampler for
                // THIS step (only consulted inside a segment): the hard-cap
                // closer script may play only in a TERMINAL free-text span (or
                // an unsteered block, where there is no stencil). Everywhere
                // else — a continuation span whose close is dropped and
                // re-steered into "But wait, " reasoning, a tool-call value,
                // a static prefill — a forced close stays bare.
                if state.in_segment {
                    state.close_would_continue = self
                        .active_decodes
                        .get(&id)
                        .and_then(|s| s.stencil.as_ref())
                        .is_some_and(|d| !d.in_terminal_close_span());
                }
                (id, state)
            })
            .collect();

        let mut sampling_states: Vec<&mut SequenceSamplingState> =
            removed_states.iter_mut().map(|(_, state)| state).collect();

        // Sample next token for all sequences in a single batched call
        let t_sample = std::time::Instant::now();
        let mut next_tokens =
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

        // Pre-decode the sampled tokens into a readable string for the timing
        // trace.  Only built when TRACE is active (gated by `tracing::enabled!`)
        // so the tokenizer call doesn't fire on the hot path under default
        // logging.  Multi-sequence batches join the per-sequence fragments
        // with `|`.
        let want_token_str = tracing::enabled!(target: "candle_conversation::scheduler::timing", tracing::Level::TRACE);
        let token_str: String = if want_token_str {
            let skip = !self.show_special_tokens;
            // Slot-labelled (`seq:token`): each fragment attributes to its slot
            // so a cross-slot row swap (one stream continuing another's text)
            // is directly visible in the trace instead of an anonymous mixture.
            next_tokens
                .iter()
                .zip(seq_ids.iter())
                .map(|(&t, id)| {
                    let frag = self
                        .tokenizer
                        .decode(&[t], skip)
                        .unwrap_or_else(|_| "<?>".to_string());
                    format!("{}:{}", id.0, frag)
                })
                .collect::<Vec<_>>()
                .join("|")
        } else {
            String::new()
        };

        tracing::trace!(
            target: "candle_conversation::scheduler::timing",
            batch = seq_ids.len(),
            fwd_ms,
            sample_ms,
            token_str = %token_str,
            "decode_step",
        );

        // Advance each sequence's tool-call stencil with the token just sampled:
        // feed it into an active walk, or start a walk if it is a trigger token
        // (e.g. `<tool_call>`).  An empty trigger registry never starts a walk.
        // Only an *active* walk needs the token's decoded bytes (free-text
        // terminators read them); starting a walk needs only the token id, so the
        // tokenizer decode stays off the hot path when no tool call is running.
        // `(row, consumed, token bytes)` for rows whose free-text span closed
        // strictly inside the sampled token — healed after this loop.
        let mut heals: Vec<(usize, usize, Vec<u8>)> = Vec::new();
        // Rows whose suppressed `</think>` close was dropped (a deep/exhaustive
        // think-steer retry): the sampled close is not committed, so it never
        // lands in the output and `inside_think_block` stays set;
        // `inject_stencil_prefills` drains the continuation static before the next
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
                            Healed::Exit { consumed } => heals.push((i, consumed, bytes)),
                            // A suppressed close: drop the closing token (do not
                            // commit it).  This is the model's own `</think>` OR an
                            // intercepted EOS (a token-closed span now closes on
                            // either) — both land here as `Healed::Drop` and are thus
                            // skipped by the commit loop below, including the EOS-seal,
                            // so neither is written to the sequence; the steering's
                            // injected closing tag / continuation prefills in its place.
                            Healed::Drop => dropped[i] = true,
                            Healed::No => {}
                        }
                        if driver.is_done() {
                            Self::log_stencil_finish(seq_id.0, driver, "completed");
                            state.stencil = None;
                        }
                        // Clear the pending mask so `inject_stencil_prefills` re-drives
                        // the driver into the continuation static this drop advanced to.
                        if dropped[i] {
                            state.pending_mask = None;
                        }
                    }
                    None => {
                        if let Some(driver) = state.triggers.driver_for(token) {
                            // A trigger token (e.g. `<tool_call>`) opened a grammar:
                            // steer the rest of this call to the catalog's shape.
                            tracing::debug!(
                                target: "candle_conversation::stencil",
                                seq_id = seq_id.0,
                                tree = driver.tree().label(),
                                trigger = token,
                                "stencil steering started (trigger token decoded)",
                            );
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
        // for real in `inject_stencil_prefills`.)
        //
        // A suppressed close also ends one span and opens the next, so the EOT
        // ramp's clock (`segment_len`) restarts here.  This is the single
        // chokepoint every suppressed close passes through — a model `</think>`,
        // an intercepted EOS (both are `TokenClosedDrop`), or a force-injected
        // close.  `</think>` already zeroed `segment_len` via `exit_segment`, but
        // an EOS close did not (EOS isn't the eot token), so reset unconditionally:
        // each span gets its own budget regardless of how it closed.
        for (i, &was_dropped) in dropped.iter().enumerate() {
            if was_dropped {
                if let Some(ss) = self.sampling_states.get_mut(&seq_ids[i]) {
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

        // Sync per-sequence DRY suppression to tool-call (stencil) state.  This
        // is the single chokepoint every stencil open/close passes through,
        // regardless of which path changed it (trigger start, completed,
        // terminal, dropped, error).  On each edge the DRY span resets, so prose
        // before and after a tool call never shares a DRY window with the call —
        // and while the call runs DRY is off entirely (the grammar is steered).
        for &seq_id in seq_ids.iter() {
            // `in_stencil` covers ANY steered span (think or tool call) and gates
            // DRY, as before.  `in_tool_call` is the tool call specifically (by
            // tree label) and gates the remaining repetition penalties — reasoning
            // keeps full repetition control, only tool-call arguments are freed to
            // reproduce the prompt's numbers/paths verbatim.
            let label: Option<&str> = self
                .active_decodes
                .get(&seq_id)
                .and_then(|s| s.stencil.as_ref())
                .map(|d| d.tree().label());
            let in_stencil = label.is_some();
            let in_tool_call = label == Some(super::TOOL_CALL_TREE_LABEL);
            if let Some(ss) = self.sampling_states.get_mut(&seq_id) {
                if in_stencil && !ss.dry_suppressed {
                    ss.enter_tool_call();
                } else if !in_stencil && ss.dry_suppressed {
                    ss.exit_tool_call();
                }
                ss.in_tool_call = in_tool_call;
            }
        }

        // Heal merged exit tokens: the model closed a free-text value with a
        // token that also carries the next node's delimiter (e.g. `",`).  Commit
        // only the re-tokenized valid prefix (the value + closing char); the
        // delimiter is dropped and re-emitted by the successor node.  Common case
        // (the valid prefix is a single token) is a plain swap; a multi-token
        // prefix forwards all-but-last and lets the last ride this step's decode.
        for (i, consumed, bytes) in heals {
            let seq_id = seq_ids[i];
            let text = String::from_utf8_lossy(&bytes[..consumed]);
            let healed: Vec<u32> = self
                .tokenizer
                .encode(text.as_ref(), false)
                .map(|e| e.get_ids().to_vec())
                .unwrap_or_default();
            let Some((&last, prefix)) = healed.split_last() else {
                continue; // nothing valid to commit (degenerate) — leave as-is
            };
            if !prefix.is_empty() && self.run_prefill(seq_id, prefix).is_ok() {
                if let Some(state) = self.active_decodes.get_mut(&seq_id) {
                    for &t in prefix {
                        state.generated_tokens.push(t);
                        let _ = state.event_tx.send(TurnEvent::Token(t));
                    }
                }
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
                //       etc.) AND more than 16 non-trigger tokens have been
                //       decoded since the last projection.  Re-orients attention
                //       at semantic transitions, but the content gate stops short
                //       lines and runs of trigger tokens from each re-projecting.
                // De-duped so a token that satisfies BOTH only queues once.
                // Skipped when the sequence just finished — finalize fires
                // via `cleanup_finished` instead.
                if !state.finished {
                    // Compute the flags via a match so the immutable borrow of
                    // `state.reprojection` ends before we mutate the counter below.
                    let flags = match state.reprojection.as_ref() {
                        Some(p) => Some((
                            !p.trigger_token_ids.is_empty()
                                && p.trigger_token_ids.contains(&next_token),
                            p.every_n_tokens > 0
                                && state.generated_tokens.len() % p.every_n_tokens == 0,
                            p.tool_call_open_id == Some(next_token),
                            p.tool_call_close_id == Some(next_token),
                        )),
                        None => None,
                    };
                    if let Some((is_trigger, cadence_fire, is_tool_open, is_tool_close)) = flags {
                        if is_tool_open {
                            // Entering the tool call: fire one lock-in reprojection so
                            // the tool committed from the reasoning so far is what the
                            // model sees, then freeze — the generic call body must not
                            // re-orient the selection.
                            Self::queue_reprojection(&mut self.pending_reprojections, seq_id);
                            state.non_punct_since_reproject = 0;
                            state.in_tool_call = true;
                        } else if is_tool_close {
                            // Leaving the call: reprojection re-enables for whatever
                            // follows (further calls, or the seal).
                            state.in_tool_call = false;
                            state.non_punct_since_reproject = 0;
                        } else if state.in_tool_call {
                            // Inside the call body — every cadence/punctuation trigger
                            // is suppressed so the committed tool stays fixed.
                        } else {
                            let punctuation_fire =
                                is_trigger && state.non_punct_since_reproject > 16;
                            if cadence_fire || punctuation_fire {
                                Self::queue_reprojection(&mut self.pending_reprojections, seq_id);
                                state.non_punct_since_reproject = 0;
                            } else if !is_trigger {
                                // A non-trigger token adds to the content accumulated
                                // toward the next punctuation-driven reprojection.
                                state.non_punct_since_reproject += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Queue a view for reprojection at the next drain, deduplicating against
    /// entries already pending. Every reprojection trigger — cadence,
    /// punctuation, tool-call lock-in, prefill promotion — enqueues through
    /// here so the queue's invariants live in one place. An associated fn over
    /// the queue field (not `&mut self`) so trigger sites that hold a
    /// `DecodeState` borrow can still call it.
    pub(super) fn queue_reprojection(pending: &mut Vec<SequenceId>, seq_id: SequenceId) {
        if !pending.contains(&seq_id) {
            pending.push(seq_id);
        }
    }

    /// Drain the queue of views whose decoded count crossed an
    /// `every_n_tokens` boundary in the last `batch_decode_step`.
    ///
    /// Each entry triggers a provenance scan + projection + view swap; on
    /// success the view's `SequenceId` changes (the new id is internal
    /// to the scheduler — never visible to the caller).  Failures mark
    /// the view as finished with an error event.
    pub(super) fn drain_pending_reprojections(&mut self) {
        if self.pending_reprojections.is_empty() {
            return;
        }
        let _t_drain = super::PhaseTimer::new("drain_reprojections");
        let pending = std::mem::take(&mut self.pending_reprojections);

        // Phase 1 — prepare each view: provenance scan + projection + tier elevate +
        // inject the sealed prefix + build the gap-fill descriptor. Removes the
        // view's `DecodeState` into the in-flight; does NOT fire the forward.
        let mut inflights: Vec<super::ReprojectInFlight> = Vec::new();
        for view_id in pending {
            // The view may have finished (EOS/health) between trigger and drain.
            if self.active_decodes.get(&view_id).is_none_or(|s| s.finished) {
                continue;
            }
            match self.reproject_view_prepare(view_id) {
                Ok(Some(inflight)) => inflights.push(inflight),
                Ok(None) => {}
                Err(e) => {
                    tracing::warn!(
                        target: "candle_conversation::scheduler::reproject",
                        view_id = view_id.0, err = %e,
                        "reprojection prepare failed; aborting decode for this view"
                    );
                    if let Some(state) = self.active_decodes.get_mut(&view_id) {
                        let _ = state.event_tx.send(TurnEvent::Error(e));
                        state.finished = true;
                    }
                }
            }
        }
        if inflights.is_empty() {
            return;
        }

        // Phase 2 — the cross-conversation wave: ONE batched multi-slot gap-fill
        // forward computes every prepared conversation's boundary glue at once,
        // amortising the per-forward fixed cost (MoE expert loads, layer loop,
        // launch) across all of them.
        let plan_refs: Vec<&super::projection_assembler::GapFillPlan> =
            inflights.iter().map(|i| &i.plan).collect();
        let glue_total: usize = inflights.iter().map(|i| i.plan.n_glue_tokens).sum();
        tracing::debug!(
            target: "candle_conversation::scheduler::reproject",
            n_slots = inflights.len(),
            glue_total,
            "gap-fill wave: batched multi-slot forward",
        );
        let t_fire = std::time::Instant::now();
        if let Err(e) = super::projection_assembler::fire_gap_fill_batch(
            &mut self.session,
            &*self.model,
            &self.device,
            &plan_refs,
        ) {
            tracing::warn!(
                target: "candle_conversation::scheduler::reproject",
                err = %e, n = inflights.len(),
                "batched gap-fill forward failed; aborting decode for all prepared views"
            );
            for inflight in inflights {
                let _ = inflight.decode_state.event_tx.send(TurnEvent::Error(
                    ConversationError::Channel(format!("gap-fill wave forward failed: {e}")),
                ));
            }
            return;
        }
        // The batched gap-fill forward time — for the wave it's shared across all
        // slots, so each view's reproject log reports the same `glue_ms`.
        let glue_ms = t_fire.elapsed().as_millis() as u64;
        super::REPROJ_GLUE_US.fetch_add(
            t_fire.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // Phase 3 — complete each view: finish (deferred user + restore tail) +
        // carve the new view + re-key. Independent per view.
        for inflight in inflights {
            let view_id = inflight.view_id;
            if let Err(e) = self.reproject_view_complete(inflight, glue_ms) {
                tracing::warn!(
                    target: "candle_conversation::scheduler::reproject",
                    view_id = view_id.0, err = %e,
                    "reprojection complete failed; decode for this view ends"
                );
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
