use super::*;
use crate::token_buffer::TokenBuffer;

impl Scheduler {
    /// Promote up to `MAX_ACTIVE_PREFILLS - active_prefills.len()` newly
    /// submitted PrefillWorks from the FIFO queue into the in-flight
    /// `active_prefills` set. Emits the initial `Prefill` and
    /// `PrefillProgress(0, total)` events so callers see their submission
    /// was picked up.
    pub(super) fn promote_new_prefills(&mut self) {
        const MAX_ACTIVE_PREFILLS: usize = 16;
        while self.active_prefills.len() < MAX_ACTIVE_PREFILLS {
            // VRAM-pressure backpressure (wave budgeting). Each admitted prefill
            // pins its conversation's KV in VRAM, so under pressure we stop
            // piling on more concurrent prefills: first force a compaction to
            // reclaim arenas, and if VRAM is still tight, leave the rest queued
            // this pass. We always keep ≥1 prefill in flight so the engine makes
            // progress — a single oversized turn is then bounded by the per-arena
            // VRAM budget gate (which compacts/fails fast rather than spilling).
            if !self.active_prefills.is_empty() && self.vram_under_pressure() {
                let _ = self.session.compact_forced();
                if self.vram_under_pressure() {
                    break;
                }
            }
            let work = match self.prefill_queue.pop_front() {
                Some(w) => w,
                None => break,
            };
            let total = work.tokens.len();
            let _ = work
                .event_tx
                .send(TurnEvent::Prefill(work.prefill_text.clone()));
            let _ = work.event_tx.send(TurnEvent::PrefillProgress {
                tokens_done: 0,
                tokens_total: total,
            });
            let error = if total == 0 {
                Some(ConversationError::Channel(
                    "prefill received zero tokens".into(),
                ))
            } else {
                None
            };
            self.active_prefills.push(ActivePrefill {
                work,
                offset: 0,
                final_logits: None,
                error,
                prefill_start: None,
            });
        }
    }

    /// True when device VRAM free space has dropped below the admission
    /// reserve (10% of total, floor 1 GiB) — the signal to stop admitting
    /// additional concurrent prefills. Sits just above the per-arena VRAM
    /// budget gate's reserve so admission throttles slightly before a hard
    /// arena OOM. `false` on non-CUDA devices / when the query is unavailable.
    pub(super) fn vram_under_pressure(&self) -> bool {
        match self.session.vram_free_total() {
            Some((free, total)) => {
                let reserve = (total / 10).max(1usize << 30);
                free < reserve
            }
            None => false,
        }
    }

    /// Number of in-flight prefills that still have tokens left to process
    /// and have not errored.
    pub(super) fn prefill_width(&self) -> usize {
        self.active_prefills
            .iter()
            .filter(|p| p.error.is_none() && p.offset < p.work.tokens.len())
            .count()
    }

    /// Number of in-flight section ingests with tokens remaining (not errored).
    pub(super) fn section_ingest_width(&self) -> usize {
        self.active_section_ingests
            .iter()
            .filter(|s| s.error.is_none() && s.offset < s.tokens.len())
            .count()
    }

    /// Run one prefill chunk across all active section ingests, batching them
    /// into a single `forward_batched` call so collection members sharing the
    /// same prefix context attend in parallel rather than serially.
    pub(super) fn run_one_section_ingest_chunk(&mut self) {
        let active: Vec<usize> = (0..self.active_section_ingests.len())
            .filter(|&i| {
                let s = &self.active_section_ingests[i];
                s.error.is_none() && s.offset < s.tokens.len()
            })
            .collect();
        if active.is_empty() {
            return;
        }

        let chunk_len = active
            .iter()
            .map(|&i| {
                let s = &self.active_section_ingests[i];
                s.tokens.len() - s.offset
            })
            .min()
            .unwrap()
            .min(self.max_prefill_chunk);

        let mut seq_ids: Vec<usize> = Vec::with_capacity(active.len());
        let mut inputs: Vec<Tensor> = Vec::with_capacity(active.len());
        let mut group_idxs: Vec<usize> = Vec::with_capacity(active.len());
        for &i in &active {
            let s = &mut self.active_section_ingests[i];
            let off = s.offset;
            let tokens = &s.tokens[off..off + chunk_len];
            match Tensor::new(tokens, &self.device).and_then(|t| t.unsqueeze(0)) {
                Ok(t) => {
                    seq_ids.push(s.sequence_id.0);
                    inputs.push(t);
                    group_idxs.push(i);
                }
                Err(e) => {
                    s.error = Some(ConversationError::Model(e));
                }
            }
        }
        if seq_ids.is_empty() {
            return;
        }

        tracing::debug!(
            target: "sched",
            "section_ingest batch={} chunk_len={} seq_ids={:?}",
            seq_ids.len(),
            chunk_len,
            seq_ids
        );

        let logits_vec = match self
            .model
            .forward_batched(&mut self.session, &seq_ids, &inputs)
        {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("batched section ingest forward failed: {e}");
                for &i in &group_idxs {
                    self.active_section_ingests[i].error =
                        Some(ConversationError::Channel(msg.clone()));
                }
                return;
            }
        };

        // Logits are produced but not used — section ingests have no decode.
        // We only need to advance the session offset and record slot tokens.
        for (_logits, &i) in logits_vec.into_iter().zip(group_idxs.iter()) {
            let s = &mut self.active_section_ingests[i];
            if let Err(e) = self.session.advance_sequence(s.sequence_id.0, chunk_len) {
                s.error = Some(ConversationError::Model(e));
                continue;
            }
            let seq_id = s.sequence_id;
            let off = s.offset;
            let chunk_tokens = s.tokens[off..off + chunk_len].to_vec();
            super::Scheduler::record_slot_tokens(&mut self.slot_tokens, seq_id, &chunk_tokens);
            s.offset += chunk_len;
        }
    }

    /// Drain completed or errored section ingest entries. Errored entries send
    /// `Err`; finished entries call `finalize_section_ingest` (seal + write)
    /// and send the `SealResult`.
    pub(super) fn finalize_done_section_ingests(&mut self) {
        let mut i = 0;
        while i < self.active_section_ingests.len() {
            let done = {
                let s = &self.active_section_ingests[i];
                s.error.is_some() || s.offset >= s.tokens.len()
            };
            if !done {
                i += 1;
                continue;
            }
            let s = self.active_section_ingests.swap_remove(i);
            if let Some(e) = s.error {
                let _ = s.response_tx.send(Err(e));
                continue;
            }
            let result = self.finalize_section_ingest(
                s.sequence_id,
                s.section_id,
                s.seal_block_from,
                std::sync::Arc::new(s.tokens.to_vec()),
                s.address,
                s.debug_name,
                s.in_collection,
            );
            let _ = s.response_tx.send(result);
            // swap_remove pulled the last element into i; don't increment.
        }
    }

    /// Run **one** prefill chunk across every still-active in-flight
    /// prefill. The chunk length equals `min(remaining across active)`
    /// capped by `max_prefill_chunk`. All inputs in the call are
    /// guaranteed to share the same dim-1 size.
    ///
    /// Sequences whose offset reaches `tokens.len()` after this chunk have
    /// their `final_logits` recorded; they will be promoted to decode by
    /// `promote_finished_prefills_to_decodes`.
    pub(super) fn run_one_prefill_chunk(&mut self) {
        // Collect indices of still-active prefills.
        let active: Vec<usize> = (0..self.active_prefills.len())
            .filter(|&i| {
                let p = &self.active_prefills[i];
                p.error.is_none() && p.final_logits.is_none() && p.offset < p.work.tokens.len()
            })
            .collect();
        if active.is_empty() {
            return;
        }

        // Ragged batch: each prefill advances by its OWN min(remaining, cap).
        // The varlen forward packs the heterogeneous lengths flat, so a short
        // scope no longer collapses the whole wave to the batch minimum.
        let cap = self.max_prefill_chunk;
        let mut seq_ids: Vec<usize> = Vec::with_capacity(active.len());
        let mut inputs: Vec<Tensor> = Vec::with_capacity(active.len());
        let mut group_idxs: Vec<usize> = Vec::with_capacity(active.len());
        let mut chunks: Vec<usize> = Vec::with_capacity(active.len());
        for &i in &active {
            let p = &mut self.active_prefills[i];
            if p.prefill_start.is_none() {
                p.prefill_start = Some(Instant::now());
            }
            let off = p.offset;
            let chunk = (p.work.tokens.len() - off).min(cap);
            let tokens = &p.work.tokens[off..off + chunk];
            match Tensor::new(tokens, &self.device).and_then(|t| t.unsqueeze(0)) {
                Ok(t) => {
                    seq_ids.push(p.work.sequence_id.0);
                    inputs.push(t);
                    group_idxs.push(i);
                    chunks.push(chunk);
                }
                Err(e) => {
                    p.error = Some(ConversationError::Model(e));
                }
            }
        }
        if seq_ids.is_empty() {
            return;
        }

        let total_tokens: usize = chunks.iter().sum();
        tracing::debug!(
            target: "sched",
            "prefill batch={} tokens={} decode_active={} seq_ids={:?}",
            seq_ids.len(),
            total_tokens,
            self.active_decodes.len(),
            seq_ids
        );

        let n_seqs = seq_ids.len();
        let t_fwd = Instant::now();
        let logits_vec = match self
            .model
            .forward_batched(&mut self.session, &seq_ids, &inputs)
        {
            Ok(v) => v,
            Err(e) => {
                let msg = format!("batched prefill forward failed: {e}");
                for &i in &group_idxs {
                    self.active_prefills[i].error = Some(ConversationError::Channel(msg.clone()));
                }
                return;
            }
        };
        // Prefill batch = n_seqs sequences, Σ chunks tokens (ragged).
        self.wave_stats
            .record(true, n_seqs, total_tokens, t_fwd.elapsed().as_millis() as u64);

        for ((logits, &i), &chunk) in logits_vec
            .into_iter()
            .zip(group_idxs.iter())
            .zip(chunks.iter())
        {
            let p = &mut self.active_prefills[i];
            if let Err(e) = self.session.advance_sequence(p.work.sequence_id.0, chunk) {
                p.error = Some(ConversationError::Model(e));
                continue;
            }
            // Mirror the just-prefilled tokens into the diagnostic
            // log so the turn-complete dump can reconstruct the
            // exact context the kernel saw — `run_one_prefill_chunk`
            // is the SubmitTurn prefill path, parallel to
            // `run_prefill_with_shift`'s synchronous path.
            let seq_id = p.work.sequence_id;
            let off = p.offset;
            let chunk_tokens = p.work.tokens[off..off + chunk].to_vec();
            super::Scheduler::record_slot_tokens(&mut self.slot_tokens, seq_id, &chunk_tokens);
            p.offset += chunk;
            let total = p.work.tokens.len();
            let _ = p.work.event_tx.send(TurnEvent::PrefillProgress {
                tokens_done: p.offset,
                tokens_total: total,
            });
            if p.offset >= total {
                p.final_logits = Some(logits);
            }
        }
    }

    /// Drain finished or errored entries from `active_prefills`. Errored
    /// entries emit `TurnEvent::Error`; finished entries are passed to
    /// `finalise_prefill` (which samples the first token and inserts into
    /// `active_decodes`).
    pub(super) fn promote_finished_prefills_to_decodes(&mut self) {
        // Use swap_remove for efficiency; iterate from the back.
        let mut i = 0;
        while i < self.active_prefills.len() {
            let done = {
                let p = &self.active_prefills[i];
                p.error.is_some() || (p.final_logits.is_some() && p.offset >= p.work.tokens.len())
            };
            if !done {
                i += 1;
                continue;
            }
            let p = self.active_prefills.swap_remove(i);
            let ActivePrefill {
                work,
                offset: _,
                final_logits,
                error,
                prefill_start,
            } = p;
            if let Some(e) = error {
                let _ = work.event_tx.send(TurnEvent::Error(e));
                continue;
            }
            let logits = match final_logits {
                Some(l) => l,
                None => {
                    let _ = work
                        .event_tx
                        .send(TurnEvent::Error(ConversationError::Channel(
                            "prefill produced no final logits".into(),
                        )));
                    continue;
                }
            };
            let prefill_ms = prefill_start
                .map(|s| s.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            let turn_start = work.submitted_at;
            let token_count = work.tokens.len();
            self.finalise_prefill(work, logits, prefill_ms, turn_start, token_count);
            // swap_remove pulled the last element into i; don't increment.
        }
    }

    /// Post-forward path shared by both single and batched prefill: sample
    /// the first token, emit it, and either transition to decode or close
    /// the turn out immediately on EOS / max_decode_tokens == 0.
    fn finalise_prefill(
        &mut self,
        work: PrefillWork,
        logits: Tensor,
        prefill_ms: f64,
        turn_start: Instant,
        token_count: usize,
    ) {
        // Total KV position after this prefill.
        let context_depth = self
            .session
            .sequence_offset(work.sequence_id.0)
            .unwrap_or(token_count);

        let mut sampling_state = self
            .sampling_states
            .remove(&work.sequence_id)
            .expect("sampling state must exist for active sequence");
        sampling_state.end_turn();
        sampling_state.record_context_tokens(&work.tokens, self.sampler.max_recent_len());

        // Send prefill progress: complete (single-prefill path needs this;
        // batched path already streams progress per-chunk, but a final
        // tokens_done==tokens_total event is always benign).
        let _ = work.event_tx.send(TurnEvent::PrefillProgress {
            tokens_done: token_count,
            tokens_total: token_count,
        });

        let first_token = match self.sample_single(&logits, &work.sampling, &mut sampling_state) {
            Ok(t) => t,
            Err(e) => {
                self.sampling_states
                    .insert(work.sequence_id, sampling_state);
                let _ = work.event_tx.send(TurnEvent::Error(e));
                return;
            }
        };

        // Detect think-mode entry from injected <think> in the prefill.
        let initial_inside_think_block = {
            let tid = work.sampling.think_start_token_id;
            if tid >= 0 {
                let tok = tid as u32;
                let prefill_has_think = work.tokens.iter().rev().take(5).any(|&t| t == tok);
                if prefill_has_think && !sampling_state.in_thinking {
                    sampling_state.enter_thinking();
                }
                prefill_has_think || first_token == tok
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
            // View sequences (SubmitTurn path): the prefill already wrote KV
            // blocks that must be finalized onto the parent and sealed into
            // the substrate.  Insert as a finished DecodeState so
            // cleanup_finished runs finalize_view + perform_seal_and_write.
            //
            // Non-view sequences (raw RULER / summarisation): no parent to
            // finalize and seal=None is correct — use the fast path.
            if self.turn_views.contains_key(&work.sequence_id) {
                self.active_decodes.insert(
                    work.sequence_id,
                    DecodeState {
                        event_tx: work.event_tx,
                        generated_tokens: TokenBuffer::from(vec![first_token]),
                        max_tokens: work.max_decode_tokens,
                        sampling_config: work.sampling,
                        seal_action: work.seal_action,
                        post_decode_tokens: work.post_decode_tokens,
                        prefill_tokens: work.tokens,
                        user_text: work.user_text,
                        finished: true,
                        decode_start: Instant::now(),
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
                        prov_sig_entries: Vec::new(),
                    },
                );
            } else {
                self.finish_immediately(
                    work.sequence_id,
                    first_token,
                    &work.event_tx,
                    prefill_ms,
                    turn_start,
                    context_depth,
                );
            }
            return;
        }

        let _ = work.event_tx.send(TurnEvent::Token(first_token));

        self.active_decodes.insert(
            work.sequence_id,
            DecodeState {
                event_tx: work.event_tx,
                generated_tokens: TokenBuffer::from(vec![first_token]),
                max_tokens: work.max_decode_tokens,
                sampling_config: work.sampling,
                seal_action: work.seal_action,
                post_decode_tokens: work.post_decode_tokens,
                prefill_tokens: work.tokens,
                user_text: work.user_text,
                finished: false,
                decode_start: Instant::now(),
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
                prov_sig_entries: Vec::new(),
            },
        );
    }

    #[allow(dead_code)]
    pub(super) fn run_prefill_with_shift(
        &mut self,
        sequence_id: SequenceId,
        tokens: &[u32],
        write_offset_shift: usize,
    ) -> Result<Tensor, ConversationError> {
        // Chunked prefill: split large prompts into bounded chunks to keep
        // intermediate activation buffers from growing unboundedly.
        // Boundary-injection shifts are always small partial blocks and are
        // handled as a single pass.
        if write_offset_shift == 0 && tokens.len() > self.max_prefill_chunk {
            let mut last_logits: Option<Tensor> = None;
            for chunk in tokens.chunks(self.max_prefill_chunk) {
                let input = Tensor::new(chunk, &self.device)
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(ConversationError::Model)?;
                let logits_vec = self
                    .model
                    .forward_batched(&mut self.session, &[sequence_id.0], &[input])
                    .map_err(ConversationError::Model)?;
                self.session
                    .advance_sequence(sequence_id.0, chunk.len())
                    .map_err(ConversationError::Model)?;
                super::Scheduler::record_slot_tokens(&mut self.slot_tokens, sequence_id, chunk);
                last_logits = logits_vec.into_iter().next();
            }
            return last_logits.ok_or_else(|| {
                ConversationError::Channel("no logits returned from chunked prefill".into())
            });
        }

        let input = Tensor::new(tokens, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(ConversationError::Model)?;

        let logits_vec = if write_offset_shift == 0 {
            self.model
                .forward_batched(&mut self.session, &[sequence_id.0], &[input])
                .map_err(ConversationError::Model)?
        } else {
            self.model
                .forward_batched_with_write_shifts(
                    &mut self.session,
                    &[sequence_id.0],
                    &[input],
                    &[write_offset_shift as u32],
                )
                .map_err(ConversationError::Model)?
        };

        self.session
            .advance_sequence(sequence_id.0, tokens.len())
            .map_err(ConversationError::Model)?;

        // Mirror these tokens into the slot's diagnostic log so the
        // turn-complete dump can reconstruct the exact context the
        // kernel saw (compiled out without the `context-dump` feature).
        super::Scheduler::record_slot_tokens(&mut self.slot_tokens, sequence_id, tokens);

        logits_vec
            .into_iter()
            .next()
            .ok_or_else(|| ConversationError::Channel("no logits returned from prefill".into()))
    }
}
