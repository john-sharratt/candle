//! Tears down finished and abandoned turns: it disposes of each turn, releases
//! its slot, and seals what completed.
//!
//! A finished turn is handled in three phases so that turns finishing in the
//! same step close together: every turn's disposition first, then one forward
//! of all their closing tails, then each turn's seal and its `Done`. A turn that
//! ended on a fault has its view discarded rather than sealed, so the parent
//! keeps the state it had before the turn began and a retry starts clean. A
//! recurrent state follows its K/V into the parent, because a turn's seal keeps
//! every token it decoded, reasoning included, and nothing may release state
//! that advanced over those blocks.

use super::*;

impl Scheduler {
    /// Free a compression scratch slot and its per-slot bookkeeping. Called on
    /// every path out of a compression pass (success or error). The decoded
    /// delta's RAII `ChunkGid`s keep the arena chunks alive after this returns.
    pub(super) fn free_summary_slot(&mut self, slot: SequenceId) {
        let _ = self.session.free_sequence(slot.0);
        let _ = self.model.release_sequence(slot.0);
        let freed_conversation = self.slot_conversations.remove(&slot);
        let freed_target = self.slot_targets.remove(&slot);
        self.gallery_sweep.slot_freed();
        self.sampling_states.remove(&slot);
        self.retire_slot_projection_state(slot, freed_conversation);
        self.compression_event_sinks.remove(&slot);
        // A queued glue plan must not outlive the slot layout it was planned
        // against (see the FreeSequence handler's purge).
        self.deferred_glue_fires.retain(|p| p.parent_id != slot);
        // Purge any in-flight prefill/decode/cohort state for this slot before
        // its id is recycled (see [`Self::purge_freed_slot_scheduling_state`]).
        self.purge_freed_slot_scheduling_state(slot);
        self.prune_ingest_timeline(freed_target);
    }

    /// Drop a turn view that will never finish its decode — its prefill failed,
    /// or the parent it borrows from is going away. Nothing on the view was
    /// completed, so its K/V is abandoned rather than transferred and its model
    /// state released rather than moved onto the parent.
    ///
    /// A view is otherwise released only when its decode completes or a
    /// reprojection swaps it. An errored turn did neither, so its view kept the
    /// parent's whole projected prefix borrowed for the daemon's lifetime, and
    /// a failed wave never gave that KV back.
    pub(super) fn discard_turn_view(&mut self, view_id: SequenceId) {
        let Some(view_state) = self.turn_views.remove(&view_id) else {
            return;
        };
        if let Err(e) = self.session.free_sequence(view_id.0) {
            tracing::warn!("failed to free turn view {}: {}", view_id, e);
        }
        // A view a reprojection re-carved holds the parent's state itself, so
        // it goes back to the parent; the release below then finds none.
        if view_state.holds_parent_state {
            if let Err(e) = self.model.move_recurrent(view_id.0, view_state.parent_id.0) {
                tracing::warn!(
                    "failed to hand turn view {}'s state back to parent {}: {}",
                    view_id,
                    view_state.parent_id,
                    e
                );
            }
        }
        if let Err(e) = self.model.release_sequence(view_id.0) {
            tracing::warn!(
                "failed to release model state for turn view {}: {}",
                view_id,
                e
            );
        }
        self.sampling_states.remove(&view_id);
        self.slot_tokens.remove(&view_id);
        // A view's working set is recorded under its PARENT, so the parent's
        // conversation is the substrate this republishes on.
        let parent_conversation = self.slot_conversations.get(&view_state.parent_id).cloned();
        self.retire_slot_projection_state(view_id, parent_conversation);
        self.purge_freed_slot_scheduling_state(view_id);
        // A group's case that will never finish is passed over, so the cases
        // behind it still seal and the group still answers.
        self.lose_group_case(view_id);
    }

    /// Drop a turn that ended on a fault ([`DecodeState::fail`]) without
    /// sealing it. Its caller already has the error.
    ///
    /// An ordinary turn's view is finalized onto its parent and its recurrent
    /// state moved there; a faulted one's is released, so the parent stays
    /// exactly where it stood before the turn began and a retry starts clean.
    /// A summarise pass answers its job with the fault and frees its slot.
    pub(super) fn discard_faulted_turn(&mut self, seq_id: SequenceId, seal_action: &SealAction) {
        tracing::warn!(
            seq = seq_id.0,
            "a turn ended on a fault; discarding what it decoded instead of sealing it"
        );
        if let SealAction::CompressionPass { job_id } = *seal_action {
            if let Some(job) = self.compression_jobs.remove(&job_id) {
                // Lifted to hot only for this pass, as in
                // `complete_compression_pass`.
                if !job.children.is_empty() {
                    job.conversation.read().demote_turns_to_warm(&job.children);
                }
                let _ = job.response_tx.send(Err(ProbeError::Soft(
                    "SubmitSummaryProbe: the summary decode ended on a fault".to_string(),
                )));
            }
            self.free_summary_slot(seq_id);
            return;
        }
        // A non-view decode owns its slot outright and leaves through the
        // slot's own release; a view goes here, group cases included.
        self.discard_turn_view(seq_id);
    }

    /// [`Self::discard_turn_view`] for every view borrowing from `parent_id`.
    pub(super) fn discard_turn_views_of(&mut self, parent_id: SequenceId) {
        // The parent is going away, so its groups end here — before their
        // views are discarded, or each discard would carve the next case's
        // view over the parent being torn down.
        self.abandon_turn_groups_of(parent_id);
        let views: Vec<SequenceId> = self
            .turn_views
            .iter()
            .filter(|(_, v)| v.parent_id == parent_id)
            .map(|(id, _)| *id)
            .collect();
        for view in views {
            self.discard_turn_view(view);
        }
    }

    /// Handle the case where generation finishes on the first token (EOS or
    /// max=0); `finish` says which.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn finish_immediately(
        &self,
        seq_id: SequenceId,
        token: Option<u32>,
        event_tx: &Sender<TurnEvent>,
        prefill_ms: f64,
        turn_start: Instant,
        prefill_token_count: usize,
        turn_prefill_tokens: usize,
        finish: FinishReason,
    ) {
        let skip = !self.show_special_tokens;
        // `None` for a turn with no decode budget, which has no reply.
        let ids: Vec<u32> = token.into_iter().collect();
        // Persist verbatim (see the main finish path) — no think-stripping here.
        let answer = decode_turn_text(&self.tokenizer, &self.tag_ids, "", &ids, skip);
        let total_ms = turn_start.elapsed().as_secs_f64() * 1000.0;
        let tokens_generated = ids.len();
        let _ = event_tx.send(TurnEvent::Done(TurnResponse {
            text: answer.text(),
            answer,
            token_ids: ids.into(),
            stats: TurnStats {
                prefill_ms,
                decode_ms: 0.0,
                total_ms,
                tokens_generated,
                tokens_per_second: 0.0,
                prefill_token_count,
                turn_prefill_tokens,
                context_tokens: self.session.sequence_offset(seq_id.0).unwrap_or(0),
                finish,
                sequence: self.session.get_sequence_stats(seq_id.0),
            },
            // `finish_immediately` fires before any decode starts and
            // does no seal — the substrate write would have nothing
            // to capture (zero new blocks).
            seal: None,
        }));
    }

    // —— Cleanup ————————————————————————————————————————————————————————

    /// Remove finished sequences and send `Done` events.
    pub(super) fn cleanup_finished(&mut self) {
        let finished_seq_ids: Vec<SequenceId> = self
            .active_decodes
            .iter()
            .filter(|(_, s)| s.finished)
            .map(|(&id, _)| id)
            .collect();
        // Their views are finalized onto their parents below: the outstanding
        // gap's rows for their blocks go in place first.
        if !finished_seq_ids.is_empty() {
            self.catch_up_outstanding_gap();
        }

        // Three phases, so the turns finishing together close together: every
        // turn's disposition, then one forward of all their closing tails, then
        // each turn's seal and `Done`.
        let mut sealing: Vec<FinishedTurn> = Vec::with_capacity(finished_seq_ids.len());
        for seq_id in finished_seq_ids {
            let high = self.decode_priority_or_high(seq_id).is_live();
            if let Some(state) = self.active_decodes.remove(&seq_id) {
                // The slot's ground is back, so an admission that did not fit
                // before may fit now.
                self.settled_since_admit = true;
                // What a `High` decode run takes, for how the device is split
                // beside the prompts (`turn_airtime`).
                if high && !state.faulted {
                    self.segment_mean.add(state.generated_tokens.len());
                }
                // A turn that ended on a fault has told its caller so; what it
                // decoded is dropped with its view and never sealed.
                if state.faulted {
                    self.discard_faulted_turn(seq_id, &state.seal_action);
                    continue;
                }
                // The summarise decode completes through the job registry, not
                // the substrate seal path: its body becomes the node's assistant
                // half and is sealed with the derived scope. No view to finalize,
                // no Done event — the summariser blocks on the job's `response_tx`,
                // fired by `complete_compression_pass`.
                if let SealAction::CompressionPass { job_id } = state.seal_action {
                    self.complete_compression_pass(seq_id, job_id, state.generated_tokens);
                    continue;
                }
                // A group's case is sealed from its own view by its group, in
                // the group's order — never finalized onto the parent, which
                // every other case is still borrowing from.
                if let SealAction::GroupCase(case) = state.seal_action {
                    self.finish_group_case(seq_id, case, state.prefill_ms);
                    continue;
                }

                // Throughput over time actually spent decoding, not turn
                // wall-clock — see `DecodeState::decode_busy_us`. Falls back to
                // wall-clock only if no forward was ever recorded.
                let decode_ms = if state.decode_busy_us > 0 {
                    state.decode_busy_us as f64 / 1000.0
                } else {
                    state.decode_start.elapsed().as_secs_f64() * 1000.0
                };
                let total_ms = state.turn_start.elapsed().as_secs_f64() * 1000.0;
                let tokens_generated = state.generated_tokens.len();
                let tokens_per_second = if decode_ms > 0.0 {
                    tokens_generated as f64 / (decode_ms / 1000.0)
                } else {
                    0.0
                };

                let skip = !self.show_special_tokens;
                // Persist the reply VERBATIM — including its `<think>…</think>`
                // reasoning — so the stored text matches the sealed token_ids and
                // a substrate reload renders identically to the live stream. The
                // `<think>` block is part of the response; consumers that don't
                // want it strip at the point of use (the summariser strips its own
                // output), and the KV already carries the reasoning tokens either
                // way. Stripping here would silently drop reasoning on F5/reload.
                //
                // **The written half comes first.** An assistant prefill — a
                // seeded `<tool_call>`, a turn grammar's opening scaffold — is
                // assistant content that happens to have been written instead of
                // sampled. Reporting only the sampled half hands the caller a
                // fragment starting mid-syntax: it prefilled `{"name": "` to
                // commit the shape, and then cannot parse the shape back,
                // because the part it supplied is missing. The substrate had the
                // same hole, storing a turn whose text began in the middle of
                // its own JSON.
                let answer = decode_turn_text(
                    &self.tokenizer,
                    &self.tag_ids,
                    &state.prefill_assistant_text,
                    &state.generated_tokens,
                    skip,
                );
                // Snapshot stats before any view finalize, since the view
                // slot is dropped during finalize and its sequence stats
                // become unavailable.
                let sequence_stats = self.session.get_sequence_stats(seq_id.0);
                let context_tokens = self.session.sequence_offset(seq_id.0).unwrap_or(0);

                // Auto-finalize: if this sequence is a scheduler-owned view
                // (created by SubmitTurn), transfer its newly-written blocks
                // back to the parent and drop the view slot before sending
                // Done.  This keeps the entire view lifecycle invisible to
                // the caller.
                let finalized_view = self.turn_views.remove(&seq_id);

                // KV-zero check: scan the VIEW slot right BEFORE finalize_view. If this
                // is clean but `substrate-write-seal` (post-finalize) is not, the seal
                // re-org drops blocks; if this is already dirty, the live decode arena
                // held zeros during generation.
                #[cfg(feature = "kv-zero-check")]
                {
                    let layers: Vec<usize> = (0..self.session.num_layers()).collect();
                    let n_real = self.session.sequence_offset(seq_id.0).unwrap_or(0);
                    // Boundary = where THIS turn's decode began = offset − generated,
                    // so `region="own"` = decoded by this turn, `region="prefix"` =
                    // present before decode (inherited projection or this turn's prefill).
                    let turn_start = n_real.saturating_sub(tokens_generated);
                    let layout = self.session.provenance_chunk_layout(seq_id.0, n_real);
                    if let Ok(dump) = self
                        .session
                        .gather_r16_kv_provenance_layers(seq_id.0, &layers, None)
                    {
                        kv_zero_check::scan_gathered(
                            "decode-final-view",
                            seq_id.0,
                            &dump,
                            &layout,
                            self.session.n_kv_head(),
                            self.session.head_dim(),
                            turn_start,
                        );
                    }
                }

                // The finalized view's `(view_id, parent_id)`, held until the
                // seal branch below decides whether its recurrent state moves
                // onto the parent or is dropped with the blocks it advanced
                // over. `None` on the non-view path, which has no child state.
                let mut pending_view_state: Option<(SequenceId, SequenceId)> = None;
                let (seal_slot, seal_block_from) = if let Some(view_state) = finalized_view {
                    let _g = profile::span("cleanup:finalize_view");
                    if let Err(e) = self.session.finalize_view(
                        seq_id.0,
                        view_state.parent_id.0,
                        view_state.original_borrowed.0,
                    ) {
                        tracing::warn!(
                            "auto-finalize failed for view {} parent {}: {}",
                            seq_id,
                            view_state.parent_id,
                            e,
                        );
                    }
                    // **The state's disposition is NOT decided here.**
                    //
                    // `finalize_view` frees the view's session slot but leaves
                    // the model's per-sequence state under the view id, and
                    // that is deliberate: whether this turn moves it onto the
                    // parent or drops it depends on a branch that has not run
                    // yet. A `SealAction::Turn` re-prefills the turn clean —
                    // the decoded blocks, thinking tokens and all, are
                    // abandoned — and the parent's state must stay at the turn
                    // boundary so the re-prefill advances it over
                    // `[user][clean response]` exactly once. Every other seal
                    // keeps the blocks, so the state moves with them.
                    //
                    // Deciding it eagerly here is the `<think>` skew: the state
                    // would end up having seen `[prefix][thinking][clean]` while
                    // the K/V holds `[prefix][clean]`, on every thinking turn,
                    // compounding. See `pending_view_state` below.
                    pending_view_state = Some((seq_id, view_state.parent_id));
                    self.sampling_states.remove(&seq_id);
                    // The view's KV blocks just got transferred onto
                    // the parent — mirror the same merge in the
                    // diagnostic log so `slot_tokens[parent]`
                    // continues to be a faithful 1:1 mirror of the
                    // parent's KV contents.  Gated on `context-dump`
                    // since the map is empty when the feature is off.
                    #[cfg(feature = "context-dump")]
                    if let Some(view_toks) = self.slot_tokens.remove(&seq_id) {
                        self.slot_tokens
                            .entry(view_state.parent_id)
                            .or_default()
                            .extend(view_toks);
                    }
                    (view_state.parent_id, view_state.turn_start_parent_blocks)
                } else {
                    // Non-view path (rare; not produced by SubmitTurn today
                    // but kept for safety).  Treat the slot itself as parent
                    // and seal from offset 0.
                    (seq_id, 0)
                };

                // Ephemeral probe: the parent slot now holds the full warm KV —
                // system prompt at `[0, seal_block_from)`, the query (+ the single
                // decoded token) at `[seal_block_from, end)`. Gather the query's
                // warm wide-Q here, where the blocks are guaranteed present (the
                // same point a real seal gathers), and stash it for the probe's
                // `ProbeWideSigs` to drain. The turn seals nothing (`SealAction::None`).
                // A slot asked to be described is described here, where it
                // holds the prompt and the turn and nothing has been sealed.
                if let Some(tx) = self.describe_at_seal.remove(&seal_slot) {
                    let described = self.describe_slot(seal_slot).map(|(mut layout, capture)| {
                        self.last_capture += 1;
                        layout.capture = self.last_capture;
                        self.captures.insert(self.last_capture, capture);
                        layout
                    });
                    let _ = tx.send(described);
                }
                if self.ephemeral_slots.contains(&seal_slot) {
                    let bc = self.session.sequence_block_count(seal_slot.0).unwrap_or(0);
                    let sigs = self.gather_wide_sigs(seal_slot, (seal_block_from, bc));
                    self.ephemeral_sigs.insert(seal_slot, sigs);
                }

                // Harvest the turn's final belief onto the conversation's parent
                // slot: the NEXT turn's submit-time projection seeds from it, so
                // provenance selection evolves across turn boundaries instead of
                // resetting to catalog order.
                //
                // MERGE, never replace: a turn whose projection lacked a
                // collection (tools dial off, projection skipped) harvests a
                // belief without that collection's key, and overwriting would
                // erase the conversation's accumulated lock-on for it. Gated on
                // the slot still being registered — a mid-decode FreeSequence
                // (client disconnect) must not resurrect a dead conversation's
                // belief under a slot id the allocator is about to recycle.
                if finalized_view.is_some() && self.slot_conversations.contains_key(&seal_slot) {
                    self.carried_beliefs
                        .entry(seal_slot)
                        .or_default()
                        .merge_from(&state.belief);
                }

                tracing::info!(
                    target: "sched",
                    tokens = tokens_generated, tps = tokens_per_second as u32,
                    prefill_ms = state.prefill_ms as u64, decode_ms = decode_ms as u64,
                    "turn complete",
                );

                // The post-decode tail (the turn's closing structural tokens —
                // ChatML's `\n` after `<|im_end|>`) is prefilled AFTER the
                // view-state disposition below, not here.
                //
                // It has to be, because `run_prefill` advances the recurrent
                // state of the slot it runs on, and this slot is the PARENT.
                // Run here it lands on the wrong side of both dispositions: on
                // the DISCARD path the tail is truncated away and re-prefilled
                // by the clean pass anyway, so this advance is a phantom the
                // sealed state carries twice (`… ⊕ tail ⊕ user ⊕ clean ⊕ tail`)
                // over a K/V that holds it once; on the MOVE path the view's
                // state — which never saw the tail — is written over the
                // parent immediately after, losing it. Both are the skew the
                // DISCARD comment below describes for `<think>`, one call
                // earlier, and both compound one marker per turn.

                // Diagnostic: dump the *entire* token stream the
                // kernel saw for this turn — every injected system
                // section, every projected turn, the user prefill,
                // every decoded token, and the post-decode tail.
                // Decoded to text so a human can eyeball it for
                // formatting bugs (missing role markers, malformed
                // ChatML, etc.) at the moment generation completes.
                //
                // Compiled in only when the `context-dump` cargo
                // feature is enabled (default: off).  With the
                // feature off the per-token recording is a no-op
                // and this entire emit is removed by the compiler
                // — no tokenizer.decode() in the hot path of a
                // production build.  When enabled, also requires
                // `RUST_LOG=candle_conversation::scheduler::context_dump=info`
                // to actually emit.
                #[cfg(feature = "context-dump")]
                if tracing::enabled!(
                    target: "candle_conversation::scheduler::context_dump",
                    tracing::Level::INFO,
                ) {
                    if let Some(toks) = self.slot_tokens.get(&seal_slot) {
                        let token_count = toks.len();
                        let decoded = self
                            .tokenizer
                            .decode(toks, false)
                            .unwrap_or_else(|e| format!("<decode error: {e}>"));
                        tracing::info!(
                            target: "candle_conversation::scheduler::context_dump",
                            seq_id = seal_slot.0,
                            token_count,
                            "=== KV cache token dump (turn complete) ===\n{decoded}\n=== end dump ==="
                        );
                    }
                }

                // **The view's decoded blocks stay on the parent, so the state
                // it advanced over them moves with them.**
                //
                // There is one disposition now, for every turn. A turn seals
                // with everything it decoded — reasoning included — and the
                // projection decides what a later turn may attend
                // (`docs/thinking_span_projection.md`). Nothing truncates the
                // decoded blocks away any more, so nothing may release the state
                // that advanced over them.
                if let Some((view_id, parent_id)) = pending_view_state.take() {
                    let _g = profile::span("cleanup:move_recurrent");
                    if let Err(e) = self.model.move_recurrent(view_id.0, parent_id.0) {
                        tracing::warn!(
                            "failed to move recurrent state from view {} to parent {}: {}",
                            view_id,
                            parent_id,
                            e,
                        );
                    }
                }

                sealing.push(FinishedTurn {
                    state,
                    answer,
                    decode_ms,
                    total_ms,
                    tokens_generated,
                    tokens_per_second,
                    sequence_stats,
                    context_tokens,
                    seal_slot,
                    seal_block_from,
                });
            }
        }

        // Post-decode forward pass: append each turn's closing structural
        // tokens into its slot before sealing, so the turn's pinned K/V closes
        // its own brackets. The model didn't emit these — we synthesise them as
        // if it did.
        //
        // Deliberately AFTER every turn's disposition above: the state that
        // absorbs the tail must be the one being sealed, and on the MOVE path
        // that state only arrives on the parent there.
        //
        // **One wave for every turn finishing here, not one per turn.** Each
        // tail is a slot's own tokens on its own slot, so forwarding them
        // together gives each slot exactly what forwarding it alone would — the
        // same pieces, the same page closes, in the same order — and turns that
        // finish in the same step share the forward instead of queueing behind
        // each other's on the loop thread.
        let tails: Vec<(SequenceId, &[u32])> = sealing
            .iter()
            .filter(|t| !t.state.post_decode_tokens.is_empty())
            .map(|t| (t.seal_slot, &t.state.post_decode_tokens[..]))
            .collect();
        if !tails.is_empty() {
            let _g = profile::span("cleanup:post_decode_prefill");
            for (slot, e) in self.run_prefill_batch(&tails) {
                tracing::warn!("post-decode prefill failed for slot {}: {}", slot, e);
            }
        }

        for turn in sealing {
            self.seal_finished_turn(turn);
        }
        self.drain_parked_forks();
    }
}
