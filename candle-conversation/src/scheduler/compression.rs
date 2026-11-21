//! The model-decode path of the summary tree: a decode-only pass whose output
//! becomes a compressed node's assistant half.
//!
//! The pass is assembled from the children's original turns, injected as sealed
//! K/V, followed by a short instruction, so the summary is conditioned on its
//! children and on nothing else. The framing sits at the tail as a new user
//! message rather than a generated run, because a recurrent model can compute an
//! append at its tail but refuses to fill a gap in the middle of its sequence.
//! The scratch slot is neutral, not seeded from the timeline, so each summary
//! depends only on the children it compresses and no state is installed for it.
//! The children are lifted to hot memory before injection and demoted again once
//! the summary has decoded.

use super::*;

impl Scheduler {
    /// Set up the summarise decode — which produces the node's assistant half —
    /// returning its scratch slot.
    ///
    /// Assembles `[compressor system][full child turns, natural roles][user_open]
    /// [instruction][user_end]` — the children's ORIGINAL user + assistant content
    /// injected as sealed K/V (`SealedKind::Turn`, zero re-prefill, natural
    /// user→assistant framing), then a follow-up user turn asking the model to
    /// summarise them. `setup_compression_decode` prefills `assistant_start` and
    /// registers the decode. The decode advances in `batch_decode_step` and
    /// completes in `cleanup_finished` → `complete_compression_pass`, which takes
    /// the decoded body as the node's assistant half and seals it with the derived
    /// scope. The decode's own K/V is not kept — the seal re-prefills the cleaned,
    /// marker-framed exchange into role-coherent K/V.
    pub(super) fn enqueue_compression_pass(
        &mut self,
        conv: &Conversation,
        target: ProjectionTarget,
        children: &[TurnIndex],
        prompt: &CompressionPrompt,
        max_tokens: usize,
        job_id: u64,
    ) -> Result<SequenceId, String> {
        // Lazily seal the summary system-prompt framing as a content section the
        // first time it is used, then inject it (zero-copy Arc clone, zero
        // re-prefill) at the head of the segment list.
        let summary_section = self
            .ensure_summary_section(conv, prompt)
            .map_err(|e| format!("SubmitSummaryProbe: seal summary section: {e}"))?;

        // Segment list: compressor system section, then the children's full turns
        // in natural roles (sealed), then the instruction user-turn.
        let mut segments: Vec<ProjectionSegment> = vec![ProjectionSegment::Sealed(
            SealedKind::Section(ResolvedSection::new(summary_section)),
        )];
        for &child in children {
            segments.push(ProjectionSegment::Sealed(SealedKind::Turn(
                ResolvedTurn {
                    id: TurnId {
                        layer_id: target.layer,
                        group_id: target.group,
                        index: child,
                    },
                    timeline: Some(target.timeline),
                },
                // Role is display-only for injection (`inject_sealed_turn` ignores
                // it and places the whole turn's sealed K/V); the turn's own baked
                // boundary markers carry the real user→assistant roles.
                Role::Assistant,
            )));
        }
        // The instruction turn: `user_start` + `/no_think` + the summarise
        // instruction + `user_end`, as ONE tail run. `/no_think` keeps the
        // compressor from reasoning (the summary budget is tiny); no
        // prior-summary anchor — the pass sees only its children and the
        // prompt, so there is nothing for it to reproduce verbatim (a leak we
        // hit when anchoring on prior summaries).
        //
        // A `NewUserMessage`, NOT `Generated` runs: the framing sits at the
        // TAIL, after every sealed piece, and the deferred-user path prefills
        // it there as a plain append — which a recurrence can compute, its
        // state having accumulated through the whole prefix. As `Generated`
        // runs the assembler reserved them as a gap-fill island, which a
        // recurrent-lineage model refuses (`reserve_glue_island`) — so the
        // summariser errored and re-enqueued every probe forever, and no
        // conversation on such a model was ever summarised. A1's suite caught
        // it the first time the WARNs were visible.
        let instruction: Vec<u32> = {
            let mut t = self.boundary_markers.user_start.as_ref().clone();
            t.extend_from_slice(&self.boundary_markers.no_think);
            t.extend_from_slice(
                self.tokenizer
                    .encode(format!("\n\n{}", prompt.user_prompt), false)
                    .map_err(|e| format!("SubmitSummaryProbe: encode instruction: {e}"))?
                    .get_ids(),
            );
            t.extend_from_slice(&self.boundary_markers.user_end);
            t
        };
        segments.push(ProjectionSegment::NewUserMessage {
            tokens: Arc::new(instruction),
        });

        // Scratch slot bound to the timeline (so sealed-turn injection can resolve
        // it from `slot_target`). Freed once the pass completes.
        //
        // `Neutral`: the timeline is an ADDRESS here, not a history to continue.
        // A compression pass must be conditioned on the children it is
        // compressing — which this projection injects — and nothing else.
        // Seeding it from the timeline would let the compressor carry forward
        // content its children never contained, make each summary depend on how
        // far the conversation had got when the probe ran, and pay a ~63 MiB
        // state install per probe.
        let slot = self
            .create_sequence_seeded(conv.clone(), Some(target), StateSeed::Neutral)
            .map_err(|e| format!("SubmitSummaryProbe: create slot: {e}"))?;
        // Lift the children (and the compressor system section) into hot VRAM
        // before `setup_compression_decode` injects them. The children are
        // just-recorded scope/turn K/V that the tier system may already have
        // evicted to warm/cold; without this the compression pass's
        // `apply_projection` finds no hot residence and drops them, so the
        // summary would be decoded over missing content. The other
        // apply_projection callers (SubmitTurn, reproject) already elevate;
        // this path is the one that didn't.
        let child_keys: Vec<TurnKey> = children
            .iter()
            .map(|&c| TurnKey::new(target.timeline, c))
            .collect();
        self.elevate_projection_working_set(conv, &[summary_section], &child_keys, "summary");
        match self.setup_compression_decode(slot, &segments, max_tokens, job_id) {
            Ok(()) => {}
            Err(e) => {
                self.free_summary_slot(slot);
                // Setup failed before decode, so `complete_compression_pass` (which
                // normally demotes) never runs — demote the just-lifted children
                // here too, else they linger hot until ordinary eviction.
                conv.read().demote_turns_to_warm(&child_keys);
                return Err(e);
            }
        }
        Ok(slot)
    }

    /// Assemble + prefill + sample-first for one compression pass, then install
    /// its [`DecodeState`]. Split out so `enqueue_compression_pass` can free the
    /// scratch slot on every failure path.
    pub(super) fn setup_compression_decode(
        &mut self,
        slot: SequenceId,
        segments: &[ProjectionSegment],
        max_tokens: usize,
        job_id: u64,
    ) -> Result<(), String> {
        self.apply_projection(slot, BlockCount(0), segments)
            .map_err(|e| format!("SubmitSummaryProbe: assemble summary pass: {e}"))?;
        self.finish_compression_pass_setup(slot, max_tokens, job_id)
    }

    /// Prefill `assistant_start`, sample the first token, and register the
    /// `CompressionPass` decode.
    pub(super) fn finish_compression_pass_setup(
        &mut self,
        slot: SequenceId,
        max_tokens: usize,
        job_id: u64,
    ) -> Result<(), String> {
        let turn_start = Instant::now();
        // Prefill `assistant_start` to get the first-token logits and frame the
        // model to *answer* rather than continue the prompt.
        //
        // The already-closed reasoning block follows it, which is how a dialect
        // with no `/no_think` soft switch suppresses thinking (see
        // `BoundaryMarkers::no_think_block`); empty on families that use the
        // switch, which the instruction above already carries. Without it this
        // probe suppressed nothing on Qwen3.5/3.8 AND had no backstop —
        // `SamplingConfig::compression()` leaves the segment ids unresolved, so
        // neither the forced close nor the EOT ramp can fire — and the
        // compressor reasoned until it hit the response cap, storing the trace
        // as the summary.
        let mut asst_start = self.boundary_markers.assistant_start.as_ref().clone();
        asst_start.extend_from_slice(&self.boundary_markers.no_think_block);
        let config = SamplingConfig::compression();
        let mut sstate = self
            .sampling_states
            .remove(&slot)
            .ok_or_else(|| "SubmitSummaryProbe: missing sampling state".to_string())?;
        let first = self
            .run_prefill_sampled(slot, &asst_start, &config, &mut sstate)
            .map_err(|e| {
                self.sampling_states.insert(slot, sstate.clone());
                format!("SubmitSummaryProbe: prefill assistant_start and sample first: {e}")
            })?;
        self.sampling_states.insert(slot, sstate);
        let prefill_ms = turn_start.elapsed().as_secs_f64() * 1000.0;
        let prefill_token_count = self.session.sequence_offset(slot.0).unwrap_or(0);

        if self.is_eos(first) {
            return Err(
                "SubmitSummaryProbe: summary produced no tokens (immediate EOS)".to_string(),
            );
        }

        // Compression passes report completion through the job, not through
        // `TurnEvent`, so the event channel is a private sink. `batch_decode_step`
        // sends streamed `Token` events into it; the dropped receiver makes those
        // sends fail, which marks the decode finished — harmless, because
        // `cleanup_finished` reaps it the same way EOS / max_tokens would.
        let (event_tx, event_rx) = flume::unbounded();
        // Keep the receiver alive for the decode's lifetime so per-token sends
        // succeed; it is dropped when the pass's slot is freed.
        self.compression_event_sinks.insert(slot, event_rx);
        let health = {
            let mut hs = DecodeHealthState::new(
                self.health_config.repetition_window,
                self.health_config.health_log_capacity,
            );
            hs.apply_baseline_config(
                self.health_config.entropy_baseline_window,
                self.health_config.entropy_trend_relative_factor,
                self.health_config.entropy_trend_absolute_min_nats,
            );
            // Argmax: temperature ~0, so entropy checks self-disable (peaked
            // distributions are expected). Repetition / phrase-loop checks stay
            // active — the normal-wave health contract a foreground decode gets.
            hs.skip_entropy_checks = config.temperature <= 0.01;
            hs
        };
        // The turn's first token goes in through `push_generated` like every
        // other, so the reasoning boundary is seen even when the prefill's own
        // logits produced it.
        let mut state = DecodeState {
            event_tx,
            generated_tokens: TokenBuffer::default(),
            think_close_at: None,
            forwarded_generated: 0,
            draft_depth: DraftDepth::default(),
            pending_page_cut: false,
            pending_page_cut_after: None,
            max_tokens,
            sampling_config: config,
            seal_action: SealAction::CompressionPass { job_id },
            // A compression pass emits no tool calls.
            free_tool_calls_from_penalties: false,
            prefill_assistant_text: String::new(),
            finished: false,
            faulted: false,
            finish: FinishReason::Stop,
            decode_start: Instant::now(),
            decode_busy_us: 0,
            prefill_ms,
            prefill_token_count,
            turn_start,
            health,
            reprojection: None,
            boundary: BoundaryState::default(),
            last_projection_end: 0,
            post_decode_tokens: TokenBuffer::default(),
            belief: PriorBelief::default(),
            prefill_tokens: TokenBuffer::default(),
            user_text: String::new(),
            tags: Vec::new(),
            user_content_start: 0,
            user_content_end: 0,
            assistant_content_start: 0,
            no_think: false,
            wrote_tool_call: false,
            triggers: Arc::new(TriggerRegistry::new()),
            stencil: None,
            pending_mask: None,
            recorded_reply: None,
        };
        // The turn's first token always opens a page — the prefill/decode
        // boundary — so the reasoning starts one.
        state.push_committed(first, self.think_close, &self.page_break_tokens);
        // No speculative rewind in flight on the summarise path either.
        flush_page_cut(self.model.as_ref(), slot, &mut state);
        self.active_decodes.insert(slot, state);
        Ok(())
    }

    /// Complete the finished summarise pass: take its decoded tokens as the
    /// node's assistant half, pair them with the scope derived before the decode,
    /// and seal both halves through the shared [`Self::seal_compression_turn`] —
    /// which strips the think block and re-prefills the marker-framed exchange
    /// into role-coherent K/V.
    ///
    /// The decode's own K/V is deliberately dropped rather than snapshotted. It
    /// is assistant-role K/V for a bare body with no question in front of it, and
    /// it covers the raw generated tokens — including any `<think>` block. Keeping
    /// it would mean either storing the think block or storing K/V that no longer
    /// matches the stored token ids. The re-prefill costs one short prefill and
    /// buys a node that is a real exchange: a question in user-role position, its
    /// answer in assistant-role position.
    pub(super) fn complete_compression_pass(
        &mut self,
        slot: SequenceId,
        job_id: u64,
        generated: TokenBuffer,
    ) {
        let Some(job) = self.compression_jobs.remove(&job_id) else {
            // Job already torn down — just reclaim the slot.
            self.free_summary_slot(slot);
            return;
        };
        let CompressionJob {
            conversation,
            target,
            kind,
            scope_tokens,
            children,
            response_tx,
        } = job;
        let timeline = target.timeline;

        // The children were lifted to hot only so this pass's decode could attend
        // over them. Now that the summary has decoded, drop their hot copies
        // (keeping warm) so a long run of summary passes can't accumulate
        // transient hot residency and exhaust VRAM — the failure mode where the
        // persist thread's hot→warm migrate then can't get scratch and OOMs.
        // Cheap: no migrate, the warm copy already exists from before the lift.
        if !children.is_empty() {
            let demoted = conversation.read().demote_turns_to_warm(&children);
            if demoted > 0 {
                tracing::trace!(
                    target: "candle_conversation::persistence::tier",
                    demoted,
                    timeline = timeline.raw(),
                    "summary pass: demoted lifted children back to warm"
                );
            }
        }

        // The wave decode pushes the final token (EOS / max_tokens-th) WITHOUT
        // forwarding it, so its K/V never landed. Drop it so the token_ids align
        // 1:1 with the snapshotted chunk grid.
        let summary_tokens: Vec<u32> = generated
            .split_last()
            .map(|(_, rest)| rest.to_vec())
            .unwrap_or_default();
        if summary_tokens.is_empty() {
            let _ = response_tx.send(Err(ProbeError::Soft(
                "SubmitSummaryProbe: summary produced no forwarded tokens".to_string(),
            )));
            self.free_summary_slot(slot);
            return;
        }

        // The decode's K/V is not kept — the seal re-prefills the cleaned,
        // marker-framed exchange — so the scratch slot can go now.
        self.free_summary_slot(slot);

        // Pair the decoded body (the assistant half) with the scope derived
        // before the decode (the user half) and seal both through the shared
        // path: think-strip, marker-frame, re-prefill to role-coherent K/V,
        // record, persist tokens + wide-Q sigs, and reply to the summariser.
        if let Err(e) = self.seal_compression_turn(
            job_id,
            conversation,
            target,
            kind,
            children.iter().map(|k| k.index).collect(),
            scope_tokens,
            summary_tokens,
            response_tx,
        ) {
            // `seal_compression_turn` has already replied to the summariser on
            // every error path it owns; log for the operator and move on.
            tracing::warn!(timeline = timeline.raw(), "summary seal failed: {e}");
        }
    }
}
