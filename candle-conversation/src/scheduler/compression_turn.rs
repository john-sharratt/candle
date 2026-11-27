//! The summary probe that compresses a summary-tree node over its children into
//! one sealed turn.
//!
//! A structural repo-map node is built deterministically from its inputs and
//! sealed at once, because decoding it would only invite fabrication. A
//! model-decode node is one argmax decode of the summary body, and its user half
//! is derived from the children's scopes before the decode, so no decode ever
//! writes a question. Relief runs before the backpressure deferral, because on an
//! idle engine the summariser's own pending work is the only thing that can free
//! ground, and a bare deferral would never end. The completed body is re-prefilled
//! as a marker-framed exchange, so the stored turn has its question in user
//! position and its answer in assistant position.

use super::*;

impl Scheduler {
    /// Compress a §6 summary node over its children into a single summary turn.
    ///
    /// A structural (repo-map) node is built deterministically from its inputs
    /// and sealed synchronously ([`Self::seal_structural_turn`]) — no model
    /// decode. A model-decode node runs ONE decode-only pass
    /// ([`Self::enqueue_compression_pass`]): the children's full turns are
    /// injected sealed (zero re-prefill) after being lifted hot, a tiny
    /// summarise instruction is appended, and the model decodes one summary into
    /// a fresh chunk range that `complete_compression_pass` snapshots and
    /// records — no stitch, no re-prefill. Nothing blocks the scheduler: the
    /// handler returns as soon as the pass is enqueued, and a setup failure
    /// replies `Err` for a soft retry. The decode is argmax, capped at the
    /// `summary` level's `max_tokens`.
    ///
    /// `children` are the node's structural children carrying the content to
    /// compress.
    pub(super) fn handle_summary_probe(
        &mut self,
        timeline: TimelineId,
        kind: TurnKind,
        children: Vec<TurnIndex>,
        height: u8,
        response_tx: Sender<Result<TurnIndex, ProbeError>>,
    ) -> Result<(), String> {
        tracing::trace!(
            target: "candle_conversation::summariser",
            timeline = %timeline,
            children = ?children,
            "compression probe: enqueuing compression pass",
        );
        // Resolve the conversation + projection target that own this timeline.
        // We look up through any slot registered against the same target.
        let conv = self
            .slot_conversations
            .values()
            .find(|c| c.read().timeline_target(timeline).is_some())
            .cloned()
            .ok_or_else(|| {
                format!("SubmitSummaryProbe: no conversation registered for timeline {timeline}")
            })?;
        let (layer, group) = conv
            .read()
            .timeline_target(timeline)
            .ok_or_else(|| format!("SubmitSummaryProbe: timeline {timeline} has no target"))?;
        let target = ProjectionTarget {
            layer,
            group,
            timeline,
        };
        if children.is_empty() {
            return Err("SubmitSummaryProbe: node has no children to compress".to_string());
        }

        // The compression prompts + decode cap come from the target layer's
        // summary config, read off the projection Builder the scheduler captured
        // for this timeline on `SubmitTurn`. A node selects its tree-level via
        // `LayerSummary::for_kind` — `summaries` for a SummaryOfSummaries node
        // (falling back to `turns`), else `turns` — and within that level the
        // user pass uses the `question` prompt (user-message half) and the
        // assistant pass the `answer` prompt (assistant-response half). Cloned so
        // it is owned for the rest of this handler (the two passes borrow
        // `&mut self`).
        // Resolve the target layer's summary config from this timeline's builder,
        // falling back to the workspace builder for timelines reloaded from the
        // substrate (not in the live per-timeline map) — the schema is
        // workspace-global, so either resolves the same layer config.
        let turn_summary = self
            .timeline_projections
            .get(&timeline)
            .or(self.workspace_projection.as_ref())
            .and_then(|b| {
                b.schema()
                    .layers
                    .iter()
                    .find(|l| l.id == target.layer)
                    .map(|l| {
                        l.summary
                            .for_kind(matches!(kind, TurnKind::SummaryOfSummaries))
                            .clone()
                    })
            })
            .ok_or_else(|| {
                format!(
                    "SubmitSummaryProbe: no projection/summary for timeline {timeline} layer {:?}",
                    target.layer
                )
            })?;

        // Structural layers (repo_map directory trees) are built deterministically
        // from their inputs — no model decode at any level. A leaf strips the size
        // annotations off its one scan turn; a summary-of-summaries reconstructs
        // and height-truncates the children's directory paths. Decoding either is
        // pure cost and invites fabrication. `seal_structural_turn` dispatches on
        // kind and derives both halves.
        if turn_summary.content == Content::Structural {
            return self.seal_structural_turn(&conv, target, kind, &children, height, response_tx);
        }

        // Backpressure: a model-decode compression pass lifts its children to hot
        // and adds VRAM churn. The summariser is a background task, so when VRAM is
        // under pressure defer this node — a Soft error it retries on its next
        // reconcile pass — rather than piling onto a tight card and starving
        // foreground decode + the persist thread's hot→warm drain. (The structural
        // path above is deterministic and cheap, so it is deliberately not gated.)
        //
        // **Relief runs before the deferral, or the deferral never ends.** The
        // free-region count this gate reads is not scarcity under the elastic
        // partition — ground can be recycled from released arenas or bought back
        // from a weight side that took the span's slack while the engine was
        // quiet. And pressure is otherwise relieved only at end of wave, so on an
        // idle engine whose only pending work IS the summariser, a bare deferral
        // is a livelock: no waves without the probe, no relief without a wave, no
        // probe under pressure. Measured: the weight side had grown to 9.9 GiB
        // against a 124-region KV span, free sat at 9–22 under a setpoint of 24,
        // and the same turn was deferred at ~4 Hz for six minutes — thousands of
        // warns, and the summary tree starved for the life of the run. The
        // scheduler is by definition between waves here, which is exactly where
        // the relief ladder is legal — same call the elevate and section paths
        // make. Deferral remains for the case relief cannot fix: a genuinely
        // busy card, which is what this gate was built for.
        if self.vram_under_pressure() {
            self.relieve_vram_pressure(
                "summary",
                prefill::VramPhase::Load,
                prefill::ReliefNeed::Now,
            );
            if self.vram_under_pressure() {
                return Err("SubmitSummaryProbe: deferred — VRAM under pressure".to_string());
            }
        }

        // Reserve the job id up front so both passes tag their `SealAction`
        // with it; the job entry registers once both passes are set up.
        let job_id = self.next_compression_job_id;
        self.next_compression_job_id += 1;

        // Derive the node's user half — its scope — before the decode. It is a
        // pure function of the children's scopes, so it needs neither the model
        // nor the decode's result, and the decode is never asked to write it.
        let scope_tokens =
            self.derive_scope_tokens(&conv, target, &children, turn_summary.scope, height)?;

        // One decode, for the assistant half only. Inject the children's FULL
        // original turns (natural user→assistant roles, sealed — zero re-prefill),
        // add a tiny summarise instruction, open a fresh writer chunk, and decode
        // ONE summary body. `complete_compression_pass` then pairs it with the
        // derived scope above and seals both halves through the shared
        // `seal_compression_turn`, which strips the think block and re-prefills the
        // marker-framed exchange into role-coherent K/V.
        let slot = self.enqueue_compression_pass(
            &conv,
            target,
            &children,
            &turn_summary.assistant,
            turn_summary.max_tokens,
            job_id,
        )?;

        self.compression_jobs.insert(
            job_id,
            CompressionJob {
                conversation: conv,
                target,
                kind,
                scope_tokens,
                children: children
                    .iter()
                    .map(|&c| TurnKey::new(target.timeline, c))
                    .collect(),
                response_tx,
            },
        );
        // `slot` is carried into `active_decodes` by the pass; the completion
        // receives it from `cleanup_finished`, so the job need not hold it.
        let _ = slot;
        Ok(())
    }

    /// Derive a summary node's user half — its scope — from its children's, and
    /// encode it (`summary_tree::scope`).
    ///
    /// A child's scope is its user-half tokens: for a `Normal` turn that is the
    /// real question the user asked, and for a summary node it is the scope this
    /// same derivation gave it one level down. So a node's scope is grounded in
    /// real user text at every height, and no decode ever writes one — a decode
    /// always speaks as the assistant, so asking it for the question half is
    /// asking it to invent a question that was never asked.
    ///
    /// A `SummaryOfTurns` leaf covers one **exchange**, which answers exactly one
    /// question — the head turn's user half. The remaining members are tool
    /// round-trips whose user halves are `<tool_response>` output, not questions,
    /// so only the head is consulted at the leaf: a single-turn exchange keeps its
    /// question verbatim, and a tool exchange is scoped by the question that
    /// started it rather than by the JSON that served it.
    pub(super) fn derive_scope_tokens(
        &self,
        conv: &Conversation,
        target: ProjectionTarget,
        children: &[TurnIndex],
        scope: Scope,
        height: u8,
    ) -> Result<Vec<u32>, String> {
        // A node always summarises at least one child; an empty set means the
        // caller built a degenerate node, and deriving a scope from nothing would
        // silently seal an empty (question-less) summary. Fail loudly instead.
        if children.is_empty() {
            return Err("derive_scope_tokens: node has no children".to_string());
        }
        // At a leaf the children are one exchange's turns, and only its head
        // carries the question (see above). Above the leaf the children are
        // summary nodes, each already scoped — those all contribute.
        let scoped: &[TurnIndex] = if height <= 1 {
            &children[..1]
        } else {
            children
        };
        let child_scopes: Vec<String> = {
            let view = conv.read();
            scoped
                .iter()
                .map(|&c| {
                    view.turn_user_token_ids(target.timeline, c)
                        .unwrap_or_default()
                })
                .collect::<Vec<Vec<u32>>>()
        }
        .into_iter()
        .map(|toks| self.tokenizer.decode(&toks, true).unwrap_or_default())
        .collect();
        let combined = scope.combine(&child_scopes, height);
        Ok(self
            .tokenizer
            .encode(combined.as_str(), false)
            .map_err(|e| format!("SubmitSummaryProbe: encode derived scope: {e}"))?
            .get_ids()
            .to_vec())
    }

    /// Build and seal a structural node deterministically, skipping the model
    /// decode entirely (`content: structural`, the repo_map directory layer). A
    /// `SummaryOfTurns` leaf strips the size annotations off its one scan turn,
    /// keeping the full skeleton; a `SummaryOfSummaries` reconstructs the
    /// children's directory paths and truncates them by tree height (coarser
    /// toward the root). See `summary_tree::structural`.
    pub(super) fn seal_structural_turn(
        &mut self,
        conv: &Conversation,
        target: ProjectionTarget,
        kind: TurnKind,
        children: &[TurnIndex],
        height: u8,
        response_tx: Sender<Result<TurnIndex, ProbeError>>,
    ) -> Result<(), String> {
        // Read each child's assistant-half *tokens* and decode them. Normal scan
        // turns are prefill-only (no stored `assistant_text`), so we slice the
        // assistant body off their token ids rather than relying on the text field.
        let child_tokens: Vec<Vec<u32>> = {
            let view = conv.read();
            children
                .iter()
                .map(|&c| {
                    view.turn_assistant_token_ids(target.timeline, c)
                        .unwrap_or_default()
                })
                .collect()
        };
        let child_texts: Vec<String> = child_tokens
            .iter()
            .map(|toks| self.tokenizer.decode(toks, true).unwrap_or_default())
            .collect();
        let rollup = match kind {
            // Leaf: its one child is the raw Normal scan turn — strip annotations.
            TurnKind::SummaryOfTurns => leaf_skeleton(child_texts.first().map_or("", |s| s)),
            // SoS: its children are already-built directory skeletons.
            _ => structural_rollup(&child_texts, height),
        };
        // An empty structural node IS the "empty repo_map" symptom. Log only when
        // it happens, with enough to place blame: no children, a child whose
        // assistant K/V read back empty (upstream seal problem), or a non-empty
        // input the transform reduced to nothing (a parsing regression). Quiet on
        // the healthy path.
        if rollup.skeleton.trim().is_empty() {
            let child_lens: Vec<(u32, usize, usize)> = children
                .iter()
                .zip(&child_tokens)
                .zip(&child_texts)
                .map(|((c, toks), text)| (c.0, toks.len(), text.trim().len()))
                .collect();
            tracing::warn!(
                target: "candle_conversation::summariser",
                timeline = target.timeline.raw(),
                ?kind,
                height,
                n_children = children.len(),
                // (turn_index, assistant_token_len, decoded_char_len) per child.
                children = ?child_lens,
                scope_empty = rollup.scope.trim().is_empty(),
                "structural seal produced an EMPTY skeleton — this becomes an empty repo_map node",
            );
        }
        let user_tokens = self
            .tokenizer
            .encode(rollup.scope.as_str(), false)
            .map_err(|e| format!("SubmitSummaryProbe: encode structural scope: {e}"))?
            .get_ids()
            .to_vec();
        let assistant_tokens = self
            .tokenizer
            .encode(rollup.skeleton.as_str(), false)
            .map_err(|e| format!("SubmitSummaryProbe: encode structural skeleton: {e}"))?
            .get_ids()
            .to_vec();
        let job_id = self.next_compression_job_id;
        self.next_compression_job_id += 1;
        self.seal_compression_turn(
            job_id,
            conv.clone(),
            target,
            kind,
            children.to_vec(),
            user_tokens,
            assistant_tokens,
            response_tx,
        )
    }

    /// Seal a compression prompt's system-prompt framing as a content section in
    /// `conv`'s substrate and return its [`SectionId`], reusing the real section
    /// ingest path (`prepare_section_ingest` + `finalize_section_ingest`) rather
    /// than a bespoke seal. Idempotent and lazy: the first compression probe to
    /// use this prompt seals it; every later probe returns the same id with no
    /// work (the `section_sealed_of` check is the authoritative guard).
    ///
    /// The section id is the one the schema allocated for `prompt.system_prompt`
    /// (a normal `1..n` schema id, but never added to `system_prompt.items`, so
    /// it only ever materialises via this seal — never in a normal projection).
    /// The K/V is pinned in the conversation's own substrate, so a scheduler
    /// hosting several conversations re-seals into each one the first time that
    /// conversation runs a compression pass.
    pub(super) fn ensure_summary_section(
        &mut self,
        conv: &Conversation,
        prompt: &CompressionPrompt,
    ) -> Result<SectionId, ConversationError> {
        let section_id = prompt.system_prompt.id;

        // Already pinned in this conversation's substrate → nothing to do.
        if conv.read().section_sealed_of(section_id).is_some() {
            return Ok(section_id);
        }

        // Tokenize the prompt once and derive a stable content address so the
        // section gets its own persistence stream, disjoint from the schema's.
        let tokens: Vec<u32> = self
            .tokenizer
            .encode(prompt.system_prompt.content.as_str(), false)
            .map_err(|e| ConversationError::Channel(format!("summary section: encode: {e}")))?
            .get_ids()
            .to_vec();
        if tokens.is_empty() {
            return Err(ConversationError::Channel(
                "summary section: prompt encoded to zero tokens".into(),
            ));
        }
        let address = ContentChain::new().push_section(&tokens);
        let tokens = TokenBuffer::from(tokens);

        // Seal on a throwaway scratch slot through the section ingest path: lay
        // down the prefix (none) + writer chunk, prefill the prompt tokens, then
        // seal `[seal_block_from..block_count)` into the substrate via
        // `SealAction::Section`. The slot is freed regardless of outcome.
        let slot = self.create_sequence(conv.clone(), None)?;
        let result = (|| -> Result<(), ConversationError> {
            let seal_block_from = self.prepare_section_ingest(slot, section_id, &[], &tokens)?;
            self.run_prefill(slot, &tokens[..])?;
            self.finalize_section_ingest(
                slot,
                section_id,
                seal_block_from,
                Arc::new(tokens.to_vec()),
                address,
                "summary_system_prompt".to_string(),
                SectionSeal::Boundary,
                false,
            )?;
            Ok(())
        })();
        self.free_summary_slot(slot);
        result?;

        Ok(section_id)
    }

    /// Frame a compressed exchange (a user-half + assistant-half, already as
    /// token ids) as one clean marker-framed turn and enqueue it for the seal.
    /// Decode `tokens`, strip any `<think>…</think>` block, and re-encode the
    /// cleaned summary text. Returns the original tokens unchanged when there is
    /// no think tag (the common case pays only a decode + scan), so the
    /// deterministic structural path — which never emits a think block — is a
    /// no-op.
    pub(super) fn strip_think_from_tokens(&self, tokens: &[u32]) -> Vec<u32> {
        let Ok(text) = self.tokenizer.decode(tokens, true) else {
            return tokens.to_vec();
        };
        let lower = text.to_ascii_lowercase();
        if !lower.contains("<think>") && !lower.contains("</think>") {
            return tokens.to_vec();
        }
        let stripped = crate::think_strip::strip_think_blocks(&text);
        self.tokenizer
            .encode(stripped.as_str(), false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_else(|_| tokens.to_vec())
    }

    /// Shared by the model-decode path ([`Self::enqueue_compression_turn`]) and
    /// the deterministic structural path ([`Self::handle_summary_probe`]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn seal_compression_turn(
        &mut self,
        job_id: u64,
        conversation: Conversation,
        target: ProjectionTarget,
        kind: TurnKind,
        children: Vec<TurnIndex>,
        user_tokens: Vec<u32>,
        assistant_tokens: Vec<u32>,
        response_tx: Sender<Result<TurnIndex, ProbeError>>,
    ) -> Result<(), String> {
        // The node inherits the union of its children's gather-scope tags —
        // exact per node and recursive by construction (a SoS's children are
        // earlier summary nodes carrying their own inherited tags).
        let tags = conversation.union_turn_tags(target.timeline, &children);
        // Despite the `/no_think` directive the model may still emit an (empty)
        // `<think></think>` block before the summary. Summaries are stored as
        // plain content, so strip the block from each half here — before the
        // halves are stitched, re-prefilled, and recorded — so it never leaks
        // into the substrate or the GUI's summary view.
        let user_tokens = self.strip_think_from_tokens(&user_tokens);
        let assistant_tokens = self.strip_think_from_tokens(&assistant_tokens);

        // Frame the compressed exchange as a clean, marker-framed turn:
        // `[user_start][question][user_end][assistant_start][answer][im_end]`.
        //
        // **The head and tail are baked here, like any other turn.** They used
        // to be live `Generated` segments the assembler re-emitted around every
        // sealed turn; now a sealed turn owns its own boundaries, so a
        // compression turn that did not bake them would be projected with no
        // opener at all — a summary that runs straight on from the previous
        // turn's closer, with no role marker to say whose words these are.
        let user_start = self.boundary_markers.user_start.as_ref().clone();
        let user_end = self.boundary_markers.user_end.as_ref().clone();
        let assistant_start = self.boundary_markers.assistant_start.as_ref().clone();
        let assistant_end = self.boundary_markers.assistant_end.as_ref().clone();
        let mut token_ids: Vec<u32> = Vec::with_capacity(
            user_start.len()
                + user_tokens.len()
                + user_end.len()
                + assistant_start.len()
                + assistant_tokens.len()
                + assistant_end.len(),
        );
        token_ids.extend_from_slice(&user_start);
        let user_content_start = token_ids.len();
        token_ids.extend_from_slice(&user_tokens);
        let user_end_at = token_ids.len();
        token_ids.extend_from_slice(&user_end);
        token_ids.extend_from_slice(&assistant_start);
        let asst_start_at = token_ids.len();
        token_ids.extend_from_slice(&assistant_tokens);
        token_ids.extend_from_slice(&assistant_end);
        let token_count = token_ids.len();

        // Decode both halves' display text. On failure reply to the summariser
        // before returning — otherwise `response_tx` drops and its `recv()` turns
        // a soft, retryable error into a hard one.
        let user_text = match self.tokenizer.decode(&user_tokens, true) {
            Ok(t) => t,
            Err(e) => {
                let e = format!("SubmitSummaryProbe: decode user-half: {e}");
                let _ = response_tx.send(Err(ProbeError::Soft(e.clone())));
                return Err(e);
            }
        };
        let assistant_text = match self.tokenizer.decode(&assistant_tokens, true) {
            Ok(t) => t,
            Err(e) => {
                let e = format!("SubmitSummaryProbe: decode assistant-half: {e}");
                let _ = response_tx.send(Err(ProbeError::Soft(e.clone())));
                return Err(e);
            }
        };

        // Create the scratch slot the wave will re-prefill the marker-framed turn
        // onto. Its question body lands in user-message position (user-role K/V),
        // its answer body after `assistant_start` (assistant-role K/V) — the
        // role-coherent K/V that replaces the passes' decode-time K/V.
        // `Neutral` for the same reason as the compression pass: a scratch slot
        // that re-prefills a marker-framed turn, addressed by the timeline but
        // not continuing it.
        let slot = match self.create_sequence_seeded(
            conversation.clone(),
            Some(target),
            StateSeed::Neutral,
        ) {
            Ok(s) => s,
            Err(e) => {
                let e = format!("SubmitSummaryProbe: reproject slot: {e}");
                let _ = response_tx.send(Err(ProbeError::Soft(e.clone())));
                return Err(e);
            }
        };

        // Private event sink, kept alive so the prefill machinery's sends never
        // fail. This path has no decode, but `PrefillWork` still carries an
        // `event_tx`; the receiver is dropped with the slot in `free_summary_slot`.
        let (event_tx, event_rx) = flume::unbounded();
        self.compression_event_sinks.insert(slot, event_rx);

        // Stash the turn content for the deferred seal, then enqueue the
        // marker-framed tokens as a `max_decode=0` prefill unit. The shared wave
        // prefills it alongside the live turn and other summaries;
        // `complete_compression_turn` snapshots + records it when the prefill
        // finishes.
        // Build the compressed turn's segment layout. The exchange is framed
        // `[user_start][question][user_end][assistant_start][answer][im_end]`,
        // so the user body spans `[user_content_start, user_end_at)` and the
        // answer body starts at `asst_start_at` and stops short of the reserved
        // tail. Summaries strip their think block, so there is no thinking split
        // here.
        let layout = TurnLayout::from_flat_grid_with_tail(
            user_content_start as u32,
            user_end_at as u32,
            asst_start_at as u32,
            token_count as u32,
            user_end.len() as u32,
            assistant_start.len() as u32,
            assistant_end.len() as u32,
            user_text,
            Some(assistant_text),
            false,
        );
        self.pending_compression_seals.insert(
            job_id,
            PendingCompressionSeal {
                conversation,
                target,
                layout,
                token_ids: token_ids.clone(),
                kind,
                children,
                tags,
                response_tx,
            },
        );
        // An arrival is an admission opportunity — see the sibling enqueue.
        self.settled_since_admit = true;
        self.prefill_queue.push_back(PrefillWork {
            sequence_id: slot,
            tokens: TokenBuffer::from(token_ids),
            prefill_text: String::new(),
            user_text: String::new(),
            tags: Vec::new(),
            user_content_start: 0,
            user_content_end: 0,
            assistant_content_start: 0,
            no_think: false,
            prefill_assistant_text: String::new(),
            event_tx,
            max_decode_tokens: 0,
            sampling: SamplingConfig::compression(),
            submitted_at: Instant::now(),
            reprojection: None,
            belief: PriorBelief::default(),
            seal_action: SealAction::CompressionTurn { job_id },
            post_decode_tokens: TokenBuffer::default(),
            projection_offsets: Vec::new(),
            staged_composition: None,
            triggers: Arc::new(TriggerRegistry::new()),
            turn_grammar: None,
            free_tool_calls_from_penalties: false,
            recorded_reply: None,
            recurrent_spans: Vec::new(),
            image_spans: Vec::new(),
            rereads: false,
        });
        Ok(())
    }

    /// Seal a re-prefilled compressed turn once the shared wave finishes its
    /// prefill. Snapshots the slot's freshly-computed (role-coherent) K/V,
    /// records the turn from its [`PendingCompressionSeal`], frees the slot, and
    /// replies to the summariser with the new `TurnIndex`.
    pub(super) fn complete_compression_turn(&mut self, slot: SequenceId, job_id: u64) {
        let pending = match self.pending_compression_seals.remove(&job_id) {
            Some(p) => p,
            None => {
                // No pending entry (e.g. enqueue raced a teardown) — just reclaim.
                self.free_summary_slot(slot);
                return;
            }
        };
        // Reconcile a windowed creep's per-layer skew before this seal —
        // see `perform_seal_and_write` and `BatchedInferenceSession::reconcile_block_counts`.
        if let Err(e) = self.session.reconcile_block_counts(slot.0) {
            let _ = pending.response_tx.send(Err(ProbeError::Soft(format!(
                "SubmitSummaryProbe: reconcile before snapshot: {e}"
            ))));
            self.free_summary_slot(slot);
            return;
        }
        // **The block count comes from the snapshot, not from the slot**, and one
        // resolution serves both the slice and the `sign(Q)` gather below so they
        // cannot disagree. `snapshot_sequence_per_layer` drops each layer's trailing
        // empty chunk while `sequence_block_count` counts it, so the two differ by one
        // whenever that chunk is empty — see the note in
        // `projection_assembler::apply_projection`, where the same pair made a boot
        // from a fresh substrate impossible.
        let block_count;
        let sealed_gpu = match self.session.snapshot_sequence_per_layer(slot.0) {
            Ok(snap) => {
                block_count = snap.iter().map(|s| s.chunks.len()).min().unwrap_or(0);
                match slice_per_layer_sealed(&snap, 0, block_count) {
                    Ok(sliced) => sliced,
                    Err(e) => {
                        let _ = pending.response_tx.send(Err(ProbeError::Soft(format!(
                            "SubmitSummaryProbe: reproject slice: {e}"
                        ))));
                        self.free_summary_slot(slot);
                        return;
                    }
                }
            }
            Err(e) => {
                let _ = pending.response_tx.send(Err(ProbeError::Soft(format!(
                    "SubmitSummaryProbe: reproject snapshot: {e}"
                ))));
                self.free_summary_slot(slot);
                return;
            }
        };
        // Capture the node's wide per-token `sign(Q)` while the slot is still
        // live — the re-prefill's K/V is freshly R16 here, exactly as at a
        // normal turn seal. The whole slot is the marker-framed turn
        // (`reprojection: None`), so the range is 1:1 with the Tokens record.
        // Live seal (code-read roundtrip summary turns) — count it for the GUI's
        // sealing phase, timing the dominant sig-gather cost.
        let t_sig = Instant::now();
        let wide_sigs = self.gather_wide_sigs(slot, (0, block_count));
        self.wave_stats
            .add_seal(0, t_sig.elapsed().as_micros() as u64, 0);
        // The slice holds RAII `ChunkGid` clones, so the K/V survives the free.
        self.free_summary_slot(slot);
        let block_end = sealed_gpu.first().map(|s| s.chunks.len()).unwrap_or(0);
        let token_count = pending.token_ids.len();

        let timeline = pending.target.timeline;
        let conversation = pending.conversation;
        let response_tx = pending.response_tx;
        // Kept for the post-injection log + token persistence below — the rest
        // move into the write.
        let summary_question = pending.layout.user_text().to_string();
        let summary_answer = pending.layout.assistant_text().unwrap_or_default();
        // A summary node stands in for the real turns it compressed and is injected
        // as a synthetic user→assistant exchange. If EITHER half's decode pass
        // produced no text, the node is hollow — it would silently drop that slice
        // of history behind an empty slot. Refuse to seal it and surface the
        // failure to the summariser rather than committing a malformed summary.
        if summary_question.trim().is_empty() || summary_answer.trim().is_empty() {
            let msg = format!(
                "summary seal aborted for timeline {timeline}: a compression pass produced an \
                 empty half (question {} chars, answer {} chars) — refusing to seal a hollow \
                 summary node",
                summary_question.trim().chars().count(),
                summary_answer.trim().chars().count(),
            );
            tracing::error!(
                target: "candle_conversation::summariser",
                timeline = %timeline,
                "{msg}",
            );
            let _ = response_tx.send(Err(ProbeError::Permanent(msg)));
            return;
        }
        let persist_token_ids = pending.token_ids.clone();
        let write = TurnPartWrite {
            layout: pending.layout,
            token_ids: TokenBuffer::from(pending.token_ids),
            token_count,
            // The node inherits the union of its children's gather-scope
            // tags, so tag-scoped provenance galleries admit the summary in
            // place of the turns it compresses; untagged (dialogue) children
            // yield an untagged summary.
            tags: pending.tags.clone(),
            block_start: 0,
            block_end: block_end as u64,
            sealed_gpu: Some(Arc::new(sealed_gpu)),
        };
        let idx = match conversation
            .record_turn(timeline, Role::Assistant, write, |seqs| Ok(seqs.to_vec()))
        {
            Ok(idx) => idx,
            Err(e) => {
                let _ = response_tx.send(Err(ProbeError::Soft(format!(
                    "SubmitSummaryProbe: record compressed turn: {e}"
                ))));
                return;
            }
        };

        // Persist the summary's token ids. `record_turn` only declared the turn
        // stream; the normal seal path writes the `Tokens` record separately, so
        // the compression path must too — otherwise the compressed text is
        // unrecoverable from the substrate (only its K/V chunks survive).
        let stream_id = turn_stream_id(timeline.raw(), idx.0);
        conversation.enqueue_tokens(stream_id, persist_token_ids.clone());
        // Persist the node's wide-Q signature under the same stream — the
        // gallery entry a provenance scan matches against the summary. Stamped
        // with this process's fold like every other capture: an unstamped record
        // reads back with group sizes `0`, which the read path treats as "not
        // stated" and scores without checking, so a summary node written under
        // one geometry would be compared against turns written under another.
        if !wide_sigs.is_empty() {
            conversation
                .enqueue_wide_q_sigs(stream_id, encode_wide_sigs_with(&wide_sigs, self.prov_fold));
        }
        // Synthesize + persist the node's projection event: the mandatory
        // provenance linkage naming the node itself and the turns it covers
        // by `(timeline, index)`, so a wide-Q hit on the summary resolves to
        // real turns. Compression frames run no projection — the event's
        // system list is honestly empty.
        {
            let (layer_name, group_name) = self
                .timeline_projections
                .get(&timeline)
                .map(|b| {
                    let schema = b.schema();
                    (
                        layer_name_of_group(schema, pending.target.group)
                            .unwrap_or_default()
                            .to_string(),
                        group_name_of(schema, pending.target.group)
                            .unwrap_or_default()
                            .to_string(),
                    )
                })
                .unwrap_or_default();
            let children_meta: Vec<(TurnIndex, TurnKind, u32)> = {
                let read = conversation.read();
                pending
                    .children
                    .iter()
                    .map(|&c| {
                        let kind = read
                            .tree_meta_of(timeline, c)
                            .map(|m| m.kind)
                            .unwrap_or(TurnKind::Normal);
                        (c, kind, read.turn_token_count_of(timeline, c) as u32)
                    })
                    .collect()
            };
            let event = summary_node_event(
                &layer_name,
                &group_name,
                timeline.raw(),
                (idx, pending.kind, token_count as u32),
                children_meta,
            );
            conversation.persist_projection_events(stream_id, encode_events(&[event]));
        }
        self.persist_triggers.fire();

        tracing::info!(
            target: "candle_conversation::summariser",
            timeline = %timeline,
            index = idx.0,
            tokens = token_count,
            blocks = block_end,
            question = %summary_question,
            answer = %summary_answer,
            "compression: summary turn decoded and injected into substrate",
        );
        let _ = response_tx.send(Ok(idx));
    }
}
