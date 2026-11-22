//! Gives a conversation's live state and the test observables over it, and the
//! settings that decide how its later turns run.
//!
//! Live recurrent memory is read without sealing a turn, and digests compare it
//! by content, so a caller can observe a conversation without changing it. The
//! projection schema, reprojection policy, LoRA adapter and decode band are set
//! between turns. An adapter must be chosen while the slot is empty, because the
//! existing keys were written under the projections in force, and an unknown name
//! is an error rather than a fall back to the base weights. Reset clears local
//! key-value state and leaves the shared turns in place.

use super::*;

impl Sequence {
    /// This conversation's **live** recurrent memory, right now, without sealing
    /// a turn to see it.
    ///
    /// `None` when the model carries no such memory. The only other view is the
    /// record a seal writes, which forces a caller to advance the conversation
    /// to observe it — changing what it is measuring — and makes "does a freed
    /// conversation still hold memory?" unaskable.
    pub fn memory(&self) -> crate::Result<Option<Vec<ExportedLayerState>>> {
        let (tx, rx) = flume::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::ReadRecurrentMemory {
                sequence_id: self.id,
                response_tx: tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        replied(&rx).map_err(|_| ConversationError::SchedulerGone)?
    }

    /// The last projection's score-density selection for this conversation's
    /// timeline, rendered as one line: node ids with their origin tags.
    ///
    /// Test observability for the continuity oracles. The summary tree
    /// summarises turns on its own thread, and the selector picks NODES —
    /// summaries where they exist, raw turns where they do not — so what a
    /// projection attends over depends on how far that thread has gotten. Two
    /// conversations fed identical tokens can legitimately select different
    /// nodes, and nothing else observable from a test says so.
    ///
    /// `None` when no projection has run or the rule-based path was used.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn last_selection_debug(&self) -> Option<String> {
        let view = self.substrate.read();
        let diag = view.last_selection_of(self.target.timeline)?.clone();
        Some(
            diag.selected_nodes
                .iter()
                .zip(diag.origins.iter())
                .map(|(n, o)| format!("{n:?}:{o:?}"))
                .collect::<Vec<_>>()
                .join(" "),
        )
    }

    /// Digest of a sealed turn's hot K/V content — formats, palettes, scales,
    /// and gathered arena bytes, per layer per chunk. `None` when the turn
    /// holds no hot sealed K/V (not materialised, or no such turn).
    ///
    /// Test observability: the continuity oracles' last discriminator. Two
    /// conversations whose recurrent state, layout, selection, and materialized
    /// glue all compare equal can still decode differently only if the K/V
    /// bytes they attend over differ — and those bytes are visible nowhere
    /// else.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn turn_kv_digest(
        &self,
        timeline: TimelineId,
        index: u32,
    ) -> crate::Result<Option<ContentHash>> {
        let (tx, rx) = flume::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::ReadTurnKvDigest {
                sequence_id: self.id,
                timeline,
                index: TurnIndex(index),
                response_tx: tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        replied(&rx).map_err(|_| ConversationError::SchedulerGone)?
    }

    /// The conversation's sealed block count after its last completed turn —
    /// the coarsest observable of the projected layout.
    ///
    /// Test observability for the continuity oracles: two conversations fed the
    /// same tokens must land on the same block count, and a resumed
    /// conversation whose count differs from its uninterrupted twin has had its
    /// history laid out differently — positions shift, RoPE shifts, and decode
    /// legitimately diverges from there, with nothing else saying why.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn sealed_block_count(&self) -> usize {
        self.current_blocks.0
    }

    /// A stable digest of this conversation's live memory — `None` when it
    /// carries none.
    ///
    /// Equality of the digest is equality of the memory: it hashes every
    /// layer's bytes in order. Tests compare digests rather than megabytes, and
    /// a digest that matches when the memory does not would defeat the point,
    /// so this is a content hash and not a summary.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn memory_digest(&self) -> crate::Result<Option<ContentHash>> {
        Ok(self.memory()?.map(|layers| {
            let mut h = crate::persistence::content_hash::ContentHasher::new();
            for l in &layers {
                h.update(&l.layer_index.to_le_bytes());
                h.update(&l.state);
                h.update(&l.conv_tail);
            }
            h.finish()
        }))
    }

    /// True when this conversation's memory is entirely zeros — the shape a
    /// conversation takes when it has forgotten everything, which is otherwise
    /// indistinguishable from one that remembers.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn memory_is_empty(&self) -> crate::Result<bool> {
        Ok(self.memory()?.is_none_or(|layers| {
            layers
                .iter()
                .all(|l| l.state.iter().all(|&b| b == 0) && l.conv_tail.iter().all(|&b| b == 0))
        }))
    }

    /// Swap the projection schema this sequence projects + reprojects with.
    ///
    /// The new builder must share the current one's section ids (e.g. a clone
    /// with one collection's selection overridden) — its sealed-section KV is
    /// reused as-is. Takes effect on the next submitted turn (the prefill
    /// projection and the reprojection policy both read this on submit).
    ///
    /// Sections submitted to this conversation stay in its collections: they
    /// are merged into the new schema, so the builder passed in never carries
    /// them.
    pub fn set_projection(&mut self, projection: Arc<Builder>) {
        self.base_projection = projection;
        self.rebuild_projection();
    }

    /// Set when the turns submitted from now on reproject mid-decode: every
    /// `every_n_tokens` decoded tokens (`0`: never by count), and at any of
    /// `trigger_texts` once more than 16 tokens have passed (empty: never by
    /// text). Both off runs a turn on the working set its opening projection
    /// chose — the one-shot retrieval `docs/unbounded_agents.md` §9.12 ablates.
    ///
    /// The same rule cuts this conversation's later turns into gallery pages
    /// ([`Self::boundary_rule`]), so a caller comparing the two should change it
    /// for the turn being measured, after the history it reads is sealed.
    pub fn set_reprojection(&mut self, every_n_tokens: usize, trigger_texts: Vec<String>) {
        self.submitter.config.reproject_every_n_tokens = every_n_tokens;
        self.submitter.config.reproject_trigger_texts = trigger_texts;
    }

    /// The projection schema this sequence was last given — the getter paired
    /// with [`Self::set_projection`]. It does not include submitted sections,
    /// which [`Self::set_projection`] merges back in.
    ///
    /// Handing it back and setting it again is how a caller asks for a
    /// reprojection without changing what is projected, which is otherwise not
    /// expressible: reprojection is policy-driven from inside the decode loop.
    pub fn projection(&self) -> Arc<Builder> {
        Arc::clone(&self.base_projection)
    }

    /// Reset this conversation to its initial state.
    ///
    /// Clears the GPU KV cache and resets client-side state.  The next
    /// `submit_turn` will re-seed the system into the substrate via
    /// `ensure_system_ingested` and apply_projection will materialise
    /// it back onto the slot.
    ///
    /// The system prompt, config, and tokenizer are preserved; all turn
    /// history is cleared.
    ///
    /// Returns an error if a turn is currently in flight.
    pub fn reset(&mut self) -> crate::Result<()> {
        if self.turn_in_flight {
            return Err(ConversationError::TurnInFlight {
                sequence_id: self.id,
            });
        }

        let (response_tx, response_rx) = flume::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::ResetSequence {
                sequence_id: self.id,
                response_tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        response_rx
            .recv()
            .map_err(|_| ConversationError::SchedulerGone)??;

        self.pending_user = None;
        // The substrate is workspace-shared — don't reset its turn store
        // (other conversations would lose their history).  Only the local
        // KV state and provenance score cache are cleared here.
        self.current_blocks = BlockCount(0);

        // Clear turn history (keeps system prompt, config, beliefs).
        self.tree.clear_turns();

        Ok(())
    }

    /// [`Self::reset`]'s async counterpart — awaits the scheduler's
    /// acknowledgement with `recv_async` instead of parking the thread in a
    /// blocking `recv`, exactly as [`crate::TurnHandle::wait_async`] mirrors
    /// [`crate::TurnHandle::wait`]. Identical effect; which one a caller uses is
    /// a property of the caller.
    ///
    /// This is the one to use on an async hot path: the per-narration narrator
    /// reset runs on the tokio worker pool under a per-character lock, and a
    /// blocking round-trip there parks the worker so no other character's turn
    /// can be polled or submitted — the batched engine then forms one-session
    /// waves and the GPU sits idle between them.
    pub async fn reset_async(&mut self) -> crate::Result<()> {
        if self.turn_in_flight {
            return Err(ConversationError::TurnInFlight {
                sequence_id: self.id,
            });
        }

        let (response_tx, response_rx) = flume::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::ResetSequence {
                sequence_id: self.id,
                response_tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        response_rx
            .recv_async()
            .await
            .map_err(|_| ConversationError::SchedulerGone)??;

        self.pending_user = None;
        self.current_blocks = BlockCount(0);
        self.tree.clear_turns();

        Ok(())
    }

    /// Decode this conversation through a LoRA adapter loaded with the model,
    /// or `None` for the base model.
    ///
    /// The name is one given to [`ModelBuilder::lora`](crate::models::ModelBuilder::lora)
    /// when the engine was built. An unknown name is an error — at the first
    /// wave rather than here, because the model owns the registry — and never a
    /// silent fall back to the base weights, which would read as the fine-tune
    /// not working with nothing to point at.
    ///
    /// # Before the first turn
    ///
    /// The adapter changes the projections that write K/V, so it must be chosen
    /// while the slot is still empty. Calling this on a conversation that has
    /// already decoded is refused: the existing keys were produced by the
    /// projections in force when they were written, and attending them with
    /// queries from a different model is not an error anything can detect — the
    /// answers simply get worse.
    ///
    /// # What it costs
    ///
    /// Nothing is merged, so an adapted conversation shares the resident
    /// checkpoint with every unadapted one. What it does change is which wave it
    /// rides in: waves are grouped by adapter, so an adapted conversation
    /// co-batches with other conversations on the same adapter rather than with
    /// all of them.
    pub fn set_lora(&mut self, adapter: Option<&str>) -> crate::Result<()> {
        if self.turn_in_flight {
            return Err(ConversationError::TurnInFlight {
                sequence_id: self.id,
            });
        }
        let (response_tx, response_rx) = flume::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::SetSequenceAdapter {
                sequence_id: self.id,
                adapter: adapter.map(Arc::from),
                response_tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        response_rx
            .recv()
            .map_err(|_| ConversationError::SchedulerGone)??;
        Ok(())
    }

    /// Abandon an in-flight turn whose stream ended without a `Done` (e.g. the
    /// scheduler shut down mid-decode, or a scheduler error). Clears the local
    /// in-flight state so the next [`Self::submit_turn`]/[`Self::reset`] is not
    /// permanently rejected by the `turn_in_flight` guard. Best-effort on the
    /// scheduler side — a gone scheduler is ignored, since the only caller is
    /// already tearing the turn down.
    pub fn abort_turn(&mut self) {
        // Best-effort scheduler-side view cleanup; ignore a gone scheduler.
        let (response_tx, response_rx) = flume::bounded(1);
        if self
            .scheduler_tx
            .send(SchedulerRequest::ResetSequence {
                sequence_id: self.id,
                response_tx,
            })
            .is_ok()
        {
            let _ = response_rx.recv();
        }

        // Clear local turn state unconditionally so the sequence is reusable.
        self.turn_in_flight = false;
        self.pending_user = None;
        self.current_blocks = BlockCount(0);
        self.tree.clear_turns();
    }

    /// Extract raw K/V/Q float data from the KV cache for a set of layer
    /// indices, synchronously on the scheduler thread.
    ///
    /// Call this after [`Self::finish_turn`] and before dropping the sequence —
    /// the parent slot's KV cache is live in that window.
    ///
    /// Returns one entry per requested layer in the same order:
    /// `(layer_idx, Vec<(block_idx, k_flat, v_flat, q_flat)>)`.
    pub fn extract_raw_kvq(
        &self,
        layer_indices: Vec<usize>,
        block_range: Option<(usize, usize)>,
    ) -> crate::Result<Vec<(usize, Vec<(usize, Vec<f32>, Vec<f32>, Vec<f32>)>)>> {
        let (tx, rx) = flume::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::ExtractRawKvq {
                sequence_id: self.id,
                layer_indices,
                block_range,
                response_tx: tx,
            })
            .map_err(|_| crate::error::ConversationError::SchedulerGone)?;
        rx.recv()
            .map_err(|_| crate::error::ConversationError::SchedulerGone)?
    }

    /// The band this conversation's own work runs at: the one it records for
    /// itself under [`DecodePriority::METADATA_KEY`], or with none recorded
    /// its target layer's. A layer the schema does not list cannot be decoded
    /// at: the protective `High`, as the scheduler reads an unresolved slot.
    ///
    /// **What a turn waits on runs at the turn's band.** A conversation's
    /// frame and the sections submitted to it are sealed before its turn can
    /// be; sealed at `Low` whatever the conversation was for, an image
    /// prompt's content check — four one-token answers asked at `Urgent` —
    /// spent eighty-five seconds behind eight characters before its first
    /// question was submitted, and the caller had given up at sixty.
    pub fn band(&self) -> DecodePriority {
        self.substrate
            .conversation_metadata_value(self.target.timeline, DecodePriority::METADATA_KEY)
            .and_then(|name| DecodePriority::parse(&name))
            .unwrap_or_else(|| {
                self.projection
                    .schema()
                    .layers
                    .iter()
                    .find(|l| l.id == self.target.layer)
                    .map_or(DecodePriority::High, |l| l.decode_priority)
            })
    }

    /// Decode this conversation's turns at `band` from the next one on, in
    /// place of the band it was opened at; `None` returns it to that band.
    /// See [`SchedulerRequest::SetBand`].
    pub fn set_band(&self, band: Option<DecodePriority>) {
        let _ = self.scheduler_tx.send(SchedulerRequest::SetBand {
            sequence_id: self.id,
            band,
        });
    }

    /// Close this conversation, releasing the sequence slot.
    ///
    /// Consumes `self`. If not called, the slot is freed on drop (best-effort).
    pub fn close(mut self) -> crate::Result<()> {
        self.freed = true;
        self.scheduler_tx
            .send(SchedulerRequest::FreeSequence {
                sequence_id: self.id,
            })
            .map_err(|_| ConversationError::SchedulerGone)
    }

    // ── Accessors ──────────────────────────────────────────────────────

    /// Get the sequence ID.  This is the slot index allocated by the
    /// scheduler — the sequence's only identifier under the
    /// substrate-as-parent model (persistence is workspace-scoped).
    pub fn id(&self) -> SequenceId {
        self.id
    }

    /// Number of turns in this conversation.
    pub fn turn_count(&self) -> u64 {
        self.turn_counter
    }

    /// Get all turns in chronological order.
    ///
    /// Synthesizes a flat `[System?, User, Assistant, User, Assistant, ...]`
    /// view from the conversation tree. Useful for inspection and testing;
    /// the tree's own `tree().turns()` iterator gives richer metadata.
    pub fn turns(&self) -> Vec<Turn> {
        let mut result: Vec<Turn> = Vec::new();
        let mut id: u64 = 0;

        // System prompt (if any).
        let sys_text = self.tree.system_prompt_text();
        if !sys_text.is_empty() {
            result.push(Turn {
                id,
                role: Role::System,
                text: sys_text.to_string(),
                token_ids: TokenBuffer::from(self.tree.system_prompt_token_ids()),
            });
            id += 1;
        }

        // Paired user↔assistant exchanges.
        for ct in self.tree.turns() {
            let inner = ct.inner();
            result.push(Turn {
                id,
                role: Role::User,
                text: inner.user.text().to_string(),
                token_ids: TokenBuffer::from(inner.user.token_ids()),
            });
            id += 1;
            result.push(Turn {
                id,
                role: Role::Assistant,
                text: inner.assistant.text().to_string(),
                token_ids: TokenBuffer::from(inner.assistant.token_ids()),
            });
            id += 1;
        }

        result
    }

    /// The raw system prompt text (with any temporal marker suffix, or empty).
    pub fn system_prompt(&self) -> &str {
        self.tree.system_prompt_text()
    }

    /// Whether the state this sequence's newest sealed turn left is in a
    /// durable record, so dropping the sequence now and resuming the timeline
    /// later, on any slot, restores it rather than an older turn's.
    ///
    /// A model that carries no recurrent state has nothing to restore beyond
    /// the timeline's K/V, which sealing made durable: true at once. So is a
    /// timeline with no turn sealed yet.
    pub fn sealed_state_durable(&self) -> bool {
        if !self.model_core.carries_recurrent_state {
            return true;
        }
        let timeline = self.target.timeline;
        let sealed = self.substrate.read().turn_count(timeline);
        match sealed.checked_sub(1) {
            None => true,
            Some(newest) => self.substrate.snapshot_durable(timeline, newest),
        }
    }

    /// Whether a turn is currently in flight.
    pub fn is_in_flight(&self) -> bool {
        self.turn_in_flight
    }

    // ── Test-introspection helpers ────────────────────────────────────────────

    /// Returns `(completed_turns, window_size)` where `completed_turns` is the
    /// number of entries in `turn_block_counts` (one per sealed `send_turn` /
    /// `insert_turn` call) and `window_size` is `config.context_window_turns`.
    ///
    /// Context masking activates when `completed_turns > window_size`; at that
    /// point the model only attends to the system prompt plus the most recent
    /// `window_size` turns.  Use this in tests to assert that the window has
    /// not (or has) become active after a given number of turns.
    pub fn window_state(&self) -> (usize, usize) {
        (
            self.substrate.read().turn_count(self.target.timeline) as usize,
            self.config.context_window_turns,
        )
    }

    /// Read-only view of the conversation tree.
    ///
    /// The tree mirrors the flat `turns()` slice but carries the full Phase 1
    /// type system: `TurnId` temporal coordinates, `TurnType`, children links,
    /// and the system prompt as a first-class field. Use `tree().nodes()` to
    /// iterate over `ConversationNode`s and inspect per-turn metadata.
    pub fn tree(&self) -> &ConversationTree {
        &self.tree
    }

    /// Writable access to the conversation tree
    pub fn tree_mut(&mut self) -> &mut ConversationTree {
        &mut self.tree
    }

    /// Apply every cognitive task (summarization, …) that has finished,
    /// without blocking on the ones still running. Returns the number of
    /// results applied to the tree.
    ///
    /// Cognitive tasks run in the background on the scheduler: a turn
    /// launches them and returns without waiting. This runs at every turn
    /// boundary, and is public so a caller that wants a result sooner — or a
    /// test waiting for one — can poll between turns.
    pub fn poll_cognitive_tasks(&mut self) -> usize {
        self.tree.poll_tasks(Some((
            &self.submitter.scheduler_tx,
            &self.submitter.tokenizer,
        )))
    }

    /// Number of cognitive tasks still running in the background.
    pub fn pending_cognitive_tasks(&self) -> usize {
        self.tree.pending_task_count()
    }

    // ── Turn-lifecycle helpers ────────────────────────────────────────────────

    /// Build the per-turn [`ReprojectionPolicy`] when continuous
    /// re-projection is enabled in config.  Returns `None` when both
    /// the cadence and punctuation triggers are disabled — the
    /// scheduler then runs the turn as a single-shot prefill+decode
    /// without mid-decode view swaps.
    pub(super) fn build_reprojection_policy(&self) -> Option<ReprojectionPolicy> {
        let boundary = self.boundary_rule();
        if !boundary.is_active() {
            return None;
        }
        Some(ReprojectionPolicy {
            target: self.target,
            projection: Arc::clone(&self.projection),
            selection: self.selection.clone(),
            substrate: self.substrate.clone(),
            boundary,
            max_probe_tokens: self.config.reproject_max_probe_tokens.max(1),
            in_thinking: self.config.reproject_in_thinking,
        })
    }

    /// The rule reprojection boundaries fire by — and that the gallery cuts
    /// ingested turns by (`provenance::boundaries`).
    pub fn boundary_rule(&self) -> BoundaryRule {
        BoundaryRule::from_tokenizer(
            &self.tokenizer,
            &self.config.reproject_trigger_texts,
            self.config.reproject_every_n_tokens,
        )
    }
}
