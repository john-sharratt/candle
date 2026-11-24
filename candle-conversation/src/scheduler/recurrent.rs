//! Restores and forks a timeline's recurrent state, which a recurrence cannot be
//! rebuilt from K/V and so must be carried across turns, forks and restarts.
//!
//! A restore resolves its seed through the fork lineage, nearest ancestor first,
//! and refuses a snapshot that describes a turn the timeline did not recover,
//! since installing one would put the recurrent layers ahead of their K/V. Every
//! rejection leaves the slot at zeros and says why, because a conversation that
//! resumes without its memory reads just as fluently as one that has it. A fork
//! copies both classes of state, and the parent's index is closed before the
//! carve. The branch-checkpoint pass prefills its tokens on a throwaway slot and
//! returns the state they produce, outside the section batcher, because it needs
//! the state and none of the K/V.

use super::*;

/// Whether a snapshot taken at `turn_index` describes a turn this timeline
/// actually recovered.
///
/// Recovered turns are `0..recovered_turns`, so a snapshot at `recovered_turns`
/// or beyond names a turn that is not there. That is reachable **by design**:
/// the seal enqueues the snapshot *before* the turn's `Tokens` record, so the
/// only tear it can produce is a snapshot ahead of its turn rather than a turn
/// ahead of its snapshot — silent staleness being the worse of the two. The
/// trade only holds if the too-new snapshot is then rejected, because installing
/// one puts the recurrent layers a turn ahead of the K/V, and a model whose
/// state remembers a token its attention never saw reads perfectly.
///
/// Also what makes the turn-scoped tombstone decision work: shedding the turn a
/// snapshot was taken at leaves its index unreachable, and this rejects it
/// without the tombstone path needing to know snapshots exist.
pub(super) fn snapshot_within_recovered_history(turn_index: u32, recovered_turns: u32) -> bool {
    turn_index < recovered_turns
}

impl Scheduler {
    /// Install a timeline's persisted recurrent state onto a freshly created
    /// slot, or say — audibly — why it could not.
    ///
    /// Never returns an error: a conversation that cannot restore its state is
    /// still a conversation, and refusing to open it would be worse than
    /// recomputing. But every path out of here that leaves the slot at zeros
    /// logs a **distinguishable** reason, because "resumed with no memory" and
    /// "resumed correctly" are indistinguishable from the outside — both read
    /// fluently, and only one of them is right.
    ///
    /// Returns whether a snapshot was found and an install attempted — `false`
    /// means this slot starts from the sequence-start state.
    pub(super) fn restore_recurrent_state(
        &mut self,
        slot_id: SequenceId,
        conversation: &Conversation,
        timeline: TimelineId,
    ) -> bool {
        // **The seed is resolved through the fork lineage, nearest ancestor
        // first.**
        //
        // A conversation that continues another starts from where that one left
        // off — that is what "forking the recurrent buffer" means. A forked
        // conversation is minted on a FRESH timeline, so it has no snapshot of
        // its own on its first turn; without this it would start from the
        // sequence-start state and remember nothing of the documents its
        // ancestors read, even though its K/V context carries their turns
        // (`Substrate::inherited_chain`) — state and context disagreeing is the
        // exact failure the rest of this function exists to prevent.
        //
        // Recursive, and bounded: `inherited_chain` walks `forked_from` all the
        // way up, guards against a cycle, and stops at the token cap. It reads
        // oldest-first, so reversing it asks the NEAREST ancestor first — the
        // most specific memory available.
        let lineage = conversation.read().inherited_chain(timeline);
        let mut resolved = None;
        for candidate in lineage.iter().rev() {
            match conversation.read_recurrent_snapshot(*candidate) {
                Ok(Some(p)) => {
                    resolved = Some((p, *candidate));
                    break;
                }
                Ok(None) => continue,
                Err(e) => {
                    tracing::warn!(
                        "RECURRENT RESUME FAILED (unreadable) for timeline {candidate}: {e} — \
                         the conversation will continue with NO recurrent memory of its \
                         history. It will read fluently and have forgotten."
                    );
                    return false;
                }
            }
        }
        let (payload, snapshot_timeline) = match resolved {
            Some(found) => found,
            None => {
                // Not an error and not always worth a warning: a model with no
                // recurrent state never writes one, and a conversation whose
                // first turn has not sealed has nothing to write yet.
                note(
                    "no recurrent snapshot: starting from the sequence-start state",
                    || format!("timeline={timeline}"),
                );
                return false;
            }
        };
        if snapshot_timeline != timeline {
            tracing::debug!(
                "timeline {timeline} seeds its recurrent state from ancestor \
                 {snapshot_timeline} (turn {})",
                payload.turn_index,
            );
        }

        // **The torn-shutdown check, and the reason the seal writes in the order
        // it does.**
        //
        // The snapshot is enqueued BEFORE the turn's `Tokens` record precisely
        // so the only reachable tear is a snapshot for a turn whose records
        // never landed — never the reverse, which would be silent staleness.
        // That trade is only sound if the too-new snapshot is then *rejected*:
        // installed, it puts the recurrent layers one turn ahead of the K/V,
        // which is `state without KV` — the mirror of the defect this whole path
        // exists to remove, and just as fluent.
        //
        // Recovered turns are `0..turn_count`, so a snapshot taken at index
        // `turn_count` or beyond describes a turn this timeline does not have.
        // It is also what makes the turn-scoped tombstone decision work (§4.2):
        // shedding the turn the snapshot was taken at leaves the index
        // unreachable, and this rejects it without the tombstone path needing to
        // know about snapshots at all.
        // Judged against the timeline the snapshot BELONGS to, not the one
        // being seeded. The check asks "does this snapshot describe a turn that
        // actually survived?" — a question about the conversation that wrote
        // it. Measuring an ancestor's snapshot against a fresh fork's own
        // (zero) turn count would reject every inherited seed on sight, which
        // is the quiet failure this whole path exists to remove.
        let recovered_turns = conversation.read().turn_count(snapshot_timeline);
        if !snapshot_within_recovered_history(payload.turn_index, recovered_turns) {
            tracing::warn!(
                "RECURRENT RESUME REJECTED (snapshot is newer than the recovered \
                 history) for timeline {snapshot_timeline}: the snapshot was taken at \
                 turn {} but only {recovered_turns} turn(s) recovered — a torn shutdown \
                 between the snapshot and the turn's records. Installing it would \
                 put the recurrent layers a turn ahead of the K/V. The conversation \
                 continues with NO recurrent memory of its history.",
                payload.turn_index,
            );
            return false;
        }

        // The decoded rows move into the staged snapshot: copied, a restore
        // would hold the whole state twice and free one of them here.
        let (turn_index, schedule_hash) = (payload.turn_index, payload.schedule_hash);
        let layers: Vec<ExportedLayerState> = payload
            .layers
            .into_iter()
            .map(ExportedLayerState::from)
            .collect();
        let n_layers = layers.len();

        // **Staged, not built.** The store is claimed where its slot is next
        // used — the view's fork at admission, or a wave on the slot — so a
        // resume, or a parked slot coming back (`recurrent_park`), holds no
        // device ground until then. A slot that already holds a store takes
        // the state at once.
        match self
            .model
            .stage_recurrent(slot_id.0, schedule_hash, Arc::from(layers))
        {
            Ok(true) => {
                tracing::debug!(
                    "restored recurrent state for timeline {timeline} at turn \
                     {turn_index} ({n_layers} layers)",
                );
            }
            Ok(false) => {
                // The model carries no recurrent state. A snapshot exists, so
                // the timeline was sealed by a different model — worth saying,
                // because it means this conversation's history was built under
                // an architecture this process is not running.
                tracing::warn!(
                    "RECURRENT RESUME SKIPPED (model carries no recurrent state) for \
                     timeline {timeline}: a snapshot exists, so this conversation was \
                     sealed by a different model"
                );
            }
            Err(e) => {
                // `import` validates the schedule hash and every layer's
                // geometry before touching a tensor, so this is a different
                // model or a changed layer schedule, and the store is
                // untouched. Recomputing is correct; doing it silently is not.
                tracing::warn!(
                    "RECURRENT RESUME REFUSED (hash or geometry mismatch) for timeline \
                     {timeline} at turn {}: {e} — the conversation will continue with \
                     NO recurrent memory of its history.",
                    payload.turn_index,
                );
            }
        }

        // The model's own state, restored from the same record. Independent of
        // the layer restore above: a model may carry one, the other, or both,
        // and the delta-rule import failing on geometry says nothing about
        // whether this blob is installable.
        self.restore_aux_from_payload(slot_id, timeline, payload.turn_index, &payload.aux);
        true
    }

    /// Install a snapshot's model-opaque blob, reporting what happened.
    ///
    /// Shared by the two paths that install a persisted state — a timeline
    /// resume and a branch checkpoint — because the interesting part is the
    /// reporting, and a path that quietly skipped the blob would produce
    /// exactly the fluent-and-forgotten resume the recurrent path is loud
    /// about.
    pub(super) fn restore_aux_from_payload(
        &self,
        slot_id: SequenceId,
        timeline: TimelineId,
        turn_index: u32,
        aux: &[u8],
    ) {
        if aux.is_empty() {
            return;
        }
        match self.model.restore_aux_state(slot_id.0, aux) {
            Ok(true) => {
                tracing::debug!(
                    "restored {} bytes of auxiliary state for timeline {timeline} at \
                     turn {turn_index}",
                    aux.len(),
                );
            }
            Ok(false) => {
                tracing::warn!(
                    "AUXILIARY RESUME SKIPPED (model reads no auxiliary state) for \
                     timeline {timeline}: {} bytes were persisted, so this conversation \
                     was sealed by a different model",
                    aux.len(),
                );
            }
            Err(e) => {
                tracing::warn!(
                    "AUXILIARY RESUME REFUSED for timeline {timeline} at turn \
                     {turn_index}: {e} — the conversation continues without that \
                     state.",
                );
            }
        }
    }

    // —— Raw KVQ extraction ————————————————————————————————————————————

    /// Extract raw K/V/Q float data for multiple layer indices.
    ///
    /// Prefill `tokens` on a throwaway slot and export the recurrent state that
    /// results — the branch checkpoint pass.
    ///
    /// Deliberately not routed through the section-ingest batcher. That path
    /// exists to make many concurrent section prefills share forwards, and it
    /// finishes by sealing K/V into the substrate; this wants neither. It runs
    /// once per branch at prompt-build time, wants no K/V kept, and wants only
    /// the state — so it drives `forward_wave` directly, in
    /// `max_prefill_pass_tokens` chunks, and throws the slot away.
    ///
    /// The chunking is not an optimisation: one forward over a whole 2,000-token
    /// prompt is a transient activation spike large enough to page, which is the
    /// same reason `build_section_batch` bounds its own per-forward budget.
    ///
    /// With `start`, the stream continues from that state: it is staged on the
    /// slot before the first token, so what comes back is the state after the
    /// prefix `start` describes and then `tokens`.
    pub(super) fn handle_branch_checkpoint_pass(
        &mut self,
        start: Option<&SharedState>,
        tokens: &[u32],
    ) -> Result<Option<ExportedState>, ConversationError> {
        if !self.model.carries_recurrent_state() {
            return Ok(None);
        }
        if tokens.is_empty() {
            return Ok(None);
        }
        let seq = self
            .session
            .create_sequence()
            .map_err(ConversationError::Model)?;
        let staged = match start {
            Some(state) => self.stage_state(seq, state).map(|_| ()),
            None => Ok(()),
        };
        let result = staged
            .and_then(|()| self.prefill_tokens_on(seq, tokens, &[]))
            .and_then(|()| self.export_state_of(seq));
        let _ = self.model.release_sequence(seq);
        let _ = self.session.free_sequence(seq);
        result
    }

    /// Stage `state` on `seq`, for the slot to take as its own the first time
    /// its state is wanted. Both classes are attempted whatever happens to the
    /// other — they validate independently, and a schedule-hash mismatch on the
    /// delta-rule rows says nothing about whether this model can read the aux
    /// bytes — and the first error is returned. `Ok(false)` when the model took
    /// neither.
    pub(super) fn stage_state(
        &self,
        seq: usize,
        state: &SharedState,
    ) -> Result<bool, ConversationError> {
        let mut staged = false;
        let mut first_err = None;
        if !state.layers.is_empty() {
            match self
                .model
                .stage_recurrent(seq, state.schedule_hash, Arc::clone(&state.layers))
            {
                Ok(did) => staged |= did,
                Err(e) => first_err = Some(ConversationError::Model(e)),
            }
        }
        if !state.aux.is_empty() {
            match self.model.restore_aux_state(seq, &state.aux) {
                Ok(did) => staged |= did,
                Err(e) => {
                    first_err.get_or_insert(ConversationError::Model(e));
                }
            }
        }
        match first_err {
            Some(e) => Err(e),
            None => Ok(staged),
        }
    }

    /// Both classes of a sequence's recurrent state, from one export.
    ///
    /// `None` only when the model produced neither — a model may carry either
    /// alone, so the layers being absent is not on its own an answer.
    pub(super) fn export_state_of(
        &self,
        seq: usize,
    ) -> Result<Option<ExportedState>, ConversationError> {
        let recurrent = self
            .model
            .export_recurrent(seq)
            .map_err(ConversationError::Model)?;
        let aux = self
            .model
            .export_aux_state(seq)
            .map_err(ConversationError::Model)?
            .unwrap_or_default();
        let (schedule_hash, layers) = recurrent.unwrap_or_default();
        let state = ExportedState {
            schedule_hash,
            layers,
            aux,
        };
        Ok(if state.is_empty() { None } else { Some(state) })
    }

    /// Run `tokens` through the model on `seq`, in [`Self::prefill_pass_budget`]
    /// chunks, the pads of `image_spans` as their rows.
    ///
    /// The chunking is not an optimisation: one forward over a whole prompt is a
    /// transient activation spike large enough to page, which is the same reason
    /// `build_section_batch` bounds its own per-forward budget.
    pub(super) fn prefill_tokens_on(
        &mut self,
        seq: usize,
        tokens: &[u32],
        image_spans: &[ImageSpan],
    ) -> Result<(), ConversationError> {
        let cap = self.prefill_pass_budget();
        let n_layers = self.model.num_layers();
        let mut off = 0usize;
        while off < tokens.len() {
            let advance = (tokens.len() - off).min(cap);
            let ids = engine_ids(&tokens[off..off + advance], off, image_spans);
            // On the host: the forward lays the ids on its own span.
            let input = Tensor::new(ids.as_slice(), &Device::Cpu)
                .and_then(|t| t.unsqueeze(0))
                .map_err(ConversationError::Model)?;
            let step = self
                .model
                .forward_wave(
                    &mut self.session,
                    &[],
                    &[],
                    &[seq],
                    std::slice::from_ref(&input),
                    &[],
                    &[],
                    0,
                    n_layers,
                    None,
                )
                .map_err(ConversationError::Model)?;
            drop(step);
            self.session
                .advance_sequence(seq, advance)
                .map_err(ConversationError::Model)?;
            off += advance;
        }
        Ok(())
    }

    /// Advance `sequence_id`'s recurrent memory over `tokens` **without adding
    /// K/V for them** — the catch-up a splice needs.
    ///
    /// `adopt_turn` moves a scope fork's sealed turns onto this conversation's
    /// timeline **by reference**: the K/V arrives without ever passing through
    /// this conversation's forward, so its memory has not seen those tokens. §4.4
    /// calls that shape a divergent join and puts it out of scope — but the
    /// `code_read` splice performs one, and left alone it leaves a conversation
    /// holding K/V its memory never saw, which is this document's whole subject.
    ///
    /// A recurrence *can* absorb them, because they land at the tail rather than
    /// mid-sequence. What it must not do is write their K/V a second time, so the
    /// work happens on a scratch slot seeded from this conversation's memory and
    /// the result is moved back: fork the memory out, run the tokens, move it
    /// home, throw the scratch K/V away.
    ///
    /// **The scratch is given the conversation's history as context.** The
    /// adopted span has to be replayed under the left context a real append
    /// would have given it — this conversation's own turns — because the
    /// attention layers produce what the recurrent layers consume. A bare slot
    /// (the first version) replays them against an empty context and moves home
    /// a state no forward could have produced: fluent, wrong, and invisible to
    /// a recall probe, since the spliced K/V answers from attention either way.
    /// `adopted_from` is the first adopted turn index — everything before it is
    /// injected as context, everything from it on is the span being replayed
    /// (already spliced by reference, so injecting it too would double it). The
    /// scratch's own written K/V is discarded with the slot, so nothing is
    /// written twice.
    pub(super) fn handle_memory_catch_up(
        &mut self,
        sequence_id: SequenceId,
        tokens: &[u32],
        adopted_from: u32,
        image_spans: &[ImageSpan],
    ) -> Result<bool, ConversationError> {
        if !self.model.carries_recurrent_state() || tokens.is_empty() {
            return Ok(false);
        }
        // Give the replay this conversation's history as its left context, by
        // INJECTING it onto the scratch from the substrate.
        //
        // The history's K/V lives in the substrate, not necessarily on the
        // conversation's slot: a slot only carries K/V once a projection has
        // materialised it, and a splice routinely lands before the file
        // conversation has submitted anything (its slot holds zero blocks —
        // measured, not assumed). So borrowing from the slot yields nothing in
        // the very flow this exists for. Arc-injecting the prior turns is the
        // same zero-copy operation `apply_projection` performs, and it works
        // whether or not the slot has been materialised.
        //
        // Turns at or after `adopted_from` are the span being replayed; their
        // K/V is already spliced onto this timeline by reference, so injecting
        // them here as well would double them.
        let scratch = self
            .session
            .create_sequence()
            .map_err(ConversationError::Model)?;
        // **Seed before injecting, then clear the per-position state the seed
        // brought.**
        //
        // The fork is here for the recurrent memory — the GDN/PLE state this
        // replay exists to rebuild — but it copies the whole carried state,
        // per-position index included, and that part describes the LIVE
        // sequence's layout rather than the one assembled below. Running it
        // after the injections also overwrote them. Scratch's index has to
        // describe scratch: cleared here, then rebuilt piece by piece as each
        // turn is injected, exactly as a projection assembles a slot.
        self.model
            .fork_recurrent(sequence_id.0, scratch)
            .map_err(ConversationError::Model)?;
        self.model
            .reset_positional_state(scratch)
            .map_err(ConversationError::Model)?;
        let mut injected = 0usize;
        if let (Some(conversation), Some(target)) = (
            self.slot_conversations.get(&sequence_id).cloned(),
            self.slot_targets.get(&sequence_id).copied(),
        ) {
            let prior: Vec<TurnIndex> = {
                let read = conversation.read();
                read.turn_indices(target.timeline)
                    .filter(|idx| idx.0 < adopted_from)
                    .collect()
            };
            // Lift them hot first: the tier system may have evicted older turns
            // to warm/cold, and an un-elevated turn injects nothing — the
            // compression pass learned the same lesson.
            let keys: Vec<TurnKey> = prior
                .iter()
                .map(|idx| TurnKey::new(target.timeline, *idx))
                .collect();
            self.elevate_projection_working_set(&conversation, &[], &keys, "memory catch-up");
            for idx in prior {
                let sealed = conversation.read().turn_sealed_of(target.timeline, idx);
                if let Some(sealed) = sealed {
                    match self.session.inject_sealed_at_tail(scratch, &sealed) {
                        Ok(placed) => {
                            injected += 1;
                            if placed.was_cut() {
                                tracing::warn!(
                                    timeline = target.timeline.raw(),
                                    index = idx.0,
                                    cut_chunks = placed.cut_chunks,
                                    injected_tokens = placed.tokens,
                                    stored_tokens = sealed.first().map_or(0, |s| s.token_count),
                                    "memory catch-up: the turn's layers describe different \
                                     token windows; the replay slot received their common prefix"
                                );
                            }
                            // The rows for the K/V just injected. Borrowing the
                            // turn's keys does not bring its index across — the
                            // turn's page, read as a projection reads it.
                            let page = conversation
                                .turn_index_page(target.timeline, idx)
                                .unwrap_or_else(|e| {
                                    tracing::warn!("memory catch-up: {e}");
                                    None
                                });
                            // The stored page is a framed LIST of pages, and the
                            // model reads one page at a time — see
                            // `index_pages::push_in_order`. However the pages fare,
                            // the slot finishes past this turn's whole width, so the
                            // next turn's rows land where its K/V does: the same rule
                            // `apply_projection` follows.
                            let width = placed.tokens;
                            let model: &(dyn ManagedBatchedModel + Send) = &*self.model;
                            let carries = model.carries_positional_state();
                            // A skip that fails leaves every later turn's rows short
                            // of their K/V, so it is reported rather than dropped.
                            let advance = |gap: usize| {
                                if !carries {
                                    return;
                                }
                                if let Err(e) = model.push_positional_gap(scratch, gap) {
                                    tracing::warn!(
                                        "memory catch-up: turn {idx} could not advance the \
                                         replay slot over {gap} unindexed token(s) ({e}) — \
                                         every later turn's rows now sit short of their K/V"
                                    );
                                }
                            };
                            match page {
                                Some(blob) => {
                                    let pushed = index_pages::push_in_order(
                                        &blob,
                                        |p, key| {
                                            model.push_positional_state(scratch, p, key).map(|_| ())
                                        },
                                        &advance,
                                    );
                                    match pushed {
                                        Ok(pushed) => {
                                            if let Some(e) = &pushed.refused {
                                                tracing::warn!(
                                                    "memory catch-up: turn {idx} index page \
                                                     refused ({e}) — advanced over {} unindexed \
                                                     token(s)",
                                                    pushed.gap
                                                );
                                            }
                                        }
                                        Err(e) => {
                                            tracing::warn!(
                                                "memory catch-up: turn {idx} index pages \
                                                 unreadable ({e}) — the replay selects against \
                                                 a prefix it never indexed"
                                            );
                                            advance(width);
                                        }
                                    }
                                }
                                None if carries => {
                                    advance(width);
                                    tracing::warn!(
                                        "memory catch-up: turn {idx} has no index page — the \
                                         replay selects against a prefix it never indexed"
                                    );
                                }
                                None => {}
                            }
                        }
                        Err(e) => tracing::warn!(
                            "memory catch-up: could not inject turn {idx} as context ({e}) — \
                             the adopted span replays with less context than a real append"
                        ),
                    }
                }
            }
        }
        tracing::debug!(
            "memory catch-up: replaying {} adopted tokens over {injected} injected turns",
            tokens.len(),
        );
        let result = self.prefill_tokens_on(scratch, tokens, image_spans).and_then(|()| {
            self.model
                .move_recurrent(scratch, sequence_id.0)
                .map_err(ConversationError::Model)
        });
        let _ = self.model.release_sequence(scratch);
        let _ = self.session.free_sequence(scratch);
        result.map(|()| true)
    }

    /// Fork `parent_id`'s per-sequence model state into its view `view_id`.
    ///
    /// The view borrows the parent's KV blocks zero-copy; per-sequence model
    /// state cannot be borrowed the same way, because the view is about to
    /// advance it. Copy it instead.
    ///
    /// Without this a hybrid's recurrent layers carry nothing across a turn
    /// boundary: the view is a distinct sequence id, so it would start from
    /// the sequence-start value while its KV holds the parent's entire
    /// history — three quarters of the stack contributing nothing but a
    /// function of the current turn, fluently and without an error.
    ///
    /// A fork that fails leaves the view with no state and the view itself in
    /// place — the caller decides whether it is freed or waits to fork again.
    pub(super) fn fork_view_state(
        &mut self,
        parent_id: SequenceId,
        view_id: SequenceId,
    ) -> Result<(), ConversationError> {
        if let Err(e) = self.model.fork_recurrent(parent_id.0, view_id.0) {
            let _ = self.model.release_sequence(view_id.0);
            return Err(ConversationError::Model(e));
        }
        FORK_RECURRENT_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // The view borrows the parent's K/V blocks but copies its per-position
        // state, so the two have to agree at the moment of the carve. A view
        // that starts short stays short for its whole life and says nothing
        // until a selection past the identity threshold refuses — so the
        // mismatch is named here, where the parent that caused it is still in
        // hand.
        if let Some(cov) = self.model.positional_coverage(view_id.0) {
            let parent_tokens = self.session.sequence_offset(parent_id.0).unwrap_or(0);
            if cov < parent_tokens {
                // **The PARENT's coverage too, because it decides where to look.**
                // A child short of a parent that is itself complete is a fork
                // that dropped rows; a child short of a parent that was already
                // short is the parent's prefix having arrived unindexed, and the
                // fork is faithfully copying nothing. Without both numbers the
                // warning names the carve for a loss that happened before it.
                let parent_cov = self.model.positional_coverage(parent_id.0);
                tracing::warn!(
                    parent = parent_id.0,
                    view = view_id.0,
                    parent_tokens,
                    parent_coverage = ?parent_cov,
                    view_coverage = cov,
                    "fork_view_state: forked a view over {} token(s) its index does not cover",
                    parent_tokens - cov
                );
            }
        }
        Ok(())
    }
}
