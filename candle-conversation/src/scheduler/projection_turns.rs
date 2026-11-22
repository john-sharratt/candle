//! Projects a submitted turn onto its slot, materialising the selected sections
//! and turns as sealed K/V and carving the view the turn decodes over.
//!
//! Writing a projection replaces the slot's prefix with the selected pieces,
//! injected as windowed chunks that share their source's memory, so the
//! substrate's record asserts only the bytes each window covers. The view borrows
//! every block the parent holds and is carved only after the parent's index is
//! closed, so the fork copies a closed index and the turn's own rows never share
//! a page with the prefix. Working-set pins are published before any unit is
//! lifted, and a retired slot's pins are released so its residences can be
//! evicted again.

use super::*;

/// Inputs the scheduler needs to run `Builder::project()` itself for a
/// `SubmitTurn` request.  The target — `(layer, group, timeline)` — is
/// **not** carried here; the scheduler reads it from
/// [`Scheduler::slot_targets`] (pinned at `NewSequence` /
/// `ResumeSequence` time).  Callers only need to supply the schema and
/// the conversation's pinned system-section id.
#[derive(Clone)]
pub(crate) struct ProjectionInputs {
    pub projection: Arc<Builder>,
    /// Section-tree selection for this turn's projection (the composer dials).
    pub selection: SelectionState,
    /// What an append-only conversation's first turn lays onto an empty slot
    /// instead of projecting: the schema's fixed sections, each tree at its
    /// default branch, then its lineage's turns (`conversation::Frame`). Its
    /// later turns skip projection (`disable_reprojection`), so this is the
    /// context its sealed K/V is computed under for the conversation's whole
    /// life, and no turn's selection can alter it. `None` on every other
    /// submit.
    pub frame: Option<Frame>,
}

impl Scheduler {
    // —— Projection helpers ———————————————————————————————————————————

    /// Write the per-turn projection onto `parent_id`.
    ///
    /// Sets `parent_id`'s contents to the system-prompt baseline
    /// (`system_block_count` blocks), followed by each projected
    /// section's sealed chunks (in `projected_sections` order),
    /// followed by each projected turn's sealed chunks (in
    /// `projected_turns` order).  Every unit's
    /// `Arc<Vec<SealedSequence>>` is fetched from the substrate and
    /// run through the per-sequence upload cache (CPU→GPU
    /// materialisation, deduped against currently-cached chunks);
    /// the resulting GPU sealed chunks are appended to the parent's
    /// tail in one batch.  The substrate's `block_range` for each
    /// injected section / turn is updated to reflect its new
    /// position so `reproject_view`'s mid-decode lookups resolve
    /// correctly.
    ///
    /// Implementation truncates the parent back to
    /// `system_block_count` and appends; the post-condition is the
    /// state above regardless of what was on the parent before.
    /// Inject the projected sections + turns onto `parent_id`'s
    /// slot as windowed sealed chunks.
    ///
    /// Each `SealedChunk` is a `(gid, offset, token_count)` window
    /// into a physical KV chunk.  Sharing a partial chunk with the
    /// source (the conversation that originally wrote it) is safe:
    /// the substrate's record asserts only the bytes the window
    /// covers; subsequent writers extending the chunk land beyond
    /// the recorded window in the R16/F16 storage and don't disturb
    /// the bytes the record references.  Arc-clone keeps the
    /// chunk's memory alive across freed slots; substrate metadata
    /// is atomic; projection sees a consistent snapshot.
    ///
    /// Answers whether the slot now holds exactly `segments`
    /// ([`projection_assembler::apply_segments`]).
    pub(crate) fn apply_projection(
        &mut self,
        parent_id: SequenceId,
        system_block_count: BlockCount,
        segments: &[ProjectionSegment],
    ) -> Result<bool, ConversationError> {
        let _ = system_block_count;

        let conversation = self
            .slot_conversations
            .get(&parent_id)
            .cloned()
            .ok_or_else(|| {
                ConversationError::Channel(format!(
                    "apply_projection: no conversation registered for slot {parent_id}"
                ))
            })?;
        let slot_target = self.slot_targets.get(&parent_id).copied();
        // During a submission drain, defer this slot's gap-fill into the one
        // batched forward at drain end (disjoint field borrow from the ctx below).
        // If this slot ALREADY has a deferred plan, that plan is now STALE: the
        // `apply_segments_build` below snapshots + truncates the slot to zero and
        // rebuilds it, so the old plan's reserved gaps — addressed by slot-relative
        // block index — no longer exist. Firing it at drain end would scatter old
        // glue K/V into whatever this rebuild placed at those indices. Drop the
        // superseded plan and defer only this latest projection, so exactly one
        // (current) plan fires against the slot as this call built it.
        // Read before any of `self`'s fields are borrowed out of it below.
        let pass_budget = self.prefill_pass_budget();
        let mut superseded = false;
        let defer = if self.batch_drain_gap_fills {
            let queued = self.deferred_glue_fires.len();
            self.deferred_glue_fires
                .retain(|p| p.parent_id != parent_id);
            superseded = self.deferred_glue_fires.len() != queued;
            Some(&mut self.deferred_glue_fires)
        } else {
            None
        };
        let state = self.slot_projection_state.entry(parent_id).or_default();
        // The dropped plan's glue was never filled, but the last assembly
        // recorded its pieces as placed: a rebuild that kept them would keep
        // zero chunks. Nothing it placed may be kept.
        if superseded {
            state.placed_pieces.clear();
        }
        // Record the sealed working set this projection attends over, so relief
        // eviction can protect it (see `evict_cold_tail`); the one it replaces
        // is held for one projection.
        state.set_working_set(projection_assembler::working_set_from_segments(segments));

        projection_assembler::apply_segments(
            state,
            projection_assembler::ApplyContext {
                session: &mut self.session,
                model: &mut self.model,
                conversation: &conversation,
                slot_target,
                parent_id,
                chunk_size: self.chunk_size,
                max_prefill_pass_tokens: pass_budget,
                tokenizer: &self.tokenizer,
                slot_tokens: &mut self.slot_tokens,
                boundary_markers: &self.boundary_markers,
                section_positional: &self.section_positional,
            },
            segments,
            defer,
        )
    }

    /// Build phase of [`Self::apply_projection`] for the cross-conversation wave:
    /// inject the sealed prefix + collect the glue descriptor, but do NOT fire
    /// the gap-fill forward. The caller fires (batched across slots via
    /// [`projection_assembler::fire_gap_fill_batch`]) then calls
    /// [`Self::apply_projection_finish`].
    pub(super) fn apply_projection_build(
        &mut self,
        parent_id: SequenceId,
        segments: &[ProjectionSegment],
    ) -> Result<projection_assembler::GapFillPlan, ConversationError> {
        let conversation = self
            .slot_conversations
            .get(&parent_id)
            .cloned()
            .ok_or_else(|| {
                ConversationError::Channel(format!(
                    "apply_projection_build: no conversation registered for slot {parent_id}"
                ))
            })?;
        let slot_target = self.slot_targets.get(&parent_id).copied();
        // Record the sealed working set this projection attends over, so relief
        // eviction can protect it (see `evict_cold_tail`). Same as the single-slot
        // `apply_projection`; the wave path threads build/finish separately, so we
        // stamp it here where the segments are in hand.
        let pass_budget = self.prefill_pass_budget();
        let state = self.slot_projection_state.entry(parent_id).or_default();
        state.set_working_set(projection_assembler::working_set_from_segments(segments));
        let mut ctx = projection_assembler::ApplyContext {
            session: &mut self.session,
            model: &mut self.model,
            conversation: &conversation,
            slot_target,
            parent_id,
            chunk_size: self.chunk_size,
            max_prefill_pass_tokens: pass_budget,
            tokenizer: &self.tokenizer,
            slot_tokens: &mut self.slot_tokens,
            boundary_markers: &self.boundary_markers,
            section_positional: &self.section_positional,
        };
        projection_assembler::apply_segments_build(state, &mut ctx, segments)
    }

    /// Finish phase of [`Self::apply_projection`] for the wave: prefill the
    /// deferred user message against the now-committed `[sealed | glue]` and
    /// re-attach the writer tail. Call after the batched fire.
    pub(super) fn apply_projection_finish(
        &mut self,
        parent_id: SequenceId,
        plan: projection_assembler::GapFillPlan,
    ) -> Result<(), ConversationError> {
        let conversation = self
            .slot_conversations
            .get(&parent_id)
            .cloned()
            .ok_or_else(|| {
                ConversationError::Channel(format!(
                    "apply_projection_finish: no conversation registered for slot {parent_id}"
                ))
            })?;
        let slot_target = self.slot_targets.get(&parent_id).copied();
        let pass_budget = self.prefill_pass_budget();
        let state = self.slot_projection_state.entry(parent_id).or_default();
        let mut ctx = projection_assembler::ApplyContext {
            session: &mut self.session,
            model: &mut self.model,
            conversation: &conversation,
            slot_target,
            parent_id,
            chunk_size: self.chunk_size,
            max_prefill_pass_tokens: pass_budget,
            tokenizer: &self.tokenizer,
            slot_tokens: &mut self.slot_tokens,
            boundary_markers: &self.boundary_markers,
            section_positional: &self.section_positional,
        };
        // The wave path fires the batched gap-fill BEFORE this finish, so the
        // fresh islands' K/V is in the gaps and capturable.
        projection_assembler::apply_segments_finish(state, &mut ctx, plan, true)
    }

    /// Carve a view of `parent_id` and fork the parent's per-sequence model
    /// state into it — [`Self::carve_view`] then [`Self::fork_view_state`].
    /// A fork that fails leaves nothing behind: the view is freed.
    pub(super) fn create_view(
        &mut self,
        parent_id: SequenceId,
        visible_block_ranges: &[BlockRange],
    ) -> Result<(SequenceId, BlockCount), ConversationError> {
        let (view_id, borrowed) = self.carve_view(parent_id, visible_block_ranges)?;
        if let Err(e) = self.fork_view_state(parent_id, view_id) {
            if let Err(free) = self.session.free_sequence(view_id.0) {
                return Err(ConversationError::Model(candle::Error::Msg(format!(
                    "{e} — and the view {} could not be freed: {free}",
                    view_id.0
                ))));
            }
            self.sampling_states.remove(&view_id);
            return Err(e);
        }
        Ok((view_id, borrowed))
    }

    /// Carve a view sequence borrowing the requested ranges from
    /// `parent_id` (or every block if `visible_block_ranges` is
    /// empty), without the parent's per-sequence model state — that is
    /// [`Self::fork_view_state`]'s. Returns `(view_id,
    /// original_borrowed_block_count)`.
    ///
    /// Called from the [`SchedulerRequest::SubmitTurn`] handler
    /// after [`apply_projection`] has populated the
    /// parent with the projected turns' sealed chunks.
    pub(super) fn carve_view(
        &mut self,
        parent_id: SequenceId,
        visible_block_ranges: &[BlockRange],
    ) -> Result<(SequenceId, BlockCount), ConversationError> {
        // Empty range list = sentinel "use all blocks of the parent".
        // Use `sequence_block_count` rather than `offset / CHUNK_SIZE`
        // — chunks with `usage < CHUNK_SIZE` from back-to-back section
        // injection are not visible to the divided value.  See
        // `BatchedInferenceSession::sequence_block_count`.
        let raw_ranges: Vec<(usize, usize)> = if visible_block_ranges.is_empty() {
            let total_blocks = self.session.sequence_block_count(parent_id.0).unwrap_or(0);
            // A zero-block parent yields a zero-LENGTH range, not an empty range
            // LIST. They borrow the same nothing, but the session rejects an empty
            // list outright: there, an empty list is a caller that never expanded
            // this sentinel, and a view built from one decodes at its parent's
            // position over empty K/V — fluently, and about nothing.
            vec![(0, total_blocks)]
        } else {
            visible_block_ranges.iter().map(|r| r.to_raw()).collect()
        };

        // The turn's own tokens start where the parent's prefix ends, so the
        // index is closed there — on the PARENT, and before the carve.
        //
        // Both halves of that are load-bearing. The view's fork
        // (`fork_view_state`) copies the parent's index cache wholesale, live tail included, so a tail left
        // open here arrives in the view as rows the turn's own tokens then
        // extend. And closing it on the parent is what makes the cut idempotent:
        // the parent keeps the page, so a slot that accumulates across turns
        // (the `skip_projection` ingest path, whose parent is never rebuilt)
        // does not re-close a growing prefix once per turn.
        self.begin_unit(parent_id, "turn");

        let vs = self
            .session
            .create_view_sequence(parent_id.0, &raw_ranges)
            .map_err(ConversationError::Model)?;
        let view_id = SequenceId(vs.view_idx);
        let borrowed = BlockCount(vs.borrowed_block_count);
        self.seal_under_conversation_of(view_id, parent_id);

        // Seed sampling state for the view (clone parent's state so
        // the DRY window survives the carve).
        if let Some(parent_state) = self.sampling_states.get(&parent_id).cloned() {
            self.sampling_states.insert(view_id, parent_state);
        } else {
            self.sampling_states.insert(
                view_id,
                SequenceSamplingState::new(
                    self.sampler.vocab_size(),
                    self.sampler.max_recent_len(),
                ),
            );
        }

        Ok((view_id, borrowed))
    }

    /// Budget-aware pre-elevate eviction (replaces the unconditional
    /// `evict_from_hot`). Frees only enough of **this** conversation's
    /// least-recently-promoted hot KV to fit the incoming cold-load within the
    /// *accurate* VRAM budget — so on a big GPU the working set stays resident
    /// instead of being dropped (and reloaded from cold) every reproject.
    ///
    /// Order: estimate the incoming cold-load size → if it already fits the free
    /// budget, evict nothing → otherwise reclaim partial-arena free space
    /// (compaction, cheap unless fragmented past the 0.20 threshold) → only then
    /// evict the oldest non-selected turns, and only as many bytes as still
    /// needed. Per-conversation scoped: `evict_hot_to_free` walks only
    /// `conversation`'s own residence, never a parallel conversation's hot KV.
    pub(super) fn evict_to_fit_incoming(
        &mut self,
        conversation: &Conversation,
        sections: &[SectionId],
        turns: &[TurnKey],
    ) -> crate::substrate::EvictionReport {
        // Incoming hot VRAM footprint of the working set that `elevate_to_hot`
        // is about to lift. BOTH warm→hot and cold→hot allocate fresh hot
        // arenas, so both must be budgeted for. Counting only cold left a
        // warm-resident selection — the common case once hot KV has been evicted
        // to the RAM tier — with zero reserved headroom, so its warm→hot batch
        // migrate hit the VRAM gate and failed; that drops the *whole* warm batch
        // (the migrate is all-or-nothing per layer), and `apply_projection` then
        // discards every selected turn as "no hot sealed K/V".
        let incoming_bytes: u64 = {
            let view = conversation.read();
            let plan = view.snapshot_promotion_state(sections, turns);
            let cold: u64 = plan
                .cold_to_hot
                .iter()
                .flat_map(|c| c.cold.iter())
                .flat_map(|s| s.chunks.iter())
                .map(|c| c.record_len)
                .sum();
            let warm: u64 = plan
                .warm_to_hot
                .iter()
                .map(|w| sealed_total_bytes(&w.warm))
                .sum();
            cold + warm
        };
        if incoming_bytes == 0 {
            return crate::substrate::EvictionReport { count: 0, bytes: 0 };
        }
        let device = self.session.device().clone();
        let avail = match candle_nn::kv_cache::vram_budget_available(&device) {
            // Budget unknown (non-CUDA / query failure) — don't evict on a guess.
            None => return crate::substrate::EvictionReport { count: 0, bytes: 0 },
            Some(a) => a as u64,
        };
        if incoming_bytes <= avail {
            // Ample VRAM — keep the whole working set hot.
            return crate::substrate::EvictionReport { count: 0, bytes: 0 };
        }
        // Tight: sweep emptied arenas back to the free list first, then
        // re-measure. This used to be a compaction — relocate chunks GPU→GPU to
        // consolidate partial arenas. Under the reservation there is nothing to
        // consolidate for: an arena holds one region whether it is one chunk
        // full or entirely full, so the only thing that returns a region is the
        // arena's last chunk going, and the sweep is what notices.
        let _ = self.session.release_empty_arenas();
        let avail = candle_nn::kv_cache::vram_budget_available(&device)
            .map(|a| a as u64)
            .unwrap_or(avail);
        let needed = incoming_bytes.saturating_sub(avail);
        if needed == 0 {
            return crate::substrate::EvictionReport { count: 0, bytes: 0 };
        }
        conversation
            .read()
            .evict_hot_to_free(sections, turns, needed)
    }

    /// Promote the projected working set (`sections` + `turns`) into hot VRAM so
    /// [`Self::apply_projection`] can inject it: budget-aware evict of the
    /// non-incoming hot residences, then a batched select-promote (warm/cold →
    /// hot). Every path that hands sealed sections/turns to `apply_projection`
    /// MUST call this first — `apply_projection` only reads the *hot* residence
    /// and silently drops any selected unit that isn't hot, so a missing elevate
    /// step means the turn decodes/summarises without that content.
    ///
    /// A promote miss is not fatal (the decode proceeds without the dropped
    /// units), but it IS surfaced: when any selected item fails to reach hot the
    /// per-bucket report is logged at WARN with `whence` identifying the caller,
    /// so the drop has a visible cause rather than only the downstream
    /// `apply_projection: ... dropping it` symptom.
    /// Publish the union of every live slot's projection working set — plus the
    /// `extra_sections` / `extra_turns` an in-flight elevate is about to lift —
    /// as the substrate's eviction keep-set (`Substrate::set_working_set_pins`).
    /// A pinned residence is protected from every automatic hot-drop for the
    /// duration of the wave, closing the free-under-read the multi-timeline
    /// belief scan opened (it can now select completed-file turns whose KV the
    /// persistence thread would otherwise evict mid-forward). Recomputed
    /// wholesale each call, so residences that have left every working set
    /// lose their pin and become evictable again. Pins are counts on the
    /// residences themselves: publishing waits on nothing and publishes no
    /// substrate version.
    pub(super) fn publish_working_set_pins(
        &mut self,
        conversation: &Conversation,
        extra_sections: &[SectionId],
        extra_turns: &[TurnKey],
    ) {
        let (mut keep_sections, mut keep_turns) = self.live_working_sets();
        // What each live slot's last projection dropped is held through that
        // slot's next one (`SlotState::previous_working_set`).
        for (&slot, st) in &self.slot_projection_state {
            if self.slot_idle(slot) {
                continue;
            }
            keep_sections.extend(st.previous_working_set.sections.iter().copied());
            keep_turns.extend(st.previous_working_set.turns.iter().copied());
        }
        keep_sections.extend(extra_sections.iter().copied());
        keep_turns.extend(extra_turns.iter().copied());
        // The publish stamps the active set with the substrate's wave clock —
        // see `Substrate::set_working_set_pins`.
        conversation
            .read()
            .set_working_set_pins(&keep_turns, &keep_sections);
    }

    /// Republish `conversation`'s substrate's keep-set — the live working sets
    /// alone: what a slot's retirement does.
    pub(super) fn republish_pins(&mut self, conversation: &Conversation) {
        self.publish_working_set_pins(conversation, &[], &[]);
    }

    /// Drop `slot`'s projection working set and republish the keep-set without
    /// it, on that slot's own substrate.
    ///
    /// **A pin outliving its slot holds VRAM it no longer needs.** The keep-set
    /// is defined as the union of every *live* slot's working set, but
    /// [`Self::publish_working_set_pins`] only ever runs from an elevate — so a
    /// slot that goes away between elevates leaves its pins standing, and
    /// nothing recomputes the union until some other slot happens to elevate.
    /// A stale pin keeps its residences off the eviction path; durability is
    /// not the pin's to defer (`Substrate::cold_pending`).
    ///
    /// Republished on the freed slot's OWN substrate: the scheduler hosts
    /// conversations on many substrates at once and the pin set lives per
    /// substrate, so the handle has to come from the slot being retired.
    ///
    /// This is the only place [`Self::slot_projection_state`] is removed from,
    /// which is what makes "a removal is always followed by a republish" hold
    /// by construction rather than by every caller remembering. Keep it that
    /// way: a bare `remove` elsewhere re-opens the wedge silently.
    pub(super) fn retire_slot_projection_state(
        &mut self,
        slot: SequenceId,
        conversation: Option<Conversation>,
    ) {
        // The pool goes with the slot's state; the model's hold on its pages
        // goes when the sequence is released.
        if self.slot_projection_state.remove(&slot).is_none() {
            return;
        }
        if let Some(conversation) = conversation {
            self.republish_pins(&conversation);
        }
    }

    pub(super) fn elevate_projection_working_set(
        &mut self,
        conversation: &Conversation,
        sections: &[SectionId],
        turns: &[TurnKey],
        whence: &str,
    ) {
        // A `section_<n>` fallback name (see `Conversation::section_debug_name`) —
        // an unnamed content/probe section, not a schema-declared unit.
        fn is_generic_section_name(name: &str) -> bool {
            name.strip_prefix("section_")
                .is_some_and(|rest| !rest.is_empty() && rest.bytes().all(|b| b.is_ascii_digit()))
                || name.starts_with("section#")
        }
        // Publish the eviction keep-set FIRST, before any KV is lifted or
        // attended: the union of every live slot's working set plus the units
        // this elevate is about to lift. Runs on every elevate (submit +
        // reproject), so the persistence thread's `evict_when_cold` hot-drops
        // never yank a residence the current wave is using — the fix for the
        // multi-timeline belief scan reaching completed-file (evicted) turns.
        // See `Substrate::working_set_pins`.
        let t_pins = Instant::now();
        let pins_span = profile::span("elevate:publish_pins");
        self.publish_working_set_pins(conversation, sections, turns);
        pins_span.end();
        let pins_us = t_pins.elapsed().as_micros() as u64;
        if sections.is_empty() && turns.is_empty() {
            return;
        }
        let t_tier = Instant::now();
        let tier_span = profile::span("elevate:tier_check");
        // PER-ITEM SKIP: only lift the units NOT already hot. A unit already hot
        // needs no work — and the machinery below (`snapshot_promotion_state`, run
        // twice, + a `backings` clone) costs ~100 ms/turn for a fixed system prompt
        // that never leaves hot. Cheap O(1) tier check per unit builds the actual
        // lift-list. `evict_to_fit_incoming` still gets the FULL working set as its
        // keep-list (it must protect the already-hot units from eviction), but only
        // budgets for the non-hot bytes. The projection tracker records only
        // ACTUAL lifts, so its board is a waste monitor — empty is good.
        let (secs_lift, turns_lift) = {
            let read = conversation.read();
            let mut sl: Vec<SectionId> = Vec::new();
            let mut tl: Vec<TurnKey> = Vec::new();
            for &sid in sections {
                if read.section_tier_state(sid).is_some_and(|t| t.hot) {
                    continue;
                }
                // A section with no schema name (`section_<n>`) is still a unit
                // of its own, so it is named by its id rather than pooled.
                let name_of = || match read.section_debug_name(sid) {
                    Some(n) if !is_generic_section_name(&n) => n,
                    _ => format!("section {}", sid.raw()),
                };
                let name = match read.section_stream(sid) {
                    Some(stream) => self
                        .section_name_cache
                        .entry(stream)
                        .or_insert_with(name_of)
                        .clone(),
                    None => name_of(),
                };
                phase_ring::record_projected(ProjectedKind::Section, &name, 1);
                sl.push(sid);
            }
            for &t in turns {
                if !read
                    .turn_tier_state(t.timeline, t.index)
                    .is_some_and(|ts| ts.hot)
                {
                    tl.push(t);
                }
            }
            // Turns by their conversation, named as the sidebar names it — for
            // an ingest layer that is the file or directory the conversation
            // reads — and counted per conversation, not per turn.
            let mut by_conversation: Vec<(TimelineId, u64)> = Vec::new();
            for t in &tl {
                match by_conversation
                    .iter_mut()
                    .find(|(tid, _)| *tid == t.timeline)
                {
                    Some((_, n)) => *n += 1,
                    None => by_conversation.push((t.timeline, 1)),
                }
            }
            for (timeline, n) in by_conversation {
                let name = match read.label_of(timeline) {
                    Some(label) => label.to_string(),
                    None => format!("conversation {}", timeline.raw()),
                };
                phase_ring::record_projected(ProjectedKind::Turns, &name, n);
            }
            (sl, tl)
        };
        tier_span.end();
        let tier_us = t_tier.elapsed().as_micros() as u64;
        // Where an elevate's time went: the keep-set publish every call pays, the
        // tier walk, and — only when something is not hot — the eviction and the
        // lift itself, with how much was lifted.
        // A lift's report says where what it lifted came from: warm (host RAM)
        // or cold (the log), and how many bytes of each.
        let split = |evict_us: u64, lift_us: u64, from: (usize, usize, u64, u64)| {
            note("elevate split", || {
                format!(
                    "whence={whence} pins_us={pins_us} tier_us={tier_us} evict_us={evict_us} \
                     lift_us={lift_us} sections={} turns={} lifted_sections={} lifted_turns={} \
                     warm_to_hot={} cold_to_hot={} warm_mib={} cold_mib={}",
                    sections.len(),
                    turns.len(),
                    secs_lift.len(),
                    turns_lift.len(),
                    from.0,
                    from.1,
                    from.2 >> 20,
                    from.3 >> 20,
                )
            });
        };
        // Everything already hot → nothing to evict or lift.
        if secs_lift.is_empty() && turns_lift.is_empty() {
            split(0, 0, (0, 0, 0, 0));
            return;
        }
        let backings = self.session.backings().to_vec();
        let device = self.session.device().clone();
        // Keep-list = FULL working set (protect the already-hot units too).
        let t_evict = Instant::now();
        let evict_span = profile::span("elevate:evict_to_fit");
        let evicted = self.evict_to_fit_incoming(conversation, sections, turns);
        evict_span.end();
        let evict_us = t_evict.elapsed().as_micros() as u64;
        if evicted.count > 0 {
            tracing::trace!(
                target: "candle_conversation::persistence::tier",
                whence, count = evicted.count, bytes = evicted.bytes,
                "select-evict complete"
            );
        }
        // Lift only the non-hot subset; already-hot units are left untouched.
        let t_lift = Instant::now();
        let lift_span = profile::span("elevate:lift");
        let lifted = warm_lift_ring(&mut self.warm_lift_ring).and_then(|ring| {
            elevate_to_hot(
                conversation,
                &backings,
                &device,
                ring,
                &mut self.cold_load_stager,
                &secs_lift,
                &turns_lift,
            )
        });
        lift_span.end();
        let from = lifted.as_ref().map_or((0, 0, 0, 0), |r| {
            (
                r.warm_to_hot,
                r.cold_to_hot,
                r.bytes_warm_to_hot,
                r.bytes_cold_to_hot,
            )
        });
        split(evict_us, t_lift.elapsed().as_micros() as u64, from);
        match lifted {
            Ok(report) => {
                if report.missing > 0 || report.failed > 0 {
                    tracing::warn!(
                        target: "candle_conversation::persistence::tier",
                        whence,
                        already_hot = report.already_hot,
                        warm_to_hot = report.warm_to_hot,
                        cold_to_hot = report.cold_to_hot,
                        missing = report.missing,
                        failed = report.failed,
                        n_sections = sections.len(),
                        n_turns = turns.len(),
                        "select-promote: some selected units could not be lifted to hot; \
                         apply_projection will drop them from the assembled context"
                    );
                } else {
                    tracing::trace!(
                        target: "candle_conversation::persistence::tier",
                        whence,
                        already_hot = report.already_hot,
                        warm_to_hot = report.warm_to_hot,
                        cold_to_hot = report.cold_to_hot,
                        bytes_warm_to_hot = report.bytes_warm_to_hot,
                        bytes_cold_to_hot = report.bytes_cold_to_hot,
                        "select-promote complete"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    target: "candle_conversation::persistence::tier",
                    whence,
                    "select-promote failed: {e} — apply_projection will drop non-hot selected units"
                );
            }
        }
    }
}
