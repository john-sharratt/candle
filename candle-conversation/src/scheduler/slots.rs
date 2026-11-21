//! Slot lifecycle: creating, freeing and detaching the GPU slots a conversation
//! runs on, and withdrawing their queued work when they go.
//!
//! A freed slot id is recycled at once, so every in-flight prefill, decode and
//! queued section keyed by it is purged or answered on the free path. Otherwise
//! the work would be attributed to the next occupant. A parked fork whose
//! requester has gone takes no slot. Detaching a substrate frees its slots and only
//! its slots, and drops its persistence trigger, while an attached substrate with
//! no live slot is still walked by compaction and eviction so that its resident
//! K/V is never moved unwitnessed. A new slot starts from its timeline's recurrent
//! state and carried belief, and a resumed one rebuilds its belief from the
//! selections it recovered, folded in turn order.

use super::*;

impl Scheduler {
    /// Withdraw the ingest of `section` on `conversation`'s substrate still
    /// waiting for admission: the section was retired — removed by its owner,
    /// or replaced under its name — so its prefill would seal what nobody holds.
    /// Answered with an error, so its seal thread stops waiting and frees its
    /// scratch slot.
    ///
    /// Left in the queue, every resubmission of a section that could not be
    /// admitted in time added another prefill behind it: on WhiteWorld
    /// (2026-11-20) the journal pre-warmer gave its run up every two minutes
    /// and 25 abandoned copies of one 3-token section sealed together, the
    /// oldest queued for over eighteen minutes.
    pub(super) fn withdraw_queued_section(
        &mut self,
        conversation: &Conversation,
        section: SectionId,
    ) {
        let conversations = &self.slot_conversations;
        let (gone, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.section_queue)
            .into_iter()
            .partition(|s| {
                s.section_id == section
                    && conversations
                        .get(&s.sequence_id)
                        .is_some_and(|c| c.same_substrate(conversation))
            });
        self.section_queue = kept.into();
        for s in gone {
            let _ = s.response_tx.send(Err(ConversationError::Channel(format!(
                "section ingest {} withdrawn: the section was retired before it was admitted",
                s.debug_name
            ))));
        }
    }

    /// Purge every in-flight scheduling entry keyed by a slot id that is being
    /// freed. The kernel recycles a freed slot id **immediately** (the code_read
    /// scope workers churn fork slots), so any prefill / decode entry left under
    /// the old id would be silently re-attributed to the NEXT occupant.
    ///
    /// The concrete failure this prevents: `form_wave_group` emits one
    /// [`WaveMember::Prefill`] per [`Self::active_prefills`] entry, keyed by its
    /// sequence id. A stale entry from an abandoned fork PLUS the reused slot's
    /// fresh prefill then yield the SAME `seq_id` twice in one wave, tripping the
    /// batched-inference "sequence appears in more than one group" guard, which
    /// aborts the whole wave. (Observed as an error cascade when a graceful
    /// shutdown cancels an in-flight scope ingest mid-wave.)
    ///
    /// Called from every TERMINAL free path (the [`SchedulerRequest::FreeSequence`]
    /// handler and [`Self::free_summary_slot`]). The mid-decode reproject swap
    /// frees-and-rebinds its own view state inline and does not route here.
    pub(super) fn purge_freed_slot_scheduling_state(&mut self, id: SequenceId) {
        self.sealed_snapshots.remove(&id);
        self.parked_at.remove(&id);
        self.prefill_queue.retain(|w| w.sequence_id != id);
        // A section still waiting for admission would otherwise be prepared on
        // whatever reuses this slot's id. Answered, so its caller is not left
        // waiting on a seal that will never come.
        let (gone, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.section_queue)
            .into_iter()
            .partition(|s| s.sequence_id == id);
        self.section_queue = kept.into();
        for s in gone {
            let _ = s.response_tx.send(Err(ConversationError::Channel(format!(
                "section ingest on slot {id} dropped: the slot was freed before it was admitted"
            ))));
        }
        self.active_prefills.retain(|p| p.work.sequence_id != id);
        self.active_decodes.remove(&id);
        // A terminal free is a completion as far as admission is concerned: the
        // slot's ground is back whether it finished or was abandoned.
        self.settled_since_admit = true;
        // The held wave cohort's residual is indexed by member POSITION, so a
        // member vanishing mid-cohort would desync every later member's slice
        // against the residual. Its rows are taken out instead, here — before
        // the id can be reissued to a slot whose prefill would otherwise be
        // matched to the old member — and the survivors carry on from the
        // layer they reached (`drop_wave_members`). Fires only under teardown
        // churn; a normally-completing turn frees its slot after its prefill has
        // already drained out of the cohort.
        let keep: Vec<bool> = self
            .wave_prefill_members
            .iter()
            .map(|m| m.seq_id() != id.0)
            .collect();
        if let Err(e) = self.drop_paused_member(id.0) {
            tracing::error!(
                slot = %id,
                error = %e,
                "could not take a freed member out of the paused creep; failing its \
                 members rather than replaying layers they already ran"
            );
            self.fail_paused_creep(&e.to_string());
        }
        if keep.contains(&false) {
            if let Err(e) = self.drop_wave_members(&keep) {
                tracing::error!(
                    slot = %id,
                    error = %e,
                    "could not take a freed member out of the creep in flight; failing \
                     the creep's members rather than replaying layers they already ran"
                );
                self.fail_wave_members(&e.to_string());
            }
        }
    }

    /// Withdraw the prefill work whose turn nobody holds any more: queued,
    /// admitted, and members of the creep in flight or paused.
    ///
    /// A decode already stops at its next token when its handle is dropped; a
    /// prefill had no such check, so a turn whose caller let go swept on to the
    /// end of its round. Its caller ends it by dropping the handle — zend does
    /// when a conversation is deleted under a running reply, and never merely
    /// because a client disconnected, whose turn runs on for a reconnect — and
    /// the slot is the caller's to free. Measured on WhiteWorld (a5e6cc3ec,
    /// 2026-10-19): twelve deleted conversations' tool rounds swept on for
    /// twelve minutes after the delete, and every new turn, chat and reader
    /// alike, waited behind them for its first token.
    ///
    /// Members leave as a freed slot's do (`drop_wave_members`,
    /// `drop_paused_member`): the survivors carry on from the layer they
    /// reached.
    pub(in super::super) fn withdraw_abandoned_prefills(&mut self) {
        self.prefill_queue.retain(|w| !w.event_tx.is_disconnected());
        let gone: Vec<SequenceId> = self
            .active_prefills
            .iter()
            .filter(|p| p.work.event_tx.is_disconnected())
            .map(|p| p.work.sequence_id)
            .collect();
        if gone.is_empty() {
            return;
        }
        self.active_prefills
            .retain(|p| !p.work.event_tx.is_disconnected());
        self.settled_since_admit = true;
        for &id in &gone {
            tracing::info!(
                target: "candle_conversation::scheduler::prefill",
                slot = %id,
                "prefill withdrawn: its turn's handle was dropped"
            );
            if let Err(e) = self.drop_paused_member(id.0) {
                tracing::error!(
                    slot = %id,
                    error = %e,
                    "could not take a withdrawn member out of the paused creep; failing its \
                     members rather than replaying layers they already ran"
                );
                self.fail_paused_creep(&e.to_string());
            }
        }
        let keep: Vec<bool> = self
            .wave_prefill_members
            .iter()
            .map(|m| !gone.iter().any(|id| id.0 == m.seq_id()))
            .collect();
        if keep.contains(&false) {
            if let Err(e) = self.drop_wave_members(&keep) {
                tracing::error!(
                    error = %e,
                    "could not take withdrawn members out of the creep in flight; failing \
                     the creep's members rather than replaying layers they already ran"
                );
                self.fail_wave_members(&e.to_string());
            }
        }
    }

    /// Drop a freed slot's timeline from [`Self::ingest_timelines`] once no
    /// other live slot still targets it. Without this, the set grows by one
    /// entry per ingested file for the daemon's lifetime (a slow leak, and a
    /// growing scan for [`demote_cold_ingest_if_pressured`]). Called from every
    /// slot-teardown path; a `None` target (raw/non-projecting slot) is a no-op.
    pub(super) fn prune_ingest_timeline(&mut self, freed_target: Option<ProjectionTarget>) {
        let Some(tgt) = freed_target else {
            return;
        };
        if !self.ingest_timelines.contains(&tgt.timeline) {
            return;
        }
        if !self
            .slot_targets
            .values()
            .any(|t| t.timeline == tgt.timeline)
        {
            self.ingest_timelines.remove(&tgt.timeline);
        }
    }

    /// Free every slot bound to `conversation`'s substrate, forget the
    /// scheduler-wide state held for its timelines, and drop attachment `id`'s
    /// persistence trigger — [`SchedulerRequest::DetachSubstrate`].
    ///
    /// Each slot goes through the same teardown a `FreeSequence` runs, which
    /// purges its in-flight prefill, decode and cohort state; a turn view freed
    /// with its parent is skipped when its own turn comes. Each freed id is
    /// then reserved until its handle frees it, and the substrate's handles'
    /// requests are dropped from here on — see [`detached`].
    pub(in super::super) fn detach_substrate(&mut self, id: u64, conversation: &Conversation) {
        let slots: Vec<SequenceId> = self
            .slot_conversations
            .iter()
            .filter(|(_, c)| c.same_substrate(conversation))
            .map(|(s, _)| *s)
            .collect();
        let mut freed = Vec::new();
        for sequence_id in slots {
            if self.slot_conversations.contains_key(&sequence_id) {
                self.handle_request(SchedulerRequest::FreeSequence { sequence_id });
                match self.session.reserve_sequence(sequence_id.0) {
                    Ok(()) => freed.push(sequence_id),
                    Err(e) => tracing::warn!(
                        "detach: slot {} not reserved for its handle: {}",
                        sequence_id,
                        e
                    ),
                }
            }
        }
        let freed_count = freed.len();
        self.detached.record(conversation, freed);
        self.forget_norm_warm(conversation);
        self.section_positional.forget(conversation);
        self.pending_section_quantize
            .retain(|p| !p.substrate.is(conversation));
        let timelines: HashSet<TimelineId> = conversation.timeline_ids().into_iter().collect();
        self.timeline_projections
            .retain(|t, _| !timelines.contains(t));
        self.ingest_timelines.retain(|t| !timelines.contains(t));
        self.persist_triggers.detach(id);
        self.attached_substrates
            .retain(|c| !c.same_substrate(conversation));
        tracing::info!(
            attachment = id,
            freed_slots = freed_count,
            attached = self.persist_triggers.len(),
            "substrate detached"
        );
    }

    /// One handle per substrate that holds resident KV: every attached one, and
    /// any other a live slot is bound to.
    ///
    /// **Attached, not only slotted.** A substrate with no live slot still holds
    /// resident KV in its residences, and a compaction sweep that skips it moves the
    /// chunks they name without rewriting them — on a shared engine, every pass run
    /// while one tenant was between conversations (`unwitnessed` in the sweep's
    /// report). Visiting it costs one walk of its residences per pass, the cost it
    /// already pays on any pass where it has a slot open.
    ///
    /// Every slot of a workspace registers its own handle on the same substrate, so
    /// a walk over [`Self::slot_conversations`] visits that substrate once per slot.
    /// A pass over residences — a compaction sweep, a relief eviction — finds nothing
    /// left to do on a second visit, and pays a full walk of every residence to learn
    /// it.
    pub(in super::super) fn distinct_substrates(&self) -> Vec<Conversation> {
        let mut out: Vec<Conversation> = self.attached_substrates.clone();
        for conv in self.slot_conversations.values() {
            if !out.iter().any(|seen| seen.same_substrate(conv)) {
                out.push(conv.clone());
            }
        }
        out
    }

    // —— Sequence creation ——————————————————————————————————————————

    /// Allocate a fresh empty GPU slot bound to a `Conversation`
    /// (workspace) and an optional projection target.
    /// Registers the conversation handle in
    /// [`Self::slot_conversations`]; when `target` is `Some`, also
    /// registers it in [`Self::slot_targets`] so the [`SubmitTurn`]
    /// handler can resolve it without re-derivation on every submit.
    /// `None` is for raw paths (RULER eval, summarisation) that don't
    /// project from the substrate.
    ///
    /// The slot starts from the timeline's persisted recurrent state and
    /// carried belief ([`StateSeed::FromTimeline`]) — see [`StateSeed`] for
    /// the scratch-slot case, which binds a target without inheriting it.
    pub(super) fn create_sequence(
        &mut self,
        conversation: Conversation,
        target: Option<ProjectionTarget>,
    ) -> Result<SequenceId, ConversationError> {
        self.create_sequence_seeded(conversation, target, StateSeed::FromTimeline)
    }

    /// [`Self::create_sequence`] with an explicit choice of what the slot's
    /// recurrent state starts as.
    pub(super) fn create_sequence_seeded(
        &mut self,
        conversation: Conversation,
        target: Option<ProjectionTarget>,
        seed: StateSeed,
    ) -> Result<SequenceId, ConversationError> {
        let raw_id = self
            .session
            .create_sequence()
            .map_err(ConversationError::Model)?;
        let slot_id = SequenceId(raw_id);

        // Create persistent sampling state for this sequence.
        // This state survives across turns so that DRY penalty can
        // see a rolling window of recent tokens spanning turn boundaries.
        self.sampling_states.insert(
            slot_id,
            SequenceSamplingState::new(self.sampler.vocab_size(), self.sampler.max_recent_len()),
        );

        // Register the conversation handle and (optional) projection
        // target for this slot — see [`Self::slot_conversations`] and
        // [`Self::slot_targets`].
        // Restore the timeline's recurrent state, if it has one.
        //
        // **Before the first wave, and after the store could exist.**
        // `create_sequence` is the funnel every entry point routes through
        // (`NewSequence`, `NewEphemeralSequence`, `ResumeSequence`) and runs
        // before any `submit_turn`, so both halves of that ordering hold here
        // and nowhere else.
        //
        // Every rejection falls back to a zero state, which is a conversation
        // that resumes fluent and having forgotten everything the recurrent
        // layers held. That is the failure this whole path exists to remove, so
        // it must not be the *quiet* error path: each reason logs at WARN, and
        // the reasons are distinguishable.
        if let (Some(target), StateSeed::FromTimeline) = (target, seed) {
            let _ = self.restore_recurrent_state(slot_id, &conversation, target.timeline);
            self.restore_carried_belief(slot_id, &conversation, target.timeline);
        }

        // A conversation that records a band of its own decodes at it.
        if let Some(band) = target.and_then(|t| recorded_band(&conversation, t.timeline)) {
            self.slot_bands.insert(slot_id, band);
        }
        self.slot_conversations.insert(slot_id, conversation);
        if let Some(target) = target {
            self.slot_targets.insert(slot_id, target);
        }
        self.seal_under_conversation_of(slot_id, slot_id);

        tracing::trace!(
            kind = if target.is_some() {
                "targeted"
            } else {
                "scratch"
            },
            "allocated empty slot {}",
            slot_id,
        );
        Ok(slot_id)
    }

    /// Restore the conversation's carried selection belief onto a freshly
    /// created slot — the third piece of a timeline's durable state, beside its
    /// K/V and its recurrent memory.
    ///
    /// [`Self::carried_beliefs`] is what a submit-time projection seeds from,
    /// and its comment says "empty only for a genuinely fresh conversation". A
    /// RESUMED conversation is not a fresh conversation — but its slot is, so
    /// without this it selected its sections and tools from an empty prior
    /// while the conversation it continues had a prior evolved over every
    /// turn. Near a selection gate (a tools-catalog entry, a grounding
    /// variant) that flips a section in or out of the context, and the first
    /// resumed decode diverges from its uninterrupted twin — deterministic,
    /// marginal, and invisible to every recall probe. A1 of the behavioural
    /// catalogue is the only oracle that sees it.
    ///
    /// Reconstruction is the same fold the live path performs: the seal
    /// harvests `merge_from(from_selection(final composition))` per turn, so
    /// folding each recovered turn's LAST persisted projection event in turn
    /// order rebuilds the harvest exactly. (The per-turn decay is applied to a
    /// clone at submit, never written back, so the stored map is the undecayed
    /// merge — the same thing this rebuilds.)
    pub(super) fn restore_carried_belief(
        &mut self,
        slot_id: SequenceId,
        conversation: &Conversation,
        timeline: TimelineId,
    ) {
        let read = conversation.read();
        let mut belief = PriorBelief::default();
        let mut turns_folded = 0usize;
        for idx in read.turn_indices(timeline) {
            let Some(blob) = read.projection_events_blob(timeline, idx) else {
                continue;
            };
            // The turn's last event carries the selection in force at its
            // seal — the composition the live harvest folded into the carry.
            let Some(last) = decode_events(blob).pop() else {
                continue;
            };
            belief.merge_from(&PriorBelief::from_selection(&last.selection));
            turns_folded += 1;
        }
        if turns_folded > 0 {
            tracing::debug!(
                "restored carried belief for slot {slot_id} from {turns_folded} recovered \
                 turn(s) on timeline {timeline}"
            );
            self.carried_beliefs.insert(slot_id, belief);
        }
    }
}
