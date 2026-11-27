//! Forks a conversation onto a timeline, deciding whether the fork inherits the
//! parent's recurrent memory.
//!
//! Memory describes a token history, so the parent's live memory goes to a fork
//! only when that fork continues the parent's own timeline. A fork onto a fresh
//! or another conversation's timeline starts from that timeline's snapshot, or
//! from the prompt branch checkpoint. The two predicates are free functions so
//! they can be asserted directly, since a wrong choice still yields correct
//! key-value state and fluent answers. A recurrence cannot fill a gap in a
//! sequence, but it can absorb an append, so a splice catches memory up over the
//! adopted turns.

use super::prefix_state::run_state;
use super::*;
use crate::scheduler::exported_state::SharedState;

/// One code scope to ingest in parallel — the `(user, assistant)` pair a single
/// A client-side conversation handle.
///
/// Manages turn history and submits GPU work to the scheduler via channel.
/// One turn at a time per conversation (`turn_in_flight` guard).
///
/// # Usage
///
/// ```ignore
/// // Blocking:
/// let response = conv.send("hello")?;
///
/// // Streaming:
/// let handle = conv.submit_turn("hello")?;
/// for event in handle.stream() { /* ... */ }
/// conv.finish_turn(&response);
/// ```
/// Whether a fork's target timeline already carries the parent's turns.
///
/// Not a mode — a fact about the timeline, and the thing that decides whether
/// the child may take the parent's recurrent memory. Memory describes a token
/// history; handing it to a conversation that does not hold that history is the
/// same defect as the reverse, and just as quiet.
///
/// It is therefore **derived** in [`Sequence::fork_onto`] from the one
/// comparison that settles it — is the fork's timeline the parent's own? — and
/// never passed in. A caller cannot know it: `fork_resuming` targets a timeline
/// whose history belongs to whichever conversation sealed it, which is the
/// parent only in the re-attach case. Deciding this per constructor is how the
/// daemon's resume path came to hand every restored conversation the *base*
/// conversation's memory over the top of its own restored snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum InheritsHistory {
    Yes,
    No,
}

/// The one comparison that settles [`InheritsHistory`], as a free function so it
/// can be asserted without a scheduler, a model, or a workspace behind it.
pub(super) fn fork_inherits_history(parent: TimelineId, fork: TimelineId) -> InheritsHistory {
    if parent == fork {
        InheritsHistory::Yes
    } else {
        InheritsHistory::No
    }
}

/// Whether [`Sequence::fork_onto`] should seed the fork's recurrent state from
/// its system-prompt branch checkpoint (§4.6), as a free function for the same
/// reason [`fork_inherits_history`] is one: asserted directly, because the
/// end-to-end oracle costs a model load and nothing else can see the defect —
/// a conversation given the wrong memory holds the right K/V, answers from it
/// fluently, and passes every recall probe.
///
/// **Three conditions, all required:**
/// - `inherits == No` — an `Yes` fork continues the parent's own live history,
///   which `fork_recurrent` already carries across; a branch checkpoint is
///   only for a fork that gets nothing from that path.
/// - `carries_recurrent_state` — a plain transformer never computed a
///   checkpoint in the first place (`build_branch_checkpoint`'s own gate at
///   construction), so the lookup would only ever miss.
/// - `timeline_turn_count == 0` — the timeline has never sealed a turn. A
///   `No` fork can also mean resuming some OTHER conversation's timeline,
///   which `create_sequence_seeded` already restores from that timeline's own
///   recurrent snapshot; installing base's prompt-only checkpoint over that
///   would silently discard a real conversation's accumulated memory. A
///   timeline with turns already sealed is exactly the case that restore does
///   NOT leave untouched, so it must be excluded here.
pub(super) fn fork_wants_branch_checkpoint(
    inherits: InheritsHistory,
    carries_recurrent_state: bool,
    timeline_turn_count: u32,
) -> bool {
    matches!(inherits, InheritsHistory::No) && carries_recurrent_state && timeline_turn_count == 0
}

impl Sequence {
    /// Fork this conversation onto a fresh timeline for one parallel scope
    /// ingest. The returned [`Sequence`] shares the substrate (so a later
    /// `adopt_turn` can reference its sealed K/V) but has its own scheduler slot
    /// + timeline, so scopes ingest concurrently without ordering conflicts.
    pub fn fork_scope(&self) -> crate::Result<Sequence> {
        // Do NOT mark the fork append-only / evict_when_cold, and DO mark it a
        // splice source: `adopt_turn` reads the fork's turns from their HOT copy
        // and nothing else, and a splice source's hot copy is exempt from every
        // automatic hot-drop — relief, the idle demote, the migrate install —
        // until the fork's `tombstone_timeline` (below) frees it, the file
        // timeline's cloned chunk handles keeping the shared KV alive.
        //
        // DO mark the fork timeline transient: its sealed KV is spliced by REFERENCE
        // onto the file timeline (which writes its own durable cold copy) and then
        // tombstoned, so the fork's own KV must never be cold-persisted — a cold copy
        // would be stranded as an unreconstructable orphan when the fork's decl drops
        // at tombstone. See `Substrate::snapshot_pending_cold` / `no_cold_persist`.
        let fork_timeline = self
            .substrate
            .mint_timeline(self.target.layer, self.target.group);
        self.substrate.mark_timeline_transient(fork_timeline);
        self.substrate.mark_timeline_splice_source(fork_timeline);
        // A scope fork is a fresh timeline: it ingests its own scope against the
        // system prompt and holds none of the file conversation's dialogue, so
        // it must not inherit the file conversation's memory either.
        //
        // `seed_recurrent_from_branch: false` — NOT because a scope fork
        // wouldn't benefit from it (it processes the same system prompt as any
        // fresh conversation), but because `install_branch_states`'s own doc
        // measures the round-trip at 157ms of 175ms, and `fork_scope`'s only
        // caller (`zend`'s `ingest_scopes`) mints up to `SCOPE_PARALLELISM`
        // forks in a sequential loop before running them concurrently. Seeding
        // each one would serialise ~700ms into that loop, ahead of the
        // parallel section it exists to feed — exactly the per-fork queue
        // wait that function was built to batch away. A background
        // summarisation pass over a couple of turns is a much smaller loss
        // from starting at zero recurrent state than a user-facing reply is.
        self.fork_onto(fork_timeline, false)
    }

    /// Splice a per-scope fork's two coupled turns onto THIS (file) timeline in
    /// order, then couple them — the ordered-merge step of parallel code_read.
    /// The fork's sealed K/V is referenced (not copied); its turns must still be
    /// HOT (splice right after the fork completes). Records at the next two file
    /// indices and returns them.
    pub fn splice_scope_turns(
        &self,
        fork_timeline: TimelineId,
        call_idx: u32,
        resp_idx: u32,
        tags: Vec<String>,
    ) -> crate::Result<(u32, u32)> {
        let file_tl = self.target.timeline;
        let new_call = self
            .substrate
            .adopt_turn(
                fork_timeline,
                TurnIndex(call_idx),
                file_tl,
                Role::Assistant,
                tags.clone(),
            )
            .map_err(ConversationError::Model)?;
        let new_resp = self
            .substrate
            .adopt_turn(
                fork_timeline,
                TurnIndex(resp_idx),
                file_tl,
                Role::Assistant,
                tags,
            )
            .map_err(ConversationError::Model)?;
        self.substrate
            .couple_turn(file_tl, new_call.0)
            .map_err(ConversationError::Model)?;
        // The adopted K/V arrived by reference, so this conversation's memory has
        // not seen it. Catch it up over the whole adopted span at once.
        self.catch_memory_up_to(&[new_call.0, new_resp.0])?;
        // The fork's turns now live on the file timeline (their K/V shared by
        // reference); tombstone the fork so its orphaned timeline isn't reloaded
        // or scored. The shared chunks stay alive via the file timeline's clones.
        let _ = self.substrate.tombstone_timeline(fork_timeline);
        Ok((new_call.0, new_resp.0))
    }

    /// Advance this conversation's recurrent memory over turns it has just
    /// adopted **by reference**.
    ///
    /// A splice moves a scope fork's sealed K/V onto this timeline without it
    /// ever passing through this conversation's forward, so its memory is behind
    /// the K/V it now holds — §1's defect, produced by the merge rather than by
    /// a restart. §4.4 calls N-children-into-one a *divergent join* and declares
    /// it out of scope on the grounds that there is no operation meaning "and
    /// also these other tokens"; there is one here, because the adopted turns
    /// land at the **tail**. A recurrence cannot fill a hole in the middle of a
    /// sequence, but it can absorb an append.
    ///
    /// One pass over the whole adopted span, not one per turn: the state is a
    /// function of the ordered stream, so the batching is free correctness-wise
    /// and turns N prefills into one.
    pub(super) fn catch_memory_up_to(&self, adopted: &[u32]) -> crate::Result<()> {
        if !self.model_core.carries_recurrent_state {
            return Ok(());
        }
        // The adopted turns, and where the adopted span starts — everything
        // before it is this conversation's own history, which the replay must
        // attend over.
        //
        // The scheduler injects those prior turns onto the replay slot so the
        // adopted span sees the left context it would have seen had this
        // conversation decoded it. Replaying against nothing — the first
        // version of this — makes the attention layers feed the recurrent
        // layers outputs no real forward ever produced.
        let turns: Vec<(TimelineId, TurnIndex)> = adopted
            .iter()
            .map(|idx| (self.target.timeline, TurnIndex(*idx)))
            .collect();
        let adopted_from = adopted.iter().copied().min().unwrap_or(0);
        match self.replay_memory_over(&turns, adopted_from) {
            Ok(_) => Ok(()),
            Err(ConversationError::SchedulerGone) => Err(ConversationError::SchedulerGone),
            Err(e) => {
                // Not fatal to the splice — the K/V is committed and the
                // conversation is usable — but it leaves memory behind its
                // history, which is invisible from the outside.
                tracing::warn!(
                    "memory catch-up after a scope splice failed ({e}) — this \
                     conversation now holds K/V its recurrent layers never saw, \
                     and will read perfectly while doing so"
                );
                Ok(())
            }
        }
    }

    /// Advance this conversation's recurrent memory over the stored turn
    /// `index` of `timeline`, all of its own turns as their left context,
    /// adding no K/V: the catch-up a splice runs, for a turn this conversation
    /// holds some other way.
    pub fn catch_memory_up_over(&self, timeline: TimelineId, index: u32) -> crate::Result<bool> {
        self.replay_memory_over(&[(timeline, TurnIndex(index))], u32::MAX)
    }

    /// Replay `turns`' stored ids, their images as their rows, through this
    /// conversation's recurrent memory, the turns of its own timeline before
    /// `adopted_from` injected as their context. `Ok(false)` when there is
    /// nothing to replay or the model carries no recurrent memory.
    fn replay_memory_over(
        &self,
        turns: &[(TimelineId, TurnIndex)],
        adopted_from: u32,
    ) -> crate::Result<bool> {
        let tokens: Vec<u32> = {
            let view = self.substrate.read();
            turns
                .iter()
                .flat_map(|&(timeline, index)| view.token_ids_of(timeline, index))
                .collect()
        };
        if tokens.is_empty() {
            return Ok(false);
        }
        let image_spans = stored_spans(
            &self.substrate,
            VisionTokens::of(&self.tokenizer),
            self.vision.map(|v| v.name),
            turns,
        )
        .map_err(ConversationError::Other)?;
        let (tx, rx) = flume::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::MemoryCatchUp {
                sequence_id: self.id,
                tokens: TokenBuffer::from(tokens),
                adopted_from,
                image_spans,
                response_tx: tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        replied(&rx).map_err(|_| ConversationError::SchedulerGone)?
    }

    /// [`Self::splice_scope_turns`] for a chain of any length: adopt every turn
    /// onto this conversation's timeline in order, couple all but the last (each
    /// prefilled turn belongs with the one answering it), and tombstone the fork.
    /// Returns the adopted indices in the same order.
    pub fn splice_chain_turns(
        &self,
        fork_timeline: TimelineId,
        indices: &[u32],
        tags: Vec<String>,
    ) -> crate::Result<Vec<u32>> {
        let target_tl = self.target.timeline;
        let mut adopted = Vec::with_capacity(indices.len());
        for idx in indices {
            let new_idx = self
                .substrate
                .adopt_turn(
                    fork_timeline,
                    TurnIndex(*idx),
                    target_tl,
                    Role::Assistant,
                    tags.clone(),
                )
                .map_err(ConversationError::Model)?;
            adopted.push(new_idx.0);
        }
        for idx in adopted.iter().take(adopted.len().saturating_sub(1)) {
            self.substrate
                .couple_turn(target_tl, *idx)
                .map_err(ConversationError::Model)?;
        }
        // One catch-up over the whole adopted chain — see `catch_memory_up_to`.
        self.catch_memory_up_to(&adopted)?;
        let _ = self.substrate.tombstone_timeline(fork_timeline);
        Ok(adopted)
    }

    /// Tombstone a per-scope fork that will NOT be spliced onto the file
    /// timeline — its chunk failed before its ordered [`Self::splice_scope_turns`]
    /// ran, so its round-trip either errored or completed-but-unadopted.
    /// [`Sequence`]'s `Drop` only frees the scheduler slot; without this the
    /// fork's registered timeline and any sealed round-trip turns linger in the
    /// substrate as an orphaned, path-less "(untitled)" scope conversation.
    /// Mirrors the tombstone `splice_scope_turns` performs after adopting.
    pub fn tombstone_fork(&self, fork_timeline: TimelineId) {
        let _ = self.substrate.tombstone_timeline(fork_timeline);
    }

    /// Fork this sequence onto a **fresh timeline**: a new conversation thread
    /// sharing the workspace substrate and the system prompt, starting with no
    /// dialogue history of its own.
    ///
    /// **It does not inherit the parent's turns**, and the naming has misled
    /// before: turns are gathered per timeline, and this mints a new one, so the
    /// child projects the system prompt and nothing else. A behavioural test
    /// settled it — asked about a codeword stated one turn before the fork, the
    /// child answered by reasoning about the *system prompt*, because that was
    /// the whole of its context.
    ///
    /// That is why the child's recurrent memory starts from the prompt branch
    /// rather than from the parent: **memory must match the history the
    /// conversation actually holds.** Handing it the parent's live memory would
    /// give it a recollection of turns its attention layers have never seen —
    /// state without K/V, which is §7.8 defect 2 and reads perfectly.
    ///
    /// For a fork that *does* continue the parent's conversation, see
    /// [`Self::fork_resuming`], which targets a timeline that already holds the
    /// history.
    pub fn fork(&self) -> crate::Result<Sequence> {
        // Mint a fresh timeline within the parent's (layer, group)
        // shape — each fork is its own conversation thread.
        let fork_timeline = self
            .substrate
            .mint_timeline(self.target.layer, self.target.group);
        self.fork_onto(fork_timeline, true)
    }

    /// [`Self::fork`], decoding at `band` in place of its layer's priority. The
    /// band is recorded before the fork's slot is made, which is when a slot
    /// reads it.
    pub fn fork_at(&self, band: DecodePriority) -> crate::Result<Sequence> {
        let fork_timeline = self
            .substrate
            .mint_timeline(self.target.layer, self.target.group);
        self.substrate.set_conversation_metadata(
            fork_timeline,
            DecodePriority::METADATA_KEY,
            band.name(),
        )?;
        self.fork_onto(fork_timeline, true)
    }

    /// Fork onto a **specific** timeline rather than a freshly minted one —
    /// the daemon resume path: a restarted daemon forks onto the timeline whose
    /// history the redo log holds.
    ///
    /// `timeline` is registered against the parent's `(layer, group)`
    /// (idempotent). If the workspace substrate already holds turns under
    /// `timeline` — reconstructed from the redo log on startup — the next
    /// `submit_turn` materialises that recovered history onto the fork's
    /// slot, exactly as an in-process fork inherits its parent's turns.
    /// For an unknown `timeline` the fork simply starts empty.
    pub fn fork_resuming(&self, timeline: TimelineId) -> crate::Result<Sequence> {
        self.substrate
            .register_timeline(timeline, self.target.layer, self.target.group);
        // The child's memory comes from `timeline`'s own sealed snapshot, not
        // from this sequence: the daemon resumes a client's conversation by
        // forking its *base* conversation onto that client's timeline, and the
        // base holds a memory of turns the client never had. `fork_onto` takes
        // the parent's live state only when `timeline` is this sequence's own.
        self.fork_onto(timeline, true)
    }

    /// Shared body of [`Self::fork`] / [`Self::fork_resuming`] /
    /// [`Self::fork_scope`]: allocate a fresh scheduler slot bound to the
    /// workspace substrate and `timeline`.
    ///
    /// `inherits` is the whole of the difference, and it is not a mode: it is a
    /// statement about `fork_timeline`. A timeline that already holds this
    /// conversation's turns gives the child that history, so the child may take
    /// the parent's memory. A freshly minted one gives it nothing, so the
    /// parent's memory would be a recollection of turns the child's attention
    /// layers have never seen.
    pub(super) fn fork_onto(
        &self,
        fork_timeline: TimelineId,
        seed_recurrent_from_branch: bool,
    ) -> crate::Result<Sequence> {
        // The parent's live memory describes the parent's own history. It is
        // therefore the child's memory exactly when the child continues that
        // history — when the fork lands back on the timeline the parent is
        // already on. Every other fork, including a resume of some *other*
        // conversation's timeline, must keep the state `create_sequence`
        // restored from that timeline's own sealed snapshot.
        let inherits = fork_inherits_history(self.target.timeline, fork_timeline);
        // **A fork must be taken at a sealed boundary.**
        //
        // The child's spliced K/V covers the parent's history *as sealed*. A
        // parent with a turn in flight has advanced its per-sequence state past
        // that point, so copying it produces a child whose recurrent layers
        // have seen tokens its attention layers have not — an asymmetry nothing
        // downstream can detect and no later operation can repair. Refusing is
        // the same rule the view carve obeys automatically by being taken at
        // the turn boundary; the difference is only that a user-facing fork
        // needs an error where the internal one has an invariant.
        if self.turn_in_flight {
            return Err(ConversationError::Channel(
                "fork with a turn in flight: the parent's per-sequence state is \
                 ahead of its sealed K/V, so the fork would carry a recurrent \
                 memory of tokens its attention layers never saw. Finish or \
                 cancel the turn, then fork."
                    .into(),
            ));
        }

        let fork_target = ProjectionTarget {
            layer: self.target.layer,
            group: self.target.group,
            timeline: fork_timeline,
        };

        // Allocate a fresh slot bound to the same workspace handle and
        // the fork's target.
        let (tx, rx) = flume::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::NewSequence {
                conversation: self.substrate.clone(),
                target: Some(fork_target),
                // The live parent, so the child's memory comes across
                // device-to-device instead of via the last seal's record — but
                // only when the child is actually continuing the parent's
                // history. A fresh timeline holds none of it.
                parent: match inherits {
                    InheritsHistory::Yes => Some(self.id),
                    InheritsHistory::No => None,
                },
                response_tx: tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;

        let new_seq_id = replied(&rx).map_err(|_| ConversationError::SchedulerGone)??;

        let fork_conv = Sequence {
            submitter: self.submitter.clone(),
            ask_state: Arc::new(Mutex::new(AskState {
                projection: Arc::clone(&self.base_projection),
                selection: self.selection.clone(),
            })),
            id: new_seq_id,
            tree: self.tree.clone(),
            selection: self.selection.clone(),
            pending_user: None,
            turn_counter: self.turn_counter,
            turn_in_flight: false,
            freed: false,
            current_blocks: self.current_blocks,
            chunk_size: self.chunk_size,
            model_core: self.model_core,
            vision: self.vision,
            // A fork starts from the schema its parent was given: the parent's
            // submitted sections belong to the parent and go when it does.
            base_projection: Arc::clone(&self.base_projection),
            projection: Arc::clone(&self.base_projection),
            submitted: SubmittedSections::default(),
            // Forks share the same substrate (Arc clone) so cross-fork
            // history aggregation continues to work.
            substrate: self.substrate.clone(),
            target: fork_target,
            birth_prefix: Arc::clone(&self.birth_prefix),
            birth_ranges: self.birth_ranges.clone(),
            // A fork of a slot that holds K/V borrows it and never reads this;
            // one of an empty slot (an ingest base that never ran a turn) is
            // framed as its parent would have been.
            frame: self.frame.clone(),
            frame_window: None,
            branch_spans: self.branch_spans.clone(),
            // A fork's state is real — the parent's live state or the
            // timeline's snapshot, never exactly the prompt checkpoint — so
            // the branch-swap window is closed.
            state_is_prompt_only: false,
            installed_branch: None,
            // Forks start with a fresh scanner state — scoring will refresh
            // on the next provenance scan.  No need to clone the parent's scores.
        };

        // Give a fork onto a genuinely fresh timeline the same recurrent head
        // start a schema-built conversation gets: the branch checkpoint
        // computed once for this exact system-prompt content (§4.6), copied
        // device-to-device into the fork's own recurrent state. Never a share
        // of the parent's live memory — `state_is_prompt_only: false` above
        // already says this fork's state is real and starts diverging from
        // the first token it decodes. See [`fork_wants_branch_checkpoint`]
        // for why the timeline/recurrence conditions are required, and
        // `fork_scope`'s call site for why `seed_recurrent_from_branch` exists.
        //
        // The two cheap checks gate the substrate lock + timeline lookup
        // below, rather than the other way round: every fork of every plain
        // transformer, and every same-timeline fork, would otherwise pay that
        // read for a predicate that was always going to be `false` anyway.
        if seed_recurrent_from_branch
            && matches!(inherits, InheritsHistory::No)
            && fork_conv.model_core.carries_recurrent_state
        {
            let turn_count = fork_conv.substrate.read().turn_count(fork_timeline);
            if fork_wants_branch_checkpoint(
                inherits,
                fork_conv.model_core.carries_recurrent_state,
                turn_count,
            ) {
                let (prefix, _tokens) =
                    fork_conv.prompt_branch(&fork_conv.birth_prefix, &fork_conv.branch_spans);
                if let Err(e) = fork_conv.restore_branch_checkpoint(prefix, None) {
                    tracing::warn!(
                        "fork onto a fresh timeline: branch checkpoint install failed ({e}) — \
                         the fork starts with zero recurrent state, exactly as it did before \
                         this existed"
                    );
                }
            }
        }

        Ok(fork_conv)
    }

    /// Seed this sequence's recurrent memory from the nearest ancestor that has
    /// a snapshot — the conversation it continues.
    ///
    /// Call it after recording the fork's lineage
    /// (`ConversationEngine::set_forked_from`) and before its first turn. The
    /// slot was seeded at creation from its own timeline, which a fresh fork
    /// has no snapshot for; the pointer that says whose memory it inherits only
    /// exists once the fork has returned, so the seed is asked for again here
    /// rather than the fork being reordered around it.
    ///
    /// `Ok(false)` when the model carries no recurrent state, or no ancestor
    /// has a snapshot yet — both ordinary, and both leave the sequence starting
    /// from the sequence-start state.
    /// Lay `turns` down in place of the inherited chain when this append-only
    /// conversation's first turn is framed ([`Frame::of`]): a calibration draft
    /// over one window of a reading too long for the chain's cap
    /// (`Substrate::INHERITED_CHAIN_TOKEN_CAP`). No effect on a conversation
    /// past its first turn, or one that reprojects — it has no frame.
    pub fn frame_window(&mut self, turns: Vec<TurnKey>) {
        self.frame_window = Some(turns);
    }

    pub fn seed_recurrent_from_lineage(&self) -> crate::Result<bool> {
        let (tx, rx) = flume::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::SeedRecurrentFromLineage {
                sequence_id: self.id,
                response_tx: tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        replied(&rx).map_err(|_| ConversationError::SchedulerGone)?
    }

    /// Run `tokens` through every layer from zero and give this slot the
    /// recurrent state they leave, before its first turn: the state a fresh
    /// prefill of `tokens` has, under whatever K/V the slot's turn is
    /// projected onto. `Ok(false)` when the model carries no recurrent state.
    pub fn install_fresh_recurrent_over(&self, tokens: Vec<u32>) -> crate::Result<bool> {
        let run = TokenRun { hole: 0, tokens };
        let Some(state) = run_state(&self.scheduler_tx, None, vec![run])? else {
            return Ok(false);
        };
        let (tx, rx) = flume::bounded(1);
        self.scheduler_tx
            .send(SchedulerRequest::InstallRecurrentState {
                sequence_ids: vec![self.id],
                state: SharedState::from(&state),
                response_tx: tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        replied(&rx).map_err(|_| ConversationError::SchedulerGone)?
    }
}
