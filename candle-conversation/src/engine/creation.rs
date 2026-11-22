//! Opens conversations on the engine: new, resumed, batched, and on a named
//! timeline.
//!
//! Each constructor mints or claims a timeline, allocates a scheduler slot and
//! builds the `Sequence` over it. The resume path reopens an existing timeline
//! with its turns, recurrent state and carried belief, and refuses a tombstoned
//! one. A plain prompt's frame section is chosen from the prompt's own tokens, so
//! two prompts never share a frame. Batch creation fires every slot request before
//! it waits, so the scheduler allocates the whole batch in one drain cycle.

use super::*;

impl ConversationEngine {
    /// Clone the workspace `Conversation` handle.
    ///
    /// The handle wraps the substrate behind its `RwLock`, so callers
    /// can take `.read()` / `.write()` views to inspect or mutate
    /// section / turn / timeline state directly.  Most production
    /// callers should instead go through `new_conversation` /
    /// `new_conversation_with_projection` / `Sequence::submit_turn`
    /// — this accessor is for tooling that needs the raw substrate
    /// (integration tests, diagnostics, the workspace inspector).
    /// How many sequences the model currently holds recurrent memory for.
    ///
    /// The leak gauge. Slot ids are recycled pool indices, so memory that
    /// outlives its conversation is not merely wasted VRAM — the next
    /// conversation on that id inherits a stranger's memory, fluently.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn live_memory_count(&self) -> usize {
        let (tx, rx) = flume::bounded(1);
        if self
            .core
            .scheduler_tx
            .send(crate::scheduler::SchedulerRequest::CountRecurrentMemories { response_tx: tx })
            .is_err()
        {
            return 0;
        }
        rx.recv().unwrap_or(0)
    }

    pub fn conversation(&self) -> Conversation {
        self.conversation.clone()
    }

    /// Create a new conversation.
    ///
    /// Creates a sequence slot on the scheduler. The system prompt is NOT
    /// prefilled here — call [`Sequence::initial_handle`] to prefill
    /// the system prompt + user header and get a [`TurnHandle`] confirming
    /// the tokens are in the KV cache.
    ///
    /// # Arguments
    ///
    /// * `system_prompt` — The formatted system prompt text. Pass `""` for none.
    /// * `config` — Per-conversation configuration (role markers, sampling, etc.).
    /// Persist a substrate-side resume key (`debug_id`) for
    /// `timeline`.  Used by the debug-id-resumable grow-conversation
    /// harness: a test can
    /// re-open the workspace, call [`Self::lookup_by_debug_id`] to
    /// find a previously-built timeline, and continue growing.
    ///
    /// Last-write-wins on replay.  Idempotent: the redo-log writer
    /// skips the append when the substrate already records the same
    /// value.
    pub fn set_conversation_debug_id(
        &self,
        timeline: TimelineId,
        debug_id: &str,
    ) -> crate::Result<()> {
        self.conversation
            .set_conversation_debug_id(timeline, debug_id)
            .map_err(ConversationError::Model)
    }

    /// Look up a timeline by its previously-set `debug_id`.  O(1).
    /// Returns `None` when no timeline carries that key.
    pub fn lookup_by_debug_id(&self, debug_id: &str) -> Option<TimelineId> {
        self.conversation.lookup_by_debug_id(debug_id)
    }

    /// Build an **engine-internal** conversation that lives on the reserved
    /// id range for `kind` — disjoint from any YAML-allocated user schema.
    ///
    /// This is the right entry point for synthetic helper conversations
    /// (the daemon's titler, future label-summarisers, etc.) that share
    /// the same workspace substrate as user conversations but must never
    /// have their turns enter user-projection retrieval.
    ///
    /// `system_prompt` may be either pre-formatted with dialect markers
    /// or raw text — `new_conversation_with_projection` handles wrapping
    /// the same way it does for user prompts. It decodes at
    /// `decode_priority` (the titler, background to the dialogue it names,
    /// is `Normal`).
    pub fn new_reserved_conversation(
        &self,
        system_prompt: &str,
        kind: Reserved,
        decode_priority: DecodePriority,
        config: SequenceConfig,
    ) -> crate::Result<Sequence> {
        let frame = self.plain_prompt_section(system_prompt)?;
        let builder =
            Builder::for_plain_prompt_reserved(system_prompt, kind, frame, decode_priority);
        let layer_id = LayerId::reserved(kind);
        let group_id = GroupId::reserved(kind);
        self.new_conversation_with_projection(system_prompt, builder, layer_id, group_id, config)
    }

    /// The section a plain prompt's frame is sealed under.
    ///
    /// Chosen from the prompt's own tokens and checked against what the
    /// substrate already holds — see [`PlainPromptFrames`] for why a fixed id
    /// hands a conversation the prompt some other conversation was opened with.
    /// Give it to [`Builder::for_plain_prompt`] or
    /// [`Builder::for_plain_prompt_reserved`] with this same text, since that
    /// text is what the section seals.
    ///
    /// The id is claimed as it is returned, so a conversation opening a
    /// different prompt at the same moment is not handed it before this one
    /// has sealed. A conversation opened for one job and discarded takes
    /// [`Self::transient_prompt_section`] instead, which leaves nothing behind.
    pub fn plain_prompt_section(&self, prompt_text: &str) -> crate::Result<SectionId> {
        let encoding = self
            .core
            .tokenizer
            .encode(prompt_text, false)
            .map_err(|e| ConversationError::Tokenizer(e.to_string()))?;
        let mut frames = self.plain_frames.lock().unwrap();
        let view = self.conversation.read();
        frames
            .resolve(encoding.get_ids(), |id| {
                view.section_exists(id).then(|| view.section_tokens_of(id))
            })
            .ok_or_else(|| {
                ConversationError::Other(format!(
                    "no frame section is free for this prompt within {} probes of its slot",
                    PlainPromptFrames::MAX_PROBES
                ))
            })
    }

    /// The section a throwaway conversation's frame is sealed under, held until
    /// [`Self::release_prompt_section`].
    ///
    /// [`Self::plain_prompt_section`] for a conversation opened for one job and
    /// discarded. The frame is **transient**: never written to disk, and retired
    /// from the substrate when the last conversation holding it releases it, so
    /// a caller whose every prompt is different leaves nothing behind. Two jobs
    /// on the same prompt share the section while both hold it.
    ///
    /// Every successful call must be matched by one release, made after the
    /// conversation that used the frame has been dropped.
    pub fn transient_prompt_section(&self, prompt_text: &str) -> crate::Result<SectionId> {
        let encoding = self
            .core
            .tokenizer
            .encode(prompt_text, false)
            .map_err(|e| ConversationError::Tokenizer(e.to_string()))?;
        let id = {
            let mut frames = self.plain_frames.lock().unwrap();
            let view = self.conversation.read();
            frames.acquire(encoding.get_ids(), |id| {
                view.section_exists(id).then(|| view.section_tokens_of(id))
            })
        }
        .ok_or_else(|| {
            ConversationError::Other(format!(
                "no transient frame section is free for this prompt within {} probes of its \
                 slot",
                PlainPromptFrames::MAX_PROBES
            ))
        })?;
        // Before the conversation that uses it opens: the seal reads this to
        // skip the disk, and a section that sealed first has already written.
        self.conversation.mark_section_transient(id);
        Ok(id)
    }

    /// Let go of a frame taken with [`Self::transient_prompt_section`].
    ///
    /// The last release retires the section, on the wave thread between waves —
    /// see [`SchedulerRequest::RetireSections`]. Sent after the conversation's
    /// own slot was freed, on the same queue, so the retirement never runs ahead
    /// of the slot that was reading the frame.
    pub fn release_prompt_section(&self, section: SectionId) {
        let last = self.plain_frames.lock().unwrap().release(section);
        if last {
            let _ = self
                .core
                .scheduler_tx
                .send(SchedulerRequest::RetireSections {
                    conversation: self.conversation.clone(),
                    sections: vec![section],
                });
        }
    }

    pub fn new_conversation(
        &self,
        system_prompt: &str,
        config: SequenceConfig,
    ) -> crate::Result<Sequence> {
        // Build a synthetic single-layer / single-group projection from the
        // raw prompt and delegate to the projection-aware constructor,
        // which mints a fresh `TimelineId` internally.
        //
        // `for_plain_prompt` stores the text as the schema section content,
        // and `new_with_projection` wraps it with dialect markers at ingest
        // time — so we strip the markers here before passing the inner text.
        let inner_prompt = {
            let s = system_prompt
                .strip_prefix(config.dialect.system_start)
                .unwrap_or(system_prompt);
            s.strip_suffix(config.dialect.system_end).unwrap_or(s)
        };
        // The frame's id comes from its text — see [`Self::plain_prompt_section`].
        let frame = self.plain_prompt_section(inner_prompt)?;
        let builder = Builder::for_plain_prompt(inner_prompt, frame);
        let (layer_id, group_id) = {
            let layer = &builder.schema().layers[0];
            (layer.id, layer.groups[0].id)
        };
        self.new_conversation_with_projection(system_prompt, builder, layer_id, group_id, config)
    }

    /// Create a new conversation backed by a full projection [`Builder`].
    ///
    /// Identical to [`Self::new_conversation`] except the given `builder`
    /// replaces the synthetic schema constructed from the prompt string,
    /// and `target` names the `(layer, group)` this conversation is for —
    /// turns are appended into `target.group`, and `target` is what gets
    /// passed to `projection.project()`.
    ///
    /// `system_prompt` must be the ChatML-formatted text (same as you would
    /// pass to `new_conversation`).
    pub fn new_conversation_with_projection(
        &self,
        system_prompt: &str,
        builder: Builder,
        layer: LayerId,
        group: GroupId,
        config: SequenceConfig,
    ) -> crate::Result<Sequence> {
        self.new_conversation_with_projection_progress(
            system_prompt,
            builder,
            layer,
            group,
            config,
            None,
        )
    }

    /// Same as [`Self::new_conversation_with_projection`] but accepts an
    /// optional progress callback fired as the schema's pinned sections
    /// are prefilled. The callback receives
    /// `(chars_done, total_chars)` — total content-bytes across every
    /// schema-declared section (including collection members). Used by
    /// the daemon's loading overlay; library callers pass `None`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_conversation_with_projection_progress(
        &self,
        system_prompt: &str,
        builder: Builder,
        layer: LayerId,
        group: GroupId,
        config: SequenceConfig,
        section_progress: Option<&dyn Fn(u64, u64)>,
    ) -> crate::Result<Sequence> {
        // Persist the projection schema as the substrate's `Template` record
        // (compare-and-insert) so the log carries the projection it was built
        // with. Programmatic schemas (no source YAML) are skipped.
        if let Some(yaml) = builder.source_yaml() {
            if let Err(e) = self.conversation.set_template(yaml.as_bytes()) {
                tracing::warn!("persist projection template failed: {e}");
            }
        }

        // Mint a fresh `TimelineId` for this conversation before
        // allocating a slot — the substrate registers it against
        // `(layer, group)` so the seal path can write turns into the
        // right timeline without consulting the schema.
        let timeline = self.conversation.mint_timeline(layer, group);
        self.bind_slot_to_timeline(
            timeline,
            Resumed::No,
            system_prompt,
            builder,
            layer,
            group,
            config,
            section_progress,
        )
    }

    /// [`Self::new_conversation_with_projection`] on `timeline`, which the
    /// caller derived rather than minted: a conversation a client names, whose
    /// id every member must derive the same way (`docs/replicated_substrate.md`
    /// §6.2). Refuses a timeline that is not [free](Self::is_timeline_free).
    #[allow(clippy::too_many_arguments)]
    pub fn new_conversation_on(
        &self,
        timeline: TimelineId,
        system_prompt: &str,
        builder: Builder,
        layer: LayerId,
        group: GroupId,
        config: SequenceConfig,
    ) -> crate::Result<Sequence> {
        // Claimed in the same edit that finds it free, so two opens of one
        // derived id cannot both take it.
        let claimed = self.conversation.update(|s| {
            let free = s.timeline_target(timeline).is_none() && !s.is_tombstoned(timeline);
            if free {
                s.register_timeline(timeline, layer, group);
            }
            free
        });
        if !claimed {
            return Err(ConversationError::Channel(format!(
                "cannot open a new conversation on timeline {timeline}: it is in use or deleted",
            )));
        }
        if let Some(yaml) = builder.source_yaml() {
            if let Err(e) = self.conversation.set_template(yaml.as_bytes()) {
                tracing::warn!("persist projection template failed: {e}");
            }
        }
        self.bind_slot_to_timeline(
            timeline,
            Resumed::No,
            system_prompt,
            builder,
            layer,
            group,
            config,
            None,
        )
    }

    /// Whether `timeline` names nothing here yet: not registered, and not a
    /// deleted conversation's — a tombstone or an expired one's entry keeps a
    /// deleted id dead for good (§6.8).
    pub fn is_timeline_free(&self, timeline: TimelineId) -> bool {
        self.conversation.timeline_target(timeline).is_none()
            && !self.is_timeline_tombstoned(timeline)
    }

    /// Open a **new GPU slot onto a conversation that already exists** in the
    /// substrate, and carry on appending to it.
    ///
    /// # What this is for
    ///
    /// Every other constructor mints a fresh `TimelineId`, so a process restart
    /// abandons whatever conversation the last one was using — the turns stay in
    /// the log, reachable by the gather, but nothing appends to them again. For
    /// an assistant that is right: a session ends. For a mind that never stops
    /// living it is not. Its conversation is *the* conversation, and a restart
    /// should rejoin it rather than start the day over with a stranger's history
    /// filed beside it.
    ///
    /// # What actually resumes
    ///
    /// The turns, the recurrent state and the carried selection belief — the
    /// three durable things a timeline owns. The first two are restored by
    /// [`crate::scheduler`]'s slot-creation funnel, which reads them from the
    /// timeline; the third is refolded from the persisted per-turn projection
    /// events. What does not resume is GPU residence: the slot starts empty and
    /// the next turn's projection materialises what it needs out of the
    /// substrate, exactly as it would after any eviction.
    ///
    /// # What the caller still owes
    ///
    /// **That this timeline is the right one to continue.** Nothing here can
    /// tell whether the schema, the prompt or the dialect has changed underneath
    /// it since it was written, and resuming into a conversation shaped by a
    /// different frame is worse than starting a new one — the model reads a
    /// history it would never have produced. Tag the conversation with a
    /// fingerprint of the frame it was opened under and check it before calling
    /// this; `npcd`'s character loop and zend's `content_sha256` ingest cache
    /// are both that pattern.
    ///
    /// Refuses an unregistered timeline, and refuses a tombstoned one — a
    /// tombstone is the record that something decided this conversation was
    /// finished, and silently reviving it would make that decision meaningless.
    pub fn resume_conversation_with_projection(
        &self,
        timeline: TimelineId,
        system_prompt: &str,
        builder: Builder,
        config: SequenceConfig,
    ) -> crate::Result<Sequence> {
        if self.is_timeline_tombstoned(timeline) {
            return Err(ConversationError::Channel(format!(
                "cannot resume conversation on timeline {timeline}: it is tombstoned",
            )));
        }
        let Some((layer, group)) = self.conversation.timeline_target(timeline) else {
            return Err(ConversationError::Channel(format!(
                "cannot resume conversation on timeline {timeline}: not registered in the substrate",
            )));
        };
        if let Some(yaml) = builder.source_yaml() {
            if let Err(e) = self.conversation.set_template(yaml.as_bytes()) {
                tracing::warn!("persist projection template failed: {e}");
            }
        }
        self.bind_slot_to_timeline(
            timeline,
            Resumed::Yes,
            system_prompt,
            builder,
            layer,
            group,
            config,
            None,
        )
    }

    /// [`Self::resume_conversation_with_projection`] for a conversation that
    /// was opened from a raw prompt string.
    ///
    /// The synthetic single-layer schema is rebuilt from `system_prompt` the
    /// same way [`Self::new_conversation`] builds it, so the frame the resumed
    /// conversation projects under is the frame it was created under — provided
    /// the caller hands over the same prompt. It cannot check that for you; see
    /// the fingerprint note on [`Self::resume_conversation_with_projection`].
    pub fn resume_conversation(
        &self,
        timeline: TimelineId,
        system_prompt: &str,
        config: SequenceConfig,
    ) -> crate::Result<Sequence> {
        let inner_prompt = {
            let s = system_prompt
                .strip_prefix(config.dialect.system_start)
                .unwrap_or(system_prompt);
            s.strip_suffix(config.dialect.system_end).unwrap_or(s)
        };
        // The frame's id comes from its text — see [`Self::plain_prompt_section`].
        let frame = self.plain_prompt_section(inner_prompt)?;
        let builder = Builder::for_plain_prompt(inner_prompt, frame);
        self.resume_conversation_with_projection(timeline, system_prompt, builder, config)
    }

    /// Allocate a slot for `timeline` and build the [`Sequence`] over it.
    ///
    /// The shared tail of [`Self::new_conversation_with_projection_progress`]
    /// and [`Self::resume_conversation_with_projection`] — everything from
    /// "the timeline exists" onwards is identical, and `resumed` chooses only
    /// which scheduler entry point allocates the slot.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn bind_slot_to_timeline(
        &self,
        timeline: TimelineId,
        resumed: Resumed,
        system_prompt: &str,
        builder: Builder,
        layer: LayerId,
        group: GroupId,
        config: SequenceConfig,
        section_progress: Option<&dyn Fn(u64, u64)>,
    ) -> crate::Result<Sequence> {
        // Register this conversation's per-conversation KV-compression
        // override (if any) before the first turn seals, so each turn
        // residence inherits it at alloc time. Utility layers set a
        // compression level (and may drop the K override or pin forced K/V
        // formats) via their SequenceConfig.
        let compression = if config.kv_compression_level.is_some()
            || config.kv_force_k_format.is_some()
            || config.kv_force_v_format.is_some()
            || config.kv_lossless
        {
            Some(ConvCompression {
                lossless: config.kv_lossless,
                level: config.kv_compression_level,
                disable_k_override: config.kv_disable_k_override,
                force_k: config.kv_force_k_format,
                force_v: config.kv_force_v_format,
            })
        } else {
            None
        };
        self.conversation
            .set_timeline_compression(timeline, compression);
        // Every layer summarises into its AVL summary tree; provenance scans then
        // No summarisation is registered for the timeline: the AVL summariser is
        // disconnected (see the `SummariserThread::disabled()` note in `new`), so
        // a timeline keeps its turns whole and retrieval ranks them by provenance
        // rather than reading a compressed stand-in.
        let target = ProjectionTarget {
            layer,
            group,
            timeline,
        };

        let (response_tx, response_rx) = flume::bounded(1);
        let request = match resumed {
            // A fresh conversation: any state comes from the timeline's own
            // snapshot, which `create_sequence` reads.
            Resumed::No => SchedulerRequest::NewSequence {
                conversation: self.conversation.clone(),
                target: Some(target),
                parent: None,
                response_tx,
            },
            // The same funnel, entered by the door that says so. Both end in
            // `create_sequence(conversation, Some(target))` and restore the
            // timeline's recurrent state and carried belief; what differs is
            // that this one re-derives `(layer, group)` from the substrate
            // registry and fails loudly if the timeline was never registered.
            Resumed::Yes => SchedulerRequest::ResumeSequence {
                conversation: self.conversation.clone(),
                timeline,
                response_tx,
            },
        };
        self.core
            .scheduler_tx
            .send(request)
            .map_err(|_| ConversationError::SchedulerGone)?;

        let sequence_id = response_rx
            .recv()
            .map_err(|_| ConversationError::SchedulerGone)??;

        // **Everything past the slot allocation has to hand the slot back.**
        //
        // A built [`Sequence`] frees its slot on drop, so the only window where
        // one can be stranded is between the scheduler handing out `sequence_id`
        // and a `Sequence` existing to own it. A `?` here would return through
        // that window and leave a GPU slot allocated, its sampling state
        // registered and its conversation handle held, with nothing left that
        // knows the id — leaked until the process ends.
        //
        // Rare, and rarer still to notice: the surviving symptom is a slot count
        // that never comes back down, which reads as a busy engine rather than
        // as an error. Resume widens the window — a conversation whose tree
        // rebuild fails is a new way in — so it is closed rather than reasoned
        // about.
        let free_slot = || {
            let _ = self
                .core
                .scheduler_tx
                .send(SchedulerRequest::FreeSequence { sequence_id });
        };
        let (conv, pending) = Sequence::new_with_projection(
            self.core.scheduler_tx.clone(),
            sequence_id,
            Arc::clone(&self.core.tokenizer),
            Arc::clone(&self.core.literal_tokenizer),
            system_prompt,
            builder,
            target,
            config,
            CHUNK_SIZE,
            self.core.model_core,
            self.conversation.clone(),
            section_progress,
        )
        .inspect_err(|_| free_slot())?;
        // A group of one — the same install path the batch create takes.
        let pending: Vec<_> = pending.into_iter().collect();
        if let Err(e) = install_branch_states(&self.core.scheduler_tx, &pending, &self.conversation)
        {
            // `conv` owns the slot by now and would free it on drop, but it is
            // dropped here without being returned, so say what happened rather
            // than letting a silent drop stand in for an error.
            drop(conv);
            return Err(e);
        }

        Ok(conv)
    }

    /// Batch-create `n` conversations that share one projection (system prompt,
    /// builder, layer/group, config), **pipelining** the `NewSequence` slot
    /// allocations: every request is fired before any response is awaited, so
    /// the scheduler drains the whole batch in a single cycle and it costs ~one
    /// round-trip instead of `n` serial ones.
    ///
    /// [`Self::new_conversation_with_projection`] blocks on its slot-alloc reply,
    /// and the scheduler interleaves those replies between forward waves — so
    /// creating a window of cases one at a time pays ~one wave-latency per case.
    /// During calibration that starved the wave-batched prefill to 2–4 sequences
    /// wide (poor MoE expert amortization). Firing the window's allocations up
    /// front lets the cases prefill together in one wide forward instead.
    ///
    /// Each conversation gets a fresh timeline; per-conversation compression and
    /// summariser settings mirror the single-create path. Returns one `Result`
    /// per requested conversation, in submission order.
    pub fn new_conversations_with_projection_batch(
        &self,
        n: usize,
        system_prompt: &str,
        builder: &Builder,
        layer: LayerId,
        group: GroupId,
        config: &SequenceConfig,
    ) -> Vec<crate::Result<Sequence>> {
        if n == 0 {
            return Vec::new();
        }
        if let Some(yaml) = builder.source_yaml() {
            if let Err(e) = self.conversation.set_template(yaml.as_bytes()) {
                tracing::warn!("persist projection template failed: {e}");
            }
        }
        let compression = if config.kv_compression_level.is_some()
            || config.kv_force_k_format.is_some()
            || config.kv_force_v_format.is_some()
            || config.kv_lossless
        {
            Some(ConvCompression {
                lossless: config.kv_lossless,
                level: config.kv_compression_level,
                disable_k_override: config.kv_disable_k_override,
                force_k: config.kv_force_k_format,
                force_v: config.kv_force_v_format,
            })
        } else {
            None
        };

        // Phase 1 — mint a timeline and fire `NewSequence` for every case
        // WITHOUT awaiting, so all `n` requests sit in the scheduler queue
        // together and one drain cycle allocates every slot.
        struct Fired {
            target: ProjectionTarget,
            rx: Receiver<crate::Result<SequenceId>>,
        }
        // Split the call's wall time three ways, because the three parts have
        // very different characters and a caller that batches creations needs to
        // know which one it is paying for. `fire` and `build` are this thread's
        // own work; `wait` is time blocked on the scheduler, which only drains
        // its request queue between waves — so a `wait` of order one wave
        // latency means the caller is stalling on a wave boundary rather than on
        // the cost of creating anything.
        let t_call = std::time::Instant::now();
        let mut fire = std::time::Duration::ZERO;
        let mut wait = std::time::Duration::ZERO;
        let mut fired: Vec<crate::Result<Fired>> = Vec::with_capacity(n);
        let t_fire = std::time::Instant::now();
        // All `n` timelines under one write-lock span, before the fire loop —
        // minting per iteration took the substrate's exclusive lock once per
        // conversation, right where conversations are opened in bursts.
        let minted = self.conversation.mint_timelines(layer, group, n);
        for timeline in minted {
            let target = ProjectionTarget {
                layer,
                group,
                timeline,
            };
            let (response_tx, rx) = flume::bounded(1);
            match self.core.scheduler_tx.send(SchedulerRequest::NewSequence {
                conversation: self.conversation.clone(),
                target: Some(target),
                // Resume by timeline: the snapshot read in `create_sequence` is
                // the whole of the state recovery here — there is no live
                // parent to copy from.
                parent: None,
                response_tx,
            }) {
                Ok(()) => {
                    // Only register per-timeline metadata for a conversation that
                    // actually exists — set it after the send succeeds (still
                    // before the scheduler drains the request or any turn seals,
                    // so residence compression inheritance is unaffected). A
                    // failed send leaves only the bare timeline mint, no metadata.
                    self.conversation
                        .set_timeline_compression(timeline, compression);
                    fired.push(Ok(Fired { target, rx }));
                }
                Err(_) => fired.push(Err(ConversationError::SchedulerGone)),
            }
        }

        fire += t_fire.elapsed();

        // Phase 2 — collect each slot id (already queued, so no extra wave wait)
        // and build its Sequence. After the warm-up case has pinned the shared
        // sections, `new_with_projection` only re-references already-hot sections
        // here, so it issues no further scheduler round-trip.
        // Collect the whole batch's pending branch states as the sequences are
        // built, then install them in ONE request below. The install is
        // overwhelmingly the wait for the scheduler to drain it, and that wait is
        // per-request — installing per sequence here would put a queue wait
        // between every pair of conversations in the burst.
        let mut pending: Vec<PendingBranchState> = Vec::with_capacity(n);
        let out: Vec<crate::Result<Sequence>> = fired
            .into_iter()
            .map(|f| {
                let f = f?;
                let t_wait = std::time::Instant::now();
                let sequence_id =
                    f.rx.recv()
                        .map_err(|_| ConversationError::SchedulerGone)??;
                wait += t_wait.elapsed();
                let (conv, p) = Sequence::new_with_projection(
                    self.core.scheduler_tx.clone(),
                    sequence_id,
                    Arc::clone(&self.core.tokenizer),
                    Arc::clone(&self.core.literal_tokenizer),
                    system_prompt,
                    builder.clone(),
                    f.target,
                    config.clone(),
                    CHUNK_SIZE,
                    self.core.model_core,
                    self.conversation.clone(),
                    None,
                )?;
                pending.extend(p);
                Ok(conv)
            })
            .collect();
        // NOT downgraded to a warning. `install_branch_states` already handles
        // every recoverable degradation internally (missing checkpoint, unreadable
        // record, refused install) by warning and carrying on — so the only error
        // it can return is `SchedulerGone`. Swallowing it would hand back a full
        // batch of `Ok(Sequence)` values bound to a dead scheduler; this function
        // reports per entry, so every entry fails.
        if let Err(e) = install_branch_states(&self.core.scheduler_tx, &pending, &self.conversation)
        {
            let msg = format!("batch branch-state install failed: {e}");
            return (0..n)
                .map(|_| Err(ConversationError::Channel(msg.clone())))
                .collect();
        }

        let call = t_call.elapsed();
        tracing::debug!(
            target: "candle_conversation::batch_creation",
            n,
            call_ms = call.as_millis() as u64,
            fire_ms = fire.as_millis() as u64,
            wait_ms = wait.as_millis() as u64,
            build_ms = call.saturating_sub(fire + wait).as_millis() as u64,
            "batch conversation creation"
        );
        out
    }
}
