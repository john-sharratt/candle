//! Restores and begins sections: it installs persisted sections as cold markers,
//! pushes their index pages, and claims the state a section starts from.
//!
//! A restored section costs no disk I/O until a projection selects it. It is
//! installed as a cold marker and lifted on first use. Its index page travels
//! beside its K/V, because the page is computed from hidden states, and a restore
//! without it would reach every projection unindexed. A section ingest is claimed
//! and prepared only after admission, and it prefills on its own slot so that its
//! build attends its native K/V before any quantization. A section without a page
//! the model can take is advanced over as a gap of its own width, so the index
//! still lines up with the slot.

use super::*;

impl Scheduler {
    /// Install a persisted section as a **cold-marker** — `cold = Some`,
    /// `hot = None`.  No disk I/O for the chunks at restore time; the
    /// K/V is materialised into VRAM on the first projection that
    /// selects this section, via `elevate_to_hot`'s
    /// `PromotionItemKind::Section` branch.
    ///
    /// Lazy reload pays cold-load only for sections that actually get
    /// projected.  For a `top_k=3` tools collection over 90 entries
    /// that's 3 cold-loads at first projection vs. 90 at startup —
    /// the trade we want once selection is honoured.
    pub(super) fn restore_section_from_persistence(
        &mut self,
        conversation: &Conversation,
        section_id: SectionId,
        stream_id: StreamId,
        tokens: TokenBuffer,
    ) -> Result<usize, ConversationError> {
        // The cold→hot section install goes through the adaptive-format
        // restore machinery (`alloc_sealed_block` + per-band arena writes),
        // which does not alias K≡V. A single-latent backing (DeepSeek's
        // fixed-FP8 plan) restored through it presents broken chunk tables
        // to the kernels — the caller must re-ingest instead (deterministic:
        // the re-prefill reproduces the exact same K/V the seal captured).
        if self
            .session
            .backings()
            .first()
            .is_some_and(|b| b.single_latent())
        {
            return Err(ConversationError::Channel(
                "section restore-from-log unsupported on a single-latent backing — re-ingest"
                    .into(),
            ));
        }
        // One chunk list per KV backing — the count the persistence thread
        // wrote the section with. On a hybrid that is the attention layers
        // (plus any draft head), not the model's transformer depth; dividing
        // the persisted grid by the depth refused every section whose chunk
        // total it did not happen to divide, so every boot prefilled the whole
        // system prompt again.
        let n_layers = self.session.num_layers();

        // 1. Resolve cold refs from the manifest — these point at
        //    each chunk's `(log_offset, record_len, token_count)` in
        //    the redo log.  Installed alone (with hot = None) so the
        //    elevate path can lift the section when a projection
        //    needs it.  A stream with no whole grid is not restorable,
        //    and saying so sends the caller to a prefill.
        let cold_refs = conversation
            .recover_section_cold_refs(stream_id, n_layers)
            .map_err(ConversationError::Model)?
            .ok_or_else(|| {
                ConversationError::Channel(format!(
                    "section {section_id:?}: stream {stream_id:?} holds no durable \
                     chunks — re-ingest"
                ))
            })?;

        // 2. The section's index page, for a model that keeps per-position
        //    state. It is computed from hidden states, so borrowing the K/V
        //    does not bring it along; a section restored without it would
        //    reach every projection unindexed, and a prefill is the only way
        //    to rebuild it.
        let page = if self.model.carries_positional_state() {
            let page = conversation.read_index_page(stream_id)?.ok_or_else(|| {
                ConversationError::Channel(format!(
                    "section {section_id:?}: persisted without its index page — re-ingest"
                ))
            })?;
            Some(page)
        } else {
            None
        };

        // 3. Install as a cold-marker.  `sealed_hot = Vec::new()`
        //    leaves `residence.hot = None`; `cold_refs` lands in
        //    `residence.cold` so `elevate_to_hot` can lift the
        //    section on the first projection that selects it.
        //    `token_count` comes from the cold refs (sum of
        //    per-chunk token counts across one layer's
        //    StoredSequence).
        let token_count = cold_refs.first().map(|s| s.token_count).unwrap_or(0);
        let chunks_per_layer = cold_refs.first().map(|s| s.chunks.len()).unwrap_or(0);
        let tokens_arc = Arc::new(tokens[..].to_vec());
        conversation.update(|substrate| {
            substrate.restore_section(
                section_id,
                stream_id,
                token_count,
                Vec::new(),
                cold_refs.clone(),
                Arc::clone(&tokens_arc),
            )
        });
        if let Some(page) = page {
            self.section_positional
                .insert(stream_id, conversation, page);
        }
        Ok(chunks_per_layer)
    }

    /// A unit's own tokens begin on `slot`: close its index so the prefix ahead
    /// of them becomes a page of its own.
    ///
    /// **Call this at the instant the unit's K/V anchor is captured, and
    /// nowhere else.** The anchor and this cut are one fact — "this unit's own
    /// tokens start here" — recorded for two stores that the seal reads back
    /// together: `[anchor, block_count)` of the K/V, and the matching token
    /// count from the index. Nothing in either store makes them agree. The
    /// prefix ahead of a unit is *forwarded* on this slot, so its rows land in
    /// the same live index tail the unit is about to extend, and the seal then
    /// asks for "the last N tokens" of a run that begins before the unit does.
    ///
    /// **Where a turn's own tokens begin depends on how its user half got
    /// there, so the boundary is taken at both places it can be.** A projection
    /// that carries the in-flight user message *prefills it onto the parent*
    /// (`deferred_user`) before the view is carved, so the turn's content starts
    /// there; a slot that skips projection (the `code_read` ingest) forwards
    /// nothing before the carve, so its turn starts on the view. The call is
    /// idempotent — an already-empty tail closes `0` — so both fire and whichever
    /// one is the real boundary is the one that closes anything.
    ///
    /// Sites: [`projection_assembler::apply_segments_finish`] (before the
    /// deferred user prefill), [`Self::create_view`] (before the carve, so
    /// `fork_recurrent` copies a cache already closed), [`Self::reproject_view`]
    /// (the same turn's anchor moving to the rebuilt prefix's end), and
    /// [`Self::prepare_section_ingest`] (a section). The anchor itself is not
    /// passed in because they do not share a coordinate system — a windowed
    /// borrow counts view blocks, an ingest counts the slot's own — while the
    /// cut is the same operation on the same slot in all of them.
    ///
    /// Recording them apart is what let them drift, and getting the *set* wrong
    /// is what survived the first consolidation: with only the carve, a turn
    /// whose user half was prefilled during projection had that half closed into
    /// the prefix's page, leaving a 7-token assistant header as the whole of its
    /// "prefill" page. Measured on turn after turn — `turn_token_count=324`,
    /// `reasoning=(296, 3)`, sealed `widths=[7, 28]` — the 289-token user
    /// message sealed as part of the prefix, every time.
    ///
    /// The closed width is logged rather than discarded. A cut placed after the
    /// tokens it was meant to separate closes `0` and is otherwise
    /// indistinguishable from one that was simply not needed, so the number is
    /// the only evidence the boundary landed where it was aimed.
    pub(super) fn begin_unit(&mut self, slot: SequenceId, unit: &'static str) {
        let held = self.session.sequence_offset(slot.0).unwrap_or(0);
        close_unit_boundary(self.model.as_ref(), slot, unit, held);
    }

    /// Install prefix section `prefix_id`'s index page on `seq`, ahead of the
    /// `tokens` of its K/V the ingest borrows.
    ///
    /// **A section without a page the model takes is advanced over as a gap.**
    /// Its span stays unindexed, which is the truth about it, and everything
    /// after it still sits where its K/V does. Left short instead, the index
    /// covers fewer tokens than the slot holds, every select of the ingest
    /// refuses, and the wave step fails on each retry for as long as the daemon
    /// runs — a startup wedged in "Prefilling tool sections".
    ///
    /// The page is keyed by the section's content address, `stream`, which the
    /// caller resolves from the substrate read it already holds. **This takes no
    /// lock of its own:** the caller walks the prefix under one read guard, and a
    /// second read of the same `RwLock` taken inside it waits behind any writer
    /// queued in between — the persistence thread commits continuously — while
    /// that writer waits for the first guard. The scheduler then parks for good,
    /// idle, mid-ingest.
    pub(super) fn push_prefix_section_index(
        &self,
        seq: SequenceId,
        prefix_id: SectionId,
        stream: Option<StreamId>,
        tokens: usize,
    ) {
        let refused = match stream.and_then(|s| self.section_positional.get(s)) {
            Some(page) => {
                match self
                    .model
                    .push_positional_state(seq.0, page.bytes(), &page.whole_key())
                {
                    Ok(_) => None,
                    Err(e) => Some(format!("index page refused: {e}")),
                }
            }
            None if self.model.carries_positional_state() => Some("has no index page".to_string()),
            None => None,
        };
        if let Some(why) = refused {
            let advanced = self.model.push_positional_gap(seq.0, tokens);
            tracing::warn!(
                "prepare_section_ingest: prefix section {prefix_id:?} {why} — its {tokens} \
                 token(s) stay unindexed (gap advanced: {})",
                advanced.is_ok()
            );
        }
    }

    /// CPU-only setup for a section ingest: truncate slot, inject prefix,
    /// capture `seal_block_from`, push writer chunk, pin tokens on substrate.
    /// Returns `seal_block_from` (the chunk index lower bound for the seal).
    /// No forward pass — the caller queues an [`ActiveSectionIngest`] entry
    /// so the batched prefill loop can interleave multiple sections.
    /// Start an admitted section: claim its slot's carried state, prepare the
    /// slot, then seal it at once when it has no tokens to prefill, or put it in
    /// flight for the wave step.
    ///
    /// `Err` hands the section back unstarted when its state cannot be claimed —
    /// see [`Self::begin_admitted`], which claims a turn's the same way and for
    /// the same reason. Any later failure is the section's own answer, sent to
    /// its caller here.
    pub(in super::super) fn begin_section(
        &mut self,
        s: QueuedSection,
    ) -> Result<(), Box<(QueuedSection, ConversationError)>> {
        let begun_at = Instant::now();
        if !s.tokens.is_empty() {
            // The prefix's state first, so the claim below takes it as the
            // slot's store instead of building one from zero.
            if let Some(start) = &s.start {
                if let Err(e) = self.stage_state(s.sequence_id.0, start) {
                    let _ = s.response_tx.send(Err(e));
                    return Ok(());
                }
            }
            if let Err(e) = self.model.claim_carried_state(s.sequence_id.0) {
                return Err(Box::new((s, ConversationError::Model(e))));
            }
        }
        let seal_block_from =
            match self.prepare_section_ingest(s.sequence_id, s.section_id, &s.prefix, &s.tokens) {
                Ok(from) => from,
                Err(e) => {
                    let _ = s.response_tx.send(Err(e));
                    return Ok(());
                }
            };
        if s.tokens.is_empty() {
            let result = self
                .perform_seal_and_write(
                    s.sequence_id,
                    s.sequence_id,
                    seal_block_from,
                    &SealAction::Section {
                        section_id: s.section_id,
                        tokens: Arc::new(s.tokens.to_vec()),
                        address: s.address,
                        debug_name: s.debug_name.clone(),
                        in_collection: s.in_collection,
                    },
                    None,
                )
                .and_then(|opt| {
                    opt.ok_or_else(|| {
                        ConversationError::Channel("ingest_section: seal returned None".into())
                    })
                });
            let _ = s.response_tx.send(result);
            return Ok(());
        }
        self.active_section_ingests.push(ActiveSectionIngest {
            sequence_id: s.sequence_id,
            section_id: s.section_id,
            tokens: s.tokens,
            offset: 0,
            seal_block_from,
            address: s.address,
            debug_name: s.debug_name,
            in_collection: s.in_collection,
            export_end: s.start.is_some(),
            band: s.band,
            response_tx: s.response_tx,
            error: None,
            timing: SectionTiming::new(s.queued_at, begun_at, Instant::now()),
        });
        Ok(())
    }

    pub(in super::super) fn prepare_section_ingest(
        &mut self,
        sequence_id: SequenceId,
        #[cfg_attr(not(feature = "context-dump"), allow(unused_variables))] section_id: SectionId,
        prefix: &[PrefixItem],
        #[cfg_attr(not(feature = "context-dump"), allow(unused_variables))] tokens: &TokenBuffer,
    ) -> Result<usize, ConversationError> {
        let prefix_section_ids = prefix_sections(prefix);
        // A section's blocks stay native until its seal: its build attends
        // them first.
        self.seal_section_natively(sequence_id);
        // 1. Defensive truncate.
        self.session
            .truncate_sequence_to_blocks(sequence_id.0, 0)
            .map_err(ConversationError::Model)?;
        if let Some(slot_state) = self.slot_projection_state.get_mut(&sequence_id) {
            slot_state.placed_pieces.clear();
        }
        // The slot's per-position state goes with its K/V. A reused slot starts
        // at position 0, and pages describing the previous occupant's positions
        // would put this section's first token at that prefix's end.
        self.model
            .reset_positional_state(sequence_id.0)
            .map_err(ConversationError::Model)?;

        // 2. Inject substrate-pinned prefix sections (zero-copy Arc clone), and
        //    leave the prefix's holes where they fall.
        let has_prefix = !prefix.is_empty();
        if has_prefix {
            if let Some(conversation) = self.slot_conversations.get(&sequence_id).cloned() {
                // Ensure every prefix section is hot-resident before reading its
                // sealed KV. A large collection ingest (e.g. the 93-tool catalog)
                // can evict earlier prelude sections from the hot tier, and
                // `section_sealed_of` is hot-only — without this, those sections
                // silently drop out of the prefix and the section being ingested
                // seals against a truncated context. `elevate_to_hot` is a no-op
                // when already resident, and re-lifts from cold otherwise.
                {
                    let backings = self.session.backings().to_vec();
                    let device = self.session.device().clone();
                    let no_turns: Vec<TurnKey> = Vec::new();
                    let lifted = warm_lift_ring(&mut self.warm_lift_ring).and_then(|ring| {
                        elevate_to_hot(
                            &conversation,
                            &backings,
                            &device,
                            ring,
                            &mut self.cold_load_stager,
                            &prefix_section_ids,
                            &no_turns,
                        )
                    });
                    if let Err(e) = lifted {
                        tracing::warn!(
                            "prepare_section_ingest: prefix elevate failed: {e} — \
                             some prefix sections may still be cold"
                        );
                    }
                }
                let n_layers = self.session.num_layers();
                let mut run = PrefixRun::new(n_layers);
                // Prefix sections whose stored layers disagree, retired below
                // once the read guard is gone — retiring takes the write lock.
                let mut skewed: Vec<(SectionId, StreamId)> = Vec::new();
                {
                    let view = conversation.read();
                    for &item in prefix {
                        let prefix_id = match item {
                            PrefixItem::Section(id) => id,
                            PrefixItem::Hole(positions) => {
                                // The sections before the hole go in as one
                                // run; the hole is left on the K/V and on the
                                // index, so what follows is read past it.
                                self.inject_prefix_run(sequence_id, &mut run)?;
                                self.session
                                    .push_sequence_position_gap(sequence_id.0, positions)
                                    .map_err(ConversationError::Model)?;
                                self.model
                                    .push_positional_gap(sequence_id.0, positions)
                                    .map_err(ConversationError::Model)?;
                                continue;
                            }
                        };
                        let Some(sealed) = view.section_sealed_of(prefix_id) else {
                            tracing::warn!(
                                "prepare_section_ingest: prefix section {:?} has no sealed substrate entry — skipping",
                                prefix_id
                            );
                            continue;
                        };
                        if sealed.len() != n_layers {
                            tracing::warn!(
                                "prepare_section_ingest: prefix section {:?} has {} layers, expected {} — skipping",
                                prefix_id,
                                sealed.len(),
                                n_layers
                            );
                            continue;
                        }
                        // Each section is cut to its own layers' common prefix
                        // BEFORE it is joined: a skewed section misaligns every
                        // section joined after it, and the cut the inject makes
                        // of the joined whole would take all of them with it.
                        let reconciled = reconcile_sealed_prefix(&sealed);
                        let source = reconciled.as_deref().unwrap_or(&sealed);
                        // Retired once, and only when the cut lost tokens —
                        // trimming an empty chunk one layer carried moves no
                        // position. Later borrows of the same copy, until the
                        // rebuild replaces it, say nothing more.
                        let stored_tokens = sealed.iter().map(|s| s.token_count).max().unwrap_or(0);
                        let lost = source[0].token_count < stored_tokens;
                        let retire = lost.then(|| view.section_stream(prefix_id));
                        if let Some(stream) =
                            retire.flatten().filter(|s| !view.is_section_tombstoned(*s))
                        {
                            tracing::warn!(
                                section = prefix_id.raw(),
                                stored_tokens,
                                injected_tokens = source[0].token_count,
                                "prepare_section_ingest: prefix section's layers describe \
                                 different token windows; borrowing their common prefix and \
                                 ending its generation so the next request prefills it fresh"
                            );
                            skewed.push((prefix_id, stream));
                        }
                        run.push(source);
                        // The one thing borrowing the chunks does not bring
                        // along. Pushed in prefix order, so the pages sit at
                        // the positions their K/V does.
                        self.push_prefix_section_index(
                            sequence_id,
                            prefix_id,
                            view.section_stream(prefix_id),
                            source[0].token_count,
                        );
                    }
                }
                for (section, stream) in skewed {
                    if let Err(e) = conversation.retire_section_generation(
                        stream,
                        "layers sealed with different token windows",
                    ) {
                        tracing::warn!(
                            "prepare_section_ingest: prefix section {} could not be retired: {e}",
                            section.raw()
                        );
                    }
                }
                self.inject_prefix_run(sequence_id, &mut run)?;
            }
        }

        // 3. Capture the seal's lower bound — and, in the same step, the index
        //    boundary that has to name the same position — before pushing the
        //    fresh writer chunk.
        let seal_block_from = self
            .session
            .sequence_block_count(sequence_id.0)
            .unwrap_or(0);
        self.begin_unit(sequence_id, "section");

        // 4. Push fresh writer chunk so section N doesn't alias the prefix's
        //    partial Arc-shared tail.  Not needed for empty prefix (slot is
        //    empty; ensure_for_batch_entries allocates the first chunk).
        if has_prefix {
            self.session
                .push_empty_writer_chunk(sequence_id.0)
                .map_err(ConversationError::Model)?;
        }

        #[cfg(feature = "context-dump")]
        if tracing::enabled!(
            target: "candle_conversation::scheduler::section_dump",
            tracing::Level::INFO,
        ) {
            let decoded = self
                .tokenizer
                .decode(&tokens[..], false)
                .unwrap_or_else(|e| format!("<decode error: {e}>"));
            tracing::info!(
                target: "candle_conversation::scheduler::section_dump",
                section_id = ?section_id,
                prefix_section_count = prefix_section_ids.len(),
                seal_block_from,
                token_count = tokens.len(),
                "=== section ingest dump ===\n{decoded}\n=== end section ==="
            );
        }

        Ok(seal_block_from)
    }
}
