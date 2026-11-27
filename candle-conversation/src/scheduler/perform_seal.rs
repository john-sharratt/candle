//! Writes a finished turn or section into the substrate: it snapshots the K/V,
//! captures the signatures and the recurrent state, and enqueues the records.
//!
//! The seal is the tail of its slot from its first new block to its last, and
//! one block range is resolved once so that the K/V slice, the token count and
//! the signature gather cannot disagree about where the turn begins. Recurrent
//! state is exported before the `Tokens` record is enqueued, so a snapshot lands
//! no later than the turn it describes, and a reload discards any snapshot newer
//! than the history it recovered. A pinned section is quantized only after the
//! turn record has committed, because the first projection reads its native form.
//! The index pages must cover the turn exactly, and a set of pages the seal knows
//! to be wrong is not persisted, since every later projection would refuse it.

use super::*;
use crate::files::is_member_stream_name;

impl Scheduler {
    /// `turn_content`, when `seal_action == SealAction::Turn`, carries
    /// the role / text / token IDs the substrate pins on the new turn
    /// entry so the on-disk record can be reconstructed later without
    /// re-tokenising.  Ignored for `SealAction::Section` and `None`.
    ///
    /// `owner` is the slot the seal is FILED under — its substrate target, its
    /// conversation and its projection state — and `seal_slot` the slot whose
    /// K/V, signatures and model state are sealed. They are the same slot for
    /// every seal but a turn group's case, which seals from its own view while
    /// the conversation the case belongs to is registered on the parent.
    pub(super) fn perform_seal_and_write(
        &mut self,
        owner: SequenceId,
        seal_slot: SequenceId,
        seal_block_from: usize,
        seal_action: &SealAction,
        turn_content: Option<TurnContent>,
    ) -> Result<Option<SealResult>, ConversationError> {
        // The substrate target (where a `SealAction::Turn` write
        // lands) is read from `slot_targets` rather than threaded
        // through the request — see [`Self::slot_targets`].
        let seal_target = self.slot_targets.get(&owner).copied();
        // A windowed creep prefill can leave layers in a later window one
        // empty writer chunk behind layers in an earlier one
        // (`BatchedInferenceSession::reconcile_block_counts`). This is the
        // seal that persists the slot's K/V to the substrate for every future
        // reload and every future conversation that borrows it, so a skew
        // reconciled here is fixed for good; left alone, it is what produces
        // "chunked decode layout diverged... on the first message of a new
        // conversation" on every subsequent replay of this exact record.
        self.session
            .reconcile_block_counts(seal_slot.0)
            .map_err(ConversationError::Model)?;
        let snapshot = self
            .session
            .snapshot_sequence(seal_slot.0)
            .map_err(ConversationError::Model)?;
        // Authoritative chunk total — `offset / CHUNK_SIZE` would
        // under-count when the slot's chunks include partials from
        // back-to-back section injection.  See
        // `BatchedInferenceSession::sequence_block_count`.
        let block_count = self.session.sequence_block_count(seal_slot.0).unwrap_or(0);
        // One resolution for the whole seal: the K/V slice, the token count and
        // the `sign(Q)` gather below all read `[block_from, block_to)`, so a
        // second derivation of either bound is a place they can disagree.
        let Some((block_from, block_to)) =
            resolve_seal_range(seal_block_from, block_count, snapshot.chunks.len())
        else {
            // Legitimate for empty section pins; for a dialogue turn it means
            // the turn's K/V range is gone and the seal would silently drop
            // the turn — always worth a loud trace.
            if matches!(seal_action, SealAction::Turn) {
                let off = self.session.sequence_offset(seal_slot.0);
                tracing::warn!(
                    "turn seal SKIPPED: empty block range from {} (slot holds \
                     {} blocks, snapshot {}) for slot {} (offset {:?}, layout {:?}) — \
                     the turn will NOT persist",
                    seal_block_from,
                    block_count,
                    snapshot.chunks.len(),
                    seal_slot,
                    off,
                    self.session
                        .provenance_chunk_layout(seal_slot.0, off.unwrap_or(0)),
                );
            }
            return Ok(None);
        };

        // **The turn's last native blocks are quantized here, before the
        // snapshot** (`docs/kv_quantize_in_the_gap.md` §5): the complete blocks
        // the gaps have not reached, the partial chunks and the writer chunk —
        // the turn takes no more writes, and its slot is emptied once sealed.
        // The snapshot below then holds every block quantized, with the signs
        // the gap captured from its Q. A section is left to its own seal
        // policy, which its build attends under first.
        if matches!(seal_action, SealAction::Turn) {
            let _g = profile::span("seal:gap");
            let t_gap = Instant::now();
            match self.session.gap_seal(seal_slot.0, block_from..block_to) {
                Ok(s) => self.kv_gap.sealed(&s),
                Err(e) => tracing::warn!(
                    target: "candle_conversation::scheduler::kv_gap",
                    "seal gap for slot {seal_slot} failed: {e} — its blocks are sealed as \
                     they stand"
                ),
            }
            self.kv_gap.spent(t_gap.elapsed());
        }

        // GPU-resident sealed sequences.  No CPU round-trip: the
        // substrate stores `Arc<Vec<SealedSequence>>` with the same
        // GPU `ChunkGid`s the slot just prefilled into.  The
        // chunks stay live via the substrate's Arc refs even after
        // the ingest slot is freed (RAII on `ChunkGid`).  Errors
        // abort the seal entirely — a missing snapshot would leave
        // the substrate entry without KV data.
        // Timed into the Sealing band's snapshot slot. It had been passing 0 for
        // this — the field existed and was never fed — so a full per-layer K/V
        // snapshot, once per sealed turn, was invisible on the dashboard and its
        // wall-clock fell through to the housekeeping remainder.
        let t_snap = Instant::now();
        // Snapshot the sealed prefix `[0, block_to)`, not the whole slot. The two
        // are the same for a settled slot, but during a windowed creep prefill the
        // layers advance incrementally: layer 0 (and its window) push an empty
        // (0-token) writer chunk for the next window ahead of the layers still
        // pending resume, so the slot's per-layer block counts differ by that one
        // trailing empty chunk (`per_layer=[626×7, 625×4]` for a section sealed
        // mid-creep; see `sequence_block_count`'s skew note). A full-slot
        // `snapshot_sequence_per_layer` walks that trailing phantom and its
        // `assert_sealed_layers_aligned` rightly refuses the uneven layers —
        // which returned `None` here and failed the seal ("seal returned None"),
        // stranding every conversation whose prime crossed the pressure point
        // that starts the creep windowing. `block_to` is `min(block_count,
        // chunk_len)` — the block count EVERY layer holds — so `[0, block_to)` is
        // the aligned common prefix, excludes the phantom, and still covers the
        // seal range `[block_from, block_to)` the slice below extracts.
        let sealed_per_layer = match self
            .session
            .snapshot_sequence_blocks(seal_slot.0, 0, block_to)
        {
            Ok(sealed) => std::sync::Arc::new(sealed),
            Err(e) => {
                tracing::error!(
                    "snapshot_sequence_blocks failed: seal_slot={} range=[0,{}) err={}",
                    seal_slot.0,
                    block_to,
                    e,
                );
                self.repair_section_if_window_divergence_confirmed(owner, seal_action, &e);
                return Ok(None);
            }
        };
        let snapshot_us = t_snap.elapsed().as_micros() as u64;

        let chunk_size = self.chunk_size;
        let turn_token_count: usize = snapshot
            .chunks
            .get(block_from..block_to)
            .map(|s| s.iter().map(|c| c.token_count as usize).sum())
            .unwrap_or(0);

        // The turn's reasoning span, read before `turn_content` moves into the
        // write below. `None` for every seal that is not a dialogue turn, and
        // for a turn that decoded no reasoning.
        let reasoning_span = turn_content.as_ref().and_then(|c| {
            c.layout.segments.iter().find_map(|s| match s {
                TurnSegment::Thinking { kv, .. } => *kv,
                _ => None,
            })
        });
        // The turn's own token count, from the buffer whose contract is that it
        // "must match the K/V chunk grid 1-1". It is the tie-breaker when the
        // chunk sum and the index walk disagree: equal to the chunk sum means
        // the K/V grid is intact and the index gained rows; equal to the index
        // means the grid lost some and this invariant is already broken here,
        // upstream of anything the index did.
        let turn_grid_tokens = turn_content.as_ref().map(|c| c.token_ids.len());

        // Capture the whole turn's wide per-token `sign(Q)` NOW — before
        // `record_turn` (below) detaches the sealed KV. All heads / all layers,
        // un-folded: from a block's R16 Q while it has one, and from the signs
        // the gap captured before it converted the block otherwise.
        // This is the LIVE turn seal (dialogue + code-read roundtrip turns) — count
        // it for the GUI's sealing phase, timing the dominant sig-gather cost.
        let t_sig = Instant::now();
        let wide_sigs = {
            let _g = profile::span("seal:wide_sigs");
            self.gather_wide_sigs(seal_slot, (block_from, block_to))
        };
        // **The signatures describe what is RETRIEVABLE, and the reasoning is
        // not.** A turn's reasoning is attendable from exactly one subsequent
        // projection, by position, and never by recall — every later projection
        // windows it out of the K/V. Leaving its tokens in the signature run lets
        // the belief scan score a turn on content the projection then removes,
        // which is a retrieval that returns nothing usable.
        let wide_sigs = match reasoning_span {
            Some(span) => sigs_without_span(wide_sigs, span.range()),
            None => wide_sigs,
        };
        self.wave_stats
            .add_seal(snapshot_us, t_sig.elapsed().as_micros() as u64, 0);

        // KV-zero check: scan the PARENT slot at the seal range, right before the
        // chunks are persisted — i.e. exactly what gets written to the substrate.
        #[cfg(feature = "kv-zero-check")]
        {
            let layers: Vec<usize> = (0..self.session.num_layers()).collect();
            let seq_off = self.session.sequence_offset(seal_slot.0).unwrap_or(0);
            let layout = self.session.provenance_chunk_layout(seal_slot.0, seq_off);
            // The sealed range [block_from, block_to) is entirely this turn's own
            // content, so the boundary is the real position where block_from begins.
            let turn_start = layout.get(block_from).map(|l| l.2).unwrap_or(0);
            if let Ok(dump) = self.session.gather_r16_kv_provenance_layers(
                seal_slot.0,
                &layers,
                Some((block_from, block_to)),
            ) {
                kv_zero_check::scan_gathered(
                    "substrate-write-seal",
                    seal_slot.0,
                    &dump,
                    &layout,
                    self.session.n_kv_head(),
                    self.session.head_dim(),
                    turn_start,
                );
            }
        }

        // Apply the substrate write on the workspace conversation
        // bound to this slot.
        let conversation = self
            .slot_conversations
            .get(&owner)
            .cloned()
            .ok_or_else(|| {
                ConversationError::Channel(format!(
                    "perform_seal_and_write: no conversation registered for slot {owner}"
                ))
            })?;
        // The substrate TurnIndex a `SealAction::Turn` records — surfaced on
        // the SealResult so per-turn persists key by the exact index instead
        // of racing `turn_count - 1` against the async summariser.
        let mut recorded_turn_index: Option<u32> = None;
        match seal_action {
            SealAction::Turn => {
                let target = seal_target.ok_or_else(|| {
                    ConversationError::Channel("SealAction::Turn missing seal_target".into())
                })?;
                let TurnContent {
                    role,
                    tags,
                    layout,
                    token_ids,
                } = turn_content.unwrap_or_default();
                let delta_gpu = slice_per_layer_sealed(&sealed_per_layer, block_from, block_to)?;
                // Snapshot what the resume path needs before the substrate
                // consumes `delta_gpu` / `token_ids` (§16.12 seal-time gather).
                let persist_token_ids: Vec<u32> = token_ids[..].to_vec();
                debug_assert_eq!(
                    persist_token_ids.len(),
                    turn_token_count,
                    "persisted token_ids must align 1:1 with the K/V chunk grid \
                     (off-by-one usually means an unforwarded token slipped through)"
                );

                // The turn is sealed as one indivisible K/V block; its
                // segment-vector `layout` carries the per-half text + spans the
                // sidebar and compressor read without re-tokenising.
                let write = TurnPartWrite {
                    layout,
                    token_ids,
                    token_count: turn_token_count,
                    tags,
                    block_start: block_from as u64,
                    block_end: block_to as u64,
                    sealed_gpu: Some(Arc::new(delta_gpu)),
                };
                let idx = {
                    let _g = profile::span("seal:record_turn");
                    conversation
                        .record_turn(target.timeline, role, write, |seqs| Ok(seqs.to_vec()))
                        .map_err(ConversationError::Model)?
                };
                recorded_turn_index = Some(idx.0);

                // Drain pending section quantizations.  Every section in
                // the queue was ingested earlier in this conversation
                // build with its native (prefill-output) K/V installed in
                // `substrate.section.hot`, and every reader up to this
                // point — this turn's projection's top-k selection, its
                // prefill kernel attending back over the injected
                // sections — has consumed that native form.  This
                // turn-seal boundary is the first moment where no
                // in-flight operation depends on those native bytes any
                // more, which makes it the only safe place to swap the
                // residence's hot to its quantized form.  Synchronous,
                // on the main scheduler thread, batched across all
                // pending sections so the per-launch kernel overhead
                // amortises.  See the bisect history in
                // `tests/section_quantize_real_model.rs` for why earlier
                // / asynchronous attempts corrupt sysprompt K/V.
                if !self.pending_section_quantize.is_empty() {
                    let _g = profile::span("seal:section_quantize");
                    if self.session.compression_policy().is_some() {
                        // Non-fatal: the turn's decl/tokens/chunks were already
                        // recorded above and do not depend on section
                        // requantization — a quantize failure (e.g. a model
                        // whose head_dim the palette4 path doesn't support)
                        // must not abort the turn seal.
                        if let Err(e) = self.quantize_pending_sections(&conversation) {
                            // The failed batch was this substrate's alone and has
                            // left the queue; every other substrate's stays.
                            tracing::warn!(
                                "turn-seal section quantize drain failed (seal continues): {e}"
                            );
                        }
                    } else {
                        self.pending_section_quantize.clear();
                    }
                }
                // Persist the wide per-token sign(Q) captured above (pre-detach) as the
                // turn's `WideQSig` record — continuous per-token shape, keyed to the
                // same turn stream.
                if !wide_sigs.is_empty() {
                    let stream_id = turn_stream_id(target.timeline.raw(), idx.0);
                    // Stamped with the same fold that produced the bits above, so
                    // a later build that folds differently refuses to score them
                    // instead of comparing incomparable bits.
                    conversation.enqueue_wide_q_sigs(
                        stream_id,
                        encode_wide_sigs_with(&wide_sigs, self.prov_fold),
                    );
                }
                // The QSA index rows describing the K/V just sealed. Taken here,
                // in the one place every turn is written, so a turn cannot be
                // sealed down a path that forgets them. It lived on a completion
                // path only one kind of turn reached, and every turn that took
                // any other route sealed with K/V and no rows at all.
                //
                // **Sealed by WIDTH, not from a position.** A turn's rows are
                // not all in the live tail: a mid-decode reprojection re-injects
                // its user half from cache and pushes it as a page, so the turn
                // spans a page boundary. `seal_positional_tail_span` walks back
                // over whole pages until the width is covered; sealing the tail
                // alone drops the user's own message, after which the model
                // decodes a question it cannot attend to and answers a different
                // one it invented.
                //
                // Each page keeps its own ragged width, so they travel as an
                // ordered list ([`index_pages`]) and are pushed back in that
                // order. Resident AND durable: the map answers this process's
                // projections, the record is what makes the turn borrowable
                // after a restart.
                match self
                    .model
                    .seal_positional_tail_span(seal_slot.0, turn_token_count)
                {
                    Ok(pages) if !pages.is_empty() => {
                        // **The pages must cover the turn EXACTLY.**
                        //
                        // `seal_positional_tail_span` walks back over WHOLE
                        // pages until the width is covered, so if the turn's own
                        // rows begin partway into a page the walk takes that
                        // page entire and the set covers more than the turn.
                        // Storing it makes the turn's index claim rows for
                        // tokens before it, and every projection that borrows
                        // the turn then carries an index wider than the K/V —
                        // measured as a constant "index N tokens past" on every
                        // ingest slot.
                        //
                        // Reported with both numbers because the overshoot is
                        // exactly one page's width, and naming that width is
                        // what identifies the piece the turn's rows share a page
                        // with.
                        let covered: usize = pages.iter().map(|(w, _)| *w).sum();
                        if covered != turn_token_count {
                            // **Both sides of the disagreement, so the next
                            // reader does not have to guess which one is
                            // wrong.** `turn_token_count` is the K/V's answer
                            // (summed chunk widths over `[block_from,
                            // block_to)`); the walk's `covered` is the index's.
                            // `reasoning_span` is here because the observed
                            // shortfall is a constant 7 tokens sitting in a page
                            // of its own between the turn's prefill and its
                            // answer — exactly where the reasoning cut falls —
                            // and its width is what says whether that page is
                            // the reasoning or something the index gained
                            // without K/V.
                            tracing::warn!(
                                turn = idx.0,
                                turn_token_count,
                                covered,
                                block_from,
                                block_to,
                                chunk_widths = ?snapshot
                                    .chunks
                                    .get(block_from..block_to)
                                    .map(|s| s.iter().map(|c| c.token_count).collect::<Vec<_>>())
                                    .unwrap_or_default(),
                                reasoning = ?reasoning_span.map(|s| (s.offset, s.len)),
                                turn_grid_tokens = ?turn_grid_tokens,
                                n_pages = pages.len(),
                                widths = ?pages.iter().map(|(w, _)| *w).collect::<Vec<_>>(),
                                "turn index pages cover {} token(s) {} the turn itself — the \
                                 walk-back crossed a page the turn does not start on",
                                covered.abs_diff(turn_token_count),
                                if covered > turn_token_count { "more than" } else { "less than" }
                            );
                        }
                        // **Do not persist an index the seal already knows is
                        // wrong.** A payload whose pages do not sum to the turn's
                        // own K/V width cannot describe it, so every later
                        // projection that selects the turn refuses to window it —
                        // and because the payload goes into the redo log, that
                        // refusal outlives the process with no repair path. Storing
                        // nothing leaves the turn in the same state as a model that
                        // seals no pages at all, which `turn_sealed_without_thinking`
                        // handles by injecting it whole. The warning above already
                        // names both numbers.
                        //
                        // The substrate holds the bytes until the write lands and
                        // the log after it; a read of them lands in the index page
                        // cache, which keeps them while a slot places the turn.
                        if covered == turn_token_count {
                            conversation.enqueue_index_page(
                                turn_stream_id(target.timeline.raw(), idx.0),
                                index_pages::encode(&pages),
                            );
                        }
                    }
                    Ok(_) => {
                        if self.model.carries_positional_state() {
                            tracing::warn!(
                                turn = idx.0,
                                turn_token_count,
                                "turn sealed with no index pages — a projection borrowing its \
                                 K/V will select against a prefix it never indexed"
                            );
                        }
                    }
                    Err(e) => tracing::warn!(
                        turn = idx.0,
                        "turn index pages not sealed ({e}); a projection borrowing this turn \
                         will select against a prefix it never indexed"
                    ),
                }
                // Persist the sequence's recurrent state, if it carries any.
                //
                // **Before the `Tokens` enqueue, deliberately.** The snapshot
                // must be durable no later than the turn it describes.
                // `turn_index` carries the rule for the torn case: reload
                // discards a snapshot NEWER than the last recovered turn and
                // recomputes. Writing first can therefore leave a snapshot for
                // a turn whose records never landed — handled — but never a
                // turn whose snapshot never landed, which would be silent
                // staleness: a resume that installs a state one turn behind its
                // K/V, reads fluently, and is wrong.
                //
                // `export_recurrent` returns `None` for a model with no such
                // state, so this site is model-agnostic and Qwen3-30B pays a
                // branch.
                //
                // **Asked BEFORE the export, not after.** `enqueue_recurrent_snapshot`
                // drops an ephemeral timeline's payload, and it has to — it is the
                // invariant, and it guards every caller. But reaching it means the
                // ~65 MB device→host export and the encode have already happened,
                // for bytes that are discarded on arrival. A stuffed calibration
                // group seals 24 turns, so that was 24 exports per group and 2,998
                // across a full build. The state is fixed-size per sequence, so a
                // 19-token question costs exactly what a full trajectory does.
                let _export_span = profile::span("seal:export_recurrent");
                let ephemeral = conversation.read().is_timeline_transient(target.timeline);
                let t_export = Instant::now();
                // The model's other recurrence, in its own encoding. Exported at
                // the same instant as `layers` — one seal, one state — so the
                // two can never describe different token counts. A model with
                // none returns `None` and the blob is empty.
                //
                // Skipped for an ephemeral timeline alongside `layers`, and for
                // the same reason: the payload is dropped on arrival, so the
                // export is pure cost.
                let aux = if ephemeral {
                    Vec::new()
                } else {
                    match self.model.export_aux_state(seal_slot.0) {
                        Ok(blob) => blob.unwrap_or_default(),
                        Err(e) => {
                            tracing::warn!(
                                "auxiliary state export failed for turn {} (the turn is \
                                 sealed; resuming it will rebuild that state by prefill): {}",
                                idx.0,
                                e,
                            );
                            Vec::new()
                        }
                    }
                };
                let exported = if ephemeral {
                    Ok(None)
                } else {
                    let _g = profile::span("seal:export_readback");
                    self.model.export_recurrent(seal_slot.0)
                };
                match exported {
                    Ok(Some((schedule_hash, layers))) => {
                        let payload = SnapshotPayload {
                            timeline_id: target.timeline.raw(),
                            turn_index: idx.0,
                            schedule_hash,
                            layers: layers.into_iter().map(SnapshotLayer::from).collect(),
                            aux,
                        };
                        // Timed around the export, which is the whole of what the
                        // seal pays synchronously: the encode and the append are
                        // the off-thread writer's.
                        use std::sync::atomic::Ordering::Relaxed;
                        SNAPSHOT_EXPORT_US
                            .fetch_add(t_export.elapsed().as_micros() as u64, Relaxed);
                        SNAPSHOT_BYTES.fetch_add(payload.encoded_len() as u64, Relaxed);
                        SNAPSHOT_COUNT.fetch_add(1, Relaxed);
                        self.sealed_snapshots
                            .insert(seal_slot, (target.timeline, idx.0));
                        conversation.enqueue_recurrent_snapshot(target.timeline, payload);
                    }
                    // **`None` from a model that declares recurrent state is a
                    // contradiction, not a fast path.**
                    //
                    // The two readings of `None` are "this architecture has no
                    // such state" and "this architecture has state it cannot
                    // export", and they are indistinguishable here while having
                    // opposite consequences: the first costs nothing, the second
                    // silently produces a conversation that resumes with no
                    // memory of its history and reads perfectly fluently. That
                    // is the exact defect this whole path exists to remove, so
                    // the model's own `carries_recurrent_state` is used to tell
                    // them apart.
                    //
                    // A model reaching this arm has declared state and returned
                    // none for a sequence that has just decoded a turn — the
                    // only innocent case, a slot that never ran a wave, cannot
                    // be sealing. It is a wiring gap in that architecture, and
                    // it is worth a warning on every seal rather than a silence
                    // that is only discovered by noticing the model has
                    // forgotten.
                    //
                    // **Only for a snapshot that was asked for.** An ephemeral
                    // timeline's export is skipped above and arrives here as the
                    // same `None`; its snapshot is dropped on arrival by design,
                    // so nothing is missing — and every calibration seal of a
                    // recurrent model is one of these.
                    Ok(None)
                        if missing_recurrent_hook(
                            ephemeral,
                            self.model.carries_recurrent_state(),
                        ) =>
                    {
                        tracing::warn!(
                            "turn {} sealed with NO recurrent snapshot, but this model \
                             declares it carries recurrent state — `export_recurrent` \
                             returned None. The turn's K/V and text are committed, but \
                             resuming this conversation will start from the \
                             sequence-start state: it will read fluently and have \
                             forgotten. This is a missing model hook, not a data loss.",
                            idx.0,
                        );
                    }
                    Ok(None) => {}
                    Err(e) => {
                        // Not fatal to the seal — the turn's K/V and text are
                        // already committed — but it means this conversation
                        // cannot be resumed from here, so it must be visible.
                        tracing::warn!(
                            "recurrent snapshot export failed for turn {} (the turn is \
                             sealed, but resuming it will recompute from zeros): {}",
                            idx.0,
                            e,
                        );
                    }
                }
                // Persist the turn's `Tokens` record — tiny and
                // load-bearing for substrate reconstruction. Enqueued onto
                // the off-thread writer (not written under the persistence
                // lock) so a compaction holding that lock can't stall the
                // seal; the heavy `Chunks` records + matching `Commit` are
                // likewise deferred to the persistence thread (trigger below).
                {
                    use turn_stream_id;
                    let stream_id = turn_stream_id(target.timeline.raw(), idx.0);
                    conversation.enqueue_tokens(stream_id, persist_token_ids.clone());
                }
                // Wake the persistence thread so it drains the new
                // turn (hot→warm migrate, warm→cold redo-log write,
                // group fsync) without waiting for its 5 s tick.
                self.persist_triggers.fire();
                // Wake the summariser thread so the
                // freshly-pending Normal turn gets absorbed into the immutable
                // summary forest on its
                // next pass instead of waiting up to 250 ms for the periodic tick.
                self.summariser_trigger.fire();
                // Trim the slot's live-prefill capture cache down to
                // the working set this projection actually touched.
                // Without trimming, each turn would append at least
                // one new boundary-run entry (the trailing
                // `user_start_current`) and pin GPU arenas via its
                // `Arc<Vec<SealedSequence>>` values indefinitely.
                // Retaining the active keys keeps the boundary K/V
                // hot for next-turn reprojection while bounding the
                // cache at ~one projection's worth of entries — see
                // `SlotState::trim_post_turn`.
                if let Some(state) = self.slot_projection_state.get_mut(&owner) {
                    state.trim_post_turn();
                }
            }
            SealAction::Section {
                section_id,
                tokens,
                address,
                debug_name,
                seal,
            } => {
                let delta_gpu = slice_per_layer_sealed(&sealed_per_layer, block_from, block_to)?;
                let stream_id = section_stream_id(*address);
                let policy_active = self.session.compression_policy().is_some();
                // The pending-quantize flag is raised inside the edit that
                // installs the residence, so no version publishes the section
                // hot without it and the persistence thread never writes the
                // interim native bytes.
                conversation.update(|substrate| {
                    let residence = substrate.set_section_full(
                        *section_id,
                        stream_id,
                        turn_token_count,
                        delta_gpu.clone(),
                        Arc::clone(tokens),
                    );
                    if policy_active {
                        substrate.mark_section_pending_quantize(residence);
                    }
                });
                // Queue the section for quantization at the next safe
                // boundary — the next `SealAction::Turn` on this substrate,
                // whether the section was ingested during a conversation's
                // build or mid-conversation.  Until
                // the drain runs, `mark_section_pending_quantize` keeps
                // the persistence thread from writing the interim
                // native bytes to disk — otherwise the cold tier would
                // hold a form that diverges from the post-drain hot
                // tier, and a daemon restart would resume in an
                // inconsistent state.
                //
                // Sections **must** be left in their native form for the
                // in-flight build: the first turn's projection reads
                // `substrate.section.hot` to pick top-k tools, the user
                // prefill kernel attends back over those injected K/V
                // to compute the prompt's own hidden states.  If we
                // quantize before that attention runs, the small Q4_KS
                // K error compounds across 48 layers into corrupted
                // prompt K/V — and everything downstream reads from a
                // poisoned cache.  See the bisect history in
                // `tests/section_quantize_real_model.rs`.
                if policy_active {
                    self.pending_section_quantize.push(PendingSectionQuantize {
                        substrate: conversation.substrate_ref(),
                        section_id: *section_id,
                        seal: *seal,
                    });
                }
                // **A shared section is quantized at its own seal**
                // (`docs/shared_journal_sections.md` §4.2). Nothing reads it
                // yet — the section continuing it waits for this seal to settle
                // — and every later reader, in this conversation or one that
                // restores it, then attends over the one form the substrate
                // keeps: a hit and a miss read the same bytes. It is persisted
                // at once, too, rather than at a turn's seal that a run sealed
                // ahead of any conversation never has.
                if policy_active && is_shared_stream_name(debug_name) {
                    let (now, keep): (Vec<_>, Vec<_>) = take(&mut self.pending_section_quantize)
                        .into_iter()
                        .partition(|p| {
                            p.section_id == *section_id && p.substrate.is(&conversation)
                        });
                    self.pending_section_quantize = keep;
                    if let Err(e) = self.quantize_section_batch(&conversation, now, false) {
                        tracing::warn!(
                            section = %debug_name,
                            "a shared section's quantize at its seal failed: {e}"
                        );
                    }
                }
                // A transient section — a throwaway conversation's frame — is
                // never read back from disk, so it declares no stream and writes
                // no tokens; its residence is already `no_cold_persist`, so its
                // chunks stay off the cold tier too. See
                // `Substrate::mark_section_transient`.
                if !conversation.is_section_transient(*section_id) {
                    // Declare the section stream in the redo log so the
                    // manifest knows the (address, debug_name) before any
                    // chunks land.  Mirrors record_turn's StreamDecl write.
                    if let Err(e) = conversation.declare_section_stream(*address, debug_name) {
                        tracing::warn!("declare section stream failed: {e}");
                    }
                    // Persist the section's token ids (off-thread writer, so the
                    // seal never blocks on the persistence lock), then fire the
                    // persistence trigger so the chunks land on disk in the next
                    // pass.
                    conversation.enqueue_tokens(stream_id, tokens.to_vec());
                    // A member of the knowledge base is retrieved by its
                    // signatures, as a turn is, so its seal writes them under
                    // its own stream; no other section is retrieved, and none
                    // carries any.
                    if is_member_stream_name(debug_name) && !wide_sigs.is_empty() {
                        conversation.enqueue_wide_q_sigs(
                            stream_id,
                            encode_wide_sigs_with(&wide_sigs, self.prov_fold),
                        );
                    }
                    self.persist_triggers.fire();
                }
            }
            SealAction::None => unreachable!("filtered above"),
            SealAction::GroupCase(_) => unreachable!(
                "a turn group's case is sealed by its group, which passes \
                 SealAction::Turn with the case's own content"
            ),
            SealAction::CompressionPass { .. } => {
                unreachable!("compression passes complete in cleanup_finished, not here")
            }
            SealAction::CompressionTurn { .. } => {
                unreachable!(
                    "compression turns seal in promote_finished_prefills_to_decodes, not here"
                )
            }
        }

        Ok(Some(SealResult {
            block_count,
            block_from,
            block_to,
            turn_token_count,
            chunk_size,
            turn_index: recorded_turn_index,
            end_state: None,
        }))
    }
}
