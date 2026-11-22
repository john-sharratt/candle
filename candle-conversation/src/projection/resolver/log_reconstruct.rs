//! Rebuilds the in-RAM substrate from the redo log when the daemon starts.
//!
//! Restart is cold-only: token history and each turn's stream metadata come back,
//! while key-value state stays on disk and is elevated on demand. Each turn is
//! isolated, so one corrupt turn is dropped without poisoning the reload, and it
//! becomes an empty placeholder so later turns keep their persisted indices.
//! Regenerable ingest timelines are tombstoned for re-ingest, while dialogue keeps
//! every turn it can still show.

use super::*;

impl Conversation {
    /// Rebuild the in-RAM [`Substrate`] from the persistence redo log — the
    /// §5.6 / §16.12 substrate-reload path run on daemon restart.
    ///
    /// **Cold-only restart.** The substrate is, by design, the on-disk redo
    /// log; warm (RAM) and hot (VRAM) tiers belong to the inference engine
    /// and are demand-populated. Reload therefore:
    /// - Walks every persisted turn stream in `(timeline, turn_index)`
    ///   order.
    /// - Replays **tokens** (for text history) into the in-RAM substrate —
    ///   small (RAM-resident) and load-bearing for text display.
    /// - Records each turn's stream metadata (`block_start`/`block_end`,
    ///   role, timeline) so projection knows the turn exists and where its
    ///   KV lives on disk.
    /// - **Does not materialize KV into VRAM.** Each restored turn's
    ///   `sealed` is an empty `Vec<SealedSequence>` — a "cold" marker. The
    ///   inject path materializes through the warm pool on demand (see
    ///   the engine's `ensure_hot` orchestrator).
    ///
    /// Returns the number of turns restored.
    pub fn reconstruct_from_log(
        &self,
        n_layers: usize,
        progress: Option<&dyn Fn(usize, usize)>,
    ) -> candle::Result<usize> {
        // Substrate's per-stream / per-timeline state was populated
        // in one walker pass during `SubstratePersistence::open_in_with_substrate`
        // — no mirror step needed here.  This pass replays turn-decl
        // records into the substrate's per-turn KV residence slots
        // (the cold-load setup that demands knowing layer count) and
        // then runs the post-reload sweeps for the summary tree.
        let t_reconstruct = Instant::now();
        let t_decls = Instant::now();
        let decls = {
            let substrate = self.read();
            crate::persistence::resume::recovered_turn_decls(&substrate)
        };
        let total = decls.len();
        let decls_build_ms = t_decls.elapsed().as_millis() as u64;
        tracing::info!(
            turns = total,
            decls_build_ms,
            "substrate reconstruct: begin replay loop",
        );
        let mut restored = 0usize;
        let mut skipped_corrupt = 0usize;
        let mut pending_remote = 0usize;
        // What each pending remote turn lacks — see `PendingCensus`.
        let mut pending = PendingCensus::default();
        // Startup integrity repair (`persistence::integrity`): completeness is
        // judged here — and only here — because reconstruct is the one moment
        // with no live writer, so on-disk state is final and a half-written
        // conversation cannot be mistaken for a damaged one.
        let corpus_has_kv = self.read().any_stream_has_chunks();
        let mut repaired_integrity = 0usize;
        let mut integrity_dropped: HashSet<TimelineId> = HashSet::new();
        for (i, mut decl) in decls.into_iter().enumerate() {
            // Report turns processed so far (restored + skipped) so the daemon's
            // loading bar advances steadily even across corrupt-turn skips.
            if let Some(p) = progress {
                p(i, total);
            }
            // A position below this turn that no decl holds is a decl lost in
            // a crash (`recovered_turn_decls`): an empty placeholder holds it,
            // so this turn lands at the index its streams resolve by.
            self.fill_turn_gap(&decl);
            // A turn already dropped by a prior `drop_turn` tombstone is dead —
            // restore it as an EMPTY PLACEHOLDER (no recover attempt, no warn)
            // instead of skipping. The placeholder matters: `restore_turn`
            // assigns indices densely, so a plain skip would renumber every
            // subsequent turn of this LIVE timeline away from the persisted
            // stream ids they are keyed by (wide-Q sigs, seams, residences),
            // and the next appended turn would collide with an existing turn's
            // stream records. A zero-token placeholder occupies the original
            // index — never selected (no content), never materialisable (no
            // cold refs) — keeping every later turn aligned.
            if let Some(tl) = TimelineId::from_raw(decl.timeline_id) {
                // **A tombstoned timeline is dead whole, and a runtime reload
                // still sees its streams**: the in-RAM stream index is
                // insert-only, while the compactor drops the timeline's records
                // from disk. Recovering it read segments compaction had already
                // unlinked, and reported a deleted conversation as corrupt — on
                // `.5`, 85 turns of conversations deleted through the API, each
                // then given a repair tombstone (2026-10-18). Replay at open
                // leaves a tombstoned timeline's records inert; so does this.
                if self.read().is_tombstoned(tl) {
                    continue;
                }
                if self.read().is_turn_tombstoned(tl, decl.turn_index) {
                    self.restore_placeholder(tl, &decl);
                    continue;
                }
            }
            // Another member's turn still arriving is pending, not corrupt: a
            // gap holds its position, the rest of it is wanted by the next
            // catch-up, and the runtime adopts it into the gap once it is all
            // here. Judging it corrupt would tombstone a conversation that is
            // whole on the member that wrote it.
            let remote = self.is_remote_turn(&decl);
            if remote && !self.is_turn_complete(&decl, n_layers) {
                pending_remote += 1;
                let (tokens, held) = {
                    let read = self.read();
                    read.stream_of(turn_stream_id(decl.timeline_id, decl.turn_index))
                        .map_or((false, 0), |s| (s.tokens.is_some(), s.chunks.len()))
                };
                let wanted = n_layers * decl.block_end.saturating_sub(decl.block_start) as usize;
                pending.note(decl.timeline_id, decl.turn_index, tokens, held, wanted);
                self.restore_pending_turn(&decl);
                continue;
            }
            // Per-turn fault isolation.  A single turn whose
            // persisted state is inconsistent (e.g. a daemon was
            // killed between chunk writes and the final TurnDecl
            // seal update, leaving a TurnDecl with
            // `block_end == block_start` but real chunks on disk)
            // should not poison the entire reload — every other turn
            // is independent.  Log the failure and tombstone the
            // corrupt turn's timeline so the next reload doesn't
            // re-encounter it.
            let substrate_read = self.read();
            let recover_result = self.persistence.call(|p| {
                // **A distilled timeline's shed records are absent, not
                // corrupt.** Its K/V was declared disposable when it was marked,
                // and the compactor drops the chunks — and, provenance-only, the
                // tokens — while the insert-only stream index keeps naming them.
                // Replay at open finds those records gone; a runtime reload must
                // read them the same way. Restored with the cold references they
                // name, the turns were chosen by every projection, failed
                // `cold-load found no chunks`, and were withdrawn — 18,724 times
                // in 53 minutes on `.5` after one reload (2026-10-18). The live
                // path clears the same references when it marks the timeline
                // (`Substrate::release_distilled_kv`).
                let distilled = TimelineId::from_raw(decl.timeline_id)
                    .is_some_and(|tl| substrate_read.distill_mode(tl).is_some());
                // Metadata only — token ids, signatures, and the chunk-index
                // token count. The turn's KV payload bytes stay on disk; a
                // later projection reads them via the cold→hot elevation
                // (`recover_turn_grid`).
                let r = crate::persistence::resume::recover_turn_meta(p, &substrate_read, &decl)
                    .or_else(|e| {
                        if distilled {
                            Ok(RecoveredTurnMeta {
                                token_ids: Vec::new(),
                                token_count: 0,
                            })
                        } else {
                            Err(e)
                        }
                    });
                let cr = if distilled {
                    Ok(None)
                } else if r.is_ok() {
                    crate::persistence::resume::recover_turn_cold_refs(
                        &substrate_read,
                        &decl,
                        n_layers,
                    )
                } else {
                    Ok(Default::default())
                };
                r.and_then(|recovered| cr.map(|cold_refs| (recovered, cold_refs)))
            });
            drop(substrate_read);
            let (recovered, cold_refs) = match recover_result {
                Ok(pair) => pair,
                // Another member's turn that cannot be read here is this
                // member's bytes at fault, not the conversation: it is held as
                // a gap until it is fetched again, never tombstoned — a
                // tombstone here would refuse the copy that repairs it.
                Err(e) if remote => {
                    tracing::warn!(
                        timeline_id = decl.timeline_id,
                        turn_index = decl.turn_index,
                        "another member's turn is unreadable here and waits to be fetched \
                         again: {e}",
                    );
                    pending_remote += 1;
                    pending.note_unreadable(decl.timeline_id, decl.turn_index);
                    self.restore_pending_turn(&decl);
                    continue;
                }
                Err(e) => {
                    tracing::warn!(
                        timeline_id = decl.timeline_id,
                        turn_index = decl.turn_index,
                        "skipping corrupt turn during substrate reload: {e}",
                    );
                    skipped_corrupt += 1;
                    if let Some(timeline) = TimelineId::from_raw(decl.timeline_id) {
                        // The corrupt-turn policy is the owning layer's (from the
                        // projection schema); default `DropConversation` when the
                        // layer is unknown.
                        let policy = LayerId::from_raw(decl.layer_id)
                            .map(|l| self.read().corrupt_turn_policy_for(l))
                            .unwrap_or_default();
                        // Durable mark carries the reason so the tombstone records
                        // WHY the turn/timeline was dropped, not just that it was.
                        let reason = format!("corrupt reload (turn {}): {e}", decl.turn_index);
                        match policy {
                            CorruptTurnPolicy::DropConversation => {
                                // Ingest layers (code_read / repo_map): one file is
                                // one timeline, so tombstoning the timeline hands the
                                // file back to the background refresh to re-ingest
                                // (absent from `content_sha256` → flagged changed →
                                // `process_one_file` rebuilds it).
                                self.update(|s| s.tombstone_timeline(timeline));
                                if !self.read_only {
                                    match self.persistence.call(|p| {
                                        p.write_repair_tombstone(timeline.raw(), &reason)
                                    }) {
                                        Ok(_) => tracing::debug!(
                                            timeline_id = decl.timeline_id,
                                            turn_index = decl.turn_index,
                                            reason = %reason,
                                            "tombstoned corrupt turn's TIMELINE (drop_conversation) \
                                             — code_read will re-ingest the file",
                                        ),
                                        Err(e) => tracing::warn!(
                                            timeline_id = decl.timeline_id,
                                            turn_index = decl.turn_index,
                                            "failed to durably tombstone corrupt turn's timeline: {e}",
                                        ),
                                    }
                                }
                            }
                            CorruptTurnPolicy::DropTurn => {
                                // Dialogue: drop only the corrupt turn, keeping the
                                // rest of the conversation live.
                                let mark =
                                    self.update(|s| s.tombstone_turn(timeline, decl.turn_index));
                                if !self.read_only {
                                    match self
                                        .persistence
                                        .call(|p| p.write_repair_turn_tombstone(mark, &reason))
                                    {
                                        Ok(_) => tracing::debug!(
                                            timeline_id = decl.timeline_id,
                                            turn_index = decl.turn_index,
                                            reason = %reason,
                                            "tombstoned corrupt TURN (drop_turn) — the rest of the \
                                             conversation is kept live",
                                        ),
                                        Err(e) => tracing::warn!(
                                            timeline_id = decl.timeline_id,
                                            turn_index = decl.turn_index,
                                            "failed to durably tombstone corrupt turn: {e}",
                                        ),
                                    }
                                }
                                // Occupy the dropped turn's index with an empty
                                // placeholder so the timeline's LATER turns keep
                                // their persisted indices (`restore_turn` assigns
                                // densely; a bare skip renumbers them away from
                                // their stream ids and the next appended turn
                                // collides with an existing turn's records).
                                self.restore_placeholder(timeline, &decl);
                            }
                        }
                    }
                    continue;
                }
            };
            let timeline = TimelineId::from_raw(decl.timeline_id).ok_or_else(|| {
                candle::Error::Msg("reconstruct: turn has zero timeline_id".into())
            })?;
            // A timeline already tombstoned by integrity repair this pass:
            // its remaining turns are dead with it — skip the restore work.
            if integrity_dropped.contains(&timeline) {
                continue;
            }
            // Completeness verdict — the CRC guards bytes that were written;
            // this guards records that were never written at all (the async
            // writer's `Tokens`/sig records can be lost to a hard kill while
            // the decl and KV chunks landed durably through other channels).
            let verdict = {
                let read = self.read();
                classify_turn(
                    read.distill_mode(timeline),
                    decl.block_end > decl.block_start,
                    !recovered.token_ids.is_empty(),
                    corpus_has_kv,
                    cold_refs.is_some(),
                )
            };
            if verdict != TurnIntegrity::Ok {
                let policy = LayerId::from_raw(decl.layer_id)
                    .map(|l| self.read().corrupt_turn_policy_for(l))
                    .unwrap_or_default();
                match policy {
                    CorruptTurnPolicy::DropConversation => {
                        // Regenerable content (code_read files, repo_map
                        // clusters): tombstone the timeline durably; the owning
                        // layer's refresh re-ingests it complete. One tombstone
                        // per timeline per pass — later turns skip above.
                        repaired_integrity += 1;
                        integrity_dropped.insert(timeline);
                        let reason = format!(
                            "integrity repair (turn {}): {}",
                            decl.turn_index,
                            verdict.describe()
                        );
                        self.update(|s| s.tombstone_timeline(timeline));
                        if !self.read_only {
                            match self
                                .persistence
                                .call(|p| p.write_repair_tombstone(timeline.raw(), &reason))
                            {
                                Ok(_) => tracing::info!(
                                    timeline_id = decl.timeline_id,
                                    turn_index = decl.turn_index,
                                    reason = %reason,
                                    "integrity repair: tombstoned incomplete conversation \
                                     — the owning layer re-ingests it",
                                ),
                                Err(e) => tracing::warn!(
                                    timeline_id = decl.timeline_id,
                                    turn_index = decl.turn_index,
                                    "integrity repair tombstone write failed: {e}",
                                ),
                            }
                        }
                        continue;
                    }
                    CorruptTurnPolicy::DropTurn => {
                        // Dialogue is user history — never auto-deleted for
                        // incompleteness. Restore what survived: layout text
                        // still renders, present KV still attends.
                        tracing::warn!(
                            timeline_id = decl.timeline_id,
                            turn_index = decl.turn_index,
                            "incomplete turn restored (user history is never \
                             auto-deleted): {}",
                            verdict.describe(),
                        );
                    }
                }
            }
            let token_count = recovered.token_count;
            // Cold-marker sealed: an empty `Vec<SealedSequence>` flags the
            // turn as on-disk-only. The runtime inject path detects the
            // empty sealed and routes through the engine's `ensure_hot`
            // orchestrator (cold → warm → hot) before borrowing into a
            // view slot.
            //
            // `cold_refs = Some(...)` lights up the residence's cold
            // tier so the new bulk `elevate_to_hot` classifier routes
            // the turn through cold_to_hot. Without it the residence
            // would be `(hot, warm, cold) = (None, None, None)` and
            // the classifier would tag the turn `missing` on the very
            // first projection that needs it. `cold_refs = None` is
            // a recoverable-token-only turn (no persisted chunks) —
            // the substrate keeps it discoverable but it stays unable
            // to materialise KV.
            let turn = RestoredTurn {
                layout: TurnLayout::new(std::mem::take(&mut decl.segments)),
                token_ids: TokenBuffer::from(recovered.token_ids),
                token_count,
                cold: cold_refs,
                block_start: decl.block_start,
                block_end: decl.block_end,
            };
            self.restore_turn_at(timeline, &decl, turn);
            restored += 1;
        }
        // Arm low-priority reconciliation per timeline. The summary forest is
        // immutable and its canonical ternary shape is a pure function of the
        // leaves, so the reload doesn't
        // re-summarise anything — it just asks the summariser to rebuild any
        // internal node that's missing (a crash between sealing leaves and their
        // parent) or non-canonical (binary nodes from the superseded AVL, which
        // `mark_for_reconcile` purges). For a clean forest `reconcile_next`
        // returns `None` on the first pass and the hint clears immediately, so a
        // healthy restart does zero probe work. Reconcile is strictly
        // lower-priority than live turns, so it never floods startup the way the
        // old "re-enqueue the orphan tail" sweep did.
        self.update(|view| {
            let timelines: Vec<TimelineId> = view.all_timeline_ids().collect();
            for tl in timelines {
                view.mark_for_reconcile(tl);
            }
        });
        if let Some(p) = progress {
            p(total, total);
        }
        let read = self.read();
        let n_sections = read.section_count();
        let n_timelines = read.timeline_count();
        let n_conversations = read.conversation_count();
        let orphan_chunk_streams = read.orphan_chunk_stream_count(self.minter.node());
        drop(read);
        tracing::info!(
            sections = n_sections,
            timelines = n_timelines,
            conversations = n_conversations,
            turns = restored,
            skipped_corrupt = skipped_corrupt,
            pending_remote = pending_remote,
            repaired_integrity = repaired_integrity,
            orphan_chunk_streams = orphan_chunk_streams,
            reconstruct_ms = t_reconstruct.elapsed().as_millis() as u64,
            "substrate reload complete",
        );
        if pending.total() > 0 {
            tracing::info!(
                pending = pending.total(),
                no_tokens = pending.no_tokens,
                no_chunks = pending.no_chunks,
                some_chunks = pending.some_chunks,
                unreadable = pending.unreadable,
                n_layers,
                samples = ?pending.samples,
                "substrate reload: what the pending remote turns lack",
            );
        }
        // Orphaned KV — chunks on disk whose owning turn's StreamDecl is gone —
        // means turns were silently dropped in a prior generation (the compaction
        // re-emit-skip bug) and their KV is now unreachable dead weight. Surface
        // it loudly: with the re-emit guard in place this must trend to zero, and
        // compaction now reclaims whatever remains instead of copying it forward.
        if orphan_chunk_streams > 0 {
            tracing::warn!(
                orphan_chunk_streams,
                "reload found orphaned KV (chunks with no StreamDecl): these turns \
                 are unreconstructable and their KV will be reclaimed by compaction",
            );
        }
        Ok(restored)
    }

    /// Read a cold turn's per-layer chunk grid from the redo log so the
    /// caller can run the warm→hot leg (`load_stream` per layer) and
    /// install the resulting `Vec<SealedSequence>`s on the substrate
    /// via the `elevate_to_hot` orchestrator (`ColdRecall`).
    ///
    /// Returns `Ok(None)` when the turn doesn't have a recoverable chunk
    /// grid — e.g. its `Tokens` record is durable but `Chunks` records
    /// haven't yet landed (the async persist callback was still pending
    /// when the daemon shut down).
    pub fn recover_turn_chunks(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
        n_layers: usize,
    ) -> candle::Result<Option<crate::persistence::resume::TurnChunkGrid>> {
        use crate::persistence::resume::{recover_turn_grid, recovered_turn_decls};
        // We need the turn's `StreamDecl` to drive `recover_turn_grid`.
        // Walk the substrate's persisted decls and pick the one matching
        // this (timeline, index). The decl set is small and rebuilt once
        // at restart, so a linear scan is fine.
        let substrate = self.read();
        let decls = recovered_turn_decls(&substrate);
        let decl = match decls
            .into_iter()
            .find(|d| d.timeline_id == timeline.raw() && d.turn_index == index.0)
        {
            Some(d) => d,
            None => return Ok(None),
        };
        let grid = self
            .persistence
            .call(|p| recover_turn_grid(p, &substrate, &decl, n_layers))
            .map_err(|e| candle::Error::Msg(format!("recover_turn_chunks: {e}")))?;
        if grid.is_empty() {
            return Ok(None);
        }
        Ok(Some(grid))
    }

    /// Batched cold→hot load. Loads every key in `keys` against one version of
    /// the substrate and one call to the log's owner, which plans every turn's
    /// read and hands over what it reads through; the reads and the uploads
    /// run here. Returns one `(key, result)` per input key, in the same order,
    /// so the caller can pair each result with its plan entry.
    pub fn cold_load_turns_into_hot(
        &self,
        keys: &[TurnKey],
        backings: &[candle_nn::kv_cache::ChunkedKvBacking],
        device: &candle::Device,
        stager: &mut crate::persistence::cold_load::ColdLoadStager,
    ) -> Vec<(TurnKey, candle::Result<Option<(Vec<SealedSequence>, u64)>>)> {
        use crate::persistence::content_hash::turn_stream_id;
        use crate::persistence::streams::StreamDecl;
        use crate::persistence::transfer::{cold_buffer_size, load_turn_into_hot};

        // Nothing cold, nothing to read: an elevate calls this whether or not
        // its plan holds a cold turn.
        if keys.is_empty() {
            return Vec::new();
        }
        let substrate = self.read();
        // Resolve each turn's decl by its deterministic stream id — an O(1)
        // `streams` map hit. The stream that carries the decl is the same one
        // whose chunks the load reads, so one `stream_of` serves both. This
        // keeps the cold-load O(keys), not O(turns-ingested-so-far).
        let turns: Vec<Option<(TurnDecl, u64)>> = keys
            .iter()
            .map(|&key| {
                let stream =
                    substrate.stream_of(turn_stream_id(key.timeline.raw(), key.index.0))?;
                let Some(StreamDecl::Turn(decl)) = &stream.decl else {
                    return None;
                };
                // The load addresses the turn by its index, and the decl
                // carries the index its writer gave it, which another
                // writer's turn may have moved it above (§6.6).
                let decl = TurnDecl {
                    turn_index: key.index.0,
                    ..decl.clone()
                };
                let kv_bytes_total: u64 = stream.chunks.values().map(|loc| loc.payload_len).sum();
                Some((decl, kv_bytes_total))
            })
            .collect();
        let buffer_size = cold_buffer_size(stager);
        let owner_wait = profile::span("cold:persistence_owner");
        let reads: Vec<Option<PersistenceResult<_>>> = self.persistence.call(|p| {
            turns
                .iter()
                .map(|turn| {
                    turn.as_ref().map(|(decl, _)| {
                        p.cold_read(
                            &substrate,
                            turn_stream_id(decl.timeline_id, decl.turn_index),
                            buffer_size,
                        )
                    })
                })
                .collect()
        });
        owner_wait.end();
        keys.iter()
            .zip(turns.iter().zip(reads))
            .map(|(&key, (turn, read))| {
                let (Some((decl, kv_bytes_total)), Some(read)) = (turn, read) else {
                    return (key, Ok(None));
                };
                let result = read
                    .map_err(|e| candle::Error::Msg(format!("cold-load plan: {e}")))
                    .and_then(|(plan, source)| {
                        load_turn_into_hot(backings, device, &plan, &source, decl, stager)
                    })
                    .map(|sealed| Some((sealed, *kv_bytes_total)));
                (key, result)
            })
            .collect()
    }

    /// Cold-load a persisted section's chunks back into hot VRAM via
    /// the shared `load_stream_into_hot` pipeline.  Returns the
    /// per-layer `SealedSequence` the substrate's residence slab
    /// installs as the section's hot tier.
    pub fn cold_load_section_into_hot(
        &self,
        stream_id: StreamId,
        chunks_per_layer: usize,
        backings: &[candle_nn::kv_cache::ChunkedKvBacking],
        device: &candle::Device,
        stager: &mut crate::persistence::cold_load::ColdLoadStager,
    ) -> candle::Result<Vec<SealedSequence>> {
        use crate::persistence::transfer::{cold_buffer_size, load_stream_into_hot};
        let substrate = self.read();
        let buffer_size = cold_buffer_size(stager);
        let (plan, source) = self
            .persistence
            .call(|p| p.cold_read(&substrate, stream_id, buffer_size))
            .map_err(|e| candle::Error::Msg(format!("cold_load_section_into_hot: {e}")))?;
        load_stream_into_hot(
            backings,
            device,
            &plan,
            &source,
            stream_id,
            chunks_per_layer,
            stager,
        )
        .map_err(|e| candle::Error::Msg(format!("cold_load_section_into_hot: {e}")))
    }

    /// Resolve a section stream's per-chunk redo-log locations into
    /// per-layer cold references — what the substrate stores under
    /// `residence.cold = Some(...)`.  Returns `None` when the stream
    /// is unknown or has no chunks recorded.
    pub fn recover_section_cold_refs(
        &self,
        stream_id: StreamId,
        n_layers: usize,
    ) -> candle::Result<Option<Vec<StoredSequence>>> {
        let substrate = self.read();
        crate::persistence::resume::recover_section_cold_refs(&substrate, stream_id, n_layers)
            .map_err(|e| candle::Error::Msg(format!("recover_section_cold_refs: {e}")))
    }
}
