//! Moves a turn's key-value state between the device, RAM and the redo log, and
//! tracks where each copy lives.
//!
//! Persistence is queued to the log's owner, so a seal never waits on the disk, and
//! a warm-to-cold migration installs its cold copy only once the data is durable. A
//! stream's commit is written to the index as well as the log, because compaction
//! rewrites commits from the index, and a commit the index does not hold is lost. A
//! read-only substrate writes nothing and records no location for a turn it never
//! wrote.

use super::*;

impl Conversation {
    /// Clear a turn's hot sealed grid, releasing VRAM arena chunks via
    /// dropping its ChunkGid Arcs. Returns `true` if hot bytes were
    /// dropped (see [`Substrate::clear_turn_sealed`]).
    pub fn clear_turn_sealed(&self, timeline: TimelineId, index: TurnIndex) -> bool {
        self.read().clear_turn_sealed(timeline, index)
    }

    /// Hot-tier VRAM byte snapshot (sum across every turn whose `sealed`
    /// carries an actual chunk grid).
    pub fn hot_turn_bytes(&self) -> usize {
        self.read().hot_turn_bytes()
    }

    /// Byte size of a single hot turn (for the pre-flight evict
    /// accounting). `None` if cold or unknown.
    pub fn turn_hot_bytes(&self, timeline: TimelineId, index: TurnIndex) -> Option<usize> {
        self.read().turn_hot_bytes(timeline, index)
    }

    /// Flag every turn residence of `timeline` for full eviction once its KV is
    /// durable on disk — see [`crate::substrate::Substrate::mark_timeline_evict_when_cold`].
    /// The persistence pipeline then frees the turns' VRAM (at warm-land) and
    /// RAM (at cold-land), leaving them cold-only. Returns the count flagged.
    pub fn mark_timeline_evict_when_cold(&self, timeline: TimelineId) -> usize {
        self.read().mark_timeline_evict_when_cold(timeline)
    }

    /// How many of `timeline`'s turns still hold a device copy — see
    /// [`crate::substrate::Substrate::timeline_hot_residences`].
    pub fn timeline_hot_residences(&self, timeline: TimelineId) -> usize {
        self.read().timeline_hot_residences(timeline)
    }

    /// Rewrite every residence's gids through a KV compaction's map — the
    /// substrate's half of the sweep. See
    /// [`crate::substrate::Substrate::rewrite_for_compaction`].
    ///
    /// Run by the thread compacting the device's KV, inside the window the
    /// compaction holds chunk locations frozen: a reader on another thread that
    /// resolves gids to device addresses holds them first
    /// (`candle_nn::kv_cache::try_hold_chunk_locations`), so it sees every old
    /// gid or every new one, never a mixture — a mixture is one turn's KV read
    /// through another turn's addresses.
    pub fn rewrite_for_compaction(&self, sweep: &mut Sweep<'_>) -> candle::Result<usize> {
        self.read().rewrite_for_compaction(sweep)
    }

    /// Set one `(timeline, turn)` summary-tree node — in RAM and as a
    /// `TreeMetadata` record, under one identity. Called by the summariser
    /// thread after every atomic tree mutation (§7.2).
    pub fn record_tree_meta(
        &self,
        timeline: TimelineId,
        idx: TurnIndex,
        meta: TreeNodeMeta,
    ) -> candle::Result<()> {
        let payload = TreeMetadataPayload {
            timeline_id: timeline.raw(),
            turn_index: idx.0,
            kind: match meta.kind {
                TurnKind::Normal => 0,
                TurnKind::SummaryOfTurns => 1,
                TurnKind::SummaryOfSummaries => 2,
            },
            tree_height: meta.tree_height,
            children: meta.children.iter().map(|c| c.0).collect(),
        };
        let identity = self.minter.mint(0);
        self.update(|s| s.set_tree_meta(timeline, idx, meta.clone(), identity));
        if self.read_only {
            return Ok(());
        }
        self.persistence
            .call(|p| p.write_tree_metadata(payload, identity))
            .map_err(|e| candle::Error::Msg(format!("write_tree_metadata: {e}")))
    }

    /// Persist only a turn's per-layer chunk records — the post-quantization
    /// half of the async seal/persist chain. Called from inside the
    /// bg-quantizer callback once float→quant migrations have landed.
    pub fn persist_turn_chunks(
        &self,
        stream_id: StreamId,
        layers: &crate::persistence::resume::TurnChunkGrid,
    ) -> candle::Result<()> {
        if self.read_only {
            return Ok(());
        }
        self.persistence
            .call(|p| crate::persistence::resume::persist_turn_chunks(p, stream_id, layers))
            .map_err(|e| candle::Error::Msg(format!("persist turn chunks: {e}")))
    }

    /// Persist a turn's chunks and return the per-layer [`StoredSequence`]
    /// references — the warm→cold leg of the persistence thread's
    /// `run_pass`. The returned references go straight into the
    /// substrate via `Substrate::install_cold`.
    ///
    /// On a read-only substrate nothing is written and nothing is returned:
    /// an empty set installs no cold tier, so the stream stays RAM-resident
    /// rather than claiming disk locations it never got.
    pub fn persist_turn_chunks_capture(
        &self,
        stream_id: StreamId,
        layers: &crate::persistence::resume::TurnChunkGrid,
    ) -> candle::Result<Vec<StoredSequence>> {
        if self.read_only {
            return Ok(Vec::new());
        }
        let (stored, locs) = self
            .persistence
            .call(|p| crate::persistence::resume::persist_turn_chunks_capture(p, stream_id, layers))
            .map_err(|e| candle::Error::Msg(format!("persist turn chunks capture: {e}")))?;
        // Fold the freshly-written chunk locations into the substrate's
        // authoritative chunk index so an in-process cold→hot elevation
        // (eviction then re-access without a daemon restart) can locate them
        // via `plan_chunked_read`. The redo-log walker only repopulates
        // `stream.chunks` on reload; without this the index stays empty until
        // then, and the cold-load pipeline plans zero records.
        self.update(|view| {
            for &(flat, loc) in &locs {
                view.apply_chunk_loc(stream_id, flat, loc.value, loc.id);
            }
        });
        Ok(stored)
    }

    /// Persist a turn's `Tokens` record only — no trailing `Commit`.
    /// Used by the seal path now that chunks (and the matching Commit)
    /// are written asynchronously by the persistence thread.
    pub fn persist_tokens_only(
        &self,
        stream_id: StreamId,
        token_ids: &[u32],
    ) -> candle::Result<()> {
        // A read-only substrate keeps the in-memory token buffer and registers
        // no location: there is no record for one to point at.
        if self.read_only {
            return Ok(());
        }
        // Append on the log's owner, then register the location in the
        // substrate index. The `apply_tokens_loc` is REQUIRED: without it
        // `entry.tokens` is `None` and the maintenance/compaction relocation
        // drops the tokens on the next pass (see
        // `crate::persistence::resume::persist_tokens_only`).
        let loc = self
            .persistence
            .call(|p| crate::persistence::resume::persist_tokens_only(p, stream_id, token_ids))
            .map_err(|e| candle::Error::Msg(format!("persist tokens: {e}")))?;
        self.update(|s| s.apply_tokens_loc(stream_id, loc.value, loc.id));
        Ok(())
    }

    /// Append a stream-level `Commit` record at the given chunk index — the
    /// post-chunks re-commit used by the async seal/persist chain to upgrade
    /// the manifest's `committed_through` once the heavy `Chunks` records
    /// have been written.
    ///
    /// The index learns the commit too, not only the log. A `Commit` is a
    /// resident record: maintenance and compaction rewrite it from the
    /// index's `committed_through`, never copy it, so a commit the index does
    /// not hold is dropped by the first rewrite of its segment — and a
    /// section without one is prefilled again on every later boot.
    pub fn commit_stream_through(
        &self,
        stream_id: StreamId,
        through_index: u64,
    ) -> candle::Result<()> {
        if self.read_only {
            return Ok(());
        }
        let commit = self
            .persistence
            .call(|p| p.commit_stream(stream_id, through_index))
            .map_err(|e| candle::Error::Msg(format!("commit stream: {e}")))?;
        self.update(|s| s.apply_commit_through(stream_id, through_index, commit));
        Ok(())
    }

    /// Enqueue a warm→cold KV migration onto the log's owner. The caller
    /// (persistence thread) has already gathered `grid` off the GPU; the owner
    /// appends the chunks, folds their locations into the substrate index,
    /// marks the stream durable, and `install_cold`s — all AFTER the append. So
    /// the persistence thread's hot→warm VRAM-relief drain is never blocked by
    /// cold-write disk I/O, and a cold copy is only ever installed on durable
    /// data, keeping the `purge_warm` RAM-unload invariant intact. Bounded by
    /// the owner's dual-cap backpressure.
    ///
    /// On a read-only substrate no cold write is coming, so the residence is
    /// left as it is — not flagged as having one in flight — and stays warm.
    pub fn enqueue_kv_cold(
        &self,
        residence: ResidenceIndex,
        stream_id: StreamId,
        grid: TurnChunkGrid,
    ) {
        if self.read_only {
            return;
        }
        // Flag the write in flight BEFORE enqueuing so the next persistence pass's
        // `snapshot_pending_cold` skips this residence until `install_cold` (on the
        // owner) clears it — no double gather/write.
        self.read().mark_cold_pending(residence);
        self.persistence.enqueue(WriteJob::KvCold {
            residence,
            stream_id,
            grid,
        });
    }

    /// Persist the projection schema/template into the substrate's
    /// `Template` record — compare-and-insert (only appends when it differs
    /// from what the log already holds), then commit if written. Lets the
    /// log carry the projection needed to reconstruct the substrate.
    pub fn set_template(&self, template: &[u8]) -> candle::Result<()> {
        if self.read_only {
            return Ok(());
        }
        self.persistence.call(|p| {
            let wrote = p
                .set_template(template)
                .map_err(|e| candle::Error::Msg(format!("persist template: {e}")))?;
            if wrote {
                p.commit()
                    .map_err(|e| candle::Error::Msg(format!("commit template: {e}")))?;
            }
            Ok(())
        })
    }

    /// Persist a turn's projection-event timeline to the redo log — the
    /// `ProjectionEvents` record. `payload` is the
    /// [`crate::projection::encode_events`] JSON. Also mirrors the bytes into
    /// the in-RAM substrate so the current session reads them back without a
    /// restart.
    /// Mirror a turn's projection-event trajectory into the in-RAM substrate
    /// (the GUI replay and any same-session read want it now) and enqueue the
    /// durable append onto the log's owner, so the caller never waits on the
    /// disk. Same crash-safety as [`Self::enqueue_tokens`]: the
    /// record is last-writer-wins per stream, and one that never lands leaves
    /// the turn without its replay trajectory rather than corrupting it.
    ///
    /// The payload is moved, not copied — the encode already happened on the
    /// caller's side, and the in-RAM mirror and the queued job share it.
    pub fn persist_projection_events(&self, stream_id: StreamId, payload: Vec<u8>) {
        let identity = self
            .minter
            .mint_for(RecordType::ProjectionEvents, stream_id.0);
        self.update(|s| s.apply_projection_events(stream_id, payload.clone(), identity));
        self.persistence.enqueue(WriteJob::ProjectionEvents {
            stream_id,
            payload,
            identity,
        });
    }

    /// Persist a turn's encoded wide-Q signature window to the redo log (`WideQSig`
    /// record, the greatest stamp per stream) and mirror it into the in-RAM
    /// substrate under the same identity.
    pub fn persist_wide_q_sigs(&self, stream_id: StreamId, payload: &[u8]) -> candle::Result<()> {
        let identity = self.minter.mint_for(RecordType::WideQSig, stream_id.0);
        self.update(|s| s.apply_wide_q_sigs(stream_id, payload.to_vec(), identity));
        if self.read_only {
            return Ok(());
        }
        let loc = self
            .persistence
            .call(|p| p.append_wide_q_sigs(stream_id, payload, identity))
            .map_err(|e| candle::Error::Msg(format!("persist wide-Q sigs: {e}")))?;
        // Appended: the substrate holds where it is, not its bytes.
        self.update(|s| s.apply_wide_q_sigs_loc(stream_id, loc, identity));
        Ok(())
    }

    /// Persist a turn's QSA index page to the redo log (`TurnIndexPage`,
    /// last-writer-wins per stream) and mirror it into the in-RAM substrate.
    ///
    /// The page has to outlive the process for the same reason the turn's K/V
    /// does: a later projection borrows both. Kept in RAM alone it is gone at
    /// restart, and every cold-loaded turn then hands a slot keys it cannot
    /// index.
    pub fn persist_index_page(&self, stream_id: StreamId, payload: &[u8]) -> candle::Result<()> {
        let identity = self.minter.mint_for(RecordType::TurnIndexPage, stream_id.0);
        self.update(|s| s.apply_index_page(stream_id, payload, identity));
        if self.read_only {
            return Ok(());
        }
        let loc = self
            .persistence
            .call(|p| p.append_turn_index_page(stream_id, payload, identity))
            .map_err(|e| candle::Error::Msg(format!("persist index page: {e}")))?;
        // Durable: the substrate holds where it is, not its bytes.
        self.update(|s| s.settle_index_page(stream_id, loc, identity));
        Ok(())
    }

    /// Enqueue a turn/section `Tokens` record onto the log's owner — the
    /// durable copy for reload. The in-memory token buffer is already set at
    /// record time, so this NEVER makes the seal wait on the disk.
    /// Metadata-class: exempt from the owner's KV byte cap, so a warm→cold
    /// backlog can't stall it either. Async loss window: a `Tokens` record that
    /// never lands (hard kill before the owner drains) does NOT remove the
    /// turn — reconstruct restores it from its
    /// decl with an empty token buffer, since the layout text, KV chunks, and sigs
    /// persist through their own channels. The startup integrity check
    /// (`persistence::integrity`) then repairs regenerable layers by re-ingest;
    /// dialogue turns restore with a warning and render from layout text.
    pub fn enqueue_tokens(&self, stream_id: StreamId, token_ids: Vec<u32>) {
        self.persistence.enqueue(WriteJob::Tokens {
            stream_id,
            token_ids,
        });
    }

    /// Mirror a turn's wide-Q signature into the in-RAM substrate synchronously
    /// (the same-session provenance scan reads it back immediately) and enqueue
    /// the durable redo-log append onto the log's owner — so the seal never
    /// waits on the disk. Same crash-safety as [`Self::enqueue_tokens`].
    pub fn enqueue_wide_q_sigs(&self, stream_id: StreamId, payload: Vec<u8>) {
        let identity = self.minter.mint_for(RecordType::WideQSig, stream_id.0);
        self.update(|s| s.apply_wide_q_sigs(stream_id, payload.clone(), identity));
        self.persistence.enqueue(WriteJob::WideQSigs {
            stream_id,
            payload,
            identity,
        });
    }

    /// Enqueue a turn's or a prompt section's QSA index page under its stream
    /// id, mirroring it in RAM first so this session's own projections can
    /// borrow the piece immediately.
    pub fn enqueue_index_page(&self, stream_id: StreamId, payload: Vec<u8>) {
        let identity = self.minter.mint_for(RecordType::TurnIndexPage, stream_id.0);
        self.update(|s| s.apply_index_page(stream_id, payload.as_slice(), identity));
        self.persistence.enqueue(WriteJob::TurnIndexPage {
            stream_id,
            payload,
            identity,
        });
    }
}
