//! Rebuilds the substrate's in-RAM state from the redo log on startup, and
//! installs the turns reloaded from it.
//!
//! Recovery applies each walked record straight into the substrate, so one pass
//! over the log fills the manifest and every per-entity table. A restored turn
//! keeps the tree metadata already replayed for it, because seeding a default
//! would collapse the summary spine on every restart. Cold references are
//! refreshed after compaction moves records, since the old offsets would read the
//! wrong bytes on the next promotion. A reloaded turn holds only its cold
//! reference: residence is demand-driven, so reload never installs hot KV.

use super::*;

impl Substrate {
    /// Whether `timeline` is marked for distillation (any mode).
    pub fn is_distilled(&self, timeline: TimelineId) -> bool {
        self.distilled_timelines.contains_key(&timeline)
    }

    /// The distillation mode `timeline` is marked at, if any.
    pub fn distill_mode(&self, timeline: TimelineId) -> Option<DistillMode> {
        self.distilled_timelines.get(&timeline).copied()
    }

    /// Direct read of the distilled-timeline map — the compactor uses it to shed
    /// each timeline's turns to its [`DistillMode`].
    pub fn distilled_timelines(&self) -> &PMap<TimelineId, DistillMode> {
        &self.distilled_timelines
    }

    /// On-disk bytes held by streams of tombstoned timelines and turns — dead
    /// weight the header-keyed accounting can't see (a timeline tombstone names
    /// its target in the payload, not the header, and the doomed records were
    /// live appends at write time). Summed from the in-RAM stream index, no
    /// disk I/O; the compaction trigger adds this to the incremental dead-byte
    /// counter.
    ///
    /// A tombstoned section is not here: its tombstone names the stream in the
    /// header, so the accounting retires its records itself, and the ended
    /// generation leaves the index ([`Self::end_section`]).
    pub fn tombstoned_stream_bytes(&self) -> u64 {
        if self.tombstoned_timelines.is_empty() && self.tombstoned_turns.is_empty() {
            return 0;
        }
        self.streams
            .iter_physical()
            .filter(|(key, s)| match &s.decl {
                Some(StreamDecl::Turn(t)) => {
                    TimelineId::from_raw(t.timeline_id)
                        .is_some_and(|tl| self.tombstoned_timelines.contains(&tl))
                        || self.is_turn_stream_retired(*key, t)
                }
                Some(StreamDecl::PromptSection(_)) | None => false,
            })
            .map(|(_, s)| {
                s.chunks.values().map(|c| c.record_size).sum::<u64>()
                    + s.tokens.map_or(0, |l| l.record_size)
            })
            .sum()
    }

    /// The turns dead by turn-scoped tombstones — consulted by compaction,
    /// which sheds their bulk records while re-emitting the tombstone itself
    /// (the reload's placeholder logic needs it).
    pub fn tombstoned_turns(&self) -> &PSet<TurnMark> {
        &self.tombstoned_turns
    }

    /// Re-point every residence's cold-tier references at the current
    /// stream index.  Compaction rewrites the log — every record moves
    /// to a new offset — then re-walks the new file into `streams`; the
    /// per-residence [`StoredSequence`]s still hold the old offsets and
    /// would read garbage on the next cold→hot elevation.  The chunk
    /// grid shape (layers × chunks per layer) is preserved by
    /// compaction, so only offsets and record sizes change.
    ///
    /// Residences whose stream is absent from the active index are left
    /// untouched: a borrowed inherited-log stream lives in its own
    /// (uncompacted) file and its references remain valid, and a
    /// tombstoned stream's records were dropped from the compacted log
    /// but its turns are filtered from every projection, so its stale
    /// references are never followed.
    ///
    /// Read against this version's stream index: the caller refreshes from the
    /// version that holds the relocated locations.
    pub fn refresh_cold_refs(&self) {
        for (idx, r) in self.residences.iter() {
            self.refresh_slot_cold_refs(idx, r);
        }
    }

    /// One residence's cold references, re-read from its stream's index. The
    /// refreshed references replace the ones they were read from; a residence
    /// whose cold references changed in between is refreshed again from the
    /// newer ones.
    pub(super) fn refresh_slot_cold_refs(&self, idx: ResidenceIndex, r: &Residence) {
        let Some(stream) = self.streams.get(r.stream_id) else {
            return;
        };
        while let Some(held) = r.cold() {
            let n_layers = held.len();
            let chunks_per_layer = held.first().map_or(0, |s| s.chunks.len());
            if chunks_per_layer == 0 || stream.chunks.len() != n_layers * chunks_per_layer {
                return;
            }
            let mut cold = (*held).clone();
            for (layer, seq) in cold.iter_mut().enumerate() {
                for (c, chunk) in seq.chunks.iter_mut().enumerate() {
                    let flat = (layer * chunks_per_layer + c) as u64;
                    if let Some(loc) = stream.chunks.get(&flat) {
                        chunk.log_offset = loc.offset;
                        chunk.record_len = loc.record_size;
                        chunk.token_count = loc.token_count as u16;
                    }
                }
            }
            if self.residences.replace_cold(idx, &held, cold) {
                return;
            }
        }
    }

    /// [`Self::refresh_cold_refs`] for the residences of `moved` alone — what
    /// a maintenance op that relocated those streams' chunks has to refresh.
    /// Every other residence's references are untouched by the op.
    pub fn refresh_cold_refs_of(&self, moved: &HashSet<StreamId>) {
        for (idx, r) in self.residences.iter() {
            if moved.contains(&r.stream_id) {
                self.refresh_slot_cold_refs(idx, r);
            }
        }
    }

    /// Apply one walked redo-log record directly into the substrate's
    /// in-RAM state.  The dispatch lives here (not on `Manifest`)
    /// because per-entity records — chunks, stream decls, labels,
    /// tree metadata, debug ids — are substrate state, not manifest
    /// state.  The manifest only sees singletons (`ModelKey`,
    /// `Template`, `Tokenizer`, `ToolSummary`).
    ///
    /// Called from `SubstratePersistence::recover_with_substrate_sink`
    /// during startup so the walker pass populates both the manifest's
    /// singleton hints and the substrate's per-entity state in one
    /// sweep over the log.
    pub fn apply_walker_entry(&mut self, entry: &WalkEntry) {
        let h = &entry.record.header;
        let stream_id = StreamId(h.stream_id);
        let id = h.identity();
        match h.record_type {
            RecordType::StreamDecl => {
                if let Ok(decl) = StreamDecl::decode(&entry.record.payload) {
                    self.apply_stream_decl(stream_id, decl, id);
                }
            }
            RecordType::TurnCoupling => {
                if let Ok(payload) = TurnCouplingPayload::decode(&entry.record.payload) {
                    self.apply_turn_coupling(&payload, id);
                }
            }
            RecordType::Chunk => {
                self.apply_chunk_loc(
                    stream_id,
                    h.chunk_index,
                    ChunkLoc {
                        // The segment the walk stamped on this entry — the
                        // physical file that holds these KV bytes (§5.1). The
                        // cold-load read routes there.
                        segment: entry.segment,
                        offset: entry.offset,
                        payload_len: h.payload_len,
                        record_size: entry.size,
                        token_count: h.token_count,
                        format: h.format,
                    },
                    id,
                );
            }
            RecordType::Tokens => {
                self.apply_tokens_loc(
                    stream_id,
                    RecordLoc {
                        segment: entry.segment,
                        offset: entry.offset,
                        payload_len: h.payload_len,
                        record_size: entry.size,
                    },
                    id,
                );
            }
            RecordType::Commit => {
                self.apply_commit_through(stream_id, h.chunk_index, id);
            }
            RecordType::Snapshot if h.retracts_snapshot() => {
                self.retract_snapshot(stream_id, id);
            }
            RecordType::Snapshot => {
                self.apply_snapshot_loc(
                    stream_id,
                    RecordLoc {
                        segment: entry.segment,
                        offset: entry.offset,
                        payload_len: h.payload_len,
                        record_size: entry.size,
                    },
                    id,
                );
            }
            RecordType::BranchCheckpoint => {
                self.apply_branch_checkpoint_loc(
                    stream_id,
                    RecordLoc {
                        segment: entry.segment,
                        offset: entry.offset,
                        payload_len: h.payload_len,
                        record_size: entry.size,
                    },
                    id,
                );
            }
            RecordType::Label => {
                if let Ok((tl, meta)) = decode_label_payload(&entry.record.payload) {
                    self.apply_conv_meta(tl, &meta, id);
                }
            }
            RecordType::ConvState => {
                if let Ok((tl, state)) = decode_conv_state_payload(&entry.record.payload) {
                    self.apply_conv_state(tl, state, id);
                }
            }
            RecordType::TreeMetadata => {
                if let Ok(payload) = TreeMetadataPayload::decode(&entry.record.payload) {
                    self.apply_tree_metadata_payload(&payload, id);
                }
            }
            RecordType::DebugId => {
                if let Ok(payload) = DebugIdPayload::decode(&entry.record.payload) {
                    self.apply_debug_id_payload(&payload, id);
                }
            }
            RecordType::CustomObject => {
                if let Ok(payload) = CustomObjectPayload::decode(&entry.record.payload) {
                    self.apply_custom_object(payload, id);
                }
            }
            RecordType::Tombstone => {
                if let Ok(payload) = TombstonePayload::decode(&entry.record.payload) {
                    self.apply_tombstone(&payload);
                }
            }
            RecordType::SectionTombstone => {
                if let Ok(payload) = SectionTombstonePayload::decode(&entry.record.payload) {
                    self.apply_section_tombstone(stream_id, &payload);
                }
            }
            RecordType::Distilled => {
                if let Ok(payload) = DistillPayload::decode(&entry.record.payload) {
                    self.apply_distill(&payload);
                }
            }
            RecordType::Deleted => {
                if let Ok(payload) = DeletedPayload::decode(&entry.record.payload) {
                    self.apply_deleted(&payload);
                }
            }
            RecordType::ProjectionEvents => {
                // Opaque JSON bytes — the projection layer decodes them on read.
                self.apply_projection_events(stream_id, entry.record.payload.clone(), id);
            }
            RecordType::WideQSig => {
                // Opaque wide-Q window bytes (provenance::wide_sig) — each
                // (re)projection writes the window again.
                self.apply_wide_q_sigs(stream_id, entry.record.payload.clone(), id);
            }
            RecordType::TurnIndexPage => {
                // Opaque QSA index-page bytes — a turn's or a prompt section's; a
                // re-seal writes the page again. **Located, not held**: recovery
                // leaves the payload in the log (`IndexPage::Stored`), and a
                // projection reads it when it needs it.
                self.apply_index_page_loc(
                    stream_id,
                    RecordLoc {
                        segment: entry.segment,
                        offset: entry.offset,
                        payload_len: h.payload_len,
                        record_size: entry.size,
                    },
                    id,
                );
            }
            // Singletons go to the manifest, not the substrate; the
            // header-index chain is consumed by recovery, never here.
            RecordType::ModelKey
            | RecordType::Template
            | RecordType::Tokenizer
            | RecordType::HeaderIndex
            // NPC records are not the substrate's to interpret. They share its
            // log because a character must outlive the process, but the
            // registry that owns them lives in `npcd`, which reads them from
            // the same walk through its own sink. The substrate holds no
            // opinion about a character.
            | RecordType::Npc
            // A conversation's file events belong to the daemon above; the
            // persistence layer locates them (`VfsIndex`) and the substrate
            // holds no opinion about them either.
            | RecordType::VfsEvent
            | RecordType::VfsTombstone
            // Blobs are located by the persistence layer too (`BlobIndex`).
            | RecordType::Blob
            // What this member has applied of other origins is the persistence
            // layer's to keep (`persistence::replica`), and so are the
            // generations fixed for what it wrote cut off
            // (`persistence::rebase`), which the walk applies before a record
            // gets here.
            | RecordType::Replica
            | RecordType::Rebase
            | RecordType::Unknown => {}
        }
    }

    /// Build an in-memory [`summary_tree::SummaryTree`] (forest) from the
    /// timeline's persisted `tree_meta`.  Used by the projection's score-density
    /// selector (§8); the peak set is derived from node parentage.
    ///
    /// The returned tree's [`NodeId`](NodeId)
    /// values are `TurnIndex.0` directly — there's a 1-to-1 mapping
    /// between substrate turns and tree nodes.
    pub fn build_summary_tree_in_memory(&self, timeline: TimelineId) -> SummaryTree {
        let mut tree = SummaryTree::new();
        let tl = match self.timelines.get(&timeline) {
            Some(t) => t,
            None => return tree,
        };
        for (idx, meta) in &tl.tree_meta {
            let token_count = tl
                .turns
                .get(idx)
                .map(|e| e.content.token_count as u32)
                .unwrap_or(0);
            let node_children: Vec<NodeId> = meta.children.iter().map(|c| NodeId(c.0)).collect();
            let node = Node {
                id: NodeId(idx.0),
                kind: meta.kind,
                children: node_children,
                tree_height: meta.tree_height,
                tokens: token_count,
            };
            tree.insert_node(node);
            match meta.kind {
                TurnKind::Normal => tree.push_chrono_normal(NodeId(idx.0)),
                TurnKind::SummaryOfTurns => tree.push_chrono_leaf(NodeId(idx.0)),
                TurnKind::SummaryOfSummaries => {}
            }
        }
        // Install the exchange couplings, projected onto the chrono-normal
        // positions the tree just collected, so selection can expand a hit on
        // either half of a tool round-trip into the whole exchange.
        let normals: Vec<TurnIndex> = tree
            .chrono_normals()
            .iter()
            .map(|n| TurnIndex(n.0))
            .collect();
        tree.set_couplings(crate::summary_tree::exchange::over_normals(
            &self.couplings_of(timeline),
            &normals,
        ));
        // The peak set (forest roots) is derived from the node parentage — no
        // single root to install.
        tree
    }

    // ── Append paths ─────────────────────────────────────────────────────────

    #[cfg(any(test, feature = "test-helpers"))]
    pub fn append_with_blocks_for_test(
        &mut self,
        layer: LayerId,
        group: GroupId,
        token_count: usize,
        block_start: u64,
        block_end: u64,
    ) -> TurnIndex {
        let existing = self.active_timelines_for_group(group).next();
        let timeline = if let Some(t) = existing {
            t
        } else {
            let raw = (self.timelines.len() as u64) + 1;
            let id = TimelineId::for_test(raw);
            self.register_timeline(id, layer, group);
            id
        };
        self.append_with_blocks(timeline, token_count, block_start, block_end)
    }

    pub fn append_with_blocks(
        &mut self,
        timeline: TimelineId,
        token_count: usize,
        block_start: u64,
        block_end: u64,
    ) -> TurnIndex {
        let idx = self
            .timelines
            .get(&timeline)
            .expect("append_with_blocks: timeline not registered")
            .next_turn_index();
        let compression = self.timeline_compression.get(&timeline).copied();
        let residence = self.alloc_residence(turn_stream_id(timeline.raw(), idx.0), compression);
        // `append_with_blocks` declares a turn's existence and block
        // range, but holds no sealed KV — the residence stays cold
        // (`hot = None`) until an elevate / restore_turn install
        // puts bytes in.
        let tl = entry_mut(&mut self.timelines, &timeline).unwrap();
        tl.turns.insert(
            idx,
            Arc::new(TurnEntryData {
                block_range: (block_start, block_end),
                content: TurnPart {
                    layout: TurnLayout::default(),
                    token_count,
                    token_ids: TokenBuffer::default(),
                    residence,
                },
            }),
        );
        *self.timeline_token_totals.entry(timeline).or_default() += token_count;
        // Every persisted turn carries a parallel `TreeNodeMeta`.  New
        // turns default to a `Normal` content sub-leaf and are pushed
        // onto the summariser's pending queue so the async thread can
        // absorb them into a `SummaryOfTurns` leaf — unless this timeline
        // opts out of summarisation (utility/reference layers).
        tl.tree_meta.insert(idx, TreeNodeMeta::default());
        if tl.summarize {
            tl.pending_summary_queue.push_back(idx);
        }
        idx
    }

    /// Append a turn with its sealed KV data as one indivisible
    /// content unit.
    ///
    /// `write` carries the turn's text, token IDs, block range, and
    /// a GPU-resident `Arc<Vec<SealedSequence>>` snapshot.
    /// `migrate_to_cpu` is called inside this function to move the
    /// bytes to the warm tier; the GPU chunks are freed as soon as
    /// the caller drops the `sealed_gpu` Arc, so no GPU arena slots
    /// are held by the substrate after this call returns.
    pub fn append_complete(
        &mut self,
        timeline: TimelineId,
        write: TurnPartWrite,
        migrate_to_cpu: impl FnMut(&[SealedSequence]) -> candle::Result<Vec<SealedSequence>>,
    ) -> candle::Result<TurnIndex> {
        let sealed_cpu = Self::migrate_sealed(&write, migrate_to_cpu)?;
        self.append_migrated(timeline, write, sealed_cpu)
    }

    /// Run a turn's GPU→CPU migration, without needing `&mut self`.
    ///
    /// Split out so a caller holding the substrate behind a lock can migrate
    /// FIRST and take the lock only for the insert — see
    /// [`Self::append_migrated`]. The migration moves K/V off the device, so
    /// doing it inside the write lock held every sealed turn serialised the
    /// copy against every reader and writer of the substrate.
    ///
    /// `Some(_)` (even an empty vec) means "this turn claims sealed bytes — run
    /// the migration to get the CPU side." `None` means "no bytes at all."
    /// Empty input to migrate is legitimate: callers in the GPU-less test paths
    /// pass an empty `sealed_gpu` and rely on the migration closure to produce
    /// the canonical CPU content.
    pub fn migrate_sealed(
        write: &TurnPartWrite,
        mut migrate_to_cpu: impl FnMut(&[SealedSequence]) -> candle::Result<Vec<SealedSequence>>,
    ) -> candle::Result<Vec<SealedSequence>> {
        match write.sealed_gpu.as_ref() {
            Some(g) => migrate_to_cpu(g),
            None => Ok(Vec::new()),
        }
    }

    /// [`Self::append_complete`] with the migration already done — the part that
    /// genuinely needs exclusive access to the substrate.
    pub fn append_migrated(
        &mut self,
        timeline: TimelineId,
        write: TurnPartWrite,
        sealed_cpu: Vec<SealedSequence>,
    ) -> candle::Result<TurnIndex> {
        let idx = self
            .timelines
            .get(&timeline)
            .expect("append_complete: timeline not registered")
            .next_turn_index();
        let compression = self.timeline_compression.get(&timeline).copied();
        let residence = self.alloc_residence(turn_stream_id(timeline.raw(), idx.0), compression);
        // A turn on a transient (scratch-fork) timeline never cold-persists — its KV
        // is spliced by reference onto a durable file timeline, so a cold copy would
        // only strand an orphan. See `mark_timeline_transient` / `no_cold_persist`.
        if self.transient_timelines.contains(&timeline) {
            self.residences.get(residence).set_flag(NO_COLD_PERSIST);
        }
        if self.splice_source_timelines.contains(&timeline) {
            self.residences.get(residence).set_flag(SPLICE_SOURCE);
        }
        let block_start = write.block_start;
        let block_end = write.block_end;
        let token_count = write.token_count;
        {
            let tl = entry_mut(&mut self.timelines, &timeline).unwrap();
            tl.turns.insert(
                idx,
                Arc::new(TurnEntryData {
                    block_range: (block_start, block_end),
                    content: TurnPart {
                        layout: write.layout,
                        token_count,
                        token_ids: write.token_ids,
                        residence,
                    },
                }),
            );
            tl.tree_meta.insert(idx, TreeNodeMeta::default());
            if tl.summarize {
                tl.pending_summary_queue.push_back(idx);
            }
        }
        *self.timeline_token_totals.entry(timeline).or_default() += token_count;
        if !sealed_cpu.is_empty() {
            self.install_hot(residence, sealed_cpu);
        }
        Ok(idx)
    }

    /// Insert a turn reconstructed from the redo log — the substrate-reload
    /// path (§16.12 of `docs/archived/kv_tier_migration.md`).
    ///
    /// The caller must [`Self::register_timeline`] first. Turns must be
    /// restored in `turn_index` order so the appended `TurnIndex` matches
    /// the persisted one.
    ///
    /// `cold` carries the per-layer `StoredSequence` references the
    /// classifier needs to route the turn through `cold_to_hot` on
    /// the next `elevate_to_hot`.  Pass `None` for a recoverable
    /// turn whose chunks haven't landed on disk yet.  The reload
    /// path never installs hot — that's always demand-driven via
    /// `elevate_to_hot`.
    #[allow(clippy::too_many_arguments)]
    pub fn restore_turn(
        &mut self,
        timeline: TimelineId,
        layout: TurnLayout,
        token_ids: TokenBuffer,
        token_count: usize,
        cold: Option<Vec<StoredSequence>>,
        block_start: u64,
        block_end: u64,
    ) -> TurnIndex {
        let idx = self
            .timelines
            .get(&timeline)
            .expect("restore_turn: timeline must be registered first")
            .next_turn_index();
        let compression = self.timeline_compression.get(&timeline).copied();
        let residence = self.alloc_residence(turn_stream_id(timeline.raw(), idx.0), compression);
        {
            let tl = entry_mut(&mut self.timelines, &timeline).unwrap();
            tl.turns.insert(
                idx,
                Arc::new(TurnEntryData {
                    block_range: (block_start, block_end),
                    content: TurnPart {
                        layout,
                        token_count,
                        token_ids,
                        residence,
                    },
                }),
            );
            // Tree metadata: the `TreeMetadata` redo-log records were
            // already replayed during the walker's open pass (see
            // `apply_tree_metadata_payload`), so a summary node's
            // `SummaryOfTurns` / `SummaryOfSummaries` kind is ALREADY set
            // here. Only seed a default `Normal` when this turn has no
            // persisted tree meta at all — using `insert` unconditionally
            // would clobber every reloaded summary node back to `Normal`,
            // which collapses the summary tree on restart (the node spine
            // fills with `Normal` turns → AVL invariant violation), re-enqueues
            // the whole history for re-summarisation, and re-summarises prior
            // summaries into garbage. Restored turns are NOT pushed onto the
            // pending queue: that captures only fresh turns the live summariser
            // hasn't seen yet.
            tl.tree_meta.entry(idx).or_default();
        }
        *self.timeline_token_totals.entry(timeline).or_default() += token_count;
        if let Some(cold_seqs) = cold {
            if !cold_seqs.is_empty() {
                // The install sums the cold record bytes into the residence's
                // `byte_size`, so the purge accounting + telemetry has a real
                // number before any hot/warm elevate fires.
                self.install_cold(residence, cold_seqs);
            }
        }
        idx
    }

    /// Drop a turn's hot copy, freeing its VRAM arena chunks once any live
    /// borrowers release them.
    ///
    /// Returns `true` if the residence had hot bytes that were dropped;
    /// `false` when the turn isn't tracked or was already cold-marker. Callers
    /// use the boolean to decide whether to charge eviction accounting — they
    /// never need the dropped bytes themselves (the residence's warm/cold tiers
    /// hold the canonical copies if any future re-elevation needs them).
    pub fn clear_turn_sealed(&self, timeline: TimelineId, index: TurnIndex) -> bool {
        let Some(residence) = self.turn(timeline, index).map(|e| e.content.residence) else {
            return false;
        };
        self.residences.take_hot(residence).is_some()
    }
}
