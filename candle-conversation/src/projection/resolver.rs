//! [`Conversation`] — the workspace-shared substrate handle, and
//! [`TargetedRead`] — the target-aware [`ContentResolver`] wrapper.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use super::ids::{GroupId, LayerId, SectionId, TimelineAllocator, TimelineId, TurnIndex, TurnKey};
use super::project::ProjectionTarget;
use super::schema::{DepthWeights, ScoreFormula};
use crate::persistence::record::TreeMetadataPayload;
use crate::persistence::streams::{
    ContentAddress, PerDepthScores, SectionDecl, StreamDecl, StreamId, TurnDecl,
};
use crate::persistence::SubstratePersistence;
use crate::provenance::SigEntry;
use crate::substrate::{
    ContentResolver, ProjectionScores, StoredSequence, Substrate, SubstrateRead, SubstrateWrite,
    TurnPartWrite,
};
use crate::summary_tree::SelectionDiagnostics;
use crate::token_buffer::TokenBuffer;
use crate::turn::Role;
use candle_nn::kv_cache::SealedSequence;

// ── Conversation ──────────────────────────────────────────────────────────────

/// Workspace-shared, lock-protected handle to the per-turn record store.
///
/// Multiple conversations in the same workspace clone this handle; they all
/// see (and write into) the same underlying [`Substrate`].  Locking is
/// coarse-grained — one `RwLock` over the whole resolver — but scans and
/// mutations are short, so contention is minimal in practice.
///
/// # Phase 4 substrate semantics
///
/// - **Append (write)** — at seal time, each conversation appends its new
///   turn into the shared store.  Index allocation is per-group, monotonic,
///   under the resolver's lock.
/// - **Read** — projection takes a read guard for the duration of a single
///   `project()` call via [`Conversation::read`].  Returns a
///   [`SubstrateRead`] that implements [`ContentResolver`].
/// - **Reset** — does *not* clear the shared store (other conversations
///   would lose their history).  Sequence-level reset only drops local
///   KV state via the scheduler.
#[derive(Clone)]
pub struct Conversation {
    inner: Arc<RwLock<Substrate>>,
    allocator: Arc<TimelineAllocator>,
    /// The mandatory persistence layer — every turn is recorded into its
    /// redo log (`docs/kv_tier_migration.md` §13.6).
    persistence: Arc<Mutex<SubstratePersistence>>,
}

impl Default for Conversation {
    /// An ephemeral conversation — see [`Conversation::ephemeral`].
    fn default() -> Self {
        Self::ephemeral()
    }
}

impl Conversation {
    /// Create a fresh ephemeral conversation (throwaway temp-dir log).
    pub fn new() -> Self {
        Self::ephemeral()
    }

    /// An ephemeral conversation: its persistence layer is backed by a
    /// throwaway log in a unique temp directory. Used by tests and by
    /// transient helper conversations (e.g. summarisation).
    pub fn ephemeral() -> Self {
        static EPHEMERAL_SEQ: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let seq = EPHEMERAL_SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("zend_ephemeral_{nanos}_{seq}"));
        let mut substrate = Substrate::new();
        let persistence = SubstratePersistence::open_in_with_substrate(&dir, &mut substrate)
            .expect("ephemeral SubstratePersistence");
        Self {
            inner: Arc::new(RwLock::new(substrate)),
            allocator: Arc::new(TimelineAllocator::new()),
            persistence: Arc::new(Mutex::new(persistence)),
        }
    }

    /// Create a conversation from a freshly-built `(Substrate,
    /// SubstratePersistence)` pair.  Callers that want the walker to
    /// dispatch into the substrate in one pass should use
    /// [`SubstratePersistence::open_in_with_substrate`] and pass both
    /// here.
    pub fn from_parts(substrate: Substrate, persistence: SubstratePersistence) -> Self {
        Self {
            inner: Arc::new(RwLock::new(substrate)),
            allocator: Arc::new(TimelineAllocator::new()),
            persistence: Arc::new(Mutex::new(persistence)),
        }
    }

    /// Create a conversation backed by a real [`SubstratePersistence`].
    /// Equivalent to `from_parts(Substrate::new(), persistence)` — for
    /// callers that already have a populated persistence and want an
    /// empty substrate.
    pub fn with_persistence(persistence: SubstratePersistence) -> Self {
        Self::from_parts(Substrate::new(), persistence)
    }

    /// Allocate a fresh [`TimelineId`] and register it against
    /// `(layer, group)` on the substrate.
    pub fn mint_timeline(&self, layer: LayerId, group: GroupId) -> TimelineId {
        let mut view = self.inner.write().unwrap();
        view.mint_timeline(layer, group, &self.allocator)
    }

    /// Look up `(layer, group)` for a previously-minted timeline.
    pub fn timeline_target(&self, timeline: TimelineId) -> Option<(LayerId, GroupId)> {
        self.inner.read().unwrap().timeline_target(timeline)
    }

    /// Register a specific [`TimelineId`] against `(layer, group)` —
    /// idempotent. Used by the resume path to bind a conversation to a
    /// timeline recovered from the redo log instead of minting a fresh one.
    pub fn register_timeline(&self, timeline: TimelineId, layer: LayerId, group: GroupId) {
        self.inner
            .write()
            .unwrap()
            .register_timeline(timeline, layer, group);
    }

    /// Acquire an unscored read guard.  The returned guard implements
    /// [`ContentResolver`] but every score lookup returns zero —
    /// appropriate for callers reading structural fields (turn counts,
    /// sealed pointers) without projection.
    ///
    /// Use [`Self::read_scored`] when projecting against a freshly-built
    /// [`ProjectionScores`] from a BDP scan.
    pub fn read(&self) -> SubstrateRead<'_> {
        SubstrateRead {
            guard: self.inner.read().unwrap(),
            scores: None,
        }
    }

    /// Acquire a read guard bound to an externally-owned
    /// [`ProjectionScores`]. The scores are transient per-projection
    /// state — typically populated by the BDP scanner on the call site's
    /// stack and dropped at end of scope. They are **not** held by the
    /// substrate.
    pub fn read_scored<'a>(&'a self, scores: &'a ProjectionScores) -> SubstrateRead<'a> {
        SubstrateRead {
            guard: self.inner.read().unwrap(),
            scores: Some(scores),
        }
    }

    /// Acquire a target-aware read guard.  The returned [`TargetedRead`]
    /// implements [`ContentResolver`] with proper sibling-timeline masking
    /// for `target.group`, with score lookups returning zero (unscored).
    pub fn read_for(&self, target: ProjectionTarget) -> TargetedRead<'_> {
        TargetedRead::new(self.read(), target)
    }

    /// Target-aware variant of [`Self::read_scored`].
    pub fn read_for_scored<'a>(
        &'a self,
        target: ProjectionTarget,
        scores: &'a ProjectionScores,
    ) -> TargetedRead<'a> {
        TargetedRead::new(self.read_scored(scores), target)
    }

    /// Acquire a write guard for mutating operations (append, set_*).
    pub fn write(&self) -> SubstrateWrite<'_> {
        SubstrateWrite {
            guard: self.inner.write().unwrap(),
        }
    }

    /// Atomically append a turn to the substrate.
    ///
    /// `write` carries the turn's text, token IDs, block range, and
    /// GPU-resident sealed K/V snapshot.  `migrate_to_cpu` is called
    /// to move the bytes to the warm (CPU) tier before storing.
    pub fn record_turn(
        &self,
        timeline: TimelineId,
        role: Role,
        write: TurnPartWrite,
        migrate_to_cpu: impl FnMut(&[SealedSequence]) -> candle::Result<Vec<SealedSequence>>,
    ) -> candle::Result<TurnIndex> {
        let block_start = write.block_start;
        let block_end = write.block_end;
        // Capture the per-half text strings before the write moves
        // into the substrate — the redo-log `TurnDecl` carries them
        // verbatim so reload can re-populate `TurnPart::user_text` /
        // `assistant_text` without re-tokenising or scanning.
        let user_text = write.user_text.clone();
        let assistant_text = write.assistant_text.clone();
        let idx = {
            let mut view = self.inner.write().unwrap();
            view.append_complete(timeline, write, migrate_to_cpu)?
        };
        // Record the turn's structure into the redo log.
        let (layer_id, group_id) = self
            .timeline_target(timeline)
            .map(|(l, g)| (l.raw(), g.raw()))
            .unwrap_or((0, 0));
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: timeline.raw(),
            turn_index: idx.0,
            turn_id_day: 0,
            turn_id_seq: idx.0 + 1,
            role: match role {
                Role::System => 0,
                Role::User => 1,
                Role::Assistant => 2,
            },
            block_start,
            block_end,
            layer_id,
            group_id,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            scores: PerDepthScores::default(),
            user_chunk_count: 0,
            user_token_count: 0,
            user_sig_count: 0,
            user_text,
            assistant_text,
        });
        self.persistence
            .lock()
            .unwrap()
            .declare_stream(&decl)
            .map_err(|e| candle::Error::Msg(format!("persist turn: {e}")))?;
        Ok(idx)
    }

    /// Append a summariser-allocated turn (SoT leaf or SoS internal)
    /// and persist its declaration to the redo log.
    ///
    /// The summariser allocates these turns to back tree nodes; they
    /// carry no KV chunks (`block_range = 0..0`) and `token_count` is
    /// the placeholder for the summary text.  Without persistence, on
    /// reopen the walker would replay [`TreeMetadata`] records for
    /// these indices but find no matching [`TurnDecl`], leaving
    /// orphan `tree_meta` entries that the score-density selector
    /// would then pick and elevate would fail to lift.
    ///
    /// Drops the auto-pending entry that [`append_with_blocks`] pushed
    /// — summary turns are not Normal and shouldn't loop back through
    /// the pending queue.
    pub fn record_summary_turn(
        &self,
        timeline: TimelineId,
        token_count: usize,
    ) -> candle::Result<TurnIndex> {
        let idx = {
            let mut view = self.inner.write().unwrap();
            let idx = view.append_with_blocks(timeline, token_count, 0, 0);
            view.pop_pending_summary(timeline);
            idx
        };
        let (layer_id, group_id) = self
            .timeline_target(timeline)
            .map(|(l, g)| (l.raw(), g.raw()))
            .unwrap_or((0, 0));
        let decl = StreamDecl::Turn(TurnDecl {
            timeline_id: timeline.raw(),
            turn_index: idx.0,
            turn_id_day: 0,
            turn_id_seq: idx.0 + 1,
            role: 2,
            block_start: 0,
            block_end: 0,
            layer_id,
            group_id,
            anchored_prefix: Vec::new(),
            view: Vec::new(),
            scores: PerDepthScores::default(),
            user_chunk_count: 0,
            user_token_count: 0,
            user_sig_count: 0,
            user_text: String::new(),
            assistant_text: String::new(),
        });
        self.persistence
            .lock()
            .unwrap()
            .declare_stream(&decl)
            .map_err(|e| candle::Error::Msg(format!("persist summary turn: {e}")))?;
        Ok(idx)
    }

    /// Rebuild the in-RAM [`Substrate`] from the persistence redo log — the
    /// §5.6 / §16.12 substrate-reload path run on daemon restart.
    ///
    /// **Cold-only restart.** The substrate is, by design, the on-disk redo
    /// log; warm (RAM) and hot (VRAM) tiers belong to the inference engine
    /// and are demand-populated. Reload therefore:
    /// - Walks every persisted turn stream in `(timeline, turn_index)`
    ///   order.
    /// - Replays **tokens** (for text history) and **BDP signatures** (for
    ///   provenance retrieval over the full persisted corpus) into the
    ///   in-RAM substrate. Sigs are small (RAM-resident) and load-bearing
    ///   for the next BDP scan; tokens are small (RAM-resident) and
    ///   load-bearing for text display.
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
        restore_sigs: impl Fn(&[(u16, Vec<u8>)]) -> candle::Result<Vec<SigEntry>>,
    ) -> candle::Result<usize> {
        // Substrate's per-stream / per-timeline state was populated
        // in one walker pass during `SubstratePersistence::open_in_with_substrate`
        // — no mirror step needed here.  This pass replays turn-decl
        // records into the substrate's per-turn KV residence slots
        // (the cold-load setup that demands knowing layer count) and
        // then runs the post-reload sweeps for the summary tree.
        let decls = {
            let substrate = self.read();
            crate::persistence::resume::recovered_turn_decls(&substrate)
        };
        let mut restored = 0usize;
        for mut decl in decls {
            let (recovered, cold_refs) = {
                let mut p = self.persistence.lock().unwrap();
                let substrate_read = self.read();
                let recovered = crate::persistence::resume::recover_turn(
                    &mut p,
                    &substrate_read,
                    &decl,
                    n_layers,
                )
                .map_err(|e| candle::Error::Msg(format!("recover turn: {e}")))?;
                let cold_refs = crate::persistence::resume::recover_turn_cold_refs(
                    &substrate_read,
                    &decl,
                    n_layers,
                )
                .map_err(|e| candle::Error::Msg(format!("recover cold refs: {e}")))?;
                (recovered, cold_refs)
            };
            // Re-append the BDP signatures into the (fresh) provenance file,
            // yielding entries that point at the rebuilt offsets. Sigs are
            // load-bearing — the BDP scan operates on signatures, not KV.
            let sig_entries = if recovered.signatures.is_empty() {
                Vec::new()
            } else {
                restore_sigs(&recovered.signatures)?
            };
            let timeline = TimelineId::from_raw(decl.timeline_id).ok_or_else(|| {
                candle::Error::Msg("reconstruct: turn has zero timeline_id".into())
            })?;
            let token_count: usize = if recovered.layers.n_layers() == 0 {
                0
            } else {
                recovered
                    .layers
                    .layer(0)
                    .iter()
                    .map(|c| c.token_count as usize)
                    .sum()
            };
            let mut view = self.write();
            if let (Some(layer), Some(group)) = (
                LayerId::from_raw(decl.layer_id),
                GroupId::from_raw(decl.group_id),
            ) {
                view.register_timeline(timeline, layer, group);
            }
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
            let idx = view.restore_turn(
                timeline,
                std::mem::take(&mut decl.user_text),
                std::mem::take(&mut decl.assistant_text),
                TokenBuffer::from(recovered.token_ids),
                token_count,
                cold_refs,
                decl.block_start,
                decl.block_end,
            );
            if !sig_entries.is_empty() {
                view.set_sig_entries(timeline, idx, sig_entries);
            }
            restored += 1;
        }
        // Post-walker sweeps on substrate state (no manifest reads —
        // all per-entity state was written during the open-time
        // walker pass into `substrate.apply_walker_entry`).
        //
        // 1. Re-enqueue orphan Normal turns onto the summariser's
        //    pending queue.  An orphan is any Normal turn whose index
        //    is NOT in any SummaryOfTurns leaf's children list — i.e.
        //    a crash interrupted its absorption before a leaf was
        //    sealed over it.  Compute the "covered" set by walking
        //    summary nodes' children.
        // 2. Re-seed the `dirty_summary_set` from `dirty: true`
        //    `TreeNodeMeta` entries — the apply path writes the flag
        //    but doesn't index into the dirty set; this sweep does.
        let n_meta_records = 0usize;
        let n_state_records = 0usize;
        let n_meta_applied = 0usize;
        let n_meta_dropped_unregistered = 0usize;
        let n_state_applied = 0usize;
        {
            let mut view = self.write();
            let timeline_ids: Vec<TimelineId> = view.all_timeline_ids().collect();
            for timeline in timeline_ids {
                // Build the "covered by a summary leaf" set from the
                // substrate's tree_meta — every Normal turn that's a
                // child of some SoT leaf is covered; every Normal
                // outside that set is orphaned.
                let mut covered: std::collections::BTreeSet<u32> =
                    std::collections::BTreeSet::new();
                for leaf_idx in view.summary_leaves_chrono(timeline) {
                    if let Some(meta) = view.tree_meta_of(timeline, leaf_idx) {
                        for c in &meta.children {
                            covered.insert(c.0);
                        }
                    }
                }
                for normal_idx in view.normal_turns_chrono(timeline) {
                    if !covered.contains(&normal_idx.0) {
                        view.push_pending_summary(timeline, normal_idx);
                    }
                }
                // Re-seed the dirty set from `dirty: true` meta entries.
                let dirty_ids: Vec<TurnIndex> = view
                    .summary_leaves_chrono(timeline)
                    .into_iter()
                    .chain(view.summary_internals_chrono(timeline).into_iter())
                    .filter(|idx| {
                        view.tree_meta_of(timeline, *idx)
                            .map(|m| m.dirty)
                            .unwrap_or(false)
                    })
                    .collect();
                for id in dirty_ids {
                    view.mark_summary_dirty(timeline, id);
                }
            }
        }
        let read = self.read();
        let n_sections = read.section_count();
        let n_timelines = read.timeline_count();
        let n_conversations = read.conversation_count();
        drop(read);
        tracing::info!(
            sections = n_sections,
            timelines = n_timelines,
            conversations = n_conversations,
            turns = restored,
            label_records = n_meta_records,
            label_records_applied = n_meta_applied,
            label_records_dropped = n_meta_dropped_unregistered,
            conv_state_records = n_state_records,
            conv_state_records_applied = n_state_applied,
            "substrate reload complete",
        );
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
        use crate::persistence::resume::{recover_turn, recovered_turn_decls};
        let stream_id = crate::persistence::content_hash::turn_stream_id(timeline.raw(), index.0);
        let mut p = self.persistence.lock().unwrap();
        // We need the turn's `StreamDecl` to drive `recover_turn`. Walk
        // the substrate's persisted decls and pick the one matching this
        // (timeline, index). The decl set is small and rebuilt once at
        // restart, so a linear scan is fine.
        let substrate_read = self.read();
        let decls = recovered_turn_decls(&substrate_read);
        let decl = match decls
            .into_iter()
            .find(|d| d.timeline_id == timeline.raw() && d.turn_index == index.0)
        {
            Some(d) => d,
            None => return Ok(None),
        };
        let substrate = self.read();
        let recovered = recover_turn(&mut p, &substrate, &decl, n_layers)
            .map_err(|e| candle::Error::Msg(format!("recover_turn_chunks: {e}")))?;
        if recovered.layers.is_empty() {
            return Ok(None);
        }
        let _ = stream_id; // (computed for diagnostics if needed later)
        Ok(Some(recovered.layers))
    }

    /// Cold-load fast path that fuses `recover_turn_chunks` + `load_to_hot`
    /// into a single batched pipeline using pinned host scratch
    /// throughout — see `transfer::load_turn_into_hot` for the pipeline.
    ///
    /// Returns:
    ///  - `Ok(None)` if no `TurnDecl` matches `(timeline, index)`
    ///    (chunks haven't landed; same semantics as `recover_turn_chunks`).
    ///  - `Ok(Some((sealed_per_layer, kv_bytes_total)))` on success;
    ///    `kv_bytes_total` is the sum of every chunk's `kv_bytes` length
    ///    (the warm-LRU / cold-budget accounting unit).
    pub fn cold_load_turn_into_hot(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
        backings: &[candle_nn::kv_cache::ChunkedKvBacking],
        device: &candle::Device,
        stager: &mut crate::persistence::cold_load::ColdLoadStager,
    ) -> candle::Result<Option<(Vec<SealedSequence>, u64)>> {
        use crate::persistence::resume::recovered_turn_decls;
        use crate::persistence::transfer::load_turn_into_hot;
        let mut p = self.persistence.lock().unwrap();
        let decls = {
            let substrate = self.read();
            recovered_turn_decls(&substrate)
        };
        let decl = match decls
            .into_iter()
            .find(|d| d.timeline_id == timeline.raw() && d.turn_index == index.0)
        {
            Some(d) => d,
            None => return Ok(None),
        };
        let substrate = self.read();
        let sealed = load_turn_into_hot(backings, device, &mut p, &substrate, &decl, stager)?;
        // Accounting bytes: sum of every chunk's `kv_bytes` size across
        // every layer in the substrate's stream snapshot — matches the
        // previous `TurnChunkGrid::bytes()` semantics.
        let stream_id = crate::persistence::content_hash::turn_stream_id(timeline.raw(), index.0);
        let kv_bytes_total: u64 = substrate
            .stream_of(stream_id)
            .map(|s| {
                s.chunks
                    .values()
                    .map(|loc| loc.payload_len.saturating_sub(0))
                    .sum::<u64>()
            })
            .unwrap_or(0);
        // `payload_len` is the whole ChunkPayload-encoded size (offset +
        // formats + pal + scales + kv_bytes + length-prefix overhead), so
        // it slightly over-counts vs. `kv_bytes` alone. For LRU/budget
        // accounting that's the better signal anyway (it tracks the
        // bytes the warm-tier writeback will produce on the next persist
        // pass). Keep the simple total here; the old `grid.bytes()`
        // delta is small.
        Ok(Some((sealed, kv_bytes_total)))
    }

    /// Clear a turn's hot sealed grid, releasing VRAM arena chunks via
    /// dropping its ChunkGid Arcs. Returns `true` if hot bytes were
    /// dropped (see [`Substrate::clear_turn_sealed`]).
    pub fn clear_turn_sealed(&self, timeline: TimelineId, index: TurnIndex) -> bool {
        self.write().clear_turn_sealed(timeline, index)
    }

    /// Hot-tier VRAM byte snapshot (sum across every turn whose `sealed`
    /// carries an actual chunk grid).
    pub fn hot_turn_bytes(&self) -> usize {
        self.read().hot_turn_bytes()
    }

    /// Pinned-section byte snapshot.
    pub fn section_bytes(&self) -> usize {
        self.read().section_bytes()
    }

    /// Byte size of a single hot turn (for the pre-flight evict
    /// accounting). `None` if cold or unknown.
    pub fn turn_hot_bytes(&self, timeline: TimelineId, index: TurnIndex) -> Option<usize> {
        self.read().turn_hot_bytes(timeline, index)
    }

    /// FIFO-oldest hot-resident turn excluding `except`.
    pub fn oldest_hot_turn_except(&self, except: TurnKey) -> Option<TurnKey> {
        self.read().oldest_hot_turn_except(except)
    }

    /// The sidebar label for `timeline`, or `None` if no label has been
    /// recorded — either the conversation hasn't had its first user turn
    /// yet, or the recovered redo log carries no label for it.
    pub fn label_of(&self, timeline: TimelineId) -> Option<String> {
        self.read().label_of(timeline).map(|s| s.to_string())
    }

    /// The client-supplied `conv_id` string for `timeline`, or `None` if
    /// no submit has been recorded yet. Recovered from the redo log on
    /// substrate reload — drives the daemon's sidebar id field.
    pub fn conv_id_of(&self, timeline: TimelineId) -> Option<String> {
        self.read().conv_id_of(timeline).map(|s| s.to_string())
    }

    /// Persist a sidebar label for `timeline`. Last-write-wins on the
    /// underlying `RecordType::Label`; this writes the same record the
    /// titler writes, preserving whatever `conv_id` is already known
    /// for this timeline.
    pub fn set_conversation_label(&self, timeline: TimelineId, label: &str) -> candle::Result<()> {
        if label.is_empty() {
            return Ok(());
        }
        let conv_id = self.conv_id_of(timeline).unwrap_or_default();
        {
            let mut p = self.persistence.lock().unwrap();
            p.write_conv_meta(timeline.raw(), &conv_id, label)
                .map_err(|e| candle::Error::Msg(format!("write_conv_meta: {e}")))?;
        }
        self.write().set_label(timeline, label);
        Ok(())
    }

    /// Persist the client-supplied `conv_id` for `timeline`. Idempotent;
    /// the typical caller is the daemon's chat handler, invoking this on
    /// every submit so the conv_id reaches the redo log immediately
    /// (well before the titler completes). The current `label` is
    /// preserved, so this can be called freely at any point in the
    /// conversation's lifecycle.
    pub fn set_conversation_conv_id(
        &self,
        timeline: TimelineId,
        conv_id: &str,
    ) -> candle::Result<()> {
        if conv_id.is_empty() {
            return Ok(());
        }
        let label = self.read().label_of(timeline).unwrap_or("").to_string();
        {
            let mut p = self.persistence.lock().unwrap();
            p.write_conv_meta(timeline.raw(), conv_id, &label)
                .map_err(|e| candle::Error::Msg(format!("write_conv_meta: {e}")))?;
        }
        self.write().set_conv_id(timeline, conv_id);
        Ok(())
    }

    /// Every conversation the workspace substrate knows about —
    /// `(timeline, conv_id, label, archived)` quads drawn from the
    /// in-RAM `Substrate::timelines` map. Drives
    /// `GET /v1/conversations` directly; no sidecar involved.
    pub fn known_conversations(&self) -> Vec<(TimelineId, String, String, bool)> {
        self.read().known_conversations()
    }

    /// Set a conversation's `archived` lifecycle flag and persist it
    /// as a `RecordType::ConvState` record. Idempotent: if the
    /// substrate already holds the requested state, the record is
    /// not written and the call returns `Ok(())` without touching the
    /// log.
    ///
    /// Last-write-wins on replay — toggling archive↔unarchive each
    /// appends one small record (~ 16 bytes payload + framing); a
    /// subsequent compaction collapses the chain to one record per
    /// timeline.
    pub fn set_conversation_archived(
        &self,
        timeline: TimelineId,
        archived: bool,
    ) -> candle::Result<()> {
        let changed = self.write().set_archived(timeline, archived);
        if !changed {
            return Ok(());
        }
        let state = crate::persistence::manifest::ConvState { archived };
        let mut p = self.persistence.lock().unwrap();
        p.write_conv_state(timeline.raw(), state)
            .map_err(|e| candle::Error::Msg(format!("write_conv_state: {e}")))?;
        Ok(())
    }

    /// Whether `timeline` is currently archived. Untouched / unknown
    /// timelines return `false`.
    pub fn is_conversation_archived(&self, timeline: TimelineId) -> bool {
        self.read().is_archived(timeline)
    }

    /// Set the substrate-side resume key (`debug_id`) for `timeline`
    /// and persist a `RecordType::DebugId` record to the redo log.
    /// Last-write-wins on replay.  Idempotent: if the substrate
    /// already holds the requested key, the record is not written and
    /// the call returns `Ok(())` without touching the log.
    pub fn set_conversation_debug_id(
        &self,
        timeline: TimelineId,
        debug_id: &str,
    ) -> candle::Result<()> {
        if debug_id.is_empty() {
            return Ok(());
        }
        self.write().set_debug_id(timeline, debug_id);
        let mut p = self.persistence.lock().unwrap();
        p.write_debug_id(timeline.raw(), debug_id)
            .map_err(|e| candle::Error::Msg(format!("write_debug_id: {e}")))?;
        Ok(())
    }

    /// Look up a timeline by `debug_id`.  O(1).
    pub fn lookup_by_debug_id(&self, debug_id: &str) -> Option<TimelineId> {
        self.read().lookup_by_debug_id(debug_id)
    }

    /// Number of turns currently waiting on the summariser thread to
    /// absorb them into the summary tree (§9 backpressure metric).
    /// `0` means steady state: the background tempo is keeping up
    /// with the foreground turn rate.
    pub fn pending_summary_len(&self, timeline: TimelineId) -> usize {
        self.read().pending_summary_len(timeline)
    }

    /// Number of summary nodes currently marked dirty (children
    /// changed since last regeneration).  `0` means the dirty sweep
    /// has caught up.
    pub fn dirty_summary_len(&self, timeline: TimelineId) -> usize {
        self.read().dirty_summary_len(timeline)
    }

    /// Most recent score-density [`SelectionDiagnostics`] for
    /// `timeline`, or `None` if no projection has run yet (or the
    /// projection used the rule-based path).  Pure test-harness
    /// instrumentation: the substrate retains only the latest
    /// selection per timeline, written by the scheduler at projection
    /// time.  Production daemons can ignore.
    pub fn last_selection_diagnostics(
        &self,
        timeline: TimelineId,
    ) -> Option<SelectionDiagnostics> {
        self.read().last_selection_of(timeline).cloned()
    }

    /// Persist a per-`(timeline, turn_index)` summary-tree metadata
    /// record to the redo log.  Idempotent: skips the append when the
    /// in-memory manifest already records the same payload.  Called
    /// by the summariser thread after every atomic tree mutation
    /// (§7.2).
    pub fn write_tree_metadata(
        &self,
        payload: TreeMetadataPayload,
    ) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        p.write_tree_metadata(payload)
            .map_err(|e| candle::Error::Msg(format!("write_tree_metadata: {e}")))
    }

    /// Persist a sealed turn's per-layer KV grid + token ids to the redo log
    /// — the seal-time half of the resume path (§16.12). All layers share
    /// one chunk count.
    pub fn persist_turn_kv(
        &self,
        stream_id: StreamId,
        layers: &crate::persistence::resume::TurnChunkGrid,
        token_ids: &[u32],
    ) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        crate::persistence::resume::persist_turn_kv(&mut p, stream_id, layers, token_ids)
            .map_err(|e| candle::Error::Msg(format!("persist turn kv: {e}")))
    }

    /// Persist only a turn's per-layer chunk records — the post-quantization
    /// half of the async seal/persist chain. Called from inside the
    /// bg-quantizer callback once float→quant migrations have landed.
    pub fn persist_turn_chunks(
        &self,
        stream_id: StreamId,
        layers: &crate::persistence::resume::TurnChunkGrid,
    ) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        crate::persistence::resume::persist_turn_chunks(&mut p, stream_id, layers)
            .map_err(|e| candle::Error::Msg(format!("persist turn chunks: {e}")))
    }

    /// Persist a turn's chunks and return the per-layer [`StoredSequence`]
    /// references — the warm→cold leg of the persistence thread's
    /// `run_pass`. The returned references go straight into the
    /// substrate via `Substrate::install_cold`.
    pub fn persist_turn_chunks_capture(
        &self,
        stream_id: StreamId,
        layers: &crate::persistence::resume::TurnChunkGrid,
    ) -> candle::Result<Vec<StoredSequence>> {
        let mut p = self.persistence.lock().unwrap();
        crate::persistence::resume::persist_turn_chunks_capture(&mut p, stream_id, layers)
            .map_err(|e| candle::Error::Msg(format!("persist turn chunks capture: {e}")))
    }

    /// Persist a turn's `Tokens` record and the trailing `Commit` — always
    /// called synchronously on seal, regardless of compression policy.
    /// `layers` is only used to compute the highest chunk index; pass an
    /// empty grid when no chunks were persisted (compression `None` path).
    pub fn persist_turn_tokens(
        &self,
        stream_id: StreamId,
        token_ids: &[u32],
        layers: &crate::persistence::resume::TurnChunkGrid,
    ) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        crate::persistence::resume::persist_turn_tokens(&mut p, stream_id, token_ids, layers)
            .map_err(|e| candle::Error::Msg(format!("persist turn tokens: {e}")))
    }

    /// Persist a turn's `Tokens` record only — no trailing `Commit`.
    /// Used by the seal path now that chunks (and the matching Commit)
    /// are written asynchronously by the persistence thread.
    pub fn persist_tokens_only(
        &self,
        stream_id: StreamId,
        token_ids: &[u32],
    ) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        crate::persistence::resume::persist_tokens_only(&mut p, stream_id, token_ids)
            .map_err(|e| candle::Error::Msg(format!("persist tokens: {e}")))
    }

    /// Append a stream-level `Commit` record at the given chunk index — the
    /// post-chunks re-commit used by the async seal/persist chain to upgrade
    /// the manifest's `committed_through` once the heavy `Chunks` records
    /// have been written.
    pub fn commit_stream_through(
        &self,
        stream_id: StreamId,
        through_index: u64,
    ) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        p.commit_stream(stream_id, through_index)
            .map_err(|e| candle::Error::Msg(format!("commit stream: {e}")))
    }

    /// Persist the projection schema/template into the substrate's
    /// `Template` record — compare-and-insert (only appends when it differs
    /// from what the log already holds), then commit if written. Lets the
    /// log carry the projection needed to reconstruct the substrate.
    pub fn set_template(&self, template: &[u8]) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        let wrote = p
            .set_template(template)
            .map_err(|e| candle::Error::Msg(format!("persist template: {e}")))?;
        if wrote {
            p.commit()
                .map_err(|e| candle::Error::Msg(format!("commit template: {e}")))?;
        }
        Ok(())
    }

    /// Persist a turn's BDP provenance signatures to the redo log — the
    /// `Signatures` record. `sigs` is the [`crate::persistence::resume::encode_signatures`]
    /// payload.
    pub fn persist_signatures(&self, stream_id: StreamId, sigs: &[u8]) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        p.append_signatures(stream_id, sigs)
            .map_err(|e| candle::Error::Msg(format!("persist signatures: {e}")))
    }

    /// Declare a section stream — appends a `StreamDecl::PromptSection`
    /// record carrying the content address and debug name.  The
    /// derived stream id matches `section_stream_id(address)`.  Called
    /// by the scheduler at section seal time; pairs with later
    /// `Tokens` / `Signatures` / `Chunks` records keyed by the same id.
    pub fn declare_section_stream(
        &self,
        address: ContentAddress,
        debug_name: &str,
    ) -> candle::Result<StreamId> {
        let decl = StreamDecl::PromptSection(SectionDecl {
            address,
            debug_name: debug_name.to_string(),
        });
        self.persistence
            .lock()
            .unwrap()
            .declare_stream(&decl)
            .map_err(|e| candle::Error::Msg(format!("declare section stream: {e}")))
    }

    /// True when the workspace's manifest already holds durable
    /// chunks for `stream_id` — i.e. a section under this content
    /// address has been persisted and can be cold-loaded back into
    /// hot without re-prefilling.  The check matches the ingest
    /// loop's skip-if-present gate.
    pub fn section_stream_is_persisted(&self, stream_id: StreamId) -> bool {
        drop(self.persistence.lock().unwrap());
        self.read()
            .stream_of(stream_id)
            .map(|s| s.committed_through.is_some() && !s.chunks.is_empty())
            .unwrap_or(false)
    }

    /// Snapshot a persisted section stream's manifest metadata for the
    /// cold-load path.  Returns `(chunks_per_layer, tokens_present,
    /// signatures_present)` when the stream is known, otherwise `None`.
    /// `chunks_per_layer = manifest.chunks.len() / n_layers`.
    pub fn section_stream_layout(
        &self,
        stream_id: StreamId,
        n_layers: usize,
    ) -> Option<(usize, bool, bool)> {
        drop(self.persistence.lock().unwrap());
        let substrate = self.read();
        let entry = substrate.stream_of(stream_id)?;
        if entry.chunks.is_empty() || n_layers == 0 {
            return None;
        }
        let total = entry.chunks.len();
        if total % n_layers != 0 {
            return None;
        }
        let chunks_per_layer = total / n_layers;
        Some((chunks_per_layer, entry.tokens.is_some(), entry.signatures.is_some()))
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
        use crate::persistence::transfer::load_stream_into_hot;
        let mut p = self.persistence.lock().unwrap();
        let substrate = self.read();
        load_stream_into_hot(
            backings,
            device,
            &mut p,
            &substrate,
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

    /// Look up a section by the human-readable `debug_name` recorded
    /// on its `SectionDecl`.  Wrapper around
    /// [`Substrate::section_id_for_debug_name`].  Used by calibration
    /// consumers that pick scenarios out of a loaded workspace by id.
    pub fn section_id_for_debug_name(&self, debug_name: &str) -> Option<SectionId> {
        self.read().section_id_for_debug_name(debug_name)
    }

    /// Read a persisted section's `Signatures` record from disk and
    /// decode it into the `(token_count, raw_bytes)` tuples the BDP
    /// scanner re-ingests.  Returns an empty Vec if the section has
    /// no signatures recorded.
    pub fn read_section_signatures(
        &self,
        stream_id: StreamId,
    ) -> candle::Result<Vec<(u16, Vec<u8>)>> {
        let mut p = self.persistence.lock().unwrap();
        let substrate = self.read();
        let bytes = match p.read_signatures(&substrate, stream_id) {
            Ok(Some(b)) => b,
            Ok(None) => return Ok(Vec::new()),
            Err(e) => return Err(candle::Error::Msg(format!("read section sigs: {e}"))),
        };
        crate::persistence::resume::decode_signatures(&bytes)
            .map_err(|e| candle::Error::Msg(format!("decode section sigs: {e}")))
    }

    /// Durably flush the persistence redo log — the group-commit point.
    /// `fsync`s every staged record so an in-flight turn survives a crash.
    pub fn commit_persistence(&self) -> candle::Result<()> {
        self.persistence
            .lock()
            .unwrap()
            .commit()
            .map_err(|e| candle::Error::Msg(format!("persist commit: {e}")))
    }

    /// Like [`Self::commit_persistence`] but a no-op when nothing is
    /// staged. Returns `Ok(true)` when an `fsync` actually happened.
    /// Used by the daemon's 5-second flush task so a quiescent
    /// workspace doesn't issue pointless syscalls.
    pub fn commit_persistence_if_pending(&self) -> candle::Result<bool> {
        self.persistence
            .lock()
            .unwrap()
            .commit_if_pending()
            .map_err(|e| candle::Error::Msg(format!("persist commit_if_pending: {e}")))
    }

    /// Flush and write a `Checkpoint` over the substrate manifest — the
    /// fast-recovery snapshot. Compacts the log first when it has accrued
    /// enough dead weight (§5.8).
    pub fn checkpoint_persistence(&self) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        p.commit()
            .map_err(|e| candle::Error::Msg(format!("persist commit: {e}")))?;
        let should = {
            let substrate = self.read();
            p.should_compact(&substrate)
                .map_err(|e| candle::Error::Msg(format!("persist compaction check: {e}")))?
        };
        if should {
            let mut substrate = self.write();
            p.compact(&mut substrate)
                .map_err(|e| candle::Error::Msg(format!("persist compaction: {e}")))?;
        }
        p.checkpoint()
            .map_err(|e| candle::Error::Msg(format!("persist checkpoint: {e}")))
    }

    /// Run `f` against the persistence layer's current manifest snapshot.
    /// Read-only accessor for callers that need to inspect the redo
    /// log's stream/chunk locations (sizes, formats, offsets) without
    /// rebuilding the whole substrate boundary.
    pub fn with_persistence_manifest<R>(
        &self,
        f: impl FnOnce(&crate::persistence::manifest::Manifest) -> R,
    ) -> R {
        let p = self.persistence.lock().unwrap();
        f(p.manifest())
    }
}

// ── TargetedRead ──────────────────────────────────────────────────────────────

/// Target-aware [`ContentResolver`] wrapper around a [`SubstrateRead`].
///
/// For `target.group`: only `target.timeline` is visible; sibling timelines
/// are masked.  For other groups: the first-registered timeline is used
/// (Phase 3 simplification for groups with a single shared timeline).
/// Sections are workspace singletons and pass straight through.
pub struct TargetedRead<'a> {
    read: SubstrateRead<'a>,
    target: ProjectionTarget,
}

impl<'a> TargetedRead<'a> {
    pub fn new(read: SubstrateRead<'a>, target: ProjectionTarget) -> Self {
        Self { read, target }
    }

    fn timeline_for(&self, group: GroupId) -> Option<TimelineId> {
        if group == self.target.group {
            Some(self.target.timeline)
        } else {
            self.read.timelines_for_group(group).next()
        }
    }
}

impl<'a> std::ops::Deref for TargetedRead<'a> {
    type Target = Substrate;
    fn deref(&self) -> &Substrate {
        &self.read
    }
}

impl<'a> ContentResolver for TargetedRead<'a> {
    fn turn_count(&self, group: GroupId) -> u32 {
        let Some(timeline) = self.timeline_for(group) else {
            return 0;
        };
        Substrate::turn_count(&self.read, timeline)
    }

    fn turn_token_count(&self, group: GroupId, index: TurnIndex) -> usize {
        let Some(timeline) = self.timeline_for(group) else {
            return 0;
        };
        self.read.turn_token_count_of(timeline, index)
    }

    fn turn_score(
        &self,
        group: GroupId,
        index: TurnIndex,
        formula: ScoreFormula,
        weights: &DepthWeights,
    ) -> f32 {
        let Some(timeline) = self.timeline_for(group) else {
            return 0.0;
        };
        self.read
            .turn_score_for_timeline(timeline, index, formula, weights)
    }

    fn turn_origin(&self, group: GroupId, _index: TurnIndex) -> Option<LayerId> {
        let timeline = self.timeline_for(group)?;
        let (layer, _) = self.read.timeline_target(timeline)?;
        Some(layer)
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        ContentResolver::section_token_count(&self.read, section)
    }

    fn section_score(
        &self,
        section: SectionId,
        formula: ScoreFormula,
        weights: &DepthWeights,
    ) -> f32 {
        ContentResolver::section_score(&self.read, section, formula, weights)
    }
}
