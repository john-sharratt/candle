//! [`Conversation`] — the workspace-shared substrate handle, and
//! [`TargetedRead`] — the target-aware [`ContentResolver`] wrapper.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use super::ids::{GroupId, LayerId, SectionId, TimelineAllocator, TimelineId, TurnIndex};
use super::schema::{DepthWeights, ScoreFormula};
use crate::persistence::streams::{PerDepthScores, StreamDecl, StreamId, TurnDecl};
use crate::persistence::SubstratePersistence;
use crate::provenance::SigEntry;
use crate::substrate::{ContentResolver, Substrate, SubstrateRead, SubstrateWrite};
use crate::substrate_cache::SubstrateCache;
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
        let persistence =
            SubstratePersistence::open_in(&dir).expect("ephemeral SubstratePersistence");
        Self {
            inner: Arc::new(RwLock::new(Substrate::new())),
            allocator: Arc::new(TimelineAllocator::new()),
            persistence: Arc::new(Mutex::new(persistence)),
        }
    }

    /// Create a conversation backed by a shared [`SubstrateCache`] and a
    /// real [`SubstratePersistence`].
    ///
    /// Pass a clone of the engine-level cache so VRAM accounting and the
    /// eviction budget are shared across all sessions.
    pub fn with_cache(cache: SubstrateCache, persistence: SubstratePersistence) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Substrate::with_cache(cache))),
            allocator: Arc::new(TimelineAllocator::new()),
            persistence: Arc::new(Mutex::new(persistence)),
        }
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

    /// Acquire a read guard.  The returned guard implements [`ContentResolver`]
    /// but is **not target-aware** — use [`Self::read_for`] when a
    /// [`super::project::ProjectionTarget`] is available.
    pub fn read(&self) -> SubstrateRead<'_> {
        SubstrateRead {
            guard: self.inner.read().unwrap(),
        }
    }

    /// Acquire a target-aware read guard.  The returned [`TargetedRead`]
    /// implements [`ContentResolver`] with proper sibling-timeline masking
    /// for `target.group`.
    pub fn read_for(&self, target: super::project::ProjectionTarget) -> TargetedRead<'_> {
        TargetedRead::new(self.read(), target)
    }

    /// Acquire a write guard for mutating operations (append, set_*).
    pub fn write(&self) -> SubstrateWrite<'_> {
        SubstrateWrite {
            guard: self.inner.write().unwrap(),
        }
    }

    /// Atomically append a turn to the substrate.
    ///
    /// `sealed_gpu` is the GPU-resident snapshot; `migrate_to_cpu` is called
    /// inside the lock to convert it to the warm (CPU) tier before storing.
    #[allow(clippy::too_many_arguments)]
    pub fn record_turn(
        &self,
        timeline: TimelineId,
        role: Role,
        text: String,
        token_ids: TokenBuffer,
        token_count: usize,
        block_start: u64,
        block_end: u64,
        sealed_gpu: Arc<Vec<SealedSequence>>,
        migrate_to_cpu: impl FnOnce(&[SealedSequence]) -> candle::Result<Vec<SealedSequence>>,
    ) -> candle::Result<TurnIndex> {
        let idx = {
            let mut view = self.inner.write().unwrap();
            view.append_full(
                timeline,
                role,
                text,
                token_ids,
                token_count,
                block_start,
                block_end,
                sealed_gpu,
                migrate_to_cpu,
            )?
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
        });
        self.persistence
            .lock()
            .unwrap()
            .declare_stream(&decl)
            .map_err(|e| candle::Error::Msg(format!("persist turn: {e}")))?;
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
        let decls = {
            let p = self.persistence.lock().unwrap();
            crate::persistence::resume::recovered_turn_decls(&p)
        };
        let mut restored = 0usize;
        for decl in decls {
            let recovered = {
                let mut p = self.persistence.lock().unwrap();
                crate::persistence::resume::recover_turn(&mut p, &decl, n_layers)
                    .map_err(|e| candle::Error::Msg(format!("recover turn: {e}")))?
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
            let role = match decl.role {
                0 => Role::System,
                1 => Role::User,
                _ => Role::Assistant,
            };
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
            let idx = view.restore_turn(
                timeline,
                role,
                String::new(),
                TokenBuffer::from(recovered.token_ids),
                token_count,
                decl.block_start,
                decl.block_end,
                std::sync::Arc::new(Vec::new()),
            );
            if !sig_entries.is_empty() {
                view.set_sig_entries(timeline, idx, sig_entries);
            }
            restored += 1;
        }
        // Conversation metadata lives in its own `RecordType::Label`
        // records — `(timeline, conv_id, label)` triples, last-write-wins.
        // Replay them after the TurnDecl pass so the in-RAM
        // `substrate.labels` and `substrate.conv_ids` maps are populated
        // for every recovered timeline.
        let metas = {
            let p = self.persistence.lock().unwrap();
            p.collected_conv_metas()
        };
        let n_meta_records = metas.len();
        let mut n_meta_applied = 0usize;
        let mut n_meta_dropped_unregistered = 0usize;
        {
            let mut view = self.write();
            for (timeline_raw, meta) in metas {
                let Some(timeline) = TimelineId::from_raw(timeline_raw) else {
                    continue;
                };
                // `set_conv_id` / `set_label` no-op when the timeline
                // isn't registered. Count any drops here so the reload
                // summary can flag "Label record without matching
                // TurnDecl" — a state we expect to be rare.
                if view.timeline_target(timeline).is_none() {
                    n_meta_dropped_unregistered += 1;
                    continue;
                }
                view.set_conv_id(timeline, &meta.conv_id);
                view.set_label(timeline, &meta.label);
                n_meta_applied += 1;
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
            "substrate reload complete",
        );
        Ok(restored)
    }

    /// Read a cold turn's per-layer chunk grid from the redo log so the
    /// caller can run the warm→hot leg (`load_stream` per layer) and write
    /// the resulting `Vec<SealedSequence>` back via
    /// [`Self::materialize_turn_sealed`].
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
        let stream_id =
            crate::persistence::content_hash::turn_stream_id(timeline.raw(), index.0);
        let mut p = self.persistence.lock().unwrap();
        // We need the turn's `StreamDecl` to drive `recover_turn`. Walk
        // the manifest's persisted decls and pick the one matching this
        // (timeline, index). The decl set is small and rebuilt once at
        // restart, so a linear scan is fine.
        let decls = recovered_turn_decls(&p);
        let decl = match decls.into_iter().find(|d| {
            d.timeline_id == timeline.raw() && d.turn_index == index.0
        }) {
            Some(d) => d,
            None => return Ok(None),
        };
        let recovered = recover_turn(&mut p, &decl, n_layers)
            .map_err(|e| candle::Error::Msg(format!("recover_turn_chunks: {e}")))?;
        if recovered.layers.is_empty() {
            return Ok(None);
        }
        let _ = stream_id; // (computed for diagnostics if needed later)
        Ok(Some(recovered.layers))
    }

    /// Cache a freshly-materialized hot `SealedSequence` set back into the
    /// substrate's turn entry. Called by the engine's `ensure_turn_hot`
    /// orchestrator after running `load_stream` per layer.
    pub fn materialize_turn_sealed(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
        sealed: std::sync::Arc<Vec<SealedSequence>>,
    ) {
        self.write().materialize_turn_sealed(timeline, index, sealed);
    }

    /// Section-side counterpart of [`Self::materialize_turn_sealed`].
    pub fn materialize_section_sealed(
        &self,
        section: SectionId,
        sealed: std::sync::Arc<Vec<SealedSequence>>,
    ) {
        self.write().materialize_section_sealed(section, sealed);
    }

    /// Clear a turn's hot Arc, releasing VRAM arena chunks. Returns
    /// the previously-held Arc so the caller can gather it into warm
    /// before it drops (see `Substrate::clear_turn_sealed`).
    pub fn clear_turn_sealed(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Option<std::sync::Arc<Vec<SealedSequence>>> {
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
    pub fn oldest_hot_turn_except(
        &self,
        except: (TimelineId, TurnIndex),
    ) -> Option<(TimelineId, TurnIndex)> {
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
    pub fn set_conversation_label(
        &self,
        timeline: TimelineId,
        label: &str,
    ) -> candle::Result<()> {
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
    /// `(timeline, conv_id, label)` triples drawn from the in-RAM
    /// `Substrate::labels` / `Substrate::conv_ids` maps. Drives
    /// `GET /v1/conversations` directly; no sidecar involved.
    pub fn known_conversations(&self) -> Vec<(TimelineId, String, String)> {
        self.read().known_conversations()
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
        if p.should_compact()
            .map_err(|e| candle::Error::Msg(format!("persist compaction check: {e}")))?
        {
            p.compact()
                .map_err(|e| candle::Error::Msg(format!("persist compaction: {e}")))?;
        }
        p.checkpoint()
            .map_err(|e| candle::Error::Msg(format!("persist checkpoint: {e}")))
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
    target: super::project::ProjectionTarget,
}

impl<'a> TargetedRead<'a> {
    pub fn new(read: SubstrateRead<'a>, target: super::project::ProjectionTarget) -> Self {
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
        self.read.turn_score_of(timeline, index, formula, weights)
    }

    fn turn_origin(&self, group: GroupId, _index: TurnIndex) -> Option<LayerId> {
        let timeline = self.timeline_for(group)?;
        let (layer, _) = self.read.timeline_target(timeline)?;
        Some(layer)
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        ContentResolver::section_token_count(&*self.read, section)
    }

    fn section_score(
        &self,
        section: SectionId,
        formula: ScoreFormula,
        weights: &DepthWeights,
    ) -> f32 {
        ContentResolver::section_score(&*self.read, section, formula, weights)
    }
}
