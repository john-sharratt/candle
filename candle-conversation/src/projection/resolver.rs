//! [`Conversation`] — the workspace-shared substrate handle, and
//! [`TargetedRead`] — the target-aware [`ContentResolver`] wrapper.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};

use super::ids::{GroupId, LayerId, SectionId, TimelineAllocator, TimelineId, TurnIndex};
use super::schema::{DepthWeights, ScoreFormula};
use crate::persistence::resume::ChunkImage as ResumeChunkImage;
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
    /// Every persisted turn stream is recovered in `(timeline, turn_index)`
    /// order; `load_layers` cold-loads each turn's per-layer chunk grid back
    /// into VRAM as `Vec<SealedSequence>` (the caller supplies this since it
    /// owns the per-layer KV backings). The turn's timeline is re-registered
    /// and the turn appended to the substrate. Returns the number of turns
    /// restored.
    pub fn reconstruct_from_log(
        &self,
        n_layers: usize,
        load_layers: impl Fn(&[Vec<ResumeChunkImage>]) -> candle::Result<Vec<SealedSequence>>,
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
            let sealed = load_layers(&recovered.layers)?;
            // Re-append the BDP signatures into the (fresh) provenance file,
            // yielding entries that point at the rebuilt offsets.
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
            let token_count: usize = recovered
                .layers
                .first()
                .map(|l| l.iter().map(|c| c.token_count as usize).sum())
                .unwrap_or(0);
            let mut view = self.write();
            if let (Some(layer), Some(group)) = (
                LayerId::from_raw(decl.layer_id),
                GroupId::from_raw(decl.group_id),
            ) {
                view.register_timeline(timeline, layer, group);
            }
            let idx = view.restore_turn(
                timeline,
                role,
                String::new(),
                TokenBuffer::from(recovered.token_ids),
                token_count,
                decl.block_start,
                decl.block_end,
                std::sync::Arc::new(sealed),
            );
            if !sig_entries.is_empty() {
                view.set_sig_entries(timeline, idx, sig_entries);
            }
            restored += 1;
        }
        Ok(restored)
    }

    /// Persist a sealed turn's per-layer KV grid + token ids to the redo log
    /// — the seal-time half of the resume path (§16.12). `layers[layer]` is
    /// that layer's ordered [`ChunkImage`] list; all layers share one
    /// chunk count.
    pub fn persist_turn_kv(
        &self,
        stream_id: StreamId,
        layers: &[Vec<ResumeChunkImage>],
        token_ids: &[u32],
    ) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        crate::persistence::resume::persist_turn_kv(&mut p, stream_id, layers, token_ids)
            .map_err(|e| candle::Error::Msg(format!("persist turn kv: {e}")))
    }

    /// Persist only a turn's per-layer chunk records — the heavy half of the
    /// async seal/persist chain. Called from inside the bg-quantizer callback
    /// once float→quant migrations have landed (slice 7); also callable
    /// directly for sync use.
    pub fn persist_turn_chunks(
        &self,
        stream_id: StreamId,
        layers: &[Vec<ResumeChunkImage>],
    ) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        crate::persistence::resume::persist_turn_chunks(&mut p, stream_id, layers)
            .map_err(|e| candle::Error::Msg(format!("persist turn chunks: {e}")))
    }

    /// Persist a turn's `Tokens` record and the trailing `Commit` — always
    /// called synchronously on seal, regardless of compression policy.
    /// `layers` is only used to compute the highest chunk index; pass an
    /// empty slice when no chunks were persisted (compression `None` path,
    /// or async chains that re-commit later).
    pub fn persist_turn_tokens(
        &self,
        stream_id: StreamId,
        token_ids: &[u32],
        layers: &[Vec<ResumeChunkImage>],
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

    /// Persist a turn's BDP provenance signatures to the redo log — the
    /// `Signatures` record. `sigs` is the
    /// [`crate::persistence::resume::encode_signatures`] payload.
    pub fn persist_signatures(&self, stream_id: StreamId, sigs: &[u8]) -> candle::Result<()> {
        let mut p = self.persistence.lock().unwrap();
        p.append_signatures(stream_id, sigs)
            .map_err(|e| candle::Error::Msg(format!("persist signatures: {e}")))
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

    /// Durably flush the persistence redo log — the group-commit point.
    /// `fsync`s every staged record so an in-flight turn survives a crash.
    pub fn commit_persistence(&self) -> candle::Result<()> {
        self.persistence
            .lock()
            .unwrap()
            .commit()
            .map_err(|e| candle::Error::Msg(format!("persist commit: {e}")))
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
