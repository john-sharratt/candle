//! Creates and opens the conversation handle, and the shared substrate it reads
//! and edits.
//!
//! Every conversation in a workspace holds the same substrate. Reads take one
//! version without waiting, and edits publish a new version. A `SubstrateRef` names
//! a substrate without keeping it alive, so a record of a gone substrate costs
//! nothing and is never mistaken for a later one. An ephemeral conversation owns a
//! throwaway log, while a conversation adopted from a host shares that host's
//! directory and never deletes it.

use super::*;

/// A substrate named without being kept alive — see
/// [`Conversation::substrate_ref`].
#[derive(Clone)]
pub(crate) struct SubstrateRef(pub(super) Weak<SubstrateLock>);

impl SubstrateRef {
    /// Whether `conversation` is a handle on this substrate.
    pub(crate) fn is(&self, conversation: &Conversation) -> bool {
        ptr::eq(self.0.as_ptr(), Arc::as_ptr(&conversation.inner))
    }

    /// Whether every handle on the substrate has gone, so nothing can name it
    /// again.
    pub(crate) fn is_gone(&self) -> bool {
        self.0.strong_count() == 0
    }

    /// The substrate, while any handle keeps it alive.
    pub(crate) fn upgrade(&self) -> Option<Arc<SubstrateLock>> {
        self.0.upgrade()
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
        let conversation = Self::from_parts(substrate, persistence);
        Self {
            // The directory goes when the last clone of this conversation does.
            // See the field's own note for what it cost not to have this.
            ephemeral_dir: Some(Arc::new(TempDirGuard::new(dir))),
            ..conversation
        }
    }

    /// Create a conversation from a freshly-built `(Substrate,
    /// SubstratePersistence)` pair.  Callers that want the walker to
    /// dispatch into the substrate in one pass should use
    /// [`SubstratePersistence::open_in_with_substrate`] and pass both
    /// here.
    pub fn from_parts(substrate: Substrate, persistence: SubstratePersistence) -> Self {
        Self::from_shared(SharedSubstrate::new(substrate, persistence))
    }

    /// Adopt a substrate the host process already opened.
    ///
    /// The path for a host that writes records of its own into the same log —
    /// `npcd` writes its character records there — and so must hold the *only*
    /// writable handle to the directory. See [`SharedSubstrate`] for what a
    /// second handle silently destroys.
    pub fn from_shared(shared: SharedSubstrate) -> Self {
        let SharedSubstrate {
            substrate: inner,
            persistence,
        } = shared;
        let segments = persistence.call(|p| p.segment_count());
        let maintenance = Arc::new(ArcSwap::from_pointee((segments, None, false)));
        Self {
            // **No guard.** This conversation's log lives in a directory the
            // caller chose — a workspace, someone's `--data`. Removing it on
            // drop would delete the substrate rather than a temp file.
            ephemeral_dir: None,
            inner,
            allocator: process_timeline_allocator(),
            cancel: CancelToken::new(),
            tenant_rank: TenantRank::FIRST,
            maintenance,
            read_only: persistence.is_read_only(),
            sealed: persistence.sealed_reader(),
            minter: persistence.minter(),
            persistence,
            normalization: Arc::new(NormalizationCache::default()),
            normalization_warm: Arc::new(AtomicBool::new(false)),
            branch_checkpoint: Arc::new(ArcSwapOption::empty()),
            gallery_cache: Arc::new(papaya::HashMap::new()),
            group_assemblies: Arc::new(Assemblies::new()),
            held_index_pages: Arc::default(),
            section_loads: Arc::default(),
        }
    }

    /// This conversation's substrate as a [`SharedSubstrate`]: the same
    /// in-RAM substrate and the same persistence handle, not a second opening —
    /// for keyed objects and blobs written beside the conversations, such as a
    /// repository's git records (`docs/substrate_git.md`).
    pub fn shared(&self) -> SharedSubstrate {
        SharedSubstrate {
            substrate: Arc::clone(&self.inner),
            persistence: Arc::clone(&self.persistence),
        }
    }

    /// Create a conversation backed by a real [`SubstratePersistence`].
    /// Equivalent to `from_parts(Substrate::new(), persistence)` — for
    /// callers that already have a populated persistence and want an
    /// empty substrate.
    pub fn with_persistence(persistence: SubstratePersistence) -> Self {
        Self::from_parts(Substrate::new(), persistence)
    }

    /// Whether this conversation's substrate was opened read-only — its
    /// in-RAM state changes as usual and nothing is written to disk.
    pub fn is_read_only_substrate(&self) -> bool {
        self.read_only
    }

    /// This handle, cancelled through `cancel` — the token the daemon that
    /// owns the substrate holds. Clones share it.
    pub fn with_cancel(self, cancel: CancelToken) -> Self {
        Self { cancel, ..self }
    }

    /// This substrate's ingest-cancel latch.
    pub fn cancel_token(&self) -> &CancelToken {
        &self.cancel
    }

    /// This handle, ranked as tenant `rank` on the engine it serves. Clones
    /// share it.
    pub fn with_tenant_rank(self, tenant_rank: TenantRank) -> Self {
        Self {
            tenant_rank,
            ..self
        }
    }

    /// The rank of the tenant this substrate belongs to.
    pub fn tenant_rank(&self) -> TenantRank {
        self.tenant_rank
    }

    /// This substrate's node: the origin of every record it writes
    /// (`docs/replicated_substrate.md` §3.7).
    pub fn node(&self) -> u64 {
        self.minter.node()
    }

    /// The `oseq` of the last record this node originated; zero before its
    /// first.
    pub fn last_oseq(&self) -> u64 {
        self.minter.origination().next_oseq() - 1
    }

    /// Whether this substrate's owner has asked its ingest to stop.
    pub fn ingest_cancelled(&self) -> bool {
        self.cancel.is_cancelled()
    }

    /// Every timeline this substrate holds.
    pub fn timeline_ids(&self) -> Vec<TimelineId> {
        self.inner.read().timeline_ids()
    }

    /// Apply a batch of records other members wrote
    /// ([`SubstratePersistence::apply_remote_shared`],
    /// `docs/replicated_substrate.md` §6.5): decided and appended on the log's
    /// owner, then published in one edit, so a projection sees the batch whole
    /// or not at all.
    ///
    /// **Every record is checked on this thread first.** The check reads every
    /// payload byte and touches no state; on the owner it would hold every
    /// write behind a full pass over the batch.
    pub fn apply_remote(
        &self,
        records: &[&[u8]],
        n_layers: usize,
    ) -> PersistenceResult<RemoteBatch> {
        let checks = SubstratePersistence::check_remote(records);
        self.persistence
            .call(|p| p.apply_remote_shared(&self.inner, records, &checks, n_layers))
    }

    /// Allocate a fresh [`TimelineId`] and register it against
    /// `(layer, group)` on the substrate.
    pub fn mint_timeline(&self, layer: LayerId, group: GroupId) -> TimelineId {
        let id = self.allocator.next();
        self.register_timeline(id, layer, group);
        id
    }

    /// Mint `n` timelines in one version edit — a batch create opens
    /// conversations in bursts, and one edit publishes them together.
    pub fn mint_timelines(&self, layer: LayerId, group: GroupId, n: usize) -> Vec<TimelineId> {
        let ids: Vec<TimelineId> = (0..n).map(|_| self.allocator.next()).collect();
        self.update(|s| {
            for &id in &ids {
                s.register_timeline(id, layer, group);
            }
        });
        ids
    }

    /// Look up `(layer, group)` for a previously-minted timeline.
    pub fn timeline_target(&self, timeline: TimelineId) -> Option<(LayerId, GroupId)> {
        self.inner.read().timeline_target(timeline)
    }

    /// Register a specific [`TimelineId`] against `(layer, group)` —
    /// idempotent. Used by the resume path to bind a conversation to a
    /// timeline recovered from the redo log instead of minting a fresh one.
    pub fn register_timeline(&self, timeline: TimelineId, layer: LayerId, group: GroupId) {
        self.update(|s| s.register_timeline(timeline, layer, group));
    }

    /// Set (or clear) the per-conversation KV-compression override for
    /// `timeline`. Called at conversation creation from the
    /// [`SequenceConfig`]; must run before the first turn seals so each
    /// turn residence inherits it. See [`crate::substrate::ConvCompression`].
    pub fn set_timeline_compression(
        &self,
        timeline: TimelineId,
        compression: Option<crate::substrate::ConvCompression>,
    ) {
        self.update(|s| s.set_timeline_compression(timeline, compression));
    }

    /// Set whether `timeline`'s turns are summarised. `false` for append-only
    /// utility/reference layers (repo_map, code_reading) so their turns never
    /// enter the summariser. Called at conversation creation, before the first
    /// turn seals.
    pub fn set_timeline_summarize(&self, timeline: TimelineId, summarize: bool) {
        self.update(|s| s.set_timeline_summarize(timeline, summarize));
    }

    /// Mark `layer` as an append-only ingest layer (code_reading/repo_map) so
    /// projections targeting it score/select self-local. See
    /// [`crate::substrate::Substrate::mark_layer_append_only`].
    pub fn mark_layer_append_only(&self, layer: LayerId) {
        self.update(|s| s.mark_layer_append_only(layer));
    }

    /// Let projections targeting the append-only `layer` draw `groups` of the
    /// layers below it across conversations. See
    /// [`crate::substrate::Substrate::set_ingest_sees`].
    pub fn set_ingest_sees(&self, layer: LayerId, groups: HashSet<GroupId>) {
        self.update(|s| s.set_ingest_sees(layer, groups.clone()));
    }

    /// Register a layer's [`CorruptTurnPolicy`] (from the projection schema),
    /// consulted at reload to decide whether an unrecoverable turn drops its whole
    /// conversation or just the turn. See
    /// [`crate::substrate::Substrate::set_layer_corrupt_turn_policy`].
    pub fn set_layer_corrupt_turn_policy(&self, layer: LayerId, policy: CorruptTurnPolicy) {
        self.update(|s| s.set_layer_corrupt_turn_policy(layer, policy));
    }

    /// The latest version of the substrate, unscored. The returned guard
    /// implements [`ContentResolver`] but every score lookup returns zero —
    /// appropriate for callers reading structural fields (turn counts,
    /// sealed pointers) without projection. Never waits.
    ///
    /// Use [`Self::read_scored`] when projecting against a freshly-built
    /// [`ProjectionScores`] from the wide-Q belief scan.
    pub fn read(&self) -> SubstrateRead<'_> {
        SubstrateRead {
            guard: self.inner.read(),
            scores: None,
        }
    }

    /// The latest version, bound to an externally-owned
    /// [`ProjectionScores`]. The scores are transient per-projection state —
    /// populated by the reprojection's wide-Q belief scan on the call site's
    /// stack and dropped at end of scope. They are **not** held by the
    /// substrate.
    pub fn read_scored<'a>(&'a self, scores: &'a ProjectionScores) -> SubstrateRead<'a> {
        SubstrateRead {
            guard: self.inner.read(),
            scores: Some(scores),
        }
    }

    /// Edit the substrate: `edit` runs on the latest version and its result
    /// is published — see [`SubstrateLock::update`], whose rules bind `edit`
    /// (it may run more than once, and changes nothing outside the version).
    #[track_caller]
    pub fn update<R>(&self, edit: impl FnMut(&mut Substrate) -> R) -> R {
        self.inner.update(edit)
    }

    /// The residences every version shares — for KV tier transitions, which
    /// publish no version.
    pub fn residences(&self) -> Arc<ResidenceSlab> {
        self.inner.residences()
    }

    /// Acquire a target-aware read guard.  The returned [`TargetedRead`]
    /// implements [`ContentResolver`] with proper sibling-timeline masking
    /// for `target.group`, with score lookups returning zero (unscored).
    #[track_caller]
    pub fn read_for(&self, target: ProjectionTarget) -> TargetedRead<'_> {
        TargetedRead::new(self.read(), target)
    }

    /// Target-aware variant of [`Self::read_scored`].
    #[track_caller]
    pub fn read_for_scored<'a>(
        &'a self,
        target: ProjectionTarget,
        scores: &'a ProjectionScores,
    ) -> TargetedRead<'a> {
        TargetedRead::new(self.read_scored(scores), target)
    }

    /// Bounded rolling-window view ranges for an append-only ingest — the
    /// system-prompt blocks plus the most recent `window_turns` sealed turns of
    /// `timeline` (design `docs/unified_wave_inference_engine.md` §4.7). Returned
    /// as raw `(start_block, end_block)` pairs for the scheduler's view borrow.
    ///
    /// `window_turns == 0` (unbounded) or fewer sealed turns than the window
    /// returns `[(0, total_blocks)]` — the whole parent, byte-for-byte the
    /// unwindowed behaviour. Used only on the disable-reprojection ingest path
    /// and only when `CANDLE_CODEREAD_WINDOW_TURNS > 0`; the KV-windowing effect
    /// itself requires golden-token validation on a live model.
    pub fn windowed_ingest_ranges(
        &self,
        timeline: TimelineId,
        window_turns: usize,
        total_blocks: usize,
    ) -> Vec<(usize, usize)> {
        let sub = self.inner.read();
        let turn_count = sub.turn_count(timeline);
        // Per-turn start block in the sealed block grid the view borrow indexes.
        // `block_range_of` is `(start, end)`; the system prompt is the blocks
        // before turn 0's start.
        let turn_starts: Vec<usize> = (0..turn_count)
            .map(|i| sub.block_range_of(timeline, TurnIndex(i)).0 as usize)
            .collect();
        let sys_end = turn_starts.first().copied().unwrap_or(0);
        crate::conversation::windowed_ingest_ranges_impl(
            sys_end,
            &turn_starts,
            total_blocks,
            window_turns,
        )
    }

    /// Whether `self` and `other` are handles on the same substrate.
    pub fn same_substrate(&self, other: &Conversation) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
    }

    /// A handle that names this substrate without keeping it alive — what a
    /// record of a substrate that has gone away holds, so the record costs no
    /// memory and is never mistaken for a substrate allocated later.
    pub(crate) fn substrate_ref(&self) -> SubstrateRef {
        SubstrateRef(Arc::downgrade(&self.inner))
    }
}
