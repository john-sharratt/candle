//! Defines the per-turn, per-section and per-timeline records the substrate keeps
//! in RAM, and the handles that tie each one to its KV residence.
//!
//! A turn's pinned content is one contiguous K/V block over its persisted tokens,
//! including its own boundary markers at both ends, so a sealed turn injects
//! complete with nothing emitted between turns. Its layout describes that K/V
//! segment by segment, and its text is kept as the caller submitted it so a
//! reload renders what streamed. The entries hold only the directory side: each
//! turn and section names the residence where its hot, warm and cold KV lives,
//! and the tier code moves that residence rather than the entry.

use super::*;

// ── Per-section record ────────────────────────────────────────────────────────

/// Per-section state stored in the substrate.  Mirrors [`TurnEntryData`]
/// for sections — sections are scoreable like turns when their content
/// has been prefilled into a conversation's KV cache.
#[derive(Debug, Clone)]
pub struct SectionEntryData {
    pub(super) token_count: usize,
    pub(super) block_range: (u64, u64),
    pub(super) tokens: Arc<Vec<u32>>,
    /// Slot in [`Substrate::residence`] holding this section's
    /// hot/warm/cold KV state. Sealed bytes live there.
    pub(super) residence: ResidenceIndex,
}

// ── Per-turn record ───────────────────────────────────────────────────────────

/// One turn's pinned content in the substrate.  The turn's K/V
/// chunks are a single contiguous block addressing the persisted
/// token sequence
/// `[user_start][/no_think?][user_msg][user_end][assistant_start][response][im_end]`.
///
/// **A turn owns its boundaries at both ends.**  The `user_start` head
/// (carrying the `/no_think` soft-switch when the turn was sealed with
/// thinking suppressed) and the `assistant_end` tail are part of the turn's
/// own grid, so a sealed turn injects complete and the projection assembler
/// emits nothing between turns.  The interior `user_end` +
/// `assistant_start` pair is baked for the same reason it always was — its
/// semantic context is the turn's own, invariant across projections.
///
/// This reverses the earlier contract, in which the head and tail were **not**
/// persisted and the assembler re-emitted them as live `Generated` runs at
/// every cross-turn boundary so their K vectors were computed under the
/// actual runtime causal prefix.  That is a real property to give up: a baked
/// marker's K is computed under the prefix present at seal and reused under a
/// different one.  It is given up deliberately, for two reasons.  It is the
/// approximation the substrate already makes for every sealed turn's own K/V,
/// extended to the two least context-sensitive tokens in the projection.  And
/// a live `Generated` run between turns is an island the engine must
/// **gap-fill**, which a recurrent model cannot do even in principle — token
/// *t*'s output depends on the accumulated state over everything before it,
/// in order — so on the hybrid lineage the alternative to an approximate bake
/// is not a better answer but a refused wave.  See
/// `docs/deltanet_state_persistence.md` §4.7.
///
/// A thinking turn's reasoning is part of `[response]`: the model
/// opens its own `<think>…</think>` as the first decoded tokens.  A
/// suppressed turn instead carries an empty `<think></think>` baked
/// right after `assistant_start`, so its `[response]` is the answer
/// alone; prefilled (inserted) turns additionally prepend a
/// `/no_think` and are always suppressed.
///
/// The [`TurnLayout`] describes the turn as an ordered segment vector,
/// the complete description of its K/V.  It carries the human-readable
/// strings (`user_text` / `assistant_text` / `thinking_text`) exactly as
/// the caller had them at submit time — no role markers, no `/no_think`
/// prefix — alongside the per-segment spans.  The assistant body is the
/// verbatim decoded reply (its `<think>…</think>` reasoning is split into a
/// dedicated `Thinking` segment), so the sidebar reload path renders
/// exactly what streamed without re-tokenising or boundary scanning.
#[derive(Debug, Clone)]
pub struct TurnPart {
    /// The turn's segment-vector layout — the complete description of its K/V:
    /// user / thinking / assistant text, each real segment's span, and the
    /// `/no_think` glue.  The per-half text and the content-boundary offsets the
    /// compressor windows on are read via [`TurnLayout`]'s accessors.
    pub layout: TurnLayout,
    /// Total token count this turn pins onto the slot — sum of the
    /// K/V chunk's `token_count` fields.  Holds the invariant
    /// `token_count == token_ids.len()` — every persisted token id
    /// has a corresponding slot position in the K/V chunk grid.  The
    /// decode-loop trailing-terminator (EOS or max-tokens edge token,
    /// sampled but never forwarded) is trimmed at seal time to
    /// preserve this; a `debug_assert_eq!` at the seal site guards
    /// against any future regression.
    pub token_count: usize,
    /// Combined turn token ids in slot order:
    /// `[user_msg][user_end][assistant_start][response]`.
    /// Stored as one buffer because the K/V chunk grid pins this
    /// exact sequence; the persisted `Tokens` record carries the
    /// same bytes so cross-process replay (`recover_turn`)
    /// reconstructs the slot K/V exactly.
    pub token_ids: TokenBuffer,
    /// Slot in [`Substrate::residence`] holding this turn's
    /// hot/warm/cold KV state.
    pub residence: ResidenceIndex,
}

#[derive(Debug, Clone)]
pub struct TurnEntryData {
    /// Slot block extent the turn's K/V occupies — `(start, end)`.
    pub(super) block_range: (u64, u64),
    /// The turn's pinned content as one indivisible unit.
    pub(super) content: TurnPart,
}

/// Caller-supplied content for a turn at append / restore time.
#[derive(Debug, Clone, Default)]
pub struct TurnPartWrite {
    /// The turn's segment-vector layout — user / thinking / assistant text and
    /// each real segment's K/V span.  See [`TurnPart::layout`].
    pub layout: TurnLayout,
    pub token_ids: TokenBuffer,
    pub token_count: usize,
    pub block_start: u64,
    pub block_end: u64,
    pub sealed_gpu: Option<Arc<Vec<SealedSequence>>>,
    /// Gather-scope tags for this turn (e.g. `"tool"` on calibration turns).
    /// Persisted onto the turn's `TurnDecl`; the provenance gallery honours a
    /// projection policy's `tags:` filter against them. Empty for live turns.
    pub tags: Vec<String>,
}

impl TurnPartWrite {
    pub fn is_empty(&self) -> bool {
        self.token_count == 0
            && self.block_start == self.block_end
            && self.sealed_gpu.as_deref().is_none_or(|v| v.is_empty())
    }
}

/// The hot KV the working-set pins hold — see
/// [`Substrate::pinned_hot_by_timeline`].
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PinnedHot {
    /// Residences pinned, hot or not.
    pub pins: usize,
    /// Hot bytes of every pinned residence, turns and sections.
    pub pinned_bytes: u64,
    /// Of them, the pinned turns' — what `by_timeline` divides.
    pub turn_bytes: u64,
    /// `(timeline, turns, bytes)`, largest first.
    pub by_timeline: Vec<(TimelineId, usize, u64)>,
}

// ── Substrate ─────────────────────────────────────────────────────────────────

/// Per-session turn state that implements [`ContentResolver`].
///
/// Owns the append history for every group.  The caller stores turn *content*
/// externally (keyed by `(GroupId, TurnIndex)`); relevance scores are supplied
/// separately per projection via a [`ProjectionScores`] cache.
///
/// # Storage
///
/// Per-(group, turn) records live in an `ahash::AHashMap` for fast lookup.
/// A parallel `HashMap<GroupId, Vec<TurnIndex>>` keeps insertion order so
/// the `Sequence { recent: N }` selection rule can iterate turns in
/// chronological order without sorting.
///
/// # Lifecycle
///
/// ```text
///   Substrate::new()
///        │
///        ▼
///   append_with_blocks(group, tokens, start, end) → TurnIndex
///        │
///        ▼
///   set_turn(timeline, index, belief_score: f32)
///        │
///        ▼
///   builder.project(target, &resolver)
///        │
///        ├── reset()
///        └── .clone()   ← fork support
/// ```
/// Per-timeline state — projection target, sidebar metadata, and the
/// ordered turn store. One struct, one HashMap lookup — replaces what
/// used to be four parallel maps keyed by `TimelineId` (target / label
/// / conv_id / turns) plus a `tails` Vec for ordered indices.
#[derive(Debug, Clone)]
pub struct TimelineEntry {
    /// Projection target this timeline is registered against. Required —
    /// set by [`Substrate::register_timeline`].
    pub layer: LayerId,
    pub group: GroupId,
    /// Sidebar title written by the daemon's titler after the first user
    /// turn. `None` until the titler completes (the substrate is the
    /// single source of truth — there is no sidecar).
    pub label: Option<String>,
    /// Client-supplied `conv_id` string — the daemon's stable id for
    /// this conversation, persisted at first-submit time alongside any
    /// label as a `RecordType::Label` record.
    pub conv_id: Option<String>,
    /// Creation-order rank: the clock reading of the least record that named a
    /// `conv_id`, or for a conversation named only by records without an
    /// identity, a count in the order the walk met them
    /// ([`Substrate::conv_order_counter`]). `0` until named. Higher = created
    /// later; the daemon sidebar orders conversations with the same
    /// [`Self::active`] by it.
    pub order: u64,
    /// Free-form key/value metadata, persisted in the same
    /// `RecordType::Label` record as `label`/`conv_id` and merged
    /// (last-write-wins per key) on update. Used as a content-addressed
    /// cache index by utility ingests; searchable via
    /// [`Substrate::timelines_with_metadata`].
    pub custom: POrdMap<String, String>,
    /// Conversation lifecycle: `true` once the user has closed
    /// (archived) the conversation. The sidebar filters archived
    /// entries out by default; the "show archived" checkbox toggles
    /// them back in. Persisted as `RecordType::ConvState`,
    /// last-write-wins.
    pub archived: bool,
    /// The branch this conversation works on in each repository, keyed by
    /// the repository's workspace name. Persisted with `archived` in the same
    /// `RecordType::ConvState` record — see [`ConvState`].
    pub branches: POrdMap<String, String>,
    /// Last-use rank, stamped from [`Substrate::conv_active_counter`] each time
    /// the conversation is used; higher = used more recently, `0` = not used
    /// since the rank was kept. Persisted with `archived` in the `ConvState`
    /// record. The daemon sidebar sorts on it, most recently used first.
    pub active: u64,
    /// Per-turn data, keyed by [`TurnIndex`]. `BTreeMap` iteration is
    /// in index order — naturally matches the append-monotonic semantic
    /// the old `tails: Vec<TurnIndex>` field used to encode separately.
    pub turns: POrdMap<TurnIndex, Arc<TurnEntryData>>,
    /// The indices of `turns` that hold an empty placeholder for a position no
    /// turn here fills yet — a declaration lost in a crash, or another
    /// member's turn still arriving (§6.5). A turn completed later takes its
    /// gap in place, so the turns above it never move.
    pub gaps: POrdSet<TurnIndex>,
    /// Per-turn tree metadata for the immutable summary forest
    /// (`docs/archived/immutable_summary_forest.md`).  Parallel to `turns`: every
    /// recorded turn carries exactly one [`TreeNodeMeta`] entry (defaults to a
    /// `Normal` content sub-leaf with no children).  Promoted to a
    /// `SummaryOfTurns` / `SummaryOfSummaries` by the async summariser thread
    /// after the §6 probe runs.  The peak set (window entry points) is derived
    /// from this map — see [`Substrate::peaks_of`].
    pub tree_meta: POrdMap<TurnIndex, TreeNodeMeta>,
    /// Turns coupled to the tool response that follows them — the two halves of
    /// one tool round-trip (`RecordType::TurnCoupling`). `from ∈ set` reads as
    /// "turn `from + 1` is the tool response to turn `from`". Replayed as a set
    /// because the records are idempotent and order-independent; consumed by
    /// [`summary_tree::exchange`](crate::summary_tree::exchange) to group turns
    /// into exchanges.
    pub couplings: Couplings,
    /// Optional substrate-side resume key.  Set via
    /// [`Substrate::set_debug_id`] and reverse-indexed by
    /// [`Substrate::timeline_by_debug_id`].  Used by the
    /// debug-id-resumable grow-conversation harness (§10.4).
    pub debug_id: Option<String>,
    /// Turns currently waiting on the async summariser to absorb them
    /// into the tree (i.e. they have arrived in the substrate but the
    /// summariser thread has not yet produced or extended a
    /// `SummaryOfTurns` leaf covering them).  Drained in FIFO order by
    /// the summariser's `pop_pending_turn` API.
    pub pending_summary_queue: PVec<TurnIndex>,
    /// Whether this timeline's turns are fed to the summariser at all.
    ///
    /// **Currently `false` for every timeline — the AVL summariser is
    /// disconnected** (see `Engine::new`). Nothing spawns the thread and nothing
    /// enqueues, so a turn is never compressed into a summary node. The gate is
    /// kept at the mechanism, rather than removed, because it is the one place
    /// every enqueue path already funnels through: re-enabling summarisation is
    /// a matter of deciding *which* timelines set it, not of re-threading the
    /// substrate.
    ///
    /// It was `true` for dialogue and `false` for append-only utility/reference
    /// layers (repo_map, code_reading), whose turns are background reference and
    /// whose summaries the ingest pipeline owns.
    pub summarize: bool,
    /// Set on cold-load (and after the old-AVL migration) to ask the summariser
    /// to reconcile this timeline's persisted forest against the canonical
    /// ternary shape — building any missing/mismatched internal nodes on the
    /// low-priority queue.  Cleared once [`Substrate::reconcile_next`] reports
    /// the forest whole.  Live appends keep the forest whole, so this stays
    /// `false` during normal operation.
    ///
    /// **Dormant with the summariser disconnected.** Its only readers live in
    /// `summary_tree::summariser`, which never runs, so a flag set on cold-load
    /// is never cleared. That is inert rather than a leak — nothing else
    /// consults it, and there is no forest to reconcile because no summary node
    /// is ever built.
    pub needs_reconcile: bool,
    /// Most recent score-density [`SelectionDiagnostics`] for this
    /// timeline, written by the scheduler at projection time and read
    /// by the test harness via
    /// [`Substrate::last_selection_of`].  Last-write-wins across
    /// re-projections within a single turn — by the time `send_turn`
    /// returns, this holds the final selection that produced the
    /// model's response.  Test-harness diagnostic only; production
    /// daemons can ignore it.
    pub last_selection: Option<Arc<SelectionDiagnostics>>,
    /// The record behind each of this timeline's replicated values.
    pub ids: TimelineIds,
    /// This timeline's belief gallery, built by the first reader that wants it
    /// and cleared by every edit of the entry ([`file_gallery`]).
    pub gallery: GalleryMemo,
}

/// Tree-bookkeeping metadata stored alongside every substrate turn.
///
/// Every persisted turn carries one of these — defaults are a `Normal` content
/// sub-leaf with no children.  Summary nodes (produced by the §6 probe)
/// overwrite the defaults with the real kind / children / level when the
/// summariser thread seals them.  Nodes are immutable once promoted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeNodeMeta {
    /// Three-kind tag from `summary_tree::TurnKind`, mirrored here so
    /// the substrate is the single source of truth and the redo-log
    /// codec can round-trip without depending on the algorithm module.
    pub kind: TurnKind,
    /// For `SummaryOfTurns`: the Normal-turn children in chronological order.
    /// For `SummaryOfSummaries`: exactly `MERGE_FANOUT` same-level summary
    /// children.  For `Normal`: empty.
    pub children: Vec<TurnIndex>,
    /// Forest level.  Always `0` for `Normal`.  `SummaryOfTurns` is `1`;
    /// `SummaryOfSummaries` carries `child_level + 1`.
    pub tree_height: u8,
}

impl Default for TreeNodeMeta {
    fn default() -> Self {
        Self {
            kind: TurnKind::Normal,
            children: Vec::new(),
            tree_height: 0,
        }
    }
}

impl TreeNodeMeta {
    /// Sensible default for a Normal content turn.
    pub fn normal() -> Self {
        Self::default()
    }
}

/// Smallest Normal-turn index covered by `idx` within a timeline's `tree_meta`.
/// Defines the chronological order of peaks (and of any subtree): a node sorts
/// by the oldest content beneath it.
pub(super) fn leftmost_normal_in(
    tree_meta: &POrdMap<TurnIndex, TreeNodeMeta>,
    idx: TurnIndex,
) -> u32 {
    match tree_meta.get(&idx) {
        None => idx.0,
        Some(meta) => match meta.kind {
            TurnKind::Normal => idx.0,
            TurnKind::SummaryOfTurns => meta.children.iter().map(|c| c.0).min().unwrap_or(idx.0),
            TurnKind::SummaryOfSummaries => meta
                .children
                .iter()
                .map(|c| leftmost_normal_in(tree_meta, *c))
                .min()
                .unwrap_or(idx.0),
        },
    }
}

/// [`Substrate::held_payload_bytes`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct HeldPayloadBytes {
    pub wide_q_sigs: u64,
    pub projection_events: u64,
}

/// Per-stream in-RAM runtime state — built by replaying the redo log
/// on startup and updated on every fresh append.
///
/// The chunk index supports O(1) `(stream_id, chunk_idx) → ChunkLoc`
/// lookup for cold-load.
#[derive(Debug, Clone, Default)]
pub struct StreamRuntime {
    /// The decoded stream declaration (`StreamDecl` record).
    pub decl: Option<StreamDecl>,
    /// Live chunk locations by chunk index — last-writer-wins.
    pub chunks: POrdMap<u64, ChunkLoc>,
    /// Latest `Tokens` record for the stream.
    pub tokens: Option<RecordLoc>,
    /// Latest `ProjectionEvents` record payload (opaque JSON bytes — the
    /// projection layer decodes). Eager bytes rather than a `RecordLoc`: the
    /// per-turn timeline is tiny, so we keep it resident and re-emit it on
    /// compaction like the other synthesised per-entity records.
    ///
    /// Shared rather than owned, like `index_page`: maintenance re-emits every
    /// live blob, and an owned one is copied into the op's plan and held there
    /// for the whole op.
    pub projection_events: Option<Arc<[u8]>>,
    /// Opaque encoded wide-Q signature window (`provenance::wide_sig`) for this turn's most
    /// recent (re)projection — the decode→decode (`Q·Q`) consensus substrate. Last-writer-wins;
    /// rebuilt from the redo log on replay. `None` until the first projection writes it.
    ///
    /// Per-token × heads × words, so hundreds of KB a turn — shared for the
    /// same reason as `projection_events`.
    pub wide_q_sigs: Option<Arc<[u8]>>,
    /// The QSA index page of a turn or a prompt section — the compressed index
    /// rows covering exactly its tokens, as sealed. Last-writer-wins, rebuilt
    /// from the redo log on replay, `None` for a piece sealed by a model that
    /// indexes nothing.
    ///
    /// A projection that borrows this turn's K/V needs these rows handed over
    /// with it: the index is computed from hidden states, so unlike the K/V it
    /// cannot be reconstructed by the slot that borrows it.
    ///
    /// Shared, not owned, and keyed once: every projection that selects the
    /// turn hands these bytes over, and a turn's page runs to a megabyte and
    /// more on a model that indexes every attention layer, so a copy and a
    /// digest per projection were most of what re-injecting a turn cost. See
    /// [`StoredPages`].
    pub index_page: Option<IndexPage>,
    /// Highest chunk index the stream is durably committed through — the
    /// greatest any `Commit` record stated.
    pub committed_through: Option<u64>,
    /// The record behind each of the values above.
    pub ids: StreamIds,
    /// What the belief scans read from `projection_events` — its seams and its
    /// selections — decoded once from these bytes. Derived from this runtime's
    /// own blob alone, so every version sharing the runtime shares a correct
    /// digest; a write of the blob makes a new runtime with an empty one.
    pub events_digest: OnceLock<Arc<EventsDigest>>,
}

/// Where a turn's or section's QSA index page is: its bytes, held, or the
/// record in the log that holds them.
///
/// **A page recovered from the log stays in the log.** Every turn's and
/// section's page was read at recovery and held for the life of the process,
/// whether its turn was ever projected or not: on WhiteWorld (2026-10-20), a
/// 260 GB substrate of deep conversations, that was ~60 GB of committed host
/// memory paged straight back out, and the process aborted three times on
/// allocations the machine could no longer commit. A recovered page is
/// located like a snapshot and read when a projection asks for it; one sealed
/// or received in this process is held, as it was written.
#[derive(Debug, Clone)]
pub enum IndexPage {
    /// The bytes, as this process sealed or received them.
    Resident(Arc<StoredPages>),
    /// The record that holds them, read on demand.
    Stored(RecordLoc),
}

impl IndexPage {
    /// The held bytes, for a page this process holds.
    pub fn resident(&self) -> Option<&Arc<StoredPages>> {
        match self {
            Self::Resident(p) => Some(p),
            Self::Stored(_) => None,
        }
    }

    /// The record's location, for a page left in the log.
    pub fn stored(&self) -> Option<RecordLoc> {
        match self {
            Self::Stored(loc) => Some(*loc),
            Self::Resident(_) => None,
        }
    }
}

#[cfg(test)]
impl IndexPage {
    /// The page's bytes, held or read from `sp`'s log — what a test compares
    /// against the bytes the seal wrote.
    pub(crate) fn read_with(&self, sp: &mut crate::persistence::SubstratePersistence) -> Vec<u8> {
        match self {
            Self::Resident(pages) => pages.bytes().to_vec(),
            Self::Stored(loc) => sp
                .read_record_payload(loc)
                .expect("a located page reads back"),
        }
    }
}

impl StreamRuntime {
    /// Whether the stream holds nothing: no declaration and no value.
    pub fn is_empty(&self) -> bool {
        self.decl.is_none()
            && self.chunks.is_empty()
            && self.tokens.is_none()
            && self.projection_events.is_none()
            && self.wide_q_sigs.is_none()
            && self.index_page.is_none()
            && self.committed_through.is_none()
    }

    /// Whether this stream is an **orphan** to node `node`: it has no
    /// declaration, and never will, so nothing can rebuild its turn or section
    /// and every record it holds is dead weight.
    ///
    /// A declaration is written before the records of its stream, so a stream
    /// this node wrote without one lost it — in a crash, or in a rewrite that
    /// predates this rule — and so did a stream of records that carry no
    /// identity. A stream another origin wrote is different: its records can
    /// reach this node before its declaration does (§6.5), and dropping them
    /// would lose them for good, since they were already applied. Such a
    /// stream waits for its declaration.
    pub fn is_orphan(&self, node: u64) -> bool {
        if self.decl.is_some() {
            return false;
        }
        let ids = &self.ids;
        let foreign = |id: &RecordIdentity| !id.is_none() && id.origin != node;
        !([
            &ids.decl,
            &ids.tokens,
            &ids.projection_events,
            &ids.wide_q_sigs,
            &ids.index_page,
            &ids.commit,
        ]
        .into_iter()
        .any(foreign)
            || ids.chunks.values().any(foreign))
    }
}

impl TimelineEntry {
    pub(super) fn new(layer: LayerId, group: GroupId) -> Self {
        Self {
            layer,
            group,
            couplings: Couplings::default(),
            label: None,
            conv_id: None,
            order: 0,
            custom: POrdMap::new(),
            archived: false,
            branches: POrdMap::new(),
            active: 0,
            turns: POrdMap::new(),
            gaps: POrdSet::new(),
            tree_meta: POrdMap::new(),
            debug_id: None,
            pending_summary_queue: PVec::new(),
            // The summariser is disconnected: no timeline enqueues.
            summarize: false,
            needs_reconcile: false,
            last_selection: None,
            ids: TimelineIds::default(),
            gallery: GalleryMemo::default(),
        }
    }

    /// Index the next appended turn will take — `turns.len()` since
    /// indices are monotonically allocated.
    pub(super) fn next_turn_index(&self) -> TurnIndex {
        TurnIndex(self.turns.len() as u32)
    }
}
