//! [`Substrate`] — the concrete per-session turn/section store.
//!
//! # Design
//!
//! The projection engine is a pure structural reconciler: it owns no content,
//! no tokenizer, and no scoring mechanism.  Everything that varies per session
//! and per projection (which turns exist, their sizes, their relevance scores)
//! flows through this module.
//!
//! ```text
//!  ┌──────────────────────────────────────────────────────────────────────┐
//!  │ ContentResolver trait — query interface                              │
//!  │  • turn_count(group)                  — how many turns exist         │
//!  │  • turn_token_count(group, i)         — stable size per turn         │
//!  │  • turn_score(group, i, formula, w)   — combined per-turn score      │
//!  │                                                                      │
//!  │ Called by: projection::run on every projection                       │
//!  └──────────────────────────────────────────────────────────────────────┘
//!
//!  ┌──────────────────────────────────────────────────────────────────────┐
//!  │ Substrate struct — concrete session-state owner                      │
//!  │  • append_with_blocks(group, tokens, start, end) → TurnIndex         │
//!  │  • set_scores(group, idx, PerDepthScores)                             │
//!  │  • block_range_of(group, idx)                                         │
//!  │  • reset()                                                            │
//!  └──────────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Storage shape
//!
//! Per-(group, turn) records live in an [`ahash::AHashMap`] keyed by
//! `(GroupId, TurnIndex)`.  An auxiliary `HashMap<GroupId, Vec<TurnIndex>>`
//! tracks insertion order so the projection's `Sequence { recent: N }`
//! rule can iterate turns in the same order they were appended.
//!
//! # Scoring contract
//!
//! Per-turn relevance scores come from the Binary Directional Provenance
//! scanner, which produces three [`TurnScores`] structs per turn (one per
//! depth: syntactic, semantic, pragmatic).  The trait method
//! [`ContentResolver::turn_score`] takes a [`ScoreFormula`] picking which
//! statistic to use (max / sum / mean / top_k_mean / count) and a
//! [`DepthWeights`] specifying how to combine the three depths.  Both are
//! supplied by the projection engine from the layer schema; the resolver is
//! agnostic to the choice.
//!
//! Scores default to all-zeroes until the first BDP scan refreshes them.

use std::sync::{OnceLock, RwLockReadGuard, RwLockWriteGuard};

use ahash::AHashMap;
use candle_nn::kv_cache::SealedSequence;
use std::collections::{BTreeMap, HashMap, LinkedList};
use std::sync::Arc;

use crate::projection::{DepthWeights, ScoreFormula};
use crate::projection::{
    GroupId, LayerId, SectionId, TimelineAllocator, TimelineId, TurnIndex, TurnKey,
};
use crate::token_buffer::TokenBuffer;
use crate::turn::Role;
use crate::SigEntry;

// ── Substrate ─────────────────────────────────────────────────────────────────

/// Per-session turn / section store and the single source of truth for KV
/// residence across hot (VRAM), warm (RAM), and cold (disk) tiers.
///
/// The substrate is the workspace-shared directory of *what* turns exist —
/// timelines, indices, role/text/tokens/sigs — plus the per-turn
/// [`SequenceResidence`] that records *where* each turn's KV bytes live
/// right now. Sections are pinned (no LRU) and held in their own table.
///
/// # Residence layout
///
/// [`Self::residence`] is a `Vec<SequenceResidence>` slab. Each
/// [`TurnEntryData`] and [`SectionEntryData`] carries a
/// [`ResidenceIndex`] into that slab — there is exactly one slot per
/// turn / per section, allocated at append time and never moved.
///
/// LRU is tracked by two [`std::collections::LinkedList`] of
/// [`ResidenceIndex`] values — [`Self::hot_lru`] and
/// [`Self::warm_lru`]. MRU is at the front; eviction pops the back.
/// A residence appears on the hot list iff `hot.is_some()`, ditto
/// warm. Position in the list **is** the recency information — no
/// timestamps, no clock.
#[derive(Debug, Default)]
pub struct Substrate {
    /// Per-turn KV residence slab. Indexed by [`ResidenceIndex`];
    /// [`TurnEntryData::residence`] and [`SectionEntryData::residence`]
    /// hold the index for their owning entity. Slots are never moved
    /// or removed from the middle so indices stay stable for the
    /// lifetime of the substrate (between [`Self::reset`] calls).
    residence: Vec<SequenceResidence>,

    /// Per-timeline directory — projection target, sidebar label,
    /// conv_id, and per-turn metadata (token counts, sig entries,
    /// role, text, token ids). Does **not** hold KV bytes; those live
    /// in [`Self::residence`] addressed by [`TurnEntryData::residence`].
    timelines: HashMap<TimelineId, TimelineEntry>,

    /// Inverse index: every timeline registered against a given group.
    /// Maintained in lockstep with [`Self::timelines`].
    timelines_by_group: HashMap<GroupId, Vec<TimelineId>>,

    /// Pinned section entries (system prompts, tool catalogs). Sections
    /// do not pass through LRU eviction — once ingested they stay hot
    /// for the session.
    sections: AHashMap<SectionId, SectionEntryData>,

    /// Hot-tier LRU list, most-recently-used at the front.
    /// `front()` = MRU, `back()` = next eviction victim. Membership
    /// mirrors `residence[idx].hot.is_some()` for every index in the
    /// list; the list and the `Option<Vec<…>>` flag are maintained
    /// together on every tier transition.
    hot_lru: LinkedList<ResidenceIndex>,

    /// Warm-tier LRU list, most-recently-used at the front. Same
    /// membership invariant as [`Self::hot_lru`] but tracking
    /// `residence[idx].warm`.
    warm_lru: LinkedList<ResidenceIndex>,
}

// ── Tier residence ────────────────────────────────────────────────────────────

/// Index into [`Substrate::residence`]. Strongly typed so it can't be
/// confused with [`TurnIndex`] or any other `usize`.
///
/// A residence slot is allocated when a turn or section is first
/// appended and stays at that index for the lifetime of the substrate
/// (until [`Substrate::reset`]). The hot- and warm-tier LRU lists store
/// `ResidenceIndex` values as their node identifiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResidenceIndex(pub usize);

/// A turn's KV residence across hot (VRAM), warm (RAM), and cold (disk).
///
/// A turn can live in **multiple tiers simultaneously** during promotion
/// windows — e.g. a turn just promoted warm→hot retains its warm copy
/// until pressure forces it out, so the next demotion is free (no copy).
///
/// [`Vec<SealedSequence>`] is the canonical KV shape for both hot and
/// warm copies. The difference between them is purely the device backing
/// of the chunks: `hot` chunks point into GPU arenas, `warm` chunks into
/// host (CPU-resident) memory. The type is device-agnostic; callers
/// maintain the invariant via the tier they place the bytes in.
///
/// [`Vec<StoredSequence>`] is the canonical cold (on-disk) shape — one
/// `StoredSequence` per layer, mirroring the per-layer shape of `hot`
/// and `warm`. Each chunk inside it is a reference into the redo log,
/// not arena GIDs. `cold` is `Option<…>` because the redo-log write
/// is **asynchronous**: a freshly-sealed turn is hot immediately but
/// `cold = None` until the persistence callback confirms its chunks
/// are durably appended and fills in the per-layer references.
///
/// # LRU membership
///
/// The hot/warm LRU position is **not** stored on the residence — it
/// lives in [`Substrate::hot_lru`] / [`Substrate::warm_lru`]. A
/// residence appears in the hot list iff `hot.is_some()`, and in the
/// warm list iff `warm.is_some()`. Tier transition methods on
/// `Substrate` enforce that invariant.
#[derive(Debug)]
pub struct SequenceResidence {
    /// VRAM-resident sealed chunks. `None` ⇒ not in VRAM.
    pub hot: Option<Vec<SealedSequence>>,
    /// RAM-resident sealed chunks. `None` ⇒ not in RAM.
    pub warm: Option<Vec<SealedSequence>>,
    /// Cold-tier references — one [`StoredSequence`] per layer. `None`
    /// until the async redo-log write for this turn lands.
    pub cold: Option<Vec<StoredSequence>>,
    /// Byte size of this sequence's sealed KV payload — the per-tier
    /// memory cost of holding it hot or warm (a turn resident in both
    /// pays the cost twice, once per tier). Set when the first bytes
    /// arrive (`install_hot` / `install_section_hot`) and stays constant
    /// across tier transitions since the payload itself doesn't change.
    /// `0` for a freshly-allocated residence with no bytes anywhere.
    pub byte_size: u64,
}

/// One layer's KV sequence as it lives in the redo log. Mirrors
/// [`SealedSequence`] (the VRAM/RAM form) but each chunk carries a
/// redo-log reference instead of GPU arena GIDs.
///
/// Promoting a `StoredSequence` back into VRAM goes through the cold-
/// load path ([`crate::persistence::transfer::load_stream`]): read each
/// chunk's bytes at its `log_offset`, decode the [`ChunkPayload`],
/// allocate arena slots, scatter the bytes, and produce a fresh
/// [`SealedSequence`].
///
/// [`ChunkPayload`]: crate::persistence::record::ChunkPayload
#[derive(Debug, Clone)]
pub struct StoredSequence {
    /// Ordered per-chunk references into the redo log, in token
    /// order. Length equals the number of sealed chunks this layer
    /// contributed when the turn was persisted.
    pub chunks: Vec<StoredChunk>,
    /// Sum of `chunks[i].token_count`. Cached so the cold-load path
    /// can size its pinned-staging buffer without re-walking chunks.
    pub token_count: usize,
}

/// One chunk's pointer into the redo log. The redo log holds a
/// [`crate::persistence::record::ChunkPayload`] encoded inside a
/// record of length `record_len` starting at `log_offset`; decoding
/// it yields the quantization metadata + arena bytes needed to
/// rebuild a [`candle_nn::kv_cache::chunked::SealedChunk`] in VRAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoredChunk {
    /// Byte offset of the chunk's full record in the redo log.
    pub log_offset: u64,
    /// Length of the chunk's record on disk (read with
    /// `read_record_at(log_offset)` and then decode the payload).
    pub record_len: u64,
    /// Valid token count within this chunk (≤ `CHUNK_SIZE`).
    pub token_count: u16,
}

// ── Per-turn statistics ───────────────────────────────────────────────────────

/// Five aggregations of per-token agreement values within a single turn at
/// a single BDP depth.  All five are computed in one scan pass so the
/// projection can pick whichever metric its layer schema asks for.
///
/// Values are typically in the range `[0, 128]` (the Hamming-agreement scale)
/// scaled by however many `(probe, corpus)` pairs contributed.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TurnScores {
    /// Maximum agreement seen across all `(probe, corpus)` token pairs.
    pub max: f32,
    /// Sum of agreement across all pairs.
    pub sum: f32,
    /// Arithmetic mean: `sum / count_pairs`.
    pub mean: f32,
    /// Mean of the top-K agreements (K from `score_formula_k`, default 8).
    pub top_k_mean: f32,
    /// Number of pairs whose agreement crossed a "high-relevance" threshold.
    pub count: f32,
    /// Span score: Σ L^α over consecutive runs of probe token positions that
    /// had at least one above-threshold corpus match.  α is configured on the
    /// BdpScanner (default 2.0).  Isolated hits (L=1) score 1.0; a run of L
    /// consecutive probe tokens scores L^α, rewarding sustained attention.
    pub span: f32,
    /// Per-token excess: Σ over probe tokens of `max(0, best_agreement − 64)`.
    /// Recentered on the random baseline (noise → ~0) and reduced per probe
    /// token (a single promiscuous token cannot inflate it), with no hit
    /// threshold so weak sub-90 signal survives.  Calibrated as the strongest
    /// prefill-phase section-scoring metric.
    pub pertok_excess: f32,
}

impl TurnScores {
    /// Read the statistic that matches `formula`.  `score_formula_k` for
    /// `TopKMean` is encoded inside the variant; this method just picks a
    /// pre-computed field.
    #[inline]
    pub fn pick(&self, formula: ScoreFormula) -> f32 {
        match formula {
            ScoreFormula::Max => self.max,
            ScoreFormula::Sum => self.sum,
            ScoreFormula::Mean => self.mean,
            ScoreFormula::TopKMean { .. } => self.top_k_mean,
            ScoreFormula::Count => self.count,
            ScoreFormula::Span { .. } => self.span,
            ScoreFormula::PerTokenExcess => self.pertok_excess,
        }
    }
}

/// Per-turn BDP scores at all three depths.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct PerDepthScores {
    pub syn: TurnScores,
    pub sem: TurnScores,
    pub prag: TurnScores,
}

/// Transient per-projection BDP score cache.
///
/// **Not part of the persistent substrate state.** Built by the BDP
/// scanner during one projection pass, consumed by the projection
/// emitter during that same pass, then discarded. Conversation
/// identity does not include this — reload from log starts with an
/// empty `ProjectionScores`, and the next BDP scan repopulates it.
///
/// Lives separately from `TurnEntryData` / `SectionEntryData` so
/// reads of those types don't surface stale-or-not-yet-scored
/// projection scratch as if it were canonical conversation state.
#[derive(Debug, Clone, Default)]
pub struct ProjectionScores {
    turns: AHashMap<TurnKey, PerDepthScores>,
    sections: AHashMap<SectionId, PerDepthScores>,
}

impl ProjectionScores {
    /// An empty score cache — every lookup defaults to zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the BDP scores for one turn.
    pub fn set_turn(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
        scores: PerDepthScores,
    ) {
        self.turns.insert(TurnKey::new(timeline, index), scores);
    }

    /// Record the BDP scores for one system-prompt section.
    pub fn set_section(&mut self, section: SectionId, scores: PerDepthScores) {
        self.sections.insert(section, scores);
    }

    /// Look up a turn's scores. Defaults to zero when the BDP scanner
    /// did not score this turn in the current projection (e.g. the turn
    /// was outside the recent window).
    pub fn turn(&self, timeline: TimelineId, index: TurnIndex) -> PerDepthScores {
        self.turns
            .get(&TurnKey::new(timeline, index))
            .copied()
            .unwrap_or_default()
    }

    /// Look up a section's scores. Defaults to zero when not scored.
    pub fn section(&self, section: SectionId) -> PerDepthScores {
        self.sections.get(&section).copied().unwrap_or_default()
    }

    /// Number of scored turns.
    pub fn turn_count(&self) -> usize {
        self.turns.len()
    }

    /// Number of scored sections.
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Discard every score, leaving the cache empty.
    pub fn clear(&mut self) {
        self.turns.clear();
        self.sections.clear();
    }

    /// Test helper: resolve `group`'s first registered timeline against
    /// `substrate`, then record `scores` for `(timeline, index)`.
    /// No-op when `group` has no registered timeline.
    ///
    /// Mirrors the old `Substrate::set_scores_for_test`, but operates on
    /// this transient cache rather than on substrate state.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn set_for_group_test(
        &mut self,
        substrate: &Substrate,
        group: GroupId,
        index: TurnIndex,
        scores: PerDepthScores,
    ) {
        if let Some(timeline) = substrate.timelines_for_group(group).next() {
            self.set_turn(timeline, index, scores);
        }
    }
}

// ── Per-section record ────────────────────────────────────────────────────────

/// Per-section state stored in the substrate.  Mirrors [`TurnEntryData`]
/// for sections — sections are scoreable like turns when their content
/// has been prefilled into a conversation's KV cache and their
/// per-chunk sig_entries captured.
#[derive(Debug, Clone)]
pub struct SectionEntryData {
    token_count: usize,
    block_range: (u64, u64),
    sig_entries: Vec<SigEntry>,
    tokens: Arc<Vec<u32>>,
    /// Slot in [`Substrate::residence`] holding this section's
    /// hot/warm/cold KV state. Sealed bytes live there.
    residence: ResidenceIndex,
}

// ── Per-turn record ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TurnEntryData {
    token_count: usize,
    block_range: (u64, u64),
    sig_entries: Vec<SigEntry>,
    role: Role,
    text: String,
    token_ids: TokenBuffer,
    /// Slot in [`Substrate::residence`] holding this turn's
    /// hot/warm/cold KV state. Sealed bytes live there.
    residence: ResidenceIndex,
}

/// Total VRAM byte footprint of a sealed Arc — sums `byte_size`
/// across every layer's chunks. `SealedChunk.byte_size` is cached at
/// snapshot time (see `record_turn` in `candle-nn`), so this is pure
/// iteration with no per-call computation.
fn sealed_bytes(sealed: &[SealedSequence]) -> usize {
    sealed
        .iter()
        .flat_map(|seq| seq.chunks.iter())
        .map(|c| c.byte_size as usize)
        .sum()
}

// ── ContentResolver trait ─────────────────────────────────────────────────────

/// Supplies dynamic content metadata to the projection engine.
///
/// The resolver never returns content itself — only the **turn count**,
/// **token counts**, and **scores** the engine needs to make budget and
/// selection decisions.  The engine emits a [`super::Projection`] of opaque
/// ids; the caller looks up content from those ids in its own store.
pub trait ContentResolver {
    /// How many turns have been appended to `group`.
    fn turn_count(&self, group: GroupId) -> u32;

    /// Token count for a turn.  Stable across projection calls.
    fn turn_token_count(&self, group: GroupId, index: TurnIndex) -> usize;

    /// Relevance score for a turn.
    ///
    /// Higher = more relevant.  Computed from per-depth BDP statistics:
    /// the `formula` picks which of `(max, sum, mean, top_k_mean, count)`
    /// to read for each depth, then `weights.combine` collapses the three
    /// depths into a single `f32`.
    fn turn_score(
        &self,
        group: GroupId,
        index: TurnIndex,
        formula: ScoreFormula,
        weights: &DepthWeights,
    ) -> f32;

    /// Layer that produced a given turn.  Used to denormalise
    /// `layer_id` onto the emitted `TurnId` without a back-lookup
    /// through the schema.
    ///
    /// Default impl returns `None` (resolver doesn't track origins).
    /// When `None`, projection emit falls back to the layer-walk's
    /// `layer_id` — which is correct for tests using mock resolvers.
    fn turn_origin(&self, _group: GroupId, _index: TurnIndex) -> Option<LayerId> {
        None
    }

    /// Token count for a system-prompt section.  Returns `0` for
    /// sections the resolver has no record of.
    fn section_token_count(&self, _section: SectionId) -> usize {
        0
    }

    /// Relevance score for a system-prompt section.  Default returns
    /// `0.0` — concrete resolvers (e.g. [`Substrate`]) override
    /// this with BDP-derived scores.
    fn section_score(
        &self,
        _section: SectionId,
        _formula: ScoreFormula,
        _weights: &DepthWeights,
    ) -> f32 {
        0.0
    }
}

// ── Substrate ─────────────────────────────────────────────────────────────────

/// Per-session turn state that implements [`ContentResolver`].
///
/// Owns the append history for every group.  The caller stores turn *content*
/// externally (keyed by `(GroupId, TurnIndex)`) and updates scores via
/// [`Substrate::set_scores`] after each BDP retrieval pass.
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
///   set_scores(group, index, PerDepthScores)
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
    /// Per-turn data, keyed by [`TurnIndex`]. `BTreeMap` iteration is
    /// in index order — naturally matches the append-monotonic semantic
    /// the old `tails: Vec<TurnIndex>` field used to encode separately.
    pub turns: BTreeMap<TurnIndex, TurnEntryData>,
}

impl TimelineEntry {
    fn new(layer: LayerId, group: GroupId) -> Self {
        Self {
            layer,
            group,
            label: None,
            conv_id: None,
            turns: BTreeMap::new(),
        }
    }

    /// Index the next appended turn will take — `turns.len()` since
    /// indices are monotonically allocated.
    fn next_turn_index(&self) -> TurnIndex {
        TurnIndex(self.turns.len() as u32)
    }
}

impl Substrate {
    pub fn new() -> Self {
        Self::default()
    }

    /// Allocate a fresh slot in the residence slab. Used by every
    /// turn/section insert path so each owning entry has a stable
    /// [`ResidenceIndex`] to address its KV state.
    ///
    /// The new slot starts with `hot = None`, `warm = None`, and
    /// `cold = None`. The cold reference is filled in later by the
    /// persistence callback once the turn's chunks are durably on
    /// disk; until then any cold-load attempt errors out. The LRU
    /// lists ignore the slot until the tier-transition methods put
    /// bytes into hot or warm.
    fn alloc_residence(&mut self) -> ResidenceIndex {
        let idx = ResidenceIndex(self.residence.len());
        self.residence.push(SequenceResidence {
            hot: None,
            warm: None,
            cold: None,
            byte_size: 0,
        });
        idx
    }

    /// Place `sealed` into the residence slot's hot tier and push the
    /// slot onto the hot-LRU front (MRU). Used by every code path that
    /// brings KV bytes into VRAM — fresh seal, cold→hot promotion,
    /// post-cold-load materialisation.
    ///
    /// Caller invariant: `sealed` is non-empty. An empty `Vec` would
    /// represent a "cold marker" — under the new design that's the
    /// `hot = None` state, reached by [`Self::clear_hot`], not by
    /// installing an empty Vec.
    fn install_hot(&mut self, residence: ResidenceIndex, sealed: Vec<SealedSequence>) {
        debug_assert!(!sealed.is_empty(), "install_hot called with empty Vec");
        let bytes = sealed_bytes(&sealed) as u64;
        let slot = &mut self.residence[residence.0];
        slot.byte_size = bytes;
        slot.hot = Some(sealed);
        self.hot_lru.push_front(residence);
    }

    /// Section variant of [`Self::install_hot`]. Sections are pinned —
    /// once installed they stay hot and do **not** appear in
    /// [`Self::hot_lru`], so eviction never touches them.
    fn install_section_hot(
        &mut self,
        residence: ResidenceIndex,
        sealed: Vec<SealedSequence>,
    ) {
        debug_assert!(!sealed.is_empty(), "install_section_hot called with empty Vec");
        let bytes = sealed_bytes(&sealed) as u64;
        let slot = &mut self.residence[residence.0];
        slot.byte_size = bytes;
        slot.hot = Some(sealed);
    }

    /// Remove `target` from the hot LRU list. O(n) — for our scale
    /// (hundreds of entries) the rebuild is cheaper than the unstable
    /// `cursor_front_mut` API would buy us.
    fn remove_from_hot_lru(&mut self, target: ResidenceIndex) {
        let mut rebuilt: LinkedList<ResidenceIndex> = LinkedList::new();
        while let Some(v) = self.hot_lru.pop_front() {
            if v != target {
                rebuilt.push_back(v);
            }
        }
        self.hot_lru = rebuilt;
    }

    // ── Timeline registry ────────────────────────────────────────────────────

    pub fn register_timeline(&mut self, timeline: TimelineId, layer: LayerId, group: GroupId) {
        // Idempotent — `HashMap::insert` *replaces*, so calling it on
        // an already-registered timeline would wipe its `turns` map (a
        // real data-loss bug on substrate replay, which calls
        // `register_timeline` once per recovered TurnDecl).
        if self.timelines.contains_key(&timeline) {
            return;
        }
        self.timelines
            .insert(timeline, TimelineEntry::new(layer, group));
        self.timelines_by_group
            .entry(group)
            .or_default()
            .push(timeline);
    }

    pub fn mint_timeline(
        &mut self,
        layer: LayerId,
        group: GroupId,
        allocator: &TimelineAllocator,
    ) -> TimelineId {
        let id = allocator.next();
        self.register_timeline(id, layer, group);
        id
    }

    pub fn timeline_target(&self, timeline: TimelineId) -> Option<(LayerId, GroupId)> {
        self.timelines.get(&timeline).map(|e| (e.layer, e.group))
    }

    /// Borrow a timeline's full entry — projection target plus optional
    /// sidebar metadata. Used by the daemon's sidebar listing.
    pub fn timeline_entry(&self, timeline: TimelineId) -> Option<&TimelineEntry> {
        self.timelines.get(&timeline)
    }

    /// Borrow a single turn by `(timeline, index)`. Two HashMap/BTreeMap
    /// hops since turns now live inside their owning [`TimelineEntry`] —
    /// both O(1) and the second is a tiny BTreeMap (one turn per index
    /// in the timeline), so the hot path is not measurably slower than
    /// the old flat AHashMap keyed by `(TimelineId, TurnIndex)`.
    fn turn(&self, timeline: TimelineId, index: TurnIndex) -> Option<&TurnEntryData> {
        self.timelines.get(&timeline).and_then(|t| t.turns.get(&index))
    }

    /// Mutable variant of [`Self::turn`].
    fn turn_mut(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Option<&mut TurnEntryData> {
        self.timelines
            .get_mut(&timeline)
            .and_then(|t| t.turns.get_mut(&index))
    }

    pub fn timelines_for_group(&self, group: GroupId) -> impl Iterator<Item = TimelineId> + '_ {
        self.timelines_by_group
            .get(&group)
            .into_iter()
            .flat_map(|v| v.iter().copied())
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
        let existing = self.timelines_for_group(group).next();
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
        let residence = self.alloc_residence();
        // `append_with_blocks` declares a turn's existence and block
        // range, but holds no sealed KV — the residence stays cold
        // (`hot = None`) until a later `materialize_turn_sealed` /
        // `append_full` puts bytes in.
        let tl = self.timelines.get_mut(&timeline).unwrap();
        tl.turns.insert(
            idx,
            TurnEntryData {
                token_count,
                block_range: (block_start, block_end),
                sig_entries: Vec::new(),
                role: Role::Assistant,
                text: String::new(),
                token_ids: TokenBuffer::default(),
                residence,
            },
        );
        idx
    }

    /// Append a turn with its sealed KV data.
    ///
    /// `sealed_gpu` is the GPU-resident snapshot produced at seal time.
    /// `migrate_to_cpu` is called **inside** this function to convert it to
    /// CPU (warm tier) — the GPU chunks are released as soon as the caller's
    /// `sealed_gpu` Arc goes out of scope; no GPU arena slots are held by the
    /// substrate after this call returns.
    #[allow(clippy::too_many_arguments)]
    pub fn append_full(
        &mut self,
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
        let sealed_cpu = migrate_to_cpu(&sealed_gpu)?;
        // GPU chunks are freed as soon as the caller drops `sealed_gpu`.
        let idx = self
            .timelines
            .get(&timeline)
            .expect("append_full: timeline not registered")
            .next_turn_index();
        let residence = self.alloc_residence();
        {
            let tl = self.timelines.get_mut(&timeline).unwrap();
            tl.turns.insert(
                idx,
                TurnEntryData {
                    token_count,
                    block_range: (block_start, block_end),
                    sig_entries: Vec::new(),
                    role,
                    text,
                    token_ids,
                    residence,
                },
            );
        }
        // Bytes are CPU-resident here — install them as the turn's hot
        // residence and put it at the front of the hot LRU.
        if !sealed_cpu.is_empty() {
            self.install_hot(residence, sealed_cpu);
        }
        Ok(idx)
    }

    /// Insert a turn reconstructed from the redo log — the substrate-reload
    /// path (§16.12 of `docs/kv_tier_migration.md`).
    ///
    /// Unlike [`Self::append_full`], the per-layer [`SealedSequence`]s are
    /// already built (cold-loaded from disk via `load_stream`), so no
    /// GPU→CPU migration closure runs. The caller must
    /// [`Self::register_timeline`] first. Turns must be restored in
    /// `turn_index` order so the appended `TurnIndex` matches the persisted
    /// one.
    #[allow(clippy::too_many_arguments)]
    pub fn restore_turn(
        &mut self,
        timeline: TimelineId,
        role: Role,
        text: String,
        token_ids: TokenBuffer,
        token_count: usize,
        block_start: u64,
        block_end: u64,
        sealed: Vec<SealedSequence>,
    ) -> TurnIndex {
        let idx = self
            .timelines
            .get(&timeline)
            .expect("restore_turn: timeline must be registered first")
            .next_turn_index();
        let residence = self.alloc_residence();
        {
            let tl = self.timelines.get_mut(&timeline).unwrap();
            tl.turns.insert(
                idx,
                TurnEntryData {
                    token_count,
                    block_range: (block_start, block_end),
                    sig_entries: Vec::new(),
                    role,
                    text,
                    token_ids,
                    residence,
                },
            );
        }
        if !sealed.is_empty() {
            self.install_hot(residence, sealed);
        }
        idx
    }

    pub fn set_turn_content(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
        role: Role,
        text: String,
        token_ids: TokenBuffer,
    ) {
        if let Some(entry) = self.turn_mut(timeline, index) {
            entry.role = role;
            entry.text = text;
            entry.token_ids = token_ids;
        }
    }

    /// Install a freshly-materialized hot KV set for a turn that
    /// currently has none (cold marker). Called by the engine's
    /// `ensure_turn_hot` orchestrator after a cold-load from NVMe.
    ///
    /// No-ops if the turn isn't tracked or if `sealed` is empty.
    pub fn materialize_turn_sealed(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
        sealed: Vec<SealedSequence>,
    ) {
        if sealed.is_empty() {
            return;
        }
        let Some(residence) = self.turn(timeline, index).map(|e| e.residence) else {
            return;
        };
        self.install_hot(residence, sealed);
    }

    /// Section-side counterpart of [`Self::materialize_turn_sealed`].
    /// Section residences don't enter [`Self::hot_lru`] — sections are
    /// pinned for the session.
    pub fn materialize_section_sealed(
        &mut self,
        section: SectionId,
        sealed: Vec<SealedSequence>,
    ) {
        if sealed.is_empty() {
            return;
        }
        let Some(residence) = self.sections.get(&section).map(|e| e.residence) else {
            return;
        };
        self.install_section_hot(residence, sealed);
    }

    /// Drop a turn's hot residence, freeing its VRAM arena chunks
    /// (the inner `Arc<ChunkGid>`s reach refcount 0 once any live
    /// borrowers release them). Removes the slot from the hot LRU.
    ///
    /// Returns the previous hot bytes so the caller can take final
    /// ownership before they drop (e.g. to gather them into the warm
    /// pool first). `None` if the turn was not tracked or was already
    /// cold.
    pub fn clear_turn_sealed(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Option<Vec<SealedSequence>> {
        let residence = self.turn(timeline, index)?.residence;
        let prev = self.residence[residence.0].hot.take()?;
        self.remove_from_hot_lru(residence);
        Some(prev)
    }

    /// Sum of bytes across every hot-resident turn. Walks
    /// [`Self::hot_lru`] directly — exactly the set of turns whose
    /// residence carries hot bytes — and reads each residence's cached
    /// `byte_size`. O(N_hot) integer sum, no chunk-walking.
    pub fn hot_turn_bytes(&self) -> usize {
        self.hot_lru
            .iter()
            .map(|idx| self.residence[idx.0].byte_size as usize)
            .sum()
    }

    /// Sum of bytes across all pinned sections — the system-prompt /
    /// catalog KV the substrate never evicts. Counts toward the same
    /// VRAM budget hot-tier eviction uses.
    pub fn section_bytes(&self) -> usize {
        self.sections
            .values()
            .filter(|e| self.residence[e.residence.0].hot.is_some())
            .map(|e| self.residence[e.residence.0].byte_size as usize)
            .sum()
    }

    /// Bytes a single hot-resident turn currently holds in VRAM. `None`
    /// for unknown / cold turns.
    pub fn turn_hot_bytes(&self, timeline: TimelineId, index: TurnIndex) -> Option<usize> {
        let residence = self.turn(timeline, index)?.residence;
        let slot = &self.residence[residence.0];
        slot.hot.as_ref()?;
        Some(slot.byte_size as usize)
    }

    /// FIFO-oldest hot-resident turn, skipping `except`. "FIFO" =
    /// insertion order within each timeline's tail; timelines are
    /// scanned in registration order. Adequate eviction heuristic
    /// (oldest persisted turn = least likely to be re-touched);
    /// upgradeable to true LRU later by walking [`Self::hot_lru`]
    /// from the back instead.
    pub fn oldest_hot_turn_except(&self, except: TurnKey) -> Option<TurnKey> {
        for (&timeline, tl) in self.timelines.iter() {
            for (&index, entry) in tl.turns.iter() {
                let key = TurnKey::new(timeline, index);
                if key == except {
                    continue;
                }
                if self.residence[entry.residence.0].hot.is_some() {
                    return Some(key);
                }
            }
        }
        None
    }

    pub fn role_of(&self, timeline: TimelineId, index: TurnIndex) -> Role {
        self.turn(timeline, index)
            .map_or(Role::Assistant, |e| e.role)
    }

    pub fn text_of(&self, timeline: TimelineId, index: TurnIndex) -> &str {
        self.turn(timeline, index)
            .map_or("", |e| e.text.as_str())
    }

    pub fn token_ids_of(&self, timeline: TimelineId, index: TurnIndex) -> &[u32] {
        self.turn(timeline, index)
            .map_or(&[][..], |e| &e.token_ids[..])
    }

    pub fn set_block_range(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
        block_start: u64,
        block_end: u64,
    ) {
        if let Some(entry) = self.turn_mut(timeline, index) {
            entry.block_range = (block_start, block_end);
        }
    }

    pub fn extend_turn(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
        additional_tokens: usize,
        new_block_end: u64,
    ) {
        if let Some(entry) = self.turn_mut(timeline, index) {
            entry.token_count = entry.token_count.saturating_add(additional_tokens);
            entry.block_range.1 = new_block_end;
        }
    }

    pub fn block_range_of(&self, timeline: TimelineId, index: TurnIndex) -> (u64, u64) {
        self.turn(timeline, index)
            .map_or((0, 0), |e| e.block_range)
    }

    pub fn set_sig_entries(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
        entries: Vec<crate::provenance::SigEntry>,
    ) {
        if let Some(entry) = self.turn_mut(timeline, index) {
            entry.sig_entries = entries;
        }
    }

    pub fn extend_sig_entries(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
        entries: impl IntoIterator<Item = crate::provenance::SigEntry>,
    ) {
        if let Some(entry) = self.turn_mut(timeline, index) {
            entry.sig_entries.extend(entries);
        }
    }

    pub fn sig_entries_of(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> &[crate::provenance::SigEntry] {
        self.turn(timeline, index)
            .map_or(&[][..], |e| &e.sig_entries)
    }

    pub fn turn_sealed_of(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Option<Arc<Vec<SealedSequence>>> {
        let residence = self.turn(timeline, index)?.residence;
        let hot = self.residence[residence.0].hot.as_ref()?;
        Some(Arc::new(hot.clone()))
    }

    pub fn turn_token_count_of(&self, timeline: TimelineId, index: TurnIndex) -> usize {
        self.turn(timeline, index)
            .map_or(0, |e| e.token_count)
    }

    pub fn turn_count(&self, timeline: TimelineId) -> u32 {
        self.timelines
            .get(&timeline)
            .map_or(0, |t| t.turns.len() as u32)
    }

    pub fn turn_indices(&self, timeline: TimelineId) -> impl Iterator<Item = TurnIndex> + '_ {
        self.timelines
            .get(&timeline)
            .into_iter()
            .flat_map(|t| t.turns.keys().copied())
    }

    pub fn all_turns(&self) -> impl Iterator<Item = TurnKey> + '_ {
        self.timelines
            .iter()
            .flat_map(|(tl, t)| t.turns.keys().map(move |idx| TurnKey::new(*tl, *idx)))
    }

    /// The conversation's sidebar label, or `None` if none has been
    /// recorded yet (the titler writes it after the first user turn).
    pub fn label_of(&self, timeline: TimelineId) -> Option<&str> {
        self.timelines
            .get(&timeline)
            .and_then(|e| e.label.as_deref())
    }

    /// Set a timeline's sidebar label. Empty values are ignored (so the
    /// first-submit placeholder doesn't clobber a real title written
    /// earlier). No-op if the timeline isn't registered.
    pub fn set_label(&mut self, timeline: TimelineId, label: &str) {
        if label.is_empty() {
            return;
        }
        if let Some(entry) = self.timelines.get_mut(&timeline) {
            entry.label = Some(label.to_string());
        }
    }

    /// The client-supplied `conv_id` string for `timeline`, or `None`
    /// if no submit has been recorded yet.
    pub fn conv_id_of(&self, timeline: TimelineId) -> Option<&str> {
        self.timelines
            .get(&timeline)
            .and_then(|e| e.conv_id.as_deref())
    }

    /// Set a timeline's `conv_id`. Empty values are ignored. No-op if
    /// the timeline isn't registered. Distinct from [`Self::set_label`]
    /// only because the two fields are written at different points in
    /// the conversation lifecycle.
    pub fn set_conv_id(&mut self, timeline: TimelineId, conv_id: &str) {
        if conv_id.is_empty() {
            return;
        }
        if let Some(entry) = self.timelines.get_mut(&timeline) {
            entry.conv_id = Some(conv_id.to_string());
        }
    }

    /// Every recovered timeline that has a `conv_id` recorded, paired
    /// with `(conv_id, label)`. Drives the daemon's sidebar — `label`
    /// is empty during the brief window between first-submit and
    /// titler-completion.
    pub fn known_conversations(&self) -> Vec<(TimelineId, String, String)> {
        self.timelines
            .iter()
            .filter_map(|(tl, entry)| {
                let conv_id = entry.conv_id.clone()?;
                let label = entry.label.clone().unwrap_or_default();
                Some((*tl, conv_id, label))
            })
            .collect()
    }

    /// Number of pinned sections — used by the substrate-reload summary log.
    pub fn section_count(&self) -> usize {
        self.sections.len()
    }

    /// Number of registered timelines — every timeline that has at least
    /// one TurnDecl (or was registered via `register_timeline` even
    /// without turns).
    pub fn timeline_count(&self) -> usize {
        self.timelines.len()
    }

    /// Number of timelines that have a `conv_id` set — i.e. the size of
    /// the daemon's sidebar list.
    pub fn conversation_count(&self) -> usize {
        self.timelines
            .values()
            .filter(|e| e.conv_id.is_some())
            .count()
    }

    pub fn reset(&mut self) {
        // `timelines` owns the per-turn store now, so clearing it
        // drops every turn alongside its parent timeline entry.
        self.timelines.clear();
        self.timelines_by_group.clear();
        self.sections.clear();
    }

    // ── Section accessors ────────────────────────────────────────────────────

    /// Create a section entry atomically with all data including sealed KV.
    ///
    /// `sealed_gpu` is the GPU-resident snapshot; `migrate_to_cpu` converts it
    /// to CPU warm-tier storage inside this call.  GPU chunks are released when
    /// the caller's `sealed_gpu` Arc drops.
    ///
    /// `block_start`/`block_end` are omitted — written later by
    /// `set_section_block_range` when the section is injected.
    #[allow(clippy::too_many_arguments)]
    pub fn set_section_full(
        &mut self,
        section: SectionId,
        token_count: usize,
        sig_entries: Vec<SigEntry>,
        sealed_gpu: Arc<Vec<SealedSequence>>,
        migrate_to_cpu: impl FnOnce(&[SealedSequence]) -> candle::Result<Vec<SealedSequence>>,
        tokens: Arc<Vec<u32>>,
    ) -> candle::Result<()> {
        let sealed_cpu = migrate_to_cpu(&sealed_gpu)?;
        // Sections don't go through the cold tier today (they're pinned
        // at conversation setup), so `cold` simply stays `None`.
        let residence = self.alloc_residence();
        let entry = SectionEntryData {
            token_count,
            block_range: (0, 0),
            sig_entries,
            tokens,
            residence,
        };
        self.sections.insert(section, entry);
        if !sealed_cpu.is_empty() {
            self.install_section_hot(residence, sealed_cpu);
        }
        Ok(())
    }

    pub fn set_section_block_range(
        &mut self,
        section: SectionId,
        block_start: u64,
        block_end: u64,
    ) {
        if let Some(e) = self.sections.get_mut(&section) {
            e.block_range = (block_start, block_end);
        }
    }

    pub fn section_sealed_of(&self, section: SectionId) -> Option<Arc<Vec<SealedSequence>>> {
        let residence = self.sections.get(&section)?.residence;
        let hot = self.residence[residence.0].hot.as_ref()?;
        Some(Arc::new(hot.clone()))
    }

    pub fn section_tokens_of(&self, section: SectionId) -> Arc<Vec<u32>> {
        self.sections
            .get(&section)
            .map(|e| Arc::clone(&e.tokens))
            .unwrap_or_else(|| Arc::new(Vec::new()))
    }

    pub fn section_block_range(&self, section: SectionId) -> (u64, u64) {
        self.sections
            .get(&section)
            .map_or((0, 0), |e| e.block_range)
    }

    pub fn section_sig_entries(&self, section: SectionId) -> &[crate::provenance::SigEntry] {
        self.sections
            .get(&section)
            .map_or(&[][..], |e| &e.sig_entries)
    }

    pub fn all_sections(&self) -> impl Iterator<Item = SectionId> + '_ {
        self.sections.keys().copied()
    }

}

/// Group-keyed [`ContentResolver`] over a bare [`Substrate`].
///
/// **The substrate owns no scoring state** — it is a directory of turns,
/// sections and timelines, not a score table. Without an attached
/// [`ProjectionScores`], every score lookup returns zero. Production
/// code attaches scores via [`SubstrateRead::scores`]; this impl exists
/// so structural-only callers (e.g. tests reading only turn counts and
/// sealed pointers) can still pass `&substrate` to `Builder::project`.
///
/// **Phase 1 caveat**: when a group has multiple timelines, this impl
/// reads from the *first registered* timeline only.  Phase 3 replaces
/// this with `TargetedRead` which filters by `target.timeline` within
/// the target group.
impl ContentResolver for Substrate {
    fn turn_count(&self, group: GroupId) -> u32 {
        let Some(timeline) = self.timelines_for_group(group).next() else {
            return 0;
        };
        Substrate::turn_count(self, timeline)
    }

    fn turn_token_count(&self, group: GroupId, index: TurnIndex) -> usize {
        let Some(timeline) = self.timelines_for_group(group).next() else {
            return 0;
        };
        self.turn(timeline, index).map_or(0, |e| e.token_count)
    }

    fn turn_score(
        &self,
        _group: GroupId,
        _index: TurnIndex,
        _formula: ScoreFormula,
        _weights: &DepthWeights,
    ) -> f32 {
        // Bare substrate has no attached scores; pair via ScoredSubstrate
        // or read through Conversation::read_scored to see non-zero values.
        0.0
    }

    fn turn_origin(&self, group: GroupId, _index: TurnIndex) -> Option<LayerId> {
        let timeline = self.timelines_for_group(group).next()?;
        let (layer, _) = self.timeline_target(timeline)?;
        Some(layer)
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        self.sections.get(&section).map_or(0, |e| e.token_count)
    }

    fn section_score(
        &self,
        _section: SectionId,
        _formula: ScoreFormula,
        _weights: &DepthWeights,
    ) -> f32 {
        0.0
    }
}

/// Pairing of a [`Substrate`] with a transient [`ProjectionScores`] cache.
///
/// Used by tests operating on a bare [`Substrate`] that want scoring
/// without going through a [`super::projection::resolver::Conversation`]
/// read guard. Production code does the same thing via
/// [`super::projection::resolver::Conversation::read_scored`].
pub struct ScoredSubstrate<'a> {
    substrate: &'a Substrate,
    scores: &'a ProjectionScores,
}

impl<'a> ScoredSubstrate<'a> {
    pub fn new(substrate: &'a Substrate, scores: &'a ProjectionScores) -> Self {
        Self { substrate, scores }
    }
}

impl<'a> std::ops::Deref for ScoredSubstrate<'a> {
    type Target = Substrate;
    fn deref(&self) -> &Substrate {
        self.substrate
    }
}

/// Combine the three per-depth statistics for a (formula, weights) pair.
/// Shared between every scored ContentResolver impl so a substrate-side
/// shape change can't drift the formula across them.
#[inline]
fn combine_per_depth(
    s: PerDepthScores,
    formula: ScoreFormula,
    weights: &DepthWeights,
) -> f32 {
    weights.combine(
        s.syn.pick(formula),
        s.sem.pick(formula),
        s.prag.pick(formula),
    )
}

/// Group-keyed [`ContentResolver`] impl over a `(Substrate, ProjectionScores)`
/// pair.
///
/// **Phase 1 caveat**: when a group has multiple timelines, this impl reads
/// from the *first registered* timeline only.  Phase 3 replaces this with
/// `TargetedRead` which filters by `target.timeline` within the target group.
impl<'a> ContentResolver for ScoredSubstrate<'a> {
    fn turn_count(&self, group: GroupId) -> u32 {
        let Some(timeline) = self.substrate.timelines_for_group(group).next() else {
            return 0;
        };
        Substrate::turn_count(self.substrate, timeline)
    }

    fn turn_token_count(&self, group: GroupId, index: TurnIndex) -> usize {
        let Some(timeline) = self.substrate.timelines_for_group(group).next() else {
            return 0;
        };
        self.substrate.turn(timeline, index).map_or(0, |e| e.token_count)
    }

    fn turn_score(
        &self,
        group: GroupId,
        index: TurnIndex,
        formula: ScoreFormula,
        weights: &DepthWeights,
    ) -> f32 {
        let Some(timeline) = self.substrate.timelines_for_group(group).next() else {
            return 0.0;
        };
        if self.substrate.turn(timeline, index).is_none() {
            return 0.0;
        }
        combine_per_depth(self.scores.turn(timeline, index), formula, weights)
    }

    fn turn_origin(&self, group: GroupId, _index: TurnIndex) -> Option<LayerId> {
        let timeline = self.substrate.timelines_for_group(group).next()?;
        let (layer, _) = self.substrate.timeline_target(timeline)?;
        Some(layer)
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        self.substrate
            .sections
            .get(&section)
            .map_or(0, |e| e.token_count)
    }

    fn section_score(
        &self,
        section: SectionId,
        formula: ScoreFormula,
        weights: &DepthWeights,
    ) -> f32 {
        if !self.substrate.sections.contains_key(&section) {
            return 0.0;
        }
        combine_per_depth(self.scores.section(section), formula, weights)
    }
}

// ── Guards ────────────────────────────────────────────────────────────────────

/// Read guard over a [`Substrate`] inside a [`super::resolver::Conversation`].
///
/// Carries an optional [`ProjectionScores`] reference: when `Some`, the
/// read implements [`ContentResolver`] using those scores; when `None`,
/// scoring methods return zero (correct default for non-projection
/// callers reading structural fields like turn counts or sealed pointers).
///
/// Construct scored variants via
/// [`super::resolver::Conversation::read_scored`]; the bare
/// [`super::resolver::Conversation::read`] returns the unscored variant.
pub struct SubstrateRead<'a> {
    pub(super) guard: RwLockReadGuard<'a, Substrate>,
    pub(super) scores: Option<&'a ProjectionScores>,
}

impl<'a> SubstrateRead<'a> {
    /// Lookup helper: returns the attached scores, or an empty default
    /// when this is an unscored read.
    fn scores_or_empty(&self) -> &ProjectionScores {
        static EMPTY_SCORES: OnceLock<ProjectionScores> = OnceLock::new();
        self.scores
            .unwrap_or_else(|| EMPTY_SCORES.get_or_init(ProjectionScores::default))
    }

    /// Per-`TimelineId` variant of [`ContentResolver::turn_score`]. Used
    /// by [`super::projection::resolver::TargetedRead`] which already
    /// knows the target-corrected `TimelineId` for the queried group.
    /// Returns `0.0` when the turn is unknown.
    pub fn turn_score_for_timeline(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
        formula: ScoreFormula,
        weights: &DepthWeights,
    ) -> f32 {
        if self.guard.turn(timeline, index).is_none() {
            return 0.0;
        }
        combine_per_depth(
            self.scores_or_empty().turn(timeline, index),
            formula,
            weights,
        )
    }
}

impl<'a> std::ops::Deref for SubstrateRead<'a> {
    type Target = Substrate;
    fn deref(&self) -> &Substrate {
        &self.guard
    }
}

impl<'a> ContentResolver for SubstrateRead<'a> {
    fn turn_count(&self, group: GroupId) -> u32 {
        let Some(timeline) = self.guard.timelines_for_group(group).next() else {
            return 0;
        };
        Substrate::turn_count(&self.guard, timeline)
    }

    fn turn_token_count(&self, group: GroupId, index: TurnIndex) -> usize {
        let Some(timeline) = self.guard.timelines_for_group(group).next() else {
            return 0;
        };
        self.guard
            .turn(timeline, index)
            .map_or(0, |e| e.token_count)
    }

    fn turn_score(
        &self,
        group: GroupId,
        index: TurnIndex,
        formula: ScoreFormula,
        weights: &DepthWeights,
    ) -> f32 {
        let Some(timeline) = self.guard.timelines_for_group(group).next() else {
            return 0.0;
        };
        if self.guard.turn(timeline, index).is_none() {
            return 0.0;
        }
        combine_per_depth(
            self.scores_or_empty().turn(timeline, index),
            formula,
            weights,
        )
    }

    fn turn_origin(&self, group: GroupId, _index: TurnIndex) -> Option<LayerId> {
        let timeline = self.guard.timelines_for_group(group).next()?;
        let (layer, _) = self.guard.timeline_target(timeline)?;
        Some(layer)
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        self.guard
            .sections
            .get(&section)
            .map_or(0, |e| e.token_count)
    }

    fn section_score(
        &self,
        section: SectionId,
        formula: ScoreFormula,
        weights: &DepthWeights,
    ) -> f32 {
        if !self.guard.sections.contains_key(&section) {
            return 0.0;
        }
        combine_per_depth(self.scores_or_empty().section(section), formula, weights)
    }
}

/// Write guard over a [`Substrate`] inside a [`super::resolver::Conversation`].
/// Holds the lock; drop to release.
pub struct SubstrateWrite<'a> {
    pub(super) guard: RwLockWriteGuard<'a, Substrate>,
}

impl<'a> std::ops::Deref for SubstrateWrite<'a> {
    type Target = Substrate;
    fn deref(&self) -> &Substrate {
        &self.guard
    }
}

impl<'a> std::ops::DerefMut for SubstrateWrite<'a> {
    fn deref_mut(&mut self) -> &mut Substrate {
        &mut self.guard
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{GroupId, LayerId, SectionId, TimelineAllocator};
    use crate::token_buffer::TokenBuffer;

    fn make_timeline() -> (LayerId, GroupId, crate::projection::TimelineId, Substrate) {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let timeline = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(timeline, layer, group);
        (layer, group, timeline, sub)
    }

    /// Identity migration: returns the input unchanged (simulates CPU == GPU in tests).
    fn identity_migrate(seqs: &[SealedSequence]) -> candle::Result<Vec<SealedSequence>> {
        Ok(seqs.to_vec())
    }

    /// `register_timeline` is data-idempotent — calling it again on a
    /// timeline that already has turns must NOT wipe those turns. This
    /// is the regression test for a real bug where substrate replay
    /// (`reconstruct_from_log`) destroyed every turn but the last,
    /// because each per-decl `register_timeline` call clobbered the
    /// previous `TimelineEntry`.
    #[test]
    fn re_registering_a_timeline_preserves_its_turns() {
        let (layer, group, timeline, mut sub) = make_timeline();

        // Land two turns under this timeline.
        sub.append_with_blocks(timeline, 3, 0, 1);
        sub.append_with_blocks(timeline, 5, 1, 2);
        assert_eq!(sub.turn_count(timeline), 2);

        // Re-register the same timeline. Used to wipe the turns map.
        sub.register_timeline(timeline, layer, group);
        assert_eq!(
            sub.turn_count(timeline),
            2,
            "re-registering a timeline must preserve its existing turns",
        );

        // Inverse index must still have exactly one entry — not two —
        // confirming we didn't double-push on re-registration.
        let listed: Vec<_> = sub.timelines_for_group(group).collect();
        assert_eq!(listed, vec![timeline]);
    }

    /// `append_full` calls the migration closure and stores the result in the
    /// main map; `turn_sealed_of` returns it.
    #[test]
    fn append_full_stores_migrated_result() {
        let (_, _, timeline, mut sub) = make_timeline();

        // Use a migration that tags the result with a known pointer.
        let migrated = Arc::new(vec![]);
        let migrated_ptr = Arc::as_ptr(&migrated);
        let migrated_clone = Arc::clone(&migrated);

        let idx = sub
            .append_full(
                timeline,
                Role::User,
                "hello".to_string(),
                TokenBuffer::default(),
                3,
                0,
                1,
                Arc::new(vec![]),
                move |_| Ok((*migrated_clone).clone()),
            )
            .unwrap();

        let stored = sub.turn_sealed_of(timeline, idx).unwrap();
        // The stored sealed is the migrated result (same content, may be new Arc).
        let _ = migrated_ptr;
        assert!(sub.turn_sealed_of(timeline, idx).is_some());
        drop(stored);
    }

    /// `set_section_full` calls the migration closure and stores the result.
    #[test]
    fn set_section_full_stores_migrated_result() {
        let mut sub = Substrate::new();
        let section = SectionId::new(42);

        sub.set_section_full(
            section,
            10,
            vec![],
            Arc::new(vec![]),
            identity_migrate,
            Arc::new(vec![1u32, 2, 3]),
        )
        .unwrap();

        assert!(sub.section_sealed_of(section).is_some());
        assert_eq!(
            sub.sections.get(&section).unwrap().tokens.as_slice(),
            &[1u32, 2, 3]
        );
    }

    /// After `reset()`, the substrate is empty.
    #[test]
    fn reset_clears_substrate() {
        let (_, _, timeline, mut sub) = make_timeline();

        sub.append_full(
            timeline,
            Role::Assistant,
            String::new(),
            TokenBuffer::default(),
            0,
            0,
            0,
            Arc::new(vec![]),
            identity_migrate,
        )
        .unwrap();

        sub.reset();
        assert_eq!(sub.turn_count(timeline), 0);
        assert!(sub.turn_sealed_of(timeline, TurnIndex(0)).is_none());
    }

    /// Two successive appends produce independent entries.
    #[test]
    fn multiple_appends_independent() {
        let (_, _, timeline, mut sub) = make_timeline();

        let idx0 = sub
            .append_full(
                timeline,
                Role::User,
                String::new(),
                TokenBuffer::default(),
                0,
                0,
                0,
                Arc::new(vec![]),
                identity_migrate,
            )
            .unwrap();
        let idx1 = sub
            .append_full(
                timeline,
                Role::Assistant,
                String::new(),
                TokenBuffer::default(),
                0,
                0,
                0,
                Arc::new(vec![]),
                identity_migrate,
            )
            .unwrap();

        assert!(sub.turn_sealed_of(timeline, idx0).is_some());
        assert!(sub.turn_sealed_of(timeline, idx1).is_some());
        assert_ne!(idx0, idx1);
    }
}
