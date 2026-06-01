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

use crate::persistence::content_hash::turn_stream_id;
use crate::persistence::streams::StreamId;
use crate::projection::{DepthWeights, ScoreFormula};
use crate::projection::{
    GroupId, LayerId, SectionId, TimelineAllocator, TimelineId, TurnIndex, TurnKey,
};
use crate::token_buffer::TokenBuffer;
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
/// until pressure forces it out, so the next eviction is free (no copy).
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
    /// Persistence-layer stream identity for this residence. Set at
    /// allocation time, immutable. Turns derive it from
    /// `turn_stream_id(timeline, index)`; sections that don't persist
    /// to disk use [`StreamId::default()`] (the reserved sentinel).
    pub stream_id: StreamId,
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

// ── Promotion plan (elevate_to_hot work shape) ────────────────────────────────

/// What kind of substrate entity owns a residence — needed because
/// section installs skip the hot LRU (pinned) while turn installs
/// push to the LRU front.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PromotionItemKind {
    Section(SectionId),
    Turn(TurnKey),
}

/// One item already in VRAM-resident hot tier — no work needed.
/// Kept in [`PromotionPlan::already_hot`] so the report can count it.
///
/// Used to plan: "the fast path of [`super::persistence::elevate::elevate_to_hot`]
/// when most callers' items are already cached."
pub type AlreadyHotEntry = PromotionItemKind;

/// One item that needs warm → hot promotion. The warm
/// `Vec<SealedSequence>` is cloned out of the substrate at snapshot
/// time so the caller can hand it to
/// [`candle_nn::kv_cache::ChunkedKvBacking::migrate_sealed_to_gpu_batch_async`]
/// without holding the substrate read lock.
#[derive(Debug)]
pub struct WarmToHotEntry {
    pub kind: PromotionItemKind,
    pub residence: ResidenceIndex,
    pub warm: Vec<SealedSequence>,
}

/// One item that needs cold (disk) → hot promotion. The cold
/// `Vec<StoredSequence>` is cloned for the same reason as
/// [`WarmToHotEntry::warm`]; the [`StreamId`] is the redo-log handle
/// the caller passes to `recover_turn_chunks` / `load_to_hot`.
#[derive(Debug, Clone)]
pub struct ColdToHotEntry {
    pub kind: PromotionItemKind,
    pub residence: ResidenceIndex,
    pub cold: Vec<StoredSequence>,
    pub stream_id: StreamId,
}

/// Classification of a batch of items by what `elevate_to_hot` must
/// do for each. Produced by [`Substrate::snapshot_promotion_state`]
/// under a single read lock; consumed by the elevation orchestrator
/// (one round of CUDA work + one substrate write lock).
#[derive(Debug, Default)]
pub struct PromotionPlan {
    pub already_hot: Vec<AlreadyHotEntry>,
    pub warm_to_hot: Vec<WarmToHotEntry>,
    pub cold_to_hot: Vec<ColdToHotEntry>,
    pub missing: Vec<PromotionItemKind>,
}

/// A full rehydration from cold (disk-only) state.
///
/// The elevation orchestrator produces one of these per turn that was
/// pulled out of the redo log: `load_to_hot` writes the bytes into
/// VRAM, then `migrate_sealed_to_cpu_batch_async` materialises a
/// fresh CPU-arena copy. The residence lands dual-tier (hot + warm)
/// in a single install so future hot evictions are no-DMA — the warm
/// copy is already there.
///
/// Conceptually a *recall*: reaching back into durable storage and
/// reactivating a turn fully into the working set. Pairs with
/// [`WarmLift`] (the faster RAM-cached hop) and is consumed
/// alongside it by [`Substrate::install_promoted`].
#[derive(Debug)]
pub struct ColdRecall {
    pub kind: PromotionItemKind,
    pub residence: ResidenceIndex,
    pub hot: Vec<SealedSequence>,
    pub warm: Vec<SealedSequence>,
}

/// A fast promotion from warm (RAM-cached) state into hot.
///
/// The elevation orchestrator produces one of these per turn already
/// resident in the warm tier: `migrate_sealed_to_gpu_batch_async`
/// scatters the warm payload into fresh VRAM arena chunks. Only hot
/// is installed — the warm copy stays in place under its existing
/// `warm_lru` entry, so the residence stays dual-tier without any
/// extra substrate writes.
///
/// Conceptually a *lift*: a short PCIe hop from the RAM cache to
/// VRAM, paying nothing on NVMe. Pairs with [`ColdRecall`].
#[derive(Debug)]
pub struct WarmLift {
    pub kind: PromotionItemKind,
    pub residence: ResidenceIndex,
    pub hot: Vec<SealedSequence>,
}

/// Per-call summary from [`Substrate::evict_hot_except`].
/// `count` is the number of residences evicted, `bytes` is the total
/// VRAM freed (sum of each residence's cached `byte_size`).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct EvictionReport {
    pub count: usize,
    pub bytes: u64,
}

/// Which tiers a residence currently occupies. Returned by
/// [`Substrate::turn_tier_state`] / [`Substrate::section_tier_state`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TierState {
    pub hot: bool,
    pub warm: bool,
    pub cold: bool,
}

/// Per-call summary from [`Substrate::purge_warm_until_headroom`].
/// `count` warm-tier residences had their warm copy dropped; `bytes`
/// is the sum of their `byte_size` (RAM freed).
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct PurgeReport {
    pub count: usize,
    pub bytes: u64,
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

/// One turn's pinned content in the substrate.  The turn's K/V
/// chunks are a single contiguous block addressing the persisted
/// token sequence
/// `[no_think_prefix][user_msg][user_end][assistant_start][/think_block][response]`
/// — the inter-turn `user_start` head and `assistant_end` tail are
/// **not** persisted: the projection assembler re-emits them as
/// live `Generated` runs at every cross-turn boundary so their K
/// vectors are computed under the actual runtime causal prefix.
/// The interior `user_end` + `assistant_start` pair stays baked
/// because its semantic context (the turn's own user message and
/// decoded response) is invariant across projections.
///
/// The text fields (`user_text` / `assistant_text`) carry the
/// human-readable strings exactly as the caller had them at
/// submit time — no role markers, no `/no_think` prefix.  They're
/// stored verbatim so the sidebar reload path renders without any
/// re-tokenising or boundary scanning.
#[derive(Debug, Clone)]
pub struct TurnPart {
    /// The user's message text, exactly as `submit_turn` received it
    /// — no role-marker envelope, no `/no_think` prefix, no
    /// boundary tokens.  Stored verbatim so the sidebar can render
    /// it without re-tokenising or pattern-matching at read time.
    pub user_text: String,
    /// The assistant's reply text — the decoded body of the model's
    /// response with special tokens skipped.  Same "what the caller
    /// already has" rule as `user_text`.
    pub assistant_text: String,
    /// Total token count this turn pins onto the slot — sum of the
    /// K/V chunk's `token_count` fields.  Must match the combined
    /// `token_ids.len()` (modulo the slot's chunk granularity).
    pub token_count: usize,
    /// Combined turn token ids in slot order:
    /// `[no_think_prefix][user_msg][user_end][assistant_start][/think_block][response]`.
    /// Stored as one buffer because the K/V chunk grid pins this
    /// exact sequence; the persisted `Tokens` record carries the
    /// same bytes so cross-process replay (`recover_turn`)
    /// reconstructs the slot K/V exactly.
    pub token_ids: TokenBuffer,
    pub sig_entries: Vec<SigEntry>,
    /// Slot in [`Substrate::residence`] holding this turn's
    /// hot/warm/cold KV state.
    pub residence: ResidenceIndex,
}

#[derive(Debug, Clone)]
pub struct TurnEntryData {
    /// Slot block extent the turn's K/V occupies — `(start, end)`.
    block_range: (u64, u64),
    /// The turn's pinned content as one indivisible unit.
    content: TurnPart,
}

/// Caller-supplied content for a turn at append / restore time.
#[derive(Debug, Clone, Default)]
pub struct TurnPartWrite {
    /// The user's message text, exactly as the caller had it before
    /// concatenation with role markers — see [`TurnPart::user_text`].
    pub user_text: String,
    /// The decoded assistant body — see [`TurnPart::assistant_text`].
    pub assistant_text: String,
    pub token_ids: TokenBuffer,
    pub token_count: usize,
    pub block_start: u64,
    pub block_end: u64,
    pub sealed_gpu: Option<Arc<Vec<SealedSequence>>>,
}

impl TurnPartWrite {
    pub fn is_empty(&self) -> bool {
        self.token_count == 0
            && self.block_start == self.block_end
            && self.sealed_gpu.as_deref().is_none_or(|v| v.is_empty())
    }
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
    /// Conversation lifecycle: `true` once the user has closed
    /// (archived) the conversation. The sidebar filters archived
    /// entries out by default; the "show archived" checkbox toggles
    /// them back in. Persisted as `RecordType::ConvState`,
    /// last-write-wins.
    pub archived: bool,
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
            archived: false,
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
    fn alloc_residence(&mut self, stream_id: StreamId) -> ResidenceIndex {
        let idx = ResidenceIndex(self.residence.len());
        self.residence.push(SequenceResidence {
            stream_id,
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

    /// Remove the first occurrence of `target` from `list`. O(n) find +
    /// O(1) splice — no allocations, the existing list nodes are
    /// re-linked around the removed one via [`LinkedList::split_off`] +
    /// [`LinkedList::append`].
    ///
    /// Free function (rather than `&mut self`) so it can be reused
    /// against either `hot_lru` or `warm_lru` without borrow-checker
    /// gymnastics over two `&mut self.field`s on the same struct.
    fn remove_from_lru(list: &mut LinkedList<ResidenceIndex>, target: ResidenceIndex) {
        if let Some(pos) = list.iter().position(|&v| v == target) {
            let mut tail = list.split_off(pos);
            tail.pop_front();
            list.append(&mut tail);
        }
    }

    // ── Tier transitions (persistence-thread API) ───────────────────────────

    /// Install RAM-resident sealed chunks into a residence slot and
    /// push the slot onto the warm-LRU front. Called by the persistence
    /// thread after `migrate_sealed_to_cpu` produces a CPU copy.
    ///
    /// Caller invariant: `warm` is non-empty and the residence currently
    /// has `warm = None`. Re-install on an already-warm slot is a bug
    /// (would duplicate the entry in the warm LRU).
    pub fn install_warm(&mut self, residence: ResidenceIndex, warm: Vec<SealedSequence>) {
        debug_assert!(!warm.is_empty(), "install_warm called with empty Vec");
        debug_assert!(
            self.residence[residence.0].warm.is_none(),
            "install_warm on already-warm residence"
        );
        self.residence[residence.0].warm = Some(warm);
        self.warm_lru.push_front(residence);
    }

    /// Install warm AND atomically drop hot — the production
    /// persistence-thread path. Once warm holds the compressed
    /// canonical copy of the turn there is no reason to keep the
    /// uncompressed hot reference around: it just pins VRAM the
    /// arena pool could reclaim. The next decode that touches this
    /// residence re-elevates warm→hot, landing quantized GPU chunks
    /// (bit-identical to what a daemon restart would reconstruct
    /// from cold) so the in-session attention path exercises the
    /// same compressed bytes that survive a restart — fidelity bugs
    /// surface immediately instead of being masked by the
    /// uncompressed hot copy until the next reload.
    ///
    /// Tests that need the "hot + warm coexist" intermediate state
    /// should call [`Self::install_warm`] directly.
    pub fn install_warm_and_evict_hot(
        &mut self,
        residence: ResidenceIndex,
        warm: Vec<SealedSequence>,
    ) {
        self.install_warm(residence, warm);
        if self.residence[residence.0].hot.take().is_some() {
            Self::remove_from_lru(&mut self.hot_lru, residence);
        }
    }

    /// Atomic dual install: replace the residence's hot AND install
    /// warm at the same time. Used by the persistence thread after a
    /// quantize-in-place pass produces new GPU Q-format chunks (hot)
    /// alongside a format-preserving copy of those chunks on CPU
    /// (warm).
    ///
    /// Replaces — not adds — `residence.hot`. The previous hot
    /// SealedSequences (which the in-session reconcile fed in as
    /// R16/F16 chunks captured at `record_turn`) drop their Arc refs
    /// here; if no other holder remains (the slot was truncated
    /// post-seal, so this is normal), the underlying arena chunks
    /// return to the pool.
    ///
    /// The crucial property: when `apply_projection` runs for the
    /// next turn, it pulls `residence.hot` and Arc-clones those
    /// gids onto the slot. Because we installed the new GPU Q chunks
    /// directly here, decode reads **the exact bytes the convert
    /// kernel wrote** with no intervening `kv_migrate` scatter — the
    /// same invariant the perf test relies on.
    pub fn install_warm_and_hot(
        &mut self,
        residence: ResidenceIndex,
        hot: Vec<SealedSequence>,
        warm: Vec<SealedSequence>,
    ) {
        debug_assert!(!hot.is_empty(), "install_warm_and_hot called with empty hot");
        debug_assert!(!warm.is_empty(), "install_warm_and_hot called with empty warm");
        // Replace hot. The LRU entry stays (residence is still hot).
        self.residence[residence.0].hot = Some(hot);
        if !self.hot_lru.contains(&residence) {
            self.hot_lru.push_front(residence);
        }
        // Install warm. Mirror install_warm's invariant — caller is
        // expected to call this when warm is currently None (first
        // persist of this turn). A second persist pass on the same
        // residence would be a redo-log duplication bug; we don't
        // guard against it here.
        debug_assert!(
            self.residence[residence.0].warm.is_none(),
            "install_warm_and_hot on already-warm residence"
        );
        self.residence[residence.0].warm = Some(warm);
        self.warm_lru.push_front(residence);
    }

    /// Install the cold-tier references for a residence slot — called
    /// by the persistence thread after writing the turn's chunks to
    /// the redo log. Cold has no LRU (it's already the cheapest tier).
    pub fn install_cold(&mut self, residence: ResidenceIndex, cold: Vec<StoredSequence>) {
        debug_assert!(!cold.is_empty(), "install_cold called with empty Vec");
        self.residence[residence.0].cold = Some(cold);
    }

    /// Snapshot indices of hot-resident slots that lack a warm copy —
    /// the work list for the persistence thread's hot→warm phase. Each
    /// entry pairs the residence index with a clone of its hot bytes so
    /// the thread can drop the substrate read lock before doing the
    /// (slow) CUDA-side `migrate_sealed_to_cpu`.
    ///
    /// The clone of `Vec<SealedSequence>` is cheap — the inner
    /// `SealedChunk`s hold `Arc<ChunkGid>` references; clone bumps the
    /// arena refcount but doesn't copy KV bytes.
    pub fn snapshot_pending_warm(&self) -> Vec<(ResidenceIndex, Vec<SealedSequence>)> {
        self.hot_lru
            .iter()
            .filter_map(|&idx| {
                let slot = &self.residence[idx.0];
                if slot.warm.is_some() {
                    return None;
                }
                slot.hot.as_ref().map(|hot| (idx, hot.clone()))
            })
            .collect()
    }

    /// Snapshot indices of warm-resident slots that lack a cold (on-
    /// disk) record — the work list for the persistence thread's
    /// warm→cold phase. Pairs each index with the slot's [`StreamId`]
    /// and a clone of its **hot** bytes (which are equivalent to warm
    /// — the same payload — but live on a device the GPU gather path
    /// can read). Skips slots where hot has been evicted; future
    /// revisions can gather from warm directly once a CPU gather
    /// exists.
    pub fn snapshot_pending_cold(
        &self,
    ) -> Vec<(ResidenceIndex, StreamId, Vec<SealedSequence>)> {
        // We persist the **warm** tier, not hot. In the legacy
        // format-preserving migration path warm and hot carry the
        // same bytes, so this doesn't change behavior. In the
        // quantize-on-evict path warm holds the compressed chunks
        // and hot still holds the source floats; reading warm is
        // the only way the redo log captures the actual stored
        // form. `warm_lru` only contains residences with warm
        // installed, so unwrapping `slot.warm` is sound here.
        self.warm_lru
            .iter()
            .filter_map(|&idx| {
                let slot = &self.residence[idx.0];
                if slot.cold.is_some() {
                    return None;
                }
                slot.warm
                    .as_ref()
                    .map(|warm| (idx, slot.stream_id, warm.clone()))
            })
            .collect()
    }

    /// Classify a batch of items (sections + turns) by what work
    /// `elevate_to_hot` will need to do to put each one in VRAM.
    ///
    /// One read-lock walk over the substrate; clones the warm
    /// `Vec<SealedSequence>` and cold `Vec<StoredSequence>` so the
    /// caller can drop the lock before running the slow CUDA work.
    /// The inner `Arc<ChunkGid>`s in warm SealedSequences are cloned
    /// cheaply (refcount bumps); cold StoredSequences are plain data.
    pub fn snapshot_promotion_state(
        &self,
        sections: &[SectionId],
        turns: &[TurnKey],
    ) -> PromotionPlan {
        let mut plan = PromotionPlan::default();
        for &sid in sections {
            let Some(entry) = self.sections.get(&sid) else {
                plan.missing.push(PromotionItemKind::Section(sid));
                continue;
            };
            self.classify_one(
                PromotionItemKind::Section(sid),
                entry.residence,
                &mut plan,
            );
        }
        for &key in turns {
            let Some(entry) = self.turn(key.timeline, key.index) else {
                plan.missing.push(PromotionItemKind::Turn(key));
                continue;
            };
            // Classify the assistant residence — the one carrying the
            // turn's K/V under today's seal path.  The user residence
            // is reserved for the Phase 5 `NewUserMessage` capture and
            self.classify_one(
                PromotionItemKind::Turn(key),
                entry.content.residence,
                &mut plan,
            );
        }
        plan
    }

    /// Helper for [`Self::snapshot_promotion_state`]: route one item
    /// into the right bucket of the plan based on its residence's tier
    /// occupancy.
    fn classify_one(
        &self,
        kind: PromotionItemKind,
        residence: ResidenceIndex,
        plan: &mut PromotionPlan,
    ) {
        let slot = &self.residence[residence.0];
        if slot.hot.is_some() {
            plan.already_hot.push(kind);
            return;
        }
        if let Some(warm) = &slot.warm {
            plan.warm_to_hot.push(WarmToHotEntry {
                kind,
                residence,
                warm: warm.clone(),
            });
            return;
        }
        if let Some(cold) = &slot.cold {
            plan.cold_to_hot.push(ColdToHotEntry {
                kind,
                residence,
                cold: cold.clone(),
                stream_id: slot.stream_id,
            });
            return;
        }
        plan.missing.push(kind);
    }

    /// Bulk install of freshly-promoted bytes — both elevation legs
    /// together under a single write lock.
    ///
    /// `recalls` lands turns pulled out of cold storage (full
    /// rehydration): warm goes in first, then hot. `lifts` lands
    /// turns lifted from the warm cache: hot only, warm is already
    /// in place. Section installs do **not** enter `hot_lru`
    /// (pinned); turn installs push to the front (MRU).
    ///
    /// One method call per batch keeps lock churn at the substrate
    /// write-lock boundary down. Empty `hot` payloads are skipped
    /// (defensive: a phase that produced nothing usable shouldn't
    /// leave a half-populated residence).
    pub fn install_promoted(&mut self, recalls: Vec<ColdRecall>, lifts: Vec<WarmLift>) {
        for recall in recalls {
            if recall.hot.is_empty() {
                continue;
            }
            // Warm goes first: the cold→hot leg produces a fresh
            // warm payload as part of the recall, and the residence
            // is guaranteed `warm = None` pre-recall (otherwise it'd
            // have classified as a WarmLift). install_hot then
            // transitions hot+warm dual-residency.
            if !recall.warm.is_empty()
                && self.residence[recall.residence.0].warm.is_none()
            {
                self.install_warm(recall.residence, recall.warm);
            }
            match recall.kind {
                PromotionItemKind::Section(_) => {
                    self.install_section_hot(recall.residence, recall.hot);
                }
                PromotionItemKind::Turn(_) => {
                    self.install_hot(recall.residence, recall.hot);
                }
            }
        }
        for lift in lifts {
            if lift.hot.is_empty() {
                continue;
            }
            // Warm already exists on the residence (that's what made
            // this a lift, not a recall) — just install hot.
            match lift.kind {
                PromotionItemKind::Section(_) => {
                    self.install_section_hot(lift.residence, lift.hot);
                }
                PromotionItemKind::Turn(_) => {
                    self.install_hot(lift.residence, lift.hot);
                }
            }
        }
    }

    /// Test/integration helper: the residence-slab index that a turn
    /// addresses. Needed by tests that drive the tier transitions
    /// (`install_warm`, `install_cold`, `evict_hot_except`) from
    /// outside the substrate's own write methods. Returns `None`
    /// when the turn isn't tracked.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn turn_residence(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Option<ResidenceIndex> {
        self.turn(timeline, index).map(|e| e.content.residence)
    }

    /// Test/integration counterpart of [`Self::turn_residence`] for
    /// section entries.
    #[cfg(any(test, feature = "test-helpers"))]
    pub fn section_residence(&self, section: SectionId) -> Option<ResidenceIndex> {
        self.sections.get(&section).map(|e| e.residence)
    }

    /// Which tiers a turn residence currently occupies. Returns `None`
    /// when the turn isn't tracked. Load-bearing in production: the
    /// SubmitTurn handler uses this as a tier-agnostic existence
    /// check before adding a turn to the projection's elevate list,
    /// so cold-marker turns (post-restart, before any elevation has
    /// fired) still survive the filter and reach `elevate_to_hot`.
    pub fn turn_tier_state(&self, timeline: TimelineId, index: TurnIndex) -> Option<TierState> {
        let residence = self
            .turn(timeline, index)
            .map(|e| e.content.residence)?;
        let slot = &self.residence[residence.0];
        Some(TierState {
            hot: slot.hot.is_some(),
            warm: slot.warm.is_some(),
            cold: slot.cold.is_some(),
        })
    }

    /// Section counterpart of [`Self::turn_tier_state`]. Sections only
    /// occupy hot today (no warm/cold equivalent), so `warm` and `cold`
    /// will always be `false`.
    pub fn section_tier_state(&self, section: SectionId) -> Option<TierState> {
        let residence = self.sections.get(&section).map(|e| e.residence)?;
        let slot = &self.residence[residence.0];
        Some(TierState {
            hot: slot.hot.is_some(),
            warm: slot.warm.is_some(),
            cold: slot.cold.is_some(),
        })
    }

    /// Drop warm-tier residences from the LRU tail until the system
    /// has at least `headroom_target` bytes of available RAM after
    /// the upcoming `incoming_bytes` allocation lands.
    ///
    /// Threshold per the design: `max(2 GiB, 5% × total_ram)`. The
    /// caller passes the OS-reported `(total_ram, available_ram)` —
    /// keeping the OS query at the orchestrator level so the substrate
    /// stays sysinfo-free and unit-testable without mocking.
    ///
    /// **Invariants:**
    /// - Only warm bytes are freed (`residence.warm = None`); hot and
    ///   cold are untouched. A purged residence that's also hot stays
    ///   in `hot_lru`; its next hot eviction will need a fresh
    ///   hot→warm DMA.
    /// - LRU-ordered: pops from the back of `warm_lru`.
    /// - Single batch: no intermediate work between victims.
    ///
    /// Returns a [`PurgeReport`] with `count` victims and `bytes`
    /// freed (sum of `residence.byte_size`).
    pub fn purge_warm_to_target(
        &mut self,
        incoming_bytes: u64,
        available_ram: u64,
        total_ram: u64,
    ) -> PurgeReport {
        let threshold: u64 = std::cmp::max(2 * 1024 * 1024 * 1024, total_ram / 20);
        let mut freed_bytes: u64 = 0;
        let mut count: usize = 0;
        loop {
            // available + freed - incoming >= threshold ?
            let projected = available_ram
                .saturating_add(freed_bytes)
                .saturating_sub(incoming_bytes);
            if projected >= threshold {
                break;
            }
            // Pop LRU off the back of the warm list.
            let Some(idx) = self.warm_lru.pop_back() else {
                break;
            };
            let slot = &mut self.residence[idx.0];
            if slot.warm.take().is_some() {
                freed_bytes = freed_bytes.saturating_add(slot.byte_size);
                count += 1;
                tracing::debug!(
                    target: "candle_conversation::persistence::tier",
                    residence = idx.0,
                    bytes = slot.byte_size,
                    "purged warm (RAM headroom)"
                );
            }
            // If warm was None, the slot was stale in the LRU; just
            // discard the index and keep looking.
        }
        if count > 0 {
            tracing::info!(
                target: "candle_conversation::persistence::tier",
                count,
                bytes = freed_bytes,
                threshold,
                incoming_bytes,
                available_ram,
                total_ram,
                "warm purge batch complete"
            );
        }
        PurgeReport {
            count,
            bytes: freed_bytes,
        }
    }

    /// Drop hot bytes from every warm-backed hot residence **except**
    /// the ones whose sections / turns are in the keep set. Steady-state
    /// working-set turnover: when a projection has produced a new
    /// working set, the items in `keep_sections` + `keep_turns` are
    /// about to be elevated (or are already hot), so evicting them
    /// would just waste a DMA round-trip.
    ///
    /// The keep-set membership is checked by **residence index** —
    /// resolved here under the same write lock from the substrate's
    /// own `sections` map and per-timeline turn lookups. Unresolvable
    /// items in the keep set are silently ignored (they can't match a
    /// real residence anyway).
    ///
    /// Invariants:
    /// - Only `hot.is_some() && warm.is_some()` slots are eligible.
    /// - Sections are pinned and never appear on `hot_lru`, so this
    ///   only ever touches turn residences.
    /// - Returned [`EvictionReport`] sums the dropped slots.
    ///
    /// Emits one `tracing::debug!` per evicted residence and an
    /// aggregate `tracing::info!` at the end, both on the
    /// `candle_conversation::persistence::tier` target.
    pub fn evict_hot_except(
        &mut self,
        keep_sections: &[SectionId],
        keep_turns: &[TurnKey],
    ) -> EvictionReport {
        // Resolve keep-items → ResidenceIndex set. Items the substrate
        // doesn't know about are silently dropped — they can't match
        // a live `hot_lru` entry anyway.
        let mut keep: std::collections::HashSet<ResidenceIndex> =
            std::collections::HashSet::with_capacity(keep_sections.len() + keep_turns.len());
        for &sid in keep_sections {
            if let Some(e) = self.sections.get(&sid) {
                keep.insert(e.residence);
            }
        }
        for &key in keep_turns {
            if let Some(e) = self.turn(key.timeline, key.index) {
                keep.insert(e.content.residence);
            }
        }

        let victims: Vec<(ResidenceIndex, u64)> = self
            .hot_lru
            .iter()
            .copied()
            .filter_map(|idx| {
                if keep.contains(&idx) {
                    return None;
                }
                let slot = &self.residence[idx.0];
                (slot.hot.is_some() && slot.warm.is_some()).then(|| (idx, slot.byte_size))
            })
            .collect();

        let count = victims.len();
        let bytes: u64 = victims.iter().map(|(_, b)| *b).sum();
        for (idx, b) in &victims {
            tracing::debug!(
                target: "candle_conversation::persistence::tier",
                residence = idx.0,
                bytes = *b,
                "evicted hot (working-set turnover)"
            );
            self.residence[idx.0].hot = None;
            Self::remove_from_lru(&mut self.hot_lru, *idx);
        }
        if count > 0 {
            tracing::info!(
                target: "candle_conversation::persistence::tier",
                count,
                bytes,
                keep_size = keep.len(),
                "evict_from_hot batch complete"
            );
        }
        EvictionReport { count, bytes }
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
        let residence = self.alloc_residence(turn_stream_id(timeline.raw(), idx.0));
        // `append_with_blocks` declares a turn's existence and block
        // range, but holds no sealed KV — the residence stays cold
        // (`hot = None`) until an elevate / restore_turn install
        // puts bytes in.
        let tl = self.timelines.get_mut(&timeline).unwrap();
        tl.turns.insert(
            idx,
            TurnEntryData {
                block_range: (block_start, block_end),
                content: TurnPart {
                    user_text: String::new(),
                    assistant_text: String::new(),
                    token_count,
                    token_ids: TokenBuffer::default(),
                    sig_entries: Vec::new(),
                    residence,
                },
            },
        );
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
        mut migrate_to_cpu: impl FnMut(&[SealedSequence]) -> candle::Result<Vec<SealedSequence>>,
    ) -> candle::Result<TurnIndex> {
        // `Some(_)` (even an empty vec) means "this turn claims
        // sealed bytes — run the migration to get the CPU side."
        // `None` means "no bytes at all."  Empty input to migrate
        // is legitimate: callers in the GPU-less test paths pass an
        // empty `sealed_gpu` and rely on the migration closure to
        // produce the canonical CPU content.
        let sealed_cpu = match write.sealed_gpu.as_ref() {
            Some(g) => migrate_to_cpu(g)?,
            None => Vec::new(),
        };
        let idx = self
            .timelines
            .get(&timeline)
            .expect("append_complete: timeline not registered")
            .next_turn_index();
        let residence = self.alloc_residence(turn_stream_id(timeline.raw(), idx.0));
        let block_start = write.block_start;
        let block_end = write.block_end;
        {
            let tl = self.timelines.get_mut(&timeline).unwrap();
            tl.turns.insert(
                idx,
                TurnEntryData {
                    block_range: (block_start, block_end),
                    content: TurnPart {
                        user_text: write.user_text,
                        assistant_text: write.assistant_text,
                        token_count: write.token_count,
                        token_ids: write.token_ids,
                        sig_entries: Vec::new(),
                        residence,
                    },
                },
            );
        }
        if !sealed_cpu.is_empty() {
            self.install_hot(residence, sealed_cpu);
        }
        Ok(idx)
    }

    /// Insert a turn reconstructed from the redo log — the substrate-reload
    /// path (§16.12 of `docs/kv_tier_migration.md`).
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
        user_text: String,
        assistant_text: String,
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
        let residence = self.alloc_residence(turn_stream_id(timeline.raw(), idx.0));
        {
            let tl = self.timelines.get_mut(&timeline).unwrap();
            tl.turns.insert(
                idx,
                TurnEntryData {
                    block_range: (block_start, block_end),
                    content: TurnPart {
                        user_text,
                        assistant_text,
                        token_count,
                        token_ids,
                        sig_entries: Vec::new(),
                        residence,
                    },
                },
            );
        }
        if let Some(cold_seqs) = cold {
            if !cold_seqs.is_empty() {
                // Sum cold record bytes into residence.byte_size so
                // the purge accounting + telemetry has a real number
                // before any hot/warm elevate fires.
                let total_bytes: u64 = cold_seqs
                    .iter()
                    .flat_map(|s| s.chunks.iter())
                    .map(|c| c.record_len)
                    .sum();
                self.residence[residence.0].byte_size = total_bytes;
                self.install_cold(residence, cold_seqs);
            }
        }
        idx
    }

    /// Drop a turn's hot residence, freeing its VRAM arena chunks
    /// (the inner `Arc<ChunkGid>`s reach refcount 0 once any live
    /// borrowers release them). Removes the slot from the hot LRU.
    ///
    /// Returns `true` if the residence had hot bytes that were
    /// dropped; `false` when the turn isn't tracked or was already
    /// cold-marker. Callers use the boolean to decide whether to
    /// charge eviction accounting — they never need the dropped
    /// bytes themselves (the residence's warm/cold tiers hold the
    /// canonical copies if any future re-elevation needs them).
    /// Drop the hot residences of both halves of the turn.  Returns
    /// `true` if either half had hot bytes to drop.
    pub fn clear_turn_sealed(&mut self, timeline: TimelineId, index: TurnIndex) -> bool {
        let Some(residence) = self
            .turn(timeline, index)
            .map(|e| e.content.residence)
        else {
            return false;
        };
        if self.residence[residence.0].hot.take().is_some() {
            Self::remove_from_lru(&mut self.hot_lru, residence);
            true
        } else {
            false
        }
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

    /// Bytes a single hot-resident turn currently holds in VRAM.
    /// `None` for unknown turns or turns with no hot bytes.
    pub fn turn_hot_bytes(&self, timeline: TimelineId, index: TurnIndex) -> Option<usize> {
        let entry = self.turn(timeline, index)?;
        let residence = &self.residence[entry.content.residence.0];
        residence
            .hot
            .as_ref()
            .map(|_| residence.byte_size as usize)
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
                if self.residence[entry.content.residence.0].hot.is_some() {
                    return Some(key);
                }
            }
        }
        None
    }

    /// The user's message text for this turn — exactly as
    /// `submit_turn` received it.
    pub fn user_text_of(&self, timeline: TimelineId, index: TurnIndex) -> String {
        self.turn(timeline, index)
            .map(|e| e.content.user_text.clone())
            .unwrap_or_default()
    }

    /// The assistant's decoded reply text for this turn.
    pub fn assistant_text_of(&self, timeline: TimelineId, index: TurnIndex) -> String {
        self.turn(timeline, index)
            .map(|e| e.content.assistant_text.clone())
            .unwrap_or_default()
    }

    /// Turn token IDs as an owned `Vec` (clones the buffer).
    pub fn token_ids_of(&self, timeline: TimelineId, index: TurnIndex) -> Vec<u32> {
        self.turn(timeline, index)
            .map(|e| e.content.token_ids[..].to_vec())
            .unwrap_or_default()
    }

    /// Turn token IDs as a borrowed slice (zero-copy).
    pub fn assistant_token_ids_of(&self, timeline: TimelineId, index: TurnIndex) -> &[u32] {
        self.turn(timeline, index)
            .map_or(&[][..], |e| &e.content.token_ids[..])
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
            entry.content.token_count =
                entry.content.token_count.saturating_add(additional_tokens);
            entry.block_range.1 = new_block_end;
        }
    }

    pub fn block_range_of(&self, timeline: TimelineId, index: TurnIndex) -> (u64, u64) {
        self.turn(timeline, index)
            .map_or((0, 0), |e| e.block_range)
    }

    /// Set the turn's BDP sig entries — one per chunk in the
    /// content's residence, in slot block order.
    pub fn set_sig_entries(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
        entries: Vec<crate::provenance::SigEntry>,
    ) {
        if let Some(entry) = self.turn_mut(timeline, index) {
            entry.content.sig_entries = entries;
        }
    }

    pub fn extend_sig_entries(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
        entries: impl IntoIterator<Item = crate::provenance::SigEntry>,
    ) {
        if let Some(entry) = self.turn_mut(timeline, index) {
            entry.content.sig_entries.extend(entries);
        }
    }

    /// Turn's BDP sig entries — one per chunk in the content's
    /// residence, slot-block ordered.
    pub fn sig_entries_of(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Vec<crate::provenance::SigEntry> {
        self.turn(timeline, index)
            .map(|e| e.content.sig_entries.clone())
            .unwrap_or_default()
    }

    /// Sealed K/V for the turn — Arc-cloned per-layer
    /// `SealedSequence` snapshot of the content residence's hot
    /// tier.  Used by the projection assembler when injecting a
    /// `SealedKind::Turn` segment onto a slot.
    pub fn turn_sealed_of(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Option<Arc<Vec<SealedSequence>>> {
        let residence = self.turn(timeline, index)?.content.residence;
        let hot = self.residence[residence.0].hot.as_ref()?;
        Some(Arc::new(hot.clone()))
    }

    /// Turn token count — pinned bytes the seal recorded.
    pub fn turn_token_count_of(&self, timeline: TimelineId, index: TurnIndex) -> usize {
        self.turn(timeline, index)
            .map_or(0, |e| e.content.token_count)
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

    /// Whether `timeline` has been archived by the user. Untouched
    /// timelines default to `false`. Returns `false` for unknown
    /// timelines (matches "not archived" since the conversation
    /// doesn't exist as far as the sidebar is concerned).
    pub fn is_archived(&self, timeline: TimelineId) -> bool {
        self.timelines
            .get(&timeline)
            .is_some_and(|e| e.archived)
    }

    /// Set a timeline's archived flag. No-op when the timeline isn't
    /// registered. Returns `true` when the flag actually changed —
    /// the daemon uses this to skip the persistence write when the
    /// caller is just re-asserting the current state.
    pub fn set_archived(&mut self, timeline: TimelineId, archived: bool) -> bool {
        let Some(entry) = self.timelines.get_mut(&timeline) else {
            return false;
        };
        if entry.archived == archived {
            return false;
        }
        entry.archived = archived;
        true
    }

    /// Every recovered timeline that has a `conv_id` recorded, paired
    /// with `(conv_id, label, archived)`. Drives the daemon's sidebar:
    /// `label` is empty during the brief window between first-submit
    /// and titler-completion, `archived` is the lifecycle filter the
    /// sidebar applies before rendering.
    pub fn known_conversations(&self) -> Vec<(TimelineId, String, String, bool)> {
        self.timelines
            .iter()
            .filter_map(|(tl, entry)| {
                let conv_id = entry.conv_id.clone()?;
                let label = entry.label.clone().unwrap_or_default();
                Some((*tl, conv_id, label, entry.archived))
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
        // at conversation setup), so `cold` simply stays `None` and the
        // stream id is the reserved sentinel until sections gain a
        // durable home.
        let residence = self.alloc_residence(StreamId::default());
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
        self.turn(timeline, index)
            .map_or(0, |e| e.content.token_count)
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
        self.substrate
            .turn(timeline, index)
            .map_or(0, |e| e.content.token_count)
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
            .map_or(0, |e| e.content.token_count)
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

    /// Structurally-minimal `SealedSequence` suitable for migration-
    /// closure tests that don't care about chunk contents. The
    /// substrate's residence model treats a `Vec<SealedSequence>`
    /// containing zero elements as cold-marker (`hot = None`), so
    /// tests that want `turn_sealed_of` / `section_sealed_of` to
    /// return `Some` must pass at least one element. The empty
    /// `chunks` field is fine — it just means "this layer's sequence
    /// has no chunks", which is structurally valid.
    fn minimal_sealed_layer() -> SealedSequence {
        SealedSequence {
            chunks: Vec::new(),
            token_count: 0,
            chunk_size: 32,
            location: candle_nn::kv_cache::ArenaLocation::Cpu,
        }
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

    /// `append_complete` calls the migration closure and installs the
    /// result into the assistant residence's hot tier — `turn_sealed_of`
    /// returns it.
    #[test]
    fn append_complete_stores_migrated_result() {
        let (_, _, timeline, mut sub) = make_timeline();

        // Migration that returns a 2-layer minimal sealed sequence
        // (non-empty so it crosses install_hot's `!is_empty()` gate
        // and the residence lands hot).
        let migrate = |_input: &[SealedSequence]| -> candle::Result<Vec<SealedSequence>> {
            Ok(vec![minimal_sealed_layer(), minimal_sealed_layer()])
        };

        let idx = sub
            .append_complete(
                timeline,
                TurnPartWrite {
                    assistant_text: "hello".to_string(),
                    token_count: 3,
                    block_end: 1,
                    sealed_gpu: Some(Arc::new(vec![])),
                    ..Default::default()
                },
                migrate,
            )
            .unwrap();

        let stored = sub.turn_sealed_of(timeline, idx).unwrap();
        assert_eq!(stored.len(), 2, "two layers installed");
    }

    /// `set_section_full` calls the migration closure and installs
    /// the result into the section residence's hot tier.
    #[test]
    fn set_section_full_stores_migrated_result() {
        let mut sub = Substrate::new();
        let section = SectionId::new(42);

        let sealed_gpu = Arc::new(vec![minimal_sealed_layer(), minimal_sealed_layer()]);
        sub.set_section_full(
            section,
            10,
            vec![],
            sealed_gpu,
            identity_migrate,
            Arc::new(vec![1u32, 2, 3]),
        )
        .unwrap();

        let stored = sub.section_sealed_of(section).expect("hot installed");
        assert_eq!(stored.len(), 2, "two layers installed");
        assert_eq!(
            sub.sections.get(&section).unwrap().tokens.as_slice(),
            &[1u32, 2, 3]
        );
    }

    /// After `reset()`, the substrate is empty.
    #[test]
    fn reset_clears_substrate() {
        let (_, _, timeline, mut sub) = make_timeline();

        sub.append_complete(
            timeline,
            TurnPartWrite {
                sealed_gpu: Some(Arc::new(vec![])),
                ..Default::default()
            },
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

        let migrate = |_: &[SealedSequence]| -> candle::Result<Vec<SealedSequence>> {
            Ok(vec![minimal_sealed_layer()])
        };

        let idx0 = sub
            .append_complete(
                timeline,
                TurnPartWrite {
                    sealed_gpu: Some(Arc::new(vec![])),
                    ..Default::default()
                },
                migrate,
            )
            .unwrap();
        let idx1 = sub
            .append_complete(
                timeline,
                TurnPartWrite {
                    sealed_gpu: Some(Arc::new(vec![])),
                    ..Default::default()
                },
                migrate,
            )
            .unwrap();

        assert!(sub.turn_sealed_of(timeline, idx0).is_some());
        assert!(sub.turn_sealed_of(timeline, idx1).is_some());
        assert_ne!(idx0, idx1);
    }

    /// Helper: install a warm-only residence with a known byte_size.
    /// Returns the residence index. Used by the purge tests below to
    /// drive the warm LRU without going through the (CUDA-only)
    /// migrate path.
    fn install_warm_only(
        sub: &mut Substrate,
        timeline: TimelineId,
        bytes: u64,
    ) -> ResidenceIndex {
        let idx = sub
            .append_complete(
                timeline,
                TurnPartWrite {
                    sealed_gpu: Some(Arc::new(vec![])),
                    ..Default::default()
                },
                identity_migrate,
            )
            .unwrap();
        let residence = sub.turn_residence(timeline, idx).unwrap();
        // Drop hot, install a marker warm payload, set byte_size for
        // accounting. The actual SealedSequence content doesn't matter
        // for the purge — purge_warm_to_target counts `residence.
        // byte_size`, not the warm vec's bytes.
        sub.residence[residence.0].hot = None;
        sub.residence[residence.0].byte_size = bytes;
        // A non-empty placeholder so the warm slot is `Some`.
        let placeholder = vec![SealedSequence {
            chunks: Vec::new(),
            token_count: 0,
            chunk_size: 32,
            location: candle_nn::kv_cache::ArenaLocation::Cpu,
        }];
        sub.install_warm(residence, placeholder);
        residence
    }

    /// Ample headroom → purge is a no-op even with warm-resident slots.
    #[test]
    fn purge_with_ample_headroom_is_noop() {
        let (_, _, timeline, mut sub) = make_timeline();
        install_warm_only(&mut sub, timeline, 1_000_000);
        install_warm_only(&mut sub, timeline, 1_000_000);

        // 64 GB total, 32 GB available, incoming 1 MB. Threshold is
        // max(2 GiB, 5% * 64 GB) = 3.2 GB. 32 GB - 1 MB >> 3.2 GB.
        let r = sub.purge_warm_to_target(
            1_000_000,
            32 * 1024 * 1024 * 1024,
            64 * 1024 * 1024 * 1024,
        );
        assert_eq!(r.count, 0);
        assert_eq!(r.bytes, 0);
    }

    /// Tight headroom → keep popping LRU until projected available
    /// covers the threshold.
    #[test]
    fn purge_drops_lru_until_threshold_met() {
        let (_, _, timeline, mut sub) = make_timeline();
        // Order matters: a is LRU (installed first → pushed to front,
        // then b → b is at front, a slides toward back). pop_back
        // returns a first.
        let a = install_warm_only(&mut sub, timeline, 500_000_000);
        let b = install_warm_only(&mut sub, timeline, 500_000_000);

        // 8 GB total, 2 GB available, incoming 1 GB. Threshold is
        // max(2 GiB = 2_147_483_648, 8 GB / 20 = 400 MB) = 2 GiB.
        // projected = 2 GB - 1 GB = 1 GB < 2 GiB → must purge.
        // After dropping a (500 MB): projected = 1 GB + 500 MB = 1.5 GB
        //   still < 2 GiB → drop b.
        // After dropping b: projected = 1 GB + 1 GB = 2 GB. 2 GB <
        //   2 GiB (2 GiB = 2.147 GB) → still under, but warm_lru is
        //   now empty, so loop exits.
        let r = sub.purge_warm_to_target(
            1_000_000_000,
            2 * 1000 * 1000 * 1000,
            8 * 1000 * 1000 * 1000,
        );
        assert_eq!(r.count, 2, "both warm residences should be popped");
        assert_eq!(r.bytes, 1_000_000_000);
        assert!(sub.residence[a.0].warm.is_none(), "a (LRU) dropped first");
        assert!(sub.residence[b.0].warm.is_none(), "b also dropped");
    }

    /// Empty warm LRU → purge exits gracefully without panicking even
    /// when the threshold can't be met.
    #[test]
    fn purge_handles_empty_warm_lru() {
        let mut sub = Substrate::new();
        // No warm residences exist; the loop should exit on the first
        // `pop_back` returning None.
        let r = sub.purge_warm_to_target(
            10 * 1024 * 1024 * 1024,
            1 * 1024 * 1024 * 1024, // 1 GiB available
            64 * 1024 * 1024 * 1024, // 64 GiB total
        );
        assert_eq!(r.count, 0);
        assert_eq!(r.bytes, 0);
    }

    /// 5% rule fires when 5% × total_ram > 2 GiB. With 256 GB total
    /// the threshold is 12.8 GB, not 2 GiB.
    #[test]
    fn purge_threshold_is_max_of_2gib_and_5_percent() {
        let (_, _, timeline, mut sub) = make_timeline();
        // One LRU warm victim of 2 GB.
        install_warm_only(&mut sub, timeline, 2_000_000_000);

        // 256 GB total, 14 GB available, incoming 0. Threshold =
        // max(2 GiB, 256 GB * 0.05 = 12.8 GB) = 12.8 GB.
        // projected = 14 GB > 12.8 GB → no purge.
        let r = sub.purge_warm_to_target(
            0,
            14 * 1000 * 1000 * 1000,
            256 * 1000 * 1000 * 1000,
        );
        assert_eq!(r.count, 0, "14 GB available > 5% × 256 GB threshold");

        // Now 13 GB available, threshold still 12.8 GB. projected =
        // 13 GB > 12.8 GB → still no purge.
        let r = sub.purge_warm_to_target(
            0,
            13 * 1000 * 1000 * 1000,
            256 * 1000 * 1000 * 1000,
        );
        assert_eq!(r.count, 0);

        // 12 GB available → projected < threshold → purge fires.
        let r = sub.purge_warm_to_target(
            0,
            12 * 1000 * 1000 * 1000,
            256 * 1000 * 1000 * 1000,
        );
        assert_eq!(r.count, 1);
        assert_eq!(r.bytes, 2_000_000_000);
    }

    /// `restore_turn` with `cold = Some(...)` lands the residence as
    /// cold-marker — classifier-ready for cold→hot promotion. With
    /// `cold = None`, the residence is left empty (legitimately
    /// recoverable but missing chunks).
    #[test]
    fn restore_turn_populates_cold_tier_when_provided() {
        let (_, _, timeline, mut sub) = make_timeline();

        let cold_payload = vec![StoredSequence {
            chunks: vec![StoredChunk {
                log_offset: 4096,
                record_len: 1024,
                token_count: 32,
            }],
            token_count: 32,
        }];
        let idx = sub.restore_turn(
            timeline,
            String::new(),
            String::new(),
            TokenBuffer::default(),
            32,
            Some(cold_payload),
            0,
            1,
        );
        let residence = sub.turn_residence(timeline, idx).unwrap();
        assert!(sub.residence[residence.0].cold.is_some(), "cold installed");
        assert!(sub.residence[residence.0].hot.is_none(), "hot empty (cold-marker)");
        assert!(sub.residence[residence.0].warm.is_none(), "warm empty");
        assert_eq!(
            sub.residence[residence.0].byte_size, 1024,
            "byte_size summed from cold record_len"
        );

        // None branch — no cold payload.
        let idx2 = sub.restore_turn(
            timeline,
            String::new(),
            String::new(),
            TokenBuffer::default(),
            0,
            None,
            0,
            0,
        );
        let r2 = sub.turn_residence(timeline, idx2).unwrap();
        assert!(sub.residence[r2.0].cold.is_none());
        assert!(sub.residence[r2.0].hot.is_none());
        assert!(sub.residence[r2.0].warm.is_none());
    }

    /// Archive flag defaults to `false`, can be toggled, and the
    /// idempotency contract holds (setting the same value twice
    /// returns `false` so the caller can short-circuit the
    /// persistence write).
    #[test]
    fn archive_flag_toggles_with_idempotency() {
        let (layer, group, timeline, mut sub) = make_timeline();
        let _ = (layer, group);
        // Untouched: archived defaults to false.
        assert!(!sub.is_archived(timeline));

        // First write: state actually changed, returns true.
        assert!(sub.set_archived(timeline, true));
        assert!(sub.is_archived(timeline));

        // Second write of the same value: idempotent, returns false.
        assert!(!sub.set_archived(timeline, true));
        assert!(sub.is_archived(timeline));

        // Unarchive: state changes, returns true.
        assert!(sub.set_archived(timeline, false));
        assert!(!sub.is_archived(timeline));

        // Setting on an unknown timeline: returns false (and no-op).
        let bogus = TimelineId::from_raw(999).unwrap();
        assert!(!sub.set_archived(bogus, true));
        assert!(!sub.is_archived(bogus));
    }

    /// `known_conversations` exposes the archived flag — the daemon
    /// uses this to drive the sidebar filter.
    #[test]
    fn known_conversations_exposes_archived_flag() {
        let (_, _, timeline, mut sub) = make_timeline();
        sub.set_conv_id(timeline, "abc");
        sub.set_label(timeline, "tour");
        sub.set_archived(timeline, true);

        let convs = sub.known_conversations();
        assert_eq!(convs.len(), 1);
        let (tl, conv_id, label, archived) = &convs[0];
        assert_eq!(*tl, timeline);
        assert_eq!(conv_id, "abc");
        assert_eq!(label, "tour");
        assert!(*archived);
    }
}
