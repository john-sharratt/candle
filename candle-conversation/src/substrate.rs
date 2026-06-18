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
use candle_nn::kv_cache::{QuantFormat, SealedSequence};
use std::collections::{BTreeMap, HashMap, HashSet, LinkedList};
use std::sync::Arc;

use crate::persistence::content_hash::turn_stream_id;
use crate::persistence::manifest::{
    decode_conv_state_payload, decode_label_payload, ChunkLoc, ConvMeta, ConvState, RecordLoc,
};
use crate::persistence::record::{
    DebugIdPayload, RecordType, TombstonePayload, TreeMetadataPayload,
};
use crate::persistence::streams::{StreamDecl, StreamId};
use crate::persistence::walker::WalkEntry;
use crate::projection::{DepthWeights, ScoreFormula};
use crate::projection::{
    GroupId, LayerId, SectionId, TimelineAllocator, TimelineId, TurnIndex, TurnKey,
};
use crate::summary_tree::{
    select_dense, Node, NodeId, RecencyConfig, SelectionDiagnostics, SelectionOrigin, SummaryTree,
    TurnKind,
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

    /// Per-timeline KV-compression override (set from the conversation's
    /// `SequenceConfig` at creation). Copied onto each turn residence at
    /// alloc time so the persistence thread can quantize different
    /// conversations at different levels. Absent ⇒ engine-wide turn policy.
    timeline_compression: HashMap<TimelineId, ConvCompression>,

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

    /// Per-stream in-RAM index of where each chunk / tokens record /
    /// signatures record / committed-through watermark sits on disk.
    /// Built by replaying the redo log on startup (and updated on
    /// every fresh append).  Cold-load and seal-time persistence read
    /// this directly — it used to live as `Manifest.streams`, but
    /// since the manifest gets serialised into every `Checkpoint`
    /// record, mirroring per-chunk pointers there forced the
    /// checkpoint payload to scale with chunk count.  Holding the
    /// index in the substrate's RAM instead bounds the checkpoint
    /// payload to a few hundred bytes regardless of stream size; the
    /// per-stream `BTreeMap` only ever sits in memory.
    streams: HashMap<StreamId, StreamRuntime>,

    /// Reverse index: stable resume keys (`debug_id`) → `TimelineId`.
    /// Populated by [`Self::set_debug_id`] and the cold-load reader.
    /// Provides O(1) lookup for the test-harness `find_or_create`
    /// pattern (§10.4 of `docs/infinite_conversations.md`).
    timeline_by_debug_id: HashMap<String, TimelineId>,

    /// Walker scratch: `Label` / `ConvState` payloads decoded for a
    /// timeline that hasn't yet been registered.  Zend writes the
    /// `Label` (carrying conv_id) immediately on conversation
    /// creation, before any TurnDecl exists, so the walker hits the
    /// Label record first and the TurnDecl second.  Stashing here
    /// lets `register_timeline` drain and apply pending meta when
    /// the timeline finally registers — otherwise restored
    /// conversations would vanish from the sidebar listing because
    /// their `conv_id` would never reach the TimelineEntry.
    pending_conv_meta: HashMap<u64, ConvMeta>,
    pending_conv_state: HashMap<u64, ConvState>,

    /// Timelines flagged by [`RecordType::Tombstone`] as logically
    /// deleted.  [`Self::active_timelines_for_group`] filters them
    /// out so projection never surfaces their turns; the compactor
    /// drops the on-disk records during the next compaction pass.
    /// Entries may name timelines that are not yet registered
    /// (walker can apply a tombstone before its target
    /// `StreamDecl::Turn`) — registration just observes them as
    /// already tombstoned, which is the correct behaviour.
    tombstoned_timelines: HashSet<TimelineId>,
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
/// Per-conversation KV-compression override, resolved from the owning
/// timeline's [`SequenceConfig`] at residence-allocation time and read by
/// the persistence thread's hot→warm quantize pass. `None` on a residence
/// means "use the engine-wide turn policy"; `Some` applies the overrides
/// below. Used to compress utility layers (e.g. `code_reading`) harder than
/// live dialogue, or to pin dialogue turns to a fixed near-lossless format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ConvCompression {
    /// Adaptive compression level override. `Some` replaces the engine-wide
    /// turn level; `None` keeps it (used when only the forced formats below
    /// are set).
    pub level: Option<u8>,
    /// Drop the global K-format override (Q4_KS) so K is adaptively quantized
    /// per-block like V.
    pub disable_k_override: bool,
    /// Force every K block of this conversation's turns to a single uniform
    /// quant format, bypassing adaptive per-block selection (and the global
    /// Q4_KS K override). `None` keeps the engine-wide K behaviour.
    pub force_k: Option<QuantFormat>,
    /// V counterpart to [`Self::force_k`].
    pub force_v: Option<QuantFormat>,
}

#[derive(Debug)]
pub struct SequenceResidence {
    /// Persistence-layer stream identity for this residence. Set at
    /// allocation time, immutable. Turns derive it from
    /// `turn_stream_id(timeline, index)`; sections that don't persist
    /// to disk use [`StreamId::default()`] (the reserved sentinel).
    pub stream_id: StreamId,
    /// Per-conversation compression override inherited from the owning
    /// timeline at alloc time (turns only; sections are `None`). The
    /// persistence thread groups residences by this when quantizing
    /// hot→warm so different conversations can target different levels.
    pub compression: Option<ConvCompression>,
    /// VRAM-resident sealed chunks. `None` ⇒ not in VRAM.
    pub hot: Option<Vec<SealedSequence>>,
    /// `true` while the scheduler still owes this section a quantize
    /// pass.  Sections are ingested with their native (prefill-output)
    /// K/V installed in `hot`; the scheduler later replaces `hot` with
    /// the quantized form at the next turn-seal boundary.  Until that
    /// drain runs, `hot` is the *interim* native form — disk
    /// persistence must skip the residence (a cold copy of native
    /// would diverge from the final in-memory Q form after the drain,
    /// and the daemon would resume in an inconsistent state on
    /// restart).  Cleared by the same drain that does the swap.
    pub pending_quantize: bool,
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
    /// Items the substrate has no record of at all — neither a turn
    /// entry in `timelines.<tl>.turns` nor a section entry in
    /// `sections`.  This is the genuinely-problematic bucket: the
    /// projection plan referenced an id that the substrate cannot
    /// resolve, which is either a stale projection or a substrate
    /// invariant violation.  `elevate_to_hot` logs a `WARN` per
    /// item in this bucket.
    pub missing: Vec<PromotionItemKind>,
    /// Items the substrate DOES have a record of, but whose
    /// residence has no hot/warm/cold tier installed — so there's
    /// nothing to elevate.  The two cases that produce this:
    ///
    ///  1. Ghost summary turns appended by the
    ///     `substrate-summariser` thread via
    ///     `record_summary_turn` → `append_with_blocks(0..0)`.
    ///     These declare a node in the per-timeline summary tree
    ///     and carry no K/V, by design.
    ///  2. A turn whose tier state was rolled back (a
    ///     `clear_turn_sealed` call between projection and elevate).
    ///
    /// `elevate_to_hot` silently skips these items.  Distinct from
    /// `missing` so the WARN doesn't fire for legitimately-tier-less
    /// turns.
    pub tier_less: Vec<PromotionItemKind>,
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
    pub fn set_turn(&mut self, timeline: TimelineId, index: TurnIndex, scores: PerDepthScores) {
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
    /// K/V chunk's `token_count` fields.  Holds the invariant
    /// `token_count == token_ids.len()` — every persisted token id
    /// has a corresponding slot position in the K/V chunk grid.  The
    /// decode-loop trailing-terminator (EOS or max-tokens edge token,
    /// sampled but never forwarded) is trimmed at seal time to
    /// preserve this; a `debug_assert_eq!` at the seal site guards
    /// against any future regression.
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

    /// Score-density selection over a timeline's summary tree
    /// (`docs/infinite_conversations.md` §8).  Returns the chrono-
    /// logically ordered `(turn_index, effective_score)` list for the
    /// given timeline, fitted into `budget` tokens, or `None` when no
    /// summary tree exists yet for that timeline.
    ///
    /// When `Some(_)`, the projection's step-9 reconciler skips the
    /// flexbox / rule-based budget allocation for this timeline's
    /// target group and emits the returned list verbatim — score-
    /// density already picked turns inside `budget`.
    ///
    /// Default impl returns `None` so existing test resolvers and
    /// non-summary-tree groups keep their current behaviour.
    fn summary_tree_select(
        &self,
        _timeline: TimelineId,
        _budget: u32,
        _formula: ScoreFormula,
        _weights: &DepthWeights,
    ) -> Option<Vec<(TurnIndex, SelectionOrigin, f32)>> {
        None
    }

    /// Number of turns currently awaiting the async summariser for
    /// `timeline`.  Used by the projection to populate the
    /// score-density backpressure metric inside its diagnostic sink
    /// (§9 of `docs/infinite_conversations.md`).  Default returns 0.
    fn pending_summary_len(&self, _timeline: TimelineId) -> usize {
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
    /// Free-form key/value metadata, persisted in the same
    /// `RecordType::Label` record as `label`/`conv_id` and merged
    /// (last-write-wins per key) on update. Used as a content-addressed
    /// cache index by utility ingests; searchable via
    /// [`Substrate::timelines_with_metadata`].
    pub custom: BTreeMap<String, String>,
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
    /// Per-turn tree metadata for the infinite-conversation summary
    /// tree (`docs/infinite_conversations.md` §5).  Parallel to
    /// `turns`: every recorded turn carries exactly one
    /// [`TreeNodeMeta`] entry (defaults to a `Normal` content
    /// sub-leaf with no children).  Promoted to a
    /// `SummaryOfTurns` / `SummaryOfSummaries` by the async
    /// summariser thread after the §6 probe runs.
    pub tree_meta: BTreeMap<TurnIndex, TreeNodeMeta>,
    /// Root of the per-timeline summary tree.  `None` until the first
    /// `SummaryOfTurns` leaf is sealed.  Persisted as part of the
    /// timeline's `TreeMetadata` records on the redo log and
    /// reconstructed on cold-load.
    pub tree_root: Option<TurnIndex>,
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
    pub pending_summary_queue: std::collections::VecDeque<TurnIndex>,
    /// Whether this timeline's turns are fed to the summariser at all.
    /// `true` for dialogue; `false` for append-only utility/reference layers
    /// (repo_map, code_reading) — they are background reference, summarising
    /// them is pointless work that storms the summariser during repo
    /// ingest/scan. When `false`, turns are never pushed onto
    /// `pending_summary_queue`, so the summariser never touches this timeline.
    pub summarize: bool,
    /// Summary nodes whose stored Q-fingerprint is stale because their
    /// children changed (most commonly after an AVL rotation).  The
    /// summariser sweep regenerates one per pass (§7.3).
    pub dirty_summary_set: std::collections::BTreeSet<TurnIndex>,
    /// Most recent score-density [`SelectionDiagnostics`] for this
    /// timeline, written by the scheduler at projection time and read
    /// by the test harness via
    /// [`Substrate::last_selection_of`].  Last-write-wins across
    /// re-projections within a single turn — by the time `send_turn`
    /// returns, this holds the final selection that produced the
    /// model's response.  Test-harness diagnostic only; production
    /// daemons can ignore it.
    pub last_selection: Option<SelectionDiagnostics>,
}

/// Tree-bookkeeping metadata stored alongside every substrate turn.
///
/// Every persisted turn carries one of these — defaults are a `Normal`
/// content sub-leaf with no children and a clean dirty flag.  Summary
/// nodes (produced by the §6 probe) overwrite the defaults with the
/// real kind / children / height when the summariser thread seals
/// them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeNodeMeta {
    /// Three-kind tag from `summary_tree::TurnKind`, mirrored here so
    /// the substrate is the single source of truth and the redo-log
    /// codec can round-trip without depending on the algorithm module.
    pub kind: TurnKind,
    /// For `SummaryOfTurns`: the Normal-turn children in
    /// chronological order.  For `SummaryOfSummaries`: exactly the
    /// `[left, right]` summary children.  For `Normal`: empty.
    pub children: Vec<TurnIndex>,
    /// AVL height for binary summary nodes.  Always `0` for `Normal`.
    /// `SummaryOfTurns` defaults to `1`; `SummaryOfSummaries` carries
    /// `max(child_height) + 1` per the standard AVL invariant.
    pub tree_height: u8,
    /// `true` when this summary's children have changed since the
    /// stored content (and its Q-fingerprint) was generated.  The
    /// summariser sweep clears this when it regenerates.
    pub dirty: bool,
}

impl Default for TreeNodeMeta {
    fn default() -> Self {
        Self {
            kind: TurnKind::Normal,
            children: Vec::new(),
            tree_height: 0,
            dirty: false,
        }
    }
}

impl TreeNodeMeta {
    /// Sensible default for a Normal content turn.
    pub fn normal() -> Self {
        Self::default()
    }
}

/// Per-stream in-RAM runtime state — built by replaying the redo log
/// on startup and updated on every fresh append.
///
/// Holds everything `Manifest.streams.<id>` used to hold; moving it
/// here keeps the `Checkpoint` record payload bounded by singleton
/// count instead of per-chunk count.  The chunk index supports O(1)
/// `(stream_id, chunk_idx) → ChunkLoc` lookup for cold-load.
#[derive(Debug, Clone, Default)]
pub struct StreamRuntime {
    /// The decoded stream declaration (`StreamDecl` record).
    pub decl: Option<StreamDecl>,
    /// Live chunk locations by chunk index — last-writer-wins.
    pub chunks: BTreeMap<u64, ChunkLoc>,
    /// Latest `Tokens` record for the stream.
    pub tokens: Option<RecordLoc>,
    /// Latest `Signatures` record for the stream.
    pub signatures: Option<RecordLoc>,
    /// Highest chunk index the stream is durably committed through.
    pub committed_through: Option<u64>,
}

impl TimelineEntry {
    fn new(layer: LayerId, group: GroupId) -> Self {
        Self {
            layer,
            group,
            label: None,
            conv_id: None,
            custom: BTreeMap::new(),
            archived: false,
            turns: BTreeMap::new(),
            tree_meta: BTreeMap::new(),
            tree_root: None,
            debug_id: None,
            pending_summary_queue: std::collections::VecDeque::new(),
            summarize: true,
            dirty_summary_set: std::collections::BTreeSet::new(),
            last_selection: None,
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
    fn alloc_residence(
        &mut self,
        stream_id: StreamId,
        compression: Option<ConvCompression>,
    ) -> ResidenceIndex {
        let idx = ResidenceIndex(self.residence.len());
        self.residence.push(SequenceResidence {
            stream_id,
            compression,
            hot: None,
            pending_quantize: false,
            warm: None,
            cold: None,
            byte_size: 0,
        });
        idx
    }

    /// Set (or clear) the per-conversation compression override for
    /// `timeline`. Called at conversation creation from the
    /// `SequenceConfig`. Must run before the timeline's first turn seals
    /// so each turn residence picks it up at alloc time.
    pub fn set_timeline_compression(
        &mut self,
        timeline: TimelineId,
        compression: Option<ConvCompression>,
    ) {
        match compression {
            Some(cc) => {
                self.timeline_compression.insert(timeline, cc);
            }
            None => {
                self.timeline_compression.remove(&timeline);
            }
        }
    }

    /// Mark whether `timeline`'s turns should be summarised. Set `false` for
    /// append-only utility/reference layers (repo_map, code_reading) so their
    /// turns never enter the summariser's pending queue. Must run after the
    /// timeline is registered and before its first turn seals.
    pub fn set_timeline_summarize(&mut self, timeline: TimelineId, summarize: bool) {
        if let Some(tl) = self.timelines.get_mut(&timeline) {
            tl.summarize = summarize;
        }
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
    fn install_section_hot(&mut self, residence: ResidenceIndex, sealed: Vec<SealedSequence>) {
        debug_assert!(
            !sealed.is_empty(),
            "install_section_hot called with empty Vec"
        );
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
        debug_assert!(
            !hot.is_empty(),
            "install_warm_and_hot called with empty hot"
        );
        debug_assert!(
            !warm.is_empty(),
            "install_warm_and_hot called with empty warm"
        );
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
    pub fn snapshot_pending_warm(
        &self,
    ) -> Vec<(ResidenceIndex, Vec<SealedSequence>, Option<ConvCompression>)> {
        self.hot_lru
            .iter()
            .filter_map(|&idx| {
                let slot = &self.residence[idx.0];
                if slot.warm.is_some() {
                    return None;
                }
                slot.hot
                    .as_ref()
                    .map(|hot| (idx, hot.clone(), slot.compression))
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
    pub fn snapshot_pending_cold(&self) -> Vec<(ResidenceIndex, StreamId, Vec<SealedSequence>)> {
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
            self.classify_one(PromotionItemKind::Section(sid), entry.residence, &mut plan);
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
        // The item is tracked in the substrate but its residence has
        // no tier installed.  This is the expected state for ghost
        // summary turns appended via `append_with_blocks(0..0)` —
        // they exist as tree-meta anchors without any K/V to load.
        // Route to `tier_less`, not `missing`, so `elevate_to_hot`'s
        // WARN doesn't fire for the design-intended case.
        plan.tier_less.push(kind);
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
            if !recall.warm.is_empty() && self.residence[recall.residence.0].warm.is_none() {
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
    pub fn turn_residence(&self, timeline: TimelineId, index: TurnIndex) -> Option<ResidenceIndex> {
        self.turn(timeline, index).map(|e| e.content.residence)
    }

    /// Resolve a section's residence index in the substrate.
    ///
    /// Used by the scheduler's section-quantize drain to look up the
    /// hot slot it needs to swap, and by integration tests that walk
    /// the section map directly.
    pub fn section_residence(&self, section: SectionId) -> Option<ResidenceIndex> {
        self.sections.get(&section).map(|e| e.residence)
    }

    /// Every section id currently registered in the substrate, in
    /// unspecified order.  Used by tooling that needs to walk the
    /// section map — integration tests, workspace diagnostics, the
    /// section-quantize regression test in `zend/tests/`.
    pub fn all_section_ids(&self) -> Vec<SectionId> {
        self.sections.keys().copied().collect()
    }

    /// Which tiers a turn residence currently occupies. Returns `None`
    /// when the turn isn't tracked. Load-bearing in production: the
    /// SubmitTurn handler uses this as a tier-agnostic existence
    /// check before adding a turn to the projection's elevate list,
    /// so cold-marker turns (post-restart, before any elevation has
    /// fired) still survive the filter and reach `elevate_to_hot`.
    pub fn turn_tier_state(&self, timeline: TimelineId, index: TurnIndex) -> Option<TierState> {
        let residence = self.turn(timeline, index).map(|e| e.content.residence)?;
        let slot = &self.residence[residence.0];
        Some(TierState {
            hot: slot.hot.is_some(),
            warm: slot.warm.is_some(),
            cold: slot.cold.is_some(),
        })
    }

    /// Section counterpart of [`Self::turn_tier_state`].  Sections
    /// can now occupy any of the three tiers: hot when prefilled
    /// fresh or post-elevate, cold-marker when restored from the
    /// redo log on daemon reload (lazy lift on next projection),
    /// and warm transiently during a tier transition.  Returns
    /// `None` for unknown section ids.
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
                (slot.hot.is_some() && slot.warm.is_some()).then_some((idx, slot.byte_size))
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

    /// Budget-aware hot eviction: evict the **least-recently-promoted** hot
    /// turns (oldest at the back of `hot_lru`) to warm, stopping as soon as
    /// `target_bytes` of VRAM has been freed. Unlike [`Self::evict_hot_except`]
    /// (which drops the entire non-selected working set every reproject), this
    /// keeps the recent working set resident and only frees what the incoming
    /// cold-load actually needs.
    ///
    /// Scoped to **this** conversation's residence (`self.hot_lru`) — it can
    /// never touch another conversation's hot KV. The selection
    /// (`keep_sections` / `keep_turns`) and sections (never on `hot_lru`) are
    /// always protected. Only items with both a hot and a warm copy are evicted,
    /// so eviction is hot→warm (the warm copy survives for a fast reload).
    pub fn evict_hot_to_free(
        &mut self,
        keep_sections: &[SectionId],
        keep_turns: &[TurnKey],
        target_bytes: u64,
    ) -> EvictionReport {
        if target_bytes == 0 {
            return EvictionReport { count: 0, bytes: 0 };
        }
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

        // Walk oldest→newest (back of the install/promote-ordered LRU first),
        // collecting victims until we've freed enough. `byte_size` is the VRAM
        // the hot copy holds, which `hot = None` releases.
        let mut freed: u64 = 0;
        let mut victims: Vec<ResidenceIndex> = Vec::new();
        for idx in self.hot_lru.iter().rev().copied() {
            if freed >= target_bytes {
                break;
            }
            if keep.contains(&idx) {
                continue;
            }
            let slot = &self.residence[idx.0];
            if slot.hot.is_some() && slot.warm.is_some() {
                freed += slot.byte_size;
                victims.push(idx);
            }
        }

        let count = victims.len();
        for idx in &victims {
            self.residence[idx.0].hot = None;
            Self::remove_from_lru(&mut self.hot_lru, *idx);
        }
        if count > 0 {
            tracing::info!(
                target: "candle_conversation::persistence::tier",
                count,
                bytes = freed,
                target_bytes,
                keep_size = keep.len(),
                "evict_hot_to_free (budget-aware) complete"
            );
        }
        EvictionReport {
            count,
            bytes: freed,
        }
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
        // Drain any walker-stashed conv meta / state that arrived
        // before this timeline registered.  Sidebar listing depends
        // on the conv_id landing on the TimelineEntry.
        if let Some(meta) = self.pending_conv_meta.remove(&timeline.raw()) {
            if !meta.conv_id.is_empty() {
                self.set_conv_id(timeline, &meta.conv_id);
            }
            if !meta.label.is_empty() {
                self.set_label(timeline, &meta.label);
            }
            if !meta.custom.is_empty() {
                self.merge_custom(timeline, &meta.custom);
            }
        }
        if let Some(state) = self.pending_conv_state.remove(&timeline.raw()) {
            let _ = self.set_archived(timeline, state.archived);
        }
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
        self.timelines
            .get(&timeline)
            .and_then(|t| t.turns.get(&index))
    }

    /// Mutable variant of [`Self::turn`].
    fn turn_mut(&mut self, timeline: TimelineId, index: TurnIndex) -> Option<&mut TurnEntryData> {
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

    /// Like [`Self::timelines_for_group`] but excludes timelines
    /// flagged `archived` or tombstoned.  Projection retrieval
    /// ([`ContentResolver`]) uses this so retired and user-archived
    /// conversations drop out of selection without their turns
    /// being physically deleted from the substrate.
    pub fn active_timelines_for_group(
        &self,
        group: GroupId,
    ) -> impl Iterator<Item = TimelineId> + '_ {
        self.timelines_by_group
            .get(&group)
            .into_iter()
            .flat_map(|v| v.iter().copied())
            .filter(move |tl| {
                if self.tombstoned_timelines.contains(tl) {
                    return false;
                }
                self.timelines.get(tl).map(|e| !e.archived).unwrap_or(true)
            })
    }

    // ── debug_id resume keys ────────────────────────────────────────────────

    /// Set the substrate-side resume key for a timeline.  Replaces any
    /// previous mapping for the same `id`; clears the old mapping if
    /// the timeline already had a different debug_id.  Used by the
    /// test harness's `find_or_create(debug_id)` path (§10.4) and
    /// reconstructed on cold-load from the timeline's record stream.
    pub fn set_debug_id(&mut self, timeline: TimelineId, id: impl Into<String>) {
        let id = id.into();
        if let Some(tl) = self.timelines.get_mut(&timeline) {
            if let Some(old) = tl.debug_id.take() {
                if old != id {
                    self.timeline_by_debug_id.remove(&old);
                }
            }
            tl.debug_id = Some(id.clone());
            self.timeline_by_debug_id.insert(id, timeline);
        }
    }

    /// Look up a timeline by its `debug_id`.  O(1).
    pub fn lookup_by_debug_id(&self, id: &str) -> Option<TimelineId> {
        self.timeline_by_debug_id.get(id).copied()
    }

    /// Current `debug_id` for a timeline, if one has been set.
    pub fn debug_id_of(&self, timeline: TimelineId) -> Option<&str> {
        self.timelines
            .get(&timeline)
            .and_then(|tl| tl.debug_id.as_deref())
    }

    // ── Tree metadata read / write ───────────────────────────────────────────

    /// Read the parallel [`TreeNodeMeta`] for a turn.  Returns `None`
    /// when the timeline or the turn index is unknown.
    pub fn tree_meta_of(&self, timeline: TimelineId, idx: TurnIndex) -> Option<&TreeNodeMeta> {
        self.timelines
            .get(&timeline)
            .and_then(|tl| tl.tree_meta.get(&idx))
    }

    /// Overwrite a turn's [`TreeNodeMeta`].  The summariser thread
    /// calls this once it has decided how a turn slots into the
    /// summary tree (Normal sub-leaf vs SummaryOfTurns leaf vs
    /// SummaryOfSummaries internal).  Clearing `dirty` is the caller's
    /// responsibility; this method writes the value verbatim.
    pub fn set_tree_meta(&mut self, timeline: TimelineId, idx: TurnIndex, meta: TreeNodeMeta) {
        if let Some(tl) = self.timelines.get_mut(&timeline) {
            tl.tree_meta.insert(idx, meta);
        }
    }

    /// Mark a summary node dirty (children changed; stored Q-fingerprint
    /// no longer reflects the subtree).  Adds the node to the dirty set
    /// so the summariser sweep picks it up.  No-op for Normal turns.
    pub fn mark_summary_dirty(&mut self, timeline: TimelineId, idx: TurnIndex) {
        if let Some(tl) = self.timelines.get_mut(&timeline) {
            if let Some(meta) = tl.tree_meta.get_mut(&idx) {
                if meta.kind.is_summary() {
                    meta.dirty = true;
                    tl.dirty_summary_set.insert(idx);
                }
            }
        }
    }

    /// Clear the dirty bit + remove from the dirty set.  Called by the
    /// summariser after a regeneration probe completes successfully.
    pub fn clear_summary_dirty(&mut self, timeline: TimelineId, idx: TurnIndex) {
        if let Some(tl) = self.timelines.get_mut(&timeline) {
            if let Some(meta) = tl.tree_meta.get_mut(&idx) {
                meta.dirty = false;
            }
            tl.dirty_summary_set.remove(&idx);
        }
    }

    /// Current tree root of a timeline.
    pub fn tree_root_of(&self, timeline: TimelineId) -> Option<TurnIndex> {
        self.timelines.get(&timeline).and_then(|tl| tl.tree_root)
    }

    /// Set the tree root for a timeline.  Used by the summariser
    /// thread after every insertion that surfaces a new root.
    pub fn set_tree_root(&mut self, timeline: TimelineId, root: Option<TurnIndex>) {
        if let Some(tl) = self.timelines.get_mut(&timeline) {
            tl.tree_root = root;
        }
    }

    // ── Pending + dirty queue accessors (§6 backpressure metrics) ──────────

    /// FIFO pop: next pending turn for the summariser to absorb, or
    /// `None` if the queue is empty.
    pub fn pop_pending_summary(&mut self, timeline: TimelineId) -> Option<TurnIndex> {
        self.timelines
            .get_mut(&timeline)
            .and_then(|tl| tl.pending_summary_queue.pop_front())
    }

    /// Push a turn onto the pending-summary queue.  Used during
    /// cold-load reconstruction (§4) to re-enqueue orphan turns whose
    /// summary parent didn't survive a crash.
    pub fn push_pending_summary(&mut self, timeline: TimelineId, idx: TurnIndex) {
        if let Some(tl) = self.timelines.get_mut(&timeline) {
            if !tl.summarize {
                return;
            }
            tl.pending_summary_queue.push_back(idx);
        }
    }

    /// Pop the oldest dirty summary node.  Returns `None` when no
    /// summary is currently dirty.  The summariser sweep regenerates
    /// at most one node per pass (§7.3).
    pub fn pop_oldest_dirty(&mut self, timeline: TimelineId) -> Option<TurnIndex> {
        let tl = self.timelines.get_mut(&timeline)?;
        let id = *tl.dirty_summary_set.iter().next()?;
        tl.dirty_summary_set.remove(&id);
        Some(id)
    }

    /// `pending_summary_queue.len()` — backpressure metric (§9).
    pub fn pending_summary_len(&self, timeline: TimelineId) -> usize {
        self.timelines
            .get(&timeline)
            .map(|tl| tl.pending_summary_queue.len())
            .unwrap_or(0)
    }

    /// `dirty_summary_set.len()` — backpressure metric (§9).
    pub fn dirty_summary_len(&self, timeline: TimelineId) -> usize {
        self.timelines
            .get(&timeline)
            .map(|tl| tl.dirty_summary_set.len())
            .unwrap_or(0)
    }

    /// Store the latest score-density [`SelectionDiagnostics`] for a
    /// timeline.  Called by the scheduler at projection time;
    /// last-write-wins across re-projections within a turn.  Pure
    /// test-harness instrumentation — production daemons can ignore.
    pub fn set_last_selection(&mut self, timeline: TimelineId, diag: SelectionDiagnostics) {
        if let Some(tl) = self.timelines.get_mut(&timeline) {
            tl.last_selection = Some(diag);
        }
    }

    /// Most recent [`SelectionDiagnostics`] for a timeline, or `None`
    /// if no projection has run yet (or the projection used the
    /// rule-based path).
    pub fn last_selection_of(&self, timeline: TimelineId) -> Option<&SelectionDiagnostics> {
        self.timelines
            .get(&timeline)
            .and_then(|tl| tl.last_selection.as_ref())
    }

    /// Chronologically-ordered `SummaryOfTurns` leaf turn indices for
    /// a timeline.  Walked by the projection's score-density selector
    /// (§8) to evaluate the right-edge recency anchor.  Empty when no
    /// summary tree exists yet.
    pub fn summary_leaves_chrono(&self, timeline: TimelineId) -> Vec<TurnIndex> {
        let tl = match self.timelines.get(&timeline) {
            Some(t) => t,
            None => return Vec::new(),
        };
        tl.tree_meta
            .iter()
            .filter(|(_, m)| m.kind == TurnKind::SummaryOfTurns)
            .map(|(idx, _)| *idx)
            .collect()
    }

    /// Chronologically-ordered `Normal` turn indices for a timeline.
    pub fn normal_turns_chrono(&self, timeline: TimelineId) -> Vec<TurnIndex> {
        let tl = match self.timelines.get(&timeline) {
            Some(t) => t,
            None => return Vec::new(),
        };
        tl.tree_meta
            .iter()
            .filter(|(_, m)| m.kind == TurnKind::Normal)
            .map(|(idx, _)| *idx)
            .collect()
    }

    /// Chronologically-ordered `SummaryOfSummaries` internal-node turn
    /// indices for a timeline.
    pub fn summary_internals_chrono(&self, timeline: TimelineId) -> Vec<TurnIndex> {
        let tl = match self.timelines.get(&timeline) {
            Some(t) => t,
            None => return Vec::new(),
        };
        tl.tree_meta
            .iter()
            .filter(|(_, m)| m.kind == TurnKind::SummaryOfSummaries)
            .map(|(idx, _)| *idx)
            .collect()
    }

    /// All registered timelines.
    pub fn all_timeline_ids(&self) -> impl Iterator<Item = TimelineId> + '_ {
        self.timelines.keys().copied()
    }

    // ── Per-stream runtime state (was Manifest.streams) ─────────────────

    /// Read the in-RAM runtime state for `stream_id` — chunk index +
    /// latest tokens/signatures locations + committed-through
    /// watermark + decl.
    pub fn stream_of(&self, stream_id: StreamId) -> Option<&StreamRuntime> {
        self.streams.get(&stream_id)
    }

    /// True iff the substrate has any record of `stream_id`.
    pub fn has_stream(&self, stream_id: StreamId) -> bool {
        self.streams.contains_key(&stream_id)
    }

    /// All stream ids known to the substrate.
    pub fn all_stream_ids(&self) -> impl Iterator<Item = StreamId> + '_ {
        self.streams.keys().copied()
    }

    /// Iterate `(stream_id, &StreamRuntime)` pairs.
    pub fn all_streams(&self) -> impl Iterator<Item = (StreamId, &StreamRuntime)> + '_ {
        self.streams.iter().map(|(k, v)| (*k, v))
    }

    /// Total live chunk count across all streams.  Used by the
    /// compaction dead-ratio calculation.
    pub fn live_chunk_count(&self) -> usize {
        self.streams.values().map(|s| s.chunks.len()).sum()
    }

    /// Clear every per-entity collection populated from redo-log
    /// records — used by `compact()` after the active log swap to
    /// re-walk the freshly-compacted log into substrate state with
    /// updated offsets.  Singleton-style state (timeline registrations,
    /// per-turn KV residence slots) is preserved.
    pub fn clear_walker_state(&mut self) {
        self.streams.clear();
        self.timeline_by_debug_id.clear();
        for tl in self.timelines.values_mut() {
            tl.label = None;
            tl.conv_id = None;
            tl.archived = false;
            tl.debug_id = None;
            tl.tree_meta.clear();
            tl.tree_root = None;
            tl.pending_summary_queue.clear();
            tl.dirty_summary_set.clear();
        }
    }

    /// Emit `(timeline_id, conv_id, label, archived, custom)` tuples for
    /// every timeline that holds non-default values.  Used by compaction
    /// to re-emit live `Label` / `ConvState` records.
    pub fn live_conv_meta(&self) -> Vec<(u64, String, String, bool, BTreeMap<String, String>)> {
        self.timelines
            .iter()
            .filter_map(|(tid, tl)| {
                let conv_id = tl.conv_id.clone().unwrap_or_default();
                let label = tl.label.clone().unwrap_or_default();
                if conv_id.is_empty() && label.is_empty() && !tl.archived && tl.custom.is_empty() {
                    None
                } else {
                    Some((tid.raw(), conv_id, label, tl.archived, tl.custom.clone()))
                }
            })
            .collect()
    }

    /// Emit one `TreeMetadataPayload` per live tree node across every
    /// timeline.  Used by compaction to re-emit live `TreeMetadata`
    /// records.
    pub fn live_tree_metadata_payloads(&self) -> Vec<TreeMetadataPayload> {
        let mut out = Vec::new();
        for (tid, tl) in &self.timelines {
            for (idx, meta) in &tl.tree_meta {
                let kind = match meta.kind {
                    TurnKind::Normal => 0,
                    TurnKind::SummaryOfTurns => 1,
                    TurnKind::SummaryOfSummaries => 2,
                };
                let root_now = if tl.tree_root == Some(*idx) {
                    Some(idx.0)
                } else {
                    None
                };
                out.push(TreeMetadataPayload {
                    timeline_id: tid.raw(),
                    turn_index: idx.0,
                    kind,
                    tree_height: meta.tree_height,
                    dirty: meta.dirty,
                    children: meta.children.iter().map(|c| c.0).collect(),
                    root_now,
                });
            }
        }
        out
    }

    /// Emit `(timeline_id, debug_id)` for every timeline that has one
    /// set.  Used by compaction to re-emit live `DebugId` records.
    pub fn live_debug_ids(&self) -> Vec<(u64, String)> {
        self.timelines
            .iter()
            .filter_map(|(tid, tl)| tl.debug_id.as_ref().map(|id| (tid.raw(), id.clone())))
            .collect()
    }

    /// Install a stream declaration.  Idempotent: subsequent decls for
    /// the same stream overwrite (last-writer-wins).
    pub fn apply_stream_decl(&mut self, stream_id: StreamId, decl: StreamDecl) {
        // Turn decls implicitly register their timeline.  The walker
        // pass applies records in log order; without this, a `Label`
        // record carrying the conv_id (written after `NewConversation`
        // but typically before any TurnDecl) would land before the
        // timeline exists in `self.timelines` and `set_conv_id`
        // would silently no-op.  Net effect was that restored
        // conversations vanished from the sidebar listing.
        if let StreamDecl::Turn(t) = &decl {
            if let (Some(timeline), Some(layer), Some(group)) = (
                TimelineId::from_raw(t.timeline_id),
                LayerId::from_raw(t.layer_id),
                GroupId::from_raw(t.group_id),
            ) {
                self.register_timeline(timeline, layer, group);
            }
        }
        self.streams.entry(stream_id).or_default().decl = Some(decl);
    }

    /// Record a chunk location for `stream_id` at chunk index `idx`.
    /// Last-writer-wins on `idx`.
    pub fn apply_chunk_loc(&mut self, stream_id: StreamId, idx: u64, loc: ChunkLoc) {
        self.streams
            .entry(stream_id)
            .or_default()
            .chunks
            .insert(idx, loc);
    }

    /// Record the latest `Tokens` record location for `stream_id`.
    pub fn apply_tokens_loc(&mut self, stream_id: StreamId, loc: RecordLoc) {
        self.streams.entry(stream_id).or_default().tokens = Some(loc);
    }

    /// Record the latest `Signatures` record location for `stream_id`.
    pub fn apply_signatures_loc(&mut self, stream_id: StreamId, loc: RecordLoc) {
        self.streams.entry(stream_id).or_default().signatures = Some(loc);
    }

    /// Record the highest chunk index the stream is durably committed
    /// through.  Last-writer-wins.
    pub fn apply_commit_through(&mut self, stream_id: StreamId, through_index: u64) {
        self.streams.entry(stream_id).or_default().committed_through = Some(through_index);
    }

    // ── Per-timeline LWW payload applicators (decode + dispatch) ────────

    /// Apply a decoded `ConvMeta` (Label payload).  When the
    /// timeline isn't registered yet (Label record written before
    /// the first TurnDecl — the common zend pattern), the meta is
    /// stashed in `pending_conv_meta` and drained by
    /// `register_timeline` once the matching TurnDecl arrives.
    pub fn apply_conv_meta(&mut self, timeline_raw: u64, meta: &ConvMeta) {
        let Some(timeline) = TimelineId::from_raw(timeline_raw) else {
            return;
        };
        if self.timelines.contains_key(&timeline) {
            if !meta.conv_id.is_empty() {
                self.set_conv_id(timeline, &meta.conv_id);
            }
            if !meta.label.is_empty() {
                self.set_label(timeline, &meta.label);
            }
            if !meta.custom.is_empty() {
                self.merge_custom(timeline, &meta.custom);
            }
        } else {
            // Merge into any prior pending entry so an earlier
            // partial Label (conv_id only) plus a later partial
            // (label only) both survive registration.
            let slot = self.pending_conv_meta.entry(timeline_raw).or_default();
            if !meta.conv_id.is_empty() {
                slot.conv_id = meta.conv_id.clone();
            }
            if !meta.label.is_empty() {
                slot.label = meta.label.clone();
            }
            for (k, v) in &meta.custom {
                slot.custom.insert(k.clone(), v.clone());
            }
        }
    }

    /// Apply a decoded `ConvState` payload.  Same stash-and-drain
    /// pattern as [`Self::apply_conv_meta`] for unregistered
    /// timelines.
    pub fn apply_conv_state(&mut self, timeline_raw: u64, state: ConvState) {
        let Some(timeline) = TimelineId::from_raw(timeline_raw) else {
            return;
        };
        if self.timelines.contains_key(&timeline) {
            let _ = self.set_archived(timeline, state.archived);
        } else {
            self.pending_conv_state.insert(timeline_raw, state);
        }
    }

    /// Apply a decoded `TreeMetadataPayload`.  Sets per-turn tree
    /// meta + (if `root_now` is set) the timeline's tree root.
    pub fn apply_tree_metadata_payload(&mut self, payload: &TreeMetadataPayload) {
        let Some(timeline) = TimelineId::from_raw(payload.timeline_id) else {
            return;
        };
        let kind = match payload.kind {
            0 => TurnKind::Normal,
            1 => TurnKind::SummaryOfTurns,
            2 => TurnKind::SummaryOfSummaries,
            _ => return,
        };
        let meta = TreeNodeMeta {
            kind,
            children: payload.children.iter().map(|c| TurnIndex(*c)).collect(),
            tree_height: payload.tree_height,
            dirty: payload.dirty,
        };
        self.set_tree_meta(timeline, TurnIndex(payload.turn_index), meta);
        if let Some(root) = payload.root_now {
            self.set_tree_root(timeline, Some(TurnIndex(root)));
        }
    }

    /// Apply a decoded `DebugIdPayload`.
    pub fn apply_debug_id_payload(&mut self, payload: &DebugIdPayload) {
        let Some(timeline) = TimelineId::from_raw(payload.timeline_id) else {
            return;
        };
        self.set_debug_id(timeline, payload.debug_id.clone());
    }

    /// Apply a decoded [`TombstonePayload`].  Works whether or not
    /// the timeline is currently registered — registration just
    /// observes the tombstone bit when it later drains the same set.
    pub fn apply_tombstone(&mut self, payload: &TombstonePayload) {
        let Some(timeline) = TimelineId::from_raw(payload.timeline_id) else {
            return;
        };
        self.tombstoned_timelines.insert(timeline);
    }

    /// Mark `timeline` as tombstoned in-RAM.  Callers writing the
    /// matching `Tombstone` record to the redo log invoke this to
    /// keep the live projection / resolver in sync — without it the
    /// deletion would only take effect on the next reload.
    pub fn tombstone_timeline(&mut self, timeline: TimelineId) {
        self.tombstoned_timelines.insert(timeline);
    }

    /// Whether `timeline` has been tombstoned.
    pub fn is_tombstoned(&self, timeline: TimelineId) -> bool {
        self.tombstoned_timelines.contains(&timeline)
    }

    /// Direct read of the tombstoned-timeline set.  Used by the
    /// compactor to filter dead records out of the next compacted
    /// log.
    pub fn tombstoned_timelines(&self) -> &HashSet<TimelineId> {
        &self.tombstoned_timelines
    }

    /// Apply one walked redo-log record directly into the substrate's
    /// in-RAM state.  The dispatch lives here (not on `Manifest`)
    /// because per-entity records — chunks, stream decls, labels,
    /// tree metadata, debug ids — are substrate state, not manifest
    /// state.  The manifest only sees singletons (`ModelSpec`,
    /// `Template`, `Tokenizer`, `Checkpoint`).
    ///
    /// Called from `SubstratePersistence::recover_with_substrate_sink`
    /// during startup so the walker pass populates both the manifest's
    /// singleton hints and the substrate's per-entity state in one
    /// sweep over the log.
    pub fn apply_walker_entry(&mut self, entry: &WalkEntry) {
        let h = &entry.record.header;
        let stream_id = StreamId(h.stream_id);
        match h.record_type {
            RecordType::StreamDecl => {
                if let Ok(decl) = StreamDecl::decode(&entry.record.payload) {
                    self.apply_stream_decl(stream_id, decl);
                }
            }
            RecordType::Chunk => {
                self.apply_chunk_loc(
                    stream_id,
                    h.chunk_index,
                    ChunkLoc {
                        offset: entry.offset,
                        payload_len: h.payload_len,
                        record_size: entry.size,
                        token_count: h.token_count,
                        format: h.format,
                    },
                );
            }
            RecordType::Tokens => {
                self.apply_tokens_loc(
                    stream_id,
                    RecordLoc {
                        offset: entry.offset,
                        payload_len: h.payload_len,
                        record_size: entry.size,
                    },
                );
            }
            RecordType::Signatures => {
                self.apply_signatures_loc(
                    stream_id,
                    RecordLoc {
                        offset: entry.offset,
                        payload_len: h.payload_len,
                        record_size: entry.size,
                    },
                );
            }
            RecordType::Commit => {
                self.apply_commit_through(stream_id, h.chunk_index);
            }
            RecordType::Label => {
                if let Ok((tl, meta)) = decode_label_payload(&entry.record.payload) {
                    self.apply_conv_meta(tl, &meta);
                }
            }
            RecordType::ConvState => {
                if let Ok((tl, state)) = decode_conv_state_payload(&entry.record.payload) {
                    self.apply_conv_state(tl, state);
                }
            }
            RecordType::TreeMetadata => {
                if let Ok(payload) = TreeMetadataPayload::decode(&entry.record.payload) {
                    self.apply_tree_metadata_payload(&payload);
                }
            }
            RecordType::DebugId => {
                if let Ok(payload) = DebugIdPayload::decode(&entry.record.payload) {
                    self.apply_debug_id_payload(&payload);
                }
            }
            RecordType::Tombstone => {
                if let Ok(payload) = TombstonePayload::decode(&entry.record.payload) {
                    self.apply_tombstone(&payload);
                }
            }
            // Singletons go to the manifest, not the substrate.
            RecordType::ModelSpec
            | RecordType::Template
            | RecordType::Tokenizer
            | RecordType::Checkpoint
            | RecordType::Unknown => {}
        }
    }

    /// Build an in-memory [`summary_tree::SummaryTree`] from the
    /// timeline's persisted `tree_meta`.  Used by the projection's
    /// score-density selector (§8) and the cold-load missing-child
    /// regeneration sweep.
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
                dirty: meta.dirty,
                tokens: token_count,
            };
            tree.insert_node(node);
            match meta.kind {
                TurnKind::Normal => tree.push_chrono_normal(NodeId(idx.0)),
                TurnKind::SummaryOfTurns => tree.push_chrono_leaf(NodeId(idx.0)),
                TurnKind::SummaryOfSummaries => {}
            }
        }
        tree.set_root(tl.tree_root.map(|r| NodeId(r.0)));
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
        let compression = self.timeline_compression.get(&timeline).copied();
        let residence = self.alloc_residence(turn_stream_id(timeline.raw(), idx.0), compression);
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
            tl.tree_meta.insert(idx, TreeNodeMeta::default());
            if tl.summarize {
                tl.pending_summary_queue.push_back(idx);
            }
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
        let compression = self.timeline_compression.get(&timeline).copied();
        let residence = self.alloc_residence(turn_stream_id(timeline.raw(), idx.0), compression);
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
            tl.tree_meta
                .entry(idx)
                .or_insert_with(TreeNodeMeta::default);
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
        let Some(residence) = self.turn(timeline, index).map(|e| e.content.residence) else {
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
        residence.hot.as_ref().map(|_| residence.byte_size as usize)
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
            entry.content.token_count = entry.content.token_count.saturating_add(additional_tokens);
            entry.block_range.1 = new_block_end;
        }
    }

    pub fn block_range_of(&self, timeline: TimelineId, index: TurnIndex) -> (u64, u64) {
        self.turn(timeline, index).map_or((0, 0), |e| e.block_range)
    }

    /// Set the turn's BDP sig entries — one per chunk in the
    /// content's residence, in slot block order.
    pub fn set_sig_entries(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
        entries: Vec<SigEntry>,
    ) {
        if let Some(entry) = self.turn_mut(timeline, index) {
            entry.content.sig_entries = entries;
        }
    }

    pub fn extend_sig_entries(
        &mut self,
        timeline: TimelineId,
        index: TurnIndex,
        entries: impl IntoIterator<Item = SigEntry>,
    ) {
        if let Some(entry) = self.turn_mut(timeline, index) {
            entry.content.sig_entries.extend(entries);
        }
    }

    /// Turn's BDP sig entries — one per chunk in the content's
    /// residence, slot-block ordered.
    pub fn sig_entries_of(&self, timeline: TimelineId, index: TurnIndex) -> Vec<SigEntry> {
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

    /// Merge key/value pairs into a timeline's `custom` metadata bag
    /// (last-write-wins per key). No-op if the timeline isn't registered;
    /// the caller (handle setter) stages it in `pending_conv_meta` first
    /// so pre-registration writes survive, mirroring label/conv_id.
    pub fn merge_custom(&mut self, timeline: TimelineId, kv: &BTreeMap<String, String>) {
        if kv.is_empty() {
            return;
        }
        if let Some(entry) = self.timelines.get_mut(&timeline) {
            for (k, v) in kv {
                entry.custom.insert(k.clone(), v.clone());
            }
        }
    }

    /// The `custom` metadata bag for `timeline`, or `None` if unregistered.
    pub fn custom_of(&self, timeline: TimelineId) -> Option<&BTreeMap<String, String>> {
        self.timelines.get(&timeline).map(|e| &e.custom)
    }

    /// All **live** (non-tombstoned) timelines whose `custom` metadata
    /// contains `key == value` (exact match). The content-addressed cache
    /// lookup used by utility ingests to skip re-building units already
    /// present after load. Tombstoned timelines are excluded — a dead
    /// conversation must never count as a cache hit (it no longer serves
    /// retrieval), or its file would silently vanish from the layer.
    pub fn timelines_with_metadata(&self, key: &str, value: &str) -> Vec<TimelineId> {
        self.timelines
            .iter()
            .filter(|(tid, tl)| {
                !self.tombstoned_timelines.contains(tid)
                    && tl.custom.get(key).map(|v| v.as_str()) == Some(value)
            })
            .map(|(tid, _)| *tid)
            .collect()
    }

    /// The set of distinct `custom[key]` values across all **live**
    /// (non-tombstoned) timelines. A one-pass snapshot so callers can
    /// probe membership in O(1) instead of an O(timelines) scan per probe
    /// (e.g. the code_read resume cache over thousands of files).
    pub fn metadata_values_for_key(&self, key: &str) -> std::collections::HashSet<String> {
        self.timelines
            .iter()
            .filter(|(tid, _)| !self.tombstoned_timelines.contains(tid))
            .filter_map(|(_, tl)| tl.custom.get(key).cloned())
            .collect()
    }

    /// All **live** (non-tombstoned) timelines that carry `key` in their
    /// `custom` metadata, paired with that key's value. Drives utility
    /// ingest reconciliation (e.g. tombstone every `code_read`
    /// conversation whose `path` is no longer on disk).
    pub fn timelines_with_metadata_key(&self, key: &str) -> Vec<(TimelineId, String)> {
        self.timelines
            .iter()
            .filter(|(tid, _)| !self.tombstoned_timelines.contains(tid))
            .filter_map(|(tid, tl)| tl.custom.get(key).map(|v| (*tid, v.clone())))
            .collect()
    }

    /// Whether `timeline` has been archived by the user. Untouched
    /// timelines default to `false`. Returns `false` for unknown
    /// timelines (matches "not archived" since the conversation
    /// doesn't exist as far as the sidebar is concerned).
    pub fn is_archived(&self, timeline: TimelineId) -> bool {
        self.timelines.get(&timeline).is_some_and(|e| e.archived)
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
    ///
    /// `stream_id` is the content-addressed stream id under which this
    /// section's chunks land in the redo log.  The substrate uses it
    /// for the residence so the persistence thread's section-persist
    /// pass picks the section up by snapshotting hot bytes for any
    /// section residence whose `stream_id != default && cold == None`.
    #[allow(clippy::too_many_arguments)]
    pub fn set_section_full(
        &mut self,
        section: SectionId,
        stream_id: StreamId,
        token_count: usize,
        sig_entries: Vec<SigEntry>,
        sealed_gpu: Arc<Vec<SealedSequence>>,
        migrate_to_cpu: impl FnOnce(&[SealedSequence]) -> candle::Result<Vec<SealedSequence>>,
        tokens: Arc<Vec<u32>>,
    ) -> candle::Result<()> {
        let sealed_cpu = migrate_to_cpu(&sealed_gpu)?;
        let residence = self.alloc_residence(stream_id, None);
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

    /// Section variant of [`Self::restore_turn`] — install a section
    /// recovered from the redo log directly into the hot tier without
    /// going through a fresh forward pass.
    ///
    /// `sealed_hot` is the per-layer K/V already materialised in VRAM
    /// (by the caller's `load_section_into_hot` cold-load).  `cold` is
    /// the on-disk reference list returned by `recover_section_cold_refs`;
    /// installing it pre-emptively marks the section as already
    /// persisted so the persistence thread's section-persist pass
    /// skips it.
    #[allow(clippy::too_many_arguments)]
    pub fn restore_section(
        &mut self,
        section: SectionId,
        stream_id: StreamId,
        token_count: usize,
        sig_entries: Vec<SigEntry>,
        sealed_hot: Vec<SealedSequence>,
        cold: Vec<StoredSequence>,
        tokens: Arc<Vec<u32>>,
    ) {
        let residence = self.alloc_residence(stream_id, None);
        self.residence[residence.0].cold = Some(cold);
        let entry = SectionEntryData {
            token_count,
            block_range: (0, 0),
            sig_entries,
            tokens,
            residence,
        };
        self.sections.insert(section, entry);
        if !sealed_hot.is_empty() {
            self.install_section_hot(residence, sealed_hot);
        }
    }

    /// True when `section` already has a hot-resident entry in the
    /// substrate — used by the ingest loop's skip-if-present check to
    /// avoid re-prefilling sections that recovered from the redo log.
    pub fn section_is_hot(&self, section: SectionId) -> bool {
        match self.sections.get(&section) {
            Some(entry) => self.residence[entry.residence.0].hot.is_some(),
            None => false,
        }
    }

    /// True when `section` is recorded in the substrate's section map
    /// at any tier (hot, warm, or cold-marker only).  Stricter than
    /// `section_is_hot`: a cold-marker section restored from the redo
    /// log returns true here even before any projection has elevated
    /// it to hot.  Used by the ingest loop to skip re-issuing a
    /// `RestoreSection` for a section the substrate already knows
    /// about (preventing duplicate residence allocations).
    pub fn section_exists(&self, section: SectionId) -> bool {
        self.sections.contains_key(&section)
    }

    /// Per-layer chunk count for the section's hot residence, or
    /// `None` if the section isn't hot.  Used by the ingest loop's
    /// skip path for the per-section block-count diagnostic.
    pub fn section_block_count(&self, section: SectionId) -> Option<usize> {
        let entry = self.sections.get(&section)?;
        let hot = self.residence[entry.residence.0].hot.as_ref()?;
        hot.first().map(|s| s.chunks.len())
    }

    /// Look up a section by its persistence `debug_name` (the symbolic
    /// id passed to `insert_section` at ingest time).  Walks the
    /// persistence manifest's `SectionDecl` records to resolve the
    /// content-addressed `StreamId`, then walks the in-RAM `sections`
    /// map to find which `SectionId` got that stream installed.  Used
    /// by calibration consumers that want to pick out individual
    /// scenarios by their human-readable identifier after loading the
    /// workspace's full substrate.
    ///
    /// Linear in section count; the calibration scenario list is
    /// O(thousands) at most so this is fine.  Returns the first match
    /// when duplicate `debug_name`s exist (callers should keep names
    /// unique per workspace).
    pub fn section_id_for_debug_name(&self, debug_name: &str) -> Option<SectionId> {
        // Find the stream id matching this debug_name from the
        // substrate's in-RAM stream index (the authoritative source
        // since Phase 3 — the manifest no longer holds per-entity
        // state).
        let stream_id =
            self.all_streams()
                .find_map(|(sid, entry)| match entry.decl.as_ref()? {
                    StreamDecl::PromptSection(s) if s.debug_name == debug_name => Some(sid),
                    _ => None,
                })?;
        // Map stream_id back to the in-RAM SectionId via the residence
        // slab — the residence holds the stream id we installed at
        // ingest time.
        self.sections.iter().find_map(|(sid, entry)| {
            if self.residence[entry.residence.0].stream_id == stream_id {
                Some(*sid)
            } else {
                None
            }
        })
    }

    /// Replace a section residence's hot bytes with `new_hot`.  Used
    /// by the persistence thread's section-persist pass after it
    /// quantizes the section's K/V to the section-compression policy
    /// (C0 by default) — the new Q-format sequences land here so
    /// subsequent in-session reads see the same bytes the cold tier
    /// will reproduce on the next reload.
    pub fn replace_section_hot(&mut self, residence: ResidenceIndex, new_hot: Vec<SealedSequence>) {
        debug_assert!(
            !new_hot.is_empty(),
            "replace_section_hot called with empty Vec"
        );
        let bytes = sealed_bytes(&new_hot) as u64;
        let slot = &mut self.residence[residence.0];
        slot.byte_size = bytes;
        slot.hot = Some(new_hot);
    }

    /// Snapshot the section residences that have hot bytes installed
    /// but haven't been written to the cold tier yet — the work list
    /// for the persistence thread's section-persist pass.  Mirrors
    /// [`Self::snapshot_pending_cold`] but walks the `sections` map
    /// rather than `warm_lru` (sections are pinned and never appear on
    /// the LRUs).  Skips sentinel-stream residences (sections still
    /// using the legacy `StreamId::default()` path won't try to
    /// persist).
    pub fn snapshot_pending_section_cold(
        &self,
    ) -> Vec<(ResidenceIndex, StreamId, Vec<SealedSequence>)> {
        self.sections
            .values()
            .filter_map(|entry| {
                let slot = &self.residence[entry.residence.0];
                if slot.cold.is_some()
                    || slot.stream_id == StreamId::default()
                    || slot.pending_quantize
                {
                    return None;
                }
                slot.hot
                    .as_ref()
                    .map(|hot| (entry.residence, slot.stream_id, hot.clone()))
            })
            .collect()
    }

    /// Mark a section residence as awaiting the scheduler's quantize
    /// drain.  Called from `SealAction::Section` right after
    /// `set_section_full` installs the native bytes, when a
    /// `compression_policy` is configured.  While this flag is set,
    /// `snapshot_pending_section_cold` ignores the residence — the
    /// persistence thread holds off on writing native bytes that the
    /// scheduler is about to replace with their quantized form.
    pub fn mark_section_pending_quantize(&mut self, residence: ResidenceIndex) {
        self.residence[residence.0].pending_quantize = true;
    }

    /// Clear the pending-quantize flag — called from the same drain
    /// that calls `replace_section_hot`.  After this the persistence
    /// thread is free to gather the residence's (now final) hot bytes
    /// and persist them.
    pub fn clear_section_pending_quantize(&mut self, residence: ResidenceIndex) {
        self.residence[residence.0].pending_quantize = false;
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

    pub fn section_sig_entries(&self, section: SectionId) -> &[SigEntry] {
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
        let Some(timeline) = self.active_timelines_for_group(group).next() else {
            return 0;
        };
        Substrate::turn_count(self, timeline)
    }

    fn turn_token_count(&self, group: GroupId, index: TurnIndex) -> usize {
        let Some(timeline) = self.active_timelines_for_group(group).next() else {
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
        let timeline = self.active_timelines_for_group(group).next()?;
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
fn combine_per_depth(s: PerDepthScores, formula: ScoreFormula, weights: &DepthWeights) -> f32 {
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
        let Some(timeline) = self.substrate.active_timelines_for_group(group).next() else {
            return 0;
        };
        Substrate::turn_count(self.substrate, timeline)
    }

    fn turn_token_count(&self, group: GroupId, index: TurnIndex) -> usize {
        let Some(timeline) = self.substrate.active_timelines_for_group(group).next() else {
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
        let Some(timeline) = self.substrate.active_timelines_for_group(group).next() else {
            return 0.0;
        };
        if self.substrate.turn(timeline, index).is_none() {
            return 0.0;
        }
        combine_per_depth(self.scores.turn(timeline, index), formula, weights)
    }

    fn turn_origin(&self, group: GroupId, _index: TurnIndex) -> Option<LayerId> {
        let timeline = self.substrate.active_timelines_for_group(group).next()?;
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

    fn summary_tree_select(
        &self,
        timeline: TimelineId,
        budget: u32,
        formula: ScoreFormula,
        weights: &DepthWeights,
    ) -> Option<Vec<(TurnIndex, SelectionOrigin, f32)>> {
        // No tree yet → fall through to the rule-based path.
        let root = self.substrate.tree_root_of(timeline)?;
        // Reachability guard: if the recorded root isn't actually
        // present, treat as "no tree" rather than crash.
        self.substrate.tree_meta_of(timeline, root)?;
        let tree = self.substrate.build_summary_tree_in_memory(timeline);
        if tree.is_empty() {
            return None;
        }
        let mut scores: ahash::AHashMap<NodeId, f32> = ahash::AHashMap::default();
        for id in tree.all_ids() {
            let idx = TurnIndex(id.0);
            // A tree node without a backing substrate turn is an
            // orphan — possible if the redo log holds TreeMetadata
            // records whose matching TurnDecl never landed (e.g. an
            // older session ran before summariser persistence was
            // wired up).  These can't be elevated, so they must be
            // excluded from the selection that flows into the
            // projection / elevate path.  Leaving them in `scores`
            // with a default 0.0 wouldn't help — `select_dense`
            // walks the tree shape and would still return them.
            if self.substrate.turn(timeline, idx).is_none() {
                continue;
            }
            let s = combine_per_depth(self.scores.turn(timeline, idx), formula, weights);
            scores.insert(id, s);
        }
        let cfg = RecencyConfig::default();
        let sel = select_dense(&tree, &scores, cfg, budget);
        // Convert (NodeId, SelectionOrigin) pairs back to TurnIndex,
        // dropping any picks whose tree node has no backing
        // substrate turn.  The substrate-turn check at scoring time
        // (above) only guards scoring — `select_dense` walks the
        // tree shape and can still return orphan NodeIds; this
        // post-filter is what actually keeps them out of the
        // elevate plan.
        let out: Vec<_> = sel
            .selected
            .iter()
            .zip(sel.origins.iter())
            .filter_map(|(id, origin)| {
                let idx = TurnIndex(id.0);
                self.substrate.turn(timeline, idx)?;
                let eff = sel.effective_scores.get(id).copied().unwrap_or(0.0);
                Some((idx, *origin, eff))
            })
            .collect();
        Some(out)
    }

    fn pending_summary_len(&self, timeline: TimelineId) -> usize {
        self.substrate.pending_summary_len(timeline)
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
        let Some(timeline) = self.guard.active_timelines_for_group(group).next() else {
            return 0;
        };
        Substrate::turn_count(&self.guard, timeline)
    }

    fn turn_token_count(&self, group: GroupId, index: TurnIndex) -> usize {
        let Some(timeline) = self.guard.active_timelines_for_group(group).next() else {
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
        let Some(timeline) = self.guard.active_timelines_for_group(group).next() else {
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
        let timeline = self.guard.active_timelines_for_group(group).next()?;
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

    fn pending_summary_len(&self, timeline: TimelineId) -> usize {
        self.guard.pending_summary_len(timeline)
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
    use crate::projection::{GroupId, LayerId, SectionId, TimelineAllocator, TimelineId};
    use crate::token_buffer::TokenBuffer;

    fn make_timeline() -> (LayerId, GroupId, TimelineId, Substrate) {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let timeline = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(timeline, layer, group);
        (layer, group, timeline, sub)
    }

    // ── Phase 1: TreeNodeMeta + debug_id substrate APIs ──────────────────

    #[test]
    fn appended_turn_gets_default_tree_meta_as_normal() {
        let (_, _, timeline, mut sub) = make_timeline();
        let idx = sub.append_with_blocks(timeline, 10, 0, 1);
        let meta = sub.tree_meta_of(timeline, idx).expect("meta present");
        assert_eq!(meta.kind, TurnKind::Normal);
        assert!(meta.children.is_empty());
        assert_eq!(meta.tree_height, 0);
        assert!(!meta.dirty);
    }

    #[test]
    fn appended_turn_pushed_onto_pending_summary_queue() {
        let (_, _, timeline, mut sub) = make_timeline();
        let idx_a = sub.append_with_blocks(timeline, 10, 0, 1);
        let idx_b = sub.append_with_blocks(timeline, 10, 1, 2);
        assert_eq!(sub.pending_summary_len(timeline), 2);
        assert_eq!(sub.pop_pending_summary(timeline), Some(idx_a));
        assert_eq!(sub.pop_pending_summary(timeline), Some(idx_b));
        assert_eq!(sub.pop_pending_summary(timeline), None);
    }

    #[test]
    fn set_tree_meta_overwrites_in_place() {
        let (_, _, timeline, mut sub) = make_timeline();
        let idx = sub.append_with_blocks(timeline, 10, 0, 1);
        let meta = TreeNodeMeta {
            kind: TurnKind::SummaryOfTurns,
            children: vec![TurnIndex(99)],
            tree_height: 1,
            dirty: false,
        };
        sub.set_tree_meta(timeline, idx, meta.clone());
        assert_eq!(sub.tree_meta_of(timeline, idx), Some(&meta));
        let leaves = sub.summary_leaves_chrono(timeline);
        assert_eq!(leaves, vec![idx]);
    }

    #[test]
    fn mark_summary_dirty_indexes_into_dirty_set() {
        let (_, _, timeline, mut sub) = make_timeline();
        let idx = sub.append_with_blocks(timeline, 10, 0, 1);
        // Marking a Normal turn is a no-op.
        sub.mark_summary_dirty(timeline, idx);
        assert_eq!(sub.dirty_summary_len(timeline), 0);
        // Convert to summary then mark dirty.
        sub.set_tree_meta(
            timeline,
            idx,
            TreeNodeMeta {
                kind: TurnKind::SummaryOfTurns,
                ..Default::default()
            },
        );
        sub.mark_summary_dirty(timeline, idx);
        assert_eq!(sub.dirty_summary_len(timeline), 1);
        assert_eq!(sub.pop_oldest_dirty(timeline), Some(idx));
        assert_eq!(sub.dirty_summary_len(timeline), 0);
        // After pop the meta's `dirty` flag is still true — the caller
        // (the summariser) clears it explicitly via `clear_summary_dirty`
        // once the regeneration probe lands.  This decoupling lets the
        // sweep batch the regeneration without losing the dirty bit on
        // a crash mid-sweep.
        assert!(sub.tree_meta_of(timeline, idx).unwrap().dirty);
        sub.clear_summary_dirty(timeline, idx);
        assert!(!sub.tree_meta_of(timeline, idx).unwrap().dirty);
    }

    #[test]
    fn debug_id_round_trip_and_lookup() {
        let (_, _, timeline, mut sub) = make_timeline();
        assert_eq!(sub.lookup_by_debug_id("foo"), None);
        sub.set_debug_id(timeline, "coherent-50");
        assert_eq!(sub.lookup_by_debug_id("coherent-50"), Some(timeline));
        assert_eq!(sub.debug_id_of(timeline), Some("coherent-50"));
        // Replacing supersedes — old key disappears.
        sub.set_debug_id(timeline, "two-topics-100");
        assert_eq!(sub.lookup_by_debug_id("coherent-50"), None);
        assert_eq!(sub.lookup_by_debug_id("two-topics-100"), Some(timeline));
    }

    #[test]
    fn tree_root_set_and_get() {
        let (_, _, timeline, mut sub) = make_timeline();
        assert_eq!(sub.tree_root_of(timeline), None);
        let idx = sub.append_with_blocks(timeline, 10, 0, 1);
        sub.set_tree_root(timeline, Some(idx));
        assert_eq!(sub.tree_root_of(timeline), Some(idx));
        sub.set_tree_root(timeline, None);
        assert_eq!(sub.tree_root_of(timeline), None);
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
            StreamId::default(),
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
    fn install_warm_only(sub: &mut Substrate, timeline: TimelineId, bytes: u64) -> ResidenceIndex {
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
        let r =
            sub.purge_warm_to_target(1_000_000, 32 * 1024 * 1024 * 1024, 64 * 1024 * 1024 * 1024);
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
            1024 * 1024 * 1024,      // 1 GiB available
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
        let r = sub.purge_warm_to_target(0, 14 * 1000 * 1000 * 1000, 256 * 1000 * 1000 * 1000);
        assert_eq!(r.count, 0, "14 GB available > 5% × 256 GB threshold");

        // Now 13 GB available, threshold still 12.8 GB. projected =
        // 13 GB > 12.8 GB → still no purge.
        let r = sub.purge_warm_to_target(0, 13 * 1000 * 1000 * 1000, 256 * 1000 * 1000 * 1000);
        assert_eq!(r.count, 0);

        // 12 GB available → projected < threshold → purge fires.
        let r = sub.purge_warm_to_target(0, 12 * 1000 * 1000 * 1000, 256 * 1000 * 1000 * 1000);
        assert_eq!(r.count, 1);
        assert_eq!(r.bytes, 2_000_000_000);
    }

    /// Helper: install a turn that is **both** hot (on `hot_lru`) and warm,
    /// with a known `byte_size`. A non-empty sealed payload is required so
    /// `append_complete` crosses `install_hot`'s `!is_empty()` gate (an empty
    /// vec is treated as a cold-marker → `hot = None`). Returns
    /// `(turn_index, residence)`.
    fn install_hot_and_warm(
        sub: &mut Substrate,
        timeline: TimelineId,
        bytes: u64,
    ) -> (TurnIndex, ResidenceIndex) {
        let idx = sub
            .append_complete(
                timeline,
                TurnPartWrite {
                    sealed_gpu: Some(Arc::new(vec![minimal_sealed_layer()])),
                    ..Default::default()
                },
                identity_migrate,
            )
            .unwrap();
        let residence = sub.turn_residence(timeline, idx).unwrap();
        sub.residence[residence.0].byte_size = bytes;
        sub.install_warm(residence, vec![minimal_sealed_layer()]);
        (idx, residence)
    }

    /// Budget-aware eviction frees the **least-recently-promoted** turns
    /// first (back of `hot_lru`) and stops as soon as the target is covered,
    /// leaving the newest turn hot and every evicted turn's **warm copy
    /// intact** (eviction is hot→warm, never hot→cold).
    #[test]
    fn evict_hot_to_free_oldest_first_keeps_warm() {
        let (_, _, timeline, mut sub) = make_timeline();
        // install_hot pushes to the FRONT, so install order a,b,c leaves
        // hot_lru = [c, b, a] (front newest). Eviction walks from the back:
        // a first, then b.
        let (_, a) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, b) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, c) = install_hot_and_warm(&mut sub, timeline, 100_000_000);

        // Target 150 MB → evict a (100 MB) then b (200 MB ≥ 150 MB) and stop.
        let report = sub.evict_hot_to_free(&[], &[], 150_000_000);
        assert_eq!(report.count, 2, "a + b cover the 150 MB target");
        assert_eq!(report.bytes, 200_000_000);
        // Evicted turns: hot dropped, warm KEPT (the fast-reload backup).
        assert!(sub.residence[a.0].hot.is_none(), "a hot evicted");
        assert!(
            sub.residence[a.0].warm.is_some(),
            "a warm KEPT (→warm not →cold)"
        );
        assert!(sub.residence[b.0].hot.is_none(), "b hot evicted");
        assert!(sub.residence[b.0].warm.is_some(), "b warm KEPT");
        // Newest turn untouched — the working set stays resident.
        assert!(sub.residence[c.0].hot.is_some(), "c (newest) still hot");
    }

    /// The selection (keep set) is never evicted, even when it's the oldest
    /// turn — eviction skips it and frees a younger one instead.
    #[test]
    fn evict_hot_to_free_protects_keep_set() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (a_idx, a) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, b) = install_hot_and_warm(&mut sub, timeline, 100_000_000);

        // a is oldest (would be evicted first) but it's in the keep set, so
        // eviction must skip it and take b instead.
        let keep = [TurnKey {
            timeline,
            index: a_idx,
        }];
        let report = sub.evict_hot_to_free(&[], &keep, 100_000_000);
        assert_eq!(report.count, 1);
        assert!(sub.residence[a.0].hot.is_some(), "a protected by keep set");
        assert!(sub.residence[b.0].hot.is_none(), "b evicted instead");
    }

    /// A zero target evicts nothing — there's no incoming load to make room
    /// for, so the working set is left fully hot.
    #[test]
    fn evict_hot_to_free_zero_target_is_noop() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_, a) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let report = sub.evict_hot_to_free(&[], &[], 0);
        assert_eq!(report.count, 0);
        assert_eq!(report.bytes, 0);
        assert!(sub.residence[a.0].hot.is_some());
    }

    /// On reload, `TreeMetadata` records are replayed (summary-node kinds set)
    /// during the walker open pass, BEFORE the per-turn `restore_turn` loop.
    /// `restore_turn` must not clobber a reloaded summary node's kind back to
    /// `Normal` — that collapse re-enqueued the whole history, re-summarised
    /// prior summaries into garbage, and broke the tree's AVL invariant on
    /// restart.
    #[test]
    fn restore_turn_preserves_reloaded_tree_meta() {
        let (_, _, timeline, mut sub) = make_timeline();

        // Stand in for the open-pass replay: index 0 is a SummaryOfTurns leaf.
        let leaf = TurnIndex(0);
        sub.set_tree_meta(
            timeline,
            leaf,
            TreeNodeMeta {
                kind: TurnKind::SummaryOfTurns,
                children: vec![TurnIndex(7)],
                tree_height: 1,
                dirty: false,
            },
        );

        // The reconstruct loop then restores turn 0.
        let idx = sub.restore_turn(
            timeline,
            String::new(),
            String::new(),
            TokenBuffer::default(),
            20,
            None,
            0,
            0,
        );
        assert_eq!(idx, leaf, "first restored turn lands at index 0");
        let meta = sub.tree_meta_of(timeline, leaf).expect("tree meta present");
        assert_eq!(
            meta.kind,
            TurnKind::SummaryOfTurns,
            "restore_turn must preserve the reloaded summary-node kind",
        );
        assert_eq!(meta.children, vec![TurnIndex(7)], "children preserved");

        // A turn with no prior tree meta still defaults to Normal.
        let idx2 = sub.restore_turn(
            timeline,
            String::new(),
            String::new(),
            TokenBuffer::default(),
            10,
            None,
            0,
            0,
        );
        assert_eq!(
            sub.tree_meta_of(timeline, idx2).map(|m| m.kind),
            Some(TurnKind::Normal),
            "a freshly restored turn with no tree meta defaults to Normal",
        );
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
        assert!(
            sub.residence[residence.0].hot.is_none(),
            "hot empty (cold-marker)"
        );
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

    /// Regression: zend's redo log writes the `Label` record carrying
    /// `conv_id` immediately on conversation creation, before any
    /// TurnDecl exists.  On reopen, the walker would replay the
    /// Label first and the TurnDecl second.  If `apply_conv_meta`
    /// dropped the meta when the timeline wasn't registered yet,
    /// every restored conversation would vanish from the sidebar
    /// because `conv_id` never reached the TimelineEntry.
    #[test]
    fn conv_meta_before_timeline_registration_survives() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let timeline = alloc.next();
        let mut sub = Substrate::new();

        let meta = super::ConvMeta {
            conv_id: "1780659918260".to_string(),
            label: String::new(),
            custom: Default::default(),
        };
        sub.apply_conv_meta(timeline.raw(), &meta);
        sub.apply_conv_state(timeline.raw(), super::ConvState { archived: false });
        assert!(
            sub.known_conversations().is_empty(),
            "pre-registration: meta stashed, not visible yet"
        );

        sub.register_timeline(timeline, layer, group);

        let convs = sub.known_conversations();
        assert_eq!(convs.len(), 1, "post-registration: drained meta visible");
        assert_eq!(convs[0].1, "1780659918260");
    }

    /// Partial Label payloads merge in the stash — an earlier
    /// conv_id-only Label and a later label-only Label must both
    /// land on the TimelineEntry when registration finally drains
    /// the pending state.
    #[test]
    fn pending_conv_meta_merges_partial_updates() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let timeline = alloc.next();
        let mut sub = Substrate::new();

        sub.apply_conv_meta(
            timeline.raw(),
            &super::ConvMeta {
                conv_id: "abc".into(),
                label: String::new(),
                custom: Default::default(),
            },
        );
        sub.apply_conv_meta(
            timeline.raw(),
            &super::ConvMeta {
                conv_id: String::new(),
                label: "tour".into(),
                custom: Default::default(),
            },
        );
        sub.register_timeline(timeline, layer, group);

        let convs = sub.known_conversations();
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0].1, "abc");
        assert_eq!(convs[0].2, "tour");
    }

    // ── Custom metadata (content-addressed cache) ──────────────────────

    #[test]
    fn custom_metadata_merges_and_is_searchable() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let tl = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(tl, layer, group);

        let mut kv = std::collections::BTreeMap::new();
        kv.insert("kind".to_string(), "code_read".to_string());
        kv.insert("path".to_string(), "src/lib.rs".to_string());
        kv.insert("content_sha256".to_string(), "abc123".to_string());
        sub.merge_custom(tl, &kv);

        let got = sub
            .custom_of(tl)
            .expect("registered timeline has custom map");
        assert_eq!(got.get("kind").map(String::as_str), Some("code_read"));
        assert_eq!(
            got.get("content_sha256").map(String::as_str),
            Some("abc123")
        );

        // Exact (key, value) search — the resume-cache + invalidation lookup.
        assert_eq!(
            sub.timelines_with_metadata("content_sha256", "abc123"),
            vec![tl]
        );
        assert_eq!(sub.timelines_with_metadata("path", "src/lib.rs"), vec![tl]);
        assert!(sub
            .timelines_with_metadata("content_sha256", "nope")
            .is_empty());
        assert!(sub.timelines_with_metadata("absent_key", "x").is_empty());

        // live_conv_meta carries custom so compaction can re-emit it.
        let live = sub.live_conv_meta();
        let entry = live
            .iter()
            .find(|e| e.0 == tl.raw())
            .expect("timeline in live meta");
        assert_eq!(entry.4.get("path").map(String::as_str), Some("src/lib.rs"));
    }

    #[test]
    fn custom_metadata_survives_pre_registration_and_merges() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let tl = alloc.next();
        let mut sub = Substrate::new();

        // Two partial Label payloads (different custom keys) arrive before
        // the timeline registers — both must survive and merge on drain.
        let mut m1 = std::collections::BTreeMap::new();
        m1.insert("path".to_string(), "src/a.rs".to_string());
        sub.apply_conv_meta(
            tl.raw(),
            &super::ConvMeta {
                conv_id: String::new(),
                label: String::new(),
                custom: m1,
            },
        );
        let mut m2 = std::collections::BTreeMap::new();
        m2.insert("content_sha256".to_string(), "h1".to_string());
        sub.apply_conv_meta(
            tl.raw(),
            &super::ConvMeta {
                conv_id: String::new(),
                label: String::new(),
                custom: m2,
            },
        );

        assert!(
            sub.custom_of(tl).is_none(),
            "pre-registration: not visible yet"
        );
        sub.register_timeline(tl, layer, group);

        let got = sub.custom_of(tl).expect("custom drained on registration");
        assert_eq!(got.get("path").map(String::as_str), Some("src/a.rs"));
        assert_eq!(got.get("content_sha256").map(String::as_str), Some("h1"));
        assert_eq!(
            sub.timelines_with_metadata("content_sha256", "h1"),
            vec![tl]
        );
    }

    #[test]
    fn timelines_with_metadata_matches_only_exact_pairs() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let a = alloc.next();
        let b = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(a, layer, group);
        sub.register_timeline(b, layer, group);
        let mut ma = std::collections::BTreeMap::new();
        ma.insert("path".to_string(), "x.rs".to_string());
        let mut mb = std::collections::BTreeMap::new();
        mb.insert("path".to_string(), "y.rs".to_string());
        sub.merge_custom(a, &ma);
        sub.merge_custom(b, &mb);

        assert_eq!(sub.timelines_with_metadata("path", "x.rs"), vec![a]);
        assert_eq!(sub.timelines_with_metadata("path", "y.rs"), vec![b]);
        assert!(sub.timelines_with_metadata("path", "z.rs").is_empty());
    }

    // ── Tombstone ─────────────────────────────────────────────────────

    #[test]
    fn tombstone_hides_timeline_from_active_iterator() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let alive = alloc.next();
        let dead = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(alive, layer, group);
        sub.register_timeline(dead, layer, group);

        // Before tombstone: both surface.
        let pre: Vec<TimelineId> = sub.active_timelines_for_group(group).collect();
        assert!(pre.contains(&alive));
        assert!(pre.contains(&dead));

        sub.tombstone_timeline(dead);
        assert!(sub.is_tombstoned(dead));
        assert!(!sub.is_tombstoned(alive));

        // After tombstone: only the alive one.
        let post: Vec<TimelineId> = sub.active_timelines_for_group(group).collect();
        assert_eq!(post, vec![alive]);
    }

    #[test]
    fn tombstone_for_unregistered_timeline_survives_registration() {
        // Walker replay order: a `Tombstone` record can precede the
        // matching `StreamDecl::Turn` (the redo log is append-only
        // and a refresh may write the tombstone before the new
        // generation's turn decls).  The tombstone bit must persist
        // through the eventual registration so the just-registered
        // timeline is observably dead from the start.
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let timeline = alloc.next();
        let mut sub = Substrate::new();

        sub.apply_tombstone(&super::TombstonePayload {
            timeline_id: timeline.raw(),
        });
        assert!(sub.is_tombstoned(timeline));

        sub.register_timeline(timeline, layer, group);
        assert!(sub.is_tombstoned(timeline));
        assert!(sub.active_timelines_for_group(group).all(|t| t != timeline));
    }

    #[test]
    fn tombstoned_timelines_set_includes_unregistered_entries() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let registered = alloc.next();
        let unregistered = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(registered, layer, group);
        sub.tombstone_timeline(registered);
        sub.apply_tombstone(&super::TombstonePayload {
            timeline_id: unregistered.raw(),
        });

        let set = sub.tombstoned_timelines();
        assert!(set.contains(&registered));
        assert!(set.contains(&unregistered));
        assert_eq!(set.len(), 2);
    }

    /// Regression: `snapshot_promotion_state` used to misclassify
    /// every tier-less turn (a turn whose residence has no
    /// hot/warm/cold installed — the design-intended state for
    /// ghost summary turns the substrate-summariser appends via
    /// `append_with_blocks(0..0)`) as `missing`, which used the
    /// same bucket as "substrate has no record of this turn at all."
    /// `elevate_to_hot` then logged a WARN per item, flooding the
    /// production daemon's trace with apparent corruption warnings
    /// every time a parallel-ingest worker submitted a turn whose
    /// projection plan included any of those tree-meta-only nodes.
    ///
    /// The fix splits the two cases — `missing` keeps the "not in
    /// substrate" semantic; `tier_less` is the new bucket for
    /// "in substrate but nothing to elevate."  Both get skipped by
    /// the elevation orchestrator, but only `missing` triggers the
    /// WARN.
    #[test]
    fn snapshot_promotion_state_classifies_tier_less_turns_separately_from_missing() {
        let (_layer, _group, timeline, mut sub) = make_timeline();
        // 4 turns with `block_range = 0..*` but no sealed K/V.
        // `append_with_blocks` allocates a residence slot with
        // hot/warm/cold all None — exactly what the summariser does
        // for ghost summary turns.
        for i in 0..4 {
            sub.append_with_blocks(timeline, 10, i, i + 1);
        }

        let stored_indices: Vec<TurnIndex> = sub.turn_indices(timeline).collect();
        assert_eq!(
            stored_indices,
            vec![TurnIndex(0), TurnIndex(1), TurnIndex(2), TurnIndex(3)],
        );

        let turn_keys: Vec<TurnKey> = stored_indices
            .iter()
            .map(|idx| TurnKey::new(timeline, *idx))
            .collect();
        let plan = sub.snapshot_promotion_state(&[], &turn_keys);

        // Every appended turn is tracked but tier-less — they
        // belong in `tier_less`, NOT `missing`.
        assert!(
            plan.missing.is_empty(),
            "no turn should be reported missing — they're all in the substrate; \
             missing={:?}",
            plan.missing,
        );
        assert_eq!(
            plan.tier_less.len(),
            4,
            "every tier-less ghost turn lands in `tier_less`; tier_less={:?}",
            plan.tier_less,
        );
        assert_eq!(plan.already_hot.len(), 0);
        assert_eq!(plan.warm_to_hot.len(), 0);
        assert_eq!(plan.cold_to_hot.len(), 0);
    }

    /// Counterpart to the regression test: a TurnKey naming an
    /// index the substrate genuinely has no record of must still
    /// land in `missing` (not `tier_less`), so the WARN in
    /// `elevate_to_hot` still fires for true corruption / stale
    /// projection plans.
    #[test]
    fn snapshot_promotion_state_marks_truly_unknown_turns_as_missing() {
        let (_layer, _group, timeline, mut sub) = make_timeline();
        sub.append_with_blocks(timeline, 10, 0, 1);

        let turn_keys = vec![
            TurnKey::new(timeline, TurnIndex(0)),  // exists, tier-less
            TurnKey::new(timeline, TurnIndex(99)), // does not exist
        ];
        let plan = sub.snapshot_promotion_state(&[], &turn_keys);

        assert_eq!(plan.tier_less.len(), 1);
        assert_eq!(plan.missing.len(), 1);
        match &plan.missing[0] {
            PromotionItemKind::Turn(k) => {
                assert_eq!(k.index, TurnIndex(99));
                assert_eq!(k.timeline, timeline);
            }
            other => panic!("expected Turn variant, got {other:?}"),
        }
    }
}
