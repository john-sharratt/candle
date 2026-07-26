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
//! Per-turn / per-section relevance scores are the wide-Q belief scores the
//! scheduler records for a projection (see
//! [`ContentResolver::turn_score`] / [`ContentResolver::section_score`], which
//! return a plain `f32`). Scores default to zero until the reprojection's belief
//! scan populates them.

use std::sync::{Mutex, OnceLock, RwLockReadGuard, RwLockWriteGuard};

use ahash::AHashMap;
use candle_nn::kv_cache::{QuantFormat, SealedSequence};
use std::collections::{BTreeMap, HashMap, HashSet, LinkedList};
use std::sync::Arc;

use crate::conversation::window_sealed_tokens;
use crate::persistence::content_hash::turn_stream_id;
use crate::persistence::manifest::{
    decode_conv_state_payload, decode_label_payload, ChunkLoc, ConvMeta, ConvState, RecordLoc,
};
use crate::persistence::record::{
    DebugIdPayload, DistillMode, DistillPayload, RecordType, TombstonePayload, TreeMetadataPayload,
    TurnCouplingPayload,
};
use crate::persistence::streams::{StreamDecl, StreamId};
use crate::persistence::walker::WalkEntry;
use crate::projection::{
    decode_events, GroupId, LayerId, ProjectionTarget, SectionId, TimelineAllocator, TimelineId,
    TurnIndex, TurnKey,
};
use crate::provenance::{decode_wide_sigs, WideQSig};
use crate::summary_tree::exchange::Couplings;
use crate::summary_tree::{
    select_dense, Node, NodeId, RecencyConfig, SelectionDiagnostics, SelectionOrigin, SummaryTree,
    TurnKind, MERGE_FANOUT,
};
use crate::token_buffer::TokenBuffer;
use crate::turn_layout::TurnLayout;

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
/// Per-stream memoized decode of wide-Q signature blobs. The belief scan would
/// otherwise `decode_wide_sigs` the whole (static) gallery on every reprojection
/// — tens of ms of repeated work on a corpus that doesn't change between
/// reprojections. Invalidation is **incremental**: a blob write evicts only that
/// stream's entry (see [`Substrate::set_wide_q_sigs_blob`]), so one turn seal
/// doesn't churn the whole gallery. `None` value = decoded to nothing (absent or
/// empty window), cached so repeated misses don't re-parse.
type SigCache = HashMap<StreamId, Option<Arc<Vec<WideQSig>>>>;

/// Per-stream memo of a turn's self-referencing sub-window seam offsets (sorted,
/// deduped). See [`Substrate::decoded_seams`].
type SeamCache = HashMap<StreamId, Arc<Vec<usize>>>;

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
    /// Monotonic counter stamped onto a [`TimelineEntry::order`] the first time
    /// its `conv_id` is set. Because the redo log replays in append order and a
    /// conversation's `conv_id` is written once at creation, ordering the
    /// sidebar by `order` reproduces creation order across recovery and live
    /// sessions — the conversation ids themselves are random u64s and carry no
    /// time information.
    conv_order_counter: u64,

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

    /// Running per-timeline sum of turn token counts — the O(1) corpus-size
    /// counter behind `total_token_count`. Maintained on turn append/extend
    /// (turns are append-only, so this only grows until `reset`), keeping the
    /// "materialized / N tokens" denominator off the O(corpus) hot path.
    timeline_token_totals: HashMap<TimelineId, usize>,
    /// Running global sum of ingested section token counts (the shared
    /// workspace corpus). Maintained on section install (overwrite-aware).
    section_token_total: usize,

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
    /// committed-through watermark sits on disk.
    /// Built by replaying the redo log on startup (and updated on
    /// every fresh append).  Cold-load and seal-time persistence read
    /// this directly; the manifest holds only the workspace
    /// singletons.  The per-stream `BTreeMap` only ever sits in
    /// memory — reload rebuilds it from record headers.
    streams: HashMap<StreamId, StreamRuntime>,

    /// Interior-mutable per-stream memo of decoded wide-Q windows for the belief
    /// scan — filled lazily under a read lock, so a session's reprojections
    /// decode the static gallery once instead of every scan, and invalidated
    /// per-stream on a blob write so one turn seal doesn't churn the whole
    /// gallery. See [`Self::decoded_wide_sig`].
    sig_cache: Mutex<SigCache>,

    /// Interior-mutable per-stream memo of a turn's self-referencing sub-window
    /// seam offsets, decoded from the (JSON) projection-events blob once per
    /// session instead of re-parsing it on every belief scan. Invalidated
    /// per-stream on an events-blob write. See [`Self::decoded_seams`].
    seam_cache: Mutex<SeamCache>,

    /// Reverse index: stable resume keys (`debug_id`) → `TimelineId`.
    /// Populated by [`Self::set_debug_id`] and the cold-load reader.
    /// Provides O(1) lookup for the test-harness `find_or_create`
    /// pattern (§10.4 of `docs/archived/infinite_conversations.md`).
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
    /// Timelines marked for distillation, each with the [`DistillMode`] degree
    /// its turns shed to at compaction. Same replay-order-independence as
    /// tombstones.
    distilled_timelines: HashMap<TimelineId, DistillMode>,
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
    /// Skip the hot→warm quantize pass entirely for this conversation's turns,
    /// persisting their K/V in the native R16/F16 form (no adaptive
    /// compression). Used to capture lossless tool-call exemplars for the
    /// provenance work. Overrides every other field here when set.
    pub lossless: bool,
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
    /// When `true`, the persistence thread fully offloads this residence as its
    /// KV becomes durable — freeing `hot` (VRAM) the moment a warm/cold copy
    /// exists and `warm` (RAM) the moment the cold copy lands (`install_cold`),
    /// leaving it cold-only on NVMe. `elevate_to_hot` pulls it back on demand.
    /// Set for two offload-only cases:
    ///   - **collection-member** sections (prefix-transparent: nothing attends
    ///     back over them during the build; hot→cold, no warm), and
    ///   - **completed-ingest** turns (e.g. a code_read file's turns, spliced
    ///     and sealed, not attended again until retrieval; hot→warm→cold).
    /// Live dialogue turns and boundary sections leave this `false` and stay
    /// resident.
    pub evict_when_cold: bool,
    /// `true` while this residence's warm→cold redo-log write is in flight on the
    /// off-thread [`crate::persistence::writer::SubstrateWriter`] (set at enqueue,
    /// cleared by `install_cold`). [`Self::snapshot_pending_cold`] skips it so the
    /// persistence thread doesn't re-gather + double-write the same turn while its
    /// first write is still queued. Distinct from `cold.is_some()`, which only
    /// becomes true once that write lands.
    pub cold_pending: bool,
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

/// Transient per-projection belief score cache.
///
/// **Not part of the persistent substrate state.** Built by the scheduler's
/// wide-Q belief scan during one reprojection, consumed by the projection
/// emitter during that same pass, then discarded. Conversation identity does not
/// include this — reload from log starts empty and the next scan repopulates it.
#[derive(Debug, Clone, Default)]
pub struct ProjectionScores {
    turns: AHashMap<TurnKey, f32>,
    sections: AHashMap<SectionId, f32>,
}

impl ProjectionScores {
    /// An empty score cache — every lookup defaults to zero.
    pub fn new() -> Self {
        Self::default()
    }

    /// Record the belief score for one turn.
    pub fn set_turn(&mut self, timeline: TimelineId, index: TurnIndex, score: f32) {
        self.turns.insert(TurnKey::new(timeline, index), score);
    }

    /// Record the belief score for one system-prompt section.
    pub fn set_section(&mut self, section: SectionId, score: f32) {
        self.sections.insert(section, score);
    }

    /// Look up a turn's belief score. Zero when not scored this projection.
    pub fn turn(&self, timeline: TimelineId, index: TurnIndex) -> f32 {
        self.turns
            .get(&TurnKey::new(timeline, index))
            .copied()
            .unwrap_or(0.0)
    }

    /// Look up a section's belief score. Zero when not scored.
    pub fn section(&self, section: SectionId) -> f32 {
        self.sections.get(&section).copied().unwrap_or(0.0)
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
        score: f32,
    ) {
        if let Some(timeline) = substrate.timelines_for_group(group).next() {
            self.set_turn(timeline, index, score);
        }
    }
}

// ── Per-section record ────────────────────────────────────────────────────────

/// Per-section state stored in the substrate.  Mirrors [`TurnEntryData`]
/// for sections — sections are scoreable like turns when their content
/// has been prefilled into a conversation's KV cache.
#[derive(Debug, Clone)]
pub struct SectionEntryData {
    token_count: usize,
    block_range: (u64, u64),
    tokens: Arc<Vec<u32>>,
    /// Slot in [`Substrate::residence`] holding this section's
    /// hot/warm/cold KV state. Sealed bytes live there.
    residence: ResidenceIndex,
}

// ── Per-turn record ───────────────────────────────────────────────────────────

/// One turn's pinned content in the substrate.  The turn's K/V
/// chunks are a single contiguous block addressing the persisted
/// token sequence
/// `[user_msg][user_end][assistant_start][response]`
/// — the inter-turn `user_start` head and `assistant_end` tail are
/// **not** persisted: the projection assembler re-emits them as
/// live `Generated` runs at every cross-turn boundary so their K
/// vectors are computed under the actual runtime causal prefix.
/// The interior `user_end` + `assistant_start` pair stays baked
/// because its semantic context (the turn's own user message and
/// decoded response) is invariant across projections.
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
    block_range: (u64, u64),
    /// The turn's pinned content as one indivisible unit.
    content: TurnPart,
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
    /// Every turn visible in `group`, across **all** of its active timelines.
    ///
    /// A group is a *shape*, not a conversation: `code_reading` declares one
    /// conversation per file, so a group routinely holds many timelines. Turn
    /// indices are per-timeline, so `(group, index)` alone is ambiguous — index 3
    /// exists in every file's timeline. Enumerating [`TurnKey`]s (timeline +
    /// index) is what lets every conversation in a multi-timeline group be scored
    /// and projected, instead of only the first-registered one.
    ///
    /// Returned in a stable order: timelines in registration order, turns
    /// ascending within each.
    fn group_turns(&self, group: GroupId) -> Vec<TurnKey>;

    /// Token count for a turn.  Stable across projection calls.
    fn turn_token_count(&self, turn: TurnKey) -> usize;

    /// Relevance score for a turn — the wide-Q belief score the scheduler
    /// recorded for this projection. Zero when unscored.
    fn turn_score(&self, turn: TurnKey) -> f32;

    /// Layer that produced a given turn.  Used to denormalise
    /// `layer_id` onto the emitted `TurnId` without a back-lookup
    /// through the schema.
    ///
    /// Default impl returns `None` (resolver doesn't track origins).
    /// When `None`, projection emit falls back to the layer-walk's
    /// `layer_id` — which is correct for tests using mock resolvers.
    fn turn_origin(&self, _turn: TurnKey) -> Option<LayerId> {
        None
    }

    /// The turn in `group` whose gather-scope decl tags contain `tag`, if any.
    /// Used to resolve a group's declared `default` member (a workspace-root
    /// cluster tagged `"."`, etc.) when normal selection is empty — so the
    /// group never drops out of the projection. Off the hot path: only consulted
    /// on an empty selection. Default `None` (mock resolvers carry no tags).
    fn turn_with_tag(&self, _group: GroupId, _tag: &str) -> Option<TurnKey> {
        None
    }

    /// Forest kind of a projected turn — `Normal` (a raw conversation turn) vs
    /// `SummaryOfTurns` / `SummaryOfSummaries` (a summary node standing in for
    /// the turns beneath it). Lets the projection record (and the GUI / inspector
    /// that read it) show whether a slot was filled with a real turn or a summary
    /// — the distinction that is otherwise invisible once a node is materialized
    /// as a `Sealed(Turn)` segment.
    ///
    /// Default impl returns `Normal` (mock resolvers and non-summarised timelines
    /// have no forest, so every turn is raw).
    fn turn_kind(&self, _turn: TurnKey) -> TurnKind {
        TurnKind::Normal
    }

    /// The turn indices `index` transitively covers in `group`'s summary forest
    /// — every descendant node (SummaryOfSummaries / SummaryOfTurns / Normal)
    /// beneath it, at every level. Empty for a raw `Normal` turn (a leaf covers
    /// nothing). A summary only ever covers turns on its OWN timeline, so this
    /// is self-contained per timeline.
    ///
    /// Used by the rule-based selection's descendant-dedup: a summary that wins
    /// a slot on provenance score is dropped when any node it covers also wins,
    /// so the projection keeps the SPECIFIC over the coarse (never a summary
    /// stacked on top of the very turns it summarises).
    ///
    /// Default impl returns empty (mock resolvers / non-summarised timelines
    /// have no forest, so nothing is covered).
    fn node_covers(&self, _turn: TurnKey) -> Vec<TurnIndex> {
        Vec::new()
    }

    /// Whether a sealed turn was decoded with `/no_think` thinking suppression —
    /// the assembler re-renders the `/no_think` soft-switch glue after
    /// `user_start` for such turns, so the materialized-glue builder needs the
    /// same bit to reproduce the engine's boundary run exactly. Default `false`
    /// (mock resolvers / non-substrate timelines have no suppression record).
    fn turn_no_think(&self, _timeline: TimelineId, _index: TurnIndex) -> bool {
        false
    }

    /// Token count for a system-prompt section.  Returns `0` for
    /// sections the resolver has no record of.
    fn section_token_count(&self, _section: SectionId) -> usize {
        0
    }

    /// Score-density selection over a timeline's summary forest
    /// (`docs/immutable_summary_forest.md` — *Window of attention*).
    /// Returns the chrono-
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
    ) -> Option<Vec<(TurnIndex, SelectionOrigin, f32)>> {
        None
    }

    /// Number of turns currently awaiting the async summariser for
    /// `timeline`.  Used by the projection to populate the
    /// score-density backpressure metric inside its diagnostic sink
    /// (§9 of `docs/archived/infinite_conversations.md`).  Default returns 0.
    fn pending_summary_len(&self, _timeline: TimelineId) -> usize {
        0
    }

    /// Relevance score for a system-prompt section — the wide-Q belief score
    /// the scheduler recorded. Default `0.0`; concrete resolvers override.
    fn section_score(&self, _section: SectionId) -> f32 {
        0.0
    }
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
    /// Creation-order rank, stamped from [`Substrate::conv_order_counter`] the
    /// first time `conv_id` is set. `0` until then. Higher = created later; the
    /// daemon sidebar sorts on this so newest conversations lead the list.
    pub order: u64,
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
    /// Per-turn tree metadata for the immutable summary forest
    /// (`docs/immutable_summary_forest.md`).  Parallel to `turns`: every
    /// recorded turn carries exactly one [`TreeNodeMeta`] entry (defaults to a
    /// `Normal` content sub-leaf with no children).  Promoted to a
    /// `SummaryOfTurns` / `SummaryOfSummaries` by the async summariser thread
    /// after the §6 probe runs.  The peak set (window entry points) is derived
    /// from this map — see [`Substrate::peaks_of`].
    pub tree_meta: BTreeMap<TurnIndex, TreeNodeMeta>,
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
    pub pending_summary_queue: std::collections::VecDeque<TurnIndex>,
    /// Whether this timeline's turns are fed to the summariser at all.
    /// `true` for dialogue; `false` for append-only utility/reference layers
    /// (repo_map, code_reading) — they are background reference, summarising
    /// them is pointless work that storms the summariser during repo
    /// ingest/scan. When `false`, turns are never pushed onto
    /// `pending_summary_queue`, so the summariser never touches this timeline.
    pub summarize: bool,
    /// Set on cold-load (and after the old-AVL migration) to ask the summariser
    /// to reconcile this timeline's persisted forest against the canonical
    /// ternary shape — building any missing/mismatched internal nodes on the
    /// low-priority queue.  Cleared once [`Substrate::reconcile_next`] reports
    /// the forest whole.  Live appends keep the forest whole, so this stays
    /// `false` during normal operation.
    pub needs_reconcile: bool,
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
fn leftmost_normal_in(tree_meta: &BTreeMap<TurnIndex, TreeNodeMeta>, idx: TurnIndex) -> u32 {
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
    pub chunks: BTreeMap<u64, ChunkLoc>,
    /// Latest `Tokens` record for the stream.
    pub tokens: Option<RecordLoc>,
    /// Latest `ProjectionEvents` record payload (opaque JSON bytes — the
    /// projection layer decodes). Eager bytes rather than a `RecordLoc`: the
    /// per-turn timeline is tiny, so we keep it resident and re-emit it on
    /// compaction like the other synthesised per-entity records.
    pub projection_events: Option<Vec<u8>>,
    /// Opaque encoded wide-Q signature window (`provenance::wide_sig`) for this turn's most
    /// recent (re)projection — the decode→decode (`Q·Q`) consensus substrate. Last-writer-wins;
    /// rebuilt from the redo log on replay. `None` until the first projection writes it.
    pub wide_q_sigs: Option<Vec<u8>>,
    /// Highest chunk index the stream is durably committed through.
    pub committed_through: Option<u64>,
}

impl TimelineEntry {
    fn new(layer: LayerId, group: GroupId) -> Self {
        Self {
            layer,
            group,
            couplings: Couplings::default(),
            label: None,
            conv_id: None,
            order: 0,
            custom: BTreeMap::new(),
            archived: false,
            turns: BTreeMap::new(),
            tree_meta: BTreeMap::new(),
            debug_id: None,
            pending_summary_queue: std::collections::VecDeque::new(),
            summarize: true,
            needs_reconcile: false,
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
            evict_when_cold: false,
            cold_pending: false,
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
        let slot = &mut self.residence[residence.0];
        slot.cold = Some(cold);
        // The async write landed — clear the in-flight flag so a re-gather is
        // gated by `cold.is_some()` from here on.
        slot.cold_pending = false;
        // Offload-as-we-go: the (quantized) bytes are now durable on disk, so
        // free EVERY resident tier and let `elevate_to_hot` pull the residence
        // back from cold (NVMe) on demand if a later projection re-selects it.
        //   - Sections are hot→cold (no warm): the `warm` drop is a no-op.
        //   - Completed-ingest turns are hot→warm→cold: `hot` was already freed
        //     at warm-land (see the migrate install loop), and dropping `warm`
        //     here reclaims the RAM copy too — leaving the turn cold-only, off
        //     both the VRAM and RAM tiers.
        // The drop returns arena chunks to the pool / frees the CPU copy. Runs
        // under the persistence thread's substrate write lock (Phase 2.5), so
        // the arena free is serialised with the scheduler's allocations.
        if !slot.evict_when_cold {
            return;
        }
        let had_hot = slot.hot.take().is_some();
        let had_warm = slot.warm.take().is_some();
        if had_hot {
            Self::remove_from_lru(&mut self.hot_lru, residence);
        }
        if had_warm {
            Self::remove_from_lru(&mut self.warm_lru, residence);
        }
    }

    /// Flag a section residence so the persistence thread frees its `hot` the
    /// moment a cold copy lands — see [`SequenceResidence::evict_when_cold`].
    /// Set by the scheduler's collection-member quantize drain.
    pub fn mark_section_evict_when_cold(&mut self, residence: ResidenceIndex) {
        self.residence[residence.0].evict_when_cold = true;
    }

    /// Whether `residence` is flagged for full eviction once its KV is durable
    /// on disk — read by the persistence thread's hot→warm install so a flagged
    /// turn goes straight to warm-only (its GPU copy freed the moment warm
    /// lands) rather than lingering hot until cold. See
    /// [`SequenceResidence::evict_when_cold`].
    pub fn residence_evict_when_cold(&self, residence: ResidenceIndex) -> bool {
        self.residence[residence.0].evict_when_cold
    }

    /// Flag every turn residence of `timeline` for full eviction the moment its
    /// KV is durable on disk (see [`SequenceResidence::evict_when_cold`]): the
    /// hot→warm install frees VRAM as warm lands, then `install_cold` frees the
    /// warm RAM copy as cold lands, leaving each turn cold-only on NVMe.
    ///
    /// Used to reclaim a **completed ingest** conversation (e.g. a code_read
    /// file) whose turns won't be attended again until retrieval pulls them
    /// back via `elevate_to_hot` — so their KV must not linger resident and
    /// accumulate through a large multi-file ingest. Returns the number of turn
    /// residences flagged.
    ///
    /// Reclaims VRAM **immediately** for any turn already warm-backed: it drops
    /// the hot copy on the spot (keeping warm — a safe demote, since the RAM copy
    /// survives and `elevate_to_hot` reloads on demand). This is the load-bearing
    /// case: a file usually completes AFTER its turns have already migrated
    /// hot→warm, so `install_warm_and_evict_hot` (which only fires DURING the
    /// migrate, for residences flagged beforehand) never ran for them — nothing
    /// else drops their hot until the async, possibly-lagging cold write lands.
    /// Without this, completed files' hot KV piles up and fills the card mid-ingest.
    pub fn mark_timeline_evict_when_cold(&mut self, timeline: TimelineId) -> usize {
        let Some(entry) = self.timelines.get(&timeline) else {
            return 0;
        };
        let residences: Vec<ResidenceIndex> =
            entry.turns.values().map(|t| t.content.residence).collect();
        for r in &residences {
            self.residence[r.0].evict_when_cold = true;
            // Immediate hot-drop for already-warm turns (VRAM back now, warm kept).
            let (has_hot, has_warm) = {
                let slot = &self.residence[r.0];
                (slot.hot.is_some(), slot.warm.is_some())
            };
            if has_hot && has_warm {
                self.residence[r.0].hot = None;
                Self::remove_from_lru(&mut self.hot_lru, *r);
            }
        }
        residences.len()
    }

    /// Flag a SINGLE turn's residence for full eviction once its KV is durable —
    /// see [`Self::mark_timeline_evict_when_cold`]. Set at splice time (while the
    /// turn is still hot, before the hot→warm migrate) so the migrate's
    /// `install_warm_and_evict_hot` drops its Q-format hot the moment warm lands,
    /// instead of installing a resident Q copy. Load-bearing for code_read: a
    /// scope turn adopted onto its file timeline is never re-attended until query
    /// time, and the file timeline is NOT in the scheduler's `ingest_timelines`,
    /// so the gentle demote never covers it — without this its migrated Q hot
    /// copy accumulates and climbs the card (`quant_live` leak).
    pub fn mark_turn_evict_when_cold(&mut self, timeline: TimelineId, idx: TurnIndex) {
        if let Some(r) = self.turn(timeline, idx).map(|t| t.content.residence) {
            self.residence[r.0].evict_when_cold = true;
        }
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

    /// Total VRAM byte footprint of hot residences that lack a warm copy — the
    /// hot→warm drain **backlog**. Cheap companion to [`Self::snapshot_pending_warm`]
    /// (which clones every hot `SealedSequence` for the migration): this only
    /// sums each slot's cached `byte_size`, so the persistence thread can stamp
    /// it every pass and the scheduler can poll it to drive ingest admission
    /// backpressure off a *leading* signal (drain deficit) rather than the
    /// lagging VRAM-pressure trip.
    pub fn pending_warm_bytes(&self) -> u64 {
        self.hot_lru
            .iter()
            .filter_map(|&idx| {
                let slot = &self.residence[idx.0];
                (slot.hot.is_some() && slot.warm.is_none()).then_some(slot.byte_size)
            })
            .sum()
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
                // Skip a turn whose cold copy already landed OR whose async cold
                // write is still queued on the writer — re-selecting the latter
                // would double-write the same KV.
                if slot.cold.is_some() || slot.cold_pending {
                    return None;
                }
                slot.warm
                    .as_ref()
                    .map(|warm| (idx, slot.stream_id, warm.clone()))
            })
            .collect()
    }

    /// Flag a residence's warm→cold write as in flight on the off-thread writer,
    /// so [`Self::snapshot_pending_cold`] won't re-gather it before the write lands
    /// (which clears the flag via `install_cold`). Set at enqueue.
    pub fn mark_cold_pending(&mut self, residence: ResidenceIndex) {
        self.residence[residence.0].cold_pending = true;
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
        // Residences popped off the LRU tail that we must NOT drop (their warm
        // copy is the turn's only surviving copy). Restored to the LRU after the
        // loop so a later pass reclaims them once a lower tier lands.
        let mut skipped: Vec<ResidenceIndex> = Vec::new();
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
            // Only drop warm when the data survives elsewhere — hot in VRAM or
            // cold on disk. A warm-only residence (its cold write hasn't landed,
            // or failed) is its turn's ONLY copy; dropping it would lose the K/V.
            // Set it aside (restored below) rather than free it.
            let slot = &self.residence[idx.0];
            if slot.warm.is_some() && slot.cold.is_none() && slot.hot.is_none() {
                skipped.push(idx);
                continue;
            }
            let slot = &mut self.residence[idx.0];
            if slot.warm.take().is_some() {
                freed_bytes = freed_bytes.saturating_add(slot.byte_size);
                count += 1;
                tracing::trace!(
                    target: "candle_conversation::persistence::tier",
                    residence = idx.0,
                    bytes = slot.byte_size,
                    "purged warm (RAM headroom)"
                );
            }
            // If warm was None, the slot was stale in the LRU; just
            // discard the index and keep looking.
        }
        // Restore the residences we couldn't safely drop; they remain valid warm
        // entries and rejoin the LRU tail in their original relative order.
        for idx in skipped {
            self.warm_lru.push_back(idx);
        }
        if count > 0 {
            tracing::trace!(
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

    /// Drop the hot copy of each of `turns` that also holds a warm copy, keeping
    /// warm — the cheap inverse of a warm→hot lift (no migrate, since the warm
    /// bytes already exist). Unlike [`Self::evict_hot_to_free`], which evicts the
    /// oldest *non*-kept residences until a byte target is met, this targets an
    /// explicit set: used to release a *transient* working set (e.g. a summary
    /// pass's children, lifted only so the compressor could attend over them)
    /// back out of VRAM the instant it's no longer needed, so background
    /// elevation churn can't accumulate hot residency and exhaust the card.
    /// Turns that are cold-only or hot-without-warm are left untouched (dropping
    /// hot without a warm copy would lose their K/V). Returns the number demoted.
    pub fn demote_turns_to_warm(&mut self, turns: &[TurnKey]) -> usize {
        let mut demoted = 0;
        for &key in turns {
            let Some(residence) = self
                .turn(key.timeline, key.index)
                .map(|e| e.content.residence)
            else {
                continue;
            };
            let (has_hot, has_warm) = {
                let slot = &self.residence[residence.0];
                (slot.hot.is_some(), slot.warm.is_some())
            };
            if has_hot && has_warm {
                self.residence[residence.0].hot = None;
                Self::remove_from_lru(&mut self.hot_lru, residence);
                demoted += 1;
            }
        }
        demoted
    }

    /// Gentle-early relief, LRU-smart: shed the **least-recently-active** ingest
    /// hot KV to warm, freeing at most `target_bytes` (relieve to the watermark,
    /// not everything), while PROTECTING the active working set so it never
    /// churns. Only ingest KV is touched — it re-elevates from warm at zero cost,
    /// unlike a live-chat turn.
    ///
    /// Two protections keep an actively-ingesting conversation resident:
    /// 1. `keep_turns` / `keep_sections` — the union working set of every live
    ///    slot (what in-flight prefills/decodes are attending RIGHT NOW).
    /// 2. `keep_recent` — the rolling window of the newest turns per ingest
    ///    timeline (what the NEXT scope's projection will re-gather). This must
    ///    cover the projection's gather width or the demote fights it.
    ///
    /// Everything else is walked oldest-first off `hot_lru` (least-recently
    /// promoted ≈ least-recently active) and demoted until `target_bytes` is met —
    /// so a settled conversation's cold tail is shed before an active one's window.
    pub fn demote_cold_ingest(
        &mut self,
        ingest_timelines: &std::collections::HashSet<TimelineId>,
        keep_turns: &[TurnKey],
        keep_sections: &[SectionId],
        keep_recent: usize,
        target_bytes: u64,
    ) -> EvictionReport {
        if target_bytes == 0 {
            return EvictionReport { count: 0, bytes: 0 };
        }
        // Protected residence: live working sets + each ingest timeline's rolling
        // window. Also collect which residence belongs to an ingest timeline —
        // only those are evictable here (zero reload cost).
        let mut protected: std::collections::HashSet<ResidenceIndex> =
            std::collections::HashSet::new();
        for &key in keep_turns {
            if let Some(e) = self.turn(key.timeline, key.index) {
                protected.insert(e.content.residence);
            }
        }
        for &sid in keep_sections {
            if let Some(e) = self.sections.get(&sid) {
                protected.insert(e.residence);
            }
        }
        let mut ingest_residence: std::collections::HashSet<ResidenceIndex> =
            std::collections::HashSet::new();
        for (tl, entry) in self.timelines.iter() {
            if !ingest_timelines.contains(tl) {
                continue;
            }
            let n = entry.turns.len();
            let cutoff = n.saturating_sub(keep_recent);
            for (i, turn_data) in entry.turns.values().enumerate() {
                let r = turn_data.content.residence;
                ingest_residence.insert(r);
                if i >= cutoff {
                    protected.insert(r); // the will-be-regathered rolling window
                }
            }
        }
        // Walk `hot_lru` oldest→newest (back is oldest), demoting warm-backed
        // ingest KV that isn't protected, until we've freed `target_bytes`.
        let mut freed: u64 = 0;
        let mut victims: Vec<ResidenceIndex> = Vec::new();
        for idx in self.hot_lru.iter().rev().copied() {
            if freed >= target_bytes {
                break;
            }
            if !ingest_residence.contains(&idx) || protected.contains(&idx) {
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
                keep_recent,
                target_bytes,
                "demote_cold_ingest (gentle-early, LRU) complete"
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

    /// Resolve which timeline a projected turn in `group` belongs to.
    ///
    /// The projection `target` pins the ACTIVE conversation: a turn in the
    /// target's own group resolves to `target.timeline` (the conversation being
    /// projected/decoded); every other group has a single registered timeline,
    /// taken in registration order.  A `None` target (e.g. a utility pass with no
    /// active conversation) falls back to that registration-order pick for all
    /// groups.
    ///
    /// This is the SINGLE source of truth for turn → timeline resolution.  It
    /// exists because open-coding `timelines_for_group(g).next()` WITHOUT the
    /// target-group special case resolves the wrong conversation's timeline as
    /// soon as more than one conversation shares a group — which silently dropped
    /// every turn of any non-first conversation on mid-decode reproject (the slot
    /// rebuilt with `turns=0`, i.e. the model lost its whole history).  Every call
    /// site — the SubmitTurn prefill, the reproject rebuild, and
    /// `inject_sealed_turn` — routes through here so they can never diverge again.
    pub fn resolve_turn_timeline(
        &self,
        target: Option<ProjectionTarget>,
        group: GroupId,
    ) -> Option<TimelineId> {
        match target {
            Some(t) if group == t.group => Some(t.timeline),
            _ => self.timelines_for_group(group).next(),
        }
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

    // ── tool-round-trip couplings ───────────────────────────────────────────

    /// Couple `from_turn` to the tool response that follows it.
    ///
    /// Idempotent — replaying the same record, or coupling twice in a live
    /// session, is a no-op. A coupling naming an unregistered timeline is
    /// dropped: with no turns to group it could only describe a round-trip that
    /// does not exist.
    pub fn couple_turn(&mut self, timeline: TimelineId, from_turn: u32) {
        if let Some(tl) = self.timelines.get_mut(&timeline) {
            tl.couplings.insert(from_turn);
        }
    }

    /// Replay a persisted [`TurnCouplingPayload`].
    pub fn apply_turn_coupling(&mut self, payload: &TurnCouplingPayload) {
        let Some(timeline) = TimelineId::from_raw(payload.timeline_id) else {
            return;
        };
        self.couple_turn(timeline, payload.from_turn);
    }

    /// The turns on `timeline` coupled to the tool response that follows them.
    /// Empty for a timeline that has never made a tool call — every turn is then
    /// its own exchange.
    pub fn couplings_of(&self, timeline: TimelineId) -> Couplings {
        self.timelines
            .get(&timeline)
            .map(|t| t.couplings.clone())
            .unwrap_or_default()
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

    /// Overwrite a turn's [`TreeNodeMeta`].  The summariser thread calls this
    /// once, to promote a turn into the forest (Normal sub-leaf → SummaryOfTurns
    /// leaf, or to record a fresh SummaryOfSummaries internal).  Nodes are
    /// immutable: an existing summary node is never rewritten.
    pub fn set_tree_meta(&mut self, timeline: TimelineId, idx: TurnIndex, meta: TreeNodeMeta) {
        if let Some(tl) = self.timelines.get_mut(&timeline) {
            tl.tree_meta.insert(idx, meta);
        }
    }

    /// The peak set — orphan summary nodes (no parent), in chronological order
    /// (oldest/leftmost-covering first), each paired with its level.  These are
    /// the window's coarse entry points (`docs/immutable_summary_forest.md`).
    pub fn peaks_of(&self, timeline: TimelineId) -> Vec<(TurnIndex, u8)> {
        let Some(tl) = self.timelines.get(&timeline) else {
            return Vec::new();
        };
        let mut claimed: std::collections::BTreeSet<TurnIndex> = std::collections::BTreeSet::new();
        for meta in tl.tree_meta.values() {
            if meta.kind == TurnKind::SummaryOfSummaries {
                for c in &meta.children {
                    claimed.insert(*c);
                }
            }
        }
        let mut peaks: Vec<(TurnIndex, u8)> = tl
            .tree_meta
            .iter()
            .filter(|(idx, meta)| meta.kind.is_summary() && !claimed.contains(*idx))
            .map(|(idx, meta)| (*idx, meta.tree_height))
            .collect();
        peaks.sort_by_key(|(idx, _)| leftmost_normal_in(&tl.tree_meta, *idx));
        peaks
    }

    /// Cheap guard: does this timeline have any summary nodes yet?
    pub fn has_summary_nodes(&self, timeline: TimelineId) -> bool {
        self.timelines
            .get(&timeline)
            .map(|tl| tl.tree_meta.values().any(|m| m.kind.is_summary()))
            .unwrap_or(false)
    }

    /// The next internal node to (re)build during reconciliation, expressed as
    /// the [`MERGE_FANOUT`] child indices it must cover — the lowest buildable
    /// node (all children already present) whose canonical `SummaryOfSummaries`
    /// is missing from the persisted forest.  `None` once the forest matches the
    /// canonical ternary shape for its leaves.
    ///
    /// Derived from the persisted state each call (the "dirty" bit is gone —
    /// staleness is computed, never stored).  See
    /// `docs/immutable_summary_forest.md`.
    pub fn reconcile_next(&self, timeline: TimelineId) -> Option<Vec<TurnIndex>> {
        let tl = self.timelines.get(&timeline)?;
        let tm = &tl.tree_meta;
        // SoT leaves in chronological order (by their single Normal child).
        let mut leaves: Vec<(u32, TurnIndex)> = tm
            .iter()
            .filter(|(_, m)| m.kind == TurnKind::SummaryOfTurns)
            .map(|(idx, m)| (m.children.first().map(|c| c.0).unwrap_or(idx.0), *idx))
            .collect();
        leaves.sort_by_key(|(normal, _)| *normal);
        // Persisted SoS indexed by exact children signature (canonical nodes
        // have `MERGE_FANOUT` children; old binary nodes never match).
        let mut sos_by_children: std::collections::HashMap<Vec<TurnIndex>, TurnIndex> =
            std::collections::HashMap::new();
        for (idx, m) in tm {
            if m.kind == TurnKind::SummaryOfSummaries {
                sos_by_children.insert(m.children.clone(), *idx);
            }
        }
        // Replay the ternary carry with persisted nodes; the first carry whose
        // canonical SoS is absent is the lowest buildable rebuild.
        let mut peaks: Vec<(TurnIndex, u8)> = Vec::new();
        for (_, leaf_idx) in &leaves {
            peaks.push((*leaf_idx, 1));
            loop {
                let n = peaks.len();
                if n < MERGE_FANOUT {
                    break;
                }
                let lvl = peaks[n - 1].1;
                if !peaks[n - MERGE_FANOUT..].iter().all(|(_, l)| *l == lvl) {
                    break;
                }
                let children: Vec<TurnIndex> =
                    peaks[n - MERGE_FANOUT..].iter().map(|(i, _)| *i).collect();
                match sos_by_children.get(&children) {
                    Some(&sos_idx) => {
                        peaks.truncate(n - MERGE_FANOUT);
                        peaks.push((sos_idx, lvl + 1));
                    }
                    None => return Some(children),
                }
            }
        }
        None
    }

    /// Whether the summariser should run a reconcile pass for this timeline.
    pub fn needs_reconcile(&self, timeline: TimelineId) -> bool {
        self.timelines
            .get(&timeline)
            .map(|tl| tl.needs_reconcile)
            .unwrap_or(false)
    }

    /// Set/clear the reconcile hint.
    pub fn set_needs_reconcile(&mut self, timeline: TimelineId, v: bool) {
        if let Some(tl) = self.timelines.get_mut(&timeline) {
            tl.needs_reconcile = v;
        }
    }

    /// Prepare a freshly-loaded timeline for reconciliation: drop any
    /// non-canonical `SummaryOfSummaries` meta (e.g. binary nodes written by the
    /// superseded AVL code) so the ternary canonical nodes can be rebuilt, and
    /// arm the reconcile hint when the forest isn't already whole.
    pub fn mark_for_reconcile(&mut self, timeline: TimelineId) {
        let Some(tl) = self.timelines.get_mut(&timeline) else {
            return;
        };
        let noncanonical: Vec<TurnIndex> = tl
            .tree_meta
            .iter()
            .filter(|(_, m)| {
                m.kind == TurnKind::SummaryOfSummaries && m.children.len() != MERGE_FANOUT
            })
            .map(|(idx, _)| *idx)
            .collect();
        for idx in noncanonical {
            tl.tree_meta.remove(&idx);
        }
        // Arm reconcile when any leaf exists; `reconcile_next` returns `None`
        // immediately when the forest is already whole, so this is cheap.
        let has_leaf = tl
            .tree_meta
            .values()
            .any(|m| m.kind == TurnKind::SummaryOfTurns);
        tl.needs_reconcile = has_leaf;
    }

    // ── Pending + reconcile accessors (§6 backpressure metrics) ────────────

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

    /// `pending_summary_queue.len()` — backpressure metric (§9).
    pub fn pending_summary_len(&self, timeline: TimelineId) -> usize {
        self.timelines
            .get(&timeline)
            .map(|tl| tl.pending_summary_queue.len())
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
    /// latest tokens location + committed-through watermark + decl.
    pub fn stream_of(&self, stream_id: StreamId) -> Option<&StreamRuntime> {
        self.streams.get(&stream_id)
    }

    /// Decoded wide-Q window for `stream_id`, memoized across reprojections.
    ///
    /// The belief scan reads the same static gallery on every reprojection;
    /// re-`decode_wide_sigs`-ing all of it each time is the single largest
    /// repeated cost of the scan. This returns a shared [`Arc`] of the decoded
    /// window, decoding on first touch and serving the memo thereafter. `None`
    /// when the stream has no signature or an empty one. Interior-mutable: safe
    /// to call under a read lock (a blob write evicts the stale entry under a
    /// write lock, so a read never observes a stale window).
    pub fn decoded_wide_sig(&self, stream_id: StreamId) -> Option<Arc<Vec<WideQSig>>> {
        // Fast path — a cached decode. Scoped so the lock releases before the
        // decode below. `unwrap_or_else(into_inner)` recovers a poisoned mutex:
        // the memo holds only plain data, so a panic elsewhere can't corrupt it.
        {
            let cache = self.sig_cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(hit) = cache.get(&stream_id) {
                return hit.clone();
            }
        }
        // Decode OUTSIDE the lock so a slow decode never blocks another reader's
        // cache hit; a concurrent miss on the same stream just decodes twice
        // (harmless — the value is identical, insert is last-writer-wins).
        let decoded = self
            .streams
            .get(&stream_id)
            .and_then(|e| e.wide_q_sigs.as_ref())
            .and_then(|b| decode_wide_sigs(b))
            .filter(|w| !w.is_empty())
            .map(Arc::new);
        self.sig_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(stream_id, decoded.clone());
        decoded
    }

    /// Evict `stream_id`'s decoded-window memo — the **incremental** invalidation
    /// used when its blob is (re)written, so one turn seal re-decodes only that
    /// one window on the next scan instead of churning the whole gallery.
    #[inline]
    fn evict_decoded_wide_sig(&mut self, stream_id: StreamId) {
        self.sig_cache
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&stream_id);
    }

    /// A turn's self-referencing sub-window seam offsets (the `start_token`s of
    /// its `self_reference` projection events), **sorted and deduped** — decoded
    /// from the events JSON blob once per session and memoized, so the belief
    /// scan reads them without re-parsing JSON for every gallery turn on every
    /// reprojection. Returns an empty (shared) list when the stream has no events
    /// or no self-referencing seams (the common case for a code-read scope, which
    /// scores as one whole-turn window). See [`Self::score_belief_groups`].
    pub fn decoded_seams(&self, stream_id: StreamId) -> Arc<Vec<usize>> {
        {
            let cache = self.seam_cache.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(hit) = cache.get(&stream_id) {
                return hit.clone();
            }
        }
        // Decode OUTSIDE the lock (mirrors `decoded_wide_sig`): a concurrent miss
        // on the same stream just decodes twice — harmless, the value is identical.
        let mut seams: Vec<usize> = self
            .streams
            .get(&stream_id)
            .and_then(|e| e.projection_events.as_deref())
            .map(decode_events)
            .map(|evs| {
                evs.iter()
                    .filter(|e| e.self_reference)
                    .map(|e| e.start_token as usize)
                    .collect()
            })
            .unwrap_or_default();
        seams.sort_unstable();
        seams.dedup();
        let arc = Arc::new(seams);
        self.seam_cache
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(stream_id, arc.clone());
        arc
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
            tl.pending_summary_queue.clear();
            tl.needs_reconcile = false;
        }
    }

    /// Emit `(timeline_id, conv_id, label, archived, custom)` tuples for
    /// every timeline that holds non-default values.  Used by compaction
    /// to re-emit live `Label` / `ConvState` records.
    pub fn live_conv_meta(&self) -> Vec<(u64, String, String, bool, BTreeMap<String, String>)> {
        // Emit in creation `order`, not `timelines` (HashMap) iteration order:
        // the compactor writes these as `Label` records, and reload re-derives
        // each timeline's `order` from the order its `conv_id` Label replays
        // (see `set_conv_id`). A nondeterministic order here would scramble the
        // sidebar's creation-order sort on every compaction. `order` is 0 for
        // timelines that never got a conv_id (label/custom only); they sort
        // first and their relative order is immaterial.
        let mut out: Vec<(u64, u64, String, String, bool, BTreeMap<String, String>)> = self
            .timelines
            .iter()
            .filter_map(|(tid, tl)| {
                let conv_id = tl.conv_id.clone().unwrap_or_default();
                let label = tl.label.clone().unwrap_or_default();
                if conv_id.is_empty() && label.is_empty() && !tl.archived && tl.custom.is_empty() {
                    None
                } else {
                    Some((
                        tl.order,
                        tid.raw(),
                        conv_id,
                        label,
                        tl.archived,
                        tl.custom.clone(),
                    ))
                }
            })
            .collect();
        out.sort_by_key(|(order, tid, ..)| (*order, *tid));
        out.into_iter()
            .map(|(_, tid, conv_id, label, archived, custom)| {
                (tid, conv_id, label, archived, custom)
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
                out.push(TreeMetadataPayload {
                    timeline_id: tid.raw(),
                    turn_index: idx.0,
                    kind,
                    tree_height: meta.tree_height,
                    children: meta.children.iter().map(|c| c.0).collect(),
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

    /// Apply a decoded `TreeMetadataPayload` — sets the turn's per-node forest
    /// meta (kind / children / level).  The peak set is derived, so there is no
    /// root marker to apply.
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
        };
        self.set_tree_meta(timeline, TurnIndex(payload.turn_index), meta);
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
        // A tombstoned timeline's KV is dead — release its resident VRAM NOW rather
        // than wait for compaction. Its chunks survive for any other holder: a
        // code_read scope fork's two turns are spliced onto the file timeline
        // (which clones the chunk handles) right before the fork is tombstoned, so
        // dropping the fork's hot only releases its redundant reference. For a
        // genuinely deleted timeline the KV is dead anyway. Without this, the
        // fork's orphaned hot copies — which nothing else evicts, the fork being
        // tombstoned and never demoted — accumulate on the card through a bulk
        // ingest (the `quant_live` climb).
        let residences: Vec<ResidenceIndex> = match self.timelines.get(&timeline) {
            Some(entry) => entry.turns.values().map(|t| t.content.residence).collect(),
            None => return,
        };
        for r in &residences {
            // Flag evict_when_cold BEFORE dropping hot: it closes the race where the
            // persistence thread snapshotted this residence's hot before the
            // tombstone and installs it after — `install_warm_and_evict_hot` (which
            // the flag selects) then drops that re-added Q copy instead of keeping it.
            self.residence[r.0].evict_when_cold = true;
            if self.residence[r.0].hot.take().is_some() {
                Self::remove_from_lru(&mut self.hot_lru, *r);
            }
        }
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

    /// Apply a decoded [`DistillPayload`] — marks the timeline for distillation
    /// at its recorded [`DistillMode`]. A later record upgrades the mode
    /// (last-writer-wins), so a conversation distilled provenance-only and then
    /// archived ends up `TextOnly`.
    pub fn apply_distill(&mut self, payload: &DistillPayload) {
        if let Some(timeline) = TimelineId::from_raw(payload.timeline_id) {
            self.distilled_timelines.insert(timeline, payload.mode);
        }
    }

    /// Mark `timeline` for distillation in-RAM at `mode` (callers also write the
    /// matching `Distilled` record so the marker survives reload).
    pub fn distill_timeline(&mut self, timeline: TimelineId, mode: DistillMode) {
        self.distilled_timelines.insert(timeline, mode);
    }

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
    pub fn distilled_timelines(&self) -> &HashMap<TimelineId, DistillMode> {
        &self.distilled_timelines
    }

    /// On-disk bytes held by streams of tombstoned timelines — dead
    /// weight the header-keyed accounting can't see (a tombstone names
    /// its timeline in the payload, and the doomed records were live
    /// appends at write time).  Summed from the in-RAM stream index, no
    /// disk I/O; the compaction trigger adds this to the incremental
    /// dead-byte counter.
    pub fn tombstoned_stream_bytes(&self) -> u64 {
        if self.tombstoned_timelines.is_empty() {
            return 0;
        }
        self.streams
            .values()
            .filter(|s| match &s.decl {
                Some(StreamDecl::Turn(t)) => TimelineId::from_raw(t.timeline_id)
                    .is_some_and(|tl| self.tombstoned_timelines.contains(&tl)),
                _ => false,
            })
            .map(|s| {
                s.chunks.values().map(|c| c.record_size).sum::<u64>()
                    + s.tokens.map_or(0, |l| l.record_size)
            })
            .sum()
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
    pub fn refresh_cold_refs(&mut self) {
        let streams = &self.streams;
        for slot in &mut self.residence {
            let Some(cold) = &mut slot.cold else { continue };
            let Some(stream) = streams.get(&slot.stream_id) else {
                continue;
            };
            let n_layers = cold.len();
            let chunks_per_layer = cold.first().map_or(0, |s| s.chunks.len());
            if chunks_per_layer == 0 || stream.chunks.len() != n_layers * chunks_per_layer {
                continue;
            }
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
        }
    }

    /// Apply one walked redo-log record directly into the substrate's
    /// in-RAM state.  The dispatch lives here (not on `Manifest`)
    /// because per-entity records — chunks, stream decls, labels,
    /// tree metadata, debug ids — are substrate state, not manifest
    /// state.  The manifest only sees singletons (`ModelSpec`,
    /// `Template`, `Tokenizer`, `ToolSummary`).
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
            RecordType::TurnCoupling => {
                if let Ok(payload) = TurnCouplingPayload::decode(&entry.record.payload) {
                    self.apply_turn_coupling(&payload);
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
            RecordType::Distilled => {
                if let Ok(payload) = DistillPayload::decode(&entry.record.payload) {
                    self.apply_distill(&payload);
                }
            }
            RecordType::ProjectionEvents => {
                // Opaque JSON bytes — the projection layer decodes them on read.
                // Last-writer-wins per turn stream id.
                self.streams.entry(stream_id).or_default().projection_events =
                    Some(entry.record.payload.clone());
                // Mirror the `WideQSig` arm: drop this stream's memoized seams so a
                // replay/apply after the seam cache warmed can't serve stale seams.
                self.seam_cache
                    .get_mut()
                    .unwrap_or_else(|e| e.into_inner())
                    .remove(&stream_id);
            }
            RecordType::WideQSig => {
                // Opaque wide-Q window bytes (provenance::wide_sig), last-writer-wins
                // per turn stream id — each (re)projection overwrites the window.
                self.streams.entry(stream_id).or_default().wide_q_sigs =
                    Some(entry.record.payload.clone());
                self.evict_decoded_wide_sig(stream_id);
            }
            // Singletons go to the manifest, not the substrate; the
            // header-index chain is consumed by recovery, never here.
            RecordType::ModelSpec
            | RecordType::Template
            | RecordType::Tokenizer
            | RecordType::HeaderIndex
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
        let tl = self.timelines.get_mut(&timeline).unwrap();
        tl.turns.insert(
            idx,
            TurnEntryData {
                block_range: (block_start, block_end),
                content: TurnPart {
                    layout: TurnLayout::default(),
                    token_count,
                    token_ids: TokenBuffer::default(),
                    residence,
                },
            },
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
        let token_count = write.token_count;
        {
            let tl = self.timelines.get_mut(&timeline).unwrap();
            tl.turns.insert(
                idx,
                TurnEntryData {
                    block_range: (block_start, block_end),
                    content: TurnPart {
                        layout: write.layout,
                        token_count,
                        token_ids: write.token_ids,
                        residence,
                    },
                },
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
            let tl = self.timelines.get_mut(&timeline).unwrap();
            tl.turns.insert(
                idx,
                TurnEntryData {
                    block_range: (block_start, block_end),
                    content: TurnPart {
                        layout,
                        token_count,
                        token_ids,
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
        *self.timeline_token_totals.entry(timeline).or_default() += token_count;
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
            .map(|e| e.content.layout.user_text().to_string())
            .unwrap_or_default()
    }

    /// The assistant's decoded reply text for this turn — the full message
    /// (reasoning block + answer), reconstructed by [`TurnLayout::assistant_text`].
    pub fn assistant_text_of(&self, timeline: TimelineId, index: TurnIndex) -> String {
        self.turn(timeline, index)
            .map(|e| e.content.layout.assistant_text().unwrap_or_default())
            .unwrap_or_default()
    }

    /// Turn token IDs as an owned `Vec` (clones the buffer).
    pub fn token_ids_of(&self, timeline: TimelineId, index: TurnIndex) -> Vec<u32> {
        self.turn(timeline, index)
            .map(|e| e.content.token_ids[..].to_vec())
            .unwrap_or_default()
    }

    /// The turn's [`TurnLayout`] — its segment-vector description: the complete,
    /// validated description of its K/V (user / thinking / assistant / boundary
    /// glue). Built at seal time and stored on the turn, so this is a direct
    /// clone with no re-derivation. `None` if the turn isn't found.
    pub fn turn_layout(&self, timeline: TimelineId, index: TurnIndex) -> Option<TurnLayout> {
        self.turn(timeline, index).map(|e| e.content.layout.clone())
    }

    /// The turn's slot block extent `(block_start, block_end)` — needed by the
    /// ordered-merge splice ([`crate::projection::resolver::Conversation::adopt_turn`])
    /// to re-record a forked scope turn onto the file timeline preserving its
    /// range verbatim. `None` if the turn isn't found.
    pub fn turn_block_range(&self, timeline: TimelineId, index: TurnIndex) -> Option<(u64, u64)> {
        self.turn(timeline, index).map(|e| e.block_range)
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
        let Some(entry) = self.turn_mut(timeline, index) else {
            return;
        };
        entry.content.token_count = entry.content.token_count.saturating_add(additional_tokens);
        entry.block_range.1 = new_block_end;
        // `entry`'s borrow ends here, so the counter bump can re-borrow self.
        *self.timeline_token_totals.entry(timeline).or_default() += additional_tokens;
    }

    pub fn block_range_of(&self, timeline: TimelineId, index: TurnIndex) -> (u64, u64) {
        self.turn(timeline, index).map_or((0, 0), |e| e.block_range)
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

    /// Sealed K/V for the turn's *user-message body* `[user_start, user_end)`
    /// (the layout's user span), derived on demand as a zero-copy window
    /// view over the turn's existing chunks via [`window_sealed_tokens`].
    /// Content-only — no leading or trailing chat-template role marker — so
    /// the compression assembler can inject it into the user-input region as
    /// role-matched sealed K/V (no re-prefill) when laying down a
    /// `SealedKind::TurnHalf` segment.
    ///
    /// Only the user half is injected; the assistant half is text-prefilled
    /// instead (its assistant-role K/V would be incoherent in the
    /// compression's user-input frame — see [`Self::turn_assistant_token_ids`]).
    pub fn turn_user_sealed_half(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Option<Arc<Vec<SealedSequence>>> {
        let full = self.turn_sealed_of(timeline, index)?;
        let layout = &self.turn(timeline, index)?.content.layout;
        let half = window_sealed_tokens(
            &full,
            layout.user_content_start() as usize,
            layout.user_content_end() as usize,
        );
        Some(Arc::new(half))
    }

    /// Token ids of the turn's *assistant-response body* `[asst_start, total)`.
    /// The compression path prefills the assistant half as text rather than
    /// injecting it (its assistant-role K/V are incoherent in the
    /// compression's user-input frame); the user half is injected instead
    /// (see [`Self::turn_user_sealed_half`]).
    pub fn turn_assistant_token_ids(
        &self,
        timeline: TimelineId,
        index: TurnIndex,
    ) -> Option<Vec<u32>> {
        let toks = self.token_ids_of(timeline, index);
        let layout = &self.turn(timeline, index)?.content.layout;
        let total = toks.len();
        let start = (layout.assistant_content_start() as usize).min(total);
        Some(toks[start..total].to_vec())
    }

    /// Token ids of the turn's *user-input body* `[user_start, user_end)`. The
    /// compression path text-prefills this alongside the assistant half so each
    /// half-summary is grounded in the full exchange rather than confabulating
    /// from one half in isolation (see [`Self::turn_assistant_token_ids`]).
    pub fn turn_user_token_ids(&self, timeline: TimelineId, index: TurnIndex) -> Option<Vec<u32>> {
        let toks = self.token_ids_of(timeline, index);
        let layout = &self.turn(timeline, index)?.content.layout;
        let total = toks.len();
        let start = (layout.user_content_start() as usize).min(total);
        let end = (layout.user_content_end() as usize).min(total).max(start);
        Some(toks[start..end].to_vec())
    }

    /// Turn token count — pinned bytes the seal recorded.
    pub fn turn_token_count_of(&self, timeline: TimelineId, index: TurnIndex) -> usize {
        self.turn(timeline, index)
            .map_or(0, |e| e.content.token_count)
    }

    /// Whether this turn was generated with thinking suppressed (the
    /// `/no_think` dial active at submit).  The projection re-injects the
    /// `/no_think` soft-switch into this turn's user opener when it re-renders
    /// the turn as history.  `false` for unknown turns.
    pub fn turn_no_think(&self, timeline: TimelineId, index: TurnIndex) -> bool {
        self.turn(timeline, index)
            .is_some_and(|e| e.content.layout.no_think())
    }

    pub fn turn_count(&self, timeline: TimelineId) -> u32 {
        self.timelines
            .get(&timeline)
            .map_or(0, |t| t.turns.len() as u32)
    }

    /// The turn on `timeline` whose decl gather-scope tags contain `tag`, if any.
    /// Backs [`ContentResolver::turn_with_tag`] — used to resolve a group's
    /// declared `default` member (e.g. the repo_map workspace-root cluster,
    /// tagged `"."`). Scans the stream decls (as `belief_gallery` does); scoped
    /// to `timeline` because a group is shared across conversations. `tag` is
    /// expected to identify a unique turn.
    pub fn turn_with_tag(&self, timeline: TimelineId, tag: &str) -> Option<TurnIndex> {
        self.all_streams().find_map(|(_sid, e)| {
            let Some(StreamDecl::Turn(d)) = e.decl.as_ref() else {
                return None;
            };
            (d.timeline_id == timeline.raw() && d.tags.iter().any(|t| t == tag))
                .then(|| TurnIndex(d.turn_index))
        })
    }

    /// Corpus size for a conversation — `timeline`'s turn tokens plus the
    /// shared section (workspace) tokens. This is the denominator the GUI shows
    /// as "materialized M / N tokens": the size of the unbounded store this
    /// projection draws from, against which its materialized subset is compared.
    ///
    /// O(1) — served from the maintained running counters, not an O(corpus)
    /// re-sum (which would be called on every reprojection during decode and
    /// scale with depth, defeating the engine's O(1) premise).
    pub fn total_token_count(&self, timeline: TimelineId) -> usize {
        self.timeline_token_totals
            .get(&timeline)
            .copied()
            .unwrap_or(0)
            + self.section_token_total
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
        // Stamp creation order the first time this timeline gets a conv_id.
        // During recovery this runs in redo-log (creation) order; live, a new
        // conversation's first Label bumps the counter last → highest rank.
        let stamp = self.conv_order_counter + 1;
        if let Some(entry) = self.timelines.get_mut(&timeline) {
            let first = entry.conv_id.is_none();
            entry.conv_id = Some(conv_id.to_string());
            if first {
                entry.order = stamp;
                self.conv_order_counter = stamp;
            }
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
    pub fn known_conversations(&self) -> Vec<(TimelineId, String, String, bool, u64)> {
        self.timelines
            .iter()
            .filter_map(|(tl, entry)| {
                let conv_id = entry.conv_id.clone()?;
                let label = entry.label.clone().unwrap_or_default();
                Some((*tl, conv_id, label, entry.archived, entry.order))
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
        self.timeline_token_totals.clear();
        self.section_token_total = 0;
        // Drop the whole decoded-signature memo — stream ids may be reused.
        self.sig_cache
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .clear();
    }

    /// Store a turn's projection-event record payload (opaque JSON bytes) on its
    /// stream runtime, keyed by the turn's `stream_id`. Called at write time
    /// and on redo-log replay.
    pub fn set_projection_events_blob(&mut self, stream_id: StreamId, payload: Vec<u8>) {
        self.streams.entry(stream_id).or_default().projection_events = Some(payload);
        // Incremental invalidation: evict only this stream's decoded seams so a
        // single seal doesn't force a full-gallery JSON re-parse on the next scan.
        self.seam_cache
            .get_mut()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&stream_id);
    }

    /// The stored projection-event record payload for a turn, if any.
    pub fn projection_events_blob(&self, timeline: TimelineId, index: TurnIndex) -> Option<&[u8]> {
        self.streams
            .get(&turn_stream_id(timeline.raw(), index.0))
            .and_then(|s| s.projection_events.as_deref())
    }

    /// Cache a turn's encoded wide-Q signature window, last-writer-wins.
    pub fn set_wide_q_sigs_blob(&mut self, stream_id: StreamId, payload: Vec<u8>) {
        self.streams.entry(stream_id).or_default().wide_q_sigs = Some(payload);
        // Incremental invalidation: evict only this stream's decoded window so a
        // single seal doesn't force a full-gallery re-decode on the next scan.
        self.evict_decoded_wide_sig(stream_id);
    }

    /// The stored wide-Q signature window payload for a turn, if any.
    pub fn wide_q_sigs_blob(&self, timeline: TimelineId, index: TurnIndex) -> Option<&[u8]> {
        self.streams
            .get(&turn_stream_id(timeline.raw(), index.0))
            .and_then(|s| s.wide_q_sigs.as_deref())
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
        sealed_gpu: Arc<Vec<SealedSequence>>,
        migrate_to_cpu: impl FnOnce(&[SealedSequence]) -> candle::Result<Vec<SealedSequence>>,
        tokens: Arc<Vec<u32>>,
    ) -> candle::Result<()> {
        let sealed_cpu = migrate_to_cpu(&sealed_gpu)?;
        let residence = self.alloc_residence(stream_id, None);
        let entry = SectionEntryData {
            token_count,
            block_range: (0, 0),
            tokens,
            residence,
        };
        let prev = self
            .sections
            .insert(section, entry)
            .map_or(0, |e| e.token_count);
        self.section_token_total = self.section_token_total + token_count - prev;
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
    pub fn restore_section(
        &mut self,
        section: SectionId,
        stream_id: StreamId,
        token_count: usize,
        sealed_hot: Vec<SealedSequence>,
        cold: Vec<StoredSequence>,
        tokens: Arc<Vec<u32>>,
    ) {
        let residence = self.alloc_residence(stream_id, None);
        self.residence[residence.0].cold = Some(cold);
        let entry = SectionEntryData {
            token_count,
            block_range: (0, 0),
            tokens,
            residence,
        };
        let prev = self
            .sections
            .insert(section, entry)
            .map_or(0, |e| e.token_count);
        self.section_token_total = self.section_token_total + token_count - prev;
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
    /// Reverse of [`Self::section_id_for_debug_name`]: the symbolic `debug_name`
    /// for a `SectionId` (tool name, `system.frame`, …). `SectionId` → residence →
    /// `stream_id` → `PromptSection.debug_name`. Used by the promote tracker.
    pub fn section_debug_name(&self, id: SectionId) -> Option<String> {
        let entry = self.sections.get(&id)?;
        let stream_id = self.residence[entry.residence.0].stream_id;
        self.all_streams().find_map(|(sid, entry)| {
            if sid != stream_id {
                return None;
            }
            match entry.decl.as_ref()? {
                StreamDecl::PromptSection(s) => Some(s.debug_name.clone()),
                _ => None,
            }
        })
    }

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
/// Enumerates **every** active timeline in a group, so a group holding many
/// conversations (`code_reading` declares one per file) is fully visible rather
/// than collapsing to the first-registered timeline.
///
/// This impl has **no projection target**, so it applies **no sibling-timeline
/// masking**: it surfaces every conversation in a group, including what would be
/// the target group. A target-masked projection (where a live slot must NOT see
/// sibling conversations of the same shape) MUST project through [`TargetedRead`]
/// — this bare impl is for structural / test callers only.
impl ContentResolver for Substrate {
    fn group_turns(&self, group: GroupId) -> Vec<TurnKey> {
        self.active_timelines_for_group(group)
            .flat_map(|tl| {
                (0..Substrate::turn_count(self, tl)).map(move |i| TurnKey::new(tl, TurnIndex(i)))
            })
            .collect()
    }

    fn turn_token_count(&self, turn: TurnKey) -> usize {
        self.turn(turn.timeline, turn.index)
            .map_or(0, |e| e.content.token_count)
    }

    fn turn_score(&self, _turn: TurnKey) -> f32 {
        // Bare substrate has no attached scores; pair via ScoredSubstrate
        // or read through Conversation::read_scored to see non-zero values.
        0.0
    }

    fn turn_origin(&self, turn: TurnKey) -> Option<LayerId> {
        let (layer, _) = self.timeline_target(turn.timeline)?;
        Some(layer)
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        self.sections.get(&section).map_or(0, |e| e.token_count)
    }

    fn section_score(&self, _section: SectionId) -> f32 {
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

/// Shared body of [`ContentResolver::summary_tree_select`] — the score-density
/// pick: build the timeline's summary forest, stamp each node with its
/// provenance turn score, and run [`select_dense`], which selects the most
/// relevant nodes (with recency anchoring/decay) that fit the window. Returns
/// the picked turns with their selection origins, each carrying its effective
/// provenance score for the diagnostics panel. `None` when the timeline has no
/// summary nodes, in which case the projection falls through to the rule-based
/// selector (which excludes summary turns). Used by the production
/// [`SubstrateRead`] resolver and the test-only [`ScoredSubstrate`].
fn select_summary_tree(
    substrate: &Substrate,
    scores: &ProjectionScores,
    timeline: TimelineId,
    budget: u32,
) -> Option<Vec<(TurnIndex, SelectionOrigin, f32)>> {
    // No summary nodes yet → fall through to the rule-based path.
    if !substrate.has_summary_nodes(timeline) {
        return None;
    }
    let tree = substrate.build_summary_tree_in_memory(timeline);
    if tree.is_empty() {
        return None;
    }
    let mut node_scores: ahash::AHashMap<NodeId, f32> = ahash::AHashMap::default();
    for id in tree.all_ids() {
        let idx = TurnIndex(id.0);
        // A tree node without a backing substrate turn is an orphan (redo-log
        // TreeMetadata whose matching TurnDecl never landed). It can't be
        // elevated, so exclude it from the selection that flows into the
        // projection / elevate path.
        if substrate.turn(timeline, idx).is_none() {
            continue;
        }
        node_scores.insert(id, scores.turn(timeline, idx));
    }
    let cfg = RecencyConfig::default();
    let sel = select_dense(&tree, &node_scores, cfg, budget);
    // Convert (NodeId, SelectionOrigin) pairs back to TurnIndex, post-filtering
    // orphan NodeIds `select_dense` may have walked in via the tree shape.
    let out: Vec<_> = sel
        .selected
        .iter()
        .zip(sel.origins.iter())
        .filter_map(|(id, origin)| {
            let idx = TurnIndex(id.0);
            substrate.turn(timeline, idx)?;
            let eff = sel.effective_scores.get(id).copied().unwrap_or(0.0);
            Some((idx, *origin, eff))
        })
        .collect();
    Some(out)
}

impl<'a> std::ops::Deref for ScoredSubstrate<'a> {
    type Target = Substrate;
    fn deref(&self) -> &Substrate {
        self.substrate
    }
}

/// [`ContentResolver`] impl over a `(Substrate, ProjectionScores)` pair.
///
/// Enumerates every active timeline in a group, so a multi-conversation group is
/// scored and projected in full rather than collapsing to the first timeline.
/// Like the bare [`Substrate`] impl, it carries **no target** and applies **no
/// sibling-timeline masking** — a target-masked projection routes through
/// [`TargetedRead`]; this pairing is for structural / test callers.
impl<'a> ContentResolver for ScoredSubstrate<'a> {
    fn group_turns(&self, group: GroupId) -> Vec<TurnKey> {
        self.substrate
            .active_timelines_for_group(group)
            .flat_map(|tl| {
                (0..Substrate::turn_count(self.substrate, tl))
                    .map(move |i| TurnKey::new(tl, TurnIndex(i)))
            })
            .collect()
    }

    fn turn_token_count(&self, turn: TurnKey) -> usize {
        self.substrate
            .turn(turn.timeline, turn.index)
            .map_or(0, |e| e.content.token_count)
    }

    fn turn_score(&self, turn: TurnKey) -> f32 {
        if self.substrate.turn(turn.timeline, turn.index).is_none() {
            return 0.0;
        }
        self.scores.turn(turn.timeline, turn.index)
    }

    fn turn_origin(&self, turn: TurnKey) -> Option<LayerId> {
        let (layer, _) = self.substrate.timeline_target(turn.timeline)?;
        Some(layer)
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        self.substrate
            .sections
            .get(&section)
            .map_or(0, |e| e.token_count)
    }

    fn section_score(&self, section: SectionId) -> f32 {
        if !self.substrate.sections.contains_key(&section) {
            return 0.0;
        }
        self.scores.section(section)
    }

    fn summary_tree_select(
        &self,
        timeline: TimelineId,
        budget: u32,
    ) -> Option<Vec<(TurnIndex, SelectionOrigin, f32)>> {
        select_summary_tree(self.substrate, self.scores, timeline, budget)
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
    pub fn turn_score_for_timeline(&self, timeline: TimelineId, index: TurnIndex) -> f32 {
        if self.guard.turn(timeline, index).is_none() {
            return 0.0;
        }
        self.scores_or_empty().turn(timeline, index)
    }
}

impl<'a> std::ops::Deref for SubstrateRead<'a> {
    type Target = Substrate;
    fn deref(&self) -> &Substrate {
        &self.guard
    }
}

/// Untargeted read guard: enumerates every active timeline in a group and applies
/// **no** sibling-timeline masking (it has no target). Wrap in [`TargetedRead`]
/// for a target-masked projection; this bare guard is for structural / test reads.
impl<'a> ContentResolver for SubstrateRead<'a> {
    fn group_turns(&self, group: GroupId) -> Vec<TurnKey> {
        self.guard
            .active_timelines_for_group(group)
            .flat_map(|tl| {
                (0..Substrate::turn_count(&self.guard, tl))
                    .map(move |i| TurnKey::new(tl, TurnIndex(i)))
            })
            .collect()
    }

    fn turn_token_count(&self, turn: TurnKey) -> usize {
        self.guard
            .turn(turn.timeline, turn.index)
            .map_or(0, |e| e.content.token_count)
    }

    fn turn_score(&self, turn: TurnKey) -> f32 {
        if self.guard.turn(turn.timeline, turn.index).is_none() {
            return 0.0;
        }
        self.scores_or_empty().turn(turn.timeline, turn.index)
    }

    fn turn_origin(&self, turn: TurnKey) -> Option<LayerId> {
        let (layer, _) = self.guard.timeline_target(turn.timeline)?;
        Some(layer)
    }

    /// Searches every active timeline in the group — a group's declared `default`
    /// member (e.g. the workspace-root cluster tagged `"."`) can live in any of
    /// its conversations, not just the first.
    fn turn_with_tag(&self, group: GroupId, tag: &str) -> Option<TurnKey> {
        self.guard.active_timelines_for_group(group).find_map(|tl| {
            self.guard
                .turn_with_tag(tl, tag)
                .map(|idx| TurnKey::new(tl, idx))
        })
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        self.guard
            .sections
            .get(&section)
            .map_or(0, |e| e.token_count)
    }

    fn section_score(&self, section: SectionId) -> f32 {
        if !self.guard.sections.contains_key(&section) {
            return 0.0;
        }
        self.scores_or_empty().section(section)
    }

    fn summary_tree_select(
        &self,
        timeline: TimelineId,
        budget: u32,
    ) -> Option<Vec<(TurnIndex, SelectionOrigin, f32)>> {
        select_summary_tree(&self.guard, self.scores_or_empty(), timeline, budget)
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
    use crate::persistence::segment::FIRST_SEGMENT;
    use crate::persistence::streams::TurnDecl;
    use crate::projection::{
        GroupId, LayerId, ProjectionTarget, SectionId, TimelineAllocator, TimelineId,
    };
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

    /// Regression: two conversations (timelines) in the SAME group is the case
    /// that lost history on reproject.  `timelines_for_group(group).next()`
    /// returns the FIRST-registered timeline, so a turn in the SECOND
    /// conversation was looked up under the first and dropped (`turn_tier_state`
    /// `None` → the slot rebuilt with `turns=0`).  `resolve_turn_timeline` pins
    /// the target group to the conversation actually being projected, so the
    /// second conversation's turns survive.  Every turn-timeline call site routes
    /// through it, so the SubmitTurn / reproject / inject paths can't disagree.
    #[test]
    fn resolve_turn_timeline_pins_target_group_to_its_own_timeline() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let first = alloc.next();
        let second = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(first, layer, group);
        sub.register_timeline(second, layer, group);

        // A turn lives ONLY in the second conversation.
        let idx = sub.append_with_blocks(second, 10, 0, 1);
        let target = |tl| ProjectionTarget {
            layer,
            group,
            timeline: tl,
        };

        // Registration order makes `.next()` resolve to `first` — the old bug.
        assert_eq!(sub.timelines_for_group(group).next(), Some(first));

        // The fix: a target pinned to the second conversation resolves to it, not
        // the first-registered timeline.
        assert_eq!(
            sub.resolve_turn_timeline(Some(target(second)), group),
            Some(second)
        );
        assert_eq!(
            sub.resolve_turn_timeline(Some(target(first)), group),
            Some(first)
        );

        // And that's what keeps the turn: under the correctly-resolved timeline
        // the second conversation's turn is found (kept); under the timeline the
        // old `.next()` picked it does not exist (dropped — the history loss).
        let resolved = sub
            .resolve_turn_timeline(Some(target(second)), group)
            .unwrap();
        assert!(
            sub.turn_tier_state(resolved, idx).is_some(),
            "turn is kept under the correctly-resolved timeline"
        );
        assert!(
            sub.turn_tier_state(first, idx).is_none(),
            "the old `.next()` timeline drops the second conversation's turn"
        );

        // A non-target group has a single timeline, so it still falls back to
        // registration order (the target only pins the target's own group).
        let other_group = GroupId::for_test(2);
        let other_tl = alloc.next();
        sub.register_timeline(other_tl, layer, other_group);
        assert_eq!(
            sub.resolve_turn_timeline(Some(target(second)), other_group),
            Some(other_tl)
        );

        // No target (utility pass) falls back to registration order everywhere.
        assert_eq!(sub.resolve_turn_timeline(None, group), Some(first));
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
        };
        sub.set_tree_meta(timeline, idx, meta.clone());
        assert_eq!(sub.tree_meta_of(timeline, idx), Some(&meta));
        let leaves = sub.summary_leaves_chrono(timeline);
        assert_eq!(leaves, vec![idx]);
    }

    /// Three SoT leaves carry into one ternary SoS; the SoS is the sole peak.
    #[test]
    fn peaks_of_derives_forest() {
        let (_, _, timeline, mut sub) = make_timeline();
        let mut leaves = Vec::new();
        for n in 0..MERGE_FANOUT as u32 {
            let idx = sub.append_with_blocks(timeline, 10, n as u64, n as u64 + 1);
            sub.set_tree_meta(
                timeline,
                idx,
                TreeNodeMeta {
                    kind: TurnKind::SummaryOfTurns,
                    children: vec![idx],
                    tree_height: 1,
                },
            );
            leaves.push(idx);
        }
        // Before the SoS exists, all MERGE_FANOUT leaves are peaks; reconcile wants
        // to build the SoS over them.
        assert_eq!(sub.peaks_of(timeline).len(), MERGE_FANOUT);
        let next = sub.reconcile_next(timeline).expect("an SoS to build");
        assert_eq!(next, leaves);
        // Record the SoS; now it is the single peak and the forest is whole.
        let sos =
            sub.append_with_blocks(timeline, 10, MERGE_FANOUT as u64, MERGE_FANOUT as u64 + 1);
        sub.set_tree_meta(
            timeline,
            sos,
            TreeNodeMeta {
                kind: TurnKind::SummaryOfSummaries,
                children: leaves.clone(),
                tree_height: 2,
            },
        );
        assert_eq!(sub.peaks_of(timeline), vec![(sos, 2)]);
        assert_eq!(sub.reconcile_next(timeline), None);
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

    /// `mark_for_reconcile` purges non-canonical (old binary) SoS meta and arms
    /// the reconcile hint so the ternary nodes get rebuilt.
    #[test]
    fn mark_for_reconcile_purges_noncanonical_sos() {
        let (_, _, timeline, mut sub) = make_timeline();
        let leaf_a = sub.append_with_blocks(timeline, 10, 0, 1);
        let leaf_b = sub.append_with_blocks(timeline, 10, 1, 2);
        for leaf in [leaf_a, leaf_b] {
            sub.set_tree_meta(
                timeline,
                leaf,
                TreeNodeMeta {
                    kind: TurnKind::SummaryOfTurns,
                    children: vec![leaf],
                    tree_height: 1,
                },
            );
        }
        // A legacy binary SoS (2 children) — not canonical under MERGE_FANOUT.
        let bin = sub.append_with_blocks(timeline, 10, 2, 3);
        sub.set_tree_meta(
            timeline,
            bin,
            TreeNodeMeta {
                kind: TurnKind::SummaryOfSummaries,
                children: vec![leaf_a, leaf_b],
                tree_height: 2,
            },
        );
        assert!(sub.tree_meta_of(timeline, bin).is_some());
        sub.mark_for_reconcile(timeline);
        // Binary SoS purged; reconcile armed; the two leaves survive.
        assert!(sub.tree_meta_of(timeline, bin).is_none());
        assert!(sub.needs_reconcile(timeline));
        assert_eq!(sub.peaks_of(timeline).len(), 2);
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
                    layout: TurnLayout::from_flat_grid(
                        0,
                        0,
                        0,
                        3,
                        0,
                        0,
                        String::new(),
                        Some("hello".to_string()),
                        false,
                    ),
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

    /// `total_token_count(timeline)` is this conversation's turn tokens plus the
    /// shared section tokens, served from the O(1) maintained counters.
    #[test]
    fn total_token_count_sums_turns_and_sections() {
        let (_, _, timeline, mut sub) = make_timeline();
        for tc in [30usize, 12] {
            sub.append_complete(
                timeline,
                TurnPartWrite {
                    token_count: tc,
                    sealed_gpu: Some(Arc::new(vec![])),
                    ..Default::default()
                },
                identity_migrate,
            )
            .unwrap();
        }
        // A 10-token ingested section adds to the total.
        sub.set_section_full(
            SectionId::new(7),
            StreamId::default(),
            10,
            Arc::new(vec![minimal_sealed_layer()]),
            identity_migrate,
            Arc::new(vec![1u32, 2, 3]),
        )
        .unwrap();
        assert_eq!(sub.total_token_count(timeline), 30 + 12 + 10);
    }

    /// The maintained counter handles section re-ingest (overwrite, not add)
    /// and turn extension without drifting.
    #[test]
    fn total_token_count_handles_overwrite_and_extend() {
        let (_, _, timeline, mut sub) = make_timeline();
        sub.append_complete(
            timeline,
            TurnPartWrite {
                token_count: 20,
                sealed_gpu: Some(Arc::new(vec![])),
                ..Default::default()
            },
            identity_migrate,
        )
        .unwrap();
        sub.set_section_full(
            SectionId::new(9),
            StreamId::default(),
            100,
            Arc::new(vec![minimal_sealed_layer()]),
            identity_migrate,
            Arc::new(vec![1u32]),
        )
        .unwrap();
        assert_eq!(sub.total_token_count(timeline), 120);
        // Re-ingest the SAME section smaller — replaces its 100 with 60.
        sub.set_section_full(
            SectionId::new(9),
            StreamId::default(),
            60,
            Arc::new(vec![minimal_sealed_layer()]),
            identity_migrate,
            Arc::new(vec![1u32]),
        )
        .unwrap();
        assert_eq!(sub.total_token_count(timeline), 80);
        // Extend the turn by 5 tokens.
        sub.extend_turn(timeline, TurnIndex(0), 5, 2);
        assert_eq!(sub.total_token_count(timeline), 85);
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

    /// Helper: install a warm residence with a known byte_size. Returns the
    /// residence index. Used by the purge tests below to drive the warm LRU
    /// without going through the (CUDA-only) migrate path.
    ///
    /// `cold_backed` mirrors the realistic post-persist state: once a turn's
    /// warm→cold write lands, its warm copy is purgeable (cold is the durable
    /// backup). When `false`, the residence is warm-only with NO lower-tier
    /// backup — the purge must refuse to drop it (its K/V would be lost).
    fn install_warm_only(
        sub: &mut Substrate,
        timeline: TimelineId,
        bytes: u64,
        cold_backed: bool,
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
        if cold_backed {
            // A non-empty cold marker (its chunks are irrelevant to the purge —
            // only `cold.is_some()` gates the drop).
            sub.install_cold(
                residence,
                vec![StoredSequence {
                    chunks: Vec::new(),
                    token_count: 0,
                }],
            );
        }
        residence
    }

    /// Ample headroom → purge is a no-op even with warm-resident slots.
    #[test]
    fn purge_with_ample_headroom_is_noop() {
        let (_, _, timeline, mut sub) = make_timeline();
        install_warm_only(&mut sub, timeline, 1_000_000, true);
        install_warm_only(&mut sub, timeline, 1_000_000, true);

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
        let a = install_warm_only(&mut sub, timeline, 500_000_000, true);
        let b = install_warm_only(&mut sub, timeline, 500_000_000, true);

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

    /// A warm residence with NO lower-tier backup (its cold write hasn't landed,
    /// or failed) is the turn's ONLY copy — the purge must refuse to drop it even
    /// under pressure, or the K/V is lost. The cold-backed victim is still freed.
    #[test]
    fn purge_keeps_warm_without_backup() {
        let (_, _, timeline, mut sub) = make_timeline();
        // a installed first → slides to the LRU tail → `pop_back` sees it first.
        // a has no cold backup; b does.
        let a = install_warm_only(&mut sub, timeline, 500_000_000, false);
        let b = install_warm_only(&mut sub, timeline, 500_000_000, true);

        // Tight headroom that would otherwise purge both: 8 GB total, 1 GB
        // available, incoming 0, threshold 2 GiB. projected = 1 GB < 2 GiB.
        let r = sub.purge_warm_to_target(0, 1_000_000_000, 8 * 1000 * 1000 * 1000);
        // Only the cold-backed residence (b) is dropped; the un-backed one (a)
        // is preserved and stays warm-resident.
        assert_eq!(r.count, 1, "only the cold-backed warm is purgeable");
        assert_eq!(r.bytes, 500_000_000);
        assert!(
            sub.residence[a.0].warm.is_some(),
            "un-backed warm must be kept (its K/V would otherwise be lost)"
        );
        assert!(
            sub.residence[b.0].warm.is_none(),
            "cold-backed warm dropped"
        );
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
        install_warm_only(&mut sub, timeline, 2_000_000_000, true);

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

    /// Targeted demotion drops the hot copy of exactly the NAMED turns (keeping
    /// warm) and leaves every other hot turn resident — the inverse of the
    /// keep-set eviction. A turn with no warm copy is left hot (dropping it would
    /// lose its K/V), and re-demoting an already-warm-only turn is a no-op.
    #[test]
    fn demote_turns_to_warm_drops_named_keeps_warm_and_rest() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (a_idx, a) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (b_idx, b) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (c_idx, c) = install_hot_and_warm(&mut sub, timeline, 100_000_000);

        let demoted = sub.demote_turns_to_warm(&[
            TurnKey {
                timeline,
                index: a_idx,
            },
            TurnKey {
                timeline,
                index: b_idx,
            },
        ]);
        assert_eq!(demoted, 2, "a + b demoted");
        assert!(sub.residence[a.0].hot.is_none(), "a hot dropped");
        assert!(
            sub.residence[a.0].warm.is_some(),
            "a warm KEPT (cheap reload)"
        );
        assert!(sub.residence[b.0].hot.is_none(), "b hot dropped");
        assert!(sub.residence[b.0].warm.is_some(), "b warm KEPT");
        assert!(sub.residence[c.0].hot.is_some(), "c (unnamed) still hot");

        // Idempotent: a is now warm-only, so re-demoting it frees nothing.
        assert_eq!(
            sub.demote_turns_to_warm(&[TurnKey {
                timeline,
                index: a_idx
            }]),
            0,
            "already warm-only → no-op"
        );

        // A hot turn with NO warm copy is left hot — dropping it would lose K/V.
        sub.residence[c.0].warm = None;
        assert_eq!(
            sub.demote_turns_to_warm(&[TurnKey {
                timeline,
                index: c_idx
            }]),
            0,
            "hot-without-warm is not demotable"
        );
        assert!(
            sub.residence[c.0].hot.is_some(),
            "c hot-without-warm untouched"
        );
    }

    /// Gentle-early ingest demotion drops the hot copy of an ingest timeline's
    /// sealed, warm-backed turns EXCEPT the newest `keep_recent` (the rolling
    /// window), keeping warm; a hot-without-warm turn (the active writer) is left
    /// hot.
    #[test]
    fn demote_cold_ingest_keeps_window_and_warm() {
        let (_, _, timeline, mut sub) = make_timeline();
        // Five sealed, hot+warm turns, oldest→newest.
        let (_, t0) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, t1) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, t2) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, t3) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, t4) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        // The newest turn is the active writer: hot-only (no warm copy yet).
        sub.residence[t4.0].warm = None;

        let mut ingest = std::collections::HashSet::new();
        ingest.insert(timeline);

        // keep_recent = 2 → protect t3, t4 (the rolling window); no live working
        // set; unbounded target → demote the whole LRU tail t0, t1, t2 (all
        // warm-backed), oldest-first.
        let report = sub.demote_cold_ingest(&ingest, &[], &[], 2, u64::MAX);
        assert_eq!(report.count, 3, "t0,t1,t2 demoted; t3,t4 window kept");
        assert_eq!(report.bytes, 300_000_000);
        for r in [t0, t1, t2] {
            assert!(
                sub.residence[r.0].hot.is_none(),
                "older ingest turn demoted"
            );
            assert!(
                sub.residence[r.0].warm.is_some(),
                "warm KEPT (cheap reload)"
            );
        }
        assert!(sub.residence[t3.0].hot.is_some(), "window turn stays hot");
        assert!(
            sub.residence[t4.0].hot.is_some(),
            "hot-without-warm writer stays hot"
        );

        // Idempotent: the demoted turns are now warm-only, so a second pass frees
        // nothing.
        assert_eq!(
            sub.demote_cold_ingest(&ingest, &[], &[], 2, u64::MAX).count,
            0,
            "already demoted"
        );
    }

    /// `target_bytes` bounds the LRU walk: with a target that covers only two
    /// turns, the demote stops after shedding the two oldest and leaves the rest
    /// hot — relief to the watermark, never the whole working set.
    #[test]
    fn demote_cold_ingest_stops_at_target() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_, t0) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, t1) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, t2) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, t3) = install_hot_and_warm(&mut sub, timeline, 100_000_000);

        let mut ingest = std::collections::HashSet::new();
        ingest.insert(timeline);

        // keep_recent = 0 (nothing protected), target = 150 MB → the walk frees t0
        // (100 MB, still < target) then t1 (200 MB, ≥ target) and stops. t2, t3 stay
        // hot.
        let report = sub.demote_cold_ingest(&ingest, &[], &[], 0, 150_000_000);
        assert_eq!(report.count, 2, "only the two oldest shed to meet target");
        assert_eq!(report.bytes, 200_000_000);
        assert!(sub.residence[t0.0].hot.is_none(), "oldest demoted");
        assert!(sub.residence[t1.0].hot.is_none(), "second-oldest demoted");
        assert!(
            sub.residence[t2.0].hot.is_some(),
            "target met → t2 kept hot"
        );
        assert!(
            sub.residence[t3.0].hot.is_some(),
            "target met → t3 kept hot"
        );
    }

    /// A timeline NOT in the ingest set is never touched — the demotion is scoped
    /// to append-only ingest timelines, so chat working sets are safe.
    #[test]
    fn demote_cold_ingest_ignores_non_ingest_timeline() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_, a) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, b) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, c) = install_hot_and_warm(&mut sub, timeline, 100_000_000);

        // Empty ingest set → this timeline is not an ingest timeline, so even an
        // unbounded target frees nothing.
        let report =
            sub.demote_cold_ingest(&std::collections::HashSet::new(), &[], &[], 0, u64::MAX);
        assert_eq!(report.count, 0);
        for r in [a, b, c] {
            assert!(
                sub.residence[r.0].hot.is_some(),
                "non-ingest timeline untouched"
            );
        }
    }

    /// `install_cold` frees a section's hot VRAM only when its residence is
    /// flagged `evict_when_cold` (the prefix-transparent collection-member
    /// offload path); a plain/boundary section keeps its hot copy resident.
    #[test]
    fn install_cold_frees_section_hot_only_when_evict_flagged() {
        let mut sub = Substrate::new();
        let install = |sub: &mut Substrate, id: u32| {
            let sealed = Arc::new(vec![minimal_sealed_layer()]);
            sub.set_section_full(
                SectionId::new(id),
                StreamId::default(),
                32,
                sealed,
                identity_migrate,
                Arc::new(vec![]),
            )
            .unwrap();
            sub.section_residence(SectionId::new(id)).unwrap()
        };
        let cold = || {
            vec![StoredSequence {
                chunks: vec![StoredChunk {
                    log_offset: 0,
                    record_len: 1024,
                    token_count: 32,
                }],
                token_count: 32,
            }]
        };

        let member = install(&mut sub, 1);
        let boundary = install(&mut sub, 2);
        assert!(sub.residence[member.0].hot.is_some());
        assert!(sub.residence[boundary.0].hot.is_some());

        // Only the collection member is flagged for offload-on-persist.
        sub.mark_section_evict_when_cold(member);

        sub.install_cold(member, cold());
        sub.install_cold(boundary, cold());

        // Flagged member: cold copy lands AND hot VRAM is freed.
        assert!(
            sub.residence[member.0].hot.is_none(),
            "flagged member: VRAM offloaded once cold lands"
        );
        assert!(sub.residence[member.0].cold.is_some());
        // Boundary section: cold copy lands but it stays hot for the build.
        assert!(
            sub.residence[boundary.0].hot.is_some(),
            "boundary section stays hot"
        );
        assert!(sub.residence[boundary.0].cold.is_some());
    }

    /// A completed-ingest turn flagged `evict_when_cold` is fully evicted from
    /// BOTH resident tiers when its cold copy lands: `install_cold` drops hot
    /// AND warm and clears both LRUs, leaving it cold-only on NVMe. An unflagged
    /// turn keeps its resident copies.
    #[test]
    fn install_cold_fully_evicts_flagged_turn_from_both_tiers() {
        let (_, _, timeline, mut sub) = make_timeline();
        let (_, flagged) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let (_, kept) = install_hot_and_warm(&mut sub, timeline, 100_000_000);
        let cold = || {
            vec![StoredSequence {
                chunks: vec![StoredChunk {
                    log_offset: 0,
                    record_len: 1024,
                    token_count: 32,
                }],
                token_count: 32,
            }]
        };

        // One turn is a completed-ingest residence flagged for full eviction; the
        // other stays live. Both start hot + warm (on both LRUs).
        sub.residence[flagged.0].evict_when_cold = true;
        assert!(sub.hot_lru.contains(&flagged) && sub.warm_lru.contains(&flagged));

        sub.install_cold(flagged, cold());
        sub.install_cold(kept, cold());

        // Flagged: cold durable, and BOTH resident tiers freed + off both LRUs.
        assert!(sub.residence[flagged.0].cold.is_some(), "cold durable");
        assert!(sub.residence[flagged.0].hot.is_none(), "VRAM reclaimed");
        assert!(sub.residence[flagged.0].warm.is_none(), "RAM reclaimed");
        assert!(!sub.hot_lru.contains(&flagged), "off hot_lru");
        assert!(!sub.warm_lru.contains(&flagged), "off warm_lru");

        // Unflagged: cold lands but the turn stays fully resident.
        assert!(sub.residence[kept.0].cold.is_some());
        assert!(sub.residence[kept.0].hot.is_some(), "kept stays hot");
        assert!(sub.residence[kept.0].warm.is_some(), "kept stays warm");
    }

    /// `mark_timeline_evict_when_cold` flags every turn residence of exactly the
    /// named timeline (leaving other timelines' turns untouched) and reports the
    /// count; an unknown timeline is a no-op.
    #[test]
    fn mark_timeline_evict_when_cold_flags_every_turn_of_that_timeline() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let ingest = alloc.next();
        let live = alloc.next();
        let mut sub = Substrate::new();
        sub.register_timeline(ingest, layer, group);
        sub.register_timeline(live, layer, group);

        let (_, a) = install_hot_and_warm(&mut sub, ingest, 10);
        let (_, b) = install_hot_and_warm(&mut sub, ingest, 10);
        let (_, c) = install_hot_and_warm(&mut sub, live, 10);

        let n = sub.mark_timeline_evict_when_cold(ingest);
        assert_eq!(n, 2, "both ingest turns flagged");
        assert!(sub.residence[a.0].evict_when_cold);
        assert!(sub.residence[b.0].evict_when_cold);
        assert!(
            !sub.residence[c.0].evict_when_cold,
            "the other timeline's turn is untouched"
        );
        // Already-warm ingest turns get their hot copy dropped immediately (VRAM
        // reclaimed now) while warm is kept; the other timeline stays fully hot.
        assert!(sub.residence[a.0].hot.is_none() && sub.residence[a.0].warm.is_some());
        assert!(sub.residence[b.0].hot.is_none() && sub.residence[b.0].warm.is_some());
        assert!(!sub.hot_lru.contains(&a) && !sub.hot_lru.contains(&b));
        assert!(sub.residence[c.0].hot.is_some() && sub.residence[c.0].warm.is_some());

        // Unknown timeline → no-op.
        assert_eq!(sub.mark_timeline_evict_when_cold(alloc.next()), 0);
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
            },
        );

        // The reconstruct loop then restores turn 0.
        let idx = sub.restore_turn(
            timeline,
            TurnLayout::default(),
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
            TurnLayout::default(),
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
            TurnLayout::default(),
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
            TurnLayout::default(),
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

    /// Compaction moves every record to a new offset and re-walks the
    /// stream index; `refresh_cold_refs` must re-point each residence's
    /// `StoredSequence`s at the rebuilt index (same grid shape, new
    /// offsets), and must leave residences whose stream is absent from
    /// the active index untouched (inherited-log streams).
    #[test]
    fn refresh_cold_refs_repoints_residences_at_the_stream_index() {
        let (_, _, timeline, mut sub) = make_timeline();
        let stream_id = turn_stream_id(timeline.raw(), 0);
        let n_layers = 2usize;
        let chunks_per_layer = 2usize;

        // Pre-compaction stream index + matching cold refs.
        let old_loc = |flat: u64| ChunkLoc {
            segment: FIRST_SEGMENT,
            offset: 4096 + flat * 4096,
            payload_len: 100,
            record_size: 4096,
            token_count: if flat % 2 == 1 { 12 } else { 32 },
            format: 4,
        };
        for flat in 0..(n_layers * chunks_per_layer) as u64 {
            sub.apply_chunk_loc(stream_id, flat, old_loc(flat));
        }
        let cold: Vec<StoredSequence> = (0..n_layers)
            .map(|l| StoredSequence {
                chunks: (0..chunks_per_layer)
                    .map(|c| {
                        let loc = old_loc((l * chunks_per_layer + c) as u64);
                        StoredChunk {
                            log_offset: loc.offset,
                            record_len: loc.record_size,
                            token_count: loc.token_count as u16,
                        }
                    })
                    .collect(),
                token_count: 44,
            })
            .collect();
        let idx = sub.restore_turn(
            timeline,
            TurnLayout::default(),
            TokenBuffer::default(),
            44,
            Some(cold),
            0,
            chunks_per_layer as u64,
        );

        // An unrelated residence whose stream is NOT in the active
        // index (an inherited-log borrow) — must stay untouched.
        let other_timeline = TimelineId::from_raw(4242).unwrap();
        sub.register_timeline(other_timeline, LayerId::for_test(1), GroupId::for_test(1));
        let untouched = vec![StoredSequence {
            chunks: vec![StoredChunk {
                log_offset: 777_216,
                record_len: 4096,
                token_count: 32,
            }],
            token_count: 32,
        }];
        let other_idx = sub.restore_turn(
            other_timeline,
            TurnLayout::default(),
            TokenBuffer::default(),
            32,
            Some(untouched),
            0,
            1,
        );

        // "Compaction": every chunk record lands at a new offset with a
        // new padded size.
        let new_loc = |flat: u64| ChunkLoc {
            segment: FIRST_SEGMENT,
            offset: 100_000 + flat * 8192,
            payload_len: 100,
            record_size: 8192,
            token_count: old_loc(flat).token_count,
            format: 4,
        };
        for flat in 0..(n_layers * chunks_per_layer) as u64 {
            sub.apply_chunk_loc(stream_id, flat, new_loc(flat));
        }
        sub.refresh_cold_refs();

        let residence = sub.turn_residence(timeline, idx).unwrap();
        let cold = sub.residence[residence.0].cold.as_ref().unwrap();
        for (l, seq) in cold.iter().enumerate() {
            for (c, chunk) in seq.chunks.iter().enumerate() {
                let flat = (l * chunks_per_layer + c) as u64;
                assert_eq!(chunk.log_offset, new_loc(flat).offset);
                assert_eq!(chunk.record_len, 8192);
                assert_eq!(chunk.token_count, new_loc(flat).token_count as u16);
            }
        }
        let other_res = sub.turn_residence(other_timeline, other_idx).unwrap();
        let other_cold = sub.residence[other_res.0].cold.as_ref().unwrap();
        assert_eq!(
            other_cold[0].chunks[0].log_offset, 777_216,
            "residence without an active-index stream stays untouched"
        );
    }

    /// Replaying a coupling record makes the round-trip visible to the
    /// summariser and selector. Idempotent: the records carry no ordering, and a
    /// recovery walk may present the same one twice.
    #[test]
    fn turn_coupling_replays_into_the_timeline_set() {
        use crate::projection::{GroupId, LayerId, TimelineAllocator};
        let mut sub = Substrate::new();
        let alloc = TimelineAllocator::new();
        let timeline = alloc.next();
        sub.register_timeline(timeline, LayerId::for_test(1), GroupId::for_test(1));

        let payload = TurnCouplingPayload {
            timeline_id: timeline.raw(),
            from_turn: 4,
        };
        sub.apply_turn_coupling(&payload);
        sub.apply_turn_coupling(&payload); // duplicate — must not change the set
        assert_eq!(sub.couplings_of(timeline), [4u32].into_iter().collect());
    }

    /// A coupling naming a timeline that was never registered describes a
    /// round-trip that cannot exist — it must be dropped, not conjure an entry.
    #[test]
    fn turn_coupling_for_an_unknown_timeline_is_dropped() {
        let mut sub = Substrate::new();
        let alloc = crate::projection::TimelineAllocator::new();
        let timeline = alloc.next();
        sub.apply_turn_coupling(&TurnCouplingPayload {
            timeline_id: timeline.raw(),
            from_turn: 1,
        });
        assert!(sub.couplings_of(timeline).is_empty());
    }

    /// A timeline that never called a tool has no couplings, so every turn is its
    /// own exchange — the pre-coupling behaviour, unchanged.
    #[test]
    fn a_timeline_without_couplings_reports_an_empty_set() {
        use crate::projection::{GroupId, LayerId, TimelineAllocator};
        let mut sub = Substrate::new();
        let alloc = TimelineAllocator::new();
        let timeline = alloc.next();
        sub.register_timeline(timeline, LayerId::for_test(1), GroupId::for_test(1));
        assert!(sub.couplings_of(timeline).is_empty());
    }

    /// `tombstoned_stream_bytes` sums the on-disk record bytes of every
    /// tombstoned timeline's turn streams — the dead weight the
    /// header-keyed accounting can't attribute.
    #[test]
    fn tombstoned_stream_bytes_sums_dead_timelines() {
        let mut sub = Substrate::new();
        let decl_for = |tl: u64| {
            StreamDecl::Turn(TurnDecl {
                timeline_id: tl,
                turn_index: 0,
                turn_id_day: 0,
                turn_id_seq: 1,
                role: 1,
                block_start: 0,
                block_end: 1,
                layer_id: 1,
                group_id: 1,
                anchored_prefix: Vec::new(),
                view: Vec::new(),
                segments: Vec::new(),
                tags: Vec::new(),
            })
        };
        let dead_sid = turn_stream_id(7, 0);
        let live_sid = turn_stream_id(8, 0);
        for (sid, tl) in [(dead_sid, 7u64), (live_sid, 8u64)] {
            sub.apply_stream_decl(sid, decl_for(tl));
            sub.apply_chunk_loc(
                sid,
                0,
                ChunkLoc {
                    segment: FIRST_SEGMENT,
                    offset: 4096,
                    payload_len: 100,
                    record_size: 8192,
                    token_count: 32,
                    format: 4,
                },
            );
            sub.apply_tokens_loc(
                sid,
                RecordLoc {
                    segment: FIRST_SEGMENT,
                    offset: 20_480,
                    payload_len: 64,
                    record_size: 4096,
                },
            );
        }
        assert_eq!(sub.tombstoned_stream_bytes(), 0, "nothing tombstoned yet");
        sub.tombstone_timeline(TimelineId::from_raw(7).unwrap());
        assert_eq!(
            sub.tombstoned_stream_bytes(),
            8192 + 4096,
            "only the tombstoned timeline's chunk + tokens bytes count"
        );
    }

    /// `turn_with_tag` resolves a declared default member to a real turn: it
    /// matches a `TurnDecl.tags` entry, is scoped to the requested timeline (a
    /// group is shared across conversations), and returns `None` for an unknown
    /// tag.
    #[test]
    fn turn_with_tag_matches_scoped_to_timeline() {
        let mut sub = Substrate::new();
        let decl = |tl: u64, idx: u32, tags: &[&str]| {
            StreamDecl::Turn(TurnDecl {
                timeline_id: tl,
                turn_index: idx,
                turn_id_day: 0,
                turn_id_seq: idx + 1,
                role: 1,
                block_start: 0,
                block_end: 1,
                layer_id: 1,
                group_id: 1,
                anchored_prefix: Vec::new(),
                view: Vec::new(),
                segments: Vec::new(),
                tags: tags.iter().map(|t| t.to_string()).collect(),
            })
        };
        // Timeline 1: turn 0 is the repo-root cluster, turn 1 a code chunk.
        sub.apply_stream_decl(turn_stream_id(1, 0), decl(1, 0, &["repo_map", "."]));
        sub.apply_stream_decl(turn_stream_id(1, 1), decl(1, 1, &["code", "foo.rs"]));
        // Timeline 2: its own repo-root cluster at turn 0.
        sub.apply_stream_decl(turn_stream_id(2, 0), decl(2, 0, &["repo_map", "."]));

        let tl1 = TimelineId::from_raw(1).unwrap();
        let tl2 = TimelineId::from_raw(2).unwrap();

        assert_eq!(sub.turn_with_tag(tl1, "."), Some(TurnIndex(0)));
        assert_eq!(sub.turn_with_tag(tl1, "foo.rs"), Some(TurnIndex(1)));
        // Absent tag ⇒ None.
        assert_eq!(sub.turn_with_tag(tl1, "missing"), None);
        // Timeline scoping: tl2 resolves its own root, not tl1's; and tl1's
        // code tag does not leak into tl2.
        assert_eq!(sub.turn_with_tag(tl2, "."), Some(TurnIndex(0)));
        assert_eq!(sub.turn_with_tag(tl2, "foo.rs"), None);
    }

    /// The decoded-signature memo serves a stable `Arc` on repeat reads, and
    /// invalidates when the underlying blob changes (gallery generation bump) so
    /// a re-sealed turn never reads a stale decoded window.
    #[test]
    fn decoded_wide_sig_caches_and_invalidates_on_blob_change() {
        use crate::provenance::encode_wide_sigs;
        let mut sub = Substrate::new();
        let sid = turn_stream_id(1, 0);
        let sig_a = WideQSig {
            n_heads: 12,
            words: vec![0xAAAA_AAAA_AAAA_AAAA; 24],
        };
        sub.set_wide_q_sigs_blob(sid, encode_wide_sigs(std::slice::from_ref(&sig_a)));

        let w1 = sub.decoded_wide_sig(sid).expect("decoded");
        assert_eq!(w1.as_slice(), std::slice::from_ref(&sig_a));
        // Second read is served from the memo — same allocation.
        let w2 = sub.decoded_wide_sig(sid).expect("decoded");
        assert!(Arc::ptr_eq(&w1, &w2), "repeat read must hit the cache");

        // Overwriting the blob bumps the generation and invalidates the memo.
        let sig_b = WideQSig {
            n_heads: 12,
            words: vec![0x5555_5555_5555_5555; 24],
        };
        sub.set_wide_q_sigs_blob(sid, encode_wide_sigs(std::slice::from_ref(&sig_b)));
        let w3 = sub.decoded_wide_sig(sid).expect("decoded");
        assert_eq!(
            w3.as_slice(),
            std::slice::from_ref(&sig_b),
            "cache must re-decode after the blob changed"
        );
        assert!(!Arc::ptr_eq(&w1, &w3), "must not serve the stale window");

        // An absent stream (and an empty window) resolve to None.
        assert!(sub.decoded_wide_sig(turn_stream_id(1, 99)).is_none());
    }

    /// Invalidation is per-stream: rewriting one turn's sig evicts only that
    /// turn's decoded window and leaves every other memo intact — so a single
    /// seal never churns the whole gallery (the point of incremental eviction).
    #[test]
    fn decoded_wide_sig_eviction_is_per_stream() {
        use crate::provenance::encode_wide_sigs;
        let sig = |fill: u64| WideQSig {
            n_heads: 12,
            words: vec![fill; 24],
        };
        let mut sub = Substrate::new();
        let sid_a = turn_stream_id(1, 0);
        let sid_b = turn_stream_id(1, 1);
        sub.set_wide_q_sigs_blob(sid_a, encode_wide_sigs(&[sig(0xAAAA_AAAA_AAAA_AAAA)]));
        sub.set_wide_q_sigs_blob(sid_b, encode_wide_sigs(&[sig(0x5555_5555_5555_5555)]));

        // Warm both windows.
        let a1 = sub.decoded_wide_sig(sid_a).expect("a");
        let b1 = sub.decoded_wide_sig(sid_b).expect("b");

        // Rewrite A's blob — evicts A's memo only.
        sub.set_wide_q_sigs_blob(sid_a, encode_wide_sigs(&[sig(0xFFFF_FFFF_FFFF_FFFF)]));
        let a2 = sub.decoded_wide_sig(sid_a).expect("a");
        let b2 = sub.decoded_wide_sig(sid_b).expect("b");

        assert!(
            !Arc::ptr_eq(&a1, &a2),
            "A re-decoded after its own blob change"
        );
        assert!(
            Arc::ptr_eq(&b1, &b2),
            "B's memo must survive A's eviction — no whole-gallery churn"
        );
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
        let (tl, conv_id, label, archived, _order) = &convs[0];
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

    /// `order` is stamped in creation sequence on first `conv_id` and is stable:
    /// re-setting a conv_id doesn't re-stamp, and `live_conv_meta` (the compactor's
    /// Label source) emits in that order so a reload re-derives the same sequence
    /// instead of scrambling the sidebar on every compaction. Production timeline
    /// ids are content hashes with no time information, so this ordinal is the only
    /// creation-order signal.
    #[test]
    fn conv_order_stamps_creation_sequence_and_emits_stably() {
        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let mut sub = Substrate::new();

        let (a, b, c) = (alloc.next(), alloc.next(), alloc.next());
        for (tl, id) in [(a, "a"), (b, "b"), (c, "c")] {
            sub.register_timeline(tl, layer, group);
            sub.set_conv_id(tl, id);
        }

        let order_of = |sub: &Substrate, want: &str| {
            sub.known_conversations()
                .into_iter()
                .find(|(_, cid, ..)| cid == want)
                .map(|(_, _, _, _, o)| o)
                .expect("conversation present")
        };
        assert_eq!(order_of(&sub, "a"), 1);
        assert_eq!(order_of(&sub, "b"), 2);
        assert_eq!(order_of(&sub, "c"), 3);

        // Re-setting the same conv_id must not bump the counter or re-stamp.
        sub.set_conv_id(a, "a");
        assert_eq!(order_of(&sub, "a"), 1);
        assert_eq!(order_of(&sub, "c"), 3, "counter untouched by a re-set");

        // The compactor's Label source emits in creation order, so reload is stable.
        let emitted: Vec<String> = sub
            .live_conv_meta()
            .into_iter()
            .map(|(_, cid, ..)| cid)
            .collect();
        assert_eq!(emitted, ["a", "b", "c"]);
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

    /// A group holding MANY conversations must surface every one of them.
    ///
    /// `code_reading` declares one conversation per file, so this is the shape
    /// the coding agent actually runs. The resolver used to collapse a group to
    /// its first-registered timeline, which made every other file structurally
    /// unreachable — not scored by provenance, not projectable — no matter how
    /// well it matched the query. Tombstoned conversations still drop out.
    #[test]
    fn group_turns_span_every_conversation_in_the_group() {
        use crate::projection::TurnKey;

        let layer = LayerId::for_test(1);
        let group = GroupId::for_test(1);
        let alloc = TimelineAllocator::new();
        let (file_a, file_b, retired) = (alloc.next(), alloc.next(), alloc.next());
        let mut sub = Substrate::new();
        for tl in [file_a, file_b, retired] {
            sub.register_timeline(tl, layer, group);
        }
        // Two turns per conversation — note the indices COLLIDE across
        // timelines, which is exactly why a bare `(group, index)` key is
        // ambiguous and the turn identity has to carry the timeline.
        for tl in [file_a, file_b, retired] {
            sub.append_with_blocks(tl, 10, 0, 0);
            sub.append_with_blocks(tl, 10, 0, 0);
        }
        sub.tombstone_timeline(retired);

        let turns = ContentResolver::group_turns(&sub, group);
        assert_eq!(
            turns,
            vec![
                TurnKey::new(file_a, TurnIndex(0)),
                TurnKey::new(file_a, TurnIndex(1)),
                TurnKey::new(file_b, TurnIndex(0)),
                TurnKey::new(file_b, TurnIndex(1)),
            ],
            "both live conversations enumerate in registration order; the \
             tombstoned one drops out",
        );
        // Each key resolves against its OWN conversation.
        for key in &turns {
            assert_eq!(ContentResolver::turn_token_count(&sub, *key), 10);
        }
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
