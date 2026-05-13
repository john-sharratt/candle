//! [`Conversation`] — the workspace-shared substrate handle, and
//! [`TargetedRead`] — the target-aware [`ContentResolver`] wrapper.

use std::sync::{Arc, RwLock};

use candle_nn::kv_cache::SealedSequence;
use super::ids::{GroupId, LayerId, SectionId, TimelineAllocator, TimelineId, TurnIndex};
use super::schema::{DepthWeights, ScoreFormula};
use crate::substrate::{ContentResolver, Substrate, SubstrateRead, SubstrateWrite};
use crate::substrate_cache::SubstrateCache;
use crate::token_buffer::TokenBuffer;
use crate::turn::Role;

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
}

impl Default for Conversation {
    fn default() -> Self {
        Self {
            inner: Arc::new(RwLock::new(Substrate::new())),
            allocator: Arc::new(TimelineAllocator::new()),
        }
    }
}

impl Conversation {
    /// Create a fresh empty in-memory substrate.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a conversation backed by a shared [`SubstrateCache`].
    ///
    /// Pass a clone of the engine-level cache so VRAM accounting and the
    /// eviction budget are shared across all sessions.
    pub fn with_cache(cache: SubstrateCache) -> Self {
        Self {
            inner: Arc::new(RwLock::new(Substrate::with_cache(cache))),
            allocator: Arc::new(TimelineAllocator::new()),
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
        )
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
