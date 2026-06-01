//! Per-slot projection assembler — full rebuild of the slot's prefix
//! K/V on every call, with a content-addressed cache for live-prefilled
//! structural template runs.
//!
//! Snapshots the slot's writer tail, truncates to zero, then walks
//! `Vec<ProjectionSegment>` in declaration order:
//!
//! - `Sealed(Section|Turn)` — resolve the substrate entry's per-layer
//!   sealed K/V, Arc-clone it onto the slot.  Fold the segment's tokens
//!   into the rolling hash.
//! - `Generated { tokens, .. }` — accumulate into the current run; do
//!   not advance the slot or the rolling hash yet.  Cache keying is by
//!   the whole run (every adjacent Generated up to the next non-Generated
//!   segment), so partial-prefix folding would produce keys that don't
//!   match any captured K/V.
//! - On a non-Generated segment (or end of list), flush the current
//!   run: compute `key = fold(rolling_hash, run_tokens)`, look up in
//!   the slot cache.  Hit → inject the cached `Vec<SealedSequence>`.
//!   Miss → run a synchronous forward pass over `run_tokens`, capture
//!   the per-layer K/V from the resulting slot range, insert into the
//!   cache.  Either way, commit `rolling_hash = key`.
//!
//! Re-attach the writer tail at the end.
//!
//! The cache is content-addressed (key = hash of preceding tokens plus
//! run tokens), so stale entries miss benignly — there is no way for a
//! wrong cache value to leak onto the slot.

use std::collections::HashMap;
use std::sync::Arc;

use candle::{Device, Tensor};
use candle_nn::kv_cache::{SealedSequence, WriterTail};
use candle_transformers::models::batched_inference::{
    BatchedInferenceSession, ManagedBatchedModel,
};

use crate::conversation::slice_per_layer_sealed;
use crate::error::ConversationError;
use crate::projection::{
    Conversation, GroupId, ProjectionSegment, ProjectionTarget, SealedKind, SectionId, TimelineId,
    TurnIndex,
};
use crate::sequence_handle::SequenceId;

// FNV-1a constants — deterministic, dependency-free, sufficient
// collision resistance over u64 for per-slot caches.
const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
const FNV_PRIME: u64 = 0x100000001b3;

/// Roll `tokens` into a 64-bit FNV-1a state.  Bytes are little-endian
/// over each `u32` so the hash is platform-stable.
fn fold_tokens(mut h: u64, tokens: &[u32]) -> u64 {
    for &t in tokens {
        for b in t.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(FNV_PRIME);
        }
    }
    h
}

/// Content-addressed cache of captured live-prefilled runs on a slot.
///
/// Keyed by `fold_tokens(preceding_slot_tokens ++ run_tokens)`.  Values
/// are the per-layer `Vec<SealedSequence>` captured from the slot just
/// after the prefill that produced them.  Lives as long as the slot;
/// dropped wholesale when the slot is freed.
#[derive(Debug, Default)]
pub(super) struct SlotProjectionCache {
    memo: HashMap<u64, Arc<Vec<SealedSequence>>>,
}

impl SlotProjectionCache {
    fn get(&self, key: u64) -> Option<Arc<Vec<SealedSequence>>> {
        self.memo.get(&key).cloned()
    }

    fn insert(&mut self, key: u64, value: Arc<Vec<SealedSequence>>) {
        self.memo.insert(key, value);
    }

    #[cfg(test)]
    pub(super) fn len(&self) -> usize {
        self.memo.len()
    }
}

/// Per-slot state owned by the projection assembler.
///
/// `cache` memoises captured live-prefill runs for cheap reuse across
/// reprojections.  `pending_user_part` holds the captured K/V for the
/// in-flight turn's user message — populated when a `NewUserMessage`
/// segment is prefilled, cleared at seal time; survives mid-decode
/// reprojection (which truncates the slot) because it lives here on
/// `SlotState`, not on the slot.
#[derive(Debug, Default)]
pub(super) struct SlotState {
    pub(super) cache: SlotProjectionCache,
    pub(super) pending_user_part: Option<Arc<Vec<SealedSequence>>>,
}

impl SlotState {
    pub(super) fn new() -> Self {
        Self::default()
    }
}

/// Pre-tokenised dialect role markers the assembler wraps around
/// every `Sealed::Turn` segment to produce attention-correct turn
/// boundaries at projection time.
///
/// The two markers are the inter-turn ones — `user_start` opens a
/// turn against whatever causal prefix actually precedes it on the
/// slot, and `assistant_end` closes a turn before the next region
/// begins.  The intra-turn `user_end` and `assistant_start` markers
/// stay baked in the persisted turn bytes because their hidden
/// state is dominated by the turn's own (invariant) content.
///
/// The scheduler owns a single instance, pre-tokenised by the
/// engine from the active model's dialect at engine construction.
/// Passed to the assembler as a borrow on `ApplyContext`.
#[derive(Debug, Clone, Default)]
pub(crate) struct BoundaryMarkers {
    pub(crate) user_start: Arc<Vec<u32>>,
    pub(crate) assistant_end: Arc<Vec<u32>>,
}

impl BoundaryMarkers {
    /// Pre-tokenise the dialect's `user_start` and `assistant_end`
    /// strings via the caller-supplied closure.  The closure form
    /// keeps this module tokenizer-agnostic — callers (the engine)
    /// wrap their `tokenizers::Tokenizer::encode`.
    pub(crate) fn from_dialect<E, F>(
        dialect: &candle_transformers::models::dialect::Dialect,
        mut tokenize: F,
    ) -> Result<Self, E>
    where
        F: FnMut(&str) -> Result<Vec<u32>, E>,
    {
        let user_start = Arc::new(tokenize(dialect.user_start)?);
        let assistant_end = Arc::new(tokenize(dialect.assistant_end)?);
        Ok(Self {
            user_start,
            assistant_end,
        })
    }
}

/// Borrowed scheduler state the assembler needs in order to run.
///
/// `model` and `device` are required so cache-miss runs can drive a
/// synchronous `forward_batched` to compute and capture the missing
/// K/V; `max_prefill_chunk` caps the per-pass token count to keep
/// activation buffers bounded.
pub(super) struct ApplyContext<'a> {
    pub(super) session: &'a mut BatchedInferenceSession,
    pub(super) model: &'a mut Box<dyn ManagedBatchedModel + Send>,
    pub(super) device: &'a Device,
    pub(super) conversation: &'a Conversation,
    pub(super) slot_target: Option<ProjectionTarget>,
    pub(super) parent_id: SequenceId,
    pub(super) chunk_size: usize,
    pub(super) max_prefill_chunk: usize,
    /// Used only under `feature = "context-dump"`; kept unconditionally
    /// so the public shape stays stable across feature toggles.
    #[cfg_attr(not(feature = "context-dump"), allow(dead_code))]
    pub(super) tokenizer: &'a tokenizers::Tokenizer,
    pub(super) slot_tokens: &'a mut HashMap<SequenceId, Vec<u32>>,
    /// Pre-tokenised inter-turn boundary markers — see [`BoundaryMarkers`].
    /// The walker pre-pends `user_start` and trails `assistant_end`
    /// around every `Sealed::Turn` segment.
    pub(super) boundary_markers: &'a BoundaryMarkers,
}

/// Apply `new_segments` onto the slot.
///
/// Post-condition: the slot's per-layer K/V is the concatenation of
/// every `new_segments[i]`'s K/V (substrate-pinned for `Sealed`,
/// live-prefilled or cache-loaded for `Generated`, captured to
/// `pending_user_part` for `NewUserMessage`), in declaration order,
/// plus the pre-existing writer tail re-attached at the end.
pub(super) fn apply_segments(
    state: &mut SlotState,
    mut ctx: ApplyContext<'_>,
    new_segments: &[ProjectionSegment],
) -> Result<(), ConversationError> {
    let parent_id = ctx.parent_id;

    // 1. Snapshot the writer tail (in-flight decode chunks).  Empty at
    //    turn-submit boundaries; non-empty during mid-decode reproject.
    let tail_per_layer = snapshot_tail(ctx.session, parent_id)?;

    // 2. Truncate the slot.  Drops Arc refs to whatever it previously
    //    held; arenas reclaim asynchronously.
    ctx.session
        .truncate_sequence_to_blocks(parent_id.0, 0)
        .map_err(ConversationError::Model)?;

    // 3. Diagnostic log: full rebuild, so the slot_tokens record is
    //    rewritten from scratch this call.
    if let Some(entry) = ctx.slot_tokens.get_mut(&parent_id) {
        entry.clear();
    }

    // 4. Walk segments in declaration order, batching adjacent
    //    Generated entries into a single run.
    let mut walker = SegmentWalker {
        rolling_hash: FNV_OFFSET_BASIS,
        run_tokens: Vec::new(),
        last_was_sealed: false,
    };

    let mut i = 0;
    while i < new_segments.len() {
        if matches!(&new_segments[i], ProjectionSegment::Generated { .. }) {
            // Collect every adjacent Generated into one run.  The
            // batched forward pass amortises the per-call kernel
            // launch overhead across all the structural tokens.
            while i < new_segments.len() {
                let ProjectionSegment::Generated { tokens, .. } = &new_segments[i] else {
                    break;
                };
                walker.run_tokens.extend(tokens.iter().copied());
                i += 1;
            }
            walker.flush_run(state, &mut ctx)?;
            continue;
        }

        match &new_segments[i] {
            ProjectionSegment::Sealed(SealedKind::Section(rs)) => {
                walker.flush_run(state, &mut ctx)?;
                inject_sealed_section(state, &mut ctx, &mut walker, rs.id)?;
            }
            ProjectionSegment::Sealed(SealedKind::Turn(rt, part)) => {
                // Wrap the turn in live-prefilled boundary markers:
                // `user_start` joins the run that flushes immediately
                // before the turn injects (batching with whatever
                // `Generated` segments — or with the previous turn's
                // trailing `assistant_end` — preceded it), and
                // `assistant_end` opens a fresh run that accumulates
                // until the next non-Generated segment (the next
                // turn's `user_start`, or the trailing
                // `Generated(UserStart)` the scheduler appends for
                // the current turn's prefill).  The result is one
                // 5-token batched prefill run at every cross-turn
                // boundary — and at every system→turn / last-turn→
                // current-turn-prefill boundary.
                walker
                    .run_tokens
                    .extend(ctx.boundary_markers.user_start.iter().copied());
                walker.flush_run(state, &mut ctx)?;
                inject_sealed_turn(
                    state,
                    &mut ctx,
                    &mut walker,
                    rt.group(),
                    rt.index(),
                    *part,
                )?;
                walker
                    .run_tokens
                    .extend(ctx.boundary_markers.assistant_end.iter().copied());
            }
            ProjectionSegment::NewUserMessage { tokens } => {
                walker.flush_run(state, &mut ctx)?;
                handle_new_user_message(state, &mut ctx, &mut walker, tokens)?;
            }
            ProjectionSegment::Generated { .. } => unreachable!("handled in run loop above"),
        }
        i += 1;
    }
    walker.flush_run(state, &mut ctx)?;

    // 5. Re-attach the writer tail at the slot's current end.
    restore_tail(ctx.session, parent_id, tail_per_layer)?;

    Ok(())
}

/// State carried across a single `apply_segments` walk.
struct SegmentWalker {
    rolling_hash: u64,
    run_tokens: Vec<u32>,
    /// True if the most recent slot mutation was a Sealed inject (Arc-
    /// shared, partial trailing chunk possible).  Set on Sealed inject,
    /// cleared after we push a fresh writer chunk for a subsequent
    /// prefill so the prefill doesn't write into the Arc-shared partial.
    last_was_sealed: bool,
}

impl SegmentWalker {
    fn flush_run(
        &mut self,
        state: &mut SlotState,
        ctx: &mut ApplyContext<'_>,
    ) -> Result<(), ConversationError> {
        if self.run_tokens.is_empty() {
            return Ok(());
        }
        let key = fold_tokens(self.rolling_hash, &self.run_tokens);

        if let Some(cached) = state.cache.get(key) {
            inject_arc_sealed(ctx.session, ctx.parent_id, ctx.chunk_size, &cached)?;
            log_injected_tokens(ctx, &self.run_tokens);
        } else {
            let captured =
                drive_prefill_and_capture(ctx, &self.run_tokens, self.last_was_sealed)?;
            state.cache.insert(key, captured);
        }

        self.rolling_hash = key;
        self.run_tokens.clear();
        // The slot now ends in chunks the writer just wrote (cache
        // miss) or just Arc-cloned (cache hit).  Either way the next
        // Sealed inject will append cleanly; the next prefill, if
        // any, will need a fresh writer chunk because the cached run
        // also ends in a (possibly partial) Arc-shared chunk.
        self.last_was_sealed = true;
        Ok(())
    }
}

// ── Sealed-segment injection ─────────────────────────────────────────────────

fn inject_sealed_section(
    _state: &mut SlotState,
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    sid: SectionId,
) -> Result<(), ConversationError> {
    let parent_id = ctx.parent_id;
    let sealed = match ctx.conversation.read().section_sealed_of(sid) {
        Some(s) => s,
        None => {
            tracing::warn!(
                target: "candle_conversation::persistence::tier",
                section = sid.raw(),
                slot = parent_id.0,
                "apply_projection: section not hot — elevate missed it; skipping borrow"
            );
            return Ok(());
        }
    };
    inject_arc_sealed(ctx.session, parent_id, ctx.chunk_size, &sealed)?;

    let toks = ctx.conversation.read().section_tokens_of(sid);
    let start_block = ctx
        .session.sequence_block_count(parent_id.0).ok_or_else(|| ConversationError::Channel(format!("apply_projection: slot {} not in session", parent_id)))?
        .saturating_sub(sealed[0].chunks.len());
    let end_block = start_block + sealed[0].chunks.len();
    ctx.conversation
        .write()
        .set_section_block_range(sid, start_block as u64, end_block as u64);

    walker.rolling_hash = fold_tokens(walker.rolling_hash, &toks);
    log_injected_tokens(ctx, &toks);
    walker.last_was_sealed = true;
    Ok(())
}

fn inject_sealed_turn(
    _state: &mut SlotState,
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    group: GroupId,
    index: TurnIndex,
    _part: crate::Role,
) -> Result<(), ConversationError> {
    let parent_id = ctx.parent_id;

    let timeline: Option<TimelineId> = match ctx.slot_target {
        Some(tgt) if group == tgt.group => Some(tgt.timeline),
        _ => ctx.conversation.read().timelines_for_group(group).next(),
    };
    let Some(timeline) = timeline else {
        return Ok(());
    };

    let sealed = match ctx.conversation.read().turn_sealed_of(timeline, index) {
        Some(s) => s,
        None => return Ok(()),
    };
    if sealed.is_empty() {
        return Ok(());
    }
    inject_arc_sealed(ctx.session, parent_id, ctx.chunk_size, &sealed)?;

    let toks: Vec<u32> = ctx
        .conversation
        .read()
        .assistant_token_ids_of(timeline, index)
        .to_vec();
    let start_block = ctx
        .session
        .sequence_block_count(parent_id.0)
        .ok_or_else(|| {
            ConversationError::Channel(format!(
                "apply_projection: slot {} not in session",
                parent_id
            ))
        })?
        .saturating_sub(sealed[0].chunks.len());
    let end_block = start_block + sealed[0].chunks.len();
    ctx.conversation
        .write()
        .set_block_range(timeline, index, start_block as u64, end_block as u64);

    walker.rolling_hash = fold_tokens(walker.rolling_hash, &toks);
    log_injected_tokens(ctx, &toks);
    walker.last_was_sealed = true;
    Ok(())
}

fn inject_arc_sealed(
    session: &mut BatchedInferenceSession,
    parent_id: SequenceId,
    chunk_size: usize,
    sealed: &Arc<Vec<SealedSequence>>,
) -> Result<(), ConversationError> {
    let n_layers = session.num_layers();
    if sealed.len() != n_layers {
        tracing::warn!(
            "apply_projection: unit has {} layers, expected {}; skipping",
            sealed.len(),
            n_layers,
        );
        return Ok(());
    }
    let mut per_layer: Vec<SealedSequence> = Vec::with_capacity(n_layers);
    for layer_seq in sealed.iter() {
        per_layer.push(SealedSequence {
            chunks: layer_seq.chunks.clone(),
            token_count: layer_seq.token_count,
            chunk_size,
            location: candle_nn::kv_cache::ArenaLocation::Gpu,
        });
    }
    session
        .inject_sealed_at_tail(parent_id.0, &per_layer)
        .map_err(ConversationError::Model)?;
    Ok(())
}

// ── NewUserMessage handling (live-prefill into pending_user_part) ────────────

fn handle_new_user_message(
    state: &mut SlotState,
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    tokens: &Arc<Vec<u32>>,
) -> Result<(), ConversationError> {
    if let Some(cached) = state.pending_user_part.clone() {
        // Mid-decode reproject path: the user's K/V was already
        // captured on an earlier apply.  Re-inject the cached bytes;
        // do not re-run the forward pass.
        inject_arc_sealed(ctx.session, ctx.parent_id, ctx.chunk_size, &cached)?;
        log_injected_tokens(ctx, tokens);
    } else {
        let captured = drive_prefill_and_capture(ctx, tokens, walker.last_was_sealed)?;
        state.pending_user_part = Some(captured);
    }
    walker.rolling_hash = fold_tokens(walker.rolling_hash, tokens);
    walker.last_was_sealed = true;
    Ok(())
}

// ── Live prefill drive (cache-miss path) ─────────────────────────────────────

/// Run a synchronous forward pass over `tokens` on the slot, then
/// extract the per-layer `SealedSequence` for the just-written block
/// range.  Returns the captured K/V wrapped in an `Arc` so the cache
/// (or `pending_user_part`) can store it without further copies.
///
/// `last_was_sealed` indicates whether the slot's current tail is
/// Arc-shared with the substrate.  If so, we push a fresh writer
/// chunk first so this prefill's writes don't alias the partial
/// trailing chunk of the previous Sealed inject.
fn drive_prefill_and_capture(
    ctx: &mut ApplyContext<'_>,
    tokens: &[u32],
    last_was_sealed: bool,
) -> Result<Arc<Vec<SealedSequence>>, ConversationError> {
    let parent_id = ctx.parent_id;

    if last_was_sealed {
        ctx.session
            .push_empty_writer_chunk(parent_id.0)
            .map_err(ConversationError::Model)?;
    }

    let start_block = ctx
        .session.sequence_block_count(parent_id.0).ok_or_else(|| ConversationError::Channel(format!("apply_projection: slot {} not in session", parent_id)))?;

    let mut offset = 0;
    while offset < tokens.len() {
        let chunk_len = (tokens.len() - offset).min(ctx.max_prefill_chunk);
        let slice = &tokens[offset..offset + chunk_len];
        let input = Tensor::new(slice, ctx.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(ConversationError::Model)?;
        let _logits = ctx
            .model
            .forward_batched(ctx.session, &[parent_id.0], &[input])
            .map_err(ConversationError::Model)?;
        ctx.session
            .advance_sequence(parent_id.0, chunk_len)
            .map_err(ConversationError::Model)?;
        offset += chunk_len;
    }

    let end_block = ctx
        .session.sequence_block_count(parent_id.0).ok_or_else(|| ConversationError::Channel(format!("apply_projection: slot {} not in session", parent_id)))?;

    let full = ctx
        .session
        .snapshot_sequence_per_layer(parent_id.0)
        .map_err(ConversationError::Model)?;
    let captured = slice_per_layer_sealed(&full, start_block, end_block);
    log_injected_tokens(ctx, tokens);

    Ok(Arc::new(captured))
}

// ── Writer-tail snapshot / restore ───────────────────────────────────────────

fn snapshot_tail(
    session: &mut BatchedInferenceSession,
    parent_id: SequenceId,
) -> Result<Vec<WriterTail>, ConversationError> {
    let mut out: Vec<WriterTail> = Vec::with_capacity(session.backings().len());
    for backing in session.backings() {
        out.push(
            backing
                .split_off_writer_tail(parent_id.0)
                .map_err(ConversationError::Model)?,
        );
    }
    Ok(out)
}

fn restore_tail(
    session: &mut BatchedInferenceSession,
    parent_id: SequenceId,
    tail_per_layer: Vec<WriterTail>,
) -> Result<(), ConversationError> {
    for (backing, tail) in session.backings().iter().zip(tail_per_layer) {
        backing
            .extend_writer_tail(parent_id.0, tail)
            .map_err(ConversationError::Model)?;
    }
    Ok(())
}

// ── Diagnostic logging ───────────────────────────────────────────────────────

fn log_injected_tokens(ctx: &mut ApplyContext<'_>, tokens: &[u32]) {
    super::Scheduler::record_slot_tokens(ctx.slot_tokens, ctx.parent_id, tokens);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{GroupId, LayerId, ResolvedSection, ResolvedTurn, TurnId};

    fn section_seg(id: u32) -> ProjectionSegment {
        ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection {
            id: SectionId::new(id),
        }))
    }

    fn turn_seg(layer: u32, group: u32, index: u32) -> ProjectionSegment {
        ProjectionSegment::Sealed(SealedKind::Turn(
            ResolvedTurn {
                id: TurnId {
                    layer_id: LayerId::for_test(layer),
                    group_id: GroupId::for_test(group),
                    index: TurnIndex(index),
                },
            },
            crate::Role::Assistant,
        ))
    }

    fn generated_seg(name: &str, position: usize, tokens: &[u32]) -> ProjectionSegment {
        ProjectionSegment::Generated {
            tokens: Arc::new(tokens.to_vec()),
            identity: crate::projection::GeneratedIdentity {
                name: name.to_string(),
                position,
            },
        }
    }

    #[test]
    fn fold_tokens_deterministic() {
        let a = fold_tokens(FNV_OFFSET_BASIS, &[1, 2, 3]);
        let b = fold_tokens(FNV_OFFSET_BASIS, &[1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn fold_tokens_order_sensitive() {
        let a = fold_tokens(FNV_OFFSET_BASIS, &[1, 2, 3]);
        let b = fold_tokens(FNV_OFFSET_BASIS, &[3, 2, 1]);
        assert_ne!(a, b);
    }

    #[test]
    fn fold_tokens_preceding_context_matters() {
        // Same run, different preceding hash → different key.
        let pre_a = fold_tokens(FNV_OFFSET_BASIS, &[100]);
        let pre_b = fold_tokens(FNV_OFFSET_BASIS, &[101]);
        let a = fold_tokens(pre_a, &[1, 2, 3]);
        let b = fold_tokens(pre_b, &[1, 2, 3]);
        assert_ne!(a, b);
    }

    #[test]
    fn fold_tokens_incremental_matches_one_shot() {
        let one_shot = fold_tokens(FNV_OFFSET_BASIS, &[1, 2, 3, 4, 5, 6]);
        let h1 = fold_tokens(FNV_OFFSET_BASIS, &[1, 2, 3]);
        let h2 = fold_tokens(h1, &[4, 5, 6]);
        assert_eq!(one_shot, h2);
    }

    #[test]
    fn cache_get_set_round_trip() {
        let mut cache = SlotProjectionCache::default();
        assert_eq!(cache.len(), 0);
        let v: Arc<Vec<SealedSequence>> = Arc::new(Vec::new());
        cache.insert(42, v.clone());
        assert_eq!(cache.len(), 1);
        let got = cache.get(42).unwrap();
        assert!(Arc::ptr_eq(&got, &v));
        assert!(cache.get(43).is_none());
    }

    #[test]
    fn slot_state_default_is_empty() {
        let s = SlotState::new();
        assert_eq!(s.cache.len(), 0);
        assert!(s.pending_user_part.is_none());
    }

    #[test]
    fn fixture_helpers_produce_expected_kinds() {
        match section_seg(7) {
            ProjectionSegment::Sealed(SealedKind::Section(rs)) => {
                assert_eq!(rs.id, SectionId::new(7));
            }
            _ => panic!("expected Sealed(Section)"),
        }
        match turn_seg(1, 2, 3) {
            ProjectionSegment::Sealed(SealedKind::Turn(rt, part)) => {
                assert_eq!(rt.group(), GroupId::for_test(2));
                assert_eq!(rt.index(), TurnIndex(3));
                assert_eq!(part, crate::Role::Assistant);
            }
            _ => panic!("expected Sealed(Turn)"),
        }
        match generated_seg("x", 0, &[1, 2, 3]) {
            ProjectionSegment::Generated { tokens, identity } => {
                assert_eq!(*tokens, vec![1, 2, 3]);
                assert_eq!(identity.name, "x");
                assert_eq!(identity.position, 0);
            }
            _ => panic!("expected Generated"),
        }
    }
}
