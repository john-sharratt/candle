//! Per-slot projection assembler — full rebuild of the slot's sealed
//! K/V prefix on every call.
//!
//! Snapshots the slot's writer tail, truncates the slot to zero,
//! resolves each [`ProjectionSegment::Sealed`] from the substrate,
//! concatenates the per-layer sealed sequences, injects them, patches
//! substrate `block_range` entries, then re-attaches the writer tail.
//!
//! [`SlotProjectionState`] records what landed on the slot so future
//! work can build on it (cache keying for live-prefill runs, etc.); the
//! current implementation never consults that record during apply —
//! every call is an independent rebuild.

use std::collections::HashMap;
use std::sync::Arc;

use candle_nn::kv_cache::SealedSequence;
use candle_transformers::models::batched_inference::BatchedInferenceSession;

use crate::error::ConversationError;
use crate::projection::{
    Conversation, GroupId, ProjectionSegment, ProjectionTarget, SealedKind, SectionId, TimelineId,
    TurnIndex,
};
use crate::sequence_handle::SequenceId;

/// Per-slot record of the projection that last landed on the slot.
///
/// Lives in [`super::Scheduler`] keyed by `SequenceId`; written by
/// [`apply_segments`] at the end of every rebuild.  The three arrays
/// are positionally aligned — `previous_segments[i]`,
/// `previous_segment_block_counts[i]` and `previous_segment_hashes[i]`
/// all describe the same injected unit.  Read-only at the moment; held
/// for future work that diffs successive projections to skip
/// re-injecting unchanged segments.
#[derive(Debug, Default)]
pub(super) struct SlotProjectionState {
    pub(super) previous_segments: Vec<ProjectionSegment>,
    pub(super) previous_segment_block_counts: Vec<u32>,
    pub(super) previous_segment_hashes: Vec<u64>,
}

impl SlotProjectionState {
    pub(super) fn new() -> Self {
        Self::default()
    }
}

/// Borrowed scheduler state the assembler needs in order to run.
///
/// Bundled into one struct so [`apply_segments`] keeps a tractable
/// signature.  All references are `&mut` only where the assembler must
/// mutate; substrate and target are immutable to the operation.
pub(super) struct ApplyContext<'a> {
    pub(super) session: &'a mut BatchedInferenceSession,
    pub(super) conversation: &'a Conversation,
    pub(super) slot_target: Option<ProjectionTarget>,
    pub(super) parent_id: SequenceId,
    pub(super) chunk_size: usize,
    /// Used only under `feature = "context-dump"` to decode injected
    /// tokens for the diagnostic projection-dump log.  Kept in the
    /// context unconditionally so the public shape stays stable across
    /// feature toggles.
    #[cfg_attr(not(feature = "context-dump"), allow(dead_code))]
    pub(super) tokenizer: &'a tokenizers::Tokenizer,
    pub(super) slot_tokens: &'a mut HashMap<SequenceId, Vec<u32>>,
}

/// Apply `new_segments` onto the slot.  Truncates the slot to zero,
/// resolves each segment's sealed bytes from the substrate, and
/// injects them in declaration order.
///
/// Post-condition: the slot's per-layer K/V is the concatenation of
/// every `new_segments[i]`'s sealed bytes, in declaration order, plus
/// the pre-existing writer tail (re-attached at the end).  Equivalent
/// at LCP = 0 to a full truncate-to-zero + inject-everything rebuild.
pub(super) fn apply_segments(
    state: &mut SlotProjectionState,
    ctx: ApplyContext<'_>,
    new_segments: &[ProjectionSegment],
) -> Result<(), ConversationError> {
    let parent_id = ctx.parent_id;

    // Snapshot any writer-owned tail (in-flight decode chunks).
    // At turn-submit boundaries this is empty; mid-decode
    // reprojection has a non-empty tail that must survive the
    // prefix rewrite.  The snapshot holds RAII refs that keep the
    // underlying arena chunks alive across the truncate.
    let tail_per_layer: Vec<candle_nn::kv_cache::WriterTail> = {
        let mut out: Vec<candle_nn::kv_cache::WriterTail> =
            Vec::with_capacity(ctx.session.backings().len());
        for backing in ctx.session.backings() {
            out.push(
                backing
                    .split_off_writer_tail(parent_id.0)
                    .map_err(ConversationError::Model)?,
            );
        }
        out
    };

    // Full rebuild: truncate the slot to zero and re-inject every
    // segment from substrate.  An earlier draft of this module diffed
    // against `state.previous_segments` and truncated only to the
    // longest-common-prefix boundary so reprojection could skip
    // re-injecting unchanged sections.  That path is correct only when
    // `state.previous_segments` describes EXACTLY what's on the slot in
    // the same block layout; under the reproject control flow the slot
    // can be partially drained (the caller snapshots the writer tail
    // before calling), and any prior call that filtered out
    // non-hot-resident sections leaves a state vs. slot mismatch that
    // a later LCP truncate happily honours — wiping the prefix.  Until
    // the state model tracks slot truth precisely, take the safe path
    // every call: truncate, re-resolve, re-inject.
    ctx.session
        .truncate_sequence_to_blocks(parent_id.0, 0)
        .map_err(ConversationError::Model)?;

    let suffix = new_segments;
    if suffix.is_empty() {
        // Nothing to inject — restore the writer tail (no-op when it
        // was already empty) and clear the per-slot segment record so
        // the next call also rebuilds from scratch.
        for (backing, tail) in ctx.session.backings().iter().zip(tail_per_layer) {
            backing
                .extend_writer_tail(parent_id.0, tail)
                .map_err(ConversationError::Model)?;
        }
        state.previous_segments.clear();
        state.previous_segment_block_counts.clear();
        state.previous_segment_hashes.clear();
        return Ok(());
    }

    // Gather per-unit per-layer sealed sequences in suffix order.
    // Each `SealedSequence` is a vector of windowed `SealedChunk`s
    // (any may be partial — sharing the underlying physical chunk
    // with the writer is safe because windows assert only the bytes
    // they cover).
    //
    // The working set has already been brought into VRAM by
    // `elevate_to_hot` ahead of this call; these lookups just pull
    // the residence's hot bytes out of the substrate.  Any item the
    // elevator missed will surface as `None` here and gets skipped —
    // the elevate report's `missing` / `failed` counters are where
    // to look if turns silently drop.
    enum InjectedUnit {
        Section(SectionId),
        Turn(GroupId, TurnIndex, TimelineId),
    }

    // `units`, `surviving_segments`, and (later) `suffix_block_counts`
    // are positionally aligned — entry `i` in each array describes the
    // same injected unit.  Segments whose substrate lookup misses are
    // skipped from all three so the bookkeeping stays consistent with
    // what actually lands on the slot.
    let mut units: Vec<(InjectedUnit, Arc<Vec<SealedSequence>>)> = Vec::with_capacity(suffix.len());
    let mut surviving_segments: Vec<ProjectionSegment> = Vec::with_capacity(suffix.len());
    let mut suffix_block_counts: Vec<u32> = Vec::with_capacity(suffix.len());
    {
        let view = ctx.conversation.read();
        for seg in suffix {
            match seg {
                ProjectionSegment::Sealed(SealedKind::Section(rs)) => {
                    let sid = rs.id;
                    if let Some(s) = view.section_sealed_of(sid) {
                        units.push((InjectedUnit::Section(sid), s));
                        surviving_segments.push(seg.clone());
                    } else {
                        tracing::warn!(
                            target: "candle_conversation::persistence::tier",
                            section = sid.raw(),
                            slot = parent_id.0,
                            "apply_projection: section not hot — elevate missed it; skipping borrow"
                        );
                    }
                }
                ProjectionSegment::Sealed(SealedKind::Turn(rt)) => {
                    let g = rt.group();
                    let t = rt.index();
                    let timeline = match ctx.slot_target {
                        Some(tgt) if g == tgt.group => Some(tgt.timeline),
                        _ => view.timelines_for_group(g).next(),
                    };
                    if let Some(timeline) = timeline {
                        if let Some(s) = view.turn_sealed_of(timeline, t) {
                            units.push((InjectedUnit::Turn(g, t, timeline), s));
                            surviving_segments.push(seg.clone());
                        } else {
                            tracing::warn!(
                                target: "candle_conversation::persistence::tier",
                                timeline = timeline.raw(),
                                turn = t.0,
                                slot = parent_id.0,
                                "apply_projection: turn not hot — elevate missed it; skipping borrow"
                            );
                        }
                    }
                }
                ProjectionSegment::Generated { .. } | ProjectionSegment::NewUserMessage { .. } => {
                    // Generated and NewUserMessage runs do not flow
                    // through this code path — their K/V is captured
                    // out-of-band by the live-prefill machinery.  If
                    // one shows up here it's a routing bug upstream;
                    // skip it rather than corrupt the slot layout.
                    tracing::warn!(
                        slot = parent_id.0,
                        "apply_projection: unexpected non-Sealed segment — skipping"
                    );
                }
            }
        }
    }

    // No units survived (e.g. every segment was missing from hot).
    // Restore the tail on an empty slot and clear the state record.
    if units.is_empty() {
        for (backing, tail) in ctx.session.backings().iter().zip(tail_per_layer) {
            backing
                .extend_writer_tail(parent_id.0, tail)
                .map_err(ConversationError::Model)?;
        }
        state.previous_segments.clear();
        state.previous_segment_block_counts.clear();
        state.previous_segment_hashes.clear();
        return Ok(());
    }

    // Concatenate per-layer + remember each unit's block extent so
    // we can patch substrate `block_range` entries after inject.
    // Layer-count mismatches drop the unit AND its `surviving_segments`
    // entry so the parallel arrays stay aligned.
    let n_layers = ctx.session.num_layers();
    let chunk_size = ctx.chunk_size;
    let mut per_layer_chunks: Vec<Vec<candle_nn::kv_cache::SealedChunk>> =
        (0..n_layers).map(|_| Vec::new()).collect();
    let mut per_layer_token_count: Vec<usize> = vec![0; n_layers];
    let mut unit_extents: Vec<(InjectedUnit, usize)> = Vec::with_capacity(units.len());
    let mut kept_segments: Vec<ProjectionSegment> = Vec::with_capacity(units.len());
    let units_iter = units.into_iter().zip(surviving_segments.into_iter());
    for ((unit, sealed), seg) in units_iter {
        if sealed.len() != n_layers {
            tracing::warn!(
                "apply_projection: unit has {} layers, expected {}; skipping",
                sealed.len(),
                n_layers
            );
            continue;
        }
        // Layer 0's block count is the canonical extent (every
        // layer's chunk count matches for a well-formed sealed
        // entry).
        let block_count = sealed[0].chunks.len();
        for layer_idx in 0..n_layers {
            let layer_seq = &sealed[layer_idx];
            per_layer_chunks[layer_idx].extend(layer_seq.chunks.iter().cloned());
            per_layer_token_count[layer_idx] += layer_seq.token_count;
        }
        unit_extents.push((unit, block_count));
        suffix_block_counts.push(block_count as u32);
        kept_segments.push(seg);
    }
    let surviving_segments = kept_segments;
    let gpu_per_layer: Vec<SealedSequence> = per_layer_chunks
        .into_iter()
        .zip(per_layer_token_count.into_iter())
        .map(|(chunks, tokens)| SealedSequence {
            chunks,
            token_count: tokens,
            chunk_size,
            location: candle_nn::kv_cache::ArenaLocation::Gpu,
        })
        .collect();

    let (start_block, _end_block) = ctx
        .session
        .inject_sealed_at_tail(parent_id.0, &gpu_per_layer)
        .map_err(ConversationError::Model)?;

    // Record each injected unit's `(start, end)` block range in the
    // substrate so `reproject_view`'s `block_range_of` lookups
    // resolve against the current parent layout.  Also mirror each
    // unit's token IDs into the slot's diagnostic log so the turn-
    // complete dump can reconstruct the exact context the kernel
    // saw.
    let mut injected_tokens: Vec<u32> = Vec::new();
    {
        let mut view = ctx.conversation.write();
        let mut cursor = start_block;
        for (unit, block_count) in unit_extents {
            let next = cursor + block_count;
            match unit {
                InjectedUnit::Section(sid) => {
                    view.set_section_block_range(sid, cursor as u64, next as u64);
                    let toks = view.section_tokens_of(sid);
                    #[cfg(feature = "context-dump")]
                    if tracing::enabled!(
                        target: "candle_conversation::scheduler::projection_dump",
                        tracing::Level::INFO,
                    ) {
                        let decoded = ctx
                            .tokenizer
                            .decode(&toks, false)
                            .unwrap_or_else(|e| format!("<decode error: {e}>"));
                        tracing::info!(
                            target: "candle_conversation::scheduler::projection_dump",
                            slot = parent_id.0,
                            label = %format!("Section({sid:?})"),
                            token_count = toks.len(),
                            "{decoded}\n---"
                        );
                    }
                    injected_tokens.extend_from_slice(&toks);
                }
                InjectedUnit::Turn(_g, t, timeline) => {
                    view.set_block_range(timeline, t, cursor as u64, next as u64);
                    let toks = view.token_ids_of(timeline, t).to_vec();
                    #[cfg(feature = "context-dump")]
                    if tracing::enabled!(
                        target: "candle_conversation::scheduler::projection_dump",
                        tracing::Level::INFO,
                    ) {
                        let decoded = ctx
                            .tokenizer
                            .decode(&toks, false)
                            .unwrap_or_else(|e| format!("<decode error: {e}>"));
                        tracing::info!(
                            target: "candle_conversation::scheduler::projection_dump",
                            slot = parent_id.0,
                            label = %format!("Turn(group={_g:?}, index={t:?})"),
                            token_count = toks.len(),
                            "{decoded}\n---"
                        );
                    }
                    injected_tokens.extend_from_slice(&toks);
                }
            }
            cursor = next;
        }
    }
    #[cfg(feature = "context-dump")]
    if tracing::enabled!(
        target: "candle_conversation::scheduler::projection_dump",
        tracing::Level::INFO,
    ) {
        tracing::info!(
            target: "candle_conversation::scheduler::projection_dump",
            slot = parent_id.0,
            total_tokens = injected_tokens.len(),
            "=== apply_projection injection order complete ==="
        );
    }
    // Diagnostic log: every call truncates the slot to zero before
    // re-injecting, so the slot's full token content is the run of
    // tokens we just injected.  Clear the prior entry and write
    // exactly what's on the slot.
    if let Some(entry) = ctx.slot_tokens.get_mut(&parent_id) {
        entry.clear();
    }
    super::Scheduler::record_slot_tokens(ctx.slot_tokens, parent_id, &injected_tokens);

    // Restore the writer-owned tail snapshot taken at the top.
    for (backing, tail) in ctx.session.backings().iter().zip(tail_per_layer) {
        backing
            .extend_writer_tail(parent_id.0, tail)
            .map_err(ConversationError::Model)?;
    }

    // Record what landed on the slot.  `surviving_segments` is built
    // alongside `units` / `suffix_block_counts` so the three arrays
    // are positionally aligned — `previous_segments[i]` corresponds
    // to `previous_segment_block_counts[i]` for the unit that
    // actually injected.  Segments whose substrate lookup returned
    // None are dropped from the record entirely.
    let n = suffix_block_counts.len();
    debug_assert_eq!(surviving_segments.len(), n);
    state.previous_segments = surviving_segments;
    state.previous_segment_block_counts = suffix_block_counts;
    state.previous_segment_hashes = vec![0u64; n];

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::projection::{GroupId, LayerId, TimelineId};
    use crate::projection::{ResolvedSection, ResolvedTurn, SectionId, TurnId, TurnIndex};

    fn section(id: u32) -> ProjectionSegment {
        ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection {
            id: SectionId::new(id),
        }))
    }

    fn turn(layer: u32, group: u32, index: u32) -> ProjectionSegment {
        ProjectionSegment::Sealed(SealedKind::Turn(ResolvedTurn {
            id: TurnId {
                layer_id: LayerId::for_test(layer),
                group_id: GroupId::for_test(group),
                index: TurnIndex(index),
            },
        }))
    }

    #[test]
    fn slot_state_default_is_empty() {
        let s = SlotProjectionState::new();
        assert!(s.previous_segments.is_empty());
        assert!(s.previous_segment_block_counts.is_empty());
        assert!(s.previous_segment_hashes.is_empty());
    }

    #[test]
    fn segment_helpers_produce_expected_kinds() {
        // Sanity check on the test fixtures used in this module —
        // catches accidental drift in `ResolvedSection` / `ResolvedTurn`
        // construction.
        let s = section(7);
        let t = turn(1, 2, 3);
        match s {
            ProjectionSegment::Sealed(SealedKind::Section(rs)) => {
                assert_eq!(rs.id, SectionId::new(7));
            }
            _ => panic!("expected Sealed(Section)"),
        }
        match t {
            ProjectionSegment::Sealed(SealedKind::Turn(rt)) => {
                assert_eq!(rt.group(), GroupId::for_test(2));
                assert_eq!(rt.index(), TurnIndex(3));
            }
            _ => panic!("expected Sealed(Turn)"),
        }
    }
}
