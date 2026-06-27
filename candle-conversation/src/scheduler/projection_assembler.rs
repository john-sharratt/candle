//! Per-slot projection assembler — full rebuild of the slot's prefix
//! K/V on every call.
//!
//! Snapshots the slot's writer tail, truncates to zero, then walks
//! `Vec<ProjectionSegment>` in declaration order:
//!
//! - `Sealed(Section|Turn)` — resolve the substrate entry's per-layer
//!   sealed K/V, Arc-clone it onto the slot.
//! - `Generated { tokens, .. }` — accumulate into the current run; do
//!   not advance the slot yet. Adjacent `Generated` segments batch into
//!   a single run so one forward pass amortises the launch cost across
//!   all the structural/boundary tokens.
//! - On a non-Generated segment (or end of list), flush the current run:
//!   a synchronous forward pass over `run_tokens` writes their K/V onto
//!   the slot.
//!
//! Re-attach the writer tail at the end.
//!
//! The prefix is re-derived from scratch every projection — there is no
//! cross-projection memoisation, so the assembled K/V always reflects the
//! exact segment sequence the resolver selected this call.

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
use crate::scheduler::profile;
use crate::sequence_handle::SequenceId;

/// Per-slot state owned by the projection assembler.
///
/// `pending_user_part` holds the captured K/V for the in-flight turn's user
/// message — populated when a `NewUserMessage` segment is prefilled, cleared at
/// seal time; survives mid-decode reprojection (which truncates the slot)
/// because it lives here on `SlotState`, not on the slot.
#[derive(Debug, Default)]
pub(super) struct SlotState {
    pub(super) pending_user_part: Option<Arc<Vec<SealedSequence>>>,
}

impl SlotState {
    /// Drop the in-flight `NewUserMessage` capture after a successful seal.
    pub(super) fn trim_post_turn(&mut self) {
        self.pending_user_part = None;
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
    /// Intra-turn markers — `user_end` closes the user message and
    /// `assistant_start` opens the assistant reply. Not used by the
    /// assembler (they stay baked into persisted turn bytes), but the
    /// summary probe needs them to frame its synthetic user→assistant
    /// exchange so the model *responds* rather than continuing the prompt.
    pub(crate) user_end: Arc<Vec<u32>>,
    pub(crate) assistant_start: Arc<Vec<u32>>,
    /// The dialect's `/no_think` soft-switch (empty for non-thinking dialects).
    /// Emitted as live glue right after `user_start` on a suppressed turn so the
    /// switch sits in the user turn (where Qwen3 honours it) without being baked
    /// into any sealed turn.
    pub(crate) no_think: Arc<Vec<u32>>,
}

impl BoundaryMarkers {
    /// Pre-tokenise the dialect's role-marker strings via the
    /// caller-supplied closure.  The closure form keeps this module
    /// tokenizer-agnostic — callers (the engine) wrap their
    /// `tokenizers::Tokenizer::encode`.
    pub(crate) fn from_dialect<E, F>(
        dialect: &candle_transformers::models::dialect::Dialect,
        mut tokenize: F,
    ) -> Result<Self, E>
    where
        F: FnMut(&str) -> Result<Vec<u32>, E>,
    {
        let user_start = Arc::new(tokenize(dialect.user_start)?);
        let assistant_end = Arc::new(tokenize(dialect.assistant_end)?);
        let user_end = Arc::new(tokenize(dialect.user_end)?);
        let assistant_start = Arc::new(tokenize(dialect.assistant_start)?);
        let no_think = Arc::new(tokenize(dialect.no_think)?);
        Ok(Self {
            user_start,
            assistant_end,
            user_end,
            assistant_start,
            no_think,
        })
    }
}

/// One step of the assembled materialized prefix, in logical order.
///
/// This is the SINGLE source of truth for how a projection's segments + dialect
/// glue lay out. Both the apply path (which injects K/V per piece) and the
/// substrate debug view (which renders them) iterate the same
/// [`assemble_pieces`] output, so they can never silently diverge.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum AssembledPiece {
    /// A run of "glue" tokens — `Generated` template runs merged with the
    /// inter-turn boundary markers (`user_start` before a turn, `assistant_end`
    /// after it) — that the engine gap-fills as one island.
    Glue(Vec<u32>),
    /// A sealed system-prompt section; its K/V comes from the substrate.
    Section(SectionId),
    /// A sealed past turn; its K/V comes from the substrate.
    Turn {
        group: GroupId,
        index: TurnIndex,
        role: crate::Role,
    },
    /// The in-flight user message, deferred to prefill after the gap-fill.
    DeferredUser(Arc<Vec<u32>>),
    /// A sealed turn's user-message half only (compression turn-half injection),
    /// with NO per-turn boundary-marker wrapping — the compression pass supplies
    /// its own framing glue.
    TurnHalf { group: GroupId, index: TurnIndex },
}

/// Lay out a projection's segments into the ordered [`AssembledPiece`]s the
/// engine materializes — the glue *decision*, factored out of the K/V injection
/// so it can be reused (the substrate debug view) and unit-tested in isolation.
///
/// Exactly mirrors what the apply walk produced inline: consecutive `Generated`
/// runs accumulate into one glue island; a `Sealed::Turn` is wrapped with
/// `user_start` (flushed into the glue island immediately before it) and
/// `assistant_end` (carried into the island after it, so it merges with the next
/// turn's `user_start`); `NewUserMessage` is deferred past the gap-fill.
pub(crate) fn assemble_pieces(
    segments: &[ProjectionSegment],
    markers: &BoundaryMarkers,
) -> Vec<AssembledPiece> {
    let mut pieces: Vec<AssembledPiece> = Vec::new();
    let mut run: Vec<u32> = Vec::new();
    fn flush(run: &mut Vec<u32>, pieces: &mut Vec<AssembledPiece>) {
        if !run.is_empty() {
            pieces.push(AssembledPiece::Glue(std::mem::take(run)));
        }
    }
    let mut i = 0;
    while i < segments.len() {
        if matches!(&segments[i], ProjectionSegment::Generated { .. }) {
            while let Some(ProjectionSegment::Generated { tokens, .. }) = segments.get(i) {
                run.extend(tokens.iter().copied());
                i += 1;
            }
            flush(&mut run, &mut pieces);
            continue;
        }
        match &segments[i] {
            ProjectionSegment::Sealed(SealedKind::Section(rs)) => {
                flush(&mut run, &mut pieces);
                pieces.push(AssembledPiece::Section(rs.id));
            }
            ProjectionSegment::Sealed(SealedKind::Turn(rt, role)) => {
                run.extend(markers.user_start.iter().copied());
                flush(&mut run, &mut pieces);
                pieces.push(AssembledPiece::Turn {
                    group: rt.group(),
                    index: rt.index(),
                    role: *role,
                });
                run.extend(markers.assistant_end.iter().copied());
            }
            ProjectionSegment::Sealed(SealedKind::TurnHalf(rt)) => {
                // The compression pass supplies its own framing glue, so the
                // user-half injects with no per-turn boundary-marker wrapping.
                flush(&mut run, &mut pieces);
                pieces.push(AssembledPiece::TurnHalf {
                    group: rt.group(),
                    index: rt.index(),
                });
            }
            ProjectionSegment::NewUserMessage { tokens } => {
                flush(&mut run, &mut pieces);
                pieces.push(AssembledPiece::DeferredUser(tokens.clone()));
            }
            ProjectionSegment::Generated { .. } => unreachable!("handled in the run loop above"),
        }
        i += 1;
    }
    flush(&mut run, &mut pieces);
    pieces
}

/// Borrowed scheduler state the assembler needs in order to run.
///
/// `model` and `device` are required so cache-miss runs can drive a
/// synchronous `forward_batched` to compute and capture the missing
/// K/V; `max_prefill_pass_tokens` caps the per-pass token count to keep
/// activation buffers bounded.
pub(super) struct ApplyContext<'a> {
    pub(super) session: &'a mut BatchedInferenceSession,
    pub(super) model: &'a mut Box<dyn ManagedBatchedModel + Send>,
    pub(super) device: &'a Device,
    pub(super) conversation: &'a Conversation,
    pub(super) slot_target: Option<ProjectionTarget>,
    pub(super) parent_id: SequenceId,
    pub(super) chunk_size: usize,
    pub(super) max_prefill_pass_tokens: usize,
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

/// A built-but-not-yet-fired gap-fill descriptor for one slot. Owned (no
/// borrows) so the cross-conversation wave can collect plans from many slots,
/// fire one batched multi-slot forward, then finish each slot independently.
pub(super) struct GapFillPlan {
    pub parent_id: SequenceId,
    /// The new region: every glue island's tokens, in logical order.
    pub glue_tokens: Vec<u32>,
    /// Flat (kv_len) TRUE sequence position of each column (sealed prefix ++
    /// glue). Computed by the wave; staged on the session by
    /// [`fire_gap_fill_batch`] and consumed by the paged-glue kernel's
    /// actual-position mask + RoPE.
    pub col_actual_pos: Vec<u32>,
    /// The in-flight user message, prefilled in `apply_segments_finish` after
    /// the gap-fill so it lands against the full `[sealed | glue]` prefix.
    pub deferred_user: Option<Arc<Vec<u32>>>,
    /// The writer tail snapshotted before truncation, re-attached on finish.
    pub tail_per_layer: Vec<WriterTail>,
    /// Glue token count (diagnostic).
    pub n_glue_tokens: usize,
}

/// Apply `new_segments` onto the slot (single-slot path).
///
/// Post-condition: the slot's per-layer K/V is `[sealed prefix | glue]` plus the
/// re-attached writer tail and the prefilled in-flight user message. Equivalent
/// to `apply_segments_build` + `fire_gap_fill_batch` (one slot) + `_finish`.
pub(super) fn apply_segments(
    state: &mut SlotState,
    mut ctx: ApplyContext<'_>,
    new_segments: &[ProjectionSegment],
) -> Result<(), ConversationError> {
    let plan = apply_segments_build(&mut ctx, new_segments)?;
    fire_gap_fill_batch(ctx.session, &**ctx.model, ctx.device, &[&plan])?;
    apply_segments_finish(state, &mut ctx, plan)
}

/// Build phase: snapshot the writer tail, truncate, then walk the segments —
/// injecting the sealed prefix and collecting all glue as the new region with
/// each column's TRUE logical position. Reserves the glue writer chunk but does
/// NOT fire the forward; the caller fires (batched across slots) and then calls
/// [`apply_segments_finish`].
pub(super) fn apply_segments_build(
    ctx: &mut ApplyContext<'_>,
    new_segments: &[ProjectionSegment],
) -> Result<GapFillPlan, ConversationError> {
    let parent_id = ctx.parent_id;

    // 1. Snapshot the writer tail (in-flight decode chunks).
    let tail_per_layer = {
        let _g = profile::span("apply:snapshot_tail");
        snapshot_tail(ctx.session, parent_id)?
    };

    // 2. Truncate the slot.
    ctx.session
        .truncate_sequence_to_blocks(parent_id.0, 0)
        .map_err(ConversationError::Model)?;

    // 3. Full rebuild — rewrite the slot_tokens record from scratch.
    if let Some(entry) = ctx.slot_tokens.get_mut(&parent_id) {
        entry.clear();
    }

    // 4. Walk segments: inject sealed as the contiguous prefix and collect all
    //    glue (boundary markers + Generated) as the new region, tracking every
    //    column's TRUE logical position. A single gap-fill forward then computes
    //    every glue island's K/V at once — each attends only logically-earlier
    //    columns via `col_actual_pos`. The NewUserMessage is deferred to after
    //    the gap-fill so it prefills against the full `[sealed | glue]` prefix.
    // The glue/order decision is owned by `assemble_pieces` (the single source
    // of truth, shared with the substrate debug view). Here we only drive the
    // per-piece K/V injection + logical-column accounting from its output: a
    // glue island is collected as one run; a sealed section/turn is injected
    // (the boundary markers around a turn are already folded into the adjacent
    // glue islands by `assemble_pieces`); the user message is deferred.
    let mut walker = SegmentWalker::new();
    for piece in assemble_pieces(new_segments, ctx.boundary_markers) {
        match piece {
            AssembledPiece::Glue(tokens) => {
                walker.run_tokens.extend(tokens);
                walker.collect_run();
            }
            AssembledPiece::Section(id) => {
                inject_sealed_section(ctx, &mut walker, id)?;
            }
            AssembledPiece::Turn { group, index, role } => {
                inject_sealed_turn(ctx, &mut walker, group, index, role)?;
            }
            AssembledPiece::DeferredUser(tokens) => {
                walker.deferred_user = Some(tokens);
            }
            AssembledPiece::TurnHalf { group, index } => {
                inject_sealed_turn_half(ctx, &mut walker, group, index)?;
            }
        }
    }
    walker.collect_run();

    // Reserve the glue's writer chunk now (before the batched forward) so the
    // gap-fill writes don't alias the Arc-shared sealed partial tail.
    if !walker.glue_tokens.is_empty() {
        push_empty_if_sealed(ctx, walker.last_was_sealed)?;
    }

    // Assembly summary: how much selected history actually reached the slot.
    // `skipped_*` > 0 means the model will decode against a context missing
    // turns/sections the resolver chose — the first thing to check when a
    // response can't see its own recent history. The skip cases also emit a
    // dedicated `warn!` below; this per-reproject line stays at `debug` so it
    // doesn't flood production logs.
    tracing::trace!(
        target: "candle_conversation::scheduler::reproject",
        slot = parent_id.0,
        segments = new_segments.len(),
        sealed_turns = walker.sealed_turns,
        sealed_sections = walker.sealed_sections,
        sealed_tokens = walker.sealed_tokens,
        skipped_turns = walker.skipped_turns,
        skipped_sections = walker.skipped_sections,
        glue_tokens = walker.n_glue_tokens,
        deferred_user = walker.deferred_user.is_some(),
        "apply_segments: assembled slot prefix"
    );

    let mut col_actual_pos = Vec::with_capacity(walker.col_prefix.len() + walker.col_new.len());
    col_actual_pos.extend_from_slice(&walker.col_prefix);
    col_actual_pos.extend_from_slice(&walker.col_new);

    Ok(GapFillPlan {
        parent_id,
        glue_tokens: std::mem::take(&mut walker.glue_tokens),
        col_actual_pos,
        deferred_user: walker.deferred_user.take(),
        tail_per_layer,
        n_glue_tokens: walker.n_glue_tokens,
    })
}

/// State carried across a single `apply_segments` walk. Accumulates the gap-fill
/// new region (every glue island) plus each column's TRUE logical position, so
/// the walk emits one batched gap-fill forward instead of N per-island prefills.
struct SegmentWalker {
    /// Glue tokens for the *current* run, drained by `collect_run`.
    run_tokens: Vec<u32>,
    /// All glue tokens across every island, in logical order — the new region.
    glue_tokens: Vec<u32>,
    /// Logical position of each sealed (prefix) column, in inject order — must
    /// match the position_map order the gap-fill reads the prefix in.
    col_prefix: Vec<u32>,
    /// Logical position of each glue (new-region) column, in collect order.
    col_new: Vec<u32>,
    /// Running TRUE sequence position as the walk advances through sealed+glue.
    logical_pos: u32,
    /// True if the most recent slot mutation was a Sealed inject (Arc-shared,
    /// partial trailing chunk possible) — so the gap-fill pushes a fresh writer
    /// chunk before writing.
    last_was_sealed: bool,
    /// The deferred in-flight user message, prefilled after the gap-fill.
    deferred_user: Option<Arc<Vec<u32>>>,
    /// Accounting: total glue tokens collected (the reproject cost scales here).
    n_glue_tokens: usize,
    /// Accounting: sealed turn / section segments actually injected into the
    /// slot, the tokens they carried, and any that were resolved-but-dropped.
    /// A non-zero `skipped_*` means the model is decoding against a context
    /// that is missing selected history — surfaced in the assembly summary so
    /// a dropped turn never goes unnoticed.
    sealed_turns: usize,
    sealed_sections: usize,
    sealed_tokens: usize,
    skipped_turns: usize,
    skipped_sections: usize,
}

impl SegmentWalker {
    fn new() -> Self {
        Self {
            run_tokens: Vec::new(),
            glue_tokens: Vec::new(),
            col_prefix: Vec::new(),
            col_new: Vec::new(),
            logical_pos: 0,
            last_was_sealed: false,
            deferred_user: None,
            n_glue_tokens: 0,
            sealed_turns: 0,
            sealed_sections: 0,
            sealed_tokens: 0,
            skipped_turns: 0,
            skipped_sections: 0,
        }
    }

    /// Close the current glue run: append its tokens to the new region and stamp
    /// each with its logical position. No forward pass — the gap-fill runs once,
    /// after the whole walk.
    fn collect_run(&mut self) {
        if self.run_tokens.is_empty() {
            return;
        }
        let n = self.run_tokens.len();
        for k in 0..n {
            self.col_new.push(self.logical_pos + k as u32);
        }
        self.glue_tokens.append(&mut self.run_tokens);
        self.logical_pos += n as u32;
        self.n_glue_tokens += n;
    }

    /// Record `count` sealed (prefix) columns at the current logical position
    /// and advance. Called by the sealed-inject helpers after a successful
    /// inject so the prefix's `col_actual_pos` matches the position_map order.
    fn record_sealed(&mut self, count: usize) {
        for k in 0..count {
            self.col_prefix.push(self.logical_pos + k as u32);
        }
        self.logical_pos += count as u32;
        self.last_was_sealed = true;
    }
}

// ── Sealed-segment injection ─────────────────────────────────────────────────

fn inject_sealed_section(
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    sid: SectionId,
) -> Result<(), ConversationError> {
    let parent_id = ctx.parent_id;
    let sealed = match ctx.conversation.read().section_sealed_of(sid) {
        Some(s) => s,
        None => {
            walker.skipped_sections += 1;
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
        .set_section_block_range(sid, start_block as u64, end_block as u64);

    log_injected_tokens(ctx, &toks);
    walker.sealed_sections += 1;
    walker.sealed_tokens += sealed[0].token_count;
    walker.record_sealed(sealed[0].token_count);
    Ok(())
}

fn inject_sealed_turn(
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
        walker.skipped_turns += 1;
        tracing::warn!(
            target: "candle_conversation::scheduler::reproject",
            slot = parent_id.0,
            group = group.raw(),
            index = index.0,
            "apply_projection: no timeline for group; dropping selected turn"
        );
        return Ok(());
    };

    let sealed = match ctx.conversation.read().turn_sealed_of(timeline, index) {
        Some(s) => s,
        None => {
            walker.skipped_turns += 1;
            // Distinguish the two failure modes:
            //   entry_exists = false → the turn does not exist under the
            //     timeline we resolved → timeline mismatch (the assembler used
            //     `slot_target.timeline`; the elevate-set builder used
            //     `timelines_for_group(group).next()` — if the group has >1
            //     timeline these disagree).
            //   entry_exists = true  → the turn exists here but its residence
            //     hot is None → elevate promoted a different (timeline, index)
            //     so this one was never lifted into VRAM.
            let conv = ctx.conversation.read();
            let group_timelines: Vec<u64> =
                conv.timelines_for_group(group).map(|t| t.raw()).collect();
            let entry_exists = conv.turn_indices(timeline).any(|i| i == index);
            // Tier flags distinguish the two root causes:
            //   all false (tier_less) → the turn has no K/V in any tier, so the
            //     projection selected an empty ghost/anchor (or a turn whose K/V
            //     was lost on replay) — nothing elevate could ever inject.
            //   warm/cold true → the K/V exists but wasn't promoted → elevate gap.
            let (tier_hot, tier_warm, tier_cold) = match conv.turn_tier_state(timeline, index) {
                Some(t) => (Some(t.hot), Some(t.warm), Some(t.cold)),
                None => (None, None, None),
            };
            let tok_count = conv.turn_token_count_of(timeline, index);
            drop(conv);
            tracing::warn!(
                target: "candle_conversation::scheduler::reproject",
                slot = parent_id.0,
                group = group.raw(),
                index = index.0,
                used_timeline = timeline.raw(),
                slot_timeline = ?ctx.slot_target.map(|t| t.timeline.raw()),
                group_timelines = ?group_timelines,
                entry_exists,
                tier_hot = ?tier_hot,
                tier_warm = ?tier_warm,
                tier_cold = ?tier_cold,
                tok_count,
                "apply_projection: selected turn has no hot sealed K/V; dropping it"
            );
            return Ok(());
        }
    };
    if sealed.is_empty() {
        walker.skipped_turns += 1;
        tracing::warn!(
            target: "candle_conversation::scheduler::reproject",
            slot = parent_id.0,
            group = group.raw(),
            index = index.0,
            "apply_projection: selected turn sealed K/V is empty; dropping it"
        );
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

    log_injected_tokens(ctx, &toks);
    walker.sealed_turns += 1;
    walker.sealed_tokens += sealed[0].token_count;
    walker.record_sealed(sealed[0].token_count);
    Ok(())
}

/// Inject the user half of a turn onto the slot as a zero-copy sealed
/// window.  Mirrors [`inject_sealed_turn`] but resolves the half via
/// [`Substrate::turn_user_sealed_half`] and does not update any
/// turn/section block range (a half is a transient compression input, not
/// a projected turn).  Used only by the summary-tree compression passes.
fn inject_sealed_turn_half(
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    group: GroupId,
    index: TurnIndex,
) -> Result<(), ConversationError> {
    let parent_id = ctx.parent_id;

    let timeline: Option<TimelineId> = match ctx.slot_target {
        Some(tgt) if group == tgt.group => Some(tgt.timeline),
        _ => ctx.conversation.read().timelines_for_group(group).next(),
    };
    let Some(timeline) = timeline else {
        walker.skipped_turns += 1;
        tracing::warn!(
            target: "candle_conversation::scheduler::reproject",
            slot = parent_id.0,
            group = group.raw(),
            index = index.0,
            "compress: no timeline for group; dropping turn-half"
        );
        return Ok(());
    };

    let sealed = match ctx
        .conversation
        .read()
        .turn_user_sealed_half(timeline, index)
    {
        Some(s) if !s.is_empty() && s[0].token_count > 0 => s,
        _ => {
            // Empty half (e.g. a turn with no user content) — nothing to
            // inject. Not an error: the other pass carries that content.
            return Ok(());
        }
    };
    inject_arc_sealed(ctx.session, parent_id, ctx.chunk_size, &sealed)?;

    walker.sealed_turns += 1;
    walker.sealed_tokens += sealed[0].token_count;
    walker.record_sealed(sealed[0].token_count);
    Ok(())
}

fn inject_arc_sealed(
    session: &mut BatchedInferenceSession,
    parent_id: SequenceId,
    chunk_size: usize,
    sealed: &Arc<Vec<SealedSequence>>,
) -> Result<(), ConversationError> {
    let _g = profile::span("inject:arc_sealed");
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
    walker.last_was_sealed = true;
    Ok(())
}

// ── Live prefill drive ───────────────────────────────────────────────────────

/// Forward `tokens` through the model on the slot, chunked by
/// `max_prefill_pass_tokens`, writing their K/V into the slot's writer tail.
fn forward_tokens(ctx: &mut ApplyContext<'_>, tokens: &[u32]) -> Result<(), ConversationError> {
    let parent_id = ctx.parent_id;
    let mut offset = 0;
    while offset < tokens.len() {
        let chunk_len = (tokens.len() - offset).min(ctx.max_prefill_pass_tokens);
        let slice = &tokens[offset..offset + chunk_len];
        let input = Tensor::new(slice, ctx.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(ConversationError::Model)?;
        {
            let _g = profile::span("prefill:forward");
            let _logits = ctx
                .model
                .forward_batched(ctx.session, &[parent_id.0], &[input])
                .map_err(ConversationError::Model)?;
        }
        ctx.session
            .advance_sequence(parent_id.0, chunk_len)
            .map_err(ConversationError::Model)?;
        offset += chunk_len;
    }
    Ok(())
}

/// If the slot's tail is Arc-shared from a prior Sealed inject, push a fresh
/// writer chunk so this prefill's writes don't alias that partial chunk.
fn push_empty_if_sealed(
    ctx: &mut ApplyContext<'_>,
    last_was_sealed: bool,
) -> Result<(), ConversationError> {
    if last_was_sealed {
        let _g = profile::span("prefill:push_empty");
        ctx.session
            .push_empty_writer_chunk(ctx.parent_id.0)
            .map_err(ConversationError::Model)?;
    }
    Ok(())
}

/// Captures the just-written per-layer
/// `SealedSequence` (for `pending_user_part`, which re-injects the in-flight
/// turn's user K/V across mid-decode reprojections instead of re-prefilling it).
fn drive_prefill_and_capture(
    ctx: &mut ApplyContext<'_>,
    tokens: &[u32],
    last_was_sealed: bool,
) -> Result<Arc<Vec<SealedSequence>>, ConversationError> {
    let parent_id = ctx.parent_id;
    push_empty_if_sealed(ctx, last_was_sealed)?;
    let start_block = ctx
        .session
        .sequence_block_count(parent_id.0)
        .ok_or_else(|| {
            ConversationError::Channel(format!(
                "apply_projection: slot {} not in session",
                parent_id
            ))
        })?;
    forward_tokens(ctx, tokens)?;
    let end_block = ctx
        .session
        .sequence_block_count(parent_id.0)
        .ok_or_else(|| {
            ConversationError::Channel(format!(
                "apply_projection: slot {} not in session",
                parent_id
            ))
        })?;

    let captured = {
        let _g = profile::span("prefill:snapshot");
        let full = ctx
            .session
            .snapshot_sequence_per_layer(parent_id.0)
            .map_err(ConversationError::Model)?;
        slice_per_layer_sealed(&full, start_block, end_block)
    };
    log_injected_tokens(ctx, tokens);
    Ok(Arc::new(captured))
}

/// Fire ONE batched gap-fill forward over `plans` (the cross-conversation wave),
/// then commit each slot's glue. Plans with no glue are skipped. Every slot's
/// glue is the new region of a ragged prefill (per-slot kv_lens via cu_seqlens);
/// a single flat `col_actual_pos` covers every slot's sealed prefix ++ glue, so
/// each glue token attends only logically-earlier columns within its own slot.
/// The resulting `[sealed | glue]` slot decodes correctly because attention is
/// order-invariant over keys and K is stored un-rotated (re-RoPE'd at read).
pub(super) fn fire_gap_fill_batch(
    session: &mut BatchedInferenceSession,
    model: &(dyn ManagedBatchedModel + Send),
    device: &Device,
    plans: &[&GapFillPlan],
) -> Result<(), ConversationError> {
    let active: Vec<&GapFillPlan> = plans
        .iter()
        .copied()
        .filter(|p| !p.glue_tokens.is_empty())
        .collect();
    if active.is_empty() {
        return Ok(());
    }
    let mut ids: Vec<usize> = Vec::with_capacity(active.len());
    let mut inputs: Vec<Tensor> = Vec::with_capacity(active.len());
    let mut glue_cols: Vec<Vec<u32>> = Vec::with_capacity(active.len());
    for p in &active {
        ids.push(p.parent_id.0);
        let input = Tensor::new(p.glue_tokens.as_slice(), device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(ConversationError::Model)?;
        inputs.push(input);
        glue_cols.push(p.col_actual_pos.clone());
    }
    // Stage each slot's TRUE column positions (sealed prefix ++ glue), aligned
    // with `ids`. The forward routes the HD128 glue to the paged-glue kernel,
    // which streams the quantized prefix once (dequant-once) and masks every
    // glue island by logical position via `col_actual_pos`.
    session.set_pending_glue(glue_cols);
    // Clear the per-op pipeline profile so the snapshot below covers only this
    // gap-fill forward (attn_core / mlp_ffn / qkv / out_proj, summed over layers).
    #[cfg(feature = "profile")]
    let _ = candle_transformers::models::profile::pipeline_snapshot_and_reset();
    {
        let _g = profile::span("prefill:gap_fill");
        model
            .forward_batched(session, &ids, &inputs)
            .map_err(ConversationError::Model)?;
    }
    #[cfg(feature = "profile")]
    {
        let snap = candle_transformers::models::profile::pipeline_snapshot_and_reset();
        let mut parts: Vec<String> = snap
            .entries
            .iter()
            .map(|(n, ms, c)| format!("{n}={ms:.1}ms({c})"))
            .collect();
        parts.sort_by(|a, b| b.cmp(a));
        tracing::info!(
            target: "candle_conversation::scheduler::reproject",
            n_slots = active.len(),
            "gap-fill forward op breakdown: {}",
            parts.join("  ")
        );
    }
    // Commit each slot's glue (consecutive writer tail).
    for p in &active {
        session
            .advance_sequence(p.parent_id.0, p.glue_tokens.len())
            .map_err(ConversationError::Model)?;
    }
    Ok(())
}

/// Finish phase: log the glue, prefill the deferred in-flight user message
/// against the now-committed `[sealed | glue]` prefix, and re-attach the writer
/// tail. Call after [`fire_gap_fill_batch`] has fired + committed the glue.
pub(super) fn apply_segments_finish(
    state: &mut SlotState,
    ctx: &mut ApplyContext<'_>,
    plan: GapFillPlan,
) -> Result<(), ConversationError> {
    let GapFillPlan {
        parent_id,
        glue_tokens,
        col_actual_pos: _,
        deferred_user,
        tail_per_layer,
        n_glue_tokens,
    } = plan;

    if !glue_tokens.is_empty() {
        log_injected_tokens(ctx, &glue_tokens);
    }

    // The in-flight user message prefills last, against [sealed | glue]. After
    // the gap-fill the slot ends in glue writer chunks (last_was_sealed=false);
    // with no glue it ends in the sealed inject (last_was_sealed=true).
    if let Some(tokens) = deferred_user {
        let mut walker = SegmentWalker::new();
        walker.last_was_sealed = glue_tokens.is_empty();
        handle_new_user_message(state, ctx, &mut walker, &tokens)?;
    }

    {
        let _g = profile::span("apply:restore_tail");
        restore_tail(ctx.session, parent_id, tail_per_layer)?;
    }

    tracing::trace!(
        target: "candle_conversation::scheduler::reproject",
        n_glue_tokens = n_glue_tokens,
        "apply_segments breakdown (single gap-fill glue prefill)",
    );
    Ok(())
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
    fn slot_state_default_is_empty() {
        let s = SlotState::default();
        assert!(s.pending_user_part.is_none());
    }

    fn test_markers() -> BoundaryMarkers {
        BoundaryMarkers {
            user_start: Arc::new(vec![100]),
            assistant_end: Arc::new(vec![200]),
            user_end: Arc::new(vec![101]),
            assistant_start: Arc::new(vec![201]),
            no_think: Arc::new(vec![]),
        }
    }

    #[test]
    fn assemble_pieces_wraps_turns_merges_glue_and_defers_user() {
        let m = test_markers();
        let segments = vec![
            generated_seg("a", 0, &[1, 2]),
            section_seg(7),
            turn_seg(1, 2, 3),
            turn_seg(1, 2, 4),
            generated_seg("b", 0, &[3]),
            ProjectionSegment::NewUserMessage {
                tokens: Arc::new(vec![9, 9]),
            },
        ];
        let pieces = assemble_pieces(&segments, &m);
        assert_eq!(
            pieces,
            vec![
                // leading Generated run
                AssembledPiece::Glue(vec![1, 2]),
                AssembledPiece::Section(SectionId::new(7)),
                // user_start flushed just before the first turn
                AssembledPiece::Glue(vec![100]),
                AssembledPiece::Turn {
                    group: GroupId::for_test(2),
                    index: TurnIndex(3),
                    role: crate::Role::Assistant
                },
                // turn 1's assistant_end MERGES with turn 2's user_start in one island
                AssembledPiece::Glue(vec![200, 100]),
                AssembledPiece::Turn {
                    group: GroupId::for_test(2),
                    index: TurnIndex(4),
                    role: crate::Role::Assistant
                },
                // turn 2's assistant_end merges with the trailing Generated run
                AssembledPiece::Glue(vec![200, 3]),
                // in-flight user message deferred past the gap-fill
                AssembledPiece::DeferredUser(Arc::new(vec![9, 9])),
            ]
        );
    }

    #[test]
    fn assemble_pieces_consecutive_generated_form_one_island() {
        let m = test_markers();
        let segments = vec![
            generated_seg("a", 0, &[1]),
            generated_seg("b", 1, &[2, 3]),
            section_seg(5),
        ];
        let pieces = assemble_pieces(&segments, &m);
        assert_eq!(
            pieces,
            vec![
                AssembledPiece::Glue(vec![1, 2, 3]),
                AssembledPiece::Section(SectionId::new(5)),
            ]
        );
    }

    #[test]
    fn assemble_pieces_empty_is_empty() {
        assert!(assemble_pieces(&[], &test_markers()).is_empty());
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
