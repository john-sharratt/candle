//! Per-slot projection assembler — full rebuild of the slot's prefix
//! K/V on every call.
//!
//! Snapshots the slot's writer tail, truncates to zero, then walks the
//! [`assemble_pieces`] output in logical order, building the slot's chunk list
//! IN PLACE:
//!
//! - `Section` / `Turn` — resolve the substrate entry's per-layer sealed K/V and
//!   Arc-clone it onto the slot.
//! - `Glue` (a boundary-marker island) — reserve a real, writer-owned **gap
//!   chunk** at this logical position (zeros, [`reserve_glue_island`]), recording
//!   each token's scatter target + forward bridge window. The gaps' K/V is filled
//!   afterwards by a single batched gap-fill forward ([`fire_gap_fill_batch`]).
//!
//! Because sealed injects and gap chunks land in logical order, every chunk's
//! cumulative-usage `rope_base` equals its true sequence position — the single
//! positional convention the glue and decode kernels both read via `slice_rope`,
//! with no `col_actual_pos` side channel. A convention assert (`slot offset ==
//! walker logical_pos`) guards it. The in-flight user message is deferred to
//! prefill after the gaps are filled; the writer tail is re-attached at the end.
//!
//! The prefix is re-derived from scratch every projection — there is no
//! cross-projection memoisation, so the assembled K/V always reflects the
//! exact segment sequence the resolver selected this call.

use std::collections::HashMap;
use std::sync::Arc;

use candle::{Device, Tensor};
use candle_nn::kv_cache::{SealedSequence, WriterTail};
use candle_transformers::models::batched_inference::{
    BatchedInferenceSession, ManagedBatchedModel, PendingGlue,
};

use crate::conversation::slice_per_layer_sealed;
use crate::error::ConversationError;
use crate::projection::event::{group_name_of, layer_name_of_group, role_str};
use crate::projection::{
    ContentResolver, Conversation, GroupId, MaterializedPiece, ProjectionSegment, ProjectionTarget,
    Schema, SealedKind, SectionId, SelectedTurn, TimelineId, TurnIndex, TurnKey,
};
use crate::scheduler::profile;
use crate::sequence_handle::SequenceId;
use crate::summary_tree::SelectionOrigin;
use crate::summary_tree::TurnKind;

/// Per-slot state owned by the projection assembler.
///
/// `pending_user_part` holds the captured K/V for the in-flight turn's user
/// message — populated when a `NewUserMessage` segment is prefilled, cleared at
/// seal time; survives mid-decode reprojection (which truncates the slot)
/// because it lives here on `SlotState`, not on the slot.
#[derive(Debug, Default)]
pub(super) struct SlotState {
    pub(super) pending_user_part: Option<Arc<Vec<SealedSequence>>>,
    /// The sealed sections + turns this slot's current projection attends over.
    /// Refreshed on every `apply_projection`; consumed by the relief-eviction
    /// path as an explicit protect-list so it never drops the hot copy of a turn
    /// an in-flight prefill/decode is attending. Lives on `SlotState` so it
    /// inherits the slot's lifecycle (removal + fork re-keying) for free.
    pub(super) working_set: SlotWorkingSet,
}

impl SlotState {
    /// Drop the in-flight `NewUserMessage` capture after a successful seal.
    pub(super) fn trim_post_turn(&mut self) {
        self.pending_user_part = None;
    }
}

/// The sealed working set (sections + turns) a slot's projection attends over.
/// See [`SlotState::working_set`] and [`working_set_from_segments`].
#[derive(Debug, Default, Clone)]
pub(super) struct SlotWorkingSet {
    pub(super) sections: Vec<SectionId>,
    pub(super) turns: Vec<TurnKey>,
}

/// Extract the sealed working set from a projection's segments. Only `Sealed`
/// entries reference persisted KV; `Generated`/`NewUserMessage` segments carry
/// live-prefilled tokens with no substrate residence, so they never contribute.
pub(super) fn working_set_from_segments(segments: &[ProjectionSegment]) -> SlotWorkingSet {
    let mut ws = SlotWorkingSet::default();
    for seg in segments {
        if let ProjectionSegment::Sealed(kind) = seg {
            match kind {
                SealedKind::Section(rs) => ws.sections.push(rs.id),
                SealedKind::Turn(rt, _) | SealedKind::TurnHalf(rt) => {
                    if let Some(key) = rt.key() {
                        ws.turns.push(key);
                    }
                }
            }
        }
    }
    ws
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
    /// The role-marker strings, kept beside their tokenised forms so the seal
    /// path can locate role boundaries baked into a turn's assistant text (the
    /// code_read tool exchange `<tool_call>…<tool_response>…confirmation`) by a
    /// plain string match, and slice the sub-segment display text at them.
    pub(crate) user_start_str: String,
    pub(crate) assistant_end_str: String,
    pub(crate) user_end_str: String,
    pub(crate) assistant_start_str: String,
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
            user_start_str: dialect.user_start.to_string(),
            assistant_end_str: dialect.assistant_end.to_string(),
            user_end_str: dialect.user_end.to_string(),
            assistant_start_str: dialect.assistant_start.to_string(),
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
    /// A sealed past turn; its K/V comes from the substrate.  `timeline` is the
    /// turn's conversation, stamped at projection (see [`ResolvedTurn::key`]) and
    /// carried here so the inject path never re-derives it from `group`.  `group`
    /// is retained only for diagnostics.
    Turn {
        group: GroupId,
        index: TurnIndex,
        role: crate::Role,
        timeline: Option<TimelineId>,
    },
    /// The in-flight user message, deferred to prefill after the gap-fill.
    DeferredUser(Arc<Vec<u32>>),
    /// A sealed turn's user-message half only (compression turn-half injection),
    /// with NO per-turn boundary-marker wrapping — the compression pass supplies
    /// its own framing glue.
    TurnHalf {
        group: GroupId,
        index: TurnIndex,
        timeline: Option<TimelineId>,
    },
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
    mut turn_no_think: impl FnMut(Option<TimelineId>, TurnIndex) -> bool,
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
                // Re-render this turn's `/no_think` soft-switch if it was sealed
                // with thinking suppressed — sits right after `user_start`, where
                // Qwen3 honours it, matching the live-turn `no_think_current`
                // glue.  Keeps a re-rendered suppressed turn self-consistent: a
                // `/no_think` opener for its empty `<think></think>`, instead of
                // an unexplained empty block the model learns to mimic.
                if turn_no_think(rt.timeline, rt.index()) {
                    run.extend(markers.no_think.iter().copied());
                }
                flush(&mut run, &mut pieces);
                pieces.push(AssembledPiece::Turn {
                    group: rt.group(),
                    index: rt.index(),
                    role: *role,
                    timeline: rt.timeline,
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
                    timeline: rt.timeline,
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

/// Lay out the **conversation region** of a projection into the ordered
/// [`MaterializedPiece`]s the projection panel renders — real boundary-glue
/// islands interleaved with the sealed turns, built from the SAME
/// [`assemble_pieces`] decision the engine injects from, so the panel's glue can
/// never drift from what the projection really does.
///
/// The system prompt is excluded (it is covered by
/// [`ProjectionSelection::system`]); the live, still-decoding user message and
/// the compression turn-halves are excluded too (the panel renders the live turn
/// from its own `um`/`dm`). Glue islands are decoded to text via `decode`; each
/// turn is classified (layer / group / role / token count / forest kind / why)
/// via `resolver` + `origins`.
pub(crate) fn materialize_conversation(
    segments: &[ProjectionSegment],
    markers: &BoundaryMarkers,
    origins: &HashMap<TurnKey, SelectionOrigin>,
    resolver: &dyn ContentResolver,
    schema: &Schema,
    mut decode: impl FnMut(&[u32]) -> String,
) -> Vec<MaterializedPiece> {
    // Conversation region = from the first sealed turn / turn-half / deferred
    // user onward; everything before is the system prompt.
    let start = segments
        .iter()
        .position(|s| {
            matches!(
                s,
                ProjectionSegment::Sealed(SealedKind::Turn(..))
                    | ProjectionSegment::Sealed(SealedKind::TurnHalf(..))
                    | ProjectionSegment::NewUserMessage { .. }
            )
        })
        .unwrap_or(segments.len());
    // `/no_think` per turn comes from the same suppression bit the assembler
    // reads, so the re-rendered soft-switch matches the engine exactly.
    let no_think =
        |tl: Option<TimelineId>, idx: TurnIndex| tl.is_some_and(|t| resolver.turn_no_think(t, idx));
    assemble_pieces(&segments[start..], markers, no_think)
        .into_iter()
        .filter_map(|piece| match piece {
            AssembledPiece::Glue(tokens) => {
                let text = decode(&tokens);
                (!text.is_empty()).then_some(MaterializedPiece::Glue { text })
            }
            AssembledPiece::Turn {
                group,
                index,
                role,
                timeline,
            } => {
                let key = timeline.map(|tl| TurnKey::new(tl, index));
                Some(MaterializedPiece::Turn {
                    turn: SelectedTurn {
                        layer: layer_name_of_group(schema, group).unwrap_or("").to_string(),
                        group: group_name_of(schema, group)
                            .unwrap_or("conversation")
                            .to_string(),
                        index: index.0,
                        role: role_str(role).to_string(),
                        // A turn's identity is (timeline, index); the group alone is
                        // ambiguous once a group holds many conversations.
                        tokens: key.map_or(0, |k| resolver.turn_token_count(k)) as u32,
                        kind: key.map_or(TurnKind::Normal, |k| resolver.turn_kind(k)),
                        reason: key.and_then(|k| origins.get(&k)).copied(),
                        timeline: timeline.map(|tl| tl.raw()),
                        // Materialized display spine — the turn is shown, and the
                        // belief score isn't threaded here (the carry reads
                        // `ProjectionSelection::turns`, not the materialized pieces).
                        selected: true,
                        score: 0.0,
                    },
                })
            }
            // The live user message is a separate prefill unit (not part of
            // `project()`'s segments), so it never reaches this spine — the panel
            // renders the in-flight turn from `um`/`dm`. The compression turn-half
            // and any stray section are not part of the displayed dialogue spine.
            AssembledPiece::DeferredUser(_)
            | AssembledPiece::TurnHalf { .. }
            | AssembledPiece::Section(_) => None,
        })
        .collect()
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
    /// The glue tokens, in logical order — the Q stream of the gap-fill forward.
    /// Each is reserved **in place** as a real gap chunk at its logical position,
    /// so the glue and decode kernels both derive its sequence position from the
    /// chunk's own `rope_base` (`slice_rope`) — no `col_actual_pos` side channel.
    pub glue_tokens: Vec<u32>,
    /// Per glue token: the writer-chunk block index its K/V scatters into (the
    /// reserved gap). Aligned with `glue_tokens`.
    pub glue_write_slice: Vec<u32>,
    /// Per glue token: the in-block offset within its gap chunk.
    pub glue_write_in_blk: Vec<u32>,
    /// Per glue token: how far it may attend FORWARD past its own position (the
    /// bridge window). `0` == backward-only (causal). Per-token so future code
    /// varies it by glue-island type with no kernel change.
    pub fwd_ahead: Vec<u32>,
    /// The in-flight user message, prefilled in `apply_segments_finish` after
    /// the gap-fill so it lands against the full interleaved prefix.
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
    defer: Option<&mut Vec<GapFillPlan>>,
) -> Result<(), ConversationError> {
    let plan = apply_segments_build(&mut ctx, new_segments)?;
    // When the caller is batching drain gap-fills AND this projection has no
    // deferred user message (ingest / compression — the content prefills through a
    // separate unit), queue the fire and finish now (restore-tail only). The glue
    // K/V isn't read until the caller fires the whole drain's batch as ONE forward
    // at drain end — collapsing N launch-floor gap-fills into one. A projection
    // WITH a deferred user must fire inline (the user prefill attends the glue).
    if let Some(sink) = defer {
        if plan.deferred_user.is_none() {
            sink.push(fire_only(&plan));
            return apply_segments_finish(state, &mut ctx, plan);
        }
    }
    let t_glue = std::time::Instant::now();
    fire_gap_fill_batch(ctx.session, &**ctx.model, ctx.device, &[&plan])?;
    super::drain_add_us(&super::DRAIN_GLUE_US, t_glue.elapsed().as_micros() as u64);
    apply_segments_finish(state, &mut ctx, plan)
}

/// A fire-only clone of a [`GapFillPlan`] (glue scatter inputs; no deferred user /
/// tail), for deferring the gap-fill forward into a batched drain-end fire.
fn fire_only(plan: &GapFillPlan) -> GapFillPlan {
    GapFillPlan {
        parent_id: plan.parent_id,
        glue_tokens: plan.glue_tokens.clone(),
        glue_write_slice: plan.glue_write_slice.clone(),
        glue_write_in_blk: plan.glue_write_in_blk.clone(),
        fwd_ahead: plan.fwd_ahead.clone(),
        deferred_user: None,
        tail_per_layer: Vec::new(),
        n_glue_tokens: plan.n_glue_tokens,
    }
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

    // 4. Walk segments in logical order, building the slot's chunk list IN
    //    PLACE: a sealed section/turn is Arc-injected; a glue island reserves a
    //    real, writer-owned GAP chunk at its position (zeros, filled later). The
    //    chunks therefore land in logical order, so every chunk's cumulative-
    //    usage `rope_base` equals its true sequence position — the single
    //    positional convention the glue and decode kernels both read via
    //    `slice_rope`, with no `col_actual_pos` side channel. A single batched
    //    gap-fill forward then scatters every island's K/V into its gap. The
    //    NewUserMessage is deferred to prefill after the gaps are filled.
    // The glue/order decision is owned by `assemble_pieces` (the single source
    // of truth, shared with the substrate debug view).
    let mut walker = SegmentWalker::new();
    // Look up a sealed turn's recorded `no_think` so the assembler can re-render
    // its `/no_think` switch.  The turn carries its own timeline (stamped at
    // projection), so this reads it directly — no group→timeline resolution.
    // Immutable borrow ends before the mutable walk below.
    let conversation = ctx.conversation;
    let no_think_for = |timeline: Option<TimelineId>, index: TurnIndex| -> bool {
        timeline.is_some_and(|tl| conversation.read().turn_no_think(tl, index))
    };
    let pieces = assemble_pieces(new_segments, ctx.boundary_markers, no_think_for);
    for i in 0..pieces.len() {
        match &pieces[i] {
            AssembledPiece::Glue(tokens) => {
                let fwd = glue_bridge_window(pieces.get(i + 1));
                reserve_glue_island(ctx, &mut walker, tokens, fwd)?;
            }
            AssembledPiece::Section(id) => {
                let t = std::time::Instant::now();
                inject_sealed_section(ctx, &mut walker, *id)?;
                super::drain_add_us(&super::DRAIN_ELEVATE_US, t.elapsed().as_micros() as u64);
            }
            AssembledPiece::Turn {
                group,
                index,
                role,
                timeline,
            } => {
                let t = std::time::Instant::now();
                inject_sealed_turn(ctx, &mut walker, *timeline, *group, *index, *role)?;
                super::drain_add_us(&super::DRAIN_ELEVATE_US, t.elapsed().as_micros() as u64);
            }
            AssembledPiece::DeferredUser(tokens) => {
                walker.deferred_user = Some(tokens.clone());
            }
            AssembledPiece::TurnHalf {
                group,
                index,
                timeline,
            } => {
                let t = std::time::Instant::now();
                inject_sealed_turn_half(ctx, &mut walker, *timeline, *group, *index)?;
                super::drain_add_us(&super::DRAIN_ELEVATE_US, t.elapsed().as_micros() as u64);
            }
        }
    }

    // Convention assert: the slot's chunks were built in logical order (gaps
    // reserved in place), so the slot offset (Σ chunk usage) must equal the
    // walker's running logical position. If they ever diverge, a chunk's
    // `rope_base` no longer equals its sequence position and the kernels would
    // silently mis-RoPE — fail loudly here instead.
    let slot_offset = ctx.session.sequence_offset(parent_id.0).ok_or_else(|| {
        ConversationError::Channel(format!("apply_segments: slot {} not in session", parent_id))
    })?;
    if slot_offset as u32 != walker.logical_pos {
        return Err(ConversationError::Channel(format!(
            "glue convention violated: slot {} offset {} != logical_pos {} \
             (chunk rope_base would diverge from sequence position)",
            parent_id, slot_offset, walker.logical_pos
        )));
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

    Ok(GapFillPlan {
        parent_id,
        glue_tokens: std::mem::take(&mut walker.glue_tokens),
        glue_write_slice: std::mem::take(&mut walker.glue_write_slice),
        glue_write_in_blk: std::mem::take(&mut walker.glue_write_in_blk),
        fwd_ahead: std::mem::take(&mut walker.fwd_ahead),
        deferred_user: walker.deferred_user.take(),
        tail_per_layer,
        n_glue_tokens: walker.n_glue_tokens,
    })
}

/// State carried across a single `apply_segments` walk. Accumulates the gap-fill
/// new region (every glue island) plus each column's TRUE logical position, so
/// the walk emits one batched gap-fill forward instead of N per-island prefills.
struct SegmentWalker {
    /// All glue tokens across every island, in logical order — the gap-fill Q.
    glue_tokens: Vec<u32>,
    /// Per glue token: the reserved gap chunk's block index (its scatter target).
    glue_write_slice: Vec<u32>,
    /// Per glue token: the in-block offset within its gap chunk.
    glue_write_in_blk: Vec<u32>,
    /// Per glue token: forward bridge window (tokens past its own position).
    fwd_ahead: Vec<u32>,
    /// Running TRUE sequence position as the walk advances through sealed+glue.
    /// Equal to `Σ chunk usage` by construction (gaps are reserved in place), so
    /// it cross-checks the slot offset in the convention assert.
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
            glue_tokens: Vec::new(),
            glue_write_slice: Vec::new(),
            glue_write_in_blk: Vec::new(),
            fwd_ahead: Vec::new(),
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

    /// Record `count` sealed (prefix) columns and advance the logical position.
    /// Called by the sealed-inject helpers after a successful inject. The
    /// position itself is carried by the injected chunks' `rope_base`; the walker
    /// only tracks the running total for the convention assert.
    fn record_sealed(&mut self, count: usize) {
        self.logical_pos += count as u32;
        self.last_was_sealed = true;
    }
}

// ── Glue-island gap reservation ──────────────────────────────────────────────

/// Reserve a glue island IN PLACE: append real, writer-owned gap chunk(s) of the
/// island's length at its logical position and record, per token, the
/// `(block, in_blk)` scatter target plus its forward bridge window. The gap's
/// K/V is filled by the batched gap-fill forward; until then it is unread (the
/// kernel scatters before it streams). An island longer than one chunk is split
/// across consecutive gap chunks — the per-token targets stay contiguous, so the
/// `slice_rope` position of token `t` is exactly its logical position.
/// Forward bridge window (tokens) for boundary glue that leads INTO a
/// conversation turn or summary. Opening it lets the `user_start`/`assistant_end`
/// glue attend the first `TURN_BRIDGE_FWD_AHEAD` tokens of the turn it introduces
/// (which is resident in-place downstream in the same slot), so the glue is
/// generated as a lead-in to that turn instead of a blind continuation of what
/// precedes it. B is read-only and only meaningful as far as its heads reach back
/// (~16-32 tokens), so the window is small. All other glue (section boundaries,
/// the pre-user tail) stays backward-only (`0`).
const TURN_BRIDGE_FWD_AHEAD: u32 = 16;

/// Forward bridge window for the glue island that immediately precedes `next` in
/// the assembled piece stream. Open it (`TURN_BRIDGE_FWD_AHEAD`) only when the
/// glue leads INTO a conversation turn/summary — a `Turn`/`TurnHalf` piece, which
/// is injected in-place downstream in the same slot and so is already resident in
/// `[0, kv_len)` for the glue forward to attend. Section boundaries and the glue
/// before the deferred (still un-prefilled) user message have no resident turn
/// ahead, so they stay backward-only (`0`).
fn glue_bridge_window(next: Option<&AssembledPiece>) -> u32 {
    match next {
        Some(AssembledPiece::Turn { .. } | AssembledPiece::TurnHalf { .. }) => {
            TURN_BRIDGE_FWD_AHEAD
        }
        _ => 0,
    }
}

fn reserve_glue_island(
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    tokens: &[u32],
    fwd_ahead: u32,
) -> Result<(), ConversationError> {
    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }
    let parent = ctx.parent_id.0;
    let chunk = ctx.chunk_size.max(1);
    let mut placed = 0;
    while placed < n {
        let take = (n - placed).min(chunk);
        // The gap is full-by-construction: it returns the base of its valid tail
        // window, and we scatter the island's K/V into `[base, base + take)`.
        // Using the reservation's own base (not a recomputed one) means the write
        // window can never drift from the chunk's `slice_offset`; the column
        // position stays `slice_rope + (in_blk - offset)` = `rope_base + i`, so
        // logical order is preserved.
        let (gap_blk, in_blk_base) = ctx
            .session
            .reserve_glue_gap(parent, take as u32)
            .map_err(ConversationError::Model)?;
        for i in 0..take as u32 {
            walker.glue_write_slice.push(gap_blk as u32);
            walker.glue_write_in_blk.push(in_blk_base + i);
            // Per-island forward bridge window, chosen by the caller from the
            // following piece: nonzero when this glue leads into a conversation
            // turn/summary (see `TURN_BRIDGE_FWD_AHEAD`), `0` (causal) otherwise.
            walker.fwd_ahead.push(fwd_ahead);
        }
        placed += take;
    }
    walker.glue_tokens.extend_from_slice(tokens);
    walker.logical_pos += n as u32;
    // A gap is a complete region like a sealed inject: the next live prefill
    // (the deferred user message) must push a fresh writer chunk after it rather
    // than extend into it.
    walker.last_was_sealed = true;
    walker.n_glue_tokens += n;
    log_injected_tokens(ctx, tokens);
    Ok(())
}

// ── Sealed-segment injection ─────────────────────────────────────────────────

fn inject_sealed_section(
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    sid: SectionId,
) -> Result<(), ConversationError> {
    let parent_id = ctx.parent_id;
    // Drop the scrutinee read guard before the match body (see the same note in
    // `inject_sealed_turn`): defensive — no arm here re-locks today, but the
    // match-scrutinee-temporary rule would turn any future re-`read()` in an arm
    // into a recursive-read self-deadlock under the writer-priority `RwLock`.
    let sealed = ctx.conversation.read().section_sealed_of(sid);
    let sealed = match sealed {
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
    timeline: Option<TimelineId>,
    group: GroupId,
    index: TurnIndex,
    _part: crate::Role,
) -> Result<(), ConversationError> {
    let parent_id = ctx.parent_id;

    // The turn carries its own conversation (stamped at projection); this path
    // never re-derives it from `group`.  A `None` here is a degenerate (mock /
    // genuinely untracked) turn, surfaced loudly rather than silently dropped.
    let Some(timeline) = timeline else {
        walker.skipped_turns += 1;
        tracing::warn!(
            target: "candle_conversation::scheduler::reproject",
            slot = parent_id.0,
            group = group.raw(),
            index = index.0,
            "apply_projection: turn carries no timeline; dropping selected turn"
        );
        return Ok(());
    };

    // Bind the read to a local so the scrutinee's `RwLockReadGuard` drops BEFORE
    // the match body runs. A guard created in a match scrutinee lives for the
    // WHOLE match (Rust temporary-lifetime rule), so the `None` arm's re-`read()`
    // below would be a *recursive* read on the substrate lock — which
    // self-deadlocks against a writer queued between the two acquisitions under
    // the writer-preferring std `RwLock`. Observed live during a heavy
    // concurrent re-ingest: the scheduler held this scrutinee read while blocked
    // re-reading it, and the ingest write path (adopt/couple/mint) had a writer
    // queued, so writer-priority refused the second read forever. `turn_sealed_of`
    // returns an owned `Arc`, so the early drop is free.
    let sealed = ctx.conversation.read().turn_sealed_of(timeline, index);
    let sealed = match sealed {
        Some(s) => s,
        None => {
            walker.skipped_turns += 1;
            // Distinguish the two failure modes (timeline resolution is now
            // unified through `Substrate::resolve_turn_timeline`, so a stale
            // group→timeline DISAGREEMENT between the selection and apply paths is
            // no longer one of them):
            //   entry_exists = false → the turn genuinely isn't tracked under the
            //     resolved timeline → a selection/registration gap (the projection
            //     picked a turn the substrate doesn't hold here).
            //   entry_exists = true  → the turn exists here but its residence
            //     hot is None → elevate promoted a different (timeline, index)
            //     so this one was never lifted into VRAM.
            let conv = ctx.conversation.read();
            // Just the group's timeline COUNT — dumping every raw id (a group can
            // hold dozens under bulk ingest) bloats the log without adding signal;
            // the count plus the tier flags below is what localises the cause.
            let group_timeline_count = conv.timelines_for_group(group).count();
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
                group_timeline_count,
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
    timeline: Option<TimelineId>,
    group: GroupId,
    index: TurnIndex,
) -> Result<(), ConversationError> {
    let parent_id = ctx.parent_id;

    // Carried from projection — see `inject_sealed_turn`.
    let Some(timeline) = timeline else {
        walker.skipped_turns += 1;
        tracing::warn!(
            target: "candle_conversation::scheduler::reproject",
            slot = parent_id.0,
            group = group.raw(),
            index = index.0,
            "compress: turn-half carries no timeline; dropping turn-half"
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
    // Drain-path prefill accounting: this forward runs during a submission drain
    // (the newly-ingested turn's content) AND during reproject (re-prefilling the
    // in-flight user message); `drain_add_us` only counts the former. The tokens go
    // to `DRAIN_PREFILL_TOKENS` so the Prefill phase can reclaim them from the
    // otherwise-token-less projection band.
    let t_fwd = std::time::Instant::now();
    let mut offset = 0;
    while offset < tokens.len() {
        let chunk_len = (tokens.len() - offset).min(ctx.max_prefill_pass_tokens);
        let slice = &tokens[offset..offset + chunk_len];
        let input = Tensor::new(slice, ctx.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(ConversationError::Model)?;
        {
            let _g = profile::span("prefill:forward");
            let nl = ctx.model.num_layers().max(1);
            let _logits = ctx
                .model
                .forward_wave(
                    ctx.session,
                    &[],
                    &[],
                    &[parent_id.0],
                    &[input],
                    &[],
                    &[],
                    0,
                    nl,
                    None,
                )
                .map(|s| s.logits.unwrap_or_default())
                .map_err(ConversationError::Model)?;
        }
        ctx.session
            .advance_sequence(parent_id.0, chunk_len)
            .map_err(ConversationError::Model)?;
        offset += chunk_len;
    }
    super::drain_add_us(&super::DRAIN_PREFILL_US, t_fwd.elapsed().as_micros() as u64);
    super::drain_add_us(&super::DRAIN_PREFILL_TOKENS, tokens.len() as u64);
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
/// then fill each slot's gaps. Plans with no glue are skipped. The gaps were
/// reserved IN PLACE during the walk, so this forward only scatters each glue
/// token's K/V into its gap (by explicit `(slice, in_blk)` target) and computes
/// its attention; the slot length already counts the gaps, so there is no commit
/// step. Every column's sequence position comes from its chunk's `rope_base`
/// (`slice_rope`) — the same convention the decode reads — so the interleaved
/// slot decodes correctly with no `col_actual_pos` side channel.
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
    let mut pending: Vec<PendingGlue> = Vec::with_capacity(active.len());
    for p in &active {
        ids.push(p.parent_id.0);
        let input = Tensor::new(p.glue_tokens.as_slice(), device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(ConversationError::Model)?;
        inputs.push(input);
        pending.push(PendingGlue {
            write_slice: p.glue_write_slice.clone(),
            write_in_blk: p.glue_write_in_blk.clone(),
            fwd_ahead: p.fwd_ahead.clone(),
        });
    }
    // Stage each slot's per-token gap scatter target + forward bridge window,
    // aligned with `ids`. The forward routes the HD128 glue to the paged-glue
    // kernel, which streams the quantized slot once (dequant-once), positions
    // every column by its chunk `rope_base`, and masks each glue token by
    // `cpos > row_pos + fwd_ahead[t]`.
    session.set_pending_glue(pending);
    // Clear the per-op pipeline profile so the snapshot below covers only this
    // gap-fill forward (attn_core / mlp_ffn / qkv / out_proj, summed over layers).
    #[cfg(feature = "profile")]
    let _ = candle_transformers::models::profile::pipeline_snapshot_and_reset();
    {
        let _g = profile::span("prefill:gap_fill");
        // Route the glue islands through the wave's GLUE group so the pending
        // per-slot scatter descriptors (staged above) drive the paged-glue kernel.
        // A glue-only wave carries no logits (it only scatters K/V) and the result
        // is discarded — the forward's whole purpose is the K/V side effect.
        let n = model.num_layers();
        model
            .forward_wave(session, &[], &[], &[], &[], &ids, &inputs, 0, n, None)
            .map_err(ConversationError::Model)?;
    }
    // The gap-fill must ONLY scatter K/V into the pre-reserved gaps — it must not
    // advance the slot. If a trailing-append advance ever creeps back in, the
    // session offset (used by the next prefill) and the backing's actual length
    // (sum of chunk usages) desync, and the final gap chunk's usage is inflated;
    // the symptom is an out-of-bounds panic deep in a later prefill's write-region
    // extension. Assert the invariant `session.offset == backing length` here, at
    // the source, so the failure is named instead of a cryptic index panic.
    for &id in &ids {
        let session_off = session.sequence_offset(id).unwrap_or(0);
        let backing_len = session
            .sequence_caches(id)
            .map(|c| c.current_seq_len())
            .unwrap_or(session_off);
        if backing_len != session_off {
            return Err(ConversationError::Channel(format!(
                "gap-fill desynced slot {id}: session offset {session_off} != backing length \
                 {backing_len}. The glue forward advanced the slot instead of only scattering \
                 into pre-reserved gaps."
            )));
        }
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
    // No commit step: the gaps were reserved in place during the walk (the slot
    // offset already counts them), so this forward only filled their K/V.
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
        glue_tokens: _,
        glue_write_slice: _,
        glue_write_in_blk: _,
        fwd_ahead: _,
        deferred_user,
        tail_per_layer,
        n_glue_tokens,
    } = plan;
    // The glue tokens were already logged into the slot_tokens debug view at
    // their interleaved positions during the walk (`reserve_glue_island`).

    // The in-flight user message prefills last, after the now-filled gaps. The
    // slot always ends in a complete region (a reserved gap chunk, or a sealed
    // inject when there is no glue), so the user message must push a fresh writer
    // chunk rather than extend it — `last_was_sealed = true` unconditionally.
    if let Some(tokens) = deferred_user {
        let mut walker = SegmentWalker::new();
        walker.last_was_sealed = true;
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
    use crate::projection::{GroupId, LayerId, ResolvedSection, ResolvedTurn, TimelineId, TurnId};

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
                timeline: Some(TimelineId::for_test(group as u64)),
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
        assert!(s.working_set.sections.is_empty() && s.working_set.turns.is_empty());
    }

    #[test]
    fn working_set_extracts_sealed_sections_and_turns() {
        // Sealed sections/turns contribute their ids; Generated glue does not.
        let segments = vec![
            section_seg(1),
            turn_seg(0, 5, 2),
            generated_seg("glue", 0, &[9, 9]),
            section_seg(3),
            turn_seg(0, 5, 7),
        ];
        let ws = working_set_from_segments(&segments);
        assert_eq!(ws.sections, vec![SectionId::new(1), SectionId::new(3)]);
        assert_eq!(
            ws.turns,
            vec![
                TurnKey::new(TimelineId::for_test(5), TurnIndex(2)),
                TurnKey::new(TimelineId::for_test(5), TurnIndex(7)),
            ]
        );

        // A projection of only live-prefilled glue attends no sealed KV.
        let none = working_set_from_segments(&[generated_seg("g", 0, &[1])]);
        assert!(none.sections.is_empty() && none.turns.is_empty());
    }

    fn test_markers() -> BoundaryMarkers {
        BoundaryMarkers {
            user_start: Arc::new(vec![100]),
            assistant_end: Arc::new(vec![200]),
            user_end: Arc::new(vec![101]),
            assistant_start: Arc::new(vec![201]),
            no_think: Arc::new(vec![]),
            user_start_str: "<|im_start|>user\n".into(),
            assistant_end_str: "<|im_end|>\n".into(),
            user_end_str: "<|im_end|>\n".into(),
            assistant_start_str: "<|im_start|>assistant\n".into(),
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
        let pieces = assemble_pieces(&segments, &m, |_, _| false);
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
                    role: crate::Role::Assistant,
                    timeline: Some(TimelineId::for_test(2)),
                },
                // turn 1's assistant_end MERGES with turn 2's user_start in one island
                AssembledPiece::Glue(vec![200, 100]),
                AssembledPiece::Turn {
                    group: GroupId::for_test(2),
                    index: TurnIndex(4),
                    role: crate::Role::Assistant,
                    timeline: Some(TimelineId::for_test(2)),
                },
                // turn 2's assistant_end merges with the trailing Generated run
                AssembledPiece::Glue(vec![200, 3]),
                // in-flight user message deferred past the gap-fill
                AssembledPiece::DeferredUser(Arc::new(vec![9, 9])),
            ]
        );
    }

    #[test]
    fn glue_bridge_window_opens_only_into_turns() {
        let turn = AssembledPiece::Turn {
            group: GroupId::for_test(2),
            index: TurnIndex(3),
            role: crate::Role::Assistant,
            timeline: Some(TimelineId::for_test(2)),
        };
        let turn_half = AssembledPiece::TurnHalf {
            group: GroupId::for_test(2),
            index: TurnIndex(3),
            timeline: Some(TimelineId::for_test(2)),
        };
        // Leads into a resident turn/summary → bridge window opens.
        assert_eq!(glue_bridge_window(Some(&turn)), TURN_BRIDGE_FWD_AHEAD);
        assert_eq!(glue_bridge_window(Some(&turn_half)), TURN_BRIDGE_FWD_AHEAD);
        // No resident turn ahead → backward-only.
        assert_eq!(
            glue_bridge_window(Some(&AssembledPiece::Section(SectionId::new(7)))),
            0
        );
        assert_eq!(
            glue_bridge_window(Some(&AssembledPiece::DeferredUser(Arc::new(vec![9])))),
            0
        );
        assert_eq!(glue_bridge_window(None), 0);
    }

    #[test]
    fn glue_bridge_window_matches_assembled_stream() {
        // Walk the same stream as `assemble_pieces_wraps_turns_…`: every glue
        // island's window is decided by the piece that follows it.
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
        let pieces = assemble_pieces(&segments, &m, |_, _| false);
        let windows: Vec<u32> = (0..pieces.len())
            .filter(|&i| matches!(pieces[i], AssembledPiece::Glue(_)))
            .map(|i| glue_bridge_window(pieces.get(i + 1)))
            .collect();
        // Glue islands in order: leading run→Section (0), user_start→Turn (16),
        // assistant_end++user_start→Turn (16), assistant_end++run→DeferredUser (0).
        assert_eq!(
            windows,
            vec![0, TURN_BRIDGE_FWD_AHEAD, TURN_BRIDGE_FWD_AHEAD, 0]
        );
    }

    #[test]
    fn assemble_pieces_reinjects_no_think_for_recorded_suppressed_turn() {
        let mut m = test_markers();
        m.no_think = Arc::new(vec![42]); // the `/no_think` soft-switch tokens
        let segments = vec![turn_seg(1, 2, 3)];

        // Recorded thinking-ON: only `user_start` [100] precedes the turn.
        let on = assemble_pieces(&segments, &m, |_, _| false);
        assert_eq!(on[0], AssembledPiece::Glue(vec![100]));

        // Recorded thinking-SUPPRESSED: `user_start` [100] ++ `/no_think` [42],
        // so the re-rendered prior turn shows its switch.
        let off = assemble_pieces(&segments, &m, |_, _| true);
        assert_eq!(off[0], AssembledPiece::Glue(vec![100, 42]));
    }

    #[test]
    fn assemble_pieces_consecutive_generated_form_one_island() {
        let m = test_markers();
        let segments = vec![
            generated_seg("a", 0, &[1]),
            generated_seg("b", 1, &[2, 3]),
            section_seg(5),
        ];
        let pieces = assemble_pieces(&segments, &m, |_, _| false);
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
        assert!(assemble_pieces(&[], &test_markers(), |_, _| false).is_empty());
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
