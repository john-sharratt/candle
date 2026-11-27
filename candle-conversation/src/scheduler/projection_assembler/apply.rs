//! Applies a projection's segments onto a slot, inline or as the build and finish
//! halves around a batched gap-fill.
//!
//! A single slot runs build, fire and finish together. A drain that batches its
//! gap-fills queues a fire-only copy and finishes at once, since the glue's K/V
//! is not read until the drain's one forward. The splice helpers cut the slot
//! after its last held piece and then apply each edit from the back, so every
//! edit's blocks and positions are still the slot's as it stood. A deferred apply
//! reports that the slot does not yet hold its segments.

use super::*;

/// Borrowed scheduler state the assembler needs in order to run.
///
/// `model` is required so cache-miss runs can drive a synchronous
/// `forward_batched` to compute and capture the missing K/V;
/// `max_prefill_pass_tokens` caps the per-pass token count to keep activation
/// buffers bounded.
pub(in super::super) struct ApplyContext<'a> {
    pub(in super::super) session: &'a mut BatchedInferenceSession,
    pub(in super::super) model: &'a mut Box<dyn ManagedBatchedModel + Send>,
    pub(in super::super) conversation: &'a Conversation,
    pub(in super::super) slot_target: Option<ProjectionTarget>,
    pub(in super::super) parent_id: SequenceId,
    pub(in super::super) chunk_size: usize,
    pub(in super::super) max_prefill_pass_tokens: usize,
    /// Used only under `feature = "context-dump"`; kept unconditionally
    /// so the public shape stays stable across feature toggles.
    #[cfg_attr(not(feature = "context-dump"), allow(dead_code))]
    pub(in super::super) tokenizer: &'a tokenizers::Tokenizer,
    pub(in super::super) slot_tokens: &'a mut HashMap<SequenceId, Vec<u32>>,
    /// Pre-tokenised inter-turn boundary markers — see [`BoundaryMarkers`].
    ///
    /// No longer emitted around sealed turns (each turn owns its own); still
    /// carried because the compression probe frames its synthetic exchange with
    /// them and the debug view decodes them.
    pub(in super::super) boundary_markers: &'a BoundaryMarkers,
    /// Each sealed section's per-position model state, so a borrowed section
    /// brings its index rows along with its chunks. Empty for a model that
    /// carries none — see `Scheduler::section_positional`.
    pub(in super::super) section_positional: &'a SectionPages,
}

/// A built-but-not-yet-fired gap-fill descriptor for one slot. Owned (no
/// borrows) so the cross-conversation wave can collect plans from many slots,
/// fire one batched multi-slot forward, then finish each slot independently.
pub(in super::super) struct GapFillPlan {
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
    /// The slot's token length BEFORE the rebuild took it apart, so
    /// [`apply_segments_finish`] can check that the rebuild put back everything
    /// it removed. `None` where the plan was cloned for a fire-only pass and
    /// there is nothing to compare against.
    pub held_before: Option<usize>,
    /// How much of `held_before` the previous projection placed
    /// ([`SlotState::placed`] when the rebuild took the slot apart).
    pub placed_before: usize,
    /// Glue token count actually COMPUTED by the wave (cache-reused islands are
    /// excluded — they were Arc-injected during the walk).
    pub n_glue_tokens: usize,
    /// Every glue island of this projection, in walk order: its content-context
    /// key, its gap-chunk block range, and whether it was reused from the
    /// island cache. Freshly-computed islands are captured into
    /// [`SlotState::glue_islands`] after the wave fires (`apply_segments_finish`
    /// with `capture_islands`).
    pub islands: Vec<PlannedIsland>,
    /// Whether the walk placed every sealed section and turn it was given. A
    /// skipped one (not hot when the walk reached it) leaves the slot short of
    /// what its segment list describes, so the list's identity does not name
    /// the slot's contents and a later reprojection must not treat it as
    /// already built.
    pub complete: bool,
}

/// One glue island's plan entry — see [`GapFillPlan::islands`].
#[derive(Debug, Clone)]
pub(in super::super) struct PlannedIsland {
    /// Content-context hash: prefix pieces + island tokens + bridge target.
    pub key: u64,
    /// Block range of the island's chunk(s) in the rebuilt slot.
    pub start_block: usize,
    pub end_block: usize,
    /// True when the K/V came from the island cache (no capture needed).
    pub reused: bool,
    /// False when the island's forward bridge window is not fully pinned by
    /// its key (see the cacheability gate in `apply_segments_build`) — such an
    /// island is never captured and recomputes every wave.
    pub cacheable: bool,
}

/// Apply `new_segments` onto the slot (single-slot path).
///
/// Post-condition: the slot's per-layer K/V is `[sealed prefix | glue]` plus the
/// re-attached writer tail and the prefilled in-flight user message. Equivalent
/// to `apply_segments_build` + `fire_gap_fill_batch` (one slot) + `_finish`.
///
/// Answers whether the slot now holds exactly `new_segments`: the walk placed
/// every sealed piece and the gap-fill ran here. A fill deferred to the drain's
/// batch has not run yet — and a later assembly of the same slot in the drain
/// drops it — so a deferred apply answers `false`.
pub(in super::super) fn apply_segments(
    state: &mut SlotState,
    mut ctx: ApplyContext<'_>,
    new_segments: &[ProjectionSegment],
    defer: Option<&mut Vec<GapFillPlan>>,
) -> Result<bool, ConversationError> {
    let plan = apply_segments_build(state, &mut ctx, new_segments)?;
    let complete = plan.complete;
    // When the caller is batching drain gap-fills AND this projection has no
    // deferred user message (ingest / compression — the content prefills through a
    // separate unit), queue the fire and finish now (restore-tail only). The glue
    // K/V isn't read until the caller fires the whole drain's batch as ONE forward
    // at drain end — collapsing N launch-floor gap-fills into one. A projection
    // WITH a deferred user must fire inline (the user prefill attends the glue).
    // No island capture on this path: the gaps are still zeros at finish time.
    if let Some(sink) = defer {
        if plan.deferred_user.is_none() {
            sink.push(fire_only(&plan));
            apply_segments_finish(state, &mut ctx, plan, false)?;
            return Ok(false);
        }
    }
    let t_glue = Instant::now();
    if let Err(e) = fire_gap_fill_batch(ctx.session, &**ctx.model, &[&plan]) {
        // The walk recorded the glue pieces as placed; unfilled, they are
        // zero chunks no later rebuild may keep.
        state.placed_pieces.clear();
        return Err(e);
    }
    super::super::drain_add_us(
        &super::super::DRAIN_GLUE_US,
        t_glue.elapsed().as_micros() as u64,
    );
    apply_segments_finish(state, &mut ctx, plan, true)?;
    Ok(complete)
}

/// A fire-only clone of a [`GapFillPlan`] (glue scatter inputs; no deferred user /
/// tail / island records), for deferring the gap-fill forward into a batched
/// drain-end fire. Islands are dropped: the deferred path finishes BEFORE the
/// fire, so there is nothing valid to capture.
pub(super) fn fire_only(plan: &GapFillPlan) -> GapFillPlan {
    GapFillPlan {
        parent_id: plan.parent_id,
        glue_tokens: plan.glue_tokens.clone(),
        glue_write_slice: plan.glue_write_slice.clone(),
        glue_write_in_blk: plan.glue_write_in_blk.clone(),
        fwd_ahead: plan.fwd_ahead.clone(),
        deferred_user: None,
        tail_per_layer: Vec::new(),
        // Fire-only: it restores no tail, so it has no rebuild to check.
        held_before: None,
        placed_before: 0,
        n_glue_tokens: plan.n_glue_tokens,
        islands: Vec::new(),
        complete: plan.complete,
    }
}

/// The slot's current block count — the walk brackets each island's chunk range
/// with this so the capture can slice exactly the island's blocks.
pub(super) fn block_count_of(
    ctx: &ApplyContext<'_>,
    parent_id: SequenceId,
) -> Result<usize, ConversationError> {
    ctx.session
        .sequence_block_count(parent_id.0)
        .ok_or_else(|| {
            ConversationError::Channel(format!("apply_segments: slot {} not in session", parent_id))
        })
}

/// How many blocks of `parent_id` hold content — the count a record of where
/// the slot stood after a piece keeps
/// (`BatchedInferenceSession::sequence_content_block_count`): a glue gap
/// leaves an empty writer chunk behind it, which the next piece fills.
pub(super) fn content_block_count_of(
    ctx: &ApplyContext<'_>,
    parent_id: SequenceId,
) -> Result<usize, ConversationError> {
    ctx.session
        .sequence_content_block_count(parent_id.0)
        .map_err(ConversationError::Model)?
        .ok_or_else(|| {
            ConversationError::Channel(format!("apply_segments: slot {} not in session", parent_id))
        })
}

/// Cut the slot after its first `keep` held pieces — K/V, per-position state
/// and token log together. Keeping nothing truncates to nothing, and the
/// per-position state goes with the K/V: pages left from the previous assembly
/// would sit at positions this one does not have.
pub(super) fn cut_slot(
    ctx: &mut ApplyContext<'_>,
    held: &[PlacedPiece],
    keep: usize,
) -> Result<(), ConversationError> {
    let parent_id = ctx.parent_id;
    match keep.checked_sub(1).map(|i| &held[i]) {
        None => {
            ctx.session
                .truncate_sequence_to_blocks(parent_id.0, 0)
                .map_err(ConversationError::Model)?;
            ctx.model
                .reset_positional_state(parent_id.0)
                .map_err(ConversationError::Model)?;
            if let Some(entry) = ctx.slot_tokens.get_mut(&parent_id) {
                entry.clear();
            }
        }
        Some(last) => {
            ctx.session
                .truncate_sequence_to_blocks(parent_id.0, last.blocks_after)
                .map_err(ConversationError::Model)?;
            // The per-position state is cut at the position the kept pieces
            // end at, holes included; the token log at their tokens.
            ctx.model
                .truncate_positional_state(parent_id.0, last.at_after as usize)
                .map_err(ConversationError::Model)?;
            if let Some(entry) = ctx.slot_tokens.get_mut(&parent_id) {
                entry.truncate(last.pos_after as usize);
            }
        }
    }
    Ok(())
}

/// Where held piece `i` starts, as the piece before it ends: its block, its
/// token count and its position.
pub(super) fn held_edge(held: &[PlacedPiece], i: usize) -> HeldEdge {
    i.checked_sub(1).map_or(HeldEdge::default(), |j| HeldEdge {
        block: held[j].blocks_after,
        token: held[j].pos_after as usize,
        at: held[j].at_after as usize,
    })
}

/// A boundary between two held pieces, in each of the measures a splice cuts
/// by: the K/V by block, the token log by token, the per-position state by
/// position.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct HeldEdge {
    pub(super) block: usize,
    pub(super) token: usize,
    pub(super) at: usize,
}

/// Apply `plan`'s edits and return the records of the pieces before
/// `plan.walk_from`, in the projection's order, laid end to end where they now
/// stand.
///
/// Back to front, so each edit's blocks and positions are the slot's as it
/// stood. An edit evicts its held pieces' K/V, walks its wanted pieces in at
/// their place ([`InsertCursor`]), and then replaces the per-position state and
/// token log the held pieces covered with what the wanted ones brought, in one
/// splice each. Every piece from the first edit on has moved, so its substrate
/// range is written again once all of them stand where they will.
pub(super) fn splice_edits(
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    plan: &SplicePlan,
    held: &[PlacedPiece],
    pieces: &[AssembledPiece],
    aligned: &[(u64, PieceKind)],
) -> Result<Vec<PlacedPiece>, ConversationError> {
    let parent_id = ctx.parent_id;
    let mut put_in: HashMap<usize, PlacedPiece> = HashMap::new();
    for edit in plan.edits.iter().rev() {
        let from = held_edge(held, edit.held.start);
        let to = held_edge(held, edit.held.end);
        let (from_block, to_block) = (from.block, to.block);
        if to_block > from_block {
            ctx.session
                .splice_sealed(
                    parent_id.0,
                    from_block,
                    to_block,
                    &no_kv(ctx.session.num_layers()),
                )
                .map_err(ConversationError::Model)?;
        }
        walker.insert = Some(InsertCursor {
            block: from_block,
            positional: Vec::new(),
            tokens: Vec::new(),
        });
        for w in edit.wanted.clone() {
            let block_before = walker.insert.as_ref().map_or(from_block, |c| c.block);
            let pos_before = walker.logical_pos;
            let contrib = walk_sealed_piece(ctx, walker, &pieces[w])?;
            let blocks = walker.insert.as_ref().map_or(block_before, |c| c.block) - block_before;
            put_in.insert(
                w,
                PlacedPiece {
                    identity: aligned[w].0,
                    kind: aligned[w].1,
                    blocks,
                    tokens: walker.logical_pos - pos_before,
                    // An edit never carries a hole (`place_hole`).
                    holes: 0,
                    blocks_after: 0,
                    pos_after: 0,
                    at_after: 0,
                    contrib,
                    prefix_h: DefaultHasher::new(),
                    range: placed_range(&pieces[w], blocks),
                    page: walker.placed_page.take(),
                },
            );
        }
        let cursor = walker.insert.take().ok_or_else(|| {
            ConversationError::Channel("apply_segments: the edit lost its cursor".into())
        })?;
        ctx.model
            .splice_positional_state(parent_id.0, from.at, to.at, &cursor.positional)
            .map_err(ConversationError::Model)?;
        if let Some(entry) = ctx.slot_tokens.get_mut(&parent_id) {
            if to.token <= entry.len() {
                entry.splice(from.token..to.token, cursor.tokens);
            }
        }
    }
    let mut records = plan
        .keeps
        .iter()
        .enumerate()
        .map(|(w, kept)| match kept {
            Some(h) => Ok(held[*h].clone()),
            None => put_in.remove(&w).ok_or_else(|| {
                ConversationError::Channel(format!(
                    "apply_segments: piece {w} is neither kept nor put in by the splice plan"
                ))
            }),
        })
        .collect::<Result<Vec<PlacedPiece>, ConversationError>>()?;
    restack(&mut records);
    if let Some(first) = plan.edits.first() {
        for r in &records[first.wanted.start..] {
            record_range(walker, r);
        }
    }
    Ok(records)
}

/// An empty piece on each of `layers` layers — what a splice that only evicts
/// puts in.
pub(super) fn no_kv(layers: usize) -> Vec<SealedSequence> {
    (0..layers)
        .map(|_| SealedSequence {
            chunks: Vec::new(),
            token_count: 0,
            chunk_size: CHUNK_SIZE,
            location: ArenaLocation::Gpu,
        })
        .collect()
}
