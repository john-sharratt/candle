//! Walks sealed pieces onto a slot, and reserves glue islands in place.
//!
//! Each sealed piece returns its contribution to a running prefix hash. A skipped
//! piece hashes as a zero-length placement, so the island cache keys on what was
//! actually placed. A glue island's gap is reserved at its logical position, and
//! a boundary that leads into a turn opens a small forward window, so the glue is
//! generated as a lead-in to that turn. Glue is refused for a model that cannot
//! gap-fill, since a hole in a recurrent state cannot be undone.

use super::*;

/// Write a piece's content identity into `h` — the same fields the prefix hash
/// records for that piece kind, minus the placed-length (unknown before
/// injection). Used for a bridging island's forward target: the island's K/V
/// attends the first `TURN_BRIDGE_FWD_AHEAD` tokens of this piece, so its
/// identity is part of the island's key.
pub(super) fn hash_piece_identity(
    h: &mut DefaultHasher,
    piece: Option<&AssembledPiece>,
    ctx: &ApplyContext<'_>,
) {
    match piece {
        Some(AssembledPiece::Turn {
            timeline, index, ..
        }) => {
            h.write_u8(3);
            h.write_u64(timeline.map(|t| t.raw()).unwrap_or(0));
            h.write_u64(index.0 as u64);
        }
        Some(AssembledPiece::Section(id)) => {
            h.write_u8(2);
            h.write_u32(id.raw());
            let toks = ctx.conversation.read().section_tokens_of(*id);
            h.write_u64(Arc::as_ptr(&toks) as u64);
            h.write_u64(toks.len() as u64);
        }
        Some(AssembledPiece::Glue(tokens)) => {
            h.write_u8(1);
            for &t in tokens.iter() {
                h.write_u32(t);
            }
        }
        Some(AssembledPiece::DeferredUser(tokens)) => {
            h.write_u8(5);
            for &t in tokens.iter() {
                h.write_u32(t);
            }
        }
        None => h.write_u8(0),
    }
}

/// Walk one sealed piece onto the slot where the walk is placing, and return
/// its contribution to the running prefix hash: its kind, what identifies it,
/// and the tokens it actually placed — a skipped piece hashes as a zero-length
/// placement, distinct from an injected one.
pub(super) fn walk_sealed_piece(
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    piece: &AssembledPiece,
) -> Result<u64, ConversationError> {
    let pos_before = walker.logical_pos;
    let t = Instant::now();
    let mut c = DefaultHasher::new();
    match piece {
        AssembledPiece::Section(id) => {
            let _g = profile::span("apply:inject_section");
            inject_sealed_section(ctx, walker, *id)?;
            c.write_u8(2);
            c.write_u32(id.raw());
            // Content stamp: the section's stored token Arc — STABLE across
            // calls (an accessor-minted Arc like `section_sealed_of`'s would
            // differ every reproject and defeat the cache); a re-seal replaces
            // the stored Arc, moving the hash.
            let toks = ctx.conversation.read().section_tokens_of(*id);
            c.write_u64(Arc::as_ptr(&toks) as u64);
            c.write_u64(toks.len() as u64);
        }
        AssembledPiece::Turn {
            group,
            index,
            role,
            timeline,
        } => {
            let _g = profile::span("apply:inject_turn");
            inject_sealed_turn(ctx, walker, *timeline, *group, *index, *role)?;
            c.write_u8(3);
            c.write_u64(timeline.map(|t| t.raw()).unwrap_or(0));
            c.write_u32(index.0);
            c.write_u8(*role as u8);
        }
        AssembledPiece::Glue(_) | AssembledPiece::DeferredUser(_) => {
            return Err(ConversationError::Channel(format!(
                "apply_segments: {} is forwarded, not sealed — only a sealed piece is \
                 placed this way",
                piece_label(piece)
            )));
        }
    }
    super::super::drain_add_us(
        &super::super::DRAIN_ELEVATE_US,
        t.elapsed().as_micros() as u64,
    );
    c.write_u32(walker.logical_pos - pos_before);
    Ok(c.finish())
}

/// The substrate range a placed piece writes where it sits: a turn's or a
/// section's, once it placed any K/V. Glue and turn halves record none.
pub(super) fn placed_range(piece: &AssembledPiece, blocks: usize) -> Option<PieceRange> {
    if blocks == 0 {
        return None;
    }
    match piece {
        AssembledPiece::Turn {
            timeline: Some(tl),
            index,
            ..
        } => Some(PieceRange::Turn(*tl, *index)),
        AssembledPiece::Section(id) => Some(PieceRange::Section(*id)),
        _ => None,
    }
}

/// Write where `r` now sits back to the substrate, with the walk's other ranges.
pub(super) fn record_range(walker: &mut SegmentWalker, r: &PlacedPiece) {
    let (start, end) = ((r.blocks_after - r.blocks) as u64, r.blocks_after as u64);
    match r.range {
        Some(PieceRange::Turn(tl, idx)) => walker.ranges.turn((tl, idx, start, end)),
        Some(PieceRange::Section(sid)) => walker.ranges.section((sid, start, end)),
        None => {}
    }
}

/// State carried across a single `apply_segments` walk. Accumulates the gap-fill
/// new region (every glue island) plus each column's TRUE logical position, so
/// the walk emits one batched gap-fill forward instead of N per-island prefills.
pub(super) struct SegmentWalker<'c> {
    /// All glue tokens across every island, in logical order — the gap-fill Q.
    pub(super) glue_tokens: Vec<u32>,
    /// Per glue token: the reserved gap chunk's block index (its scatter target).
    pub(super) glue_write_slice: Vec<u32>,
    /// Per glue token: the in-block offset within its gap chunk.
    pub(super) glue_write_in_blk: Vec<u32>,
    /// Per glue token: forward bridge window (tokens past its own position).
    pub(super) fwd_ahead: Vec<u32>,
    /// Running TRUE sequence position as the walk advances through sealed+glue.
    /// Equal to `Σ chunk usage` by construction (gaps are reserved in place), so
    /// it cross-checks the slot offset in the convention assert.
    pub(super) logical_pos: u32,
    /// True if the most recent slot mutation was a Sealed inject (Arc-shared,
    /// partial trailing chunk possible) — so the gap-fill pushes a fresh writer
    /// chunk before writing.
    pub(super) last_was_sealed: bool,
    /// The deferred in-flight user message, prefilled after the gap-fill.
    pub(super) deferred_user: Option<Arc<Vec<u32>>>,
    /// Accounting: total glue tokens collected (the reproject cost scales here).
    pub(super) n_glue_tokens: usize,
    /// Accounting: glue tokens Arc-injected from the island cache (not computed).
    pub(super) reused_glue_tokens: usize,
    /// Accounting: sealed turn / section segments actually injected into the
    /// slot, the tokens they carried, and any that were resolved-but-dropped.
    /// A non-zero `skipped_*` means the model is decoding against a context
    /// that is missing selected history — surfaced in the assembly summary so
    /// a dropped turn never goes unnoticed.
    pub(super) sealed_turns: usize,
    pub(super) sealed_sections: usize,
    pub(super) sealed_tokens: usize,
    pub(super) skipped_turns: usize,
    pub(super) skipped_sections: usize,
    /// Index pages handed to the model during this walk. Counted so the
    /// assembled prefix can be checked against it: a walk that injects sealed
    /// tokens and pushes no pages has built a slot whose K/V nothing indexes.
    pub(super) pages_pushed: usize,
    /// Leading pieces kept from the slot's last assembly rather than walked.
    pub(super) kept_pieces: usize,
    /// Where each turn and section this walk placed now sits in the slot,
    /// written back to the substrate under one lock once the walk ends — or
    /// as it leaves early on an error ([`PendingRanges`]).
    pub(super) ranges: PendingRanges<'c, Conversation>,
    /// Microseconds the walk's turns spent resolving their sealed K/V and
    /// index rows, injecting the K/V, and pushing the index pages — the parts of
    /// a turn's cost the `apply split` line breaks out.
    pub(super) turn_resolve_us: u64,
    /// `turn_resolve_us` by part: reading each turn's index page (from the log
    /// for `turn_fetched` of them), waiting for the substrate's read lock, and
    /// windowing the reasoning out of the K/V and the rows.
    pub(super) turn_fetch_us: u64,
    pub(super) turn_fetched: u32,
    pub(super) turn_lock_us: u64,
    pub(super) turn_window_us: u64,
    pub(super) turn_inject_us: u64,
    pub(super) turn_pages_us: u64,
    /// The index page the piece being walked was placed with, taken into its
    /// [`PlacedPiece`] when the piece is recorded.
    pub(super) placed_page: Option<Arc<StoredPages>>,
    /// Where a splice is putting pieces in, while one is (`apply_edit`); the
    /// slot's tail otherwise.
    pub(super) insert: Option<InsertCursor>,
    /// The slot's own turns that go in with their reasoning
    /// ([`Substrate::reasoning_turns`]); every other turn is windowed.
    pub(super) reasoning: Vec<TurnIndex>,
}

/// An edit's insertion point and what its pieces bring that is placed after
/// them all: the per-position state and the token log, spliced in one go once
/// the edit's last piece is in.
pub(super) struct InsertCursor {
    /// The block the next piece's K/V goes in at.
    pub(super) block: usize,
    pub(super) positional: Vec<PositionalPiece>,
    pub(super) tokens: Vec<u32>,
}

impl<'c> SegmentWalker<'c> {
    pub(super) fn new(conversation: &'c Conversation, reasoning: Vec<TurnIndex>) -> Self {
        Self {
            glue_tokens: Vec::new(),
            glue_write_slice: Vec::new(),
            glue_write_in_blk: Vec::new(),
            fwd_ahead: Vec::new(),
            logical_pos: 0,
            last_was_sealed: false,
            deferred_user: None,
            n_glue_tokens: 0,
            reused_glue_tokens: 0,
            sealed_turns: 0,
            sealed_sections: 0,
            sealed_tokens: 0,
            skipped_turns: 0,
            skipped_sections: 0,
            pages_pushed: 0,
            kept_pieces: 0,
            ranges: PendingRanges::new(conversation),
            turn_resolve_us: 0,
            turn_fetch_us: 0,
            turn_fetched: 0,
            turn_lock_us: 0,
            turn_window_us: 0,
            turn_inject_us: 0,
            turn_pages_us: 0,
            placed_page: None,
            insert: None,
            reasoning,
        }
    }

    /// Record `count` sealed (prefix) columns and advance the logical position.
    /// Called by the sealed-inject helpers after a successful inject. The
    /// position itself is carried by the injected chunks' `rope_base`; the walker
    /// only tracks the running total for the convention assert.
    pub(super) fn record_sealed(&mut self, count: usize) {
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
pub(super) const TURN_BRIDGE_FWD_AHEAD: u32 = 16;

/// Forward bridge window for the glue island that immediately precedes `next` in
/// the assembled piece stream. Open it (`TURN_BRIDGE_FWD_AHEAD`) only when the
/// glue leads INTO a conversation turn/summary — a `Turn` piece, which is
/// injected in-place downstream in the same slot and so is already resident in
/// `[0, kv_len)` for the glue forward to attend. Section boundaries and the glue
/// before the deferred (still un-prefilled) user message have no resident turn
/// ahead, so they stay backward-only (`0`).
pub(super) fn glue_bridge_window(next: Option<&AssembledPiece>) -> u32 {
    match next {
        Some(AssembledPiece::Turn { .. }) => TURN_BRIDGE_FWD_AHEAD,
        _ => 0,
    }
}

pub(super) fn reserve_glue_island(
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    tokens: &[u32],
    fwd_ahead: u32,
) -> Result<(), ConversationError> {
    let n = tokens.len();
    if n == 0 {
        return Ok(());
    }
    // **A model that cannot gap-fill must never reach here.**
    //
    // Reserving an island commits the projection to computing K/V for tokens
    // in the MIDDLE of the sequence, which a recurrence cannot do: its state has
    // already accumulated past the hole and there is no way to re-enter it
    // there. The wave would bail on `n_glue > 0` a layer down, after the
    // projection had been planned and the chunks reserved — so the refusal
    // belongs here, where it can name what asked for an island and why.
    if !ctx.model.model_core_properties().can_gap_fill {
        return Err(ConversationError::Channel(format!(
            "projection reserved a {n}-token glue island, but this model cannot \
             gap-fill: its per-sequence memory is a recurrence, so a token \
             inserted mid-sequence has no computable output. Every producer of a \
             live `Generated` run has to become sealed content for this lineage \
             (docs/deltanet_state_persistence.md §4.6-§4.7d)."
        )));
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
