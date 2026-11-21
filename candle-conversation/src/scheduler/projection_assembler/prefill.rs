//! Prefills a slot's own tokens, and re-injects a span the slot forwarded once
//! from its cached K/V and index rows.
//!
//! A forwarded span keeps its index page beside its K/V, because the index keys
//! come from hidden states that cannot be recovered from the bytes. A cached span
//! without its page would hold tokens no cache accounts for. The in-flight user
//! message is captured once and re-injected across reprojections instead of being
//! computed again. Prefill runs in passes capped by the token limit, so activation
//! buffers stay bounded.

use super::*;

/// A span this slot forwarded once and can re-inject instead of recomputing.
///
/// **The page travels with the K/V, always.** Re-injecting the bytes is what
/// makes a reproject cheap, but the index cannot be recovered from them — its
/// keys come from hidden states, so the only copy is the one taken when the
/// forward ran. A cached span whose page is dropped comes back as tokens no
/// cache accounts for, which is silent under the QSA identity threshold and
/// refuses the select above it.
#[derive(Clone)]
pub(in super::super) struct CapturedSpan {
    pub kv: Arc<Vec<SealedSequence>>,
    /// `None` only for a model that indexes nothing. One page, keyed whole.
    pub page: Option<Arc<StoredPages>>,
}

/// Hand a re-injected span's index rows back to the model.
///
/// `injected` is what the slot actually received. With no page, or with a cut
/// (whose page describes tokens the slot does not hold), the slot is advanced
/// over the tokens placed instead, as the section and turn paths do: a page
/// pushed after this one is placed at a position measured from here.
pub(super) fn push_captured_page(
    ctx: &mut ApplyContext<'_>,
    span: &CapturedSpan,
    injected: &SealedInjection,
    what: &InjectedPiece,
) {
    match span.page.as_ref().filter(|_| !injected.was_cut()) {
        Some(page) => {
            if let Err(e) =
                ctx.model
                    .push_positional_state(ctx.parent_id.0, page.bytes(), &page.whole_key())
            {
                tracing::warn!("apply_projection: {what} index page refused: {e}");
            }
        }
        None => {
            if !ctx.model.carries_positional_state() {
                return;
            }
            let advanced = ctx
                .model
                .push_positional_gap(ctx.parent_id.0, injected.tokens);
            if span.page.is_none() || advanced.is_err() {
                tracing::warn!(
                    advanced = advanced.is_ok(),
                    "apply_projection: {what} has no index page — its {} token(s) are \
                     unindexed",
                    injected.tokens
                );
            }
        }
    }
}

// ── NewUserMessage handling (live-prefill into pending_user_part) ────────────

pub(super) fn handle_new_user_message(
    state: &mut SlotState,
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    tokens: &Arc<Vec<u32>>,
) -> Result<(), ConversationError> {
    if let Some(cached) = state.pending_user_part.clone() {
        // Mid-decode reproject path: the user's K/V was already
        // captured on an earlier apply.  Re-inject the cached bytes;
        // do not re-run the forward pass.
        let what = InjectedPiece::CachedUserMessage {
            tokens: tokens.len(),
        };
        let injected = inject_arc_sealed(ctx.session, ctx.parent_id, &cached.kv, &what)?;
        if let Some(inj) = injected.filter(SealedInjection::was_cut) {
            report_cut(
                &what,
                ctx.parent_id,
                &cached.kv,
                &inj,
                "captured by this process's own forward, so this is a live fault",
            );
        }
        // And the rows that go with them — `apply_segments_build` reset the
        // index before this walk, so nothing else puts them back.
        if let Some(inj) = &injected {
            push_captured_page(ctx, &cached, inj, &what);
        }
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
pub(super) fn forward_tokens(
    ctx: &mut ApplyContext<'_>,
    tokens: &[u32],
) -> Result<(), ConversationError> {
    let parent_id = ctx.parent_id;
    // Drain-path prefill accounting: this forward runs during a submission drain
    // (the newly-ingested turn's content) AND during reproject (re-prefilling the
    // in-flight user message); `drain_add_us` only counts the former. The tokens go
    // to `DRAIN_PREFILL_TOKENS` so the Prefill phase can reclaim them from the
    // otherwise-token-less projection band.
    let t_fwd = Instant::now();
    let mut offset = 0;
    while offset < tokens.len() {
        let chunk_len = (tokens.len() - offset).min(ctx.max_prefill_pass_tokens);
        let slice = &tokens[offset..offset + chunk_len];
        // On the host: the forward lays the ids on its own span.
        let input = Tensor::new(slice, &Device::Cpu)
            .and_then(|t| t.unsqueeze(0))
            .map_err(ConversationError::Model)?;
        {
            let _g = profile::span("loop:prefill:forward");
            let nl = ctx.model.num_layers().max(1);
            // Only the K/V this writes is wanted; the result drops here, with
            // the logits still on the span they were carved from.
            ctx.model
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
                .map_err(ConversationError::Model)?;
        }
        ctx.session
            .advance_sequence(parent_id.0, chunk_len)
            .map_err(ConversationError::Model)?;
        offset += chunk_len;
    }
    super::super::drain_add_us(
        &super::super::DRAIN_PREFILL_US,
        t_fwd.elapsed().as_micros() as u64,
    );
    super::super::drain_add_us(&super::super::DRAIN_PREFILL_TOKENS, tokens.len() as u64);
    // Monotone OOM-free-throughput evidence for the admission regulator —
    // every path above returns on error, so reaching here means the whole
    // token run forwarded successfully at the current admission width.
    super::super::PREFILL_OK_TOKENS
        .fetch_add(tokens.len() as u64, std::sync::atomic::Ordering::Relaxed);
    super::super::TURN_CONTENT_TOKENS
        .fetch_add(tokens.len() as u64, std::sync::atomic::Ordering::Relaxed);
    Ok(())
}

/// If the slot's tail is Arc-shared from a prior Sealed inject, push a fresh
/// writer chunk so this prefill's writes don't alias that partial chunk.
pub(super) fn push_empty_if_sealed(
    ctx: &mut ApplyContext<'_>,
    last_was_sealed: bool,
) -> Result<(), ConversationError> {
    if last_was_sealed {
        let _g = profile::span("loop:prefill:push_empty");
        ctx.session
            .push_empty_writer_chunk(ctx.parent_id.0)
            .map_err(ConversationError::Model)?;
    }
    Ok(())
}

/// Captures the just-written per-layer
/// `SealedSequence` (for `pending_user_part`, which re-injects the in-flight
/// turn's user K/V across mid-decode reprojections instead of re-prefilling it).
pub(super) fn drive_prefill_and_capture(
    ctx: &mut ApplyContext<'_>,
    tokens: &[u32],
    last_was_sealed: bool,
) -> Result<CapturedSpan, ConversationError> {
    let parent_id = ctx.parent_id;
    // A prior windowed creep prefill on this slot can have left layers in a
    // later window one empty writer chunk behind layers in an earlier one
    // (`BatchedInferenceSession::reconcile_block_counts`). This forward and
    // the snapshot below both trust every layer's block count as one number —
    // `start_block`/`end_block` and the slot headers `forward_tokens` builds —
    // so an unreconciled skew here is captured into `CapturedSpan` and
    // re-injected into every future conversation that borrows this span,
    // permanently: the exact "chunked decode layout diverged... on the first
    // message of a new conversation" failure this reconciles away while the
    // skew is still genuinely empty and padding it is lossless.
    ctx.session
        .reconcile_block_counts(parent_id.0)
        .map_err(ConversationError::Model)?;
    push_empty_if_sealed(ctx, last_was_sealed)?;
    // Where this span's tokens start, so the index rows it is about to build can
    // be sealed back out of the cache as the span's own page. Taken before the
    // forward: afterwards the tail has grown and the span's start is gone.
    let start_pos = ctx.session.sequence_offset(parent_id.0).unwrap_or(0);
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

    let captured = {
        let _g = profile::span("loop:prefill:snapshot");
        let full = ctx
            .session
            .snapshot_sequence_per_layer(parent_id.0)
            .map_err(ConversationError::Model)?;
        // **The end of the range comes from the SNAPSHOT, never from the live
        // slot.** `snapshot_sequence_per_layer` drops each layer's trailing empty
        // chunk (`SealedSequence::drop_empty_tail`) so a skew can never be captured
        // into a `CapturedSpan` and re-injected; `sequence_block_count` counts that
        // chunk, because the slot really does hold it. Asking the slot and slicing
        // the snapshot therefore disagreed by exactly one whenever the tail chunk was
        // empty — which `reconcile_block_counts` above can make true on *every* layer
        // at once, so all of them dropped it and the range ran one past all of them.
        // That is a hard refusal from `slice_per_layer_sealed`, and it made a boot
        // from a fresh substrate impossible: measured `seal range 40..49 is outside
        // layer 0's 48 sealed chunk(s)`, identical on every attempt, while the old
        // substrate masked it by having nothing left to ingest.
        //
        // Deriving it here is what the two sibling seal paths already do
        // (`start_block + sealed[0].chunks.len()`), and it is lossless by the same
        // argument that justifies the drop: an empty chunk holds no token and no
        // position moves. The minimum across layers rather than layer 0's, so a
        // layer that dropped one while others did not still yields a range every
        // layer can satisfy — and a `min` that lands *below* `start_block` is left to
        // fail, because that is a real divergence and not a tail to trim.
        let end_block = full
            .iter()
            .map(|s| s.chunks.len())
            .min()
            .unwrap_or(start_block);
        slice_per_layer_sealed(&full, start_block, end_block)?
    };
    // The index rows this forward just built, taken as a page so a later
    // re-inject of the same K/V can hand them over with it. Without this the
    // cached bytes come back and the rows do not: a reproject resets the index,
    // re-injects the K/V from cache, and leaves the slot holding this span's
    // tokens with nothing indexing them.
    let page = match ctx.model.seal_positional_range(parent_id.0, start_pos) {
        Ok(page) => page.map(|p| Arc::new(StoredPages::new(p))),
        Err(e) => {
            tracing::warn!(
                "apply_projection: captured span at {start_pos} has no index page ({e}); \
                 a reproject re-injecting it will select against a prefix it never indexed"
            );
            None
        }
    };
    log_injected_tokens(ctx, tokens);
    Ok(CapturedSpan {
        kv: Arc::new(captured),
        page,
    })
}

// ── Diagnostic logging ───────────────────────────────────────────────────────

pub(super) fn log_injected_tokens(ctx: &mut ApplyContext<'_>, tokens: &[u32]) {
    super::super::Scheduler::record_slot_tokens(ctx.slot_tokens, ctx.parent_id, tokens);
}

/// [`log_injected_tokens`] where the walk is placing: inside an edit the
/// tokens go in with the edit's splice of the log, at its position.
#[cfg_attr(not(feature = "context-dump"), allow(dead_code))]
pub(super) fn log_placed_tokens(
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    tokens: &[u32],
) {
    match walker.insert.as_mut() {
        None => log_injected_tokens(ctx, tokens),
        Some(cursor) => cursor.tokens.extend_from_slice(tokens),
    }
}
