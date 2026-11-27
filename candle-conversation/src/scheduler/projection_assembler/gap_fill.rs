//! Fires one batched forward that scatters glue K/V into its reserved gaps, and
//! checks the slot's offset against its backing.
//!
//! The forward only scatters into gaps reserved before it, so a stalled run can
//! be repeated and write the same bytes again. It must never advance the slot. An
//! advance would desynchronise the session offset from the backing length, and an
//! assert names that at the source. The profile is read as a snapshot, so the
//! run's other spans are left intact.

use super::*;

/// Times one gap-fill forward runs again after an expert stall
/// ([`is_expert_stall`]) before its error fails the turns it carries. Each
/// attempt waits on the reads the last one started, so a stall that does not
/// clear in a few is a drive that has stopped, not one that is slow.
pub(super) const GAP_FILL_STALL_RETRIES: usize = 4;

/// Each of `plans`' slots' gap scatter targets and forward bridge window, in
/// order — what the gap-fill forward is staged with.
pub(super) fn pending_glue(plans: &[&GapFillPlan]) -> Vec<PendingGlue> {
    plans
        .iter()
        .map(|p| PendingGlue {
            write_slice: p.glue_write_slice.clone(),
            write_in_blk: p.glue_write_in_blk.clone(),
            fwd_ahead: p.fwd_ahead.clone(),
        })
        .collect()
}

/// Fire ONE batched gap-fill forward over `plans` (the cross-conversation wave),
/// then fill each slot's gaps. Plans with no glue are skipped. The gaps were
/// reserved IN PLACE during the walk, so this forward only scatters each glue
/// token's K/V into its gap (by explicit `(slice, in_blk)` target) and computes
/// its attention; the slot length already counts the gaps, so there is no commit
/// step. Every column's sequence position comes from its chunk's `rope_base`
/// (`slice_rope`) — the same convention the decode reads — so the interleaved
/// slot decodes correctly with no `col_actual_pos` side channel.
pub(in super::super) fn fire_gap_fill_batch(
    session: &mut BatchedInferenceSession,
    model: &(dyn ManagedBatchedModel + Send),
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
    for p in &active {
        ids.push(p.parent_id.0);
        // On the host: the forward lays the ids on its own span.
        let input = Tensor::new(p.glue_tokens.as_slice(), &Device::Cpu)
            .and_then(|t| t.unsqueeze(0))
            .map_err(ConversationError::Model)?;
        inputs.push(input);
    }
    // Reconcile each slot's per-layer block count before this wave —
    // `BatchedInferenceSession::reconcile_block_counts`. This wave's own
    // invariant check below reads `sequence_backing_tokens`, which (like
    // `sequence_block_count`) reports the MIN across layers and trusts that
    // whatever sits past it on an ahead layer is empty writer-chunk
    // structure, never real content. A prior windowed creep or borrowed
    // section can leave that untrue; reconciling here while the excess is
    // still genuinely empty is what keeps that trust honest for the check
    // that follows.
    for &p in &active {
        session
            .reconcile_block_counts(p.parent_id.0)
            .map_err(ConversationError::Model)?;
    }
    // Read the per-op pipeline profile so the delta below covers only this
    // gap-fill forward (attn_core / mlp_ffn / qkv / out_proj, summed over layers).
    //
    // **A snapshot, not a snapshot-and-reset.** The reset is all-thread, so
    // bracketing this one forward with it wiped every `loop:*` and `drain:*` span
    // the run had accumulated — the engine's own profile table then covered only
    // the window since the last gap-fill, which is a partial window dressed as a
    // whole run. Subtracting two readings gives the same interval and destroys
    // nothing.
    #[cfg(feature = "profile")]
    let prof_before = candle_transformers::models::profile::pipeline_snapshot();
    {
        let _g = profile::span("loop:prefill:gap_fill");
        // Route the glue islands through the wave's GLUE group so the pending
        // per-slot scatter descriptors (staged above) drive the paged-glue kernel.
        // A glue-only wave carries no logits (it only scatters K/V) and the result
        // is discarded — the forward's whole purpose is the K/V side effect.
        let n = model.num_layers();
        let mut stalls = 0;
        loop {
            // Stage each slot's per-token gap scatter target + forward bridge
            // window, aligned with `ids` — again on a retry, since the forward
            // takes them. The forward routes the HD128 glue to the paged-glue
            // kernel, which streams the quantized slot once (dequant-once),
            // positions every column by its chunk `rope_base`, and masks each
            // glue token by `cpos > row_pos + fwd_ahead[t]`.
            session.set_pending_glue(pending_glue(&active));
            match model.forward_wave(session, &[], &[], &[], &[], &ids, &inputs, 0, n, None) {
                Ok(_) => break,
                // **A stall runs again, here.** The forward only scatters into
                // gaps reserved before it — no commit, no advance — so running
                // it again writes the same bytes to the same places, and the
                // rolled-back one left nothing to undo. Failing it instead
                // ended the turn of every conversation it carried over one
                // slow pack read.
                Err(e) if is_expert_stall(&e) && stalls < GAP_FILL_STALL_RETRIES => {
                    stalls += 1;
                    tracing::warn!(
                        target: "candle_conversation::scheduler::reproject",
                        attempt = stalls,
                        n_slots = active.len(),
                        "gap-fill forward stalled on an expert read; running it again: {e}"
                    );
                }
                Err(e) => return Err(ConversationError::Model(e)),
            }
        }
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
        // Authoritative backing length = Σ chunk usage (the same source
        // `reconcile_wave_offsets` reads). The legacy per-sequence Cache
        // counter is only maintained by Cache-level write paths, which
        // in-kernel writers never touch — it would read 0 and fail the
        // invariant spuriously.
        let backing_len = session.sequence_backing_tokens(id).unwrap_or(session_off);
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
        let after = candle_transformers::models::profile::pipeline_snapshot();
        // Only what this forward added: spans absent before, and the increase on
        // spans already there. A span whose total did not move is left out.
        let before: std::collections::HashMap<&str, (f64, u64)> = prof_before
            .entries
            .iter()
            .map(|(n, ms, c)| (n.as_str(), (*ms, *c)))
            .collect();
        let mut parts: Vec<String> = after
            .entries
            .iter()
            .filter_map(|(n, ms, c)| {
                let (ms0, c0) = before.get(n.as_str()).copied().unwrap_or((0.0, 0));
                let (d_ms, d_c) = (ms - ms0, c.saturating_sub(c0));
                (d_c > 0).then(|| format!("{n}={d_ms:.1}ms({d_c})"))
            })
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
pub(in super::super) fn apply_segments_finish(
    state: &mut SlotState,
    ctx: &mut ApplyContext<'_>,
    plan: GapFillPlan,
    capture_islands: bool,
) -> Result<(), ConversationError> {
    let GapFillPlan {
        parent_id,
        glue_tokens: _,
        glue_write_slice: _,
        glue_write_in_blk: _,
        fwd_ahead: _,
        deferred_user,
        tail_per_layer,
        held_before,
        placed_before,
        n_glue_tokens,
        islands,
        complete: _,
    } = plan;
    // The glue tokens were already logged into the slot_tokens debug view at
    // their interleaved positions during the walk (`reserve_glue_island`).

    // ── Island capture: the wave has scattered every fresh island's K/V into
    // its gap chunks, so slice them out of a per-layer snapshot (Arc borrows,
    // zero-copy — the `pending_user_part` capture pattern) and file them under
    // their content-context keys for the next reproject to Arc-inject. Prune
    // stale keys first so the cache tracks exactly the live projection's
    // islands. Skipped when the caller deferred the fire (`capture_islands =
    // false`) — the gaps are still zeros there.
    if capture_islands && !islands.is_empty() {
        let gen = state.begin_island_capture();
        for isl in &islands {
            if !isl.reused && isl.cacheable && isl.end_block > isl.start_block {
                // Ranged snapshot: records only the island's own chunks on
                // every layer (a whole-slot record per wave costs tens of ms
                // at deep slots; the range costs microseconds).
                //
                // A window-divergence failure here is NOT the same case
                // `Scheduler::repair_section_if_window_divergence_confirmed`
                // handles: that hook tombstones the PERSISTED section a seal
                // was writing. This range is glue tokens — ephemeral,
                // in-memory-only cache entries (`state.glue_islands`), never
                // written to the substrate — built on top of whatever this
                // slot's prefix already holds, which may include an earlier
                // injection of a corrupted persisted section. There is no
                // single `SectionId` to confirm-and-tombstone at this point
                // without walking back through the slot's own prefix history
                // (the "prefix chain" tracing the reactive repair's own doc
                // comment names as future scope, not attempted here) — so
                // this only logs plainly instead of silently propagating a
                // bare model error, to leave a trace pointing at the real
                // mechanism instead of nothing.
                let sealed = ctx
                    .session
                    .snapshot_sequence_blocks(parent_id.0, isl.start_block, isl.end_block)
                    .inspect_err(|e| {
                        if e.to_string().contains(WINDOW_DIVERGENCE_MARKER) {
                            tracing::warn!(
                                parent_id = parent_id.0,
                                start_block = isl.start_block,
                                end_block = isl.end_block,
                                err = %e,
                                "glue-island capture hit a per-layer window divergence — \
                                 likely rooted in an earlier-injected section this slot's \
                                 prefix already holds; not repaired from here (see comment)",
                            );
                        }
                    })
                    .map_err(ConversationError::Model)?;
                // **No page, and today that is provable rather than assumed.**
                // An island exists only for a model that gap-fills, and
                // `can_gap_fill` is `!carries_recurrent_state()` — while the
                // only stack that carries an index (`carries_positional_state`)
                // is recurrent and refuses islands outright. So a cached island
                // never has rows to carry. If that ever stops holding,
                // `push_captured_page` says so on the re-inject instead of
                // leaving the island's tokens silently unindexed.
                state.glue_islands.insert(
                    isl.key,
                    (
                        gen,
                        CapturedSpan {
                            kv: Arc::new(sealed),
                            page: None,
                        },
                    ),
                );
            }
        }
    }

    // **The turn's own tokens begin below, so its index boundary is here.**
    //
    // Everything assembled above is prefix: injected turns and sections, and any
    // of it this slot FORWARDED rather than borrowed has left its rows in the
    // live index tail. The deferred user message that prefills below is the
    // turn's own first content, and it extends that same tail — so without a cut
    // here the two become one run and the seal, which asks for "the last N
    // tokens", takes a page that starts before the turn does.
    //
    // `Scheduler::create_view` closes again at the carve, and that is not a
    // duplicate: it is the boundary for a slot that reaches the carve having
    // forwarded nothing (`skip_projection`, the `code_read` ingest), which is
    // why the cut cannot live at only one of the two. Closing an empty tail is a
    // no-op, so whichever is the real boundary is the one that closes anything.
    //
    // Leaving this out was measured, on turn after turn: `turn_token_count=324`
    // with `reasoning=(296, 3)` and sealed `widths=[7, 28]` — the 289-token user
    // message swallowed into the prefix's page, leaving a 7-token assistant
    // header as the whole of the turn's "prefill".
    if deferred_user.is_some() {
        let held = ctx.session.sequence_offset(parent_id.0).unwrap_or(0);
        super::super::close_unit_boundary(&**ctx.model, parent_id, "turn user half", held);
    }

    // The in-flight user message prefills last, after the now-filled gaps. The
    // slot always ends in a complete region (a reserved gap chunk, or a sealed
    // inject when there is no glue), so the user message must push a fresh writer
    // chunk rather than extend it — `last_was_sealed = true` unconditionally.
    if let Some(tokens) = deferred_user {
        let mut walker = SegmentWalker::new(ctx.conversation, Vec::new());
        walker.last_was_sealed = true;
        handle_new_user_message(state, ctx, &mut walker, &tokens)?;
    }

    // Everything so far is this projection's placement; the tail restored next
    // is the slot's own.
    let placed = ctx.session.sequence_offset(parent_id.0).unwrap_or(0);
    {
        let _g = profile::span("apply:restore_tail");
        restore_tail(ctx.session, parent_id, tail_per_layer)?;
    }

    // **The rebuild put back everything it took apart.**
    //
    // A slot that comes back SHORT has lost content that existed a moment ago
    // and exists nowhere else: the substrate holds only what was sealed, and
    // the writer-tail snapshot holds only what this slot forwarded. Anything
    // that was neither is simply gone, and nothing downstream faults — the turn
    // decodes fluently against a prefix missing part of itself, which is the
    // most expensive kind of wrong.
    //
    // LONG is reported too. The rebuild is supposed to reproduce the slot, and
    // a slot that grew has injected something twice — a piece restored both
    // from the substrate and from the tail.
    //
    // **A slot holding only a placement is populated, not reproduced.** What
    // the previous projection placed came from the substrate, and this one
    // places afresh by its own selection — more history hot, a tool selected in
    // or out — so its size is free to change. A slot holding nothing beyond
    // that placement has nothing of its own to lose: an empty slot, or one
    // fresh from construction that projected its system prompt alone.
    // Comparing there reports the arriving history as "duplicated" — measured
    // at 19 warns per startup with `before=0`, and at 3,486 tokens on every
    // turn of a resumed passthrough conversation whose slot was built
    // system-only — the volume of false alarm that makes a real divergence in
    // this check unreadable.
    if let Some(before) = held_before {
        let after = ctx.session.sequence_offset(parent_id.0).unwrap_or(0);
        if let Some((tokens, duplicated)) = rebuild_mismatch(before, placed_before, after) {
            tracing::warn!(
                slot = parent_id.0,
                before,
                placed_before,
                after,
                "apply_segments_finish: the rebuild did not reproduce the slot — {} \
                 token(s) {}",
                tokens,
                if duplicated { "duplicated" } else { "lost" }
            );
        }
        state.placed = placed;
    }

    // The rebuild is responsible for leaving the slot's per-position state
    // covering the K/V it assembled, exactly. Both directions are faults, and
    // for different reasons: SHORT means a source put K/V back without its rows,
    // and the select refuses once the conversation passes the identity
    // threshold; LONG means rows were installed twice, and the surplus claims
    // positions this slot does not hold — which selects against them without
    // erroring at all. This is the same invariant `handle_memory_catch_up`
    // asserts after its own rebuild, on the same pair of quantities.
    //
    // **Reported, not refused — and that is a measured decision, not caution.**
    // Promoting this to an `Err` was tried and reverted: a live run showed the
    // repo_map section-ingest path arriving here already diverged by exactly one
    // ingest turn's width (323 tokens, 408 times, always the same number), which
    // predates the thinking-span work. Refusing aborted 120 of 355 directory
    // ingests and killed every conversation with it. The invariant is right; the
    // system does not satisfy it yet on that path, so the check reports until it
    // does. Promote it when a run is silent here — that run is the evidence, and
    // this comment is the record that it has not happened.
    // Both sides in positions: the index places pages past the slot's holes as
    // the K/V does (`docs/placed_section_layout.md` §4.1), so its coverage is
    // the slot's position frontier, not its token count.
    if let Some(cov) = ctx.model.positional_coverage(parent_id.0) {
        let held = ctx.session.sequence_position(parent_id.0).unwrap_or(0);
        if cov != held {
            tracing::warn!(
                slot = parent_id.0,
                held,
                coverage = cov,
                "apply_segments_finish: the assembled slot's index is {} position(s) {} its K/V — {}",
                cov.abs_diff(held),
                if cov < held { "short of" } else { "past" },
                if cov < held {
                    "a source injected K/V without its index pages"
                } else {
                    "pages were installed twice, claiming positions this slot does not hold"
                }
            );
        }
    }

    tracing::trace!(
        target: "candle_conversation::scheduler::reproject",
        n_glue_tokens = n_glue_tokens,
        "apply_segments breakdown (single gap-fill glue prefill)",
    );
    Ok(())
}

/// Whether a rebuild failed to reproduce its slot: `Some((tokens, duplicated))`
/// when it did not. `before` is the slot's length when the rebuild took it
/// apart, `placed_before` how much of that the previous projection placed, and
/// `after` the rebuilt length. A slot holding nothing beyond a placement is
/// populated rather than reproduced, so it never mismatches.
pub(super) fn rebuild_mismatch(
    before: usize,
    placed_before: usize,
    after: usize,
) -> Option<(usize, bool)> {
    if before <= placed_before || after == before {
        return None;
    }
    Some((after.abs_diff(before), after > before))
}

// ── Writer-tail snapshot / restore ───────────────────────────────────────────

pub(super) fn snapshot_tail(
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

pub(super) fn restore_tail(
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
