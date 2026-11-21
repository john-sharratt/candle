//! Injects sealed sections and turns onto a slot, with their index pages, and
//! records how far each one advanced the slot.
//!
//! A section or turn's K/V is borrowed by `Arc`, so injection is cheap. Its index
//! page is pushed after it, or the slot is advanced over its width, so every later
//! page sits where its K/V does. A page the model refuses is treated as missing:
//! the K/V still goes in, the span stays unindexed, and the section's generation
//! ends so it is prefilled fresh. A turn keeps its reasoning only for the slot's
//! own request, since an inherited thought read as the slot's own made a
//! conversation skip its own work.

use super::*;

// ── Sealed-segment injection ─────────────────────────────────────────────────

pub(super) fn inject_sealed_section(
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
    // The page is keyed by the section's content address: `sid` is this
    // substrate's schema's own id, and names another section in a tenant's.
    let stream = ctx.conversation.read().section_stream(sid);
    let page = stream.and_then(|s| ctx.section_positional.get(s)).cloned();
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
    let what = InjectedPiece::Section(sid);
    let Some(injected) = place_sealed(ctx, walker, &sealed, &what)? else {
        walker.skipped_sections += 1;
        return Ok(());
    };
    // **A skewed section is rebuilt, not re-cut on every projection.** Its
    // stored layers disagree, so every slot that borrows it gets the common
    // prefix and the stored record never improves. Ending its generation makes
    // the next request for the section prefill it fresh; this copy keeps serving
    // until then. Once ended, later borrows of the same copy say nothing more.
    if injected.was_cut() {
        if let Some(stream) = stream {
            let retired = ctx.conversation.read().is_section_tombstoned(stream);
            if !retired {
                report_cut(
                    &what,
                    parent_id,
                    &sealed,
                    &injected,
                    "ending its generation so the next request prefills it fresh",
                );
                if let Err(e) = ctx
                    .conversation
                    .retire_section_generation(stream, "layers sealed with different token windows")
                {
                    tracing::warn!(
                        "apply_projection: section {} could not be retired: {e}",
                        sid.raw()
                    );
                }
            }
        } else {
            report_cut(
                &what,
                parent_id,
                &sealed,
                &injected,
                "it has no persisted stream to retire",
            );
        }
    }
    // **A cut section's page is not pushed.** The page describes every token the
    // section stored, and the slot now holds fewer: pushed, it claims blocks the
    // slot does not hold and places every later page past where its K/V sits —
    // a select then reaches blocks nothing wrote. The cut span goes in as a gap
    // of the tokens actually placed, unindexed and aligned.
    let page = page.filter(|_| !injected.was_cut());
    // **Only the anomaly earns a debug line.** The case worth seeing is a section
    // that reaches here with no positional blob: `section_positional` is filled by
    // `ingest_section` and by nothing else, so a RECOVERED section — one the
    // substrate reload brought back rather than re-ingested — has no entry no
    // matter how sound its K/V is, and that is what leaves a slot holding an
    // unindexed prefix.
    //
    // This was unconditional, on the reasoning that a section which never reaches
    // here at all is invisible and telling the two apart is the whole question.
    // Measured over one daemon run, that cost 1,609 lines of a 4,315-line log —
    // 37% of it — and the anomaly fired **zero** times, so the volume was paid
    // entirely to restate the ordinary case. The count is not lost: the
    // `reproject (zero-copy rebuild)` engine event already carries `sections`
    // (`GET /v1/events`), so "did sections reach here" is answerable without
    // a line each.
    if page.is_some() {
        tracing::trace!(
            target: "candle_conversation::scheduler::reproject",
            slot = parent_id.0,
            section = sid.raw(),
            "apply_projection: injecting section K/V",
        );
    } else {
        tracing::debug!(
            target: "candle_conversation::scheduler::reproject",
            slot = parent_id.0,
            section = sid.raw(),
            "apply_projection: injecting section K/V with no positional page",
        );
    }
    // The rows that go with those chunks. Borrowing the K/V is what makes this
    // path cheap; the index cannot be borrowed the same way, because its keys
    // come from hidden states this slot never computed.
    //
    // **A refused page is a missing page.** The K/V went in above whatever the
    // page does, so a page the model will not read leaves the slot holding tokens
    // no index accounts for — and a view forked from it inherits them. On
    // WhiteWorld (2026-11-17) a 108-token section whose page was written before
    // the index was Q8_0 was refused on every projection that borrowed it, placed
    // nothing in its place, and every turn its views ran failed its select 108
    // tokens short. So it goes in as a gap like a section with no page, and its
    // generation is ended: the next request prefills it fresh, and that writes a
    // page the model reads.
    let had_page = page.is_some();
    let refused = match page {
        Some(page) => match place_page(ctx, walker, page.bytes(), &page.whole_key()) {
            Ok(()) => {
                walker.pages_pushed += 1;
                false
            }
            Err(e) => {
                let retired = stream.is_some_and(|s| {
                    ctx.conversation.read().is_section_tombstoned(s)
                        || ctx
                            .conversation
                            .retire_section_generation(s, "its index page cannot be read")
                            .is_ok()
                });
                tracing::warn!(
                    slot = parent_id.0,
                    retired,
                    "apply_projection: section {} index page refused ({e}) — placed as a gap; \
                     its generation ended so the next request prefills it fresh",
                    sid.raw()
                );
                true
            }
        },
        None => false,
    };
    if (refused || !had_page) && ctx.model.carries_positional_state() {
        // The section's K/V goes in either way, so the index has to be told how
        // far the slot advanced — a page pushed after this one is placed at a
        // position, and that position is measured from here. The span itself
        // stays unindexed, which is the truth about it: nothing indexed it, and
        // a query landing there finds no candidates.
        let tokens = injected.tokens;
        if let Err(e) = place_gap(ctx, walker, tokens) {
            tracing::warn!(
                "apply_projection: section {} has no index page and the slot could not be \
                 advanced past its {tokens} token(s) ({e}) — pages injected after it will be \
                 placed short of where their K/V is",
                sid.raw()
            );
        } else if !injected.was_cut() && !refused {
            // A cut section's missing page was reported with the cut, and a refused
            // one with its refusal.
            tracing::warn!(
                "apply_projection: section {} has no index page — its {tokens} token(s) are \
                 unindexed",
                sid.raw()
            );
        }
    }

    // Written with the walk's other ranges, under one write lock (see the turn
    // path). A piece an edit puts in is written once the rebuild has placed
    // everything, since the edits before it move it.
    if walker.insert.is_none() {
        let (start_block, end_block) = injected.range;
        walker
            .ranges
            .section((sid, start_block as u64, end_block as u64));
    }

    #[cfg(feature = "context-dump")]
    {
        let toks = ctx.conversation.read().section_tokens_of(sid);
        log_placed_tokens(ctx, walker, &toks[..injected.tokens.min(toks.len())]);
    }
    walker.sealed_sections += 1;
    walker.sealed_tokens += injected.tokens;
    walker.record_sealed(injected.tokens);
    Ok(())
}

/// Whether a selected turn may be injected WHOLE, reasoning included.
///
/// Only the slot's own rounds of the request it is continuing may — `reasoning`,
/// from [`Substrate::reasoning_turns`]. Every other turn — the slot's finished
/// requests, and every turn borrowed from another conversation — goes in with
/// its `<think>` span windowed out.
///
/// `slot_timeline` is `None` for a projection with no slot identity (mocks,
/// structural callers). That continues no request, so nothing keeps its
/// reasoning — the safe direction, since windowing only ever removes reasoning
/// from the context.
pub(super) fn keeps_reasoning(
    slot_timeline: Option<TimelineId>,
    turn_timeline: TimelineId,
    reasoning: &[TurnIndex],
    index: TurnIndex,
) -> bool {
    slot_timeline == Some(turn_timeline) && reasoning.contains(&index)
}

pub(super) fn inject_sealed_turn(
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
    // **The rule: the newest rounds of the request the slot is continuing go in
    // whole, every other turn with its reasoning windowed out**
    // (`Substrate::reasoning_turns`, `docs/thinking_span_projection.md` §6.3).
    // The request is read off the turn couplings — no mode, no flag, no tool
    // state — so a tool loop decodes against the thinking that planned it, and
    // a new user message starts with none.
    //
    // **By turn index, not by position in the piece walk.** Groups emit in score
    // order with the highest last, so a tool-scope or repo_map group can emit
    // turns after the conversation group's — the last `Turn` piece is not
    // reliably the most recent turn.
    //
    // The rows come from the turn's stored record: its bytes while the
    // substrate still holds them, a slot's held copy while one places the turn
    // (`HeldIndexPages`), and the log otherwise.
    let t_resolve = Instant::now();
    let fetch = profile::span("inject:turn:fetch_page");
    // Before the substrate lock: a page left in the log is read through the
    // persistence path, which takes its own.
    let in_memory = ctx
        .conversation
        .turn_index_page_in_memory(timeline, index)
        .is_some();
    let stored = ctx.conversation.turn_index_page(timeline, index)?;
    walker.turn_fetched += u32::from(!in_memory && stored.is_some());
    walker.turn_fetch_us += t_resolve.elapsed().as_micros() as u64;
    // The page as stored, before any windowing filters it: what the placed
    // piece holds, so the next rebuild finds it again by its record.
    let held_page = stored.clone();
    let (sealed, page) = {
        let lock = profile::span("inject:turn:substrate_read_lock");
        let t_lock = Instant::now();
        let conv = ctx.conversation.read();
        walker.turn_lock_us += t_lock.elapsed().as_micros() as u64;
        let t_window = Instant::now();
        lock.end();
        fetch.end();
        let _window = profile::span("inject:turn:window");
        // **Only the SLOT's own request keeps reasoning, never another
        // timeline's.** A projection can span timelines — retrieval pulls in
        // other conversations' turns, and a forked conversation inherits its
        // ancestors' (`Substrate::inherited_chain`) — and the model reads any
        // `<think>` it is shown as its own. Measured: an inherited block reading
        // "The user wants me to read ARCHITECTURE.md — I already have lines
        // 1-200 from a previous read" made the next conversation skip its own
        // `file_read` and summarise the wrong file, and made a dialogue answer
        // that inherited instruction instead of the question it was asked.
        let slot_timeline = ctx.slot_target.map(|t| t.timeline);
        let resolved = if keeps_reasoning(slot_timeline, timeline, &walker.reasoning, index) {
            let _g = profile::span("inject:turn:sealed_whole");
            (conv.turn_sealed_of(timeline, index), stored)
        } else {
            let _g = profile::span("inject:turn:without_thinking");
            // An `Err` here is a turn whose reasoning cannot be windowed. It is
            // propagated rather than skipped: injecting it whole would show the
            // reasoning the projection was asked to remove, and skipping it would
            // silently drop a turn the selection chose.
            match conv.turn_sealed_without_thinking(timeline, index, stored)? {
                Some((kv, page)) => (Some(kv), page),
                None => (None, None),
            }
        };
        walker.turn_window_us += t_window.elapsed().as_micros() as u64;
        resolved
    };
    walker.turn_resolve_us += t_resolve.elapsed().as_micros() as u64;
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
    let what = InjectedPiece::Turn(timeline, index);
    let t_inject = Instant::now();
    let injected = place_sealed(ctx, walker, &sealed, &what)?;
    walker.turn_inject_us += t_inject.elapsed().as_micros() as u64;
    let Some(injected) = injected else {
        walker.skipped_turns += 1;
        return Ok(());
    };
    if injected.was_cut() {
        report_cut(
            &what,
            parent_id,
            &sealed,
            &injected,
            "a conversation's turn is its history and is not rebuilt; its index pages \
             are left out and the span goes in unindexed",
        );
    }
    let t_pages = Instant::now();
    push_turn_pages(ctx, walker, timeline, index, page, &injected);
    walker.turn_pages_us += t_pages.elapsed().as_micros() as u64;
    // Recorded on the walker and written once the walk ends, under one write
    // lock for every turn it placed: a write lock per turn queued each one
    // behind whatever reader held the substrate — the persistence thread mid-pass.
    // A turn an edit puts in is written once the rebuild has placed everything,
    // since the edits before it move it.
    if walker.insert.is_none() {
        let (start_block, end_block) = injected.range;
        walker
            .ranges
            .turn((timeline, index, start_block as u64, end_block as u64));
    }

    // The slot's token log is a `context-dump` diagnostic; without the feature
    // the turn's ids are not read at all — copying them was a full turn's worth
    // of ids, under a read lock, for every turn a rebuild placed.
    #[cfg(feature = "context-dump")]
    {
        let toks: Vec<u32> = ctx
            .conversation
            .read()
            .assistant_token_ids_of(timeline, index)
            .to_vec();
        log_placed_tokens(ctx, walker, &toks);
    }
    walker.sealed_turns += 1;
    walker.sealed_tokens += injected.tokens;
    walker.record_sealed(injected.tokens);
    walker.placed_page = held_page;
    Ok(())
}

/// Push the index pages of a turn — or of one of its records — just placed as
/// `injected`, and advance the slot over whatever they do not cover, so every
/// later page sits where its K/V does.
pub(super) fn push_turn_pages(
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    timeline: TimelineId,
    index: TurnIndex,
    page: Option<Arc<StoredPages>>,
    injected: &SealedInjection,
) {
    // A cut turn's pages describe tokens the slot does not hold — see the same
    // rule in `inject_sealed_section`.
    let page = page.filter(|_| !injected.was_cut());
    // The rows that go with those chunks. Borrowing the K/V is what makes a
    // reprojection cheap; the index cannot be borrowed the same way, because
    // its keys come from hidden states this slot never computed.
    // Resolved above, beside the K/V it belongs to, so a windowed turn's rows
    // are filtered in the same call that windowed its chunks.
    if let Some(blob) = page {
        // A turn's index is a LIST of pages — one per piece its rows span (see
        // `index_pages`) — pushed in the order the seal produced them, because
        // each carries its own ragged width. A malformed payload injects
        // nothing: half a turn's index is worse than none, since the slot would
        // then hold K/V that only part of the cache accounts for.
        // **However this ends, the slot must finish past the K/V just injected.**
        // `inject_arc_sealed` above put the turn's chunks in unconditionally, so
        // a page path that gives up early leaves the index short of the cache and
        // every LATER page is placed at a position its K/V does not occupy — the
        // slot then selects against a prefix that is silently misaligned, which
        // is a wrong answer rather than a missing one. The `page.is_none()`
        // branch below already advances for exactly this reason; these two
        // failure exits did not.
        let carries = ctx.model.carries_positional_state();
        let mut advanced = true;
        let _push = profile::span("inject:turn:push_pages");
        // Both callbacks place through the walk, and the walk calls them in
        // turn, never together — so one cell lends them the walk and context,
        // for this scope.
        let pushed = {
            let placing = RefCell::new((&mut *ctx, &mut *walker));
            index_pages::push_in_order(
                &blob,
                |p, key| {
                    let mut placing = placing.borrow_mut();
                    let (c, w) = &mut *placing;
                    place_page(c, w, p, key)
                },
                |gap| {
                    if carries {
                        let mut placing = placing.borrow_mut();
                        let (c, w) = &mut *placing;
                        advanced = place_gap(c, w, gap).is_ok();
                    }
                },
            )
        };
        match pushed {
            Ok(pushed) => {
                walker.pages_pushed += pushed.pages;
                if let Some(e) = &pushed.refused {
                    tracing::warn!(
                        "apply_projection: turn {}/{} index page refused: {e}",
                        timeline,
                        index.0
                    );
                }
                if pushed.gap > 0 && carries {
                    tracing::warn!(
                        timeline = timeline.raw(),
                        index = index.0,
                        gap = pushed.gap,
                        advanced,
                        "apply_projection: index pages stopped short of the turn's K/V; \
                         advancing the slot over the remainder"
                    );
                }
            }
            Err(e) => {
                tracing::warn!(
                    "apply_projection: turn {}/{} index pages unreadable ({e}); injecting none — \
                     this slot will select against a prefix it never indexed",
                    timeline,
                    index.0
                );
                if carries {
                    // Nothing was pushed, so the whole injected width is the gap.
                    let _ = place_gap(ctx, walker, injected.tokens);
                }
            }
        }
    } else if ctx.model.carries_positional_state() {
        // Same reasoning as the section path: the turn's K/V is injected either
        // way, so the slot has to be advanced past it or the next page is placed
        // short of where its K/V actually sits.
        let tokens = injected.tokens;
        let declared = place_gap(ctx, walker, tokens);
        // A cut turn's pages were left out above and reported with the cut.
        if !injected.was_cut() || declared.is_err() {
            tracing::warn!(
                stream_id = format!("{:#018x}", turn_stream_id(timeline.raw(), index.0).0),
                advanced = declared.is_ok(),
                "apply_projection: turn {}/{} has no index page, resident or stored — its \
                 {tokens} token(s) are unindexed",
                timeline,
                index.0
            );
        }
    }
}

/// Inject a piece's already-sealed per-layer K/V onto `parent_id`'s slot.
///
/// Passes `sealed` straight through — no per-layer copy. An earlier version
/// rebuilt a fresh `Vec<SealedSequence>` here solely to override `chunk_size`
/// and `location`, which meant cloning every chunk's six palette/scale/format
/// `Arc`s (`SealedChunk` derives `Clone`) for every layer of every piece, on
/// every projection — work `ChunkedKvBacking::inject_sealed_at_tail` repeats a
/// moment later when it builds its own `ChunkWindow`s from the same chunks.
/// Neither `inject_sealed_at_tail` (here or in `candle-nn`) nor
/// `truncate_sealed_to_tokens` (the one place that constructs a fresh
/// `SealedSequence` on this path) ever reads a `SealedSequence`'s own
/// `chunk_size` or `location` — both fields are copied through, never
/// branched on — so the override was dead work from the moment it ran.
///
/// `None` when the piece's layer count does not match the session's and nothing
/// was injected. A piece whose layers disagree on their chunk windows is cut to
/// the common prefix by the session and reported in the returned
/// [`SealedInjection`]; the caller knows which stored record the piece is, and
/// so whether it can be repaired and how loudly to say so ([`report_cut`]).
pub(super) fn inject_arc_sealed(
    session: &mut BatchedInferenceSession,
    parent_id: SequenceId,
    sealed: &Arc<Vec<SealedSequence>>,
    what: &InjectedPiece,
) -> Result<Option<SealedInjection>, ConversationError> {
    let _g = profile::span("inject:arc_sealed");
    let n_layers = session.num_layers();
    if sealed.len() != n_layers {
        tracing::warn!(
            "apply_projection: {what} has {} layers, expected {}; skipping",
            sealed.len(),
            n_layers,
        );
        return Ok(None);
    }
    session
        .inject_sealed_at_tail(parent_id.0, sealed)
        .map(Some)
        .map_err(ConversationError::Model)
}

/// Put a sealed piece's K/V where the walk is placing: onto the slot's tail, or
/// in at the open edit's insertion point, which then moves past it.
pub(super) fn place_sealed(
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    sealed: &Arc<Vec<SealedSequence>>,
    what: &InjectedPiece,
) -> Result<Option<SealedInjection>, ConversationError> {
    let Some(cursor) = walker.insert.as_mut() else {
        return inject_arc_sealed(ctx.session, ctx.parent_id, sealed, what);
    };
    let _g = profile::span("inject:splice_sealed");
    let n_layers = ctx.session.num_layers();
    if sealed.len() != n_layers {
        tracing::warn!(
            "apply_projection: {what} has {} layers, expected {}; skipping",
            sealed.len(),
            n_layers,
        );
        return Ok(None);
    }
    let placed = ctx
        .session
        .splice_sealed(ctx.parent_id.0, cursor.block, cursor.block, sealed)
        .map_err(ConversationError::Model)?;
    cursor.block = placed.range.1;
    Ok(Some(placed))
}

/// Hand a piece's page to the model where the walk is placing. Inside an edit
/// it is queued for the edit's one splice of per-position state, once the
/// model has said it reads it — so a page refused at the tail is refused here.
pub(super) fn place_page(
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    bytes: &[u8],
    key: &PieceKey,
) -> candle::Result<()> {
    match walker.insert.as_mut() {
        None => ctx
            .model
            .push_positional_state(ctx.parent_id.0, bytes, key)
            .map(|_| ()),
        Some(cursor) => {
            ctx.model.positional_lift_bytes(bytes, key)?;
            cursor.positional.push(PositionalPiece::Page {
                bytes: Arc::from(bytes),
                key: *key,
            });
            Ok(())
        }
    }
}

/// Advance the per-position state over `tokens` no page covers, where the walk
/// is placing.
pub(super) fn place_gap(
    ctx: &mut ApplyContext<'_>,
    walker: &mut SegmentWalker,
    tokens: usize,
) -> candle::Result<()> {
    match walker.insert.as_mut() {
        None => ctx.model.push_positional_gap(ctx.parent_id.0, tokens),
        Some(cursor) => {
            if tokens > 0 {
                cursor.positional.push(PositionalPiece::Gap(tokens));
            }
            Ok(())
        }
    }
}

/// Name a piece the session cut to its layers' common prefix: which piece,
/// what it stored, what the slot received, and what is being done about it.
pub(super) fn report_cut(
    what: &InjectedPiece,
    parent_id: SequenceId,
    sealed: &[SealedSequence],
    injected: &SealedInjection,
    repair: &str,
) {
    tracing::warn!(
        slot = parent_id.0,
        cut_chunks = injected.cut_chunks,
        injected_tokens = injected.tokens,
        stored_tokens = sealed.first().map_or(0, |s| s.token_count),
        "apply_projection: {what}'s layers describe different token windows; the slot \
         received their common prefix — {repair}"
    );
}
