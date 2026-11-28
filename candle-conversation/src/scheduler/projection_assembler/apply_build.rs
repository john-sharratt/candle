//! Builds a slot's prefix from a projection's segments, keeping what the slot
//! already holds.
//!
//! A sealed piece is kept by its own identity. A forwarded piece is kept only
//! while the chain of identities up to it still matches. Everything else is walked
//! onto the tail in logical order, and each glue island is reserved as a
//! writer-owned gap. Every chunk's rope base then equals its true position, which
//! an assert checks. A rebuild that comes back short or long is reported, because
//! a short slot decodes against a prefix missing part of itself.

use super::*;

/// Build phase: snapshot the writer tail, truncate, then walk the segments —
/// injecting the sealed prefix and collecting all glue as the new region with
/// each column's TRUE logical position. Reserves the glue writer chunk but does
/// NOT fire the forward; the caller fires (batched across slots) and then calls
/// [`apply_segments_finish`].
pub(in super::super) fn apply_segments_build(
    state: &mut SlotState,
    ctx: &mut ApplyContext<'_>,
    new_segments: &[ProjectionSegment],
) -> Result<GapFillPlan, ConversationError> {
    let parent_id = ctx.parent_id;

    // 1. Snapshot the writer tail (in-flight decode chunks).
    // What the slot held before the rebuild takes it apart. The rebuild is
    // responsible for putting all of it back — from the substrate for sealed
    // pieces, from the writer-tail snapshot for anything this slot forwarded
    // itself — and [`apply_segments_finish`] checks that it did.
    //
    // The check exists because the invariant it guards is about to widen.
    // "Unsealed content is at most the current turn" was true only by accident
    // of the seal firing before every continuation, and nothing anywhere
    // asserted it. A slot that comes back short does not fault: it decodes
    // fluently against a prefix missing its own last turn.
    let held_before = ctx.session.sequence_offset(parent_id.0).unwrap_or(0);
    let tail_per_layer = {
        let _g = profile::span("apply:snapshot_tail");
        snapshot_tail(ctx.session, parent_id)?
    };

    // 2–4. Walk segments in logical order, building the slot's chunk list IN
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
    // The turns that go in with their reasoning, read once for the whole walk:
    // the newest rounds of the request the slot's next turn continues.
    let reasoning = match ctx.slot_target {
        Some(t) => ctx.conversation.read().reasoning_turns(t.timeline),
        None => Vec::new(),
    };
    let mut walker = SegmentWalker::new(ctx.conversation, reasoning);
    // A sealed turn's `/no_think` switch needs no look-up here: it is baked into
    // the turn's own grid, so injecting the turn injects the switch it was
    // sealed with. Its ROLE MARKERS do need one: a turn sealed before the
    // boundaries moved into the grid carries ethereal ones, and the assembler
    // is still its only source (see `assemble_pieces_with`).
    // Answered for every sealed turn in ONE read-lock pass, before assembly:
    // the closure below is consulted per turn per assembly, and taking the
    // substrate lock (and cloning a layout) inside it put that cost on the
    // projection hot path.
    let unbaked: std::collections::HashSet<(TimelineId, TurnIndex)> = {
        let read = ctx.conversation.read();
        new_segments
            .iter()
            .filter_map(|seg| match seg {
                ProjectionSegment::Sealed(SealedKind::Turn(rt, _)) => {
                    rt.timeline.map(|tl| (tl, rt.index()))
                }
                _ => None,
            })
            .filter(|(tl, idx)| !read.turn_bakes_own_boundaries(*tl, *idx))
            .collect()
    };
    let bakes = |tl: TimelineId, idx: TurnIndex| !unbaked.contains(&(tl, idx));
    let pieces = assemble_pieces_with(new_segments, ctx.boundary_markers, &bakes);

    // ── Keep every piece this slot already holds ────────────────────────────
    // Each piece's alignment identity — a sealed piece's own, a forwarded one's
    // chained through the pieces before it — against what the slot's last
    // complete assembly placed (`super::piece_identity`). The slot is cut after
    // the last held piece still wanted, each change before that is spliced in
    // where it falls, and only the rest is walked (`super::piece_splice`): a
    // projection that changed one piece costs that piece, wherever it is.
    let t_identities = Instant::now();
    let aligned = {
        let _g = profile::span("apply:identities");
        let read = ctx.conversation.read();
        let slot_timeline = ctx.slot_target.map(|t| t.timeline);
        let reasoning = &walker.reasoning;
        let whole =
            |tl: TimelineId, idx: TurnIndex| keeps_reasoning(slot_timeline, tl, reasoning, idx);
        let stamp = |id: SectionId| section_content_stamp(&read.section_tokens_of(id));
        let causal_only = ctx.boundary_markers.causal_only_glue;
        let bridge = |next: Option<&AssembledPiece>| {
            if causal_only {
                0
            } else {
                glue_bridge_window(next)
            }
        };
        let reads = PieceReads {
            whole: &whole,
            section_stamp: &stamp,
            bridge: &bridge,
        };
        alignment_identities(&pieces, &chained_identities(&pieces, &reads), &reads)
    };
    let identities_us = t_identities.elapsed().as_micros() as u64;
    // A record is trusted only while the slot still reaches its end: anything
    // that cut the slot shorter since has taken the kept pieces with it. The
    // records live until the rebuild ends, and with them every page they hold:
    // a piece re-placed finds its page still in the index page cache, which
    // evicts nothing held.
    let slot_blocks = block_count_of(ctx, parent_id)?;
    let held_records: Vec<PlacedPiece> = std::mem::take(&mut state.placed_pieces);
    let trusted = held_records
        .iter()
        .take_while(|p| p.blocks_after <= slot_blocks)
        .count();
    let held: Vec<HeldPiece> = held_records[..trusted]
        .iter()
        .map(|r| HeldPiece {
            identity: r.identity,
            kind: r.kind,
            // Where the model's per-position state can be cut is asked by
            // position, holes included.
            end_pos: r.at_after as usize,
        })
        .collect();
    let wanted: Vec<WantedPiece> = aligned
        .iter()
        .map(|&(identity, kind)| WantedPiece { identity, kind })
        .collect();
    // Every splice and the cut must land where the model can cut its
    // per-position state: that state may be grouped more coarsely than the
    // pieces (one index page over a whole injected prefix).
    let plan = plan_splice(&held, &wanted, |pos| {
        ctx.model.positional_cut_floor(parent_id.0, pos)
    })
    .map_err(ConversationError::Model)?;
    // Every piece this rebuild projects, in order — the context the turn reads,
    // turn by turn (`timeline:index`). What a "the model answered about the
    // wrong file" report needs first: whether the turn it should have attended
    // to was in front of it at all.
    note("apply_segments: projected pieces", || {
        format!(
            "slot={} timeline={} pieces={}",
            parent_id.0,
            ctx.slot_target.map_or(0, |t| t.timeline.raw()),
            pieces.iter().map(piece_label).collect::<Vec<_>>().join(" "),
        )
    });
    // What the rebuild keeps, splices and walks, and what stands where the walk
    // starts — the facts that say why a rebuild cost what it did.
    note("apply_segments: splice plan", || {
        format!(
            "slot={} placed={} trusted={trusted} kept={} edits={} inserted={} evicted={} \
             cut_after={} walk_from={} pieces={} first_walked={:?}",
            parent_id.0,
            held_records.len(),
            plan.kept(),
            plan.edits.len(),
            plan.inserted(),
            plan.evicted(),
            plan.cut_after,
            plan.walk_from,
            pieces.len(),
            pieces.get(plan.walk_from).map(piece_label),
        )
    });
    let t_cut = Instant::now();
    let cut_span = profile::span("apply:cut_to_kept");
    cut_slot(ctx, &held_records, plan.cut_after)?;
    cut_span.end();
    let cut_us = t_cut.elapsed().as_micros() as u64;
    let t_splice = Instant::now();
    let splice_span = profile::span("apply:splice_edits");
    let records = splice_edits(ctx, &mut walker, &plan, &held_records, &pieces, &aligned)?;
    splice_span.end();
    let splice_us = t_splice.elapsed().as_micros() as u64;
    let mut prefix_h = records
        .last()
        .map_or_else(DefaultHasher::new, |r| r.prefix_h.clone());
    let mut content_before = records.last().map_or(0, |r| r.blocks_after);
    if let Some(last) = records.last() {
        walker.logical_pos = last.pos_after;
        walker.holes = last.at_after - last.pos_after;
        // Every piece ends on a complete region — a sealed inject or a
        // reserved gap — so the next one starts a fresh writer chunk.
        walker.last_was_sealed = true;
    }
    walker.kept_pieces = plan.kept();
    state.placed_pieces = records;
    // Per piece kind: microseconds walked and pieces walked.
    let mut walk_us = [0u64; 3];
    let mut walked = [0usize; 3];
    // ── Island-cache keying: a rolling hash over the CONTENT identity of every
    // piece the walk has placed so far. A glue island's K/V is a function of its
    // own tokens, its (backward-unbounded) attended prefix, and — when its
    // forward bridge is open — the first tokens of the piece it leads into.
    // Glue K is stored un-rotated and re-RoPE'd at read from its chunk
    // `rope_base`, so the hash deliberately excludes absolute positions: a
    // uniform shift of an identical piece stream reuses exactly. Each piece
    // contributes its kind, its identifiers, the token count it ACTUALLY placed
    // (a skipped not-hot section hashes as a zero-length placement, distinct
    // from an injected one), and — for sections, whose content can re-seal
    // under the same id — the sealed unit's `Arc` identity (a reseal mints a
    // new Arc, forcing a conservative recompute). Each piece's contribution is
    // folded in as one value (`PlacedPiece::contrib`), so after a splice the hash
    // is folded again over the pieces as they now stand (`restack`), and the
    // walk resumes from the last of them.
    let mut islands: Vec<PlannedIsland> = Vec::new();
    // The convention assert below names only the slot; this names the first
    // piece (or the kept-prefix restore) after which the two counts parted.
    let mut drift_seen = false;
    let mut check_drift =
        |ctx: &ApplyContext<'_>, walker: &SegmentWalker, at: &dyn Fn() -> String| {
            if drift_seen {
                return;
            }
            let Some(off) = ctx.session.sequence_offset(parent_id.0) else {
                return;
            };
            if off as u32 != walker.logical_pos {
                drift_seen = true;
                tracing::warn!(
                    target: "candle_conversation::scheduler::reproject",
                    slot = parent_id.0,
                    slot_offset = off,
                    logical_pos = walker.logical_pos,
                    blocks = ?ctx.session.sequence_block_count(parent_id.0),
                    kept = plan.kept(),
                    edits = plan.edits.len(),
                    at = %at(),
                    "apply_segments: slot offset and walker position parted"
                );
            }
        };
    check_drift(ctx, &walker, &|| "kept and spliced pieces".to_string());
    for i in plan.walk_from..pieces.len() {
        let pos_before = walker.logical_pos;
        let holes_before = walker.holes;
        let t_piece = Instant::now();
        // This piece's contribution to the running prefix hash.
        let mut contrib = 0u64;
        let kind = match &pieces[i] {
            AssembledPiece::Section(_) => 0,
            AssembledPiece::Turn { .. } => 1,
            AssembledPiece::Glue(_) | AssembledPiece::DeferredUser(_) | AssembledPiece::Hole(_) => {
                2
            }
        };
        match &pieces[i] {
            AssembledPiece::Glue(tokens) => {
                let _g = profile::span("apply:glue_island");
                // Causal-only dialects (DeepSeek) drop the forward bridge — glue
                // attends backward-unbounded only (§E), matching the paged-glue
                // kernel's `fwd_ahead ≡ 0` contract.
                let fwd = if ctx.boundary_markers.causal_only_glue {
                    0
                } else {
                    glue_bridge_window(pieces.get(i + 1))
                };
                // ── Bridge-window cacheability gate ──────────────────────────
                // A cached island's K/V must be a pure function of its key. A
                // backward-only island (`fwd == 0`) is prefix-keyed and always
                // eligible. A BRIDGING island also attended `fwd` tokens
                // forward, so it is cacheable only when that whole window is
                // pinned by the key: the next piece is an identified turn that
                // is sealed-available NOW (the same check its inject will make
                // — a skipped target would silently swap the attended content)
                // AND long enough to cover the window (a shorter turn lets the
                // bridge spill into piece i+2, which the key does not name).
                // Anything else — no next piece, no timeline, short or missing
                // target — recomputes every wave.
                let bridge_cacheable = match (fwd, pieces.get(i + 1)) {
                    (0, _) => true,
                    (
                        _,
                        Some(AssembledPiece::Turn {
                            timeline: Some(tl),
                            index,
                            ..
                        }),
                    ) => {
                        let conv = ctx.conversation.read();
                        conv.turn_sealed_of(*tl, *index).is_some()
                            && conv.turn_token_count_of(*tl, *index) >= fwd as usize
                    }
                    _ => false,
                };
                // Island key = prefix identity + island tokens + bridge window +
                // (when bridging) the identity of the piece it leads into.
                let mut h = prefix_h.clone();
                h.write_u8(1);
                for &t in tokens.iter() {
                    h.write_u32(t);
                }
                h.write_u32(fwd);
                if fwd > 0 {
                    hash_piece_identity(&mut h, pieces.get(i + 1), ctx);
                }
                let key = h.finish();
                let start_block = block_count_of(ctx, parent_id)?;
                let gen = state.glue_generation;
                if let Some(cached) = bridge_cacheable
                    .then(|| {
                        state.glue_islands.get_mut(&key).map(|e| {
                            e.0 = gen; // refresh: this shape is live again
                            e.1.clone()
                        })
                    })
                    .flatten()
                {
                    // Cache hit: Arc-inject the captured K/V — the wave never
                    // sees this island. Mirrors the sealed-inject bookkeeping,
                    // the index page included: the wave not seeing the island is
                    // exactly why nothing would otherwise index it.
                    let what = InjectedPiece::CachedGlue {
                        tokens: tokens.len(),
                    };
                    let injected = inject_arc_sealed(ctx.session, parent_id, &cached.kv, &what)?;
                    let placed = injected.map_or(0, |inj| inj.tokens);
                    if let Some(inj) = injected.filter(SealedInjection::was_cut) {
                        // Captured by this process's own forward, so a skew here
                        // is a live fault, not a stored one: drop the capture and
                        // let the next rebuild forward the island again.
                        report_cut(
                            &what,
                            parent_id,
                            &cached.kv,
                            &inj,
                            "dropping the capture so the next rebuild forwards it",
                        );
                        state.glue_islands.remove(&key);
                    }
                    if let Some(inj) = &injected {
                        push_captured_page(ctx, &cached, inj, &what);
                    }
                    walker.logical_pos += placed as u32;
                    walker.last_was_sealed = true;
                    walker.reused_glue_tokens += tokens.len();
                    log_injected_tokens(ctx, tokens);
                    let end_block = block_count_of(ctx, parent_id)?;
                    islands.push(PlannedIsland {
                        key,
                        start_block,
                        end_block,
                        reused: true,
                        cacheable: true,
                    });
                } else {
                    // First cache miss of the walk: name the island so a
                    // whole-prefix divergence (a varying token in generated
                    // glue, a re-sealed early section) is attributable — the
                    // miss index says WHERE the prefix stopped matching.
                    if islands.iter().all(|i| i.reused) {
                        tracing::debug!(
                            target: "candle_conversation::scheduler::reproject",
                            island_idx = islands.len(),
                            piece_idx = i,
                            n_tokens = tokens.len(),
                            head = ?&tokens[..tokens.len().min(8)],
                            fwd,
                            "glue island cache: first miss of this walk"
                        );
                    }
                    reserve_glue_island(ctx, &mut walker, tokens, fwd)?;
                    let end_block = block_count_of(ctx, parent_id)?;
                    islands.push(PlannedIsland {
                        key,
                        start_block,
                        end_block,
                        reused: false,
                        cacheable: bridge_cacheable,
                    });
                }
                // The island is itself part of every LATER island's prefix.
                let mut c = DefaultHasher::new();
                c.write_u8(1);
                for &t in tokens.iter() {
                    c.write_u32(t);
                }
                c.write_u32(fwd);
                contrib = c.finish();
            }
            AssembledPiece::Section(_) | AssembledPiece::Turn { .. } => {
                contrib = walk_sealed_piece(ctx, &mut walker, &pieces[i])?;
            }
            AssembledPiece::DeferredUser(tokens) => {
                walker.deferred_user = Some(tokens.clone());
            }
            AssembledPiece::Hole(positions) => {
                place_hole(ctx, &mut walker, *positions)?;
                contrib = hole_contrib(*positions);
            }
        }
        check_drift(ctx, &walker, &|| {
            format!("piece {i} {}", piece_label(&pieces[i]))
        });
        // What the slot holds after this piece, for the next rebuild to keep.
        // The deferred user message is prefilled after the walk and never
        // part of a kept prefix.
        let page = walker.placed_page.take();
        if !matches!(pieces[i], AssembledPiece::DeferredUser(_)) {
            prefix_h.write_u64(contrib);
            let blocks_after = content_block_count_of(ctx, parent_id)?;
            let blocks = blocks_after.saturating_sub(content_before);
            content_before = blocks_after;
            state.placed_pieces.push(PlacedPiece {
                identity: aligned[i].0,
                kind: aligned[i].1,
                blocks,
                tokens: walker.logical_pos - pos_before,
                holes: walker.holes - holes_before,
                blocks_after,
                pos_after: walker.logical_pos,
                at_after: walker.logical_pos + walker.holes,
                contrib,
                prefix_h: prefix_h.clone(),
                range: placed_range(&pieces[i], blocks),
                page,
            });
        }
        walk_us[kind] += t_piece.elapsed().as_micros() as u64;
        walked[kind] += 1;
    }
    // Every re-placed piece now holds its own page; the rest leave with this.
    drop(held_records);
    let t_ranges = Instant::now();
    walker.ranges.flush();
    let ranges_us = t_ranges.elapsed().as_micros() as u64;
    // What the rebuild cost, by what it walked — a kept piece is free, a spliced
    // one costs itself, and every piece walked onto the tail is re-injected.
    note("apply split", || {
        format!(
            "slot={} identities_us={identities_us} cut_us={cut_us} splice_us={splice_us} \
             edits={} inserted={} evicted={} sections_us={} sections={} turns_us={} turns={} \
             glue_us={} glue={} ranges_us={ranges_us} turn_resolve_us={} turn_fetch_us={} \
             turn_fetched={} turn_lock_us={} turn_window_us={} turn_inject_us={} \
             turn_pages_us={} kept={} held_pages={}",
            parent_id.0,
            plan.edits.len(),
            plan.inserted(),
            plan.evicted(),
            walk_us[0],
            walked[0],
            walk_us[1],
            walked[1],
            walk_us[2],
            walked[2],
            walker.turn_resolve_us,
            walker.turn_fetch_us,
            walker.turn_fetched,
            walker.turn_lock_us,
            walker.turn_window_us,
            walker.turn_inject_us,
            walker.turn_pages_us,
            plan.kept(),
            state
                .placed_pieces
                .iter()
                .filter(|p| p.page.is_some())
                .count(),
        )
    });

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
        reused_glue_tokens = walker.reused_glue_tokens,
        deferred_user = walker.deferred_user.is_some(),
        pages_pushed = walker.pages_pushed,
        kept_pieces = walker.kept_pieces,
        walked_pieces = pieces.len() - walker.kept_pieces,
        "apply_segments: assembled slot prefix"
    );
    // Sealed K/V went in and no index page came with it. Every borrow site
    // pushes one or says why, so reaching here means the walk took a route that
    // does neither — the slot ends up holding a prefix nothing indexed, which is
    // silent until the select passes the identity threshold and refuses.
    if ctx.model.carries_positional_state() && walker.sealed_tokens > 0 && walker.pages_pushed == 0
    {
        tracing::warn!(
            slot = parent_id.0,
            sealed_tokens = walker.sealed_tokens,
            sealed_turns = walker.sealed_turns,
            sealed_sections = walker.sealed_sections,
            "apply_segments: injected sealed K/V but pushed no index page at all"
        );
    }

    // Where the system prompt ends, for a model whose selection treats it apart
    // (a stratified QSA selection ranks it in every window). Read from the
    // placement records before an incomplete assembly clears them.
    let prompt_end = prompt_end_of(&pieces, &state.placed_pieces);
    let prompt_span = profile::span("apply:set_selection_prompt");
    ctx.model
        .set_selection_prompt(parent_id.0, prompt_end as usize)
        .map_err(ConversationError::Model)?;
    // The pinned sections' places, which every selecting query attends whole
    // (`docs/hit_driven_projection.md` §5) — declared from the same records,
    // so they describe the layout the pages do.
    let pins = pin_ranges_of(new_segments, &pieces, &state.placed_pieces);
    ctx.model
        .set_selection_pins(parent_id.0, &pins)
        .map_err(ConversationError::Model)?;
    // The placed turns, declared so the sampled layers count what the model's
    // own selection reads of each until the next reprojection reads them back.
    // The slot is sized to the trained window, but the system prompt is not
    // counted where the shares are checked: a prompt past their margin takes
    // the slot past it, and the rung schedule changes every position's
    // rotation (`docs/hit_driven_projection.md` §4, §5).
    let reach = state
        .placed_pieces
        .last()
        .map_or(0, |p| p.at_after as usize);
    let trained = ctx
        .model
        .rope_select()
        .ceilings()
        .first()
        .copied()
        .unwrap_or(usize::MAX);
    if reach > trained {
        tracing::warn!(
            slot = parent_id.0,
            reach,
            trained,
            "projection placed past the trained window: the slot leaves the trained RoPE"
        );
    }
    // Hits only advise the scoring, so a declaration that fails costs this
    // stride its counts and nothing else. What the old pieces ran up since the
    // last read is read back first and kept for the next reprojection.
    match ctx.model.take_hits(parent_id.0) {
        Ok(Some(h)) if (h.queries > 0 || h.shadow_queries > 0) && !state.hit_keys.is_empty() => {
            state
                .unread_hits
                .push((std::mem::take(&mut state.hit_keys), h));
            if state.unread_hits.len() > UNREAD_HIT_PARTS {
                state.unread_hits.remove(0);
            }
        }
        Ok(_) => {}
        Err(e) => tracing::warn!(slot = parent_id.0, error = %e, "hits not read back"),
    }
    // The placed rounds, then every unplaced round of what the pool holds,
    // then each resident section out of scope — the rounds the shadow scan
    // reads (§6.3; `docs/placed_section_layout.md` §3.4).
    let (hit_keys, hit_ranges, pooled) = {
        let conv = ctx.conversation.read();
        let sections = kb_sections(&conv, ctx.slot_target);
        state.kb_rounds = sections
            .values()
            .copied()
            .chain(state.kb_resident.iter().map(|r| r.key))
            .map(|k| (k.scan_id(), k))
            .collect();
        let (mut keys, ranges) = hit_rounds_of(&pieces, &state.placed_pieces, &conv, &sections);
        let placed: HashSet<RoundKey> = keys.iter().copied().collect();
        let (pooled_keys, mut pooled) =
            pooled_rounds(&state.pool, &placed, |tl| conv.exchanges_of(tl));
        keys.extend(pooled_keys);
        let in_scope: HashSet<SectionKey> = sections.values().copied().collect();
        let (kb_keys, kb_pooled) = kb_pooled_rounds(&state.kb_resident, &in_scope);
        keys.extend(kb_keys);
        pooled.extend(kb_pooled);
        (keys, ranges, pooled)
    };
    match ctx.model.set_hit_rounds(parent_id.0, &hit_ranges, &pooled) {
        Ok(()) => state.hit_keys = hit_keys,
        Err(e) => {
            tracing::warn!(
                slot = parent_id.0,
                pieces = hit_ranges.len(),
                error = %e,
                "hit pieces not declared: this stride counts no hits"
            );
            state.hit_keys.clear();
        }
    }
    prompt_span.end();

    let complete = walker.skipped_turns == 0 && walker.skipped_sections == 0;
    // A slot missing a piece its list names does not hold what the list
    // describes, so no later rebuild may keep any of it.
    if !complete {
        state.placed_pieces.clear();
    }

    Ok(GapFillPlan {
        parent_id,
        glue_tokens: std::mem::take(&mut walker.glue_tokens),
        glue_write_slice: std::mem::take(&mut walker.glue_write_slice),
        glue_write_in_blk: std::mem::take(&mut walker.glue_write_in_blk),
        fwd_ahead: std::mem::take(&mut walker.fwd_ahead),
        deferred_user: walker.deferred_user.take(),
        tail_per_layer,
        held_before: Some(held_before),
        placed_before: state.placed,
        n_glue_tokens: walker.n_glue_tokens,
        islands,
        complete,
    })
}
