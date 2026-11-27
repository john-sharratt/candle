//! The prepare phase of a reprojection: it scores the turn's live probe,
//! reselects the working set, and detaches the turn's tail so the rebuild can run.
//!
//! The probe covers only the turn's own content and never reaches into the
//! system-prompt prefix, which would pin the selection to a tool's own definition.
//! The rebuild moves no bytes: the turn's tail and its index rows are taken from
//! the view, the view is freed, and the parent is truncated and rebuilt over the
//! new selection. The rebuild is priced as an admission is, and it waits while the
//! decodes are below their weight floor, since nothing has moved yet and not
//! starting costs nothing. The tail is captured from the turn's own start rather
//! than from the borrowed prefix, so the first half of the turn survives the swap.

use super::*;

impl Scheduler {
    /// Prepare phase of reprojection: BDP scan + projection + tier elevate +
    /// inject the sealed prefix + build the gap-fill descriptor. Returns the
    /// in-flight state for [`Self::reproject_view_complete`] (after the caller
    /// fires the batched gap-fill), or `None` if the view needs no reprojection.
    /// Removes the view's `DecodeState` into the in-flight on success.
    ///
    /// # The zero-copy rebuild
    ///
    /// Reprojection rebuilds the parent + view around the freshly-projected
    /// `(sections, turns)` selection, preserving the active turn's in-flight
    /// tokens **without any data copies or forward-pass work**. This function
    /// runs steps 1–6; [`Self::reproject_view_complete`] runs 7–8.
    ///
    /// An earlier implementation kept the same parent slot and just narrowed the
    /// view's borrow window onto it. That assumed the new selection was already
    /// materialised on parent in a contiguous prefix. Provenance-driven `top_k`
    /// swaps broke the assumption: newly-picked sections existed only in the
    /// substrate, so they were filtered out of the new borrow ranges, the view
    /// borrowed a gappy subset, `finalize_view`'s truncate-to-borrowed step then
    /// dropped the in-parent sections the new selection didn't keep, and the next
    /// reproject's range request landed past parent's now-shrunken end.
    ///
    /// The rebuild is fully metadata-only. The chunked backing's primitives
    /// (`inject_sealed_at_tail`, `create_view_sequence`, `extend_chunks`) all
    /// move chunks via `Arc<ChunkGid>` clones — no DMA, no kernel work, no
    /// forward pass. Steps:
    ///
    /// 1. **Probe + scan + project**: probe live Q from the recent decode
    ///    window, run provenance against substrate sigs to refresh section /
    ///    turn scores, project the new `(sections, turns)` selection.
    /// 2. **Capture the active turn's tail**: snapshot the view per layer as
    ///    `SealedSequence`s and slice off `chunks[original_borrowed..]`. Those
    ///    chunks hold every K/V byte computed since the view was carved (user
    ///    prefill + decoded-so-far). The Q vectors and provenance signatures for
    ///    the decoded tokens are already captured by provenance — nothing in the
    ///    tail needs regenerating.
    /// 3. **Free the old view**: drops its `ChunkGid` refs. The captured tail
    ///    Arc-refs keep the underlying chunks alive across the free, so the GPU
    ///    bytes survive untouched.
    /// 4. **Reset parent**: truncate to 0 chunks. Substrate-pinned section
    ///    chunks survive via the substrate's own Arc refs; parent just drops its
    ///    metadata pointers.
    /// 5. **Re-run `apply_projection`**: Arc-clones the new selection's sealed
    ///    chunks onto parent and patches their `block_range` in the substrate to
    ///    the new layout.
    /// 6. **`inject_sealed_at_tail` the captured tail onto parent**: Arc-clones
    ///    again, so parent now holds `[new prefix] + [active-turn tail]` with the
    ///    tail's exact K/V bytes preserved.
    /// 7. **Carve a fresh view borrowing all of parent**: the new view's
    ///    writer-owned chunk is the standard CoW copy of parent's trailing
    ///    partial chunk, ready for the next decoded token to extend.
    /// 8. **Re-key per-view state** (`active_decodes`, `sampling_states`,
    ///    `turn_views`) onto the new view id.
    ///
    /// # RoPE is recomputed transparently
    ///
    /// K bytes are stored un-rotated. Per-chunk `rope_base` is rederived every
    /// forward pass by [`SlotStateHost::from_sealed_chunks`] from the
    /// destination slot's cumulative chunk usage. Moving chunks between slots or
    /// shifting the prefix size automatically yields the right rotated K when the
    /// kernel reads them.
    ///
    /// # The one real semantic compromise: stale K_raw
    ///
    /// The tail's stored K and V bytes were computed by the model against the
    /// *old* prefix. Under the new prefix, a fresh forward pass would have
    /// produced slightly different K/V because the residual stream at every layer
    /// absorbed different attention output. Re-prefilling would burn a forward
    /// pass to "fix" this; the zero-copy path doesn't.
    ///
    /// For typical reproject use (small provenance-driven catalog swaps of
    /// semantically-adjacent sections), the staleness is negligible. Compounded
    /// over many reprojects fired in quick succession (e.g. every newline via
    /// `trigger_token_ids`), the drift can degrade generation quality — set the
    /// reproject cadence conservatively (`every_n_tokens` ~50+, or trigger on
    /// semantic-paragraph boundaries) to keep K_raw staleness within tolerance.
    pub(super) fn reproject_view_prepare(
        &mut self,
        view_id: SequenceId,
    ) -> Result<Option<ReprojectInFlight>, ConversationError> {
        let _t_total = PhaseTimer::new("reproject_total");
        // **Not while the decodes are short of their weights.** A reprojection's
        // scan uploads gallery pages for the turns it scores, before any lift is
        // known — and below the decode floor admission defends, every byte it
        // places is one the decode set is already streaming for. On the 3090
        // (2026-10-18) the gallery held 2.38 GB of scanned working set while the
        // experts stood at 3.07 of the 6.45 GB the engine needs. The turn keeps
        // its view; a later trigger asks again.
        if !self.decode_floor_holds_after(0) {
            return Ok(None);
        }
        // What the reprojection reads of its turn — a decode, or an opening
        // stopped at a boundary (`held_turn`) — cloned out, since the swap below
        // takes the turn out of the map it sits in.
        let Some(reading) = self.turn_reading(view_id) else {
            return Ok(None);
        };
        let policy = reading.policy;
        // Seed this reprojection's belief from the last one (empty on the
        // first) — the RelLeak decay/reinforcement carries across the turn.
        // The new belief is written back in `reproject_view_complete`.
        let mut prior_belief = reading.belief;
        // What the turn has appended so far — the reply, or the opening's
        // prefilled part: drives the early-decode grace window (lowered band +
        // carried-belief floor), so a correct pick whose decode-Q is still
        // accruing over the first ~64 tokens isn't evicted at token 1, and is
        // what the slot's room is counted less.
        let decode_pos = Some(reading.appended);
        // This turn's first reprojection is the turn boundary: no projection has
        // run against the turn's own content yet.
        let is_turn_boundary = reading.is_turn_boundary;

        // 1. Compute the probe's CHUNK range — the turn's own content, in the
        //    same chunk coordinates the seal captures for the belief gallery.
        //
        //    The seal gathers `gather_wide_sigs(seal_slot, (seal_block_from,
        //    block_count))` where `seal_block_from == turn_start_parent_blocks`
        //    (see the turn-complete handler) and `block_count ==
        //    sequence_block_count`.  Those stored signatures are the belief
        //    gallery, so the live probe covers the same turn-content domain:
        //      • It never reaches before the turn boundary into the materialized
        //        system-prompt / tools prefix — a probe that swept the prefix
        //        would score the selected tool's own definition against its
        //        gallery and pin the selection to it with a turn-independent
        //        constant.
        //      • Chunk indices are partial-aware: `sequence_block_count` is the
        //        authoritative chunk total (the same call the seal uses), so
        //        partial chunks from glue/section boundaries can't shrink the
        //        probe below the turn's real tokens.
        //
        //    For turns longer than `max_probe_tokens`, the probe is the most
        //    recent chunks capped to that budget PLUS the turn's first chunks
        //    (the user query): the query is the strongest intent signal in the
        //    gallery domain, so it stays in every scan of the turn rather than
        //    sliding out of a purely trailing window.
        let view_state = self.turn_views.get(&view_id).copied().ok_or_else(|| {
            ConversationError::Channel(format!("reproject: missing view state {view_id}"))
        })?;
        // Wall-clock start of the whole reproject — drives `total_ms` so the log
        // reports the real end-to-end cost, not a sum of (partly overlapping)
        // phase fields.
        let t_repro = Instant::now();

        // 2. Gather the live wide-Q probe — folded per-token sign(Q): the query
        //    head (when the trailing window has slid past it) followed by the
        //    most recent turn chunks, and the pinned question window, which
        //    policies with `question_pin` scan as a second probe and max-fuse
        //    (`turn_probe`). A gather covering no real tokens means nothing to
        //    score.
        let t_probe = Instant::now();
        let probe_span = profile::span("reproject:probe");
        let Some(TurnProbe {
            probe,
            question: probe_q,
        }) = self.turn_probe(
            view_id,
            view_state.turn_start_parent_blocks,
            view_state.question_tokens,
            policy.max_probe_tokens,
        )
        else {
            return Ok(None);
        };
        probe_span.end();
        let probe_ms = t_probe.elapsed().as_millis() as u64;
        self.reproject_census
            .add_since(ReprojectPart::Probe, t_probe);
        record_phase(t_probe, "reproject_probe_extract");
        REPROJ_SCAN_US.fetch_add(
            t_probe.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // 4. Wide-Q belief scoring: scan the probe against each belief-driven
        //    collection's tag-scoped gallery of past turns→selected-section AND
        //    each belief-driven turn group's own turns (self-match), producing the
        //    per-section / per-turn scores the belief policy selects from in
        //    `project()`. No provenance scan, no persisted scores — projection-local.
        //    `group_candidates` carries each turn group's freshly-scored turns for
        //    the turn-boundary challenger below.
        let t_scan = Instant::now();
        let warm_span = profile::span("reproject:queue_norm_warm");
        self.queue_norm_warm(
            &policy.substrate,
            &policy.projection,
            policy.target,
            policy.max_probe_tokens,
        );
        warm_span.end();
        let warm_us = t_scan.elapsed().as_micros() as u64;
        let schema = policy.projection.schema();
        let t_score = Instant::now();
        let scan_span = profile::span("reproject:score_beliefs");
        // observe = false: a live reprojection only READS the normalization hit
        // levels; learning happens once per turn at seal (last_turn_belief_scores).
        // The hot reproject path scans on the GPU (paged segmented BDP over the
        // resident gallery arena, per-file z) — one launch for the whole group,
        // numerically equivalent to the CPU per-file scan up to fast-math ULP /
        // same ranking (see `examples/gpu_belief_parity.rs`). Seal-time learning
        // runs the same launches here (`SchedulerRequest::ArenaScan`), so the
        // levels are learned from the scores they later normalize.
        let (mut projection_scores, group_candidates) = policy.substrate.score_beliefs(
            schema,
            policy.target,
            &probe,
            (!probe_q.is_empty()).then_some(probe_q.as_slice()),
            // A live reprojection only READS the levels; the turn teaches them
            // once, at seal, where its tags are known.
            Observe::No,
            self.gallery_arena.as_deref().map(|a| a as &dyn BeliefArena),
        );
        scan_span.end();
        let score_us = t_score.elapsed().as_micros() as u64;
        // The same fresh scores feed the dialogue's working-set momentum — once
        // per reprojection, the cadence the momentum's leak is defined over
        // (`docs/zend_working_set.md` §4.3).
        let t_observe = Instant::now();
        let observe_span = profile::span("reproject:observe_working_set");
        // What the model's own selection read of each placed conversation since
        // the last projection — read once here, at the boundary, and zeroed.
        // Hits only advise the scoring: a read that fails leaves this
        // reprojection judging by momentum alone.
        // The pieces were declared on the parent this view was carved from —
        // the slot the projection rebuilds — and the view counts into them.
        let slot = self
            .turn_views
            .get(&view_id)
            .map_or(view_id, |v| v.parent_id);
        // Read only where there is a slot state to fold them into: a read zeroes
        // the counters.
        let hits = match self.slot_projection_state.contains_key(&slot) {
            true => match self.model.take_hits(slot.0) {
                Ok(h) => h,
                Err(e) => {
                    tracing::warn!(slot = slot.0, error = %e, "hits not read: scored without them");
                    None
                }
            },
            false => None,
        };
        // What the read held and what it lifted, for the observe split: the
        // turns whose rounds' momentum lifts their scores, and the strongest.
        let mut hit_read = hit_density::HitSummary::default();
        let mut lifted = 0usize;
        let mut top_lift = 0.0f32;
        let density = match self.slot_projection_state.get_mut(&slot) {
            Some(state) => {
                let mut parts = std::mem::take(&mut state.unread_hits);
                if let Some(h) = hits {
                    parts.push((state.hit_keys.clone(), h));
                }
                hit_read = hit_density::hit_summary(&parts);
                // Each placed round's hit momentum moves, and lifts this
                // reprojection's scores of the turns whose rounds it keeps
                // reading (`docs/hit_driven_projection.md` §7.1, §8.1).
                state
                    .round_hits
                    .fold(hit_density::round_density(&parts), &schema.hits);
                // Reads are credited over the same base the target's lineage
                // is ranked from, so a turn displaces the lineage only once
                // its reads pass it (§8.2).
                let base = schema
                    .layers
                    .iter()
                    .find(|l| l.id == policy.target.layer)
                    .and_then(|l| {
                        l.groups.iter().find_map(|g| match g.selection {
                            SelectionRule::Sequence { lineage_base, .. } => Some(lineage_base),
                            _ => None,
                        })
                    })
                    .unwrap_or(LINEAGE_BASE);
                // Credit for unscored reads goes to the conversation, its
                // lineage and the pooled pieces: every round of the fill is
                // ranked on the one band (§7.1), so a pooled turn the shadow
                // scan keeps reading is brought in though provenance did not
                // score it.
                let chain: HashSet<TimelineId> = policy
                    .substrate
                    .read()
                    .inherited_chain(policy.target.timeline)
                    .into_iter()
                    .collect();
                let turn_lifts = {
                    let read = policy.substrate.read();
                    let mut exchanges: HashMap<TimelineId, Vec<Vec<TurnIndex>>> = HashMap::new();
                    hit_density::turn_lifts(state.round_hits.lifts(&schema.hits), |tl, head| {
                        exchanges
                            .entry(tl)
                            .or_insert_with(|| read.exchanges_of(tl))
                            .iter()
                            .find(|turns| turns.first() == Some(&head))
                            .cloned()
                    })
                };
                lifted = turn_lifts.len();
                top_lift = turn_lifts.values().copied().fold(0.0, f32::max);
                note("reproject piece lifts", || {
                    hit_density::piece_lifts_line(&turn_lifts, 10)
                });
                let pool = &state.pool;
                projection_scores.lift_turns(turn_lifts, base, |k| {
                    chain.contains(&k.timeline) || pool.contains_key(&k.timeline)
                });
                hit_density::hit_density(&parts)
            }
            None => HashMap::new(),
        };
        let hits_us = t_observe.elapsed().as_micros() as u64;
        let t_ws = Instant::now();
        // The working set is the pool: it chooses its pieces within this
        // card's size (`docs/hit_driven_projection.md` §7.2 step 1).
        if policy.substrate.read().pool_size() != self.pool_size {
            let size = self.pool_size;
            policy.substrate.update(|s| s.set_pool_size(size));
        }
        // The boundary: what the slot holds now, for the fill to keep seated
        // and to count entries against (§7.2 step 5).
        let placed_turns = self
            .slot_projection_state
            .get(&slot)
            .map(|s| s.placed_turns())
            .unwrap_or_default();
        policy.substrate.observe_working_set(
            schema,
            policy.target,
            &placed_turns,
            &group_candidates,
            &density,
        );
        let ws_us = t_ws.elapsed().as_micros() as u64;
        let ws_tracked = policy
            .substrate
            .read()
            .working_set(policy.target.timeline)
            .map_or(0, |ws| ws.tracked());
        // The pool behind the slot: the pieces it places and the working
        // set's members (§7.2 step 1).
        let t_pool = Instant::now();
        let pool = self.refresh_pool(slot, &policy.substrate, policy.target.timeline);
        let pool_us = t_pool.elapsed().as_micros() as u64;
        observe_span.end();
        let observe_us = t_observe.elapsed().as_micros() as u64;
        for (part, us) in [
            (ReprojectPart::Warm, warm_us),
            (ReprojectPart::Score, score_us),
            (ReprojectPart::Hits, hits_us),
            (ReprojectPart::WorkingSet, ws_us),
            (ReprojectPart::Pool, pool_us),
        ] {
            self.reproject_census.add(part, us);
        }
        // `observe_us` by part, on every reprojection — one whose selection is
        // unchanged reports no prepare split, and it is the one a decode waits
        // on most often.
        note("reproject observe split", || {
            format!(
                "view={} observe_us={observe_us} hits_us={hits_us} ws_us={ws_us} \
                 ws_tracked={ws_tracked} \
                 pool_us={pool_us} pooled={} pool_admitted={} pool_dropped={} \
                 pool_revived={} pool_lingering={} pool_cold={} pool_fetch_us={} \
                 pool_place_us={} \
                 pool_read_us={} pool_read_bytes={} pool_reused={} pool_walked_turns={} \
                 candidates={} hit_parts={} queries={} shadow_queries={} \
                 loaded_rounds={} loaded_hit={} loaded_hits={} pooled_rounds={} \
                 pooled_hit={} pooled_hits={} turns_lifted={lifted} top_lift={top_lift:.3}",
                view_id.0,
                pool.pooled,
                pool.admitted,
                pool.dropped,
                pool.revived,
                pool.lingering,
                pool.cold,
                pool.fetch_us,
                pool.place_us,
                pool.read_us,
                pool.read_bytes,
                pool.reused,
                pool.walked,
                group_candidates.iter().map(|(_, c)| c.len()).sum::<usize>(),
                hit_read.parts,
                hit_read.queries,
                hit_read.shadow_queries,
                hit_read.loaded,
                hit_read.loaded_hit,
                hit_read.loaded_hits,
                hit_read.pooled,
                hit_read.pooled_hit,
                hit_read.pooled_hits,
            )
        });

        let scan_ms = t_scan.elapsed().as_millis() as u64;
        record_phase(t_scan, "reproject_belief_scan");
        REPROJ_SCAN_US.fetch_add(
            t_scan.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // Turn-boundary challenger: on this turn's FIRST reprojection, give each
        // top-N belief collection's strongest fresh signal a slot even when it's
        // below `min_score`, evicting the weakest carried incumbent only if the
        // selection is full. Lets a topic-changed query's new intent break in
        // without lowering the threshold (which would admit noise mid-turn); the
        // strong carried signals survive, and RelLeak decays the challenger back
        // out over the turn if its fresh score doesn't hold up.
        let t_challenger = Instant::now();
        if is_turn_boundary {
            // Collections (tool catalog) live in the shared system prompt:
            // challenge on section fresh scores.
            {
                for item in &schema.system_prompt.items {
                    if let SystemPromptItem::Collection(coll) = item {
                        let budget_max = coll.policy.config.budget_max;
                        if budget_max < 3 {
                            continue;
                        }
                        let fresh: Vec<(String, f32)> = coll
                            .sections
                            .iter()
                            .map(|s| (s.name.clone(), projection_scores.section(s.id)))
                            .collect();
                        // Seed the challenger at the *windowed* selection bar so, in
                        // the opening grace window, it lands beside the floored
                        // carried picks (250) rather than dominating them at 1000.
                        let (wcfg, _) = coll.policy.config.windowed(decode_pos);
                        prior_belief.seat_turn_boundary_challenger(
                            GroupKey::Collection(coll.name.clone()),
                            &fresh,
                            budget_max,
                            wcfg.min_score,
                        );
                    }
                }
            }
            // Belief-driven turn groups span every layer (memory tiers, repo
            // map): look each candidate group up across the whole schema and
            // challenge on its per-turn fresh scores, keyed by turn index.
            for (gid, cands) in &group_candidates {
                let Some(group) = schema
                    .layers
                    .iter()
                    .flat_map(|l| l.groups.iter())
                    .find(|g| g.id == *gid)
                else {
                    continue;
                };
                // A working-set group is not selected by belief, so there is no
                // carried selection for a challenger to break into.
                if group.is_working_set() {
                    continue;
                }
                let cfg = group.belief_config(cands.len());
                if cfg.budget_max < 3 {
                    continue;
                }
                let (wcfg, _) = cfg.windowed(decode_pos);
                // Key by the SAME `<timeline>:<index>` the belief carry uses — a
                // group holds many conversations, so a bare index would seat the
                // challenger under a key the carry never reads back.
                let fresh: Vec<(String, f32)> = cands
                    .iter()
                    .map(|(key, score)| (crate::projection::turn_belief_key(*key), *score))
                    .collect();
                prior_belief.seat_turn_boundary_challenger(
                    GroupKey::TurnGroup(group.name.clone()),
                    &fresh,
                    cfg.budget_max,
                    wcfg.min_score,
                );
            }
        }
        self.reproject_census
            .add_since(ReprojectPart::Challenger, t_challenger);

        // 5. Re-project against the freshly-scored substrate.
        //
        //    Produce the same `(projected_sections, projected_turns)`
        //    pair that `SchedulerRequest::SubmitTurn` builds: only
        //    sealed substrate entries make the cut.  These two lists
        //    drive the zero-copy rebuild in step 6.
        let t_project = Instant::now();
        let project_span = profile::span("reproject:project");
        let mut turn_keys_for_elevate: Vec<TurnKey> = Vec::new();
        let (projected_sections, projected_segments, composition) = {
            let view = policy
                .substrate
                .read_for_scored(policy.target, &projection_scores);
            // Prefill mode: the section corpus is prefill-Q of tool
            // descriptions, so reprojection scores it with the calibrated
            // prefill profile (Max / semantic, no threshold gate).
            let projection = policy.projection.project_with_mode_and_sink(
                policy.target,
                &view,
                ProjectionMode::Prefill,
                &policy.selection,
                &prior_belief,
                decode_pos,
                &mut |_| {},
            );
            // Walk segments once, populating both the elevate side-lists
            // and the segment list `apply_projection` will diff against
            // the slot's previous projection.
            let mut sections: Vec<SectionId> = Vec::new();
            let mut segments: Vec<ProjectionSegment> = Vec::new();
            for seg in &projection.segments {
                match seg {
                    ProjectionSegment::Sealed(SealedKind::Section(rs)) => {
                        // Any-tier filter — same logic as the
                        // SubmitTurn path.  Cold-marker sections
                        // survive the filter; `elevate_to_hot`
                        // below lifts them before
                        // `apply_projection` injects them.
                        if view.section_tier_state(rs.id).is_some() {
                            sections.push(rs.id);
                            segments.push(seg.clone());
                        }
                    }
                    ProjectionSegment::Sealed(SealedKind::Turn(rt, _)) => {
                        // The turn carries its conversation (stamped at projection),
                        // so read it directly — no group→timeline re-derivation,
                        // which is what once resolved the first-registered timeline
                        // and dropped every non-first conversation's turns here (the
                        // reproject `turns=0` history loss).  A timeline-less turn
                        // (mock/untracked) is skipped; a genuinely-missing one is
                        // surfaced loudly at inject, not silently filtered.
                        let Some(timeline) = rt.timeline else {
                            continue;
                        };
                        turn_keys_for_elevate.push(TurnKey::new(timeline, rt.index()));
                        segments.push(seg.clone());
                    }
                    ProjectionSegment::Generated { .. } => {
                        // Keep prefix structural / live-prefill runs in the
                        // segment chain, exactly as the SubmitTurn path does
                        // (see the matching arm where submit builds its
                        // `segments`). These structural tokens are part of the
                        // assembled prefix; dropping them here would make the
                        // reproject's rebuilt prefix diverge from submit's.
                        segments.push(seg.clone());
                    }
                    // A hole is part of the rebuilt prefix's layout, as in
                    // the SubmitTurn path: every position after it depends on it.
                    ProjectionSegment::Hole { .. } => segments.push(seg.clone()),
                    ProjectionSegment::NewUserMessage { .. } => {
                        // The active turn's user message is captured as the
                        // writer tail (snapshot/restore in the swap), so it must
                        // NOT be re-injected into the rebuilt prefix here.
                    }
                }
            }
            // Bucket the materialized segments by category (system / section
            // groups / turns) for the GUI. Timing is zeroed here and filled in
            // at `complete` from the per-sequence span anchor.
            //
            // Build the GUI composition from the FULL `projection.segments`, not
            // the tier-filtered `segments` used for apply: the filtered list
            // drops the active turn's user message (captured as the writer tail,
            // not re-injected) and any tier-less turn, which would leave the
            // panel's conversation section empty on a first turn. The decode-end
            // path (`projection_event`) buckets the full segments too, so this
            // keeps the live reproject view consistent with it.
            let mut composition = crate::projection::from_projection_with_origins(
                &projection.segments,
                &projection.selection_origins,
                policy.projection.schema(),
                &view,
                &projection.selection_scores,
                view.total_token_count(policy.target.timeline) as u32,
                0,
                0.0,
            );
            composition.pieces = projection.fill.clone();
            composition.unpooled = projection.unpooled.clone();
            composition.placed = projection.files_placed.clone();
            // The knowledge base's range as placed, for the next projection
            // to keep.
            if let Some(scope) = projection.files_scope.clone() {
                policy
                    .substrate
                    .update(|s| s.files_scope(policy.target.timeline, scope.clone()));
            }
            // The dialogue glue, sourced from the SAME `assemble_pieces` decision
            // the engine injects from — so the panel shows the real boundary
            // markers, never a reconstruction. `decode` keeps special tokens (the
            // markers ARE special tokens).
            composition.materialized = projection_assembler::materialize_conversation(
                &projection.segments,
                &projection.selection_origins,
                &view,
                policy.projection.schema(),
                |toks| self.tokenizer.decode(toks, false).unwrap_or_default(),
            );
            (sections, segments, composition)
        };
        project_span.end();
        let project_ms = t_project.elapsed().as_millis() as u64;
        self.reproject_census
            .add_since(ReprojectPart::Project, t_project);
        record_phase(t_project, "reproject_project");
        REPROJ_LAYOUT_US.fetch_add(
            t_project.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        self.reproject_rebuild(
            view_id,
            Selected {
                sections: projected_sections,
                segments: projected_segments,
                turns: turn_keys_for_elevate,
                composition,
            },
            PrepareTimes {
                t_repro,
                probe_ms,
                scan_ms,
                project_ms,
                warm_us,
                score_us,
                observe_us,
                probe_tokens: probe.len(),
            },
        )
    }

    /// The rebuild half of a reprojection of `view_id` onto `selected`: kept
    /// when the parent already holds it, declined below the decode floor,
    /// and otherwise the zero-copy rebuild described on
    /// [`Self::reproject_view_prepare`] — the turn's tail captured, the view
    /// freed, the parent re-assembled — up to the gap-fill descriptor.
    /// `times` are what the selection took.
    pub(super) fn reproject_rebuild(
        &mut self,
        view_id: SequenceId,
        selected: Selected,
        times: PrepareTimes,
    ) -> Result<Option<ReprojectInFlight>, ConversationError> {
        let Selected {
            sections: projected_sections,
            segments: projected_segments,
            turns: turn_keys_for_elevate,
            composition,
        } = selected;
        let PrepareTimes {
            t_repro,
            probe_ms,
            scan_ms,
            project_ms,
            warm_us,
            score_us,
            observe_us,
            probe_tokens,
        } = times;
        let view_state = self.turn_views.get(&view_id).copied().ok_or_else(|| {
            ConversationError::Channel(format!("reproject: missing view state {view_id}"))
        })?;
        let policy = self
            .turn_reading(view_id)
            .ok_or_else(|| {
                ConversationError::Channel(format!("reproject: view {view_id} has no turn"))
            })?
            .policy;
        let parent_id = view_state.parent_id;

        // Count of sealed turns this projection selected — the reproject cost
        // scales linearly with it (each adds one boundary-glue prefill).
        let n_turns_selected = projected_segments
            .iter()
            .filter(|s| matches!(s, ProjectionSegment::Sealed(SealedKind::Turn(..))))
            .count();

        // The parent already holds this selection when its last rebuild came
        // from a segment list with the same identity and placed all of it:
        // rebuilding would reproduce it exactly, at ~190 ms a time.
        let t_identity = Instant::now();
        let identity = {
            let _g = profile::span("reproject:identity");
            let read = policy.substrate.read();
            segments_identity(&projected_segments, |id| {
                section_content_stamp(&read.section_tokens_of(id))
            })
        };
        let identity_us = t_identity.elapsed().as_micros() as u64;
        self.reproject_census
            .add(ReprojectPart::Identity, identity_us);
        if view_state.applied_identity == Some(identity) {
            self.keep_unchanged_projection(view_id, composition);
            self.end_reprojection(ReprojectOutcome::Kept, t_repro);
            tracing::debug!(
                target: "candle_conversation::scheduler::reproject",
                view = view_id.0,
                total_ms = t_repro.elapsed().as_millis() as u64,
                probe_ms,
                scan_ms,
                project_ms,
                "reproject: selection unchanged — the parent is kept"
            );
            return Ok(None);
        }
        // **A rebuild spends residency as an admission does, and is judged by
        // the same line.** Its lift — the newly selected history brought hot and
        // its index pages placed — is ground the weight side loses, and a
        // decoding turn's rebuild adds no rows to repay it: the decodes it would
        // slow are the only thing at stake. Below the decode floor admission
        // defends, the turn keeps decoding on the view it has, and a later
        // trigger asks again. Nothing has moved yet, so not starting is free.
        let t_lift = Instant::now();
        let lift = self.projection_lift(
            &policy.substrate,
            &projected_sections,
            &turn_keys_for_elevate,
        );
        let floor_holds = self.decode_floor_holds_after(lift);
        self.reproject_census.add_since(ReprojectPart::Lift, t_lift);
        if !floor_holds {
            self.end_reprojection(ReprojectOutcome::Declined, t_repro);
            tracing::debug!(
                target: "candle_conversation::scheduler::reproject",
                view = view_id.0,
                lift_mib = lift >> 20,
                "reproject: the rebuild's lift would take the weights below the decode floor — \
                 the turn keeps its view"
            );
            return Ok(None);
        }
        // The rebuild below — capture, seal and re-inject the active turn, then
        // re-assemble the parent — scales with the turn's length, and on a turn's
        // first reprojection that is the whole prompt.
        let _rebuild = profile::span("reproject:rebuild");

        // 6. Zero-copy rebuild.
        //
        //    The previous implementation tried to *narrow* the view's
        //    borrow window over a still-correct parent.  That assumed
        //    the new projection's selected sections / turns were
        //    already materialised on parent in a contiguous prefix.
        //    provenance-driven `top_k` swaps invalidated that assumption mid-
        //    decode (newly-picked sections only exist in the substrate,
        //    not in parent), `finalize_view`'s truncate-to-borrowed
        //    step then dropped the in-parent sections that the new
        //    selection didn't keep, and the next reproject borrowed
        //    block indices past parent's now-shrunken end.
        //
        //    The rebuild is fully metadata-only:
        //
        //      a. Snapshot the view per-layer to capture every
        //         `ChunkGid` it holds — the borrowed prefix *and*
        //         the writer-owned tail.  Slice off chunks beyond
        //         `original_borrowed`; those are the active turn's
        //         writes (user prefill + decoded-so-far).  Their K
        //         bytes are un-rotated, V bytes are
        //         position-independent, Q vectors and provenance signatures
        //         for the decoded tokens are already captured by
        //         provenance — nothing needs regenerating.
        //
        //      b. Free the old view slot.  The captured tail
        //         `ChunkGid`s keep the underlying chunks alive via
        //         their Arc refs even after the view drops.
        //
        //      c. Truncate parent to 0.  The old prefix's section
        //         chunks survive in the substrate (which still holds
        //         Arc refs to them); parent's metadata pointers are
        //         the only thing dropped.
        //
        //      d. Re-run `apply_projection` against parent with the
        //         new `(sections, turns)` list.  This Arc-clones
        //         each selected section / turn's sealed chunks onto
        //         parent and patches their `block_range` in the
        //         substrate so future lookups resolve to the new
        //         layout.
        //
        //      e. `inject_sealed_at_tail` the captured tail onto
        //         parent — Arc-clones again, no DMA.  Parent now
        //         holds `[new prefix] + [active-turn tail]`.
        //
        //      f. Carve a fresh view borrowing all of parent.  The
        //         view's writer-owned tail will be the CoW copy of
        //         parent's trailing partial chunk (create_view's
        //         normal partial-tail handling), ready for the next
        //         decoded token to extend.
        //
        //      g. Re-key every per-view map (`active_decodes`,
        //         `sampling_states`, `turn_views`) onto the new
        //         view id.
        //
        //    RoPE just works: each chunk's `rope_base` is rederived
        //    from cumulative usage at every forward pass by
        //    `SlotStateHost::from_sealed_chunks` — the same un-rotated
        //    K bytes yield the right rotated K at the new positions
        //    when the kernel reads them.  Zero bytes copied; the only
        //    side effect is the chunk pool's refcounts rebalancing.
        let t_swap = Instant::now();
        // Capture the active turn's tail from `turn_start_parent_blocks`,
        // not `original_borrowed`.
        //
        // The active turn's content begins at `turn_start_parent_blocks`
        // in the view's chunk list: for a freshly-submitted turn that
        // index is the COW chunk (`= original_borrowed` when the
        // parent had a partial tail); for a view that's already been
        // through one or more zero-copy reprojects, the active turn's
        // earliest content (the original COW, then later writes) lives
        // in the *borrowed* prefix of the view (Arc'd from parent),
        // *not* in the writer-owned tail.  Using `original_borrowed`
        // here would silently drop everything between
        // `turn_start_parent_blocks..original_borrowed` — i.e. the
        // user prefill and all decoding up to the previous reproject —
        // leaving only the latest CoW + most recent writer chunks.
        // That dropped content is exactly the "the model has lost the
        // user message" symptom that derails generation past the
        // first reproject.
        // The view is rebuilt from this snapshot and freed: the outstanding
        // gap's rows for its blocks go in place first, so the snapshot carries
        // them converted.
        let t_gap = Instant::now();
        self.catch_up_outstanding_gap();
        self.reproject_census.add_since(ReprojectPart::Gap, t_gap);
        let t_snapshot = Instant::now();
        let tail_per_layer = {
            let _g = profile::span("reproject:snapshot_tail");
            // Reconcile a windowed creep's per-layer skew before this snapshot
            // — see `perform_seal_and_write` and
            // `BatchedInferenceSession::reconcile_block_counts`. This capture
            // takes each layer's own `chunks.len()` rather than a shared
            // min-based bound, so an unreconciled skew would carry a
            // different tail length per layer into the rebuilt view.
            self.session
                .reconcile_block_counts(view_id.0)
                .map_err(ConversationError::Model)?;
            // The turn's blocks only — never the whole slot, which holds the
            // entire projected context ahead of them.
            self.session
                .snapshot_sequence_tail(view_id.0, view_state.turn_start_parent_blocks)
                .map_err(ConversationError::Model)?
        };
        // Pull the turn and its sampling state off the old view id before
        // freeing the slot, then re-bind to the new view id after the
        // rebuild. Failing to find the turn is an internal invariant
        // violation — a reprojection is queued only for a decoding view or an
        // opening stopped at a boundary.
        let turn = self.take_turn(view_id).ok_or_else(|| {
            ConversationError::Channel(format!("reproject: missing turn state for view {view_id}"))
        })?;
        let sampling_state = self.sampling_states.remove(&view_id);
        // Carry the turn-scoped view metadata (the Concept F question boundary)
        // into the re-carved view below.
        let old_view_state = self.turn_views.get(&view_id).copied();
        // Diagnostic log: drop the freed view's per-slot token mirror.
        // The kernel will recycle this slot index for the new view we
        // carve below; without this drop, the new view's `slot_tokens`
        // would carry every decode-step input ever pushed under this
        // slot id across the turn's reprojects, producing duplicated
        // tokens in the turn-complete context dump that don't reflect
        // anything actually in the KV cache.
        self.slot_tokens.remove(&view_id);
        // **This path finalizes, so the state MOVES — it is not discarded.**
        //
        // The KV is preserved: `tail_per_layer` above captured the active
        // turn's chunks and they are re-injected onto the parent below. By the
        // disposition rule the recurrent state follows the K/V, so the turn's
        // decode so far has to survive with it. Releasing here instead would
        // silently lose every token decoded before the reprojection — the
        // reprojected turn would read fluently and have forgotten its own
        // first half.
        //
        // **The index rows for `tail_per_layer`, taken from the VIEW, before
        // anything moves.**
        //
        // The K/V for the turn so far survives this swap as `tail_per_layer` and
        // is injected back onto the rebuilt parent. Its index rows do not travel
        // with it: the move below puts them on the parent, and the re-projection
        // immediately resets the parent's per-position state and replays the
        // prefix from pages. So the turn's K/V comes back and the rows
        // describing it do not, leaving the slot short by exactly this turn's
        // width — silent while the conversation sits under the QSA identity
        // threshold, and a refused select the moment it passes it.
        //
        // **Sealed by WIDTH, not from a position, and read from the view.** The
        // turn's rows are not all in its live tail: a previous reprojection
        // re-injects the turn's user half from cache and pushes it as a page, so
        // the turn spans a page boundary. `seal_positional_tail_span` walks back
        // over whole pages until the width is covered and flushes the partial
        // block, which is the only formulation that takes the whole turn.
        // Sealing the live tail alone drops the user's own message — measured at
        // 14 tokens, after which the model decoded a question it could not
        // attend to and answered a different one it invented.
        //
        // **From here the turn's state is held by this reprojection alone**, so
        // every exit goes through `or_abandon!`: the turn is told why and
        // discarded, never dropped unannounced (`reproject_abandon.rs`). Until
        // the free below, the view's slot is still held.
        let snapshot_us = t_snapshot.elapsed().as_micros() as u64;
        let turn_tokens = tail_per_layer.first().map(|s| s.token_count).unwrap_or(0);
        let t_seal = Instant::now();
        let seal_span = profile::span("reproject:seal_tail_index");
        let tail_index_page = or_abandon!(
            self,
            view_id,
            parent_id,
            turn,
            true,
            self.model
                .seal_positional_tail_span(view_id.0, turn_tokens)
                .map_err(ConversationError::Model)
        );
        seal_span.end();
        let seal_us = t_seal.elapsed().as_micros() as u64;
        let t_free = Instant::now();
        let free_span = profile::span("reproject:move_free_truncate");
        // Ordering: move before the free, or there is nothing left to read.
        // The move already removes the view's entry, so the release below is
        // just the general per-sequence cleanup.
        or_abandon!(
            self,
            view_id,
            parent_id,
            turn,
            true,
            self.model
                .move_recurrent(view_id.0, parent_id.0)
                .map_err(ConversationError::Model)
        );
        let move_us = t_free.elapsed().as_micros() as u64;
        // The parent has the turn's state now. Until here the old view stayed
        // registered, so a failure above could hand back a state it held
        // (`abandon_reprojection`).
        self.turn_views.remove(&view_id);
        let t_free_view = Instant::now();
        or_abandon!(
            self,
            view_id,
            parent_id,
            turn,
            true,
            self.session
                .free_sequence(view_id.0)
                .map_err(ConversationError::Model)
        );
        let free_view_us = t_free_view.elapsed().as_micros() as u64;
        let t_release = Instant::now();
        or_abandon!(
            self,
            view_id,
            parent_id,
            turn,
            false,
            self.model
                .release_sequence(view_id.0)
                .map_err(ConversationError::Model)
        );
        let release_us = t_release.elapsed().as_micros() as u64;
        let t_truncate = Instant::now();

        // Cut the parent back to the pieces its last assembly placed — the
        // active turn it also holds past them was captured above from the
        // view and goes back on after the rebuild. The assembler then keeps
        // whatever prefix of those pieces the new projection shares and
        // re-injects only from the first that differs.
        or_abandon!(
            self,
            view_id,
            parent_id,
            turn,
            false,
            self.session
                .truncate_sequence_to_blocks(parent_id.0, view_state.turn_start_parent_blocks)
                .map_err(ConversationError::Model)
        );
        let truncate_us = t_truncate.elapsed().as_micros() as u64;
        // Reset the slot_tokens diagnostic log so it stays in sync with
        // the post-rebuild slot contents.  apply_projection re-populates
        // it with the new prefix's tokens; the active turn tokens get
        // re-added below.  Gated on `context-dump` — the map is empty
        // when the feature is off, so the get_mut would no-op anyway.
        #[cfg(feature = "context-dump")]
        if let Some(toks) = self.slot_tokens.get_mut(&parent_id) {
            toks.clear();
        }
        free_span.end();
        let free_us = t_free.elapsed().as_micros() as u64;
        for (part, us) in [
            (ReprojectPart::Snapshot, snapshot_us),
            (ReprojectPart::Seal, seal_us),
            (ReprojectPart::Move, move_us),
            (ReprojectPart::FreeView, free_view_us),
            (ReprojectPart::Release, release_us),
            (ReprojectPart::Truncate, truncate_us),
        ] {
            self.reproject_census.add(part, us);
        }
        // The pure swap work (tail snapshot + free + truncate) ends here —
        // captured disjoint from elevate/glue/complete so `swap_ms` reflects
        // only the chunk-pool rebalance, not the whole reproject.
        let swap_ms = t_swap.elapsed().as_millis() as u64;
        record_phase(t_swap, "reproject_swap");
        REPROJ_LAYOUT_US.fetch_add(
            t_swap.elapsed().as_micros() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );

        // Select-promote (warm/cold → hot) the new working set before re-applying,
        // turning the per-unit `ensure_*_hot` calls inside `apply_projection` into
        // hot-hit no-ops — see `elevate_projection_working_set`.
        let t_elevate = Instant::now();
        let elevate_span = profile::span("reproject:elevate");
        self.elevate_projection_working_set(
            &policy.substrate,
            &projected_sections,
            &turn_keys_for_elevate,
            "reproject",
        );
        elevate_span.end();
        let elevate_ms = t_elevate.elapsed().as_millis() as u64;
        self.reproject_census
            .add_since(ReprojectPart::Elevate, t_elevate);
        // Every other conversation whole, or not this time — and a rebuild that
        // withdrew one does not hold the selection its identity names.
        let t_whole = Instant::now();
        let mut projected_segments = projected_segments;
        let identity = (!Self::keep_conversations_whole(
            &policy.substrate,
            Some(policy.target.timeline),
            &mut projected_segments,
            "reproject",
        ))
        .then_some(identity);
        self.reproject_census
            .add_since(ReprojectPart::Whole, t_whole);

        // Time the per-slot apply work (segment build now + finish later),
        // EXCLUDING the shared gap-fill wave — that wait is reported as `glue_ms`.
        let t_build = Instant::now();
        let _build_span = profile::span("reproject:apply_build");
        // Build the gap-fill descriptor + inject the sealed prefix, but DON'T
        // fire the forward — the caller batches it across conversations.
        let plan = or_abandon!(
            self,
            view_id,
            parent_id,
            turn,
            false,
            self.apply_projection_build(parent_id, &projected_segments)
        );
        let build_ms = t_build.elapsed().as_millis() as u64;
        self.reproject_census
            .add_since(ReprojectPart::Build, t_build);
        // The prepare half step by step, where the reproject line has one field
        // per phase: which part of `scan_ms` / `swap_ms` / `apply_ms` the time is.
        note("reproject prepare split", || {
            format!(
                "view={} warm_us={warm_us} score_us={score_us} observe_us={observe_us} \
                 identity_us={identity_us} snapshot_us={snapshot_us} seal_us={seal_us} \
                 free_us={free_us} (move_us={move_us} free_view_us={free_view_us} \
                 release_us={release_us} truncate_us={truncate_us}) build_us={} \
                 probe_tokens={}",
                view_id.0,
                t_build.elapsed().as_micros(),
                probe_tokens,
            )
        });
        // What the slot now holds, for a step's re-layout to start from
        // (`section_relayout`).
        if let Some(state) = self.slot_projection_state.get_mut(&parent_id) {
            state.applied_composition = Some(composition.clone());
        }
        Ok(Some(ReprojectInFlight {
            view_id,
            parent_id,
            tail_per_layer,
            tail_index_page,
            turn,
            sampling_state,
            sections_len: projected_sections.len(),
            segments_len: projected_segments.len(),
            n_turns_selected,
            composition,
            plan,
            build_ms,
            t_repro,
            swap_ms,
            probe_ms,
            scan_ms,
            project_ms,
            elevate_ms,
            question_tokens: old_view_state.map(|v| v.question_tokens).unwrap_or(0),
            identity,
        }))
    }
}
