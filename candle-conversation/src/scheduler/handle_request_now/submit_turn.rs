//! Runs a `SubmitTurn` request on the scheduler loop: projects the turn, lifts its
//! working set onto the device and prefills its opening.
//!
//! This is the one handler that does real work on the loop thread, and the other
//! request handlers are bookkeeping by comparison. A submit over a parent that
//! still has a turn registered first winds that earlier turn down, because its
//! caller has dropped the handle; a live decode or a turn still in prefill cannot
//! be wound down, so the new turn is refused instead, since racing a second decode
//! onto the slot would interleave their K/V. A detached handle frees only the slot
//! id it reserved. Projection seeds from the conversation's carried belief, decayed
//! at the turn boundary so a stale prior does not hold the opening window. The
//! projection's history is priced as an admission before any of it is lifted, so
//! a turn does not take the expert zone's ground unpriced, and a turn that arrives
//! at an idle engine is offered admission at once. Slot free and reset clear every
//! per-occupant record, including any deferred glue plan, because slot ids recycle
//! and a plan built for the old layout would write into the new occupant's cache.

use super::*;

impl Scheduler {
    pub(in super::super) fn handle_request_now(&mut self, req: SchedulerRequest) -> bool {
        tracing::trace!(request = req.span_name(), slots = ?req.slots(), "scheduler request");
        // A detached substrate's handle: its own free releases the slot id it
        // reserved, and anything else it sends is dropped unanswered.
        if let SchedulerRequest::FreeSequence { sequence_id } = &req {
            if self.detached.release(*sequence_id) {
                if let Err(e) = self.session.free_sequence(sequence_id.0) {
                    tracing::warn!("failed to release reserved slot {}: {}", sequence_id, e);
                }
                return true;
            }
        }
        if self.detached.names(&req) {
            tracing::warn!(
                request = req.span_name(),
                "a request from a detached substrate's handle — dropped unanswered"
            );
            return true;
        }
        match req {
            SchedulerRequest::NewSequence {
                conversation,
                target,
                parent,
                response_tx,
            } => {
                let result = self.create_sequence(conversation, target).and_then(|slot| {
                    // A live parent's state beats the snapshot: it is current
                    // rather than as-of-the-last-seal, and the copy is
                    // device-to-device with no host round trip. `create_sequence`
                    // already tried the snapshot read, so this overwrites a
                    // strictly older state when both are available.
                    //
                    // "Older" is only true because `parent` is `Some` solely
                    // when the fork targets the parent's OWN timeline
                    // (`fork_onto` derives it; `fork_inherits_history_tests`
                    // asserts it). For any other timeline the snapshot is the
                    // right state and the parent's is a different
                    // conversation's — overwriting it here is exactly how every
                    // daemon resume once ran on the base conversation's memory.
                    if let Some(parent_id) = parent {
                        self.model
                            .fork_recurrent(parent_id.0, slot.0)
                            .map_err(ConversationError::Model)?;
                    }
                    Ok(slot)
                });
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::NewEphemeralSequence {
                conversation,
                target,
                response_tx,
            } => {
                // Bind the target (so `apply_projection` runs → warm system
                // prompt), then mark ephemeral so the turn never seals.
                let result = self.create_sequence(conversation, Some(target));
                if let Ok(slot) = &result {
                    self.ephemeral_slots.insert(*slot);
                }
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::ForkEphemeralSequence {
                parent,
                band,
                response_tx,
            } => {
                self.request_ephemeral_fork(parent, band, response_tx);
                true
            }

            SchedulerRequest::ResumeSequence {
                conversation,
                timeline,
                response_tx,
            } => {
                // Look up `(layer, group)` from the substrate registry
                // to construct the slot's target, then create the
                // sequence as if it were a fresh `NewSequence` —
                // `slot_targets` ends up the same shape regardless of
                // which entry point produced it.
                let result = match conversation.timeline_target(timeline) {
                    Some((layer, group)) => {
                        let target = ProjectionTarget {
                            layer,
                            group,
                            timeline,
                        };
                        self.create_sequence(conversation, Some(target))
                    }
                    None => Err(ConversationError::Channel(format!(
                        "ResumeSequence: timeline {timeline} not registered in substrate",
                    ))),
                };
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::SubmitTurn {
                seal_group,
                sequence_id,
                projection_inputs,
                prefill_tokens,
                prefill_text,
                user_text,
                tags,
                user_content_start,
                user_content_end,
                assistant_content_start,
                no_think,
                projection_offsets,
                prefill_assistant_text,
                post_decode_tokens,
                max_decode_tokens,
                sampling,
                event_tx,
                reprojection,
                disable_reprojection,
                triggers,
                turn_grammar,
                free_tool_calls_from_penalties,
                recorded_reply,
                recurrent_spans,
                image_spans,
            } => {
                // **The whole SubmitTurn handler, because it is the one that runs real
                // work on the loop thread.** Its siblings are bookkeeping; this one
                // projects the turn, elevates warm KV, carves a view and gap-fills. The
                // drain it sits inside measured 23 ms per call against a decode kernel of
                // a fraction of that, so the question "is that this handler or the
                // channel" needs its own answer.
                let _g = profile::span("drain:submit_turn");
                // The sequence acts as the parent slot for a carved
                // view inside this handler — rebind for clarity.
                let parent_id = sequence_id;

                // ── Abandoned-turn wind-down ─────────────────────────────
                // The sequence-side guard admits one turn at a time, so a
                // submit arriving while this parent still has a turn
                // registered means the caller dropped that turn's handle —
                // an async cancellation. That turn is wound down HERE,
                // before the projection resets the parent under it: its
                // decode is marked finished — the closed channel is what
                // its next step would have discovered anyway — and the
                // ordinary cleanup runs, so its view finalizes and its
                // truncated turn seals first. Without this, the new turn's
                // reset raced the old decode's finalize on one slot.
                //
                // Two states cannot be wound down and are refused instead,
                // with an error on the NEW turn's channel: a turn whose
                // event channel is still open (a live turn — racing a
                // second decode onto the slot would interleave their KV),
                // and a turn still in prefill (there is no finished-path to
                // run for it; the caller retries once it drains).
                let in_flight_views: Vec<SequenceId> = self
                    .turn_views
                    .iter()
                    .filter(|(_, v)| v.parent_id == parent_id)
                    .map(|(&view, _)| view)
                    .collect();
                let mut wind_down_refused = false;
                for view in in_flight_views {
                    match self.active_decodes.get_mut(&view) {
                        Some(state) if state.event_tx.is_disconnected() => {
                            state.finished = true;
                        }
                        // A live decode, or a turn still in prefill.
                        Some(_) | None => wind_down_refused = true,
                    }
                }
                // A submission still waiting for admission on this parent is a turn
                // in flight as well: one whose caller left is dropped, a live one
                // refuses the new turn as a live decode does.
                self.pending_submits
                    .retain(|p| p.parent() != parent_id || !p.work.event_tx.is_disconnected());
                if self.pending_submits.iter().any(|p| p.parent() == parent_id) {
                    wind_down_refused = true;
                }
                self.cleanup_finished();
                if wind_down_refused {
                    let _ = event_tx.send(TurnEvent::Error(ConversationError::TurnInFlight {
                        sequence_id: parent_id,
                    }));
                    return true;
                }
                // An idle slot's state may be parked in its record
                // (`recurrent_park`): staged back before the projection or the
                // view's fork reads it.
                if let Err(e) = self.unpark_from_record(parent_id) {
                    let _ = event_tx.send(TurnEvent::Error(e));
                    return true;
                }

                // Derive the post-Done seal directly from the
                // projection inputs.  When projection is supplied AND
                // the slot has a registered target the request is by
                // definition a turn submission and the result lands in
                // the substrate at `slot_targets[parent_id]`.  When
                // either is absent the request is a raw prefill+decode
                // (RULER, summarisation) and no substrate write
                // happens.  Section ingestion uses
                // [`SchedulerRequest::IngestSection`] instead.
                let slot_target = self.slot_targets.get(&parent_id).copied();
                // Capture this timeline's projection Builder so the summariser's
                // compression probes can read the target layer's `summary`.
                if let (Some(inputs), Some(tgt)) = (&projection_inputs, slot_target.as_ref()) {
                    self.timeline_projections
                        .insert(tgt.timeline, inputs.projection.clone());
                    self.workspace_projection
                        .get_or_insert_with(|| inputs.projection.clone());
                }
                let seal_action = if self.ephemeral_slots.contains(&parent_id) {
                    // Ephemeral probe: project (warm) but NEVER write back — the
                    // query's wide-Q is gathered at completion instead of sealed.
                    SealAction::None
                } else {
                    match (&projection_inputs, slot_target) {
                        (Some(_), Some(_)) => SealAction::Turn,
                        _ => SealAction::None,
                    }
                };
                // Append-only ingests (e.g. code_reading) opt out of the
                // per-turn projection rebuild once their slot is seeded:
                // skip the reset + re-project and prefill straight onto the
                // cumulative slot. The seal is unaffected — turns still land
                // in the substrate.
                //
                // The context is laid down by the conversation's first turn: a
                // submit that carries a frame (`ProjectionInputs::frame`) and
                // finds the slot empty injects exactly the frame — the fixed
                // sections, then its lineage's turns — instead of projecting, as
                // part of its own admission and prefill. From then on the slot
                // holds blocks and `skip_projection` holds. The trade-off is
                // that collection-member / `depends_on`-gated sections (which
                // only materialize via a projection) are NOT added; utility-layer
                // system prompts (code_reading, repo_map) use only fixed
                // sections, so this is exact for them.
                let slot_holds_blocks =
                    self.session.sequence_block_count(parent_id.0).unwrap_or(0) > 0;
                let skip_projection = disable_reprojection && slot_holds_blocks;
                let frame = projection_inputs
                    .as_ref()
                    .and_then(|i| i.frame.as_ref())
                    .filter(|_| frames(disable_reprojection, slot_holds_blocks));
                // Mark this as an append-only ingest timeline so the gentle-early
                // ladder can demote its sealed, warm-backed KV to RAM at ~50%
                // capacity — it is never re-attended until query time. See
                // `demote_cold_ingest_if_pressured`.
                if disable_reprojection {
                    if let Some(tgt) = slot_target {
                        self.ingest_timelines.insert(tgt.timeline);
                    }
                }
                // Step 1: run projection (if requested) and apply it
                // — reset `parent_id` to empty and write the
                // projected sections + projected turns from the
                // substrate onto it.
                //
                // `turn_keys_for_elevate` is the parallel
                // `Vec<TurnKey>` we feed to `elevate_to_hot` before
                // `apply_projection`. Built inside the same closure
                // so the resolved `timeline` for each `(g, t)` pair
                // doesn't get thrown away.
                let mut turn_keys_for_elevate: Vec<TurnKey> = Vec::new();
                // Captured from the projection if it ran the
                // score-density path; written to the substrate's
                // side-channel below, after the read guard drops.
                let mut diag_to_write: Option<(TimelineId, SelectionDiagnostics)> = None;
                // Staged calibration prefill: one projection composition, cloned
                // per segment with its own span, emitted as the wave crosses each
                // `projection_offsets` point. Computed inside the projection block
                // (below) while the view read guard is live; `None` for a normal
                // prefill.
                let mut staged_composition: Option<crate::projection::ProjectionEvent> = None;
                // The turn's opening belief — assigned by the projection path
                // below (carried belief stepped through the submit projection);
                // default when projection is skipped.
                let mut turn_belief = PriorBelief::default();
                // Whether a new user message waits for its first boundary — the
                // turn is a request, read first without the seeds.
                let mut question_pending = false;
                let (projected_sections, projected_segments) = if let (Some(frame), Some(_)) =
                    (frame, slot_target)
                {
                    let (sections, segments, turns) = framed_segments(frame);
                    turn_keys_for_elevate = turns;
                    (sections, segments)
                } else if let (Some(inputs), Some(target)) = (
                    projection_inputs.as_ref().filter(|_| !skip_projection),
                    slot_target,
                ) {
                    let conversation = match self.slot_conversations.get(&parent_id) {
                        Some(c) => c.clone(),
                        None => {
                            let _ = event_tx.send(TurnEvent::Error(ConversationError::Channel(
                                format!(
                                    "submit_turn: no conversation registered for slot {parent_id}"
                                ),
                            )));
                            return true;
                        }
                    };
                    // Target-aware read: the projection sees only
                    // `target.timeline` within `target.group`,
                    // masking sibling timelines.
                    // A request's second reading projects by its first
                    // reading's scores (`first_reading`); every other
                    // submit reads the substrate unscored.
                    let first_reading = self.first_readings.remove(&parent_id);
                    let view = match &first_reading {
                        Some(scores) => conversation.read_for_scored(target, scores),
                        None => conversation.read_for(target),
                    };
                    question_pending = ContentResolver::working_set_boundary(&view)
                        .is_some_and(|b| b.question_pending);
                    // Prefill mode: section scoring uses the calibrated
                    // prefill profile (Max / semantic depth, no threshold)
                    // against the prefill-Q section corpus.  The sink
                    // captures score-density diagnostics for the test
                    // harness (§10.8.4) on a substrate-side side-channel.
                    // Last-write-wins per timeline; the recall test reads
                    // it via `Conversation::last_selection_diagnostics`
                    // after `send_turn` returns.  Sink is never invoked
                    // when the rule-based path runs (no tree on the
                    // target timeline).
                    // Submit-time projection seeds from the conversation's belief
                    // as of its last completed turn (`carried_beliefs`) — empty
                    // only for a genuinely fresh conversation. The belief then
                    // evolves through this turn's reprojections and is harvested
                    // back at seal, so provenance selection is continuous across
                    // turn boundaries instead of resetting to catalog order.
                    //
                    // Decayed at the boundary so it seeds as a SOFT PRIOR, not a
                    // hard pin: the incoming query has no decode-Q evidence yet
                    // (its prefill-Q hits the call↔definition domain gap), so a
                    // full-strength prior would let the previous turn's tool own
                    // the whole opening window and the model could commit to a
                    // wrong framing before the correct tool is selected. Halving
                    // lets the fresh decode-Q overtake a stale tool within a few
                    // tokens; a real continuation re-accumulates just as fast.
                    let mut carried_belief = self
                        .carried_beliefs
                        .get(&parent_id)
                        .cloned()
                        .unwrap_or_default();
                    carried_belief.decay_scores(CARRIED_BELIEF_TURN_DECAY);
                    // Choosing the context: the provenance scan and the section-tree walk
                    // that decide which turns this reply attends over. Pure selection —
                    // no K/V has moved yet.
                    let _g_project = profile::span("drain:project");
                    let projection = inputs.projection.project_with_mode_and_sink(
                        target,
                        &view,
                        ProjectionMode::Prefill,
                        &inputs.selection,
                        &carried_belief,
                        // Opening projection: decode position 0. The early-decode
                        // window excludes position 0 (its prior is the previous
                        // turn's decayed belief), so submit selects on the steady
                        // band; the grace window engages once decode starts.
                        Some(0),
                        &mut |diag| {
                            diag_to_write = Some((target.timeline, diag));
                        },
                    );
                    // Emit the OPENING projection as a timeline event: a POINT at
                    // token 0 (nothing decoded yet) carrying the initial composition
                    // this turn decodes against. A projection governs everything
                    // forward until the next reprojection supersedes it, so even a
                    // short / no-think turn that never reprojects still gets this one
                    // clickable record. `start_token`/`seconds` stay 0 — the turn's
                    // opening point.
                    {
                        let mut opening = crate::projection::from_projection_with_origins(
                            &projection.segments,
                            &projection.selection_origins,
                            inputs.projection.schema(),
                            &view,
                            &projection.selection_scores,
                            view.total_token_count(target.timeline) as u32,
                            0,
                            0.0,
                        );
                        opening.pieces = projection.fill.clone();
                        opening.unpooled = projection.unpooled.clone();
                        opening.placed = projection.files_placed.clone();
                        // Where the knowledge base's files sit, for the
                        // next projection to keep.
                        conversation
                            .update(|s| s.files_placed(target.timeline, projection.files_order()));
                        // The turn's decode state seeds its belief from what this
                        // opening projection actually selected (the carried belief
                        // stepped against this turn's selection), so the first
                        // mid-decode reprojection continues the evolution.
                        turn_belief = PriorBelief::from_selection(&opening.selection);
                        opening.materialized = projection_assembler::materialize_conversation(
                            &projection.segments,
                            &projection.selection_origins,
                            &view,
                            inputs.projection.schema(),
                            |toks| self.tokenizer.decode(toks, false).unwrap_or_default(),
                        );
                        let _ = event_tx.send(TurnEvent::Projection(opening));
                    }
                    // The schema's projection is the single source
                    // of truth for system-side sections — emit
                    // exactly what the projection picked, in
                    // declaration order.  The legacy
                    // "always prepend `system_section_id`" path
                    // (a monolithic ChatML-wrapped duplicate of
                    // the schema fragments) has been removed; the
                    // schema items now compose the full system
                    // prompt on their own.
                    // Single walk of `projection.segments` builds three
                    // outputs:
                    //   - `projected_sections` / `turn_keys_for_elevate`
                    //     feed `elevate_to_hot` (per-id form expected
                    //     by the persistence API).
                    //   - `segments` carries the same items in
                    //     declaration order for `apply_projection`.
                    // Each turn already carries its resolved timeline
                    // (stamped at projection), so no group→timeline
                    // resolution happens here.
                    let mut sections: Vec<SectionId> = Vec::new();
                    let mut segments: Vec<ProjectionSegment> = Vec::new();
                    for seg in &projection.segments {
                        match seg {
                            ProjectionSegment::Sealed(SealedKind::Section(rs)) => {
                                // Any-tier filter — cold-marker
                                // sections (post-reload, before
                                // elevate) must survive this filter
                                // so the `elevate_to_hot` call below
                                // can lift them.  Filtering on
                                // `section_sealed_of` (hot-only)
                                // would silently drop every
                                // not-yet-elevated section the
                                // projection selected.
                                if view.section_tier_state(rs.id).is_some() {
                                    sections.push(rs.id);
                                    segments.push(seg.clone());
                                }
                            }
                            ProjectionSegment::Sealed(SealedKind::Turn(rt, _)) => {
                                // The turn carries its conversation (stamped at
                                // projection); read it directly — no group→timeline
                                // re-derivation.  `elevate_to_hot` below brings
                                // cold-marker turns into hot before apply runs; a
                                // timeline-less (mock/untracked) turn is skipped.
                                let Some(timeline) = rt.timeline else {
                                    continue;
                                };
                                turn_keys_for_elevate.push(TurnKey::new(timeline, rt.index()));
                                segments.push(seg.clone());
                            }
                            ProjectionSegment::Generated { .. }
                            | ProjectionSegment::NewUserMessage { .. } => {
                                // Live-prefill runs and new-user-message
                                // captures don't need an elevate side-list
                                // entry — their K/V is produced on the
                                // slot, not loaded from the substrate.
                                segments.push(seg.clone());
                            }
                        }
                    }

                    // **No trailing `Generated(UserStart)`.**
                    //
                    // The live turn's opener is now the head of its own prefill
                    // (`Sequence::submit_prefill_unit`), so it arrives as
                    // ordinary tail prefill rather than as a glue island the
                    // engine has to gap-fill. The distinction is the whole
                    // difference for this lineage: appending at the writer tail
                    // is prefill, which any model can do; filling a hole in the
                    // middle of a sequence is gap-fill, which a recurrence
                    // cannot do even in principle.
                    //
                    // It is invisible in the token stream — the same ids in the
                    // same order — and visible only in whether a gap chunk was
                    // reserved, which is what `T5a.1` asserts on.
                    // Build the composition for a staged prefill's per-segment
                    // projection events now, while the view guard is live. The
                    // calibration projection is pinned (`SelectionRule::Named`), so
                    // one composition serves every segment — the wave overrides its
                    // point `start_token` per emission.
                    if !projection_offsets.is_empty() {
                        let total = view.total_token_count(target.timeline) as u32;
                        staged_composition = Some(crate::projection::from_projection(
                            &projection.segments,
                            inputs.projection.schema(),
                            &view,
                            &projection.selection_scores,
                            total,
                            0,
                            0.0,
                        ));
                    }
                    // **No live `/no_think` run either.**
                    //
                    // The soft-switch is baked into the head of the turn that
                    // carries it, decided once at submit from the dial that was
                    // live then. That is *stronger* than re-deciding it every
                    // projection, not weaker: the leak this used to guard
                    // against — a past suppressed turn putting a stale switch on
                    // a later thinking-on turn — cannot happen when each turn's
                    // grid holds its own.
                    (sections, segments)
                } else {
                    (Vec::new(), Vec::new())
                };

                // **What the projection will put on the device is priced here, and
                // put there only when admission takes the turn** (`submit_admit`).
                // The lift of the selected history and its index pages is ground
                // the weight side loses exactly as the turn's own K/V is; spent at
                // submit, it was spent before admission could weigh it, and a
                // burst of reads took the experts' ground with no decision made.
                let (lift, depth) = match self.slot_conversations.get(&parent_id).cloned() {
                    Some(conversation) => (
                        self.projection_lift(
                            &conversation,
                            &projected_sections,
                            &turn_keys_for_elevate,
                        ),
                        submit_admit::projected_depth(&conversation, &projected_segments),
                    ),
                    None => (0, 0),
                };
                // **An arrival is an admission opportunity too.** The pass is
                // otherwise gated on completions — nothing new can fit that did
                // not fit before unless something freed ground — but that is
                // only true of work already queued. A turn arriving at an idle
                // engine has never been offered at all, and waiting for a
                // completion that will never come (nothing is running) leaves it
                // queued indefinitely.
                self.settled_since_admit = true;
                let rereads = first_reading::rereads(first_reading::Request {
                    question_pending,
                    seals_a_turn: matches!(seal_action, SealAction::Turn),
                    grouped: seal_group.is_some(),
                    reprojects: reprojection.is_some(),
                    staged: !projection_offsets.is_empty() || !recurrent_spans.is_empty(),
                    replayed: recorded_reply.is_some(),
                    tokens: prefill_tokens.len(),
                });
                self.pending_submits.push_back(submit_admit::PendingSubmit {
                    work: PrefillWork {
                        sequence_id: parent_id,
                        tokens: prefill_tokens,
                        prefill_text,
                        user_text,
                        tags,
                        user_content_start,
                        user_content_end,
                        assistant_content_start,
                        no_think,
                        prefill_assistant_text,
                        event_tx,
                        max_decode_tokens,
                        sampling,
                        submitted_at: Instant::now(),
                        reprojection,
                        belief: turn_belief,
                        seal_action,
                        post_decode_tokens,
                        projection_offsets,
                        staged_composition,
                        triggers,
                        turn_grammar,
                        free_tool_calls_from_penalties,
                        recorded_reply,
                        recurrent_spans,
                        image_spans,
                        rereads,
                    },
                    lift,
                    depth,
                    diag_to_write,
                    projected_sections,
                    projected_segments,
                    turn_keys_for_elevate,
                    slot_target,
                    skip_projection,
                    seal_group,
                });
                true
            }

            SchedulerRequest::FreeSequence { sequence_id } => {
                tracing::debug!(target: "sched", "free_sequence {}", sequence_id);
                // A submission still waiting on this slot cannot be projected onto it
                // any more: its caller hears so, rather than waiting on nothing.
                let (gone, kept): (Vec<_>, Vec<_>) = std::mem::take(&mut self.pending_submits)
                    .into_iter()
                    .partition(|p| p.parent() == sequence_id);
                self.pending_submits = kept.into_iter().collect();
                for p in gone {
                    let _ = p
                        .work
                        .event_tx
                        .send(TurnEvent::Error(ConversationError::Channel(format!(
                            "slot {sequence_id} was freed before its turn was admitted"
                        ))));
                }
                self.request_laps.lap("pending_submits");
                // A turn view borrows this slot's blocks, so it cannot outlive
                // them; and a view freed directly takes its registration with it.
                self.discard_turn_views_of(sequence_id);
                self.turn_views.remove(&sequence_id);
                self.request_laps.lap("turn_views");
                if let Err(e) = self.session.free_sequence(sequence_id.0) {
                    tracing::warn!("failed to free sequence {}: {}", sequence_id, e);
                }
                self.request_laps.lap("session_free");
                // The session owns the KV; per-sequence state the MODEL keeps
                // outside it (a recurrent store) is freed here or not at all.
                // Slot ids recycle, so this is correctness, not housekeeping.
                if let Err(e) = self.model.release_sequence(sequence_id.0) {
                    tracing::warn!(
                        "failed to release model state for sequence {}: {}",
                        sequence_id,
                        e
                    );
                }
                self.request_laps.lap("model_release");
                // Clean up persistent sampling state.
                self.sampling_states.remove(&sequence_id);
                // Drop the conversation handle and projection target
                // bound to this slot.
                let freed_conversation = self.slot_conversations.remove(&sequence_id);
                let freed_target = self.slot_targets.remove(&sequence_id);
                // Its conversation may now be unloaded, and its own gallery
                // turns readable by no scan.
                self.gallery_sweep.slot_freed();
                self.ephemeral_slots.remove(&sequence_id);
                self.slot_bands.remove(&sequence_id);
                self.ephemeral_sigs.remove(&sequence_id);
                self.describe_at_seal.remove(&sequence_id);
                self.carried_beliefs.remove(&sequence_id);
                self.slot_tokens.remove(&sequence_id);
                self.request_laps.lap("slot_maps");
                self.retire_slot_projection_state(sequence_id, freed_conversation);
                self.request_laps.lap("projection_state");
                // Purge any DEFERRED glue plan for this slot. A queued gap-fill
                // must never outlive the slot layout it was planned against: the
                // freed id is recycled immediately (the code_read scope workers
                // churn fork slots), and `take_wave_glue` would fire the dead
                // tenant's plan into the NEW tenant's slot — its slot-relative
                // gap indices scatter stale glue K/V into whatever the new
                // occupant placed there (cross-scope KV contamination), or, when
                // the old gap chunks are gone, write out of bounds (the
                // deterministic decode-onset CUDA_ERROR_ILLEGAL_ADDRESS at the
                // first scope completions). The reproject path already purges
                // superseded plans (see `batch_drain_gap_fills`); the free path
                // must too.
                self.deferred_glue_fires
                    .retain(|p| p.parent_id != sequence_id);
                // Purge in-flight prefill/decode/cohort state before the id is
                // recycled — a stale prefill entry keyed by the freed id would
                // collide with the reused slot in the next wave-group assembly.
                self.purge_freed_slot_scheduling_state(sequence_id);
                self.request_laps.lap("scheduling_state");
                self.prune_ingest_timeline(freed_target);
                self.request_laps.lap("ingest_timeline");
                true
            }

            SchedulerRequest::ResetSequence {
                sequence_id,
                response_tx,
            } => {
                // A turn view borrows the prefix this reset discards.
                self.discard_turn_views_of(sequence_id);
                let result = self
                    .session
                    .reset_sequence(sequence_id.0)
                    .map_err(ConversationError::Model);
                if result.is_ok() {
                    // Reset sampling state counters while preserving the shape.
                    // A reset slot is reused for new content, so everything the
                    // previous occupant said goes — its cross-turn history
                    // included, which `end_turn` would have carried over and
                    // penalised in the next occupant's first turn. The context
                    // tokens fed in the next prefill re-seed DRY.
                    if let Some(state) = self.sampling_states.get_mut(&sequence_id) {
                        state.clear();
                    }
                    // A reset slot is reused for NEW content (the titler resets
                    // between title jobs): the previous occupant's belief must
                    // not seed the next occupant's projections.
                    self.carried_beliefs.remove(&sequence_id);
                    // And the previous occupant's deferred glue plan must not
                    // fire into the new content's layout (see FreeSequence).
                    self.deferred_glue_fires
                        .retain(|p| p.parent_id != sequence_id);
                }
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::SetSequenceAdapter {
                sequence_id,
                adapter,
                response_tx,
            } => {
                // **Only on an empty slot.** The KV already in this sequence was
                // written by whichever projections were in force at the time,
                // and the adapter changes them — so switching mid-conversation
                // would attend adapted keys with unadapted queries, or the
                // reverse. Nothing about the shapes objects; the answers just
                // quietly degrade, which is why this is refused rather than
                // documented.
                let offset = self.session.sequence_offset(sequence_id.0);
                let result = if offset.unwrap_or(0) > 0 {
                    Err(ConversationError::Model(candle::Error::Msg(format!(
                        "sequence {} already holds {} tokens of KV — a LoRA must be \
                         chosen before a conversation's first turn, not after",
                        sequence_id.0,
                        offset.unwrap_or(0)
                    ))))
                } else {
                    self.session
                        .set_sequence_adapter(sequence_id.0, adapter)
                        .map_err(ConversationError::Model)
                };
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::ArenaScan {
                scan,
                response_tx,
                background,
            } => {
                if background {
                    self.queue_held_scan(norm_warm::HeldJob { scan, response_tx });
                    return true;
                }
                let reply = match self.gallery_arena.as_deref() {
                    Some(arena) => scan.run(arena),
                    None => scan.no_arena(),
                };
                let _ = response_tx.send(reply);
                true
            }

            SchedulerRequest::ArenaGroups { response_tx } => {
                let _ = response_tx.send(self.gallery_arena.as_deref().map(GalleryArena::n_groups));
                true
            }

            SchedulerRequest::IngestSection {
                sequence_id,
                section_id,
                prefix_section_ids,
                tokens,
                address,
                debug_name,
                in_collection,
                start,
                band,
                response_tx,
            } => {
                let section = QueuedSection {
                    sequence_id,
                    section_id,
                    prefix_section_ids,
                    tokens,
                    address,
                    debug_name,
                    in_collection,
                    start,
                    band,
                    response_tx,
                    queued_at: Instant::now(),
                };
                if section.tokens.is_empty() {
                    // No forward pass, so nothing to admit or claim: seal now.
                    if let Err(refused) = self.begin_section(section) {
                        let (s, why) = *refused;
                        let _ = s.response_tx.send(Err(why));
                    }
                } else {
                    // A first admission like a prefill's — see `QueuedSection`.
                    self.section_queue.push_back(section);
                    self.settled_since_admit = true;
                }
                true
            }

            SchedulerRequest::RestoreSection {
                conversation,
                section_id,
                stream_id,
                tokens,
                response_tx,
            } => {
                let result = self.restore_section_from_persistence(
                    &conversation,
                    section_id,
                    stream_id,
                    tokens,
                );
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::SetBand { sequence_id, band } => {
                let recorded = || {
                    let conversation = self.slot_conversations.get(&sequence_id)?;
                    let target = self.slot_targets.get(&sequence_id)?;
                    recorded_band(conversation, target.timeline)
                };
                match band.or_else(recorded) {
                    Some(band) => self.slot_bands.insert(sequence_id, band),
                    None => self.slot_bands.remove(&sequence_id),
                };
                true
            }

            SchedulerRequest::RetireSections {
                conversation,
                sections,
            } => {
                // A transient section leaves the substrate here; an owned
                // section already left it when its owner released it, and only
                // the scheduler's own records of it remain. Each is this
                // substrate's alone to let go: section ids are a schema's own,
                // and a page is kept while another substrate holds the section.
                for id in sections {
                    self.withdraw_queued_section(&conversation, id);
                    let stream = conversation.read().section_stream(id);
                    conversation.retire_section(id);
                    if let Some(stream) = stream {
                        self.section_positional.release(stream, &conversation);
                        self.section_name_cache.remove(&stream);
                    }
                    self.pending_section_quantize
                        .retain(|p| p.section_id != id || !p.substrate.is(&conversation));
                }
                true
            }

            SchedulerRequest::OffloadCollectionMembers {
                conversation,
                response_tx,
            } => {
                let result = (|| -> Result<(), ConversationError> {
                    if let Some(turn_policy) = self.session.compression_policy() {
                        let boundary_policy = Self::section_compression_policy_boundary();
                        let member_policy = Self::section_compression_policy_member(&turn_policy);
                        // Quantize just the prefix-transparent members + flag
                        // them for offload; boundary sections stay pending for
                        // the turn-seal boundary.
                        self.quantize_pending_collection_members(
                            &conversation,
                            &boundary_policy,
                            &member_policy,
                        )?;
                    }
                    // Block on a full persistence pass so the members' cold
                    // copies land and `install_cold` frees their VRAM before the
                    // next batch prefills.  The wave-batched prefill within a
                    // batch keeps its own concurrency; this seam only gates
                    // between batches so the native catalog can't outrun the
                    // offload under VRAM pressure.
                    timed_wait(|| {
                        self.persist_triggers
                            .flush_blocking(std::time::Duration::from_secs(30))
                    });
                    Ok(())
                })();
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::ExtractRawKvq {
                sequence_id,
                layer_indices,
                block_range,
                response_tx,
            } => {
                let result =
                    self.handle_extract_raw_kvq(sequence_id.0, &layer_indices, block_range);
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::BranchCheckpointPass {
                start,
                tokens,
                response_tx,
            } => {
                let result = self.handle_branch_checkpoint_pass(start.as_ref(), &tokens);
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::MemoryCatchUp {
                sequence_id,
                tokens,
                adopted_from,
                image_spans,
                response_tx,
            } => {
                let result =
                    self.handle_memory_catch_up(sequence_id, &tokens, adopted_from, &image_spans);
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::SeedRecurrentFromLineage {
                sequence_id,
                response_tx,
            } => {
                let seeded = match (
                    self.slot_conversations.get(&sequence_id).cloned(),
                    self.slot_targets.get(&sequence_id).copied(),
                ) {
                    (Some(conversation), Some(target)) => {
                        self.restore_recurrent_state(sequence_id, &conversation, target.timeline)
                    }
                    // A scratch slot, or one already gone. Neither is an error:
                    // there is no conversation whose memory this would be.
                    _ => false,
                };
                let _ = response_tx.send(Ok(seeded));
                true
            }

            SchedulerRequest::InstallRecurrentState {
                sequence_ids,
                state,
                response_tx,
            } => {
                // Every slot gets the same state, so the scatter is per-slot but
                // the queue wait and the host payload are shared.
                //
                // EVERY slot is attempted, even after a failure. Stopping at the
                // first error does not prevent a partial install — the slots
                // already done stay done — it only adds arbitrarily-skipped slots
                // on top, which is strictly worse: those conversations start from
                // zero with no error of their own. There is no rollback here, so
                // best-effort plus a reported error is the honest contract. The
                // first error is kept and returned.
                //
                // **Staged, not restored.** These slots have not run a turn and
                // may not for a while; restored now, each held a whole recurrent
                // store outside admission, and a window of new conversations took
                // the span from under turns admission had already priced. The
                // model holds the shared rows and builds a store when the slot's
                // state is first wanted — its view forked at admission, or a wave
                // on the slot itself.
                let mut installed = false;
                let mut first_err = None;
                for sequence_id in sequence_ids {
                    match self.stage_state(sequence_id.0, &state) {
                        Ok(did) => installed |= did,
                        Err(e) => {
                            if first_err.is_none() {
                                first_err = Some(e);
                            }
                        }
                    }
                }
                let result = match first_err {
                    Some(e) => Err(e),
                    None => Ok(installed),
                };
                let _ = response_tx.send(result);
                true
            }

            SchedulerRequest::DescribeAtSeal {
                sequence_id,
                response_tx,
            } => {
                self.describe_at_seal.insert(sequence_id, response_tx);
                true
            }

            SchedulerRequest::Transplant {
                sequence_id,
                capture,
                prompt_tokens,
                kv,
                index,
                response_tx,
            } => {
                let _ = response_tx.send(self.transplant(
                    sequence_id,
                    capture,
                    prompt_tokens,
                    kv,
                    index,
                ));
                true
            }

            SchedulerRequest::ReleaseCapture { capture } => {
                self.captures.remove(&capture);
                true
            }

            SchedulerRequest::SealNatively {
                sequence_id,
                response_tx,
            } => {
                self.seal_section_natively(sequence_id);
                let _ = response_tx.send(());
                true
            }

            SchedulerRequest::QuantizeSpans {
                sequence_id,
                spans,
                response_tx,
            } => {
                let _ = response_tx.send(self.quantize_spans(sequence_id, &spans));
                true
            }

            SchedulerRequest::ReadRecurrentMemory {
                sequence_id,
                response_tx,
            } => {
                let result = self
                    .model
                    .export_recurrent(sequence_id.0)
                    .map(|opt| opt.map(|(_hash, layers)| layers))
                    .map_err(ConversationError::Model);
                let _ = response_tx.send(result);
                true
            }

            #[cfg(any(test, feature = "test-helpers"))]
            SchedulerRequest::ReadTurnKvDigest {
                sequence_id,
                timeline,
                index,
                response_tx,
            } => {
                let result = self.handle_read_turn_kv_digest(sequence_id, timeline, index);
                let _ = response_tx.send(result);
                true
            }

            #[cfg(any(test, feature = "test-helpers"))]
            SchedulerRequest::CountRecurrentMemories { response_tx } => {
                let _ = response_tx.send(self.model.recurrent_memory_count());
                true
            }

            SchedulerRequest::ProbeWideSigs {
                sequence_id,
                response_tx,
            } => {
                // Hand back the warm query wide-Q the ephemeral probe turn stashed
                // at its completion; drain it so a re-probe of a recycled slot id
                // can't read a stale window.
                let sigs = self.ephemeral_sigs.remove(&sequence_id).unwrap_or_default();
                let _ = response_tx.send(Ok(sigs));
                true
            }

            SchedulerRequest::SubmitSummaryProbe {
                timeline,
                kind,
                children,
                height,
                response_tx,
            } => {
                // Dispatches the node to its compression: a structural (repo-map)
                // node seals synchronously; a model-decode node enqueues ONE
                // decode-only pass that rides `batch_decode_step` concurrently
                // with foreground turns and is recorded by
                // `complete_compression_pass` when it lands. Either way the
                // handler returns immediately; a setup failure replies with `Err`
                // here for a soft retry on the summariser's next pass.
                if let Err(e) =
                    self.handle_summary_probe(timeline, kind, children, height, response_tx.clone())
                {
                    // "no projection/summary" means this layer has no
                    // summary-of-summaries projection to compress into — a permanent
                    // structural condition, not a transient setup failure. Classify
                    // it Permanent so the summariser disarms this timeline's reconcile
                    // instead of re-submitting a doomed probe every pass forever
                    // (which starves the summariser and floods the log).
                    let pe = if e.contains("no projection/summary") {
                        ProbeError::Permanent(e)
                    } else {
                        ProbeError::Soft(e)
                    };
                    let _ = response_tx.send(Err(pe));
                }
                true
            }

            SchedulerRequest::ReconstructSubstrate {
                conversation,
                status,
            } => {
                self.reconstruct_substrate(&conversation, &status);
                true
            }

            SchedulerRequest::AttachSubstrate {
                id,
                trigger,
                conversation,
                status,
            } => {
                self.persist_triggers.attach(id, trigger);
                self.register_substrate(&conversation);
                self.detached.reattached(&conversation);
                self.reconstruct_substrate(&conversation, &status);
                true
            }

            SchedulerRequest::DetachSubstrate {
                id,
                conversation,
                response_tx,
            } => {
                self.detach_substrate(id, &conversation);
                let _ = response_tx.send(());
                true
            }

            SchedulerRequest::WarmCollectionNormalization {
                conversation,
                schema,
                response_tx,
            } => {
                let warm = conversation
                    .warm_collection_normalization(&schema, self.gallery_arena.as_deref());
                let _ = response_tx.send(warm);
                true
            }

            SchedulerRequest::WarmDialogueNormalization {
                conversation,
                projection,
                target,
                max_probe_tokens,
            } => {
                // Background work: queued here, scored from the loop top
                // (`step_norm_warm`) one turn at a time.
                self.queue_norm_warm(&conversation, &projection, target, max_probe_tokens);
                true
            }

            SchedulerRequest::PreloadGalleries {
                conversation,
                schema,
                response_tx,
            } => {
                let preload =
                    conversation.preload_galleries(&schema, self.gallery_arena.as_deref());
                let _ = response_tx.send(preload);
                true
            }

            SchedulerRequest::WarmIngestSlice {
                conversation,
                schema,
                layer,
                group,
                timelines,
                response_tx,
            } => {
                // Background work: scored at ingest priority from the loop top
                // (`step_norm_warm`), never inside the drain that received it.
                self.queue_ingest_warm_slice(norm_warm::IngestSliceJob {
                    conversation,
                    schema,
                    layer,
                    group,
                    timelines,
                    response_tx,
                });
                true
            }

            SchedulerRequest::ScanGroupVotes {
                conversation,
                group,
                calibrations,
                masked,
                excluded,
                probes,
                groups,
                keep,
                response_tx,
            } => {
                // Background work, like an ingest slice: run from the loop top
                // (`step_norm_warm`), never inside the drain that received it.
                self.queue_group_scan(norm_warm::ScanVotesJob {
                    conversation,
                    group,
                    calibrations,
                    masked,
                    excluded,
                    probes,
                    groups,
                    keep,
                    response_tx,
                });
                true
            }

            SchedulerRequest::DemoteTimelinesHot {
                conversation,
                timelines,
                response_tx,
            } => {
                // Drop the hot copy of each timeline's turns that already hold a
                // warm copy (a turn without one is left hot — see
                // `demote_turns_to_warm`). Any needed hot→warm flush was run
                // caller-side before this request, so this thread never blocks on
                // the persistence pass.
                let demoted = {
                    let view = conversation.read();
                    let keys: Vec<TurnKey> = timelines
                        .iter()
                        .flat_map(|&tl| {
                            view.turn_indices(tl)
                                .map(move |idx| TurnKey::new(tl, idx))
                                .collect::<Vec<_>>()
                        })
                        .collect();
                    view.demote_turns_to_warm(&keys)
                };
                // The demote returned the hot chunks to the pool free-list;
                // release now-empty arenas so `pool_used` actually drops.
                let _ = self.session.release_empty_arenas();
                let _ = response_tx.send(Ok(demoted));
                true
            }

            // Nothing to do. Returning to the top of the loop *is* the work —
            // that is where the guest queue is polled.
            SchedulerRequest::Wake => true,

            // Answered by the idle branch's settle, never from here: this drain
            // runs between forwards of live decodes, and a settle is unbudgeted.
            SchedulerRequest::Settle { response_tx } => {
                self.settle_waiters.push(response_tx);
                self.settle_owed = true;
                true
            }

            SchedulerRequest::Shutdown => false,
        }
    }
}
