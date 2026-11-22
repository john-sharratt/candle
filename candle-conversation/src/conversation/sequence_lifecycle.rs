//! Runs a turn from submission to seal, and clears a turn's in-flight guard
//! however it ends.
//!
//! Only `finish_turn` and `abort_turn` clear the in-flight guard, so a dropped
//! async send clears it through `CancelClearsInFlight`, and a failed turn is
//! aborted in `settle_turn`. Without those, one failed decode rejected every later
//! turn. A cancelled turn seals what it decoded as a truncated turn, the same
//! record a crash leaves. Finishing a turn applies the seal payload and commits
//! the exchange; no header is prefilled between turns, since each turn carries its
//! own opener and closer.

use super::*;

/// Keeps a drop-cancelled [`Sequence::send_turn_with_options_async`] from
/// wedging its sequence.
///
/// Submitting a turn latches `turn_in_flight`, and only `finish_turn` /
/// `abort_turn` unlatch it — neither of which runs when the async send's
/// future is dropped mid-await (a task abort, a timeout). The blocking send
/// can never be abandoned there, so this state was unreachable before the
/// async path existed. Armed across the await, this clears the caller-side
/// turn state on drop; disarmed on the settled path, where `finish_turn` /
/// `abort_turn` own the cleanup.
///
/// The scheduler needs no message and gets none: the dropped [`TurnHandle`]
/// closes the event channel, and the scheduler winds the turn down — at its
/// next decode step, or at this sequence's next `SubmitTurn`, whose handler
/// finalizes an abandoned turn before resetting the parent under it (a
/// submit that catches the old turn still live or still prefilling is
/// refused with `TurnInFlight` and retried, never raced). Deliberately NOT
/// [`Sequence::abort_turn`], which resets the slot's whole KV through a
/// blocking scheduler round trip — a decode's worth of state and a parked
/// executor thread are the wrong price for a cancellation.
///
/// What a cancelled turn leaves behind, stated rather than hidden: the
/// decoded-so-far seals into the timeline as a truncated turn (the same
/// record a crash between decode and finish leaves), the client-side tree
/// does not carry the cancelled exchange (the substrate is what projections
/// compose from), and `current_blocks` stays at its pre-turn value until the
/// next seal's absolute count self-heals it.
pub(super) struct CancelClearsInFlight<'a> {
    pub(super) seq: Option<&'a mut Sequence>,
}

impl<'a> CancelClearsInFlight<'a> {
    pub(super) fn arm(seq: &'a mut Sequence) -> Self {
        Self { seq: Some(seq) }
    }

    /// The settled path: hand the sequence back and clear nothing.
    pub(super) fn disarm(mut self) -> &'a mut Sequence {
        self.seq
            .take()
            .expect("disarm is called once, on the armed guard")
    }
}

impl Drop for CancelClearsInFlight<'_> {
    fn drop(&mut self) {
        if let Some(seq) = self.seq.take() {
            seq.turn_in_flight = false;
            seq.pending_user = None;
        }
    }
}

impl Sequence {
    /// Blocking convenience: submit + wait.
    ///
    /// Automatically records the assistant response, prefills the next
    /// user header, and clears in-flight.
    pub fn send_turn(&mut self, user_message: &str) -> crate::Result<TurnResponse> {
        self.send_turn_with_options(user_message, TurnOptions::default())
    }

    /// Blocking convenience with per-turn options: submit + wait.
    ///
    /// The [`Self::send_turn`] path for callers that arm a stencil — pass a
    /// tool-call [`TriggerRegistry`](crate::stencil::TriggerRegistry) in
    /// `options.triggers` and any `<tool_call>` the model starts is forced to
    /// the catalog's exact shape. Without one the turn free-decodes, which for a
    /// caller that then *parses* the output means hoping for a shape rather than
    /// requiring it.
    pub fn send_turn_with_options(
        &mut self,
        user_message: &str,
        options: TurnOptions,
    ) -> crate::Result<TurnResponse> {
        let handle = self.submit_turn_with_options(user_message, options)?;
        let waited = handle.wait();
        self.settle_turn(handle, waited)
    }

    /// Async convenience: submit + await. [`Self::send_turn`]'s counterpart —
    /// the same submit, the same failure handling, the same `finish_turn`;
    /// only the wait yields instead of holding a thread.
    pub async fn send_turn_async(&mut self, user_message: &str) -> crate::Result<TurnResponse> {
        self.send_turn_with_options_async(user_message, TurnOptions::default())
            .await
    }

    /// Async counterpart of [`Self::send_turn_with_options`].
    ///
    /// The returned future owns this turn's [`TurnHandle`], so dropping the
    /// future — an aborted task, a caller that gave up mid-await — drops the
    /// handle, the scheduler sees the closed event channel, and the decode
    /// stops at its next step. The drop also clears this sequence's in-flight
    /// guard ([`CancelClearsInFlight`]), so the conversation takes its next
    /// turn normally; the cancelled turn's decoded-so-far winds down and seals
    /// scheduler-side, exactly as a dropped blocking handle's does.
    /// Cancellation is the drop; nothing extra to call.
    ///
    /// **What yields is the wait; the ends stay synchronous.** The submit
    /// composes the prompt and enqueues (fast, but a first turn's branch
    /// checkpoint does scheduler round trips), and the settle runs
    /// `finish_turn` — or, on a failed turn, `abort_turn`, whose slot reset
    /// waits on the scheduler's next drain. Callers multiplexing many
    /// sequences on one executor thread should treat a turn's ends as brief
    /// host work, not as await points.
    pub async fn send_turn_with_options_async(
        &mut self,
        user_message: &str,
        options: TurnOptions,
    ) -> crate::Result<TurnResponse> {
        let handle = self.submit_turn_with_options(user_message, options)?;
        let armed = CancelClearsInFlight::arm(self);
        let waited = handle.wait_async().await;
        armed.disarm().settle_turn(handle, waited)
    }

    /// Resolve a submitted turn from its wait's result — one function behind
    /// both the blocking and async sends, so their failure handling cannot
    /// diverge.
    ///
    /// **A turn that fails never reaches `finish_turn`**, which is the only
    /// other thing that clears the in-flight guard — so without the abort here
    /// one failed decode rejected every turn the conversation was ever asked
    /// for afterwards, with `TurnInFlight`, and nothing short of dropping the
    /// sequence brought it back. That is what a character looks like when a
    /// single wave faults under it: alive, scheduled, and never acting again.
    pub(super) fn settle_turn(
        &mut self,
        handle: TurnHandle,
        waited: crate::Result<TurnResponse>,
    ) -> crate::Result<TurnResponse> {
        let response = match waited {
            Ok(response) => response,
            Err(e) => {
                drop(handle);
                self.abort_turn();
                return Err(e);
            }
        };
        // Record the assistant turn and prefill next user header.
        self.finish_turn(handle, &response)?;
        Ok(response)
    }

    /// Record the assistant response after streaming, prefill the next
    /// user turn header (`<|im_end|>\n<|im_start|>user\n`) into the KV
    /// cache, and clear the in-flight guard.
    ///
    /// Takes the handle returned by [`submit_turn`](Self::submit_turn) so it
    /// can finalize any windowed-context view sequence that was created for
    /// this turn.
    ///
    /// Does **not** return a handle — the next-user-header prefill fires
    /// fire-and-forget; FIFO ordering in the scheduler guarantees it completes
    /// before the next `submit_turn` starts.
    ///
    /// Returns the exact boundary text that was prefilled into the KV cache
    /// (e.g. `"<|im_end|>\n<|im_start|>user\n"`).  Callers that need to
    /// display the full token stream can print this string immediately after
    /// `finish_turn` returns — at which point the seal has completed and the
    /// tokens are guaranteed to be in the cache.
    pub fn finish_turn(
        &mut self,
        handle: TurnHandle,
        response: &TurnResponse,
    ) -> crate::Result<String> {
        // Take the pending user turn recorded at submit time.
        let user_tt = self.pending_user.take().unwrap_or_default();
        let assistant_tt = TokenizedText::new(response.text.as_str(), response.token_ids.clone());

        self.turn_in_flight = false;

        // The view sequence is owned by the scheduler and was already
        // auto-finalized in `cleanup_finished` before this `Done` event was
        // sent — so the parent now holds all of the turn's KV blocks.  We
        // drop the handle here purely to release the event channel.
        drop(handle);

        self.finalize_turn_post_done(user_tt, assistant_tt, response.seal.as_ref())
    }

    /// [`finish_turn`](Self::finish_turn) + staged provenance linkage for a
    /// DECODED ingest turn (the code_read per-file summary): synthesizes and
    /// persists the same two staged [`ProjectionEvent`]s as
    /// [`insert_turn_staged`](Self::insert_turn_staged), with the decode's
    /// wall-clock on the assistant-half event. The assistant half's grid
    /// start comes from the sealed turn's own [`TurnLayout`] rather than
    /// re-tokenizing.
    pub fn finish_turn_staged(
        &mut self,
        handle: TurnHandle,
        response: &TurnResponse,
    ) -> crate::Result<String> {
        let turn_index = response.seal.as_ref().and_then(|s| s.turn_index);
        let seconds = response.stats.decode_ms / 1000.0;
        let text = self.finish_turn(handle, response)?;
        if let Some(idx) = turn_index {
            let assistant_start = {
                let read = self.substrate.read();
                read.turn_layout(self.target.timeline, TurnIndex(idx))
                    .map(|l| l.assistant_content_start())
                    .unwrap_or(0)
            };
            self.persist_staged_ingest_events(idx, assistant_start, seconds)?;
        }
        Ok(text)
    }

    /// Synthesize + persist the two staged [`ProjectionEvent`]s for the
    /// just-sealed ingest turn `turn_index`: one governing the user half
    /// (`start_token: 0`), one governing the assistant half. `selection`
    /// carries the layer's fixed system sections and — the mandatory
    /// provenance linkage — the turn itself plus its immediate predecessor as
    /// `(timeline, index)`-resolvable [`SelectedTurn`]s, keyed to the turn's
    /// stream id (the same key its wide-Q signature persists under).
    pub(super) fn persist_staged_ingest_events(
        &self,
        turn_index: u32,
        assistant_content_start: u32,
        seconds: f64,
    ) -> crate::Result<()> {
        use crate::persistence::streams::StreamDecl;
        use crate::projection::event::group_name_of;
        use crate::projection::{encode_events, staged_ingest_event, SelectedTurn, SystemItem};
        use crate::substrate::ContentResolver;
        use crate::summary_tree::TurnKind;

        let timeline = self.target.timeline;
        let schema = self.projection.schema();
        let layer = schema.layers.iter().find(|l| l.id == self.target.layer);
        let layer_name = layer.map(|l| l.name.clone()).unwrap_or_default();
        let group_name = group_name_of(schema, self.target.group)
            .unwrap_or_default()
            .to_string();

        let (system, turns) = {
            let read = self.substrate.read();
            // The trunk's fixed system sections — utility-layer system
            // prompts are fixed-only, so this is the complete composition.
            let mut system: Vec<SystemItem> = Vec::new();
            for item in &schema.system_prompt.items {
                if let SystemPromptItem::Section(s) = item {
                    let tokens = ContentResolver::section_token_count(&read, s.id) as u32;
                    system.push(SystemItem::Section {
                        name: s.name.clone(),
                        tokens,
                    });
                }
            }
            let count = crate::substrate::Substrate::turn_count(&read, timeline);
            let selected = |idx: u32| -> Option<SelectedTurn> {
                let t = TurnIndex(idx);
                if idx >= count {
                    return None;
                }
                let role = read
                    .stream_of(turn_stream_id(timeline.raw(), idx))
                    .and_then(|s| s.decl.as_ref())
                    .and_then(|d| match d {
                        StreamDecl::Turn(t) => Some(t.role),
                        _ => None,
                    })
                    .map(|r| match r {
                        0 => "system",
                        2 => "assistant",
                        _ => "user",
                    })
                    .unwrap_or("user")
                    .to_string();
                let kind = read
                    .tree_meta_of(timeline, t)
                    .map(|m| m.kind)
                    .unwrap_or(TurnKind::Normal);
                Some(SelectedTurn {
                    layer: layer_name.clone(),
                    group: group_name.clone(),
                    index: idx,
                    role,
                    tokens: read.turn_token_count_of(timeline, t) as u32,
                    kind,
                    reason: None,
                    timeline: Some(timeline.raw()),
                    selected: true,
                    score: 0.0,
                    qualified: false,
                })
            };
            let mut turns = Vec::with_capacity(2);
            if let Some(prev) = turn_index.checked_sub(1).and_then(selected) {
                turns.push(prev);
            }
            if let Some(own) = selected(turn_index) {
                turns.push(own);
            }
            (system, turns)
        };

        let events = vec![
            staged_ingest_event(0, 0.0, system.clone(), turns.clone()),
            staged_ingest_event(assistant_content_start, seconds, system, turns),
        ];
        self.substrate.persist_projection_events(
            turn_stream_id(timeline.raw(), turn_index),
            encode_events(&events),
        );
        Ok(())
    }

    /// Belief scores for the just-sealed turn, scored against each belief node
    /// (the target layer's collections AND every layer's belief-driven turn
    /// groups) using the turn's own persisted wide-Q signature as the probe. The
    /// post-turn counterpart of the scheduler's live reproject scan — same
    /// galleries + scorer, but the probe is the finished turn's stored signature
    /// rather than a live gather. Empty when the turn has no signature (nothing
    /// to score against).
    pub(super) fn last_turn_belief_scores(&self, observe: bool) -> ProjectionScores {
        let empty = ProjectionScores::new();
        let timeline = self.target.timeline;
        let (probe, q_span, tags, source) = {
            let read = self.substrate.read();
            let count = read.turn_count(timeline);
            if count == 0 {
                return empty;
            }
            let idx = TurnIndex(count - 1);
            let probe = match read
                .wide_q_sigs_blob(timeline, idx)
                .ok()
                .flatten()
                .and_then(|b| decode_wide_sigs(&b))
            {
                // Same head+tail window the live reproject scans (see
                // `scheduler`'s `QUERY_HEAD_CHUNKS` / `max_probe_tokens`). Without
                // the cap this scan probes with the WHOLE turn, and a turn can be
                // arbitrarily large — a single `file_list` result ran to 5.7k
                // tokens, taking ~7s of scan and scoring every tool in the catalog
                // nonzero (the listing happened to name every tool's own
                // definition file), which then carried into the next turn's
                // opening belief.
                Some(p) => cap_probe_window(p, self.config.reproject_max_probe_tokens),
                None => return empty,
            };
            // Concept F: the sealed turn's question window is its persisted
            // user span — the sig grid is 1:1 with the real-KV layout.
            let q_span = read.user_sig_span(timeline, idx);
            // The turn's own gather-scope tags, which route the observation below
            // to the scopes this turn belongs to. A working-set mark is not
            // scope, so a dialogue turn carrying one still observes as dialogue.
            let tags: Vec<String> = gather_tags(&read.turn_tags(timeline, idx))
                .cloned()
                .collect();
            // The turn's own stream id keys the observation, so a later replay of
            // this turn folds nothing a second time.
            let source = turn_stream_id(timeline.raw(), idx.0).0;
            (probe, q_span, tags, source)
        };
        let probe_q: Vec<WideQSig> = q_span
            .and_then(|r| probe.get(r))
            .map(<[WideQSig]>::to_vec)
            .unwrap_or_default();
        // The seal scan is the once-per-turn learning point for the
        // score-normalization backgrounds — and it teaches only the scopes this
        // turn is inside, so a dialogue turn cannot rewrite a tag-scoped
        // collection's lens. It runs here, on this conversation's thread: the
        // galleries, the turn groups' files and the normalization are substrate
        // work any thread may do, and only the launches go to the scheduler's
        // wave thread, the one thread that may scan the gallery arena
        // (`RemoteArena`). With no arena it scores on the host.
        let arena = RemoteArena::connect(&self.scheduler_tx);
        let (scores, _) = self.substrate.score_beliefs(
            self.projection.schema(),
            self.target,
            &probe,
            (!probe_q.is_empty()).then_some(probe_q.as_slice()),
            if observe {
                Observe::Yes {
                    tags: &tags,
                    source,
                }
            } else {
                Observe::No
            },
            arena.as_ref().map(|a| a as &dyn BeliefArena),
        );
        scores
    }

    /// Recompute the materialized projection for this conversation and pair it
    /// with the just-finished decode's throughput into a [`ProjectionEvent`].
    ///
    /// Call this immediately after [`finish_turn`](Self::finish_turn): the turn
    /// is sealed, so its wide-Q signature is persisted and serves as the belief
    /// probe. Scoring that stored signature against each collection's tag-scoped
    /// gallery gives the same per-section belief the live reproject scan
    /// produces, so the persisted event reflects what provenance selects — not a
    /// zero-score min-budget fill. Returns `None` for layers that don't reproject
    /// (`disable_reprojection` — utility/reference ingestion), where a projection
    /// event carries no meaning.
    ///
    /// The composition (system / section groups / turns), per-category token
    /// counts, substrate total, and decode throughput are all real; the only
    /// approximation is the probe — the recompute scores against the
    /// just-finished turn's whole signature rather than replaying each live
    /// decode-step Q vector.
    pub fn projection_event(&self, stats: &crate::stats::TurnStats) -> Option<ProjectionEvent> {
        if self.config.disable_reprojection {
            return None;
        }
        Some(self.projection_event_scored(stats, &self.last_turn_belief_scores(true)))
    }

    /// What a one-turn ingest unit's just-sealed turn drew in — the
    /// projection its own signature selects, for a caller that must see what
    /// it drew where no live reprojection ever records it — and what that
    /// signature scored each candidate conversation at (its best turn),
    /// highest first, the whole field the selection kept its few from.
    ///
    /// One scan, read-only: it is a record of the turn, not evidence about any
    /// scope, so it teaches no score level the dialogue's retrieval reads.
    pub fn unit_projection(
        &self,
        stats: &crate::stats::TurnStats,
    ) -> (ProjectionEvent, Vec<(TimelineId, f32)>) {
        let scores = self.last_turn_belief_scores(false);
        let mut best: std::collections::HashMap<TimelineId, f32> = std::collections::HashMap::new();
        for (key, score) in scores.turn_scores() {
            let e = best.entry(key.timeline).or_insert(f32::MIN);
            *e = e.max(score);
        }
        let mut candidates: Vec<(TimelineId, f32)> = best.into_iter().collect();
        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        (self.projection_event_scored(stats, &scores), candidates)
    }

    /// The [`ProjectionEvent`] the just-sealed turn's `scores` select.
    pub(super) fn projection_event_scored(
        &self,
        stats: &crate::stats::TurnStats,
        scores: &ProjectionScores,
    ) -> ProjectionEvent {
        let resolver = self.substrate.read_for_scored(self.target, scores);
        let projection = self.projection.project_with_selection(
            self.target,
            &resolver,
            ProjectionMode::Decode,
            &self.selection,
        );
        let substrate_total = resolver.total_token_count(self.target.timeline) as u32;
        // A projection is a POINT on the timeline, not a span. This is the
        // post-seal projection: the turn has finished decoding and its answer is
        // now materialized, so the point sits at the FINAL generated position
        // (`tokens_generated`), governing the closing interval `[last_reproj, end]`.
        // `seconds` is the full decode elapsed — the wall-clock at this final point.
        let mut ev = from_projection_with_origins(
            &projection.segments,
            &projection.selection_origins,
            self.projection.schema(),
            &resolver,
            &projection.selection_scores,
            substrate_total,
            stats.tokens_generated as u32,
            stats.decode_ms / 1000.0,
        );
        ev.pieces = projection.fill.clone();
        ev.unpooled = projection.unpooled.clone();
        // Materialized dialogue, from the SAME `assemble_pieces` decision the
        // engine injects from, so the persisted panel can never drift from what
        // the projection really does. It no longer needs the dialect's boundary
        // markers: a sealed turn carries its own, so there is nothing left for
        // the assembler to emit between turns.
        ev.materialized = materialize_conversation(
            &projection.segments,
            &projection.selection_origins,
            &resolver,
            self.projection.schema(),
            |toks| self.tokenizer.decode(toks, false).unwrap_or_default(),
        );
        ev
    }

    /// Shared post-prefill turn-completion path.
    ///
    /// After a `SubmitTurn` request emits `Done` (whether decoded by
    /// the model or fully prefilled inline), both `finish_turn` and
    /// `insert_turn` need the same fold-down work:
    ///
    /// 1. Apply the scheduler's seal payload (cold store register +
    ///    sig-blocks counter advance + provenance scan probe).  The
    ///    substrate write itself happened scheduler-side before
    ///    `Done` arrived.
    /// 2. Persist the user/assistant entries to the cold store.
    /// 3. Commit the exchange to the conversation tree (which may
    ///    launch cognitive tasks like summarisation).
    /// 4. Apply any cognitive task that has finished — without waiting
    ///    for the ones still running.
    /// 5. Prefill the next user-turn header into the KV cache.
    ///
    /// Returns the boundary text prefilled in step 5 so callers can
    /// echo it for streaming displays.
    pub(super) fn finalize_turn_post_done(
        &mut self,
        user_tt: TokenizedText,
        assistant_tt: TokenizedText,
        seal: Option<&crate::handle::SealResult>,
    ) -> crate::Result<String> {
        // Apply the seal payload that came back on Done.  The
        // scheduler already wrote the turn into the substrate; we
        // just register the bytes with the cold-store and advance
        // local counters.
        if let Some(seal) = seal {
            self.current_blocks = BlockCount(seal.block_count);
            // The scheduler already wrote the turn's sealed bytes,
            // per-block sig entries, role/text/token_ids, AND the
            // workspace persistence layer (when configured) into the
            // substrate before sending `Done` — see
            // `Conversation::record_turn`.  Persistence is per-
            // workspace now, not per-Sequence; nothing for this
            // method to forward.
        }

        // Bump the local turn counter for diagnostics; the canonical
        // per-group append index lives on the substrate now.
        self.turn_counter += 2;

        // Commit the completed exchange to the conversation tree.
        self.tree.finish_turn(
            user_tt,
            assistant_tt,
            TurnType::Reality,
            vec![],
            Some((&self.submitter.scheduler_tx, &self.submitter.tokenizer)),
        );

        // Apply any cognitive task that has finished. Tasks still running —
        // including one `finish_turn` just launched — stay in flight and are
        // picked up at a later turn boundary: the turn never waits on them.
        self.poll_cognitive_tasks();

        // Each turn opens its own user role marker via the
        // `prefill_tokens` of the next `submit_turn` and closes the
        // assistant tail via `post_decode_tokens` after decode.
        // No inter-turn header prefill is needed.
        Ok(String::new())
    }
}
