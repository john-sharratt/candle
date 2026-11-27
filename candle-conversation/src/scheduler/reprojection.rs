//! The continuous reprojection policy, and the completion phase that rebuilds the
//! view and re-keys the decode onto it.
//!
//! A reprojection re-runs the belief scan over the turn's content and records the
//! new selection as a point in the timeline, so a consumer reconstructs each
//! selection's interval from the sequence of points. The completion appends the
//! captured tail and reattaches its index rows as pages. A refused page makes the
//! turn wait for ground rather than decode over K/V that nothing indexes. The
//! turn's state moves onto the new view instead of being copied, and a view that
//! is discarded hands the state back. Belief is carried from one reprojection to
//! the next, so its decay and reinforcement accumulate across the whole turn.

use super::*;
use crate::projection::ProjectionEvent;

/// All the per-conversation context the scheduler needs to do continuous
/// re-projection on its own thread, without round-tripping back to the
/// caller.  Built once at [`SchedulerRequest::SubmitTurn`] time and stored
/// on the active `DecodeState`.
///
/// At each boundary its `boundary` rule fires, the scheduler:
///
/// 1. Tracks decoded-token count per-view.
/// 2. Extracts live Q sigs from the view's R16 backing at the three
///    provenance layers.
/// 3. Runs provenance against the substrate corpus and writes fresh per-turn
///    scores under [`Conversation::write`].
/// 4. Re-runs `Builder::project` for `target` to obtain a new
///    visible-block set.
/// 5. Drops the current view (transferring its decoded blocks to the
///    parent via `finalize_view`) and carves a fresh view borrowing the
///    new ranges plus the active turn's own decoded tail.  Decode state
///    (sampler, generated-token buffer, health) is re-keyed onto the new
///    view id, invisible to the caller.
#[derive(Clone)]
pub(crate) struct ReprojectionPolicy {
    pub(crate) target: ProjectionTarget,
    pub(crate) projection: Arc<Builder>,
    /// Section-tree selection (the composer dials), so decode reprojection emits
    /// the same selector options as the initial prefill.
    pub(crate) selection: SelectionState,
    pub(crate) substrate: Conversation,
    /// Where reprojections fire: the cadence, the trigger tokens and the
    /// tool-call gate (`provenance::boundaries`) — the same rule the gallery
    /// cuts ingested turns by.
    pub(crate) boundary: BoundaryRule,
    /// Maximum tokens looking back from the current decode position to
    /// include in the wide-Q probe.  Caps the "thought window" — beyond
    /// this many tokens the prior reprojection already captured the
    /// older intent.  Must be `>= 1`.
    pub(crate) max_probe_tokens: usize,
    /// Which decode boundaries reproject: the turn's reprojection
    /// (`ReprojectionMode`), over `SequenceConfig::reproject_in_thinking`;
    /// see [`think_gate`]. The opening boundary ([`Self::opening_chunk`])
    /// selects whatever this says.
    pub(crate) decode: DecodeReprojection,
}

impl ReprojectionPolicy {
    /// How far an opening is prefilled before a boundary re-chooses the slot
    /// (`docs/hit_driven_projection.md` §7.4): the target layer's
    /// `reserve_tokens`, on an engine that selects. `None` where the slot is
    /// not filled — a target with no working set, or an engine that attends
    /// every position.
    pub(crate) fn opening_chunk(&self) -> Option<usize> {
        self.substrate.read().trained_window()?;
        self.projection
            .schema()
            .layers
            .iter()
            .find(|l| l.id == self.target.layer)?
            .working_set
            .as_ref()
            .map(|ws| ws.reserve_tokens)
            .filter(|&r| r > 0)
    }
}

// ————————————————————————————————————————————————————————————————————————————
// Internal state
// ————————————————————————————————————————————————————————————————————————————

/// What a reprojection selected, for its rebuild
/// ([`Scheduler::reproject_rebuild`]): the sealed sections to lift, the
/// segments to apply, the turns to lift, and the composition the turn's
/// belief and the projection event are taken from.
pub(super) struct Selected {
    pub(super) sections: Vec<SectionId>,
    pub(super) segments: Vec<ProjectionSegment>,
    pub(super) turns: Vec<TurnKey>,
    pub(super) composition: ProjectionEvent,
}

/// What a reprojection's selection took, reported with its rebuild.
pub(super) struct PrepareTimes {
    /// When the reprojection began.
    pub(super) t_repro: Instant,
    pub(super) probe_ms: u64,
    pub(super) scan_ms: u64,
    pub(super) project_ms: u64,
    pub(super) warm_us: u64,
    pub(super) score_us: u64,
    pub(super) observe_us: u64,
    /// Tokens the probe covered.
    pub(super) probe_tokens: usize,
}

/// In-flight reprojection state carried from
/// [`Scheduler::reproject_view_prepare`] to
/// [`Scheduler::reproject_view_complete`], so the cross-conversation wave can
/// prepare many views (scan + project + inject sealed + build the gap-fill
/// descriptor), fire ONE batched multi-slot gap-fill forward, then complete
/// each view (finish + carve new view + re-key) independently.
pub(super) struct ReprojectInFlight {
    pub(super) view_id: SequenceId,
    pub(super) parent_id: SequenceId,
    pub(super) tail_per_layer: Vec<candle_nn::kv_cache::SealedSequence>,
    /// The index rows describing `tail_per_layer`, taken in `prepare` before the
    /// re-projection resets the parent's per-position state, and re-attached in
    /// `complete` where that K/V is injected back. Travels with the K/V it
    /// describes for the same reason a sealed section's page does: the rows come
    /// from hidden states, so nothing downstream can recompute them.
    ///
    /// `(token width, bytes)` per page, as the seal produced them. The widths
    /// are unused on this path — the pages go straight back in order — but they
    /// travel rather than being discarded, so this list and a stored one are the
    /// same shape.
    pub(super) tail_index_page: Vec<(usize, Vec<u8>)>,
    /// The turn being swapped: a decode, or an opening stopped at a boundary
    /// (`held_turn`).
    pub(super) turn: HeldTurn,
    pub(super) sampling_state: Option<SequenceSamplingState>,
    pub(super) sections_len: usize,
    pub(super) segments_len: usize,
    pub(super) n_turns_selected: usize,
    /// The materialized-context composition this reprojection selected
    /// (buckets + materialized/substrate totals; the decode-span timing is
    /// filled in at `complete` from the per-sequence anchor). Emitted as a
    /// [`TurnEvent::Projection`] so the GUI drops a timeline dot per reproject.
    pub(super) composition: ProjectionEvent,
    pub(super) plan: projection_assembler::GapFillPlan,
    pub(super) build_ms: u64,
    /// Wall-clock start of the whole reproject (set at the top of `prepare`);
    /// drives `total_ms` reported at the end of `complete`.
    pub(super) t_repro: Instant,
    /// Pure swap work in `prepare` (tail snapshot + free + truncate) — disjoint
    /// from `elevate_ms`/`glue_ms`/`apply_ms`, unlike the old umbrella that
    /// spanned the whole reproject and hid the glue wave inside it.
    pub(super) swap_ms: u64,
    pub(super) probe_ms: u64,
    pub(super) scan_ms: u64,
    pub(super) project_ms: u64,
    pub(super) elevate_ms: u64,
    /// The turn's Concept F question boundary, carried from the freed view's
    /// [`ViewState`] into the re-carved one.
    pub(super) question_tokens: usize,
    /// The identity of the segment list this rebuild applies — recorded on the
    /// re-carved view's [`ViewState::applied_identity`] when the rebuild placed
    /// every piece. `None` when a conversation was withdrawn for not being
    /// whole in VRAM (`whole_conversations`): the slot then does not hold the
    /// selection the identity names, and claiming it would keep the next
    /// reprojection of the same selection from ever bringing that conversation.
    pub(super) identity: Option<u64>,
}

impl Scheduler {
    /// Count a reprojection that ended as `outcome`, started at `started`, and
    /// write the census once its minute is up.
    pub(super) fn end_reprojection(&mut self, outcome: ReprojectOutcome, started: Instant) {
        let now = Instant::now();
        self.reproject_census.ended(
            outcome,
            now.saturating_duration_since(started).as_micros() as u64,
        );
        if let Some([prepare, apply]) = self.reproject_census.due(now) {
            tracing::info!(
                target: "candle_conversation::scheduler::reproject",
                "reprojection census {prepare}"
            );
            tracing::info!(
                target: "candle_conversation::scheduler::reproject",
                "reprojection census apply {apply}"
            );
            // The belief scans behind the census's `score` part, by phase.
            tracing::info!(
                target: "candle_conversation::scheduler::reproject",
                "reprojection census scan {}",
                take_scan_census().line()
            );
        }
    }

    /// Complete phase of reprojection — runs after the batched gap-fill forward
    /// has fired + committed every prepared slot's glue. Finishes the projection
    /// (deferred user prefill + restore tail), appends the active turn's captured
    /// tail, carves a fresh view from the rebuilt parent, and re-keys the
    /// per-view maps onto the new view id.
    pub(super) fn reproject_view_complete(
        &mut self,
        inflight: ReprojectInFlight,
        glue_ms: u64,
    ) -> Result<SequenceId, ConversationError> {
        let ReprojectInFlight {
            view_id,
            parent_id,
            tail_per_layer,
            tail_index_page,
            mut turn,
            sampling_state,
            sections_len,
            segments_len,
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
            question_tokens,
            identity,
        } = inflight;

        // Carry the belief forward: the next reprojection seeds from what this one
        // selected, so the RelLeak decay/reinforcement accumulates across the turn.
        // Migrates with the decode state into `new_view_id` below.
        turn.set_belief(PriorBelief::from_selection(&composition.selection));

        // A projection is a POINT, not a span: it is selected here (by the Q of
        // the tokens decoded so far) and governs everything forward until the next
        // projection supersedes it. So emit it the instant it occurs — `start_token`
        // is the generated-token position at which it was selected, `seconds` is the
        // wall-clock elapsed since the turn began producing. The consumer
        // reconstructs each projection's effective interval `[pos_i, pos_{i+1})`
        // and its throughput from the *sequence* of events; there is nothing to
        // close or flush. Marking the point advances a decode's span cursor for
        // the next reprojection / final seal. (The wide `sign(Q)` history is
        // captured continuously and persisted at seal — it is not tied to
        // reprojections; the event is just a marker.)
        {
            let (start_token, seconds) = turn.projection_point(Instant::now());
            let event = ProjectionEvent {
                start_token,
                seconds,
                ..composition
            };
            let _ = turn.event_tx().send(TurnEvent::Projection(event));
        }

        // The turn's state is this reprojection's until it is re-bound below,
        // so every exit goes through `or_abandon!` (`reproject_abandon.rs`).
        // The old view's slot was freed in prepare.
        let t_finish = Instant::now();
        let finish_span = profile::span("reproject:apply_finish");
        let applied_identity = identity.filter(|_| plan.complete);
        or_abandon!(
            self,
            view_id,
            parent_id,
            turn,
            false,
            self.apply_projection_finish(parent_id, plan)
        );
        // The rebuilt prefix ends here, so the turn's anchor moves here — and
        // the index closes with it. The rebuild re-supplies the turn's user half
        // by *prefilling* it (`deferred_user`), which leaves rows in the live
        // tail; without this cut the tail pages pushed below land on a
        // non-empty tail and are refused outright, and the slot then holds the
        // turn's K/V with nothing indexing it.
        //
        // The anchor is the blocks holding content, not the block count: an
        // empty writer chunk at the tail is where the turn's captured tail
        // lands below (`inject_sealed_at_tail` trims it first), so a count that
        // included it would start the turn one chunk into its own tail — the
        // turn's seal missing that chunk, and the next rebuild's cut back to the
        // anchor keeping it.
        let new_prefix_block_count = or_abandon!(
            self,
            view_id,
            parent_id,
            turn,
            false,
            self.session
                .sequence_content_block_count(parent_id.0)
                .map_err(ConversationError::Model)
                .and_then(|n| {
                    n.ok_or_else(|| {
                        ConversationError::Channel(format!(
                            "reproject: parent slot {parent_id} not in session"
                        ))
                    })
                })
        );
        self.begin_unit(parent_id, "reprojected turn");
        finish_span.end();
        self.reproject_census
            .add(ReprojectPart::Glue, glue_ms * 1000);
        self.reproject_census
            .add_since(ReprojectPart::Finish, t_finish);
        // apply_ms = this slot's segment build + finish, NOT the shared gap-fill
        // wave (reported separately as glue_ms), so it no longer absorbs the
        // cross-slot wave wait.
        let apply_ms = build_ms + t_finish.elapsed().as_millis() as u64;

        // Under the read-only projection model there's no COW prefix
        // duplication to guard against: the view never copied bytes from
        // the parent's partial — it borrowed the partial read-only and
        // wrote new K/V into a fresh active chunk. So the captured tail
        // can be re-injected directly on top of the freshly-projected
        // parent with no truncate dance.

        // Append the captured tail to parent (metadata-only Arc clone).
        let tail_token_count: usize = tail_per_layer.first().map(|s| s.token_count).unwrap_or(0);
        let t_inject = Instant::now();
        let inject_span = profile::span("reproject:inject_tail");
        if tail_token_count > 0 {
            let placed = or_abandon!(
                self,
                view_id,
                parent_id,
                turn,
                false,
                self.session
                    .inject_sealed_at_tail(parent_id.0, &tail_per_layer)
                    .map_err(ConversationError::Model)
            );
            // Captured from this slot a moment ago: a cut here would drop
            // decoded tokens the turn's text still holds.
            if placed.was_cut() {
                let cut: Result<(), ConversationError> = Err(ConversationError::Other(format!(
                    "reproject: the captured decode tail's layers describe different token \
                     windows — {} chunk(s) cut, {} of {tail_token_count} token(s) placed",
                    placed.cut_chunks, placed.tokens
                )));
                or_abandon!(self, view_id, parent_id, turn, false, cut);
            }
            // The rows that describe those tokens, back at the position the
            // tokens landed. The rebuilt prefix ended flush with its last
            // injected piece and forwarded nothing, so these rows sit on an
            // empty live tail and go in as a page — the same shape every other
            // borrowed piece uses.
            if !tail_index_page.is_empty() {
                // In order: each piece the turn spanned keeps its own width.
                // Sealed a moment ago and handed over once, so keyed here.
                //
                // **A refused page is ground, and the turn waits for it rather
                // than decoding over K/V nothing indexes.** The pages are the
                // turn's own, freed with the old view a moment ago, so relief
                // can find them again. Decoding on regardless refused the turn's
                // first select past the gap — and failed every turn that rode
                // its decode step. Measured on the RTX 3090 (run 42,
                // 2026-10-21): 3 of 14 index slots refused with the span full,
                // a slot 14,171 tokens short, and two shift turns failed.
                for (_tokens, page) in tail_index_page.iter() {
                    let key = PieceKey::of(page);
                    let pushed = match self.model.push_positional_state(parent_id.0, page, &key) {
                        Err(e) if candle_nn::kv_cache::is_span_full(&e) => {
                            self.relieve_vram_pressure(
                                "reproject index",
                                prefill::VramPhase::Decode,
                                prefill::ReliefNeed::Now,
                            );
                            self.model.push_positional_state(parent_id.0, page, &key)
                        }
                        pushed => pushed,
                    };
                    let refused: Result<(), ConversationError> = pushed.map(|_| ()).map_err(|e| {
                        ConversationError::Channel(format!(
                            "reproject: the turn's index page was refused ({e}); the slot \
                             would hold the turn's K/V with nothing indexing it"
                        ))
                    });
                    or_abandon!(self, view_id, parent_id, turn, false, refused);
                }
            } else if self.model.carries_positional_state() {
                tracing::warn!(
                    slot = parent_id.0,
                    tail_token_count,
                    "reproject: the turn's K/V came back with no index page"
                );
            }
            // The turn's K/V is back; its rows have to be too. A gap here is the
            // width of whatever part of the turn the pages did not cover, and it
            // is the last point at which that is attributable to this rebuild —
            // downstream it surfaces only as a refused select on some later turn.
            if let Some(cov) = self.model.positional_coverage(parent_id.0) {
                let held = self.session.sequence_offset(parent_id.0).unwrap_or(0);
                // Both directions. Short drops rows the turn needs; long means
                // a piece was restored twice — the rebuild re-supplies the
                // turn's user half from cache and the span seal carries it too
                // — and the surplus rows claim positions this slot does not
                // hold, which selects against them without erroring.
                //
                // Reported, not refused, for the reason `apply_segments_finish`
                // records at the same check: a live run showed this path already
                // diverged before the thinking-span work, so refusing here kills
                // conversations over a fault that is not theirs.
                if cov != held {
                    tracing::warn!(
                        slot = parent_id.0,
                        held,
                        coverage = cov,
                        tail_token_count,
                        pages_pushed = tail_index_page.len(),
                        "reproject: the rebuilt slot's index is {} token(s) {} its K/V",
                        cov.abs_diff(held),
                        if cov < held { "short of" } else { "past" }
                    );
                }
            }
            // Mirror the tail's tokens back into the slot_tokens log
            // (we recorded them at original prefill / decode time
            // under the now-freed view id, but the parent's log was
            // also wiped above). Reconstruct from the turn's own record of
            // what it forwarded: the opening, then what it decoded.
            let (opening, decoded) = turn.slot_tokens();
            Self::record_slot_tokens(&mut self.slot_tokens, parent_id, opening);
            Self::record_slot_tokens(&mut self.slot_tokens, parent_id, decoded);
        }

        inject_span.end();
        let inject_ms = t_inject.elapsed().as_millis() as u64;
        self.reproject_census
            .add_since(ReprojectPart::Inject, t_inject);

        // Carve a fresh view from the rebuilt parent, borrowing every chunk, and
        // re-bind the turn to it. The active turn's content starts where the new
        // prefix ends — `cleanup_finished`'s seal extracts
        // `parent[turn_start_parent_blocks..end]` to produce the substrate turn
        // entry, so the anchor excludes the freshly-injected sections / turns
        // and includes the tail.
        //
        // **The turn's state moves onto the new view; it is not copied.** The
        // move in prepare left the parent holding the turn's only copy, and the
        // turn's seal moves it back there, so nothing reads the parent's while
        // the turn runs. A fork here claimed a second whole store for the
        // length of every reprojection — on the RTX 3090 (2026-10-18), 36 slots
        // refused "after 20 of 36" with the weight side at its floor, ending the
        // reply, and later parking it 0.1–6 s per rebuild. Moved, the rebuild
        // claims nothing, and a view that is discarded hands its state back
        // (`ViewState::holds_parent_state`).
        let parent_block_count = self.session.sequence_block_count(parent_id.0).unwrap_or(0);
        let effective_ranges: Vec<BlockRange> = if parent_block_count == 0 {
            Vec::new()
        } else {
            vec![BlockRange::new(0, parent_block_count)]
        };
        let t_view = Instant::now();
        let view_span = profile::span("reproject:carve_view");
        let (new_view_id, new_borrowed) = or_abandon!(
            self,
            view_id,
            parent_id,
            turn,
            false,
            self.carve_view(parent_id, &effective_ranges)
        );
        if let Err(e) = self.model.move_recurrent(parent_id.0, new_view_id.0) {
            if let Err(free) = self.session.free_sequence(new_view_id.0) {
                tracing::warn!("failed to free view {new_view_id} after a failed move: {free}");
            }
            self.sampling_states.remove(&new_view_id);
            let e = ConversationError::Model(e);
            self.abandon_reprojection(view_id, parent_id, turn, false, &e);
            return Err(e);
        }
        view_span.end();
        let view_ms = t_view.elapsed().as_millis() as u64;
        self.reproject_census.add_since(ReprojectPart::View, t_view);
        // The new view's slot may have been recycled from the freed old view's.
        self.slot_tokens.remove(&new_view_id);
        self.turn_views.insert(
            new_view_id,
            ViewState {
                parent_id,
                original_borrowed: new_borrowed,
                turn_start_parent_blocks: new_prefix_block_count,
                question_tokens,
                applied_identity,
                state_forked: true,
                holds_parent_state: true,
            },
        );
        // `carve_view` seeded a fresh sampling state from the parent; the live
        // one carried across keeps DRY history and per-turn counters. A decode
        // goes back to the loop; an opening to the prefill group, its next
        // stride against the rebuilt slot.
        self.rebind_turn(view_id, new_view_id, turn);
        if let Some(state) = sampling_state {
            self.sampling_states.insert(new_view_id, state);
        }
        // True end-to-end wall-clock of the whole reproject (prepare → wave →
        // complete), so the phase fields below — which are individually disjoint
        // but separated by the shared glue wave — can be read against a real
        // total instead of summed by eye.
        let total_ms = t_repro.elapsed().as_millis() as u64;
        self.end_reprojection(ReprojectOutcome::Rebuilt, t_repro);

        // Per-reproject timing breakdown — an event, not a log line: a
        // continuous-reproject run reprojects several times a second.
        note("reproject (zero-copy rebuild)", || {
            format!(
                "from_view={} to_view={} new_prefix_blocks={new_prefix_block_count} \
                 tail_tokens={tail_token_count} parent_blocks_after={parent_block_count} \
                 new_borrowed={} total_ms={total_ms} probe_ms={probe_ms} scan_ms={scan_ms} \
                 project_ms={project_ms} swap_ms={swap_ms} elevate_ms={elevate_ms} \
                 glue_ms={glue_ms} apply_ms={apply_ms} inject_ms={inject_ms} view_ms={view_ms} \
                 sections={sections_len} segments={segments_len} turns={n_turns_selected}",
                view_id.0, new_view_id.0, new_borrowed.0,
            )
        });

        Ok(new_view_id)
    }
}
