//! A request's first reading (`docs/hit_driven_projection.md` §7.4).
//!
//! The question a new user message opens is read twice on the conversation's
//! own slot. The first reading prefills it, to the token before its last,
//! against the projection with the seeds held out (`WorkingSet::holds_out`),
//! and scores it: its fresh scores feed the working set's momentum and the
//! boundary takes the question, which brings the seeds back. Its hits are
//! discarded — they were earned against a slot the reply never sees — and so
//! is its view: nothing was sealed, so the parent stands where it stood. The
//! turn is then submitted again, unchanged, and its opening projection places
//! the seeds and what the question ranked in one ranking, chosen whole, before
//! the question is prefilled a second time and the reply decodes.
//!
//! The seeds are background every conversation starts from, and the question
//! read against them scored a different set of files: on BlackWorld
//! (2026-11-20) the question naming `recorded_reply` ranked
//! `recorded_reply.rs` second read alone and fifth read against the seeds.
//! A first reading that cannot be scored fails the turn: nothing decodes on a
//! slot chosen half-way.

use super::*;

/// What a submitted turn is, as far as reading it first depends on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Request {
    /// A new user message waits for its first boundary
    /// (`SlotBoundary::question_pending`).
    pub(super) question_pending: bool,
    /// It seals into the substrate as a turn.
    pub(super) seals_a_turn: bool,
    /// It is a turn group's.
    pub(super) grouped: bool,
    /// It reprojects.
    pub(super) reprojects: bool,
    /// Its prefill carries a calibration's staged points or recurrent arms.
    pub(super) staged: bool,
    /// It replays a recorded turn, whose reprojections are its recording's.
    pub(super) replayed: bool,
    /// Tokens it prefills.
    pub(super) tokens: usize,
}

/// Whether the turn is read first without the seeds: a request — a dialogue
/// turn a new user message opens, sealing as itself and reprojecting — of at
/// least two tokens, so its first reading stops before anything is sampled.
pub(super) fn rereads(r: Request) -> bool {
    r.question_pending
        && r.seals_a_turn
        && !r.grouped
        && r.reprojects
        && !r.staged
        && !r.replayed
        && r.tokens >= 2
}

/// The turn `work` submitted again on `parent`, exactly as it first was.
fn resubmission(work: PrefillWork, parent: SequenceId) -> Option<SchedulerRequest> {
    let policy = work.reprojection.as_ref()?;
    let projection_inputs = ProjectionInputs {
        projection: Arc::clone(&policy.projection),
        selection: policy.selection.clone(),
        frame: None,
    };
    Some(SchedulerRequest::SubmitTurn {
        seal_group: None,
        sequence_id: parent,
        projection_inputs: Some(projection_inputs),
        prefill_tokens: work.tokens,
        prefill_text: work.prefill_text,
        user_text: work.user_text,
        tags: work.tags,
        user_content_start: work.user_content_start,
        user_content_end: work.user_content_end,
        assistant_content_start: work.assistant_content_start,
        no_think: work.no_think,
        projection_offsets: work.projection_offsets,
        prefill_assistant_text: work.prefill_assistant_text,
        post_decode_tokens: work.post_decode_tokens,
        max_decode_tokens: work.max_decode_tokens,
        sampling: work.sampling,
        event_tx: work.event_tx,
        reprojection: work.reprojection,
        disable_reprojection: false,
        triggers: work.triggers,
        turn_grammar: work.turn_grammar,
        free_tool_calls_from_penalties: work.free_tool_calls_from_penalties,
        recorded_reply: work.recorded_reply,
        recurrent_spans: work.recurrent_spans,
    })
}

impl Scheduler {
    /// Score `view_id`'s first reading, discard it, and submit its turn again.
    /// A reading that cannot be scored fails the turn with the cause.
    pub(super) fn read_first(&mut self, view_id: SequenceId) {
        let Some(at) = self
            .active_prefills
            .iter()
            .position(|p| p.work.sequence_id == view_id && p.awaiting && p.work.rereads)
        else {
            return;
        };
        let prefill = self.active_prefills.remove(at);
        let parent = self.turn_views.get(&view_id).map(|v| v.parent_id);
        let t = Instant::now();
        let scored = self.score_first_reading(view_id, &prefill.work);
        self.discard_turn_view(view_id);
        let event_tx = prefill.work.event_tx.clone();
        let fail = |why: ConversationError| {
            let _ = event_tx.send(TurnEvent::Error(why));
        };
        match (scored, parent) {
            (Err(e), _) => fail(e),
            (Ok(_), None) => fail(ConversationError::Channel(format!(
                "first reading: view {view_id} has no parent to submit its turn again on"
            ))),
            (Ok((candidates, scores)), Some(parent)) => {
                self.first_readings.insert(parent, scores);
                note("first reading", || {
                    format!(
                        "view={} parent={} tokens={} candidates={candidates} scored_ms={}",
                        view_id.0,
                        parent.0,
                        prefill.offset,
                        t.elapsed().as_millis()
                    )
                });
                let tokens = prefill.work.tokens.len();
                match resubmission(prefill.work, parent) {
                    Some(request) => {
                        self.handle_request_now(request);
                    }
                    None => fail(ConversationError::Channel(format!(
                        "first reading: the turn of {tokens} tokens on view {view_id} carries \
                         no reprojection policy to submit it again with"
                    ))),
                }
            }
        }
    }

    /// Score the reading `view_id` holds: its probe through the belief scan,
    /// folded into the working set's momentum at a boundary that takes the
    /// question. Its hits are read and dropped. Returns how many candidates
    /// the scan scored, and its scores, which the second reading's opening
    /// projection is ranked by.
    fn score_first_reading(
        &mut self,
        view_id: SequenceId,
        work: &PrefillWork,
    ) -> Result<(usize, ProjectionScores), ConversationError> {
        let policy = work.reprojection.clone().ok_or_else(|| {
            ConversationError::Channel(format!(
                "first reading: view {view_id} carries no reprojection policy"
            ))
        })?;
        let view_state = self.turn_views.get(&view_id).copied().ok_or_else(|| {
            ConversationError::Channel(format!("first reading: missing view state {view_id}"))
        })?;
        let probe = self
            .turn_probe(
                view_id,
                view_state.turn_start_parent_blocks,
                view_state.question_tokens,
                policy.max_probe_tokens,
            )
            .ok_or_else(|| {
                ConversationError::Other(format!(
                    "first reading: view {view_id} captured no probe of its question"
                ))
            })?;
        self.queue_norm_warm(
            &policy.substrate,
            &policy.projection,
            policy.target,
            policy.max_probe_tokens,
        );
        let schema = policy.projection.schema();
        let (scores, group_candidates) = policy.substrate.score_beliefs(
            schema,
            policy.target,
            &probe.probe,
            (!probe.question.is_empty()).then_some(probe.question.as_slice()),
            // Read, not taught: the turn teaches the levels once, at seal.
            Observe::No,
            self.gallery_arena.as_deref().map(|a| a as &dyn BeliefArena),
        );
        // The reading's hits were counted against a slot the reply never
        // sees: read, so the counters start the second reading at zero, and
        // dropped.
        let slot = view_state.parent_id;
        if let Some(state) = self.slot_projection_state.get_mut(&slot) {
            state.unread_hits.clear();
            self.model
                .take_hits(slot.0)
                .map_err(ConversationError::Model)?;
        }
        if policy.substrate.read().pool_size() != self.pool_size {
            let size = self.pool_size;
            policy.substrate.update(|s| s.set_pool_size(size));
        }
        let placed = self
            .slot_projection_state
            .get(&slot)
            .map(|s| s.placed_turns())
            .unwrap_or_default();
        let target = policy.target.timeline;
        policy
            .substrate
            .update(|s| s.working_set_boundary(target, &placed));
        policy.substrate.observe_working_set(
            schema,
            policy.target,
            &group_candidates,
            &HashMap::new(),
        );
        Ok((group_candidates.iter().map(|(_, c)| c.len()).sum(), scores))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> Request {
        Request {
            question_pending: true,
            seals_a_turn: true,
            grouped: false,
            reprojects: true,
            staged: false,
            replayed: false,
            tokens: 40,
        }
    }

    /// **A request is read first**: a dialogue turn a new user message
    /// opens, sealing as itself and reprojecting.
    #[test]
    fn a_request_is_read_first() {
        assert!(rereads(request()));
    }

    /// **Every other turn is read once**: a tool result (no question
    /// waiting), a probe or summary that seals nothing, a turn group's case,
    /// a turn that does not reproject, a calibration's staged prefill, a
    /// replay, and a turn too short to stop before its last token.
    #[test]
    fn every_other_turn_is_read_once() {
        let r = request();
        for (why, turn) in [
            (
                "a tool result",
                Request {
                    question_pending: false,
                    ..r
                },
            ),
            (
                "seals nothing",
                Request {
                    seals_a_turn: false,
                    ..r
                },
            ),
            ("a group's case", Request { grouped: true, ..r }),
            (
                "does not reproject",
                Request {
                    reprojects: false,
                    ..r
                },
            ),
            ("staged", Request { staged: true, ..r }),
            (
                "a replay",
                Request {
                    replayed: true,
                    ..r
                },
            ),
            ("one token", Request { tokens: 1, ..r }),
        ] {
            assert!(!rereads(turn), "{why}");
        }
    }
}
