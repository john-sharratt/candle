//! Turn groups: many prefilled turns submitted at once, each prefilled on a
//! view of its own and sealed as a turn of its own.
//!
//! # Every case is masked to itself
//!
//! A group's cases share one projection — the parent slot, rebuilt once by the
//! submit — and nothing else. Each case is carved its own view over the
//! parent's projected ranges ([`Scheduler::create_view`]): the prefix's K/V is
//! borrowed zero-copy and its per-sequence state (a hybrid's recurrence, the
//! positional index) is forked. So a case's prefill attends to the prefix and
//! to itself, never to a sibling, and its recurrence starts from the prefix's
//! state rather than from the end of the case before it. The views ride the
//! ordinary multi-sequence prefill waves together, which is what makes a group
//! cheaper than its cases submitted one at a time.
//!
//! # Sealed in order, from the view
//!
//! A case is sealed straight from its own view, filed under the parent's
//! conversation and target ([`Scheduler::perform_seal_and_write`]'s `owner`),
//! and its view is released after. Nothing is finalized onto the parent: the
//! parent is what every later case is carved from, so it holds the projected
//! prefix and the prefix's state until the group ends. The last case's state
//! is the one moved onto the parent, as a lone turn's would be.
//!
//! Cases seal in submission order, whatever order their prefills finish in —
//! a turn's substrate index is assigned as it is written, and the caller maps
//! cases to turns by that order.
//!
//! # Carved at admission
//!
//! Each view holds a forked copy of the parent's recurrent state for as long as
//! it lives — on Qwen3.6-35B-A3B about 126 MiB, on Qwen3.8-Flash-Next about
//! 226 MB — and a writer chunk on every attention layer. So a case's view is
//! carved only when admission takes its prefill ([`Scheduler::begin_admitted`]),
//! priced like any other first admission: its store, its K/V, its rows in the
//! tier.
//!
//! The group queues its cases in order, at most [`Scheduler::MAX_PREFILL_WIDTH`]
//! at a time — a bookkeeping bound, not the throttle — and each seal makes room
//! for the next. Carving the views as the group queues them would claim them
//! against no judgement of the card — up to 24 a group, sixteen groups in
//! flight, on a 16 GB card — and the carves that failed would lose their cases
//! and leave half-built views pinning the parent's prefix.
//!
//! The parent's state does not move while its group runs, so a case carved late
//! forks the same state one carved early would have. And a group's cases share
//! one parent and so one band, which admission takes FIFO: the case that must
//! seal next is never admitted behind one after it, so carving late cannot
//! stall the seal order.

use super::*;

/// One case of one turn group — what a case's [`SealAction::GroupCase`]
/// carries back to its group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct GroupCaseId {
    /// The group, as filed in [`Scheduler::turn_groups`].
    pub group: u64,
    /// The case's position in the group's submission.
    pub index: usize,
}

/// Where one case stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CaseState {
    /// Not yet offered: no prefill queued, no view.
    Waiting,
    /// Its prefill is queued for admission; its view is carved when admission
    /// takes it.
    Queued,
    /// Prefilling on this view.
    Prefilling(SequenceId),
    /// Prefilled on this view, waiting for the cases before it to seal.
    Finished(SequenceId),
    /// Sealed, or given up on; its view is gone either way.
    Closed,
}

/// The group's cases and the order they open and seal in — the part of a group
/// that is pure bookkeeping, kept apart so its ordering is testable without a
/// session.
#[derive(Debug)]
struct CaseLedger {
    states: Vec<CaseState>,
    /// The first case not yet closed. Every case before it is closed, and it is
    /// the only case that may seal.
    next_to_seal: usize,
    /// The most views the group holds at once.
    window: usize,
}

impl CaseLedger {
    fn new(cases: usize, window: usize) -> Self {
        Self {
            states: vec![CaseState::Waiting; cases],
            next_to_seal: 0,
            window: window.max(1),
        }
    }

    /// Cases the group has in play: queued for admission or holding a view.
    fn live(&self) -> usize {
        self.states
            .iter()
            .filter(|s| {
                matches!(
                    s,
                    CaseState::Queued | CaseState::Prefilling(_) | CaseState::Finished(_)
                )
            })
            .count()
    }

    /// The next case to queue, while the window has room. Cases queue in
    /// order, and admission takes one band FIFO, so the case that must seal
    /// next is never admitted behind a case after it.
    fn to_open(&self) -> Option<usize> {
        if self.live() >= self.window {
            return None;
        }
        self.states.iter().position(|s| *s == CaseState::Waiting)
    }

    /// The case's prefill went into the queue.
    fn queued(&mut self, index: usize) {
        self.states[index] = CaseState::Queued;
    }

    /// Whether the case is queued and still waiting for its view.
    fn is_queued(&self, index: usize) -> bool {
        self.states.get(index) == Some(&CaseState::Queued)
    }

    /// Admission took the case and carved it `view`.
    fn opened(&mut self, index: usize, view: SequenceId) {
        self.states[index] = CaseState::Prefilling(view);
    }

    /// A case's prefill finished on `view`.
    fn finished(&mut self, index: usize, view: SequenceId) {
        if self.states[index] == CaseState::Prefilling(view) {
            self.states[index] = CaseState::Finished(view);
        }
    }

    /// A case that will never seal — its view could not be carved, or its
    /// prefill failed. Returns the case it was, if `view` is one of this
    /// group's.
    fn lost(&mut self, view: SequenceId) -> Option<usize> {
        let index = self
            .states
            .iter()
            .position(|s| *s == CaseState::Prefilling(view))?;
        self.states[index] = CaseState::Closed;
        Some(index)
    }

    fn lost_at(&mut self, index: usize) {
        self.states[index] = CaseState::Closed;
    }

    /// The next case to seal, if it has finished: its index and its view, now
    /// counted as closed. Closed cases at the head are stepped over.
    fn take_sealable(&mut self) -> Option<(usize, SequenceId)> {
        while self.next_to_seal < self.states.len() {
            match self.states[self.next_to_seal] {
                CaseState::Closed => self.next_to_seal += 1,
                CaseState::Finished(view) => {
                    let index = self.next_to_seal;
                    self.states[index] = CaseState::Closed;
                    self.next_to_seal += 1;
                    return Some((index, view));
                }
                CaseState::Waiting | CaseState::Queued | CaseState::Prefilling(_) => return None,
            }
        }
        None
    }

    /// Every case is closed.
    fn complete(&self) -> bool {
        self.next_to_seal == self.states.len()
    }

    /// The views whose prefill has finished — the ones no discard will reach,
    /// because they have left the prefill queue.
    fn finished_views(&self) -> Vec<SequenceId> {
        self.states
            .iter()
            .filter_map(|s| match s {
                CaseState::Finished(v) => Some(*v),
                _ => None,
            })
            .collect()
    }
}

/// A turn group in flight — see the module docs.
pub(super) struct TurnGroup {
    /// The conversation's slot: the projected prefix every case is carved
    /// from, and the owner every case is sealed under.
    parent_id: SequenceId,
    /// The parent's ranges every case's view borrows.
    ranges: Vec<BlockRange>,
    /// Each case's turn — its tokens are the whole of what it prefills.
    cases: Vec<TurnContent>,
    ledger: CaseLedger,
    sampling: SamplingConfig,
    belief: PriorBelief,
    /// The caller's channel. It hears the submit's opening projection and the
    /// group's one `Done`.
    event_tx: Sender<TurnEvent>,
    /// Every case's own channel. A case's events are its prefill's — a first
    /// token nobody decodes, an error — and are the group's to read, not the
    /// caller's: the caller is told about the group as a whole.
    case_tx: Sender<TurnEvent>,
    case_rx: Receiver<TurnEvent>,
    /// The latest case's seal, which rides the `Done`.
    last_seal: Option<SealResult>,
    sealed: usize,
    /// Cases that will never seal — their view could not be carved, their
    /// prefill failed, or their seal wrote nothing. The caller is told how
    /// many, so a group that lost cases is not taken for a complete one.
    lost: usize,
    prefill_ms: f64,
    prefill_tokens: usize,
    started: Instant,
}

impl Scheduler {
    /// Start a turn group on `parent_id`, which the submit has just projected.
    ///
    /// The cases are carved over `ranges` — the same ranges a lone turn's view
    /// would borrow — and queued for prefill as their views open.
    pub(super) fn start_turn_group(
        &mut self,
        parent_id: SequenceId,
        ranges: Vec<BlockRange>,
        cases: Vec<TurnContent>,
        sampling: SamplingConfig,
        belief: PriorBelief,
        event_tx: Sender<TurnEvent>,
    ) {
        let group = self.next_turn_group;
        self.next_turn_group += 1;
        let (case_tx, case_rx) = flume::unbounded();
        let ledger = CaseLedger::new(cases.len(), Self::MAX_PREFILL_WIDTH);
        tracing::debug!(
            target: "candle_conversation::scheduler",
            group,
            parent = parent_id.0,
            cases = cases.len(),
            window = ledger.window,
            "turn group opened",
        );
        self.turn_groups.insert(
            group,
            TurnGroup {
                parent_id,
                ranges,
                cases,
                ledger,
                sampling,
                belief,
                event_tx,
                case_tx,
                case_rx,
                last_seal: None,
                sealed: 0,
                lost: 0,
                prefill_ms: 0.0,
                prefill_tokens: 0,
                started: Instant::now(),
            },
        );
        // An arrival is an admission opportunity — see the single-turn path.
        self.settled_since_admit = true;
        self.advance_turn_group(group);
    }

    /// A case's prefill finished on `view`: seal whatever is now in order.
    pub(super) fn finish_group_case(
        &mut self,
        view: SequenceId,
        case: GroupCaseId,
        prefill_ms: f64,
    ) {
        let Some(group) = self.turn_groups.get_mut(&case.group) else {
            // The group was abandoned under it; the view is all that is left.
            self.release_case_view(view, None);
            return;
        };
        group.ledger.finished(case.index, view);
        group.prefill_ms = group.prefill_ms.max(prefill_ms);
        self.advance_turn_group(case.group);
    }

    /// `view` was discarded without finishing. If it was a group's case, the
    /// group passes over it.
    pub(super) fn lose_group_case(&mut self, view: SequenceId) {
        let Some((&id, index)) = self
            .turn_groups
            .iter_mut()
            .find_map(|(id, g)| g.ledger.lost(view).map(|index| (id, index)))
        else {
            return;
        };
        let group = self.turn_groups.get_mut(&id).expect("found above");
        group.lost += 1;
        for event in group.case_rx.try_iter() {
            if let TurnEvent::Error(e) = event {
                tracing::warn!(group = id, case = index, "turn group case failed: {e}");
            }
        }
        tracing::warn!(
            group = id,
            case = index,
            "turn group case lost before it sealed — the group goes on without it",
        );
        self.advance_turn_group(id);
    }

    /// The turn-group case `work` is, while it is still queued without a view.
    pub(super) fn uncarved_case(&self, work: &PrefillWork) -> Option<GroupCaseId> {
        let SealAction::GroupCase(case) = &work.seal_action else {
            return None;
        };
        self.turn_groups
            .get(&case.group)
            .filter(|g| g.ledger.is_queued(case.index))
            .map(|_| *case)
    }

    /// Carve the view of a queued case admission has just taken, and point its
    /// prefill at it.
    ///
    /// The whole of what a case claims happens here: the parent's prefix
    /// borrowed, a writer chunk per attention layer, the parent's recurrent
    /// state forked. A failure leaves nothing behind (`create_view` frees a view
    /// that fails part way) and the case still queued, so the caller decides
    /// whether it waits or is lost.
    pub(super) fn carve_group_case(
        &mut self,
        work: &mut PrefillWork,
        case: GroupCaseId,
    ) -> Result<(), ConversationError> {
        let group = self.turn_groups.get(&case.group).ok_or_else(|| {
            ConversationError::Channel(format!("turn group {} is gone", case.group))
        })?;
        let (parent, ranges) = (group.parent_id, group.ranges.clone());
        let (view, borrowed) = self.create_view(parent, &ranges)?;
        self.turn_views.insert(
            view,
            ViewState {
                parent_id: parent,
                original_borrowed: borrowed,
                turn_start_parent_blocks: borrowed.0,
                question_tokens: 0,
                applied_identity: None,
                state_forked: true,
            },
        );
        self.turn_groups
            .get_mut(&case.group)
            .expect("held across the carve")
            .ledger
            .opened(case.index, view);
        work.sequence_id = view;
        Ok(())
    }

    /// A queued case that could not be started — its view not carved, or its
    /// state not claimed — with nothing else in flight to free the ground it
    /// needs: it will not seal, and the group goes on without it.
    pub(super) fn lose_queued_case(&mut self, case: GroupCaseId, why: &ConversationError) {
        let Some(group) = self.turn_groups.get_mut(&case.group) else {
            return;
        };
        group.ledger.lost_at(case.index);
        group.lost += 1;
        tracing::warn!(
            group = case.group,
            case = case.index,
            "turn group case could not be started with nothing in flight — this case \
             is lost: {why}",
        );
        self.advance_turn_group(case.group);
    }

    /// End every group carved from `parent_id`, which is going away. Each
    /// group's finished views are released here; its prefilling ones are the
    /// caller's to discard, and find no group left to tell.
    pub(super) fn abandon_turn_groups_of(&mut self, parent_id: SequenceId) {
        let ids: Vec<u64> = self
            .turn_groups
            .iter()
            .filter(|(_, g)| g.parent_id == parent_id)
            .map(|(&id, _)| id)
            .collect();
        for id in ids {
            let Some(group) = self.turn_groups.remove(&id) else {
                continue;
            };
            // Its queued cases sit on the parent's own slot until admission
            // carves them a view. Left queued with the group gone, nothing would
            // carve one, and a case's tokens would be prefilled straight into
            // whatever the parent holds next.
            self.prefill_queue
                .retain(|w| !matches!(&w.seal_action, SealAction::GroupCase(c) if c.group == id));
            for view in group.ledger.finished_views() {
                self.release_case_view(view, None);
            }
            let _ = group
                .event_tx
                .send(TurnEvent::Error(ConversationError::Channel(format!(
                    "turn group on slot {parent_id} abandoned: its slot was freed after {} of {} \
                 cases sealed",
                    group.sealed,
                    group.cases.len(),
                ))));
        }
    }

    /// Seal every case now in order, open views for the cases the window has
    /// room for, and answer the caller once every case is closed.
    fn advance_turn_group(&mut self, id: u64) {
        loop {
            let Some(group) = self.turn_groups.get_mut(&id) else {
                return;
            };
            if let Some((index, view)) = group.ledger.take_sealable() {
                let parent = group.parent_id;
                let content = std::mem::take(&mut group.cases[index]);
                let last = index + 1 == group.cases.len();
                let seal = self.seal_group_case(parent, view, content);
                // The last case's state is the one the parent carries on with;
                // every earlier case's is dropped with its view.
                self.release_case_view(view, last.then_some(parent));
                let group = self.turn_groups.get_mut(&id).expect("held across the seal");
                match seal {
                    Some(result) => {
                        group.sealed += 1;
                        group.last_seal = Some(result);
                    }
                    None => {
                        group.lost += 1;
                        tracing::warn!(
                            group = id,
                            case = index,
                            "turn group case sealed nothing — this case is lost",
                        );
                    }
                }
                continue;
            }
            if let Some(index) = group.ledger.to_open() {
                // Queued on the PARENT's slot, with no view: admission carves
                // the view when it takes the case (`begin_admitted`). Until then
                // the case holds nothing, and its priority and depth are the
                // parent's, which is what its view would read.
                let parent = group.parent_id;
                let content = &group.cases[index];
                let tokens = content.token_ids.clone();
                let tags = content.tags.clone();
                let sampling = group.sampling.clone();
                let belief = group.belief.clone();
                let case_tx = group.case_tx.clone();
                group.prefill_tokens += tokens.len();
                group.ledger.queued(index);
                self.prefill_queue.push_back(PrefillWork {
                    sequence_id: parent,
                    tokens,
                    prefill_text: String::new(),
                    user_text: String::new(),
                    tags,
                    // The case's own bounds ride its `TurnContent`, which the
                    // seal uses whole; nothing reads these for a prefill that
                    // decodes nothing.
                    user_content_start: 0,
                    user_content_end: 0,
                    assistant_content_start: 0,
                    no_think: false,
                    prefill_assistant_text: String::new(),
                    event_tx: case_tx,
                    max_decode_tokens: 0,
                    sampling,
                    submitted_at: Instant::now(),
                    reprojection: None,
                    belief,
                    seal_action: SealAction::GroupCase(GroupCaseId { group: id, index }),
                    // Every case closed its own bracket in its tokens.
                    post_decode_tokens: TokenBuffer::new(),
                    projection_offsets: Vec::new(),
                    staged_composition: None,
                    triggers: Arc::new(TriggerRegistry::new()),
                    turn_grammar: None,
                    free_tool_calls_from_penalties: false,
                    recorded_reply: None,
                });
                // An arrival is an admission opportunity.
                self.settled_since_admit = true;
                continue;
            }
            if group.ledger.complete() {
                let group = self.turn_groups.remove(&id).expect("held above");
                self.close_turn_group(id, group);
            }
            return;
        }
    }

    /// Seal one case from its own view, filed under the parent.
    fn seal_group_case(
        &mut self,
        parent: SequenceId,
        view: SequenceId,
        content: TurnContent,
    ) -> Option<SealResult> {
        // The view's own tokens begin past the prefix it borrowed.
        let block_from = self
            .turn_views
            .get(&view)
            .map(|v| v.original_borrowed.0)
            .unwrap_or(0);
        let _g = profile::span("cleanup:seal");
        self.perform_seal_and_write(parent, view, block_from, &SealAction::Turn, Some(content))
            .unwrap_or_else(|e| {
                tracing::warn!("turn group case seal failed on view {view}: {e}");
                None
            })
    }

    /// Release a case's view. Its K/V is the substrate's now, or abandoned; its
    /// model state moves onto `state_to` when given, and is dropped otherwise.
    fn release_case_view(&mut self, view: SequenceId, state_to: Option<SequenceId>) {
        let parent = self.turn_views.remove(&view).map(|v| v.parent_id);
        if let Err(e) = self.session.free_sequence(view.0) {
            tracing::warn!("failed to free turn group view {}: {}", view, e);
        }
        let state = match state_to {
            Some(parent) => self.model.move_recurrent(view.0, parent.0),
            None => self.model.release_sequence(view.0),
        };
        if let Err(e) = state {
            tracing::warn!(
                "failed to settle model state of turn group view {}: {}",
                view,
                e
            );
        }
        self.sampling_states.remove(&view);
        self.slot_tokens.remove(&view);
        let parent_conversation = parent.and_then(|p| self.slot_conversations.get(&p).cloned());
        self.retire_slot_projection_state(view, parent_conversation);
        self.purge_freed_slot_scheduling_state(view);
    }

    /// Every case is closed: hand the parent back as a finished turn leaves it,
    /// and answer the caller.
    fn close_turn_group(&mut self, id: u64, group: TurnGroup) {
        let parent = group.parent_id;
        let context_tokens = self.session.sequence_offset(parent.0).unwrap_or(0);
        let sequence_stats = self.session.get_sequence_stats(parent.0);
        // As after any turn: the slot's chunks are the substrate's now, and the
        // next turn's projection rebuilds it.
        if let Err(e) = self.session.truncate_sequence_to_blocks(parent.0, 0) {
            tracing::warn!("post-group slot truncate failed for slot {}: {}", parent, e);
        }
        if let Some(slot_state) = self.slot_projection_state.get_mut(&parent) {
            slot_state.placed_pieces.clear();
        }
        if self.slot_conversations.contains_key(&parent) {
            self.carried_beliefs
                .entry(parent)
                .or_default()
                .merge_from(&group.belief);
        }
        let total_ms = group.started.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            target: "sched",
            group = id,
            sealed = group.sealed,
            lost = group.lost,
            of = group.cases.len(),
            tokens = group.prefill_tokens,
            total_ms = total_ms as u64,
            "turn group complete",
        );
        if let Some(e) = turn_group_verdict(id, group.sealed, group.cases.len(), group.lost) {
            let _ = group.event_tx.send(TurnEvent::Error(e));
            return;
        }
        let _ = group.event_tx.send(TurnEvent::Done(TurnResponse {
            text: String::new(),
            answer: TurnText::default(),
            token_ids: TokenBuffer::new(),
            stats: TurnStats {
                prefill_ms: group.prefill_ms,
                decode_ms: 0.0,
                total_ms,
                tokens_generated: 0,
                tokens_per_second: 0.0,
                prefill_token_count: context_tokens + group.prefill_tokens,
                turn_prefill_tokens: group.prefill_tokens,
                context_tokens: context_tokens + group.prefill_tokens,
                finish: FinishReason::Length,
                sequence: sequence_stats,
            },
            seal: group.last_seal,
        }));
    }
}

/// How a closed group is answered: `None` for done, or the error naming what it
/// lost.
///
/// **A group that lost cases is not answered as complete.** Its caller keeps a
/// complete group and redoes an incomplete one — zend archives a calibration
/// group it is told is done and never regenerates it, so a lost case's
/// exemplars would be missing for good.
pub(super) fn turn_group_verdict(
    id: u64,
    sealed: usize,
    cases: usize,
    lost: usize,
) -> Option<ConversationError> {
    (lost > 0).then(|| {
        ConversationError::Channel(format!(
            "turn group {id}: {sealed} of {cases} cases sealed, {lost} lost"
        ))
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(n: usize) -> SequenceId {
        SequenceId(n)
    }

    /// A group whose every case sealed is done; one that lost any says how
    /// many of how many sealed, in the words the caller logs.
    #[test]
    fn a_group_is_done_only_when_it_lost_nothing() {
        assert!(turn_group_verdict(3, 24, 24, 0).is_none());
        let partial = turn_group_verdict(14, 23, 24, 1).expect("a lost case is not done");
        assert_eq!(
            partial.to_string(),
            "internal channel error: turn group 14: 23 of 24 cases sealed, 1 lost"
        );
    }

    /// Cases open in order and no more than the window at once.
    #[test]
    fn cases_open_in_order_within_the_window() {
        let mut ledger = CaseLedger::new(4, 2);
        assert_eq!(ledger.to_open(), Some(0));
        ledger.opened(0, v(10));
        assert_eq!(ledger.to_open(), Some(1));
        ledger.opened(1, v(11));
        assert_eq!(ledger.to_open(), None, "the window is full");
        assert_eq!(ledger.live(), 2);
    }

    /// **A queued case holds its place and blocks the seal behind it** until
    /// admission carves it a view, and a queued case lost is stepped over.
    #[test]
    fn a_queued_case_holds_its_place_until_carved_or_lost() {
        let mut ledger = CaseLedger::new(3, 2);
        ledger.queued(0);
        ledger.queued(1);
        assert!(ledger.is_queued(0));
        assert_eq!(ledger.live(), 2, "a queued case counts against the window");
        assert_eq!(ledger.to_open(), None);

        ledger.opened(1, v(11));
        ledger.finished(1, v(11));
        assert_eq!(
            ledger.take_sealable(),
            None,
            "case 0 is still queued, so case 1 waits behind it",
        );
        ledger.lost_at(0);
        assert!(!ledger.is_queued(0));
        assert_eq!(ledger.take_sealable(), Some((1, v(11))));
        assert_eq!(ledger.to_open(), Some(2), "the window has room again");
    }

    /// **Cases seal in submission order, whatever order they finish in.** A
    /// case that finishes early waits for the cases before it.
    #[test]
    fn a_case_that_finishes_early_waits_for_the_cases_before_it() {
        let mut ledger = CaseLedger::new(3, 3);
        for (i, view) in [(0, 10), (1, 11), (2, 12)] {
            ledger.opened(i, v(view));
        }
        ledger.finished(2, v(12));
        ledger.finished(1, v(11));
        assert_eq!(ledger.take_sealable(), None, "case 0 has not finished");
        ledger.finished(0, v(10));
        assert_eq!(ledger.take_sealable(), Some((0, v(10))));
        assert_eq!(ledger.take_sealable(), Some((1, v(11))));
        assert_eq!(ledger.take_sealable(), Some((2, v(12))));
        assert_eq!(ledger.take_sealable(), None);
        assert!(ledger.complete());
    }

    /// A sealed case frees its place in the window for the next case.
    #[test]
    fn a_seal_frees_a_place_in_the_window() {
        let mut ledger = CaseLedger::new(3, 1);
        ledger.opened(0, v(10));
        assert_eq!(ledger.to_open(), None);
        ledger.finished(0, v(10));
        assert_eq!(
            ledger.to_open(),
            None,
            "a finished view still holds its place"
        );
        assert_eq!(ledger.take_sealable(), Some((0, v(10))));
        assert_eq!(ledger.to_open(), Some(1));
    }

    /// A lost case is stepped over: the cases behind it still seal, and the
    /// group still completes.
    #[test]
    fn a_lost_case_is_stepped_over() {
        let mut ledger = CaseLedger::new(3, 3);
        for (i, view) in [(0, 10), (1, 11), (2, 12)] {
            ledger.opened(i, v(view));
        }
        assert_eq!(ledger.lost(v(10)), Some(0));
        assert_eq!(ledger.lost(v(99)), None, "not one of this group's views");
        ledger.lost_at(1);
        ledger.finished(2, v(12));
        assert_eq!(ledger.take_sealable(), Some((2, v(12))));
        assert!(ledger.complete());
    }

    /// A finish reported for a view the case no longer holds changes nothing —
    /// a lost case cannot come back.
    #[test]
    fn a_stale_finish_does_not_revive_a_lost_case() {
        let mut ledger = CaseLedger::new(1, 1);
        ledger.opened(0, v(10));
        ledger.lost(v(10));
        ledger.finished(0, v(10));
        assert_eq!(ledger.take_sealable(), None);
        assert!(ledger.complete());
        assert!(ledger.finished_views().is_empty());
    }

    /// Only finished views are the group's own to release when it is
    /// abandoned; a prefilling one is still the prefill queue's.
    #[test]
    fn only_finished_views_are_released_with_the_group() {
        let mut ledger = CaseLedger::new(3, 3);
        ledger.opened(0, v(10));
        ledger.opened(1, v(11));
        ledger.finished(1, v(11));
        assert_eq!(ledger.finished_views(), vec![v(11)]);
    }

    /// A window of zero still opens one case, or the group could never finish.
    #[test]
    fn a_zero_window_still_opens_one_case() {
        let ledger = CaseLedger::new(2, 0);
        assert_eq!(ledger.to_open(), Some(0));
    }
}
