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
//! # A window of views
//!
//! Each view holds a forked copy of the parent's recurrent state for as long as
//! it lives — on Qwen3.6-35B-A3B about 126 MiB — so a group does not carve
//! every case at once. At most [`Scheduler::turn_group_window`] views are live;
//! each seal frees a place for the next case.

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
    /// No view carved yet.
    Waiting,
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

    /// Views the group holds now.
    fn live(&self) -> usize {
        self.states
            .iter()
            .filter(|s| matches!(s, CaseState::Prefilling(_) | CaseState::Finished(_)))
            .count()
    }

    /// The next case to carve a view for, while the window has room. Cases open
    /// in order, so the case that must seal next always holds a view.
    fn to_open(&self) -> Option<usize> {
        if self.live() >= self.window {
            return None;
        }
        self.states.iter().position(|s| *s == CaseState::Waiting)
    }

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
                CaseState::Waiting | CaseState::Prefilling(_) => return None,
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
    prefill_ms: f64,
    prefill_tokens: usize,
    started: Instant,
}

impl Scheduler {
    /// Views a turn group holds at once.
    ///
    /// Each holds a forked recurrent store for as long as it lives, so the
    /// window is what the admission setpoint funds in stores — the same budget
    /// that decides how much prefill may be in flight — capped by
    /// [`Self::MAX_PREFILL_WIDTH`], past which no wave could carry more of them
    /// anyway. A model with no recurrent state has nothing to bound it but the
    /// cap.
    fn turn_group_window(&self) -> usize {
        let per_view = self.model.recurrent_store_bytes() as u64;
        if per_view == 0 {
            return Self::MAX_PREFILL_WIDTH;
        }
        ((self.admit_budget / per_view) as usize).clamp(1, Self::MAX_PREFILL_WIDTH)
    }

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
        let ledger = CaseLedger::new(cases.len(), self.turn_group_window());
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
        let group = &self.turn_groups[&id];
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
                    None => tracing::warn!(
                        group = id,
                        case = index,
                        "turn group case sealed nothing — this case is lost",
                    ),
                }
                continue;
            }
            if let Some(index) = group.ledger.to_open() {
                let parent = group.parent_id;
                let ranges = group.ranges.clone();
                let content = &group.cases[index];
                let tokens = content.token_ids.clone();
                let tags = content.tags.clone();
                let sampling = group.sampling.clone();
                let belief = group.belief.clone();
                let case_tx = group.case_tx.clone();
                group.prefill_tokens += tokens.len();
                match self.create_view(parent, &ranges) {
                    Ok((view, borrowed)) => {
                        self.turn_views.insert(
                            view,
                            ViewState {
                                parent_id: parent,
                                original_borrowed: borrowed,
                                turn_start_parent_blocks: borrowed.0,
                                question_tokens: 0,
                                applied_identity: None,
                            },
                        );
                        self.prefill_queue.push_back(PrefillWork {
                            sequence_id: view,
                            tokens,
                            prefill_text: String::new(),
                            user_text: String::new(),
                            tags,
                            // The case's own bounds ride its `TurnContent`, which
                            // the seal uses whole; nothing reads these for a
                            // prefill that decodes nothing.
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
                        self.settled_since_admit = true;
                        self.turn_groups
                            .get_mut(&id)
                            .expect("held across the carve")
                            .ledger
                            .opened(index, view);
                    }
                    Err(e) => {
                        tracing::warn!(
                            group = id,
                            case = index,
                            "turn group case could not be carved a view: {e}",
                        );
                        self.turn_groups
                            .get_mut(&id)
                            .expect("held across the carve")
                            .ledger
                            .lost_at(index);
                    }
                }
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
            of = group.cases.len(),
            tokens = group.prefill_tokens,
            total_ms = total_ms as u64,
            "turn group complete",
        );
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

#[cfg(test)]
mod tests {
    use super::*;

    fn v(n: usize) -> SequenceId {
        SequenceId(n)
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
