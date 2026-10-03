//! An ephemeral fork of a live conversation's slot, taken between its turns.
//!
//! A question put to a character that must leave no trace
//! ([`crate::Sequence::ask_unsealed`]) used to run on a fresh slot: empty K/V, no
//! placed-piece record, no glue cache, so the whole system prompt and history
//! were projected and prefilled again for every question. The fork starts from
//! what the live slot already holds instead. Its K/V is a zero-copy borrow of the
//! parent's blocks (the carve a turn's own view uses), its recurrent and
//! per-position state is copied, and its projection caches are cloned, so the
//! projection the question runs under keeps the prefix the parent placed and
//! re-injects only from the first piece that differs.
//!
//! The fork is a slot in its own right, not a turn view: it is registered
//! ephemeral (its turn seals nothing), carries the parent's conversation, target
//! and carried belief, and is freed by an ordinary `FreeSequence`, which drops its
//! references to the parent's blocks and leaves the parent untouched.
//!
//! A request that arrives while a turn runs on the parent is parked and answered
//! at the turn boundary, so a caller needs no lock on the conversation to ask: it
//! gets the state as of the end of the turn in flight.

use flume::Sender;

use super::Scheduler;
use crate::error::ConversationError;
use crate::sequence_handle::SequenceId;

/// A fork request waiting for its parent's turn to end.
pub(super) struct ParkedFork {
    parent: SequenceId,
    response_tx: Sender<Result<SequenceId, ConversationError>>,
}

impl Scheduler {
    /// Whether a turn is in flight on `parent`: a view carved from it is alive.
    fn turn_in_flight(&self, parent: SequenceId) -> bool {
        self.turn_views.values().any(|v| v.parent_id == parent)
    }

    /// Answer a fork request now, or park it until the turn on `parent` ends.
    pub(super) fn request_ephemeral_fork(
        &mut self,
        parent: SequenceId,
        response_tx: Sender<Result<SequenceId, ConversationError>>,
    ) {
        if self.slot_conversations.contains_key(&parent) && self.turn_in_flight(parent) {
            self.parked_forks.push(ParkedFork {
                parent,
                response_tx,
            });
            return;
        }
        let _ = response_tx.send(self.fork_ephemeral_sequence(parent));
    }

    /// Answer every parked request whose parent has no turn in flight, and fail
    /// those whose parent is gone. A requester that stopped waiting is dropped
    /// without forking, so an abandoned ask costs no slot.
    pub(super) fn drain_parked_forks(&mut self) {
        if self.parked_forks.is_empty() {
            return;
        }
        let parked = std::mem::take(&mut self.parked_forks);
        for fork in parked {
            if fork.response_tx.is_disconnected() {
                continue;
            }
            if self.slot_conversations.contains_key(&fork.parent)
                && self.turn_in_flight(fork.parent)
            {
                self.parked_forks.push(fork);
                continue;
            }
            let _ = fork
                .response_tx
                .send(self.fork_ephemeral_sequence(fork.parent));
        }
    }

    /// Fork `parent` into an ephemeral slot.
    ///
    /// Refused while a turn is in flight on `parent`: its view is borrowing the
    /// blocks the fork would carve, and what the slot holds mid-turn is not a
    /// state anything should fork. Callers go through
    /// [`Self::request_ephemeral_fork`], which waits instead.
    pub(super) fn fork_ephemeral_sequence(
        &mut self,
        parent: SequenceId,
    ) -> Result<SequenceId, ConversationError> {
        let conversation = self
            .slot_conversations
            .get(&parent)
            .cloned()
            .ok_or(ConversationError::SequenceNotFound(parent.0))?;
        if self.turn_in_flight(parent) {
            return Err(ConversationError::TurnInFlight {
                sequence_id: parent,
            });
        }

        let (child, _borrowed) = self.create_view(parent, &[])?;

        self.slot_conversations.insert(child, conversation);
        if let Some(target) = self.slot_targets.get(&parent).copied() {
            self.slot_targets.insert(child, target);
        }
        if let Some(belief) = self.carried_beliefs.get(&parent).cloned() {
            self.carried_beliefs.insert(child, belief);
        }
        if let Some(state) = self.slot_projection_state.get(&parent) {
            let forked = state.fork();
            self.slot_projection_state.insert(child, forked);
        }
        self.ephemeral_slots.insert(child);
        Ok(child)
    }
}
