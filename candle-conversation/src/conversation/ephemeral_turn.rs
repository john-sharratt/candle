//! A turn taken against a conversation that leaves no trace in it.
//!
//! [`UnsealedAsker::ask`] runs one user turn on an ephemeral fork of a
//! conversation's live slot, taken at a turn boundary. The fork projects exactly
//! as a real turn does, so the model sees the same system prompt and the same
//! selected history, but it resolves to `SealAction::None`: neither the question
//! nor the answer is written to the substrate, the timeline's turn count does not
//! move, and the conversation's own state (selection, in-flight guard, pending
//! user turn) is untouched.
//!
//! The asker is a handle, not a borrow of the [`Sequence`]: it carries the
//! channel, tokenizers and config a turn is laid out with, and reads the
//! projection and selection the conversation last published. A caller can
//! therefore ask while the conversation's own turn is running, with no lock on
//! the conversation; the scheduler parks the fork until that turn ends.

use std::sync::{Arc, Mutex};

use super::{AskState, Sequence, TurnSubmitter, TurnText};
use crate::error::ConversationError;
use crate::handle::TurnResponse;
use crate::projection::SelectionState;
use crate::scheduler::{ProjectionInputs, SchedulerRequest};
use crate::sequence_handle::SequenceId;
use crate::turn::TurnOptions;

/// Puts questions to one conversation without borrowing it.
#[derive(Clone)]
pub struct UnsealedAsker {
    pub(super) submitter: TurnSubmitter,
    pub(super) id: SequenceId,
    pub(super) state: Arc<Mutex<AskState>>,
}

impl UnsealedAsker {
    /// The selection the conversation's last turn ran under, which is what a
    /// question asked now reads the history through.
    pub fn selection(&self) -> SelectionState {
        self.state.lock().unwrap().selection.clone()
    }

    /// Put `question` to the model as a user turn and return its reply, writing
    /// nothing back.
    ///
    /// The turn is decoded under `options` (grammar, prefill, sampling, token
    /// cap, selection) like any other. The question runs on an ephemeral fork of
    /// the conversation's live slot, taken at a turn boundary: the fork borrows
    /// the slot's K/V and copies its recurrent state and projection caches, so the
    /// projection keeps the prefix already placed instead of rebuilding it. When a
    /// turn is running on the conversation the fork waits for it to end.
    pub async fn ask(&self, question: &str, options: TurnOptions) -> crate::Result<TurnResponse> {
        let (tx, rx) = flume::bounded(1);
        self.submitter
            .scheduler_tx
            .send(SchedulerRequest::ForkEphemeralSequence {
                parent: self.id,
                response_tx: tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        let slot = rx
            .recv_async()
            .await
            .map_err(|_| ConversationError::SchedulerGone)??;
        let _free = FreeOnDrop {
            tx: self.submitter.scheduler_tx.clone(),
            slot,
        };

        let handle = {
            let inputs = ProjectionInputs {
                projection: Arc::clone(&self.state.lock().unwrap().projection),
                selection: options.selection.clone(),
            };
            let message = TurnText::from(question);
            self.submitter
                .submit_turn_on(slot, inputs, &message, question.to_string(), options, None)?
                .0
        };
        handle.wait_async().await
    }
}

impl Sequence {
    /// [`UnsealedAsker::ask`] for a caller that holds the conversation.
    pub async fn ask_unsealed(
        &mut self,
        question: &str,
        options: TurnOptions,
    ) -> crate::Result<TurnResponse> {
        let asker = self.unsealed_asker();
        asker.ask(question, options).await
    }
}

/// Releases the ephemeral slot however the ask ends, including a dropped future.
struct FreeOnDrop {
    tx: flume::Sender<SchedulerRequest>,
    slot: SequenceId,
}

impl Drop for FreeOnDrop {
    fn drop(&mut self) {
        let _ = self.tx.send(SchedulerRequest::FreeSequence {
            sequence_id: self.slot,
        });
    }
}
