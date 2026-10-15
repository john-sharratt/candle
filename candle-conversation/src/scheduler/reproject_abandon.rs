//! A reprojection that fails after taking its turn's [`DecodeState`] out of
//! [`Scheduler::active_decodes`].
//!
//! A reprojection swaps a decoding turn onto a rebuilt view in two halves —
//! prepare, then complete, with the batched gap-fill forward between — and
//! between them the turn's state belongs to the reprojection alone, in no map
//! the decode loop looks at. A failure there that only returned its error
//! dropped the state, and with it the turn's channel: the caller saw the turn
//! end with neither `Done` nor `Error`, could not tell a fault from a vanished
//! turn, and kept nothing of it. Every exit from those halves goes through
//! [`or_abandon!`] instead, which tells the turn why ([`DecodeState::fail`])
//! and discards it unsealed, like any faulted turn.

use super::*;

/// `$result`'s value, or — on its error — the turn `$state` abandoned
/// ([`Scheduler::abandon_reprojection`]) and the error returned.
macro_rules! or_abandon {
    ($sched:ident, $view:expr, $parent:expr, $state:ident, $view_live:expr, $result:expr) => {
        match $result {
            Ok(value) => value,
            Err(e) => {
                $sched.abandon_reprojection($view, $parent, $state, $view_live, &e);
                return Err(e);
            }
        }
    };
}
pub(super) use or_abandon;

impl Scheduler {
    /// Tell the turn its reprojection failed, and discard it unsealed. The
    /// caller logs the failure.
    ///
    /// `view_live` says the old view's slot is still held: the failure came
    /// before the swap freed it, so it is freed here, as a faulted turn's view
    /// is ([`Self::discard_turn_view`]). After the swap the slot is gone and its
    /// id may already be another view's, so nothing is freed under it.
    pub(super) fn abandon_reprojection(
        &mut self,
        view_id: SequenceId,
        parent_id: SequenceId,
        mut decode_state: DecodeState,
        view_live: bool,
        error: &ConversationError,
    ) {
        decode_state.fail(ConversationError::Channel(format!(
            "the reprojection failed: {error}"
        )));
        if view_live {
            if let Err(e) = self.session.free_sequence(view_id.0) {
                tracing::warn!("failed to free turn view {view_id}: {e}");
            }
            if let Err(e) = self.model.release_sequence(view_id.0) {
                tracing::warn!("failed to release model state for turn view {view_id}: {e}");
            }
            let parent_conversation = self.slot_conversations.get(&parent_id).cloned();
            self.retire_slot_projection_state(view_id, parent_conversation);
            self.purge_freed_slot_scheduling_state(view_id);
        }
        // A group's case is looked up by the view it was carved on, which the
        // group's ledger still names.
        if matches!(decode_state.seal_action, SealAction::GroupCase(_)) {
            self.lose_group_case(view_id);
        }
    }
}
