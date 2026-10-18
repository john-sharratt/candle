//! A reprojected turn waiting for the ground to carve its new view.
//!
//! A reprojection's last claim is the rebuilt view's fork of its parent's
//! recurrent state (`create_view`), made after the gap-fill forward and after the
//! turn's captured tail is back on the parent. By then the parent holds
//! everything the turn needs and its old view is gone, so a fork refused for
//! ground used to abandon the turn. Nothing about the turn was wrong; the span
//! was full for a moment. Measured on the RTX 3090 (2026-10-18): a reply ended
//! "no region for recurrent state after 20 of 36 slots", with the weight side at
//! its floor, while a check made before the gap-fill forward had seen room.
//! Ground priced at one moment and spent at another always disagree a little
//! under load, so the turn is parked at the point of use instead: kept whole
//! beside its parent, and carved on a later pass of the loop when relief, a
//! finishing turn or the settle has handed the ground back.

use std::time::{Duration, Instant};

use super::decode::is_ground_refusal;
use super::*;

/// How long a turn may wait for its view before it fails with the refusal. Long
/// against a turn finishing and its ground coming back, short against a caller
/// looking at a reply that stopped.
const OWED_VIEW_PATIENCE: Duration = Duration::from_secs(60);

/// The loop's wait between carve attempts when nothing else is running.
pub(super) const OWED_VIEW_RETRY: Duration = Duration::from_millis(50);

/// A reprojected turn whose new view is not carved yet.
pub(super) struct OwedView {
    /// The view the turn decoded on before the rebuild; already freed, kept for
    /// the log and [`Scheduler::abandon_reprojection`].
    pub view_id: SequenceId,
    pub parent_id: SequenceId,
    pub decode_state: DecodeState,
    pub sampling_state: Option<SequenceSamplingState>,
    /// [`ViewState::turn_start_parent_blocks`] for the view to come.
    pub turn_start_parent_blocks: usize,
    pub question_tokens: usize,
    pub applied_identity: Option<u64>,
    /// When the first carve was refused.
    pub since: Instant,
}

impl Scheduler {
    /// Carve `owed`'s view and re-bind its turn to it. Answers the new view, or
    /// gives `owed` back when the span refused the fork for ground, for a later
    /// pass to try again. Any other failure abandons the turn.
    pub(super) fn carve_owed_view(
        &mut self,
        owed: OwedView,
    ) -> Result<Result<SequenceId, OwedView>, ConversationError> {
        let parent_block_count = self
            .session
            .sequence_block_count(owed.parent_id.0)
            .unwrap_or(0);
        let ranges: Vec<BlockRange> = if parent_block_count == 0 {
            Vec::new()
        } else {
            vec![BlockRange::new(0, parent_block_count)]
        };
        let (new_view_id, borrowed) = match self.create_view(owed.parent_id, &ranges) {
            Ok(carved) => carved,
            Err(e) if is_ground_refusal(&e) && owed.since.elapsed() < OWED_VIEW_PATIENCE => {
                return Ok(Err(owed));
            }
            Err(e) => {
                let OwedView {
                    view_id,
                    parent_id,
                    decode_state,
                    ..
                } = owed;
                self.abandon_reprojection(view_id, parent_id, decode_state, false, &e);
                return Err(e);
            }
        };
        // The new view's slot may have been recycled from the freed old view's.
        self.slot_tokens.remove(&new_view_id);
        self.turn_views.insert(
            new_view_id,
            ViewState {
                parent_id: owed.parent_id,
                original_borrowed: borrowed,
                // The turn's content starts where the rebuilt prefix ends — see
                // `reproject_view_complete`.
                turn_start_parent_blocks: owed.turn_start_parent_blocks,
                question_tokens: owed.question_tokens,
                applied_identity: owed.applied_identity,
                state_forked: true,
            },
        );
        // `create_view` seeded a fresh sampling state from the parent; the live
        // one carried across keeps DRY history and per-turn counters.
        self.active_decodes.insert(new_view_id, owed.decode_state);
        if let Some(state) = owed.sampling_state {
            self.sampling_states.insert(new_view_id, state);
        }
        Ok(Ok(new_view_id))
    }

    /// Try every owed view again; the ones still refused stay owed. A turn whose
    /// caller stopped listening is abandoned rather than carved.
    pub(super) fn retry_owed_views(&mut self) {
        if self.owed_views.is_empty() {
            return;
        }
        for owed in std::mem::take(&mut self.owed_views) {
            if owed.decode_state.event_tx.is_disconnected() {
                let e = ConversationError::Channel("the caller stopped waiting".into());
                let OwedView {
                    view_id,
                    parent_id,
                    decode_state,
                    ..
                } = owed;
                self.abandon_reprojection(view_id, parent_id, decode_state, false, &e);
                continue;
            }
            let (view_id, waited) = (owed.view_id, owed.since.elapsed());
            match self.carve_owed_view(owed) {
                Ok(Ok(new_view_id)) => tracing::info!(
                    target: "candle_conversation::scheduler::reproject",
                    from_view = view_id.0,
                    to_view = new_view_id.0,
                    waited_ms = waited.as_millis() as u64,
                    "reprojected turn carved its view after waiting for ground"
                ),
                Ok(Err(owed)) => self.owed_views.push(owed),
                Err(e) => tracing::warn!(
                    target: "candle_conversation::scheduler::reproject",
                    view_id = view_id.0,
                    waited_ms = waited.as_millis() as u64,
                    err = %e,
                    "reprojected turn could not carve its view; decode for this view ends"
                ),
            }
        }
    }

    /// Whether a reprojected turn on `parent` is waiting for its view.
    pub(super) fn owes_view(&self, parent: SequenceId) -> bool {
        self.owed_views.iter().any(|o| o.parent_id == parent)
    }
}
