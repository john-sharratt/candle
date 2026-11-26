//! Rebuilds a slot's per-position index as a projection places it: one page
//! per piece, pushed in order, instead of the pages its own prefill forwarded
//! ([`SchedulerRequest::SealPieceIndex`], [`SchedulerRequest::ReplaceIndex`]).
//!
//! A projection hands a slot each sealed section's index as a single stored
//! page, keyed by its bytes; a slot that prefilled the same tokens itself holds
//! the pages its forwards closed, cut wherever a break token fell. Sealing each
//! piece's index right after its prefill and pushing those pages back over a
//! reset index leaves the slot indexed the way a projection would have, so a
//! slot built by prefill can be compared with a projected one page for page.

use super::*;

impl Scheduler {
    /// The index `slot` has forwarded from `start` on, as one page; `None`
    /// for a model that keeps no index.
    pub(super) fn seal_piece_index(
        &self,
        slot: SequenceId,
        start: usize,
    ) -> Result<Option<Vec<u8>>, ConversationError> {
        self.model
            .seal_positional_range(slot.0, start)
            .map_err(ConversationError::Model)
    }

    /// Replace `slot`'s index with `pages`, in order, each keyed by its bytes
    /// as a projection keys a section's page; the tokens the index covers
    /// afterwards.
    pub(super) fn replace_index(
        &self,
        slot: SequenceId,
        pages: &[Vec<u8>],
    ) -> Result<Option<usize>, ConversationError> {
        self.model
            .reset_positional_state(slot.0)
            .map_err(ConversationError::Model)?;
        for page in pages {
            self.model
                .push_positional_state(slot.0, page, &PieceKey::of(page))
                .map_err(ConversationError::Model)?;
        }
        Ok(self.model.positional_coverage(slot.0))
    }
}
