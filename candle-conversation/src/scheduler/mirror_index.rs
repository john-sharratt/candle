//! Rebuilds a slot's per-position index as a projection places it, piece by
//! piece: one sealed page per piece, every earlier piece's page injected
//! ahead of it ([`SchedulerRequest::IndexPiece`]).
//!
//! A section is sealed on a scratch slot whose prefix arrives as injected
//! pages; its own tokens are forwarded behind them and sealed, on a block
//! boundary, as one page — the page a projection later pushes, keyed by its
//! bytes. A slot that prefills the same pieces itself holds the pages its
//! forwards closed instead, cut wherever a break token fell. Sealing each
//! piece's page once it is prefilled, and putting the slot's index back as
//! every page so far, injected, leaves the slot indexed as a projection
//! would have placed it before the next piece is forwarded over it.

use super::*;

impl Scheduler {
    /// Seal the index `slot` has forwarded since its index was last placed as
    /// one page, then place the index again as `placed` followed by that
    /// page, each pushed in order and keyed by its bytes. The new page;
    /// `None` for a model that keeps no index.
    pub(super) fn index_piece(
        &self,
        slot: SequenceId,
        placed: &[Vec<u8>],
    ) -> Result<Option<Vec<u8>>, ConversationError> {
        let Some(page) = self
            .model
            .seal_positional_state(slot.0)
            .map_err(ConversationError::Model)?
        else {
            return Ok(None);
        };
        self.model
            .reset_positional_state(slot.0)
            .map_err(ConversationError::Model)?;
        for p in placed.iter().chain(std::iter::once(&page)) {
            self.model
                .push_positional_state(slot.0, p, &PieceKey::of(p))
                .map_err(ConversationError::Model)?;
        }
        Ok(Some(page))
    }
}
