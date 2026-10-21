//! What a splice of a slot's per-position state puts in place — the argument to
//! `ManagedBatchedModel::splice_positional_state`.
//!
//! A rebuild that changes one piece in the middle of a slot splices that
//! piece's K/V into place (`BatchedInferenceSession::splice_sealed`) and its
//! per-position state alongside it. The state comes in the same two kinds a
//! piece injected at the tail brings: a sealed page, or a span its K/V covers
//! that nothing indexed.

use std::sync::Arc;

use super::piece_key::PieceKey;

/// One piece of per-position state, in slot order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PositionalPiece {
    /// A sealed page, the counterpart of `push_positional_state`: its bytes and
    /// the key its holder took when the page entered memory.
    Page { bytes: Arc<[u8]>, key: PieceKey },
    /// Tokens of injected K/V that carry no rows, the counterpart of
    /// `push_positional_gap`.
    Gap(usize),
}

impl PositionalPiece {
    /// A page of `bytes`, keyed by them.
    pub fn page(bytes: &[u8]) -> Self {
        Self::Page {
            bytes: Arc::from(bytes),
            key: PieceKey::of(bytes),
        }
    }
}
