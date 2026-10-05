//! A wave step's scored rows, read where the head wrote them.
//!
//! The head carves its logits from the forward's span, and the model hands the
//! span's guard back inside the [`WaveResult`] so the rows stay valid while the
//! caller reads them. Copying them off the span to outlive the result cost one
//! `[rows, vocab]` allocation and copy per wave — and the accept walk then
//! stacked each block position into another. Held instead, the rows are read
//! in place and the walk's stacks land on the same span, priced there
//! (`WaveBuffer::AcceptRows`).
//!
//! The holder drops the guard when it drops, so a caller keeps it for exactly
//! as long as it reads the rows and drops it before anything that can run the
//! next forward — a rollback can.

use candle::Tensor;
use candle_transformers::models::batched_inference::WaveResult;

/// The rows a wave step hands back, and the result that keeps them valid.
pub(crate) struct WaveLogits {
    /// One row per scored position the caller asked for, as views on the
    /// forward's span while [`Self`] lives.
    pub rows: Vec<Tensor>,
    /// Held, never read: dropping it reclaims the span the rows name.
    _held: Option<WaveResult>,
}

impl WaveLogits {
    /// No rows — a step that ran no head, or ran nothing.
    pub fn empty() -> Self {
        Self {
            rows: Vec::new(),
            _held: None,
        }
    }

    /// The first `keep` of `result`'s scored rows, held behind it.
    pub fn held(result: WaveResult, keep: usize) -> Self {
        let mut rows = result.logits_on_span();
        rows.truncate(keep);
        Self {
            rows,
            _held: Some(result),
        }
    }

    /// `rows` already taken from `result`, held behind it.
    pub fn held_rows(result: WaveResult, rows: Vec<Tensor>) -> Self {
        Self {
            rows,
            _held: Some(result),
        }
    }
}
