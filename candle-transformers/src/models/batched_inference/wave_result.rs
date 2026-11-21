//! Wraps a forward's outputs with the guard that keeps the device span they sit
//! on from being reclaimed while they are read.
//!
//! The head's outputs are carved from a forward-scoped span that is reclaimed
//! when its generation drops, so the guard travels with the result and
//! dereferences to the step. Reading through it borrows the guard, and moving the
//! logits out is a compile error, which flags every site that would have taken
//! the tensors away from their span. A caller that must keep the values past the
//! span copies them with `logits_owned`, which joins every row into one
//! allocation and makes one copy.

use super::*;

/// Result of a wave forward.
///
/// `logits` when it ran the head (one row per scored input sequence), `None`
/// when it stopped short — whose residual went to the caller's
/// [`CreepResidual`] — or carried only glue rows, which score nothing.
pub struct WaveStep {
    /// Per-sequence logits (in `seq_indices` order), when the range reached the head.
    pub logits: Option<Vec<Tensor>>,
}

/// The scored rows a verify step hands its accept walk: the plain cohort's row
/// each, then every block's rows by position.
///
/// When one wave produced them they are views on its head's span and `head` is
/// that wave, held so the views stay valid — the walk reads them in place and
/// drops this before the rewind opens anything. `None` when they are owned
/// copies (a path that ran several forwards).
pub struct VerifiedRows {
    pub plain: Vec<Tensor>,
    pub blocks: Vec<Vec<Tensor>>,
    pub head: Option<WaveResult>,
}

/// A [`WaveStep`] together with what keeps it valid.
///
/// The head's outputs are carved from the forward-scoped span, which is
/// reclaimed when its generation drops. Returning the step alone would hand the
/// caller tensors whose memory is reusable the moment the callee returns — sound
/// only by the convention that everyone samples before the next forward, which
/// nothing checks.
///
/// So the guard travels **with** the result. The span cannot be reclaimed while
/// this value is alive, and [`Deref`](std::ops::Deref) is what makes that
/// checkable rather than merely true: reading through it borrows `self`, so a
/// reference to the logits keeps the guard alive, while *moving* the logits out
/// is rejected — you cannot move out of a `Deref`. Every site that only reads
/// compiles unchanged; every site that would have taken the tensors away from
/// their guard is a compile error, which is exactly the set worth looking at.
pub struct WaveResult {
    pub(super) step: WaveStep,
    /// Held, never read. `None` off-CUDA and for waves that never opened a
    /// forward span.
    #[cfg(feature = "cuda")]
    pub(super) _forward: Option<candle_nn::kv_cache::WaveGeneration>,
}

impl WaveResult {
    /// Wrap a step whose outputs do not sit on a forward span.
    pub fn owned(step: WaveStep) -> Self {
        Self {
            step,
            #[cfg(feature = "cuda")]
            _forward: None,
        }
    }

    /// Wrap a step whose outputs were carved from `forward`'s span.
    #[cfg(feature = "cuda")]
    pub fn on_span(step: WaveStep, forward: Option<candle_nn::kv_cache::WaveGeneration>) -> Self {
        Self {
            step,
            _forward: forward,
        }
    }

    /// The logits, copied off the span so they outlive this result.
    ///
    /// The sanctioned escape, and it really copies. Prefer reading through
    /// [`Deref`](std::ops::Deref) — `result.logits` — which costs nothing and
    /// keeps the guard doing its job. This exists for callers that genuinely
    /// need the values after the span is reclaimed: a caller that returns them
    /// upward, or one that accumulates across several forwards.
    ///
    /// Empty when the wave paused before the head: a glue-only wave carries no
    /// logits and that is not a failure. A *failed copy* is a different thing and
    /// propagates — the two used to collapse into the same empty `Vec`, which the
    /// scheduler reads as "this wave produced nothing" and turns into a silently
    /// token-less turn.
    ///
    /// **One copy for every row, not one per row.** The rows are the head's
    /// logits block, so they are joined along their leading axis straight into
    /// one owned allocation (`Tensor::cat_owned`, which copies a run of adjacent
    /// views as one range — one launch when they are that block's rows in order)
    /// and handed back as views of it in the same shapes: one allocation and one
    /// copy however many rows the wave scored. A caller that keeps one row past
    /// the others takes it with `to_owned_tensor`, which copies the row alone;
    /// `copy` would clone the whole block it views.
    pub fn logits_owned(&self) -> Result<Vec<Tensor>> {
        let Some(ls) = self.step.logits.as_ref() else {
            return Ok(Vec::new());
        };
        if ls.is_empty() {
            return Ok(Vec::new());
        }
        let block = Tensor::cat_owned(ls, 0)?;
        let mut start = 0;
        let mut out = Vec::with_capacity(ls.len());
        for t in ls {
            let n = t.dim(0)?;
            out.push(block.narrow(0, start, n)?);
            start += n;
        }
        Ok(out)
    }

    /// The scored rows as views on the head's span, for reading **while this
    /// result is held** — the counterpart of [`Self::logits_owned`] that copies
    /// nothing. The views name memory the span reclaims when this result drops,
    /// so the caller keeps the result alive for as long as it reads them; the
    /// verify path does exactly that, carrying it beside the rows
    /// ([`crate::models::verify_wave::VerifyWaveOutput`]) until the accept walk
    /// is done.
    pub fn logits_on_span(&self) -> Vec<Tensor> {
        self.step.logits.clone().unwrap_or_default()
    }
}

impl std::ops::Deref for WaveResult {
    type Target = WaveStep;

    fn deref(&self) -> &WaveStep {
        &self.step
    }
}
