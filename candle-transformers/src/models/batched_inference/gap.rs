//! The session's gap quantize (`docs/kv_quantize_in_the_gap.md` §3): the
//! scheduler calls [`BatchedInferenceSession::gap_catch_up`] at the head of the
//! gap between forwards, its own arena creation and compaction next, then
//! [`BatchedInferenceSession::gap_quantize`].
//!
//! Every layer seals under its own policy ([`BatchedInferenceSession::
//! layer_seal_policies`]), as today's seal does, so the gap's groups and the
//! seal's agree. A session that does not quantize — F16/BF16 — and a geometry
//! the convert has no record for (a head_dim other than 128 or 256, or the
//! single-latent layout, which has its own per-band compressor) take no gap.

use candle::quantized::cuda::kvhead_supported_head_dim;
use candle::{Device, Result};
use candle_nn::kv_cache::{GapCatchUp, GapCycle, GapKind, GapQuantize};

use super::BatchedInferenceSession;

/// The most jobs — blocks × layers — one gap's table holds. An 8K-token
/// prompt's prefill gap on a 48-layer model is 12,288; a table past it is cut,
/// and the rest goes to the next gap. Sizes the summary in mapped memory:
/// `jobs × n_kv_head × (head_dim / 2 + 136)` bytes, 13 MB at 4 heads of 128.
pub const GAP_MAX_JOBS: usize = 16_384;

impl BatchedInferenceSession {
    /// Whether this session's K/V is quantized in the gap.
    pub fn gaps_quantize(&self) -> bool {
        matches!(self.device, Device::Cuda(_))
            && self.seal_policy().is_some()
            && self
                .backings
                .first()
                .is_some_and(|b| kvhead_supported_head_dim(b.head_dim()) && !b.single_latent())
    }

    /// Put the outstanding gap's converted blocks in place, once it has
    /// completed — the head of every gap. Nothing when no gap is outstanding,
    /// when it still runs, or when the session does not quantize.
    pub fn gap_catch_up(&mut self) -> Result<GapCatchUp> {
        match self.gap.as_mut() {
            Some(cycle) => cycle.catch_up(&self.backings),
            None => Ok(GapCatchUp::default()),
        }
    }

    /// Quantize every live sequence's complete blocks past its cursor, the
    /// retried blocks first — after the gap's arena creation and compaction.
    /// `kind` keeps prefill and decode gaps' shares apart. Nothing while a gap
    /// is outstanding or when the session does not quantize.
    pub fn gap_quantize(&mut self, kind: GapKind) -> Result<GapQuantize> {
        if !self.gaps_quantize() {
            return Ok(GapQuantize::default());
        }
        let Some(session_policy) = self.seal_policy() else {
            return Ok(GapQuantize::default());
        };
        let policies = self.layer_seal_policies(&session_policy);
        let seqs: Vec<usize> = self
            .sequences
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|_| i))
            .collect();
        if self.gap.is_none() {
            self.gap = Some(GapCycle::new(&self.backings, GAP_MAX_JOBS)?);
        }
        let cycle = self.gap.as_mut().expect("made above");
        cycle.quantize(&self.backings, &seqs, &policies, kind)
    }

    /// Whether a gap is outstanding — launched and not yet caught up.
    pub fn gap_outstanding(&self) -> bool {
        self.gap.as_ref().is_some_and(GapCycle::is_outstanding)
    }
}
