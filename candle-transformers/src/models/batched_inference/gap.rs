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

use std::time::Instant;

use candle::quantized::cuda::kvhead_supported_head_dim;
use candle::{Device, Result};
use candle_nn::kv_cache::{gap_jobs_within, GapCatchUp, GapCycle, GapKind, GapQuantize};

use super::BatchedInferenceSession;

/// The device scratch one gap's table runs in — its selection blocks, convert
/// buffer and inputs, held for the session's life (§3.1). The table is cut to
/// the jobs — blocks × layers — whose scratch fits ([`gap_jobs_within`]):
/// 3,866 at 4 KV heads of 128, so an 8K-token prompt on a 48-layer model
/// (12,288 jobs) is quantized over four gaps; the rest of a table goes to the
/// next gap.
pub const GAP_SCRATCH_BYTES: usize = 64 << 20;

impl BatchedInferenceSession {
    /// Make the gap's state, once, for a session that quantizes: its scratch
    /// and summary are allocated here, never between two forwards.
    pub(super) fn ensure_gap(&mut self) -> Result<()> {
        if self.gap.is_some() || !self.gaps_quantize() {
            return Ok(());
        }
        let first = &self.backings[0];
        let max_jobs = gap_jobs_within(GAP_SCRATCH_BYTES, first.n_kv_head(), first.head_dim());
        self.gap = Some(GapCycle::new(&self.backings, max_jobs)?);
        Ok(())
    }

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
        let t_setup = Instant::now();
        self.ensure_gap()?;
        let setup = t_setup.elapsed();
        let cycle = self.gap.as_mut().expect("made above");
        let mut q = cycle.quantize(&self.backings, &seqs, &policies, kind)?;
        q.host.setup = setup;
        Ok(q)
    }

    /// Whether a gap is outstanding — launched and not yet caught up.
    pub fn gap_outstanding(&self) -> bool {
        self.gap.as_ref().is_some_and(GapCycle::is_outstanding)
    }
}
