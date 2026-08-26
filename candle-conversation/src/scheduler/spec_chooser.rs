//! Choosing a speculative step's tokens with the scheduler's own sampler.
//!
//! The speculative driver scores rows; this decides what each row commits. It
//! is the production sampler — temperature, nucleus truncation, repetition and
//! DRY penalties, the EOT ramp, grammar stencils — applied to a verify block's
//! rows exactly as it would be applied to a plain decode row, which is what
//! makes speculation draw from the same distribution plain decoding would.
//!
//! # Why sampling each row is enough
//!
//! Every drafter here proposes greedily, so its proposal distribution is a point
//! mass and the textbook accept/reject rule collapses to "sample the row, accept
//! the proposal iff the sample agrees" — the argument is in
//! `candle_transformers::models::speculative_choice`. That is the entire reason
//! this type can be a thin wrapper over [`BatchedSampler::sample_batch`] instead
//! of a bespoke accept/reject kernel: there is nothing to compute beyond the
//! sample the sampler was already going to draw.
//!
//! # History comes from the sampler's own state
//!
//! Repetition and DRY penalties at block position `j` must see the tokens
//! committed at `0..j`. They do, without anything here replaying them: the
//! driver walks positions in order, `sample_batch` records each sampled token
//! into the sequence's [`SequenceSamplingState`], and a sequence that leaves the
//! walk is simply absent from later positions. So the state advances along
//! exactly the committed path and no further. The `prefix` each row carries is
//! used to *check* that — a state that has not advanced by one token per
//! position has desynced, and silently mispriced penalties are the kind of bug
//! that reads as a mysterious quality regression rather than a failure.

use candle::{IndexOp, Result, Tensor};
use candle_transformers::models::speculative_choice::{SpecRow, TokenChooser};

use crate::batched_sampler::{BatchedSampler, SequenceSamplingState};
use crate::config::SamplingConfig;

/// The scheduler's [`TokenChooser`].
///
/// Owns the cohort's sampling states for the duration of one speculative step —
/// they are lifted out of the scheduler's map the same way the plain decode path
/// lifts them, so the sampler can borrow them mutably — and hands them back
/// through [`Self::into_states`].
pub(super) struct SpecChooser<'a> {
    sampler: &'a BatchedSampler,
    /// Per cohort index, in the order the driver was given its sequences.
    states: Vec<SequenceSamplingState>,
    configs: Vec<SamplingConfig>,
    /// Per cohort index, the logits row that produced each committed token, in
    /// position order. The decode-health checks read the distribution a token
    /// was drawn from, so a multi-token step has to keep one row per token
    /// rather than one per sequence.
    rows: Vec<Vec<Tensor>>,
    /// Tokens committed per cohort index so far this step — the counter the
    /// desync check above compares against a row's `prefix`.
    committed: Vec<usize>,
}

impl<'a> SpecChooser<'a> {
    pub(super) fn new(
        sampler: &'a BatchedSampler,
        states: Vec<SequenceSamplingState>,
        configs: Vec<SamplingConfig>,
    ) -> Self {
        let n = states.len();
        Self {
            sampler,
            states,
            configs,
            rows: vec![Vec::new(); n],
            committed: vec![0; n],
        }
    }

    /// The logits rows that produced each sequence's committed tokens, in
    /// position order, to hand to the per-token decode-health checks.
    pub(super) fn rows(&self) -> &[Vec<Tensor>] {
        &self.rows
    }

    /// Give the sampling states back so the scheduler can reinsert them.
    pub(super) fn into_states(self) -> Vec<SequenceSamplingState> {
        self.states
    }
}

impl TokenChooser for SpecChooser<'_> {
    fn choose(&mut self, logits: &Tensor, rows: &[SpecRow<'_>]) -> Result<Vec<u32>> {
        if rows.is_empty() {
            return Ok(Vec::new());
        }
        // The walk hands live sequences over in ascending cohort order, which is
        // what lets `iter_mut().enumerate().filter(..)` below produce mutable
        // borrows lined up with `rows`. A caller that reordered them would pair
        // each row with the wrong sequence's penalties and RNG.
        if rows.windows(2).any(|w| w[0].seq >= w[1].seq) {
            candle::bail!("SpecChooser: rows are not in ascending cohort order");
        }
        for r in rows {
            if r.seq >= self.states.len() {
                candle::bail!(
                    "SpecChooser: row names sequence {} of a {}-sequence cohort",
                    r.seq,
                    self.states.len()
                );
            }
            // One token committed per position walked, or the penalties this row
            // is about to be priced with are stale.
            if self.committed[r.seq] != r.prefix.len() {
                candle::bail!(
                    "SpecChooser: sequence {} has committed {} tokens this step but its row \
                     at position {} carries a {}-token prefix — the sampling state has \
                     desynced from the accept walk",
                    r.seq,
                    self.committed[r.seq],
                    r.position,
                    r.prefix.len()
                );
            }
        }

        let live: Vec<usize> = rows.iter().map(|r| r.seq).collect();
        let mut states: Vec<&mut SequenceSamplingState> = self
            .states
            .iter_mut()
            .enumerate()
            .filter(|(i, _)| live.contains(i))
            .map(|(_, s)| s)
            .collect();
        let configs: Vec<&SamplingConfig> = live.iter().map(|&i| &self.configs[i]).collect();
        // `sample_batch` records each sampled token into its sequence's state,
        // so the next position is priced against this one.
        let tokens = self.sampler.sample_batch(logits, &mut states, &configs)?;

        // Keep the row each token was drawn from, shaped like a plain decode
        // step's row (`[1, vocab]`) so the health checks read it identically.
        for (m, r) in rows.iter().enumerate() {
            self.rows[r.seq].push(logits.i(m..m + 1)?);
            self.committed[r.seq] += 1;
        }
        Ok(tokens)
    }
}
