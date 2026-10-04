//! Choosing a speculative step's tokens with the scheduler's own sampler.
//!
//! The speculative driver scores rows; this decides what each row commits. It
//! is the production sampler — temperature, nucleus truncation, repetition and
//! DRY penalties, the EOT ramp, grammar stencils — applied to a verify block's
//! rows exactly as it would be applied to a plain decode row. A row commits its
//! sample unless typical acceptance (below) takes its draft instead, so the
//! only departure from what plain decoding would draw is the drafts that rule
//! lets through; at temperature zero there is none, and the output is plain
//! greedy decode token for token.
//!
//! # Why sampling each row is enough
//!
//! Every drafter here proposes greedily, so its proposal distribution is a point
//! mass and the textbook accept/reject rule collapses to "sample the row, accept
//! the proposal iff the sample agrees" — the argument is in
//! `candle_transformers::models::speculative_choice`. That is the entire reason
//! this type can be a thin wrapper over the sampler instead of a bespoke
//! accept/reject kernel: there is nothing to compute beyond the sample the
//! sampler was already going to draw.
//!
//! # Typical acceptance
//!
//! Each row also hands the sampler the draft it tests, with the model's
//! [`TypicalAcceptance`] thresholds ([`BatchedSampler::sample_verify_rows`]).
//! The kernel commits that draft instead of the sample when the row's
//! distribution gives it enough mass, so a step keeps every draft the exact
//! rule would and some it would not. The rule and why it needs no argmax
//! clause are in `speculative_choice`.
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
use candle_transformers::models::speculative_choice::{SpecRow, TokenChooser, TypicalAcceptance};

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
    /// The thresholds a draft the sample missed is accepted on.
    typical: TypicalAcceptance,
}

impl<'a> SpecChooser<'a> {
    pub(super) fn new(
        sampler: &'a BatchedSampler,
        states: Vec<SequenceSamplingState>,
        configs: Vec<SamplingConfig>,
        typical: TypicalAcceptance,
    ) -> Self {
        let n = states.len();
        Self {
            sampler,
            states,
            configs,
            rows: vec![Vec::new(); n],
            committed: vec![0; n],
            typical,
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
        // The sampler records each committed token into its sequence's state,
        // so the next position is priced against this one.
        let drafts: Vec<Option<u32>> = rows.iter().map(|r| r.draft).collect();
        let tokens = self.sampler.sample_verify_rows(
            logits,
            &mut states,
            &configs,
            &drafts,
            self.typical,
        )?;

        // Keep the row each token was drawn from, shaped like a plain decode
        // step's row (`[1, vocab]`) so the health checks read it identically.
        for (m, r) in rows.iter().enumerate() {
            self.rows[r.seq].push(logits.i(m..m + 1)?);
            self.committed[r.seq] += 1;
        }
        Ok(tokens)
    }
}

#[cfg(test)]
mod tests {
    use candle::Device;
    use candle_transformers::models::speculative_choice::AcceptWalk;

    use super::*;

    const VOCAB: usize = 100;

    /// One `[1, VOCAB]` row with `p(5) = 0.8` and `p(7) = 0.2`; every other
    /// token sits ~2e-9.
    fn row() -> Tensor {
        let mut data = vec![0.0f32; VOCAB];
        data[5] = 20.0;
        data[7] = 20.0 + 0.25f32.ln();
        Tensor::from_vec(data, (1, VOCAB), &Device::Cpu).expect("row")
    }

    fn sampled(seed: u64) -> SamplingConfig {
        let mut c = SamplingConfig::argmax();
        c.temperature = 1.0;
        c.top_k = 0;
        c.top_p = 1.0;
        c.seed = seed;
        c
    }

    /// **The chooser hands each row's draft to the sampler, in walk order.**
    /// Sequence 0 drafts 7 then 30, sequence 1 drafts 9. At position 0, 7 has
    /// 0.2 of its row and is committed whatever the sample; 9 has ~2e-9, so
    /// sequence 1 commits its sample (5 or 7) and leaves the walk. At position
    /// 1, 30 has ~2e-9 too, so sequence 0 commits its sample. Every committed
    /// token is recorded into its own sequence's state.
    #[test]
    fn each_row_is_sampled_against_its_own_draft() -> Result<()> {
        let sampler = BatchedSampler::new(Device::Cpu, VOCAB, VOCAB, 32, vec![2].into(), None);
        for seed in 0..8 {
            let states = vec![
                SequenceSamplingState::new(VOCAB, 32),
                SequenceSamplingState::new(VOCAB, 32),
            ];
            let mut chooser = SpecChooser::new(
                &sampler,
                states,
                vec![sampled(seed), sampled(seed + 100)],
                TypicalAcceptance::MEDUSA,
            );
            let blocks = vec![vec![1u32, 7, 30], vec![1u32, 9]];
            let mut walk = AcceptWalk::new(&blocks);

            let first = chooser.choose(&Tensor::cat(&[row(), row()], 0)?, &walk.rows())?;
            assert_eq!(first[0], 7, "seed {seed}: the 0.2 draft is committed");
            assert!(matches!(first[1], 5 | 7), "seed {seed}: {first:?}");
            walk.commit(&first, |_, _| true)?;
            assert_eq!(walk.alive(), &[0]);

            let second = chooser.choose(&row(), &walk.rows())?;
            assert!(matches!(second[0], 5 | 7), "seed {seed}: {second:?}");

            let states = chooser.into_states();
            assert_eq!(states[0].rng_offset, 2, "seed {seed}: two rows sampled");
            assert_eq!(states[1].rng_offset, 1, "seed {seed}: one row sampled");
        }
        Ok(())
    }
}
