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
//! [`TypicalAcceptance`] thresholds. The kernel commits that draft instead of
//! the sample when the row's distribution gives it enough mass, so a step keeps
//! every draft the exact rule would and some it would not. The rule and why it
//! needs no argmax clause are in `speculative_choice`.
//!
//! # One dispatch per step, committed position by position
//!
//! Repetition and DRY penalties at block position `j` must see the tokens
//! committed at `0..j`. The walk reaches position `j` only by committing the
//! block's first `j` drafts, so that history is known before the walk starts:
//! [`SpecChooser::score`] prices every position against the state those drafts
//! would leave and scores the whole step in one dispatch
//! ([`BatchedSampler::score_rows`]). The walk then commits position by position
//! ([`SpecChooser::commit`]), each commit advancing the sequence's state along
//! exactly the committed path and no further — and refusing, rather than
//! committing, a row whose scored state is not the one the walk arrived at.
//! The `prefix` each row carries checks the walk's side of the same agreement:
//! a state that has not advanced by one token per position has desynced, and
//! silently mispriced penalties are the kind of bug that reads as a mysterious
//! quality regression rather than a failure.

use candle::{IndexOp, Result, Tensor};
use candle_transformers::models::speculative_choice::{SpecRow, TypicalAcceptance};

use crate::batched_sampler::{BatchedSampler, RowSpec, Scored, SequenceSamplingState};
use crate::config::SamplingConfig;

/// The scheduler's chooser for one speculative step.
///
/// Owns the cohort's sampling states for the duration of the step — they are
/// lifted out of the scheduler's map the same way the plain decode path lifts
/// them, so the sampler can borrow them mutably — and hands them back through
/// [`Self::into_states`].
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
    /// The step's logits block, once [`Self::score`] has read it.
    block: Option<Tensor>,
    /// Per cohort index and block position: the block row the position was
    /// scored from and its pick, or `None` for a position the earlier ones
    /// cannot reach.
    scored: Vec<Vec<Option<(usize, Scored)>>>,
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
            block: None,
            scored: vec![Vec::new(); n],
        }
    }

    /// Score every position of every block in one dispatch. `blocks[i]` is
    /// cohort sequence `i`'s committed token followed by its drafts, and
    /// `positions[i][p]` the row of `block` that scores its position `p`.
    pub(super) fn score(
        &mut self,
        block: Tensor,
        positions: &[Vec<usize>],
        blocks: &[Vec<u32>],
    ) -> Result<()> {
        if blocks.len() != self.states.len() || positions.len() != blocks.len() {
            candle::bail!(
                "SpecChooser::score: {} blocks and {} position lists for a {}-sequence cohort",
                blocks.len(),
                positions.len(),
                self.states.len()
            );
        }
        let mut specs: Vec<RowSpec<'_>> = Vec::new();
        let mut at: Vec<(usize, usize)> = Vec::new();
        for (i, b) in blocks.iter().enumerate() {
            if positions[i].len() != b.len() {
                candle::bail!(
                    "SpecChooser::score: sequence {i} has {} scored rows for a {}-token block",
                    positions[i].len(),
                    b.len()
                );
            }
            for (p, &logits_row) in positions[i].iter().enumerate() {
                specs.push(RowSpec {
                    seq: i,
                    logits_row,
                    draft: b.get(p + 1).copied(),
                    ahead: &b[1..=p],
                });
                at.push((i, p));
            }
        }
        let mut states: Vec<&mut SequenceSamplingState> = self.states.iter_mut().collect();
        let configs: Vec<&SamplingConfig> = self.configs.iter().collect();
        let scored =
            self.sampler
                .score_rows(&block, &mut states, &configs, &specs, Some(self.typical))?;
        self.scored = blocks.iter().map(|b| vec![None; b.len()]).collect();
        for ((i, p), (spec, s)) in at.into_iter().zip(specs.iter().zip(scored)) {
            self.scored[i][p] = s.map(|s| (spec.logits_row, s));
        }
        self.block = Some(block);
        Ok(())
    }

    /// Commit the walk's rows at one position, in ascending cohort order,
    /// returning each row's token.
    pub(super) fn commit(&mut self, rows: &[SpecRow<'_>]) -> Result<Vec<u32>> {
        let Some(block) = self.block.as_ref() else {
            candle::bail!("SpecChooser::commit: the step was never scored");
        };
        // The walk hands live sequences over in ascending cohort order; a caller
        // that reordered them would pair each row with the wrong sequence.
        if rows.windows(2).any(|w| w[0].seq >= w[1].seq) {
            candle::bail!("SpecChooser: rows are not in ascending cohort order");
        }
        let mut tokens = Vec::with_capacity(rows.len());
        for r in rows {
            if r.seq >= self.states.len() {
                candle::bail!(
                    "SpecChooser: row names sequence {} of a {}-sequence cohort",
                    r.seq,
                    self.states.len()
                );
            }
            // One token committed per position walked, or the penalties this row
            // was priced with are stale.
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
            let Some((logits_row, scored)) = self.scored[r.seq].get(r.position).copied().flatten()
            else {
                candle::bail!(
                    "SpecChooser: the walk reached sequence {} position {}, which its scoring \
                     found unreachable",
                    r.seq,
                    r.position
                );
            };
            let token = self.sampler.commit_row(
                block,
                logits_row,
                &mut self.states[r.seq],
                &self.configs[r.seq],
                scored,
            )?;
            // Keep the row the token was drawn from, shaped like a plain decode
            // step's row (`[1, vocab]`) so the health checks read it identically.
            self.rows[r.seq].push(block.i(logits_row..logits_row + 1)?);
            self.committed[r.seq] += 1;
            tokens.push(token);
        }
        Ok(tokens)
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

/// The wave's scored rows as one `[rows, vocab]` block, and where each of
/// `per_seq`'s rows sits in it.
///
/// A wave run in one forward hands its rows back as views on the one tensor
/// its head wrote, so the block is a view as well and a row's place in it is
/// its offset: the step's single dispatch reads the rows where the head wrote
/// them, with nothing copied to line them up. A wave run in slices owns each
/// slice's rows separately (`wave_driver`'s sliced path copies them off each
/// slice's span), and those are joined once, in walk order — one copy for the
/// step.
pub(super) fn locate_rows(
    wave: &[Tensor],
    per_seq: &[Vec<Tensor>],
) -> Result<(Tensor, Vec<Vec<usize>>)> {
    let flat: Vec<Tensor> = wave
        .iter()
        .map(|t| t.flatten_all()?.unsqueeze(0))
        .collect::<Result<_>>()?;
    let Some(block) = Tensor::cat_view(&flat, 0) else {
        let rows: Vec<Tensor> = per_seq
            .iter()
            .flatten()
            .map(|t| t.flatten_all()?.unsqueeze(0))
            .collect::<Result<_>>()?;
        let joined = Tensor::cat(&rows, 0)?;
        let mut next = 0;
        let positions = per_seq
            .iter()
            .map(|rows| {
                let at: Vec<usize> = (next..next + rows.len()).collect();
                next += rows.len();
                at
            })
            .collect();
        return Ok((joined, positions));
    };
    let (n, vocab) = block.dims2()?;
    let base = block.layout().start_offset();
    let positions = per_seq
        .iter()
        .map(|rows| {
            rows.iter()
                .map(|row| {
                    let off = row.layout().start_offset();
                    let within = off.checked_sub(base).filter(|d| d % vocab == 0);
                    match within.map(|d| d / vocab) {
                        Some(i)
                            if i < n
                                && row.elem_count() == vocab
                                && row.is_contiguous()
                                && row.same_storage(&block) =>
                        {
                            Ok(i)
                        }
                        _ => candle::bail!(
                            "locate_rows: a scored row at offset {off} is not a row of the \
                             wave's {n} × {vocab} block at offset {base}"
                        ),
                    }
                })
                .collect::<Result<Vec<usize>>>()
        })
        .collect::<Result<Vec<_>>>()?;
    Ok((block, positions))
}

#[cfg(test)]
mod tests {
    use candle::Device;
    use candle_transformers::models::speculative_choice::AcceptWalk;

    use super::*;

    const VOCAB: usize = 100;

    /// One `[1, VOCAB]` row with `p(5) = 0.8` and `p(7) = 0.2`; every other
    /// token sits ~2e-9.
    fn row_values() -> Vec<f32> {
        let mut data = vec![0.0f32; VOCAB];
        data[5] = 20.0;
        data[7] = 20.0 + 0.25f32.ln();
        data
    }

    /// `n` such rows as one `[n, VOCAB]` block.
    fn block(n: usize) -> Tensor {
        let data: Vec<f32> = (0..n).flat_map(|_| row_values()).collect();
        Tensor::from_vec(data, (n, VOCAB), &Device::Cpu).expect("block")
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
            // Sequence 0 scores rows 0..3 of the block, sequence 1 rows 3..5.
            chooser.score(block(5), &[vec![0, 1, 2], vec![3, 4]], &blocks)?;
            let mut walk = AcceptWalk::new(&blocks);

            let first = chooser.commit(&walk.rows())?;
            assert_eq!(first[0], 7, "seed {seed}: the 0.2 draft is committed");
            assert!(matches!(first[1], 5 | 7), "seed {seed}: {first:?}");
            walk.commit(&first, |_, _| true)?;
            assert_eq!(walk.alive(), &[0]);

            let second = chooser.commit(&walk.rows())?;
            assert!(matches!(second[0], 5 | 7), "seed {seed}: {second:?}");

            let states = chooser.into_states();
            assert_eq!(states[0].rng_offset, 2, "seed {seed}: two rows sampled");
            assert_eq!(states[1].rng_offset, 1, "seed {seed}: one row sampled");
        }
        Ok(())
    }

    /// A sliced wave's rows are separate tensors: they are joined in walk order
    /// and located by that order.
    #[test]
    fn a_sliced_waves_rows_are_joined_in_walk_order() -> Result<()> {
        let wave: Vec<Tensor> = (0..3).map(|_| block(1)).collect();
        let per_seq = vec![
            vec![wave[2].clone()],
            vec![wave[0].clone(), wave[1].clone()],
        ];
        let (joined, positions) = locate_rows(&wave, &per_seq)?;
        assert_eq!(positions, vec![vec![0], vec![1, 2]]);
        assert_eq!(joined.dims(), &[3, VOCAB]);
        Ok(())
    }

    /// The rows of a wave are found where the head wrote them: a block's views
    /// locate by offset, and a row of some other tensor is refused.
    #[test]
    fn rows_are_located_in_the_wave_block_by_offset() -> Result<()> {
        let head = block(4);
        let wave: Vec<Tensor> = (0..4)
            .map(|i| head.narrow(0, i, 1))
            .collect::<Result<_>>()?;
        let per_seq = vec![
            vec![wave[2].clone(), wave[3].clone()],
            vec![wave[0].clone()],
        ];
        let (located, positions) = locate_rows(&wave, &per_seq)?;
        assert_eq!(positions, vec![vec![2, 3], vec![0]]);
        assert!(located.same_storage(&head), "a view, not a copy");

        let stranger = block(1);
        assert!(locate_rows(&wave, &[vec![stranger]]).is_err());
        Ok(())
    }
}
