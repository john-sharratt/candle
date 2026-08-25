//! Choosing the token a scored row of a speculative step actually commits.
//!
//! The speculative driver ([`super::batched_inference::ManagedBatchedModel::speculative_decode_step_batch`])
//! owns the draft/verify/rollback machinery but deliberately does not own the
//! *token decision*: that belongs to whoever is generating, because it is where
//! temperature, nucleus truncation, repetition penalties and grammar stencils
//! live. The driver scores rows and asks a [`TokenChooser`] what each one says.
//!
//! # Why a chooser makes speculation exact under sampling
//!
//! Speculative decoding is only worth having if the tokens it commits are drawn
//! from exactly the distribution plain decoding would have drawn them from. The
//! textbook construction (Leviathan et al.) draws a proposal `x ~ q`, accepts it
//! with probability `min(1, p(x)/q(x))`, and on rejection draws from
//! `norm(max(0, p - q))`.
//!
//! Every drafter in this engine proposes **greedily** — the MTP/NextN head
//! argmaxes its own logits — so `q` is a point mass at `x`: `q(x) = 1`, and
//! `q(y) = 0` elsewhere. Substituting:
//!
//! * the accept probability is `min(1, p(x)/1)` = `p(x)`;
//! * the rejection residual `max(0, p - q)` is `p` with `x` removed, since
//!   `p(x) - 1 <= 0` kills the proposed token and leaves every other mass
//!   untouched.
//!
//! So the committed token is `x` with probability `p(x)` and otherwise a draw
//! from `p` restricted to `y != x`, renormalised by `1 - p(x)` — which is `p`.
//! That is indistinguishable from a much simpler procedure: **draw `y ~ p` and
//! call the draft accepted exactly when `y == x`.** Both commit a draw from `p`,
//! and both accept with probability `p(x)`.
//!
//! The consequence is the whole reason this interface is small: the driver never
//! needs the drafter's distribution, never needs a probability read back to the
//! host, and needs no accept/reject kernel. It samples each scored row the way
//! it would have sampled a plain decode row, and keeps the longest prefix whose
//! samples happen to agree with the proposals. [`GreedyChooser`] is then not a
//! special case bolted on — it is what this rule becomes at temperature zero,
//! where `p` is a point mass and agreement is argmax equality.
//!
//! # History
//!
//! A chooser that applies repetition penalties needs to score row `j` under the
//! history that would have reached it. That history is not in doubt: a row is
//! only reachable when every earlier proposal in its block was accepted, so it
//! is the block's own draft prefix, known before the verify forward runs. The
//! driver hands it over as [`SpecRow::prefix`] and walks positions in order, so
//! a chooser can advance its per-sequence state exactly along the committed path.

use candle::{DType, Result, Tensor};

/// One scored row of a speculative step, as the chooser sees it.
#[derive(Debug, Clone, Copy)]
pub struct SpecRow<'a> {
    /// Index into the step's `seqs` slice — which sequence of the cohort this
    /// row belongs to. Not the session's sequence id; the caller built `seqs`
    /// and can map back.
    pub seq: usize,
    /// Position within that sequence's verify block. This row predicts the
    /// token that follows `block[..=position]`.
    pub position: usize,
    /// The tokens this sequence has already committed *within this block*, in
    /// order — empty at `position == 0`. A chooser with repetition penalties or
    /// a grammar stencil must score the row as though these had just been
    /// generated, or it prices the row against a history one or more tokens
    /// stale.
    pub prefix: &'a [u32],
}

/// Picks the token each scored row commits.
///
/// Called once per block position with every sequence still alive at that
/// position, so an implementation batches across the cohort and pays one
/// dispatch per position rather than one per row.
pub trait TokenChooser {
    /// Choose one token per row of `logits` (`[rows, vocab]`), in row order.
    /// `rows[m]` describes row `m`.
    fn choose(&mut self, logits: &Tensor, rows: &[SpecRow<'_>]) -> Result<Vec<u32>>;
}

/// Whether a sequence walks on to the next position of its verify block after
/// committing `token` at `position`.
///
/// The whole accept rule, in one place because the block's own layout decides
/// it and getting the indexing wrong is silent: `block` is `[seed, proposal…]`,
/// so the proposal this row was testing is `block[position + 1]`. A sequence
/// continues only when both hold —
///
/// * **there is a next row.** `position + 1 == block.len()` means this row was
///   the block's last, and the token it committed is the free bonus token that
///   follows a fully-accepted block. There is nothing left to test.
/// * **the model agreed with the proposal.** Otherwise `token` IS the
///   correction, every later row was scored against a prefix that will not
///   happen, and the sequence stops here.
///
/// A sequence that was never drafted has `block.len() == 1` and so stops after
/// its single row — a plain decode step, by the same rule.
pub fn continues(block: &[u32], position: usize, token: u32) -> bool {
    position + 1 < block.len() && token == block[position + 1]
}

/// One speculative step's accept walk, position by position.
///
/// The rule in [`continues`] applied across a cohort, holding the bookkeeping
/// that the rule alone does not: who is still walking, how many tokens each
/// sequence kept (which is where its KV rolls back to), and what its next
/// `committed` seed is.
///
/// It is a state machine rather than a loop because the caller has to do real
/// work between positions — gather that position's logits rows, ask a chooser
/// for tokens, run each token through the caller's own per-token handling —
/// and because two callers drive it. The standalone driver
/// (`ManagedBatchedModel::speculative_decode_step_batch`) owns its forward and
/// runs the walk inline; the scheduler owns a much richer forward (the
/// continuous-fair wave) and cannot hand that ownership to a model method, so
/// it drives the same walk itself. One rule, one set of off-by-one hazards,
/// tested once.
///
/// ```text
/// let mut walk = AcceptWalk::new(&blocks);
/// while !walk.finished() {
///     let rows = walk.rows();
///     let tokens = chooser.choose(&logits_at(walk.position(), walk.alive()), &rows)?;
///     walk.commit(&tokens, |i, t| sink(i, t))?;
/// }
/// let (next, kept) = walk.finish();
/// ```
pub struct AcceptWalk<'b> {
    blocks: &'b [Vec<u32>],
    alive: Vec<usize>,
    position: usize,
    kept: Vec<usize>,
    stopped: Vec<bool>,
    last: Vec<Option<u32>>,
}

impl<'b> AcceptWalk<'b> {
    /// Start a walk over one block per sequence. A sequence that drafted
    /// nothing has a one-token block (just its seed) and leaves after the
    /// first position — a plain decode step.
    pub fn new(blocks: &'b [Vec<u32>]) -> Self {
        let n = blocks.len();
        Self {
            blocks,
            alive: (0..n).collect(),
            position: 0,
            kept: vec![0; n],
            stopped: vec![false; n],
            last: vec![None; n],
        }
    }

    /// The block position this step is about to score.
    pub fn position(&self) -> usize {
        self.position
    }

    /// The sequences still walking, as indices into the cohort, **ascending**.
    ///
    /// The order is load-bearing: callers gather logits rows and per-sequence
    /// sampler state by this list, and a chooser that holds its state in cohort
    /// order relies on the subset arriving in that order too.
    pub fn alive(&self) -> &[usize] {
        &self.alive
    }

    /// Whether every sequence has stopped.
    pub fn finished(&self) -> bool {
        self.alive.is_empty()
    }

    /// This position's rows, in `alive()` order, for a [`TokenChooser`].
    pub fn rows(&self) -> Vec<SpecRow<'b>> {
        self.alive
            .iter()
            .map(|&i| SpecRow {
                seq: i,
                position: self.position,
                // Skips the seed at `block[0]` and covers every proposal
                // accepted to reach here; empty at position 0.
                prefix: &self.blocks[i][1..=self.position],
            })
            .collect()
    }

    /// Commit one token per alive sequence, in `alive()` order, and advance.
    ///
    /// `emit` receives `(cohort index, token)` and returns `false` to stop that
    /// sequence — an EOS, a budget, a steering decision. A stopped sequence
    /// still counts the token it stopped on in [`Self::finish`]'s kept count,
    /// because the token was generated and its KV must be kept.
    pub fn commit<F>(&mut self, tokens: &[u32], mut emit: F) -> Result<()>
    where
        F: FnMut(usize, u32) -> bool,
    {
        if tokens.len() != self.alive.len() {
            candle::bail!(
                "AcceptWalk::commit: {} tokens for {} live sequences at position {}",
                tokens.len(),
                self.alive.len(),
                self.position
            );
        }
        let mut still = Vec::with_capacity(self.alive.len());
        for (m, &i) in self.alive.iter().enumerate() {
            let token = tokens[m];
            self.kept[i] += 1;
            self.last[i] = Some(token);
            if !emit(i, token) {
                self.stopped[i] = true;
                continue;
            }
            if continues(&self.blocks[i], self.position, token) {
                still.push(i);
            }
        }
        self.alive = still;
        self.position += 1;
        Ok(())
    }

    /// `(next committed seed per sequence, tokens kept per sequence)`.
    ///
    /// The seed is `None` where the sink stopped: that sequence is done and has
    /// nothing to seed a following step with. The kept count is what the
    /// sequence's KV truncates to, counted from where it stood before the step.
    pub fn finish(self) -> (Vec<Option<u32>>, Vec<usize>) {
        let next = (0..self.blocks.len())
            .map(|i| if self.stopped[i] { None } else { self.last[i] })
            .collect();
        (next, self.kept)
    }
}

/// Argmax over each row.
///
/// The temperature-zero case of the rule in this module's docs: `p` is a point
/// mass on the argmax, so a proposal is accepted exactly when it *is* the
/// argmax, and the committed token is the model's greedy continuation. This is
/// what a correctness gate wants — output bit-identical to plain greedy decode
/// regardless of draft quality — and what a caller that does not sample wants.
pub struct GreedyChooser;

impl TokenChooser for GreedyChooser {
    fn choose(&mut self, logits: &Tensor, _rows: &[SpecRow<'_>]) -> Result<Vec<u32>> {
        logits
            .argmax(candle::D::Minus1)?
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::Device;

    /// The greedy chooser returns one argmax per row, in row order.
    #[test]
    fn greedy_chooser_argmaxes_each_row() -> Result<()> {
        let logits = Tensor::from_vec(
            vec![
                0.0f32, 1.0, 0.5, // row 0 -> 1
                3.0, 0.0, 0.5, // row 1 -> 0
                0.0, 0.5, 9.0, // row 2 -> 2
            ],
            (3, 3),
            &Device::Cpu,
        )?;
        let rows = [
            SpecRow {
                seq: 0,
                position: 0,
                prefix: &[],
            },
            SpecRow {
                seq: 1,
                position: 0,
                prefix: &[],
            },
            SpecRow {
                seq: 1,
                position: 1,
                prefix: &[7],
            },
        ];
        assert_eq!(GreedyChooser.choose(&logits, &rows)?, vec![1, 0, 2]);
        Ok(())
    }

    /// An undrafted sequence's block is just its seed, so it commits one token
    /// and stops — the plain-decode case, falling out of the same rule rather
    /// than being special-cased around it.
    #[test]
    fn undrafted_block_stops_after_one_row() {
        assert!(!continues(&[7], 0, 42));
    }

    /// An accepted proposal walks on; the first disagreement stops the walk and
    /// the token that disagreed is the one kept.
    #[test]
    fn walk_continues_only_while_proposals_hold() {
        let block = [7u32, 22, 33];
        assert!(continues(&block, 0, 22));
        assert!(!continues(&block, 0, 99));
        assert!(continues(&block, 1, 33));
        assert!(!continues(&block, 1, 99));
    }

    /// The last row of a fully-accepted block yields a bonus token and stops —
    /// even though it "agrees", there is no further proposal to test, and
    /// reading `block[position + 1]` there would run off the end.
    #[test]
    fn last_row_stops_even_when_every_proposal_held() {
        let block = [7u32, 22, 33];
        assert_eq!(block.len(), 3);
        assert!(!continues(&block, 2, 33));
        assert!(!continues(&block, 2, 44));
    }

    /// Drive a walk with a scripted chooser: `script[position]` gives the token
    /// each still-alive sequence commits, keyed by cohort index. Returns the
    /// tokens each sequence emitted, plus the walk's own result.
    fn drive(
        blocks: &[Vec<u32>],
        script: &[Vec<(usize, u32)>],
        stop_after: Option<usize>,
    ) -> (Vec<Vec<u32>>, Vec<Option<u32>>, Vec<usize>) {
        let mut emitted = vec![Vec::new(); blocks.len()];
        let mut walk = AcceptWalk::new(blocks);
        let mut positions = 0usize;
        while !walk.finished() {
            let at = &script[walk.position()];
            let tokens: Vec<u32> = walk
                .alive()
                .iter()
                .map(|&i| at.iter().find(|(j, _)| *j == i).expect("scripted").1)
                .collect();
            let total: usize = emitted.iter().map(|e: &Vec<u32>| e.len()).sum();
            walk.commit(&tokens, |i, t| {
                emitted[i].push(t);
                stop_after.is_none_or(|n| total + 1 < n)
            })
            .unwrap();
            positions += 1;
            assert!(positions <= 8, "walk did not terminate");
        }
        let (next, kept) = walk.finish();
        (emitted, next, kept)
    }

    /// Every proposal holds: the walk runs one position per block token and the
    /// last row yields the free bonus token, so a `k`-proposal block commits
    /// `k + 1` tokens.
    #[test]
    fn a_fully_accepted_block_commits_one_more_token_than_it_proposed() {
        let blocks = vec![vec![7u32, 22, 33]];
        let script = vec![vec![(0, 22)], vec![(0, 33)], vec![(0, 99)]];
        let (emitted, next, kept) = drive(&blocks, &script, None);
        assert_eq!(emitted[0], vec![22, 33, 99]);
        assert_eq!(kept, vec![3]);
        // The bonus token seeds the next step.
        assert_eq!(next, vec![Some(99)]);
    }

    /// The first disagreement ends the block, and the token that disagreed is
    /// the one kept — it is the model's correction, not a discard.
    #[test]
    fn the_first_disagreement_is_kept_as_the_correction() {
        let blocks = vec![vec![7u32, 22, 33]];
        let script = vec![vec![(0, 55)]];
        let (emitted, next, kept) = drive(&blocks, &script, None);
        assert_eq!(emitted[0], vec![55]);
        assert_eq!(kept, vec![1]);
        assert_eq!(next, vec![Some(55)]);
    }

    /// A sequence that drafted nothing commits exactly one token — the plain
    /// decode step, reached through the same walk rather than around it.
    #[test]
    fn an_undrafted_sequence_commits_exactly_one_token() {
        let blocks = vec![vec![7u32]];
        let script = vec![vec![(0, 42)]];
        let (emitted, next, kept) = drive(&blocks, &script, None);
        assert_eq!(emitted[0], vec![42]);
        assert_eq!(kept, vec![1]);
        assert_eq!(next, vec![Some(42)]);
    }

    /// A sink that stops still keeps the token it stopped on — that token was
    /// generated and its KV must survive the rollback — but the sequence gets no
    /// seed, because there is no next step for it.
    #[test]
    fn a_stopped_sink_keeps_its_last_token_but_takes_no_seed() {
        let blocks = vec![vec![7u32, 22, 33]];
        let script = vec![vec![(0, 22)], vec![(0, 33)], vec![(0, 99)]];
        let (emitted, next, kept) = drive(&blocks, &script, Some(2));
        assert_eq!(emitted[0], vec![22, 33]);
        assert_eq!(kept, vec![2]);
        assert_eq!(next, vec![None]);
    }

    /// A mixed cohort: sequences leave the walk at different positions, and the
    /// live set stays in ascending cohort order the whole way — which is what
    /// the caller's logits gather and the chooser's per-sequence state rely on.
    #[test]
    fn a_mixed_cohort_drops_out_in_ascending_order() {
        // 0: undrafted. 1: rejects immediately. 2: accepts everything.
        let blocks = vec![vec![7u32], vec![8, 80], vec![9, 90, 91]];
        let mut walk = AcceptWalk::new(&blocks);
        assert_eq!(walk.alive(), &[0, 1, 2]);
        walk.commit(&[1, 999, 90], |_, _| true).unwrap();
        // 0 had a one-token block; 1's token disagreed with its proposal 80.
        assert_eq!(walk.alive(), &[2]);
        walk.commit(&[91], |_, _| true).unwrap();
        assert_eq!(walk.alive(), &[2]);
        walk.commit(&[7], |_, _| true).unwrap();
        assert!(walk.finished());
        let (next, kept) = walk.finish();
        assert_eq!(kept, vec![1, 1, 3]);
        assert_eq!(next, vec![Some(1), Some(999), Some(7)]);
    }

    /// Each position's rows carry the prefix that reaches them, so a chooser
    /// with repetition penalties prices the row against the history it would
    /// really have had.
    #[test]
    fn rows_carry_the_prefix_that_reaches_them() {
        let blocks = vec![vec![7u32, 22, 33]];
        let mut walk = AcceptWalk::new(&blocks);
        assert_eq!(walk.rows()[0].prefix, &[] as &[u32]);
        walk.commit(&[22], |_, _| true).unwrap();
        assert_eq!(walk.rows()[0].prefix, &[22]);
        walk.commit(&[33], |_, _| true).unwrap();
        assert_eq!(walk.rows()[0].prefix, &[22, 33]);
    }

    /// A token count that disagrees with the live set is a caller bug that
    /// would otherwise commit one sequence's token to another.
    #[test]
    fn commit_refuses_a_token_count_that_is_not_the_live_set() {
        let blocks = vec![vec![7u32, 22], vec![8, 80]];
        let mut walk = AcceptWalk::new(&blocks);
        assert!(walk.commit(&[1], |_, _| true).is_err());
    }

    /// The prefix a row carries is the block's own draft prefix, so a chooser
    /// that tracks history can reconstruct it without the driver replaying
    /// tokens. Pinned here because the driver's accept walk depends on the
    /// exact convention: `prefix` excludes the seed token at `block[0]` and
    /// includes every proposal accepted before this row.
    #[test]
    fn prefix_grows_by_one_accepted_proposal_per_position() {
        let block = [11u32, 22, 33];
        let at = |position: usize| SpecRow {
            seq: 0,
            position,
            prefix: &block[1..=position],
        };
        assert_eq!(at(0).prefix, &[] as &[u32]);
        assert_eq!(at(1).prefix, &[22]);
        assert_eq!(at(2).prefix, &[22, 33]);
    }
}
