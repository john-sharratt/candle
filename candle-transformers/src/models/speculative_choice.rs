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
//!
//! # Typical acceptance
//!
//! A sampling chooser also accepts a draft the sample did not land on, when the
//! row's distribution gives it enough mass: `p(draft) > min(ε, δ·e^(−H))`, the
//! Medusa rule ([`TypicalAcceptance`]). Otherwise it commits the sample. That
//! keeps every step at least as long as the exact rule's — a sample that lands
//! on the draft still accepts it — and a row whose draft fails the threshold
//! commits exactly what plain sampling would have, so the departure from the
//! target distribution is confined to the drafts the threshold let through.
//! The rule needs no argmax clause: `e^(−H) ≤ max p`, so with `δ < 1` the
//! argmax always clears the threshold. At temperature zero the row is a point
//! mass and the sample is the argmax, so greedy verification is unchanged.

use candle::{DType, Result, Tensor};

/// The Medusa typical-acceptance thresholds: a draft is accepted when its
/// probability under the row's sampling distribution exceeds
/// `min(epsilon, delta · exp(−entropy))`.
///
/// `epsilon` caps the bar on a peaked distribution; `delta · exp(−entropy)`
/// lowers it as the distribution flattens, where many tokens are plausible and
/// demanding a high probability of any one of them would reject good drafts.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TypicalAcceptance {
    pub epsilon: f32,
    pub delta: f32,
}

impl TypicalAcceptance {
    /// The Medusa reference values.
    pub const MEDUSA: Self = Self {
        epsilon: 0.09,
        delta: 0.3,
    };

    /// Panics unless `0 < epsilon <= 1` and `0 < delta < 1`.
    ///
    /// Outside that range the rule stops meaning what it says: a zero bar
    /// accepts every draft the row gives any mass at all, and `delta >= 1`
    /// breaks the property that the argmax always clears the bar. Thresholds
    /// are a model constant, so a value out of range is a bug in that model,
    /// caught where the engine loads it rather than as quietly worse text.
    pub fn assert_valid(&self) {
        let Self { epsilon, delta } = *self;
        assert!(
            epsilon > 0.0 && epsilon <= 1.0 && delta > 0.0 && delta < 1.0,
            "typical acceptance needs 0 < epsilon <= 1 and 0 < delta < 1, got \
             epsilon = {epsilon}, delta = {delta}"
        );
    }
}

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
    /// The proposal this row tests — `block[position + 1]` — or `None` on the
    /// block's last row, which scores the bonus token and tests nothing. A
    /// chooser applying [`TypicalAcceptance`] reads it.
    pub draft: Option<u32>,
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

    /// Whether a row's token depends on nothing but the row itself and its
    /// [`SpecRow`] — no state that committing one position's token advances
    /// before the next is priced.
    ///
    /// Such a chooser is asked for every scored row of a step in one call —
    /// one launch and one readback — and the accept walk then commits from
    /// those picks position by position, by the same rule. Rows past where a
    /// sequence stops are chosen and never read, which is only sound when
    /// choosing them changed nothing. A chooser carrying penalties, an RNG
    /// stream or a grammar stencil advances that state per committed token, so
    /// it is walked position by position and says `false`.
    fn stateless(&self) -> bool {
        false
    }
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

/// One cohort speculative step's outcome, per sequence in the order the step was given them.
pub struct SpeculativeStep {
    /// The next committed seed (already emitted, held out of the KV), or `None` where the
    /// sequence's sink asked to stop.
    pub next: Vec<Option<u32>>,
    /// The tokens the sequence actually drafted — its requested depth, or fewer where the
    /// drafter proposed less (none at all takes a plain decode row). An acceptance estimate
    /// divides by this, never by the depth asked for.
    pub drafted: Vec<usize>,
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
                draft: self.blocks[i].get(self.position + 1).copied(),
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
///
/// One launch of the fused batched sampler's greedy path over every row — the
/// kernel the scheduler's sampler runs at temperature zero — not the generic
/// `argmax` reduction, whose half-precision path addresses every element
/// through the strided-index walk.
///
/// The argmax runs over the first `live_vocab` columns, the tokens the
/// tokenizer can name: a checkpoint pads its logits row past them, and the
/// padded tail is not a token.
pub struct GreedyChooser {
    live_vocab: usize,
    /// Where each position's picks are written — grown to the widest cohort
    /// the chooser has walked and reused, so a walk position allocates nothing.
    picks: Option<Tensor>,
}

impl GreedyChooser {
    /// Greedy over the first `live_vocab` columns of each row; the row width
    /// for a model whose logits are not padded.
    pub fn new(live_vocab: usize) -> Self {
        Self {
            live_vocab,
            picks: None,
        }
    }

    /// Greedy over every column of each row, whatever its width — for a
    /// caller that holds no tokenizer and compares against its own whole-row
    /// greedy decode, which a bound on one side only would make disagree.
    pub fn whole_row() -> Self {
        Self::new(usize::MAX)
    }
}

impl TokenChooser for GreedyChooser {
    fn choose(&mut self, logits: &Tensor, _rows: &[SpecRow<'_>]) -> Result<Vec<u32>> {
        let n = logits.dim(0)?;
        let fits = self
            .picks
            .as_ref()
            .is_some_and(|p| p.elem_count() >= n && p.device().same_device(logits.device()));
        if !fits {
            self.picks = Some(Tensor::empty(n, DType::U32, logits.device())?);
        }
        let picks = self.picks.as_ref().expect("sized above").narrow(0, 0, n)?;
        logits.batched_sample_argmax_into(self.live_vocab, &picks)?;
        picks.to_vec1::<u32>()
    }

    /// An argmax reads its row and nothing else.
    fn stateless(&self) -> bool {
        true
    }
}

/// Every scored row of a step at once, in row order — what a
/// [stateless](TokenChooser::stateless) chooser is asked for.
///
/// `blocks[i]` is sequence `i`'s verify block and `row_of[i]` the first of its
/// `blocks[i].len()` rows; every row is described as the accept walk would
/// describe it on reaching it, prefix and draft included.
pub fn all_rows<'b>(blocks: &'b [Vec<u32>], row_of: &[(usize, usize)]) -> Vec<SpecRow<'b>> {
    let total: usize = blocks.iter().map(Vec::len).sum();
    let mut rows: Vec<Option<SpecRow<'b>>> = vec![None; total];
    for (i, block) in blocks.iter().enumerate() {
        for position in 0..block.len() {
            rows[row_of[i].0 + position] = Some(SpecRow {
                seq: i,
                position,
                prefix: &block[1..=position],
                draft: block.get(position + 1).copied(),
            });
        }
    }
    rows.into_iter()
        .map(|r| r.expect("row_of tiles the rows"))
        .collect()
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
                draft: None,
            },
            SpecRow {
                seq: 1,
                position: 0,
                prefix: &[],
                draft: Some(7),
            },
            SpecRow {
                seq: 1,
                position: 1,
                prefix: &[7],
                draft: None,
            },
        ];
        assert_eq!(GreedyChooser::new(3).choose(&logits, &rows)?, vec![1, 0, 2]);
        Ok(())
    }

    /// The padded tail of a row is never the greedy pick: with two live
    /// tokens, row 2's 9.0 in the third column is padding, and the row's pick
    /// is its best live token.
    #[test]
    fn greedy_chooser_never_picks_the_padded_tail() -> Result<()> {
        let logits = Tensor::from_vec(
            vec![
                0.0f32, 1.0, 0.5, // row 0 -> 1
                0.0, 0.5, 9.0, // row 1 -> 1, not the padded 2
            ],
            (2, 3),
            &Device::Cpu,
        )?;
        let row = SpecRow {
            seq: 0,
            position: 0,
            prefix: &[],
            draft: None,
        };
        assert_eq!(
            GreedyChooser::new(2).choose(&logits, &[row, row])?,
            vec![1, 1]
        );
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

    #[test]
    fn the_medusa_thresholds_are_valid() {
        TypicalAcceptance::MEDUSA.assert_valid();
    }

    /// A zero bar would accept every draft the row gives any mass.
    #[test]
    #[should_panic(expected = "typical acceptance needs")]
    fn a_zero_epsilon_is_refused() {
        TypicalAcceptance {
            epsilon: 0.0,
            delta: 0.3,
        }
        .assert_valid();
    }

    /// `delta >= 1` would let the bar rise above the argmax's probability.
    #[test]
    #[should_panic(expected = "typical acceptance needs")]
    fn a_delta_of_one_is_refused() {
        TypicalAcceptance {
            epsilon: 0.09,
            delta: 1.0,
        }
        .assert_valid();
    }

    /// Each row names the proposal it tests, and the last row — the bonus
    /// position — names none.
    #[test]
    fn rows_carry_the_draft_they_test() {
        let blocks = vec![vec![7u32, 22, 33]];
        let mut walk = AcceptWalk::new(&blocks);
        assert_eq!(walk.rows()[0].draft, Some(22));
        walk.commit(&[22], |_, _| true).unwrap();
        assert_eq!(walk.rows()[0].draft, Some(33));
        walk.commit(&[33], |_, _| true).unwrap();
        assert_eq!(walk.rows()[0].draft, None);
    }

    /// Every row comes out where `row_of` puts it, described exactly as the
    /// walk describes it on reaching it — so committing from these picks is
    /// committing from what a position-by-position walk would have asked for.
    #[test]
    fn all_rows_describes_each_row_as_the_walk_does() {
        // Cohort order 0, 1, 2; row order puts sequence 2 first.
        let blocks = vec![vec![7u32, 22, 33], vec![8u32], vec![9u32, 90]];
        let row_of = vec![(2, 3), (5, 1), (0, 2)];
        let rows = all_rows(&blocks, &row_of);
        let got: Vec<(usize, usize, Vec<u32>, Option<u32>)> = rows
            .iter()
            .map(|r| (r.seq, r.position, r.prefix.to_vec(), r.draft))
            .collect();
        assert_eq!(
            got,
            vec![
                (2, 0, vec![], Some(90)),
                (2, 1, vec![90], None),
                (0, 0, vec![], Some(22)),
                (0, 1, vec![22], Some(33)),
                (0, 2, vec![22, 33], None),
                (1, 0, vec![], None),
            ]
        );
        let mut walk = AcceptWalk::new(&blocks);
        while !walk.finished() {
            for r in walk.rows() {
                let at = &rows[row_of[r.seq].0 + r.position];
                assert_eq!((at.prefix, at.draft), (r.prefix, r.draft));
            }
            let picks: Vec<u32> = walk
                .alive()
                .iter()
                .map(|&i| blocks[i].get(walk.position() + 1).copied().unwrap_or(0))
                .collect();
            walk.commit(&picks, |_, _| true).unwrap();
        }
    }

    /// Greedy reads its row and nothing else; the default is to walk.
    #[test]
    fn only_a_chooser_that_reads_nothing_but_its_row_is_stateless() {
        struct Walks;
        impl TokenChooser for Walks {
            fn choose(&mut self, _: &Tensor, rows: &[SpecRow<'_>]) -> Result<Vec<u32>> {
                Ok(vec![0; rows.len()])
            }
        }
        assert!(GreedyChooser::new(3).stateless());
        assert!(!Walks.stateless());
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
            draft: block.get(position + 1).copied(),
        };
        assert_eq!(at(0).prefix, &[] as &[u32]);
        assert_eq!(at(1).prefix, &[22]);
        assert_eq!(at(2).prefix, &[22, 33]);
    }
}
