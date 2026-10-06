//! How deep one sequence drafts, from how much of its drafting it keeps.
//!
//! A model's draft ladder ([`super::draft_ladder`]) says how deep drafting can
//! pay at a given wave width — measured on a rewrite, which the drafter predicts
//! almost perfectly. What a sequence actually keeps depends on what it is
//! writing: free continuation on Flash-Next accepts ~2.1 tokens a step against
//! the rewrite's 4.85. Every proposal past what the verify keeps is a step of the
//! drafter's serial walk and a verify row that bought nothing.
//!
//! # The depth that buys the most tokens per second
//!
//! A step that drafts `d` commits the proposals accepted before the first miss
//! plus the one token the verify adds after them, and costs a fixed `T0` for
//! the verify forward and everything around it plus `c` per drafted token for
//! its walk step and its verify row. So each sequence drafts the `d` that
//! maximises
//!
//! ```text
//!   tokens(d) / time(d)  =  (1 + q₁ + q₁q₂ + … + q₁⋯q_d)  /  (1 + r·d)
//! ```
//!
//! where `q_j` is the chance the `j`-th proposal is accepted given every one
//! before it was, and `r = c / T0` is a property of the model, carried beside
//! its ladder ([`super::draft_ladder::DraftLadder::token_cost`]).
//!
//! # Acceptance by position, not one rate
//!
//! **A drafter's proposals are not equally good, and the depth turns on exactly
//! where they stop being good.** On Flash-Next alone the rewrite's acceptance
//! holds to about the twelfth proposal — a fixed depth of 8 ran 279 tok/s, 12
//! ran 330 — while a free-written essay's thins after the second: a fixed 2 ran
//! 96.8 tok/s, 3 ran 83.7. One acceptance rate for every position cannot see
//! either shape: priced as one rate, the rewrite drafted ~15 deep (310 tok/s).
//!
//! So `q_j` is estimated per position. A step that drafted `d` and kept `a`
//! proposals reached positions `1..=a+1` (to `d`), accepted the first `a` and
//! rejected the next, and each reached position's counts are a running mean
//! with [`RETAIN`] of their weight carried across steps.
//!
//! A position a step did not reach learned nothing, and ages back toward the
//! [`PRIOR`] — slowly ([`RELAX`]), so what it showed when last reached still
//! counts for several steps. Two failures sit either side of that rate:
//!
//! - **Never ageing** leaves a position that once missed unreachable for good —
//!   a sequence that missed at its third proposal drafts two forever, even once
//!   its text turns predictable.
//! - **Ageing fast** makes every position past the last miss look like the
//!   optimistic prior within a couple of steps, and a sequence writing free text
//!   drafts deep again and again into positions that keep missing: 88 tok/s on
//!   the essay at the rate unreached and reached positions share, 92 at this one.
//!
//! A sequence never drafts none: one whose proposals keep missing drafts one,
//! so it can find out when its text turns predictable again.
//!
//! Measured on the RTX PRO 5000, 255 tokens a run, this rule runs the rewrite
//! at 316 tok/s and the essay at 92. The one-past-acceptance rule it replaced
//! stopped the rewrite at a depth of 8 (279 tok/s); on the essay it settles
//! near 2, the depth whose fixed run measured 97.

/// How many draft positions keep their own counts. Past it, a position is
/// priced at the prior; no ladder reaches it.
const TRACKED: usize = 32;

/// The weight a reached position's counts keep from one step to the next —
/// about three steps of memory.
const RETAIN: f32 = 0.7;

/// The weight an unreached position's counts keep from one step to the next as
/// they age back toward the [`PRIOR`] — about ten steps.
const RELAX: f32 = 0.9;

/// A position's starting counts, `(accepted, reached)`: one observation's
/// worth of evidence at 90%. Optimistic, because until a sequence has been
/// measured the ladder's own figure is the best estimate there is; weak, so one
/// miss at a position is enough to stop there.
const PRIOR: (f32, f32) = (0.9, 1.0);

/// One sequence's drafting depth, learned from its own verify steps.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DraftDepth {
    /// Per draft position `j` (index `j − 1`), running `(accepted, reached)`.
    counts: [(f32, f32); TRACKED],
}

impl Default for DraftDepth {
    fn default() -> Self {
        Self {
            counts: [PRIOR; TRACKED],
        }
    }
}

impl DraftDepth {
    /// How many tokens to draft this step, never more than `ceiling`, and one
    /// at least when `ceiling` allows any.
    ///
    /// `token_cost` is what one drafted token costs relative to the step's
    /// fixed cost — see the module docs.
    pub fn budget(&self, ceiling: usize, token_cost: f32) -> usize {
        if ceiling == 0 {
            return 0;
        }
        let mut best = (0.0f32, 0usize);
        // Expected tokens at depth `d`: the verify's own token, plus each
        // proposal weighted by the chance every one up to it is accepted.
        let mut tokens = 1.0f32;
        let mut reach = 1.0f32;
        for d in 1..=ceiling {
            reach *= self.acceptance(d);
            tokens += reach;
            let score = tokens / (1.0 + token_cost * d as f32);
            if d == 1 || score > best.0 {
                best = (score, d);
            }
        }
        best.1
    }

    /// `q_j`: the chance proposal `j` (1-based) is accepted given every one
    /// before it was.
    fn acceptance(&self, j: usize) -> f32 {
        let (accepted, reached) = self.counts.get(j - 1).copied().unwrap_or(PRIOR);
        accepted / reached
    }

    /// Record one verify step: `drafted` proposals went in, and the accept
    /// walk kept `kept` tokens of the block — the accepted proposals plus the
    /// token the verify produced after them.
    ///
    /// A step that drafted nothing says nothing about acceptance and is not
    /// recorded.
    pub fn record(&mut self, drafted: usize, kept: usize) {
        if drafted == 0 {
            return;
        }
        let accepted = kept.saturating_sub(1).min(drafted);
        // Positions `1..=accepted` were reached and accepted; the next, if the
        // block went that far, was reached and rejected.
        let reached = (accepted + 1).min(drafted).min(TRACKED);
        for (j, (a, r)) in self.counts.iter_mut().enumerate() {
            if j < reached {
                let hit = if j < accepted { 1.0 } else { 0.0 };
                *a = RETAIN * *a + hit;
                *r = RETAIN * *r + 1.0;
            } else {
                *a = RELAX * *a + (1.0 - RELAX) * PRIOR.0;
                *r = RELAX * *r + (1.0 - RELAX) * PRIOR.1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Flash-Next's cost, the one the examples below price.
    const COST: f32 = 0.125;

    fn close(a: (f32, f32), b: (f32, f32)) -> bool {
        (a.0 - b.0).abs() < 1e-5 && (a.1 - b.1).abs() < 1e-5
    }

    /// Unmeasured, every position is priced at the prior's 0.9: scores
    /// d=8 3.0629, d=9 3.0650, d=10 3.0497 — nine, and every narrower ceiling
    /// in full.
    #[test]
    fn an_unmeasured_sequence_drafts_the_priors_optimum() {
        let d = DraftDepth::default();
        assert_eq!(d.budget(16, COST), 9);
        assert_eq!(d.budget(4, COST), 4);
        assert_eq!(d.budget(2, COST), 2);
        assert_eq!(d.budget(0, COST), 0);
    }

    /// A block accepted whole: positions 1–16 each move to (1.63, 1.7),
    /// `q ≈ 0.959`, and the score still rises through Flash-Next's ceiling of
    /// twelve. Past it the cost catches up: d=15 4.1368, d=16 4.1346.
    #[test]
    fn a_full_accept_drafts_deep() {
        let mut d = DraftDepth::default();
        d.record(16, 17);
        assert!(close(d.counts[0], (1.63, 1.7)));
        assert!(close(d.counts[15], (1.63, 1.7)));
        assert!(close(d.counts[16], PRIOR), "position 17 was not reached");
        assert_eq!(d.budget(12, COST), 12);
        assert_eq!(d.budget(16, COST), 15);
    }

    /// **Free text's cliff.** Three accepted and the fourth missed, three steps
    /// running: `q₁..q₃ ≈ 0.986`, `q₄ ≈ 0.122`. Scores d=3 2.8505, d=4 2.6910 —
    /// three, where one rate for every position (`p = 0.75`) would have drafted
    /// four.
    #[test]
    fn a_cliff_at_one_position_stops_the_depth_there() {
        let mut d = DraftDepth::default();
        for _ in 0..3 {
            d.record(8, 4);
        }
        assert!(close(d.counts[2], (2.4987, 2.533)));
        assert!(close(d.counts[3], (0.3087, 2.533)));
        assert!(close(d.counts[4], PRIOR), "position 5 was never reached");
        assert_eq!(d.budget(16, COST), 3);
    }

    /// A position that missed and is no longer reached ages back toward the
    /// prior: ten fully-accepted steps at depth 3 after the cliff above take
    /// `q₄` from 0.122 to 0.452, and the depth past it again — scores d=4
    /// 2.9667, d=5 2.9886, d=6 2.9842.
    #[test]
    fn an_unreached_miss_is_tried_again() {
        let mut d = DraftDepth::default();
        for _ in 0..3 {
            d.record(8, 4);
        }
        for _ in 0..10 {
            d.record(3, 4);
        }
        assert!(close(d.counts[3], (0.69383, 1.53452)));
        assert_eq!(d.budget(16, COST), 5);
    }

    /// The dearer a drafted token, the shallower the same acceptance drafts.
    #[test]
    fn a_dearer_token_drafts_shallower() {
        let d = DraftDepth::default();
        assert_eq!(d.budget(16, 0.3), 5);
    }

    /// Ten straight misses at the first proposal leave `q₁ ≈ 0.008`, and one
    /// proposal — never none, so the sequence can find out when its text turns
    /// predictable again.
    #[test]
    fn misses_shrink_to_one_proposal_not_none() {
        let mut d = DraftDepth::default();
        for _ in 0..10 {
            d.record(4, 1);
        }
        assert_eq!(d.budget(16, COST), 1);
    }

    /// The bonus token is not a kept proposal, and a block cannot keep more
    /// proposals than it drafted.
    #[test]
    fn kept_counts_proposals_only() {
        let mut d = DraftDepth::default();
        d.record(2, 3);
        assert!(close(d.counts[1], (1.63, 1.7)));
        assert!(close(d.counts[2], PRIOR));
        let mut e = DraftDepth::default();
        e.record(2, 9);
        assert_eq!(d, e);
    }

    /// A plain row teaches nothing.
    #[test]
    fn an_undrafted_step_is_not_recorded() {
        let mut d = DraftDepth::default();
        d.record(0, 1);
        assert_eq!(d, DraftDepth::default());
    }
}
