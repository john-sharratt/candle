//! How deep one sequence drafts, from how much of its drafting it keeps.
//!
//! A model's draft ladder ([`super::draft_ladder`]) says how deep drafting can
//! pay at a given wave width — measured on a rewrite, which the drafter predicts
//! almost perfectly. What a sequence actually keeps depends on what it is
//! writing: free continuation on Flash-Next accepts ~2.1 tokens a step against
//! the rewrite's 4.85. Every proposal past what the verify keeps is a step of the
//! drafter's serial walk and a verify row that bought nothing.
//!
//! So each sequence tracks its acceptance and drafts one past it: a sequence
//! that keeps everything grows toward the ladder's ceiling a token per step, and
//! one whose proposals keep missing shrinks toward a single proposal — never to
//! none, so a sequence whose text turns predictable can grow back.

/// Weight of the newest step in the running acceptance.
///
/// One half: the depth follows a change in what the sequence is writing within
/// a couple of steps — a reasoning span giving way to a quoted file is exactly
/// that change — while one unlucky step only moves it halfway.
const NEWEST_WEIGHT: f32 = 0.5;

/// One sequence's drafting depth, learned from its own verify steps.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DraftDepth {
    /// Running mean of drafted tokens kept per drafting step, or `None` before
    /// the first one.
    acceptance: Option<f32>,
}

impl DraftDepth {
    /// How many tokens to draft this step, never more than `ceiling`.
    ///
    /// A sequence with no history drafts the full ceiling: until it has been
    /// measured, the ladder's own figure is the best estimate there is.
    pub fn budget(&self, ceiling: usize) -> usize {
        match self.acceptance {
            None => ceiling,
            Some(a) => (a.round() as usize + 1).min(ceiling),
        }
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
        let accepted = kept.saturating_sub(1).min(drafted) as f32;
        self.acceptance = Some(match self.acceptance {
            None => accepted,
            Some(a) => a + NEWEST_WEIGHT * (accepted - a),
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unmeasured_sequence_drafts_the_ceiling() {
        let d = DraftDepth::default();
        assert_eq!(d.budget(4), 4);
        assert_eq!(d.budget(2), 2);
        assert_eq!(d.budget(0), 0);
    }

    /// The depth is the rounded acceptance plus one.
    #[test]
    fn the_depth_is_one_past_the_rounded_acceptance() {
        let mut d = DraftDepth::default();
        d.record(4, 3); // kept 2 proposals
        assert_eq!(d.acceptance, Some(2.0));
        assert_eq!(d.budget(4), 3);
        d.record(3, 2); // kept 1: 2.0 + 0.5 × (1 − 2) = 1.5
        assert_eq!(d.acceptance, Some(1.5));
        assert_eq!(d.budget(4), 3, "1.5 rounds away from zero, to 2");
        d.record(3, 2); // 1.5 + 0.5 × (1 − 1.5) = 1.25
        assert_eq!(d.acceptance, Some(1.25));
        assert_eq!(d.budget(4), 2);
    }

    /// A sequence that keeps every proposal grows a token a step up to the
    /// ceiling and stays there.
    #[test]
    fn a_full_accept_grows_to_the_ceiling() {
        let mut d = DraftDepth::default();
        d.record(1, 2);
        assert_eq!(d.budget(4), 2);
        d.record(2, 3); // 1 + 0.5 × (2 − 1) = 1.5 → 2 + 1
        assert_eq!(d.budget(4), 3);
        d.record(3, 4); // 1.5 + 0.5 × (3 − 1.5) = 2.25 → 3
        assert_eq!(d.budget(4), 3);
        d.record(3, 4); // 2.25 + 0.375 = 2.625 → 4
        assert_eq!(d.budget(4), 4);
        d.record(4, 5);
        assert_eq!(d.budget(4), 4, "clipped at the ceiling");
        assert_eq!(d.budget(2), 2, "and at a narrower one");
    }

    /// Nothing kept drafts one, never none — so the sequence can find out when
    /// its text turns predictable again.
    #[test]
    fn a_miss_shrinks_to_one_proposal_not_none() {
        let mut d = DraftDepth::default();
        d.record(4, 1);
        assert_eq!(d.acceptance, Some(0.0));
        assert_eq!(d.budget(4), 1);
    }

    /// The bonus token is not a kept proposal, and a block cannot keep more
    /// proposals than it drafted.
    #[test]
    fn kept_counts_proposals_only() {
        let mut d = DraftDepth::default();
        d.record(2, 3);
        assert_eq!(d.acceptance, Some(2.0));
        let mut e = DraftDepth::default();
        e.record(2, 9);
        assert_eq!(e.acceptance, Some(2.0));
    }

    /// A plain row teaches nothing.
    #[test]
    fn an_undrafted_step_is_not_recorded() {
        let mut d = DraftDepth::default();
        d.record(0, 1);
        assert_eq!(d, DraftDepth::default());
    }
}
