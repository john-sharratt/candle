//! A range's entries from its sections' hit rates
//! (`docs/placed_section_layout.md` §3.4).
//!
//! **The opening** places the mandatory sections, then the ranked sections,
//! strongest first, each that fits; the projection then orders the range by
//! relevance, the mandatory ones among the rest. **At a step after it**, a
//! section out of scope at or above the entry bar enters at its rank by
//! momentum, strongest first, a full range making room only past weaker
//! sections; which sections leave is the [`super::watch`]'s to say. A step's
//! decision is the layout the next step reads.

use std::collections::HashMap;

use super::bars::Bars;
use super::range_scope::RangeScope;
use crate::files::key::SectionKey;

/// A section that may enter: its key and its positions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Candidate {
    pub key: SectionKey,
    pub tokens: usize,
}

/// What one decision changed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StepChange {
    /// Entered in front, in the order they entered.
    pub entered: Vec<SectionKey>,
    /// Left, each leaving a hole.
    pub left: Vec<SectionKey>,
    /// Passed the top of a full range.
    pub slid: Vec<SectionKey>,
}

impl StepChange {
    /// Whether the layout is as it was.
    pub fn is_empty(&self) -> bool {
        self.entered.is_empty() && self.left.is_empty() && self.slid.is_empty()
    }
}

impl RangeScope {
    /// Open an empty range on `mandatory` — in their order — then `ranked`,
    /// strongest first: each that fits in what is left enters, the first at
    /// the top. Nothing slides at the opening.
    pub fn open(&mut self, mandatory: &[Candidate], ranked: &[Candidate]) -> StepChange {
        let mut change = StepChange::default();
        for (c, is_mandatory) in mandatory
            .iter()
            .map(|c| (c, true))
            .chain(ranked.iter().map(|c| (c, false)))
        {
            if self.holds(c.key) || c.tokens > self.positions.saturating_sub(self.used()) {
                continue;
            }
            if self.enter(c.key, c.tokens, is_mandatory).is_some() {
                change.entered.push(c.key);
            }
        }
        change
    }

    /// Admit every candidate out of scope at or above the entry bar by its
    /// step `rate`, strongest first, each at its rank by `score` — the placed
    /// sections' momentum, a candidate's own or its rate when it has none — a
    /// full range making room only past weaker sections, and never moving or
    /// cutting a `locked` one ([`RangeScope::enter_over`]); a section pushed
    /// out this step does not come back in it.
    pub fn admit(
        &mut self,
        rate: impl Fn(SectionKey) -> f32,
        score: impl Fn(SectionKey) -> Option<f32>,
        locked: impl Fn(SectionKey) -> bool,
        candidates: &[Candidate],
        bars: &Bars,
    ) -> StepChange {
        let mut change = StepChange::default();
        let mut entering: Vec<(Candidate, f32)> = candidates
            .iter()
            .filter(|c| !self.holds(c.key))
            .map(|&c| (c, rate(c.key)))
            .filter(|&(_, r)| r >= bars.entry)
            .collect();
        entering.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.key.cmp(&b.0.key)));
        let mut entered: HashMap<SectionKey, f32> = HashMap::new();
        for (c, _) in entering {
            if change.slid.contains(&c.key) {
                continue;
            }
            let value = score(c.key).unwrap_or_else(|| rate(c.key));
            // A section entered earlier this step ranks by the value it entered
            // at, its momentum not yet holding it.
            let ranked = |k: SectionKey| entered.get(&k).copied().or_else(|| score(k));
            if let Some(slid) = self.enter_over(c.key, c.tokens, value, ranked, &locked) {
                entered.insert(c.key, value);
                change.entered.push(c.key);
                change.slid.extend(slid);
            }
        }
        change
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(key: u64, tokens: usize) -> Candidate {
        Candidate {
            key: SectionKey(key),
            tokens,
        }
    }

    fn keys(s: &RangeScope) -> Vec<u64> {
        s.sections().map(|e| e.key.0).collect()
    }

    /// **The opening places the mandatory sections, then the ranked ones that
    /// fit, the first at the top**: m (30), then a (40) and c (20); b (50)
    /// does not fit the 30 left after a and is passed over.
    #[test]
    fn the_opening_places_mandatory_then_ranked_that_fit() {
        let mut s = RangeScope::new(100);
        let change = s.open(&[c(9, 30)], &[c(1, 40), c(2, 50), c(3, 20)]);
        assert_eq!(
            change.entered,
            vec![SectionKey(9), SectionKey(1), SectionKey(3)]
        );
        assert_eq!(keys(&s), vec![9, 1, 3]);
        assert!(s.entry(SectionKey(9)).unwrap().mandatory);
    }

    /// **Candidates at or over the entry bar enter at their rank, strongest
    /// first; one under it does not; a full range makes room past weaker
    /// sections only**: in a 60-range holding a (30, momentum 700) and b (20,
    /// momentum 300), c (900) enters above a, then d (850) pushes b out from
    /// the front and takes its rank between c and a; e (799) is under the bar.
    #[test]
    fn candidates_over_the_bar_enter_at_their_rank() {
        let mut s = RangeScope::new(60);
        s.open(&[], &[c(1, 30), c(2, 20)]);
        let rate = |k: SectionKey| match k.0 {
            3 => 900.0,
            4 => 850.0,
            5 => 799.0,
            _ => 0.0,
        };
        let momentum = |k: SectionKey| match k.0 {
            1 => Some(700.0),
            2 => Some(300.0),
            _ => None,
        };
        let change = s.admit(
            rate,
            momentum,
            |_| false,
            &[c(3, 10), c(4, 20), c(5, 5)],
            &Bars::DEFAULT,
        );
        assert_eq!(change.entered, vec![SectionKey(3), SectionKey(4)]);
        assert_eq!(change.slid, vec![SectionKey(2)]);
        assert_eq!(keys(&s), vec![3, 4, 1]);
        assert!(change.left.is_empty());
    }

    /// **A full range does not give way to a weaker entry**: a (30, 900) and
    /// b (30, 820) fill a 60-range; c (810, over the bar) would need b's room
    /// and is weaker, so it does not enter.
    #[test]
    fn a_full_range_holds_against_a_weaker_entry() {
        let mut s = RangeScope::new(60);
        s.open(&[], &[c(1, 30), c(2, 30)]);
        let momentum = |k: SectionKey| match k.0 {
            1 => Some(900.0),
            2 => Some(820.0),
            _ => None,
        };
        let change = s.admit(|_| 810.0, momentum, |_| false, &[c(3, 10)], &Bars::DEFAULT);
        assert!(change.entered.is_empty());
        assert_eq!(keys(&s), vec![1, 2]);
    }

    /// **An entry never moves or cuts a locked section**: in a 60-range a
    /// (20, locked at the top, momentum 100) and b (20, 300) leave a 20-hole
    /// at the front. c (40, 900) needs b's room: it cuts b, the weaker, from
    /// the front, and takes its rank — above b's place, but a, locked, keeps
    /// the top positions. d (45, 950) would need c's room and a's, and does
    /// not enter.
    #[test]
    fn an_entry_never_moves_or_cuts_a_locked_section() {
        let mut s = RangeScope::new(60);
        s.open(&[], &[c(1, 20), c(2, 20)]);
        let momentum = |k: SectionKey| match k.0 {
            1 => Some(100.0),
            2 => Some(300.0),
            _ => None,
        };
        let locked = |k: SectionKey| k.0 == 1;
        let rate = |k: SectionKey| match k.0 {
            3 => 900.0,
            4 => 950.0,
            _ => 0.0,
        };
        let change = s.admit(rate, momentum, locked, &[c(3, 40)], &Bars::DEFAULT);
        assert_eq!(change.entered, vec![SectionKey(3)]);
        assert_eq!(change.slid, vec![SectionKey(2)]);
        assert_eq!(keys(&s), vec![1, 3], "a keeps the top; c below it");
        let held = s.admit(rate, momentum, locked, &[c(4, 45)], &Bars::DEFAULT);
        assert!(held.entered.is_empty(), "d would need a's room");
        assert_eq!(keys(&s), vec![1, 3]);
    }
}
