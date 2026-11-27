//! A range's decisions from its sections' hit rates
//! (`docs/placed_section_layout.md` §3.4).
//!
//! **The opening** places the mandatory sections, then the sections the
//! question's prefill selects in descending rate, each that fits — the first
//! at the top, nearest the question. **Each step after it**, a placed section
//! under the eviction bar counts the step against it and leaves once it has
//! stayed there [`Bars::evict_steps`] steps; a section out of scope at or
//! above the entry bar enters in front, strongest first, sliding a full range.
//! A step's decision is the layout the next step reads.

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

    /// One step: every placed section's rate against the eviction bar, then
    /// every candidate's against the entry bar.
    pub fn step(
        &mut self,
        rate: impl Fn(SectionKey) -> f32,
        candidates: &[Candidate],
        bars: &Bars,
    ) -> StepChange {
        let mut change = StepChange::default();
        let mut leaving = Vec::new();
        for e in self.entries_mut() {
            if e.mandatory {
                continue;
            }
            if rate(e.key) < bars.evict {
                e.below += 1;
            } else {
                e.below = 0;
            }
            if e.below >= bars.evict_steps {
                leaving.push(e.key);
            }
        }
        for key in leaving {
            if self.leave(key) {
                change.left.push(key);
            }
        }
        let mut entering: Vec<(Candidate, f32)> = candidates
            .iter()
            .filter(|c| !self.holds(c.key) && !change.left.contains(&c.key))
            .map(|&c| (c, rate(c.key)))
            .filter(|&(_, r)| r >= bars.entry)
            .collect();
        entering.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.key.cmp(&b.0.key)));
        for (c, _) in entering {
            if change.slid.contains(&c.key) {
                continue;
            }
            if let Some(slid) = self.enter(c.key, c.tokens, false) {
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

    const BARS: Bars = Bars {
        entry: 800.0,
        evict: 600.0,
        evict_steps: 3,
    };

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

    /// **A section leaves only after staying under the eviction bar for its
    /// steps, and a step at or over it resets the count**; a mandatory one
    /// never counts.
    #[test]
    fn a_section_leaves_after_its_steps_under_the_bar() {
        let mut s = RangeScope::new(100);
        s.open(&[c(9, 10)], &[c(1, 20), c(2, 20)]);
        let rates = |low: u64| move |k: SectionKey| if k.0 == low { 100.0 } else { 700.0 };
        assert!(s.step(rates(1), &[], &BARS).is_empty());
        assert!(s.step(rates(1), &[], &BARS).is_empty());
        // A step at the bar resets 1's count.
        assert!(s.step(rates(0), &[], &BARS).is_empty());
        assert!(s.step(rates(1), &[], &BARS).is_empty());
        assert!(s.step(rates(1), &[], &BARS).is_empty());
        let change = s.step(rates(1), &[], &BARS);
        assert_eq!(change.left, vec![SectionKey(1)]);
        assert_eq!(keys(&s), vec![9, 2]);
        // 9 is mandatory: under the bar for ever, it stays.
        for _ in 0..10 {
            s.step(rates(9), &[], &BARS);
        }
        assert_eq!(keys(&s), vec![9, 2]);
    }

    /// **Candidates at or over the entry bar enter in front, strongest
    /// first; one under it does not; a full range slides**: in a 60-range
    /// holding a (30) and b (20), c (900) and d (850) enter — c in front of b,
    /// then d in front of c, sliding a out past the top; e (799) does not.
    #[test]
    fn candidates_over_the_bar_enter_in_front_strongest_first() {
        let mut s = RangeScope::new(60);
        s.open(&[], &[c(1, 30), c(2, 20)]);
        let rate = |k: SectionKey| match k.0 {
            3 => 900.0,
            4 => 850.0,
            5 => 799.0,
            _ => 700.0,
        };
        let change = s.step(rate, &[c(3, 10), c(4, 25), c(5, 5)], &BARS);
        assert_eq!(change.entered, vec![SectionKey(3), SectionKey(4)]);
        assert_eq!(change.slid, vec![SectionKey(1)]);
        assert_eq!(keys(&s), vec![2, 3, 4]);
        assert!(change.left.is_empty());
    }
}
