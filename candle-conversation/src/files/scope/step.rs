//! A range's entries from its sections' hit rates
//! (`docs/placed_section_layout.md` §3.4).
//!
//! **The opening** places the mandatory sections, then the sections the
//! question's prefill selects in descending rate, each that fits — the first
//! at the top, nearest the question. **At a step after it**, a section out of
//! scope at or above the entry bar enters in front, strongest first, sliding a
//! full range; which sections leave is the [`super::watch`]'s to say. A step's
//! decision is the layout the next step reads.

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

    /// Admit every candidate out of scope at or above the entry bar, strongest
    /// first, each in front; a section a full range slid out this step does
    /// not come back in it.
    pub fn admit(
        &mut self,
        rate: impl Fn(SectionKey) -> f32,
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
        let change = s.admit(rate, &[c(3, 10), c(4, 25), c(5, 5)], &Bars::DEFAULT);
        assert_eq!(change.entered, vec![SectionKey(3), SectionKey(4)]);
        assert_eq!(change.slid, vec![SectionKey(1)]);
        assert_eq!(keys(&s), vec![2, 3, 4]);
        assert!(change.left.is_empty());
    }
}
