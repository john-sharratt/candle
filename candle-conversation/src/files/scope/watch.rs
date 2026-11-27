//! Which placed sections leave a range (`docs/placed_section_layout.md`
//! §3.4): each step, a placed section whose hit rate is under the eviction
//! bar counts the step against it, one at or over it starts again, and one
//! that has stayed under for [`Bars::evict_steps`] steps leaves. Mandatory
//! sections are never watched.
//!
//! The counts are the decode's own, kept beside the slot and never on the
//! substrate: a step that changes no layout publishes nothing.

use std::collections::HashMap;

use super::bars::Bars;
use super::range_scope::RangeScope;
use crate::files::key::SectionKey;

/// The steps each placed section has stayed under the eviction bar.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct EvictWatch {
    below: HashMap<SectionKey, u32>,
}

impl EvictWatch {
    /// One step of `scope`'s placed sections at `rate`: the sections that
    /// leave now, top first. Each is forgotten here; the caller has it leave.
    pub fn observe(
        &mut self,
        scope: &RangeScope,
        rate: impl Fn(SectionKey) -> f32,
        bars: &Bars,
    ) -> Vec<SectionKey> {
        let mut leaving = Vec::new();
        let mut watched: HashMap<SectionKey, u32> = HashMap::new();
        for e in scope.sections().filter(|e| !e.mandatory) {
            let steps = match rate(e.key) < bars.evict {
                true => self.below.get(&e.key).copied().unwrap_or(0) + 1,
                false => 0,
            };
            if steps >= bars.evict_steps {
                leaving.push(e.key);
            } else {
                watched.insert(e.key, steps);
            }
        }
        self.below = watched;
        leaving
    }

    /// The steps `key` has stayed under the bar.
    pub fn steps_below(&self, key: SectionKey) -> u32 {
        self.below.get(&key).copied().unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const BARS: Bars = Bars {
        entry: 800.0,
        evict: 600.0,
        evict_steps: 3,
    };

    /// **A section leaves only after staying under the eviction bar for its
    /// steps, and a step at or over it starts the count again**; a mandatory
    /// one is never watched.
    #[test]
    fn a_section_leaves_after_its_steps_under_the_bar() {
        let mut s = RangeScope::new(100);
        s.enter(SectionKey(9), 10, true);
        s.enter(SectionKey(1), 20, false);
        s.enter(SectionKey(2), 20, false);
        let low =
            |low: u64| move |k: SectionKey| if k.0 == low || k.0 == 9 { 100.0 } else { 700.0 };
        let mut w = EvictWatch::default();
        assert!(w.observe(&s, low(1), &BARS).is_empty());
        assert!(w.observe(&s, low(1), &BARS).is_empty());
        assert_eq!(w.steps_below(SectionKey(1)), 2);
        // A step at the bar starts 1's count again.
        assert!(w.observe(&s, low(0), &BARS).is_empty());
        assert_eq!(w.steps_below(SectionKey(1)), 0);
        assert!(w.observe(&s, low(1), &BARS).is_empty());
        assert!(w.observe(&s, low(1), &BARS).is_empty());
        assert_eq!(w.observe(&s, low(1), &BARS), vec![SectionKey(1)]);
        assert_eq!(w.steps_below(SectionKey(1)), 0, "forgotten once it leaves");
        assert_eq!(
            w.steps_below(SectionKey(9)),
            0,
            "a mandatory one is never watched"
        );
    }
}
