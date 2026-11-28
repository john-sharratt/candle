//! Each section's momentum over the decode (`docs/placed_section_layout.md`
//! §3.2, §3.4): an exponential average of its per-step hit rate — the share
//! of its blocks the queries selected, on the 0–1000 band — which orders the
//! range, strongest at the top beside the question. The decisions are taken
//! on the momenta as shares of the range's (`super::nucleus`), so no bar is
//! set on the band itself.
//!
//! A section takes its first rate whole, so the order locks on from the first
//! step that reads it; after that each step moves it a quarter of the way to
//! its rate, so a section the decode keeps reading climbs to the top within a
//! few steps and one it stops reading sinks back over several.

use std::collections::HashMap;

use crate::files::key::SectionKey;

/// The share of a section's momentum each step keeps; the rest is the step's
/// rate.
pub const MOMENTUM_KEEP: f32 = 0.75;

/// Every section's momentum, for the slot's life.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Momentum {
    score: HashMap<SectionKey, f32>,
}

impl Momentum {
    /// Fold one step's `rates` in: a section seen for the first time takes
    /// its rate, one seen before keeps [`MOMENTUM_KEEP`] of its momentum and
    /// takes the rest from the rate. A section the step did not rate keeps
    /// its momentum.
    pub fn observe(&mut self, rates: impl IntoIterator<Item = (SectionKey, f32)>) {
        for (key, rate) in rates {
            self.score
                .entry(key)
                .and_modify(|m| *m = MOMENTUM_KEEP * *m + (1.0 - MOMENTUM_KEEP) * rate)
                .or_insert(rate);
        }
    }

    /// `key`'s momentum; `None` before any step has rated it.
    pub fn score(&self, key: SectionKey) -> Option<f32> {
        self.score.get(&key).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A first rate is taken whole; after it each step moves a quarter of
    /// the way**: 800, then 0.75·800 + 0.25·0 = 600, then 0.75·600 + 0.25·1000
    /// = 700; an unrated section keeps what it had.
    #[test]
    fn a_first_rate_is_whole_and_each_step_moves_a_quarter() {
        let (a, b) = (SectionKey(1), SectionKey(2));
        let mut m = Momentum::default();
        assert_eq!(m.score(a), None);
        m.observe([(a, 800.0), (b, 100.0)]);
        assert_eq!(m.score(a), Some(800.0));
        m.observe([(a, 0.0)]);
        assert_eq!(m.score(a), Some(600.0));
        m.observe([(a, 1000.0)]);
        assert_eq!(m.score(a), Some(700.0));
        assert_eq!(m.score(b), Some(100.0), "unrated, it keeps its momentum");
    }
}
