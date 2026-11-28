//! Each section's momentum over the decode (`docs/placed_section_layout.md`
//! §3.2, §3.4): an exponential average of its per-step hit rate — a lift
//! over the step's own selection, 1 for a section read as any block of the
//! slot is — which orders the range, strongest at the top beside the
//! question. Every bar here is in those units, so it holds whatever the
//! prompt's length, the selection's budget or the model.
//!
//! A section takes its first rate whole, so the order locks on from the first
//! step that reads it; after that each step moves it a quarter of the way to
//! its rate, so a section the decode keeps reading climbs to the top within a
//! few steps and one it stops reading sinks back over several.
//!
//! At a step a section the index reads strongly — its momentum at or over
//! [`LOCK_BAR`] — is locked to its positions, and the rest move around it
//! (`RangeScope::reorder_locked`). It unlocks once its momentum falls under
//! the bar, and only then moves.

use std::collections::HashMap;

use crate::files::key::SectionKey;

/// The share of a section's momentum each step keeps; the rest is the step's
/// rate.
pub const MOMENTUM_KEEP: f32 = 0.75;

/// How far a section's momentum must pass the one above it before it takes
/// its place, as a lift: neighbours a hair apart do not trade places step
/// after step. The top of the noise a section read no more than chance shows.
pub const REORDER_MARGIN: f32 = 2.5;

/// The hit momentum at or above which a section is locked to its positions at
/// a step's reorder, as a lift: the index reads it strongly. Under it the
/// section is unlocked, and drifts by momentum like any other. Measured on
/// BlackWorld (2026-11-28, the depth-1 battery at 16 slots, the selection
/// reading 16–25 on the band): a range holds one to five sections the decode
/// reads at 8–30 times chance, a band of five to ten read mildly at 2.5–8,
/// and a tail under 2.5. The bar is where the strong part from the mild; set
/// inside the mild band, the band hovered on it and flipped in and out of the
/// lock.
pub const LOCK_BAR: f32 = 7.5;

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
