//! One candidate's **background**: the reference-scaled scores it reached on
//! probes it was not the known answer to, kept as counts in log-spaced bins.
//!
//! A score is read against it as tail surprise, `-ln P(background ≥ score)`:
//! under the null — a probe not about the candidate — every candidate's
//! surprise follows the same distribution, so a loud candidate (a hub that
//! scores on everything) wins no more often by chance than a quiet one, and a
//! genuine hit sits in its own far tail.
//!
//! Counts commute, so what a background holds — and every score read against
//! it — is the same whatever order its evidence arrived in. The hit-level EWMA
//! this replaced depended on that order: a warm-up that folded each tool's
//! exemplars in catalog order left the early tools' levels decayed to their
//! background and the last tools' at their hits, so a tool's score followed its
//! name's place in the alphabet.

/// Bins per nat of `ln(1 + score)`: a bin spans a 13% step in the score.
pub(super) const BINS_PER_NAT: f32 = 8.0;

/// Whether `x` is a score of any evidence: positive. None, a negative, or not
/// a number is none.
fn is_evidence(x: f32) -> bool {
    x > 0.0
}

/// Where `x > 0` falls on the bin axis.
fn position(x: f32) -> f32 {
    x.ln_1p() * BINS_PER_NAT
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Background {
    /// Samples that scored nothing.
    zeros: u32,
    /// `(bin, count)` of the samples that scored, ascending by bin.
    bins: Vec<(u16, u32)>,
    total: u32,
}

impl Background {
    /// A background holding exactly `scores`.
    pub(super) fn of(scores: impl IntoIterator<Item = f32>) -> Background {
        let mut b = Background::default();
        for x in scores {
            b.add(x);
        }
        b
    }

    /// One more sample. A score that is not positive — none, or not a number —
    /// is a sample of no evidence.
    pub(super) fn add(&mut self, x: f32) {
        self.total = self.total.saturating_add(1);
        if !is_evidence(x) {
            self.zeros = self.zeros.saturating_add(1);
            return;
        }
        let bin = position(x).min(u16::MAX as f32) as u16;
        match self.bins.binary_search_by_key(&bin, |(b, _)| *b) {
            Ok(i) => self.bins[i].1 = self.bins[i].1.saturating_add(1),
            Err(i) => self.bins.insert(i, (bin, 1)),
        }
    }

    /// How many samples it holds.
    pub(super) fn total(&self) -> u32 {
        self.total
    }

    /// How many samples score at least `x`, counting `x`'s own bin in
    /// proportion to how far into it `x` sits.
    fn at_least(&self, x: f32) -> f32 {
        if !is_evidence(x) {
            return self.total as f32;
        }
        let p = position(x);
        let bin = p.floor();
        let into = p - bin;
        let mut n = 0.0;
        for &(b, count) in self.bins.iter().rev() {
            let b = b as f32;
            if b > bin {
                n += count as f32;
            } else {
                if b == bin {
                    n += count as f32 * (1.0 - into);
                }
                break;
            }
        }
        n
    }

    /// `-ln P(background ≥ x)`, add-one smoothed. Past the top of the highest
    /// bin it goes on rising a nat for every nat of `ln(1 + x)`, so scores
    /// beyond everything the background holds still rank by size. Zero for a
    /// score of no evidence.
    pub(super) fn surprise(&self, x: f32) -> f32 {
        if !is_evidence(x) {
            return 0.0;
        }
        let n = self.total as f32;
        let tail = ((n + 1.0) / (self.at_least(x) + 1.0)).ln();
        let edge = self.bins.last().map_or(0.0, |(b, _)| *b as f32 + 1.0);
        let beyond = (position(x) - edge).max(0.0) / BINS_PER_NAT;
        tail + beyond
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(got: f32, want: f32, tol: f32) {
        assert!(
            (got - want).abs() <= tol,
            "expected ~{want}, got {got} (tol {tol})"
        );
    }

    /// `x` at exactly `bins` on the bin axis.
    fn at(bins: f32) -> f32 {
        (bins / BINS_PER_NAT).exp() - 1.0
    }

    /// **A zero is no evidence**, and a sample of zero counts in the total.
    #[test]
    fn a_zero_is_no_evidence() {
        let b = Background::of([0.0, f32::NAN, 3.0]);
        assert_eq!(b.total(), 3);
        assert_eq!(b.zeros, 2);
        assert_eq!(b.surprise(0.0), 0.0);
        assert_eq!(b.surprise(-1.0), 0.0);
    }

    /// Samples land in log-spaced bins: 1.0 at `8 ln 2 = 5.55`, bin 5; 10.0 at
    /// `8 ln 11 = 19.18`, bin 19.
    #[test]
    fn samples_land_in_log_spaced_bins() {
        let b = Background::of([1.0, 1.0, 10.0]);
        assert_eq!(b.bins, vec![(5, 2), (19, 1)]);
    }

    /// **The tail is read off the counts**: against 99 samples of 1.0, a score
    /// at the top of their bin is beyond all of them — `ln(100 / 1)` — and a
    /// score of 1.0 itself is 54.5% into its bin, so 99 × 0.4548 = 45.03
    /// samples are at least it: `ln(100 / 46.03) = 0.7759`.
    #[test]
    fn the_tail_is_read_off_the_counts() {
        let b = Background::of(std::iter::repeat_n(1.0, 99));
        approx(b.surprise(at(6.0)), 100f32.ln(), 1e-4);
        approx(b.surprise(1.0), 0.7759, 1e-3);
        approx(b.surprise(0.5), 0.0, 1e-6);
    }

    /// **Beyond the highest bin a score rises a nat per nat**, from the tail at
    /// the bin's top, so two scores past everything still rank by size.
    #[test]
    fn past_the_background_a_score_still_ranks_by_size() {
        let b = Background::of(std::iter::repeat_n(1.0, 99));
        approx(b.surprise(at(14.0)), 100f32.ln() + 1.0, 1e-3);
        assert!(b.surprise(at(20.0)) > b.surprise(at(14.0)));
        // An empty background: nothing to beat but the axis itself.
        let empty = Background::default();
        approx(empty.surprise(at(8.0)), 1.0, 1e-4);
    }

    /// **The same evidence in any order is the same background.**
    #[test]
    fn the_order_of_the_evidence_changes_nothing() {
        let scores = [0.0, 4.0, 120.0, 4.5, 0.0, 33.0, 4.0, 900.0];
        let mut reversed = scores;
        reversed.reverse();
        assert_eq!(Background::of(scores), Background::of(reversed));
    }
}
