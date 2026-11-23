//! The battery's statistics (`docs/unbounded_agents.md` §9.12, *Scoring and
//! statistics*): a percentile bootstrap over questions, the Wilcoxon
//! signed-rank test on paired per-question differences, and Holm's step-down
//! correction across the contrasts.
//!
//! Everything is deterministic: the bootstrap draws from a SplitMix64 stream
//! seeded by the caller, so a figure's interval is reproduced exactly from the
//! same results.

/// SplitMix64 (Steele, Lea and Flood, 2014): a 64-bit generator whose whole
/// state is one word, so a seed names its stream.
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A draw in `0..n`, by Lemire's multiply-shift (its bias at `n` of a few
    /// hundred is below 2⁻⁵⁰).
    pub fn below(&mut self, n: usize) -> usize {
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }
}

pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

/// The `q`-quantile of sorted `xs`, interpolated linearly between the two
/// order statistics around it (the R-7 definition).
pub fn quantile(sorted: &[f64], q: f64) -> f64 {
    let h = (sorted.len() - 1) as f64 * q;
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
}

/// The 95% percentile-bootstrap interval of the mean of `xs`, over
/// `resamples` resamples drawn from `seed`'s stream.
pub fn bootstrap_mean_interval(xs: &[f64], resamples: usize, seed: u64) -> (f64, f64) {
    let mut rng = SplitMix64::new(seed);
    let mut means: Vec<f64> = (0..resamples)
        .map(|_| (0..xs.len()).map(|_| xs[rng.below(xs.len())]).sum::<f64>() / xs.len() as f64)
        .collect();
    means.sort_by(f64::total_cmp);
    (quantile(&means, 0.025), quantile(&means, 0.975))
}

/// A Wilcoxon signed-rank test's result.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wilcoxon {
    /// The differences that were not zero: the test's sample size.
    pub n: usize,
    /// The sum of the ranks of the positive differences.
    pub w_plus: f64,
    /// The normal approximation's statistic, continuity-corrected.
    pub z: f64,
    /// The two-sided p-value.
    pub p: f64,
}

/// The ranks of `xs`, 1-based, ties given the mean of the ranks they span.
pub fn ranks(xs: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..xs.len()).collect();
    order.sort_by(|&a, &b| xs[a].total_cmp(&xs[b]));
    let mut out = vec![0.0; xs.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && xs[order[j + 1]] == xs[order[i]] {
            j += 1;
        }
        let rank = (i + j) as f64 / 2.0 + 1.0;
        for &k in &order[i..=j] {
            out[k] = rank;
        }
        i = j + 1;
    }
    out
}

/// The Wilcoxon signed-rank test of `diffs` against a median of zero: zero
/// differences dropped (Wilcoxon's own treatment), tied magnitudes given
/// their mean rank, and the variance reduced for the ties. The p-value is the
/// two-sided normal approximation with a continuity correction, which at the
/// battery's 128 questions is within a few thousandths of the exact one.
pub fn wilcoxon(diffs: &[f64]) -> Wilcoxon {
    let nonzero: Vec<f64> = diffs.iter().copied().filter(|d| *d != 0.0).collect();
    let n = nonzero.len();
    if n == 0 {
        return Wilcoxon {
            n,
            w_plus: 0.0,
            z: 0.0,
            p: 1.0,
        };
    }
    let magnitudes: Vec<f64> = nonzero.iter().map(|d| d.abs()).collect();
    let r = ranks(&magnitudes);
    let w_plus: f64 = nonzero
        .iter()
        .zip(&r)
        .filter(|(d, _)| **d > 0.0)
        .map(|(_, r)| r)
        .sum();
    let nf = n as f64;
    let expected = nf * (nf + 1.0) / 4.0;
    // Each run of t tied magnitudes lowers the variance by (t³ − t) / 48.
    let mut sorted = magnitudes.clone();
    sorted.sort_by(f64::total_cmp);
    let mut ties = 0.0;
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i;
        while j + 1 < sorted.len() && sorted[j + 1] == sorted[i] {
            j += 1;
        }
        let t = (j - i + 1) as f64;
        ties += t * t * t - t;
        i = j + 1;
    }
    let variance = nf * (nf + 1.0) * (2.0 * nf + 1.0) / 24.0 - ties / 48.0;
    let off = w_plus - expected;
    let corrected = (off.abs() - 0.5).max(0.0) * off.signum();
    let z = if variance > 0.0 {
        corrected / variance.sqrt()
    } else {
        0.0
    };
    Wilcoxon {
        n,
        w_plus,
        z,
        p: erfc(z.abs() / std::f64::consts::SQRT_2).min(1.0),
    }
}

/// Holm's step-down adjustment of `ps`, in their own order: the k-th smallest
/// is multiplied by the number not yet passed, and every adjusted value kept
/// at least the one before it.
pub fn holm(ps: &[f64]) -> Vec<f64> {
    let m = ps.len();
    let mut order: Vec<usize> = (0..m).collect();
    order.sort_by(|&a, &b| ps[a].total_cmp(&ps[b]));
    let mut out = vec![0.0; m];
    let mut floor: f64 = 0.0;
    for (k, &i) in order.iter().enumerate() {
        floor = floor.max(((m - k) as f64 * ps[i]).min(1.0));
        out[i] = floor;
    }
    out
}

/// The complementary error function, by the Chebyshev fit of *Numerical
/// Recipes* §6.2 (`erfcc`): fractional error below 1.2 × 10⁻⁷ everywhere.
pub fn erfc(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.5 * z);
    let poly = -z * z - 1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    let ans = t * poly.exp();
    if x >= 0.0 {
        ans
    } else {
        2.0 - ans
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published SplitMix64 stream from seed 0.
    #[test]
    fn splitmix_reproduces_its_reference_stream() {
        let mut r = SplitMix64::new(0);
        assert_eq!(r.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(r.next_u64(), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(r.next_u64(), 0x06c4_5d18_8009_454f);
    }

    #[test]
    fn a_draw_below_n_stays_below_n() {
        let mut r = SplitMix64::new(7);
        assert!((0..10_000).all(|_| r.below(3) < 3));
        assert_eq!(SplitMix64::new(1).below(1), 0);
    }

    #[test]
    fn quantiles_interpolate_between_order_statistics() {
        let s = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(quantile(&s, 0.0), 1.0);
        assert_eq!(quantile(&s, 0.5), 3.0);
        assert_eq!(quantile(&s, 0.25), 2.0);
        assert_eq!(quantile(&s, 0.1), 1.4);
    }

    #[test]
    fn a_constant_sample_has_a_point_interval_and_a_seed_reproduces_one() {
        assert_eq!(bootstrap_mean_interval(&[0.5; 10], 1_000, 3), (0.5, 0.5));
        let xs: Vec<f64> = (0..50).map(|i| (i % 7) as f64 / 6.0).collect();
        let a = bootstrap_mean_interval(&xs, 2_000, 11);
        assert_eq!(a, bootstrap_mean_interval(&xs, 2_000, 11));
        assert!(a.0 < mean(&xs) && mean(&xs) < a.1);
    }

    #[test]
    fn ties_share_their_mean_rank() {
        assert_eq!(ranks(&[3.0, 1.0, 2.0]), vec![3.0, 1.0, 2.0]);
        assert_eq!(ranks(&[1.0, 2.0, 2.0, 5.0]), vec![1.0, 2.5, 2.5, 4.0]);
    }

    /// Ranks of |d| = 3,1,2,5,4 are 3,1,2,5,4; the positive ones sum to
    /// 3 + 2 + 5 = 10. E[W⁺] = 7.5, Var = 5·6·11/24 = 13.75, so
    /// z = (10 − 7.5 − 0.5)/√13.75 = 0.539360 and p = erfc(z/√2) = 0.589639.
    #[test]
    fn the_signed_rank_test_matches_its_hand_computation() {
        let w = wilcoxon(&[3.0, -1.0, 2.0, 5.0, -4.0, 0.0]);
        assert_eq!((w.n, w.w_plus), (5, 10.0));
        assert!((w.z - 0.539_360).abs() < 1e-6, "{}", w.z);
        assert!((w.p - 0.589_640).abs() < 2e-5, "{}", w.p);
        assert_eq!(wilcoxon(&[0.0, 0.0]).p, 1.0);
    }

    /// |d| = 1,1,2: ranks 1.5,1.5,3, all positive, W⁺ = 6; E = 3,
    /// Var = 3·4·7/24 − (8 − 2)/48 = 3.375, z = 2.5/√3.375 = 1.360828.
    #[test]
    fn tied_magnitudes_reduce_the_variance() {
        let w = wilcoxon(&[1.0, 1.0, 2.0]);
        assert_eq!(w.w_plus, 6.0);
        assert!((w.z - 1.360_828).abs() < 1e-6, "{}", w.z);
    }

    #[test]
    fn holm_steps_down_and_stays_monotone() {
        let adj = holm(&[0.01, 0.04, 0.03, 0.005, 0.2]);
        let want = [0.04, 0.09, 0.09, 0.025, 0.2];
        for (a, w) in adj.iter().zip(want) {
            assert!((a - w).abs() < 1e-12, "{adj:?}");
        }
        assert_eq!(holm(&[0.6, 0.9]), vec![1.0, 1.0]);
    }

    #[test]
    fn erfc_matches_its_table() {
        assert!((erfc(0.0) - 1.0).abs() < 1e-7);
        assert!((erfc(1.0) - 0.157_299_207).abs() < 1e-7);
        assert!((erfc(-1.0) - 1.842_700_793).abs() < 1e-7);
        assert!((erfc(2.0) - 0.004_677_735).abs() < 1e-8);
    }
}
