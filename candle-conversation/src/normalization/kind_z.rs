//! A competition's scores read against the scores of their own kind on the
//! same probe, put on the band.
//!
//! The background normalizer ([`super::NormalizationCache`]) reads a score
//! against what that one candidate reached on probes not about it. The code
//! files of a workspace compete with its documents, and a document scores on
//! nearly every natural-language question, so documents took half the top ten
//! of a question about code. Reading each file as a z-score among the scoring
//! files of its own kind (its extension) on this probe lets a document and a
//! source file meet on equal terms with no weight set by hand: MRR 0.532 on the
//! dependency battery against 0.522 for a hand-set ×0.25 on documents, and the
//! background (other files' questions) read the same way scored 0.326
//! (`docs/results/provenance_scoring_lab_2026-11-16.md`).
//!
//! The z-score goes onto the band through the normal tail, so it reads as the
//! background normalizer's band does: `1000 × −ln P(Z ≥ z) / ln 100`. 1000 is a
//! score as far above its kind as 1 in 100, a file at its kind's mean reads
//! about 150, and a score of no evidence reads 0, below every scoring file.
//!
//! **A kind too small to have a spread is read against every kind.** A mean
//! and a spread over a handful of files say nothing about any of them: a lone
//! file reads at its own mean, z 0, however far its probe lifted it, and two
//! read ±1 whatever their scores. A kind with fewer than [`MIN_KIND_SAMPLE`]
//! scoring files — a workspace's few `Makefile`s, its one `rust-toolchain.toml`
//! among its manifests — is read against the scoring files of every kind
//! together, which is the competition it would be in with no kinds at all.

/// The fewest scoring files a kind is read against itself with; a smaller
/// kind is read against every kind's.
pub const MIN_KIND_SAMPLE: usize = 8;

/// A sum, a sum of squares and a count: what a mean and a spread come from.
#[derive(Clone, Copy, Default)]
struct Moments {
    sum: f64,
    sq: f64,
    n: usize,
}

impl Moments {
    fn add(&mut self, x: f64) {
        self.sum += x;
        self.sq += x * x;
        self.n += 1;
    }

    /// `x`'s z-score among these; 0 with fewer than two.
    fn z(&self, x: f64) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        let n = self.n as f64;
        let mean = self.sum / n;
        let var = (self.sq / n - mean * mean).max(0.0);
        (x - mean) / var.sqrt().max(f64::EPSILON)
    }
}

/// Each of `raw`'s scores as a z-score on the band. `kinds[i]` is score `i`'s
/// kind. A kind with at least [`MIN_KIND_SAMPLE`] scoring (positive) members
/// is read against them; a smaller one against every kind's scoring members
/// together. A score at or below 0 reads 0.
pub fn kind_z_band<K: AsRef<str>>(raw: &[f32], kinds: &[K]) -> Vec<f32> {
    let mut all = Moments::default();
    let mut stats: Vec<(&str, Moments)> = Vec::new();
    for (x, k) in raw.iter().zip(kinds) {
        if *x <= 0.0 {
            continue;
        }
        let (k, x) = (k.as_ref(), *x as f64);
        all.add(x);
        match stats.iter_mut().find(|s| s.0 == k) {
            Some(s) => s.1.add(x),
            None => {
                let mut m = Moments::default();
                m.add(x);
                stats.push((k, m));
            }
        }
    }
    raw.iter()
        .zip(kinds)
        .map(|(x, k)| {
            if *x <= 0.0 {
                return 0.0;
            }
            let own = stats.iter().find(|s| s.0 == k.as_ref()).map(|s| s.1);
            let against = match own {
                Some(m) if m.n >= MIN_KIND_SAMPLE => m,
                _ => all,
            };
            band_of_z(against.z(*x as f64))
        })
        .collect()
}

/// A path's kind: its file name's extension, or the empty kind for a name
/// with none (a `Makefile`, a dotfile).
pub fn path_kind(path: &str) -> &str {
    let name = path.rsplit('/').next().unwrap_or(path);
    match name.rsplit_once('.') {
        Some((stem, ext)) if !stem.is_empty() => ext,
        _ => "",
    }
}

/// `z` on the band: `1000 × −ln P(Z ≥ z) / ln 100`, `Z` standard normal. Never
/// below the smallest positive value, so a scoring file however far below its
/// kind still reads above a file of no evidence.
pub fn band_of_z(z: f64) -> f32 {
    ((1000.0 * -ln_upper_tail(z) / 100f64.ln()) as f32).max(f32::MIN_POSITIVE)
}

/// `ln P(Z ≥ z)` for standard normal `Z`, accurate to a relative 1.2e-7 of the
/// tail and finite however far out `z` is: Chebyshev's `erfc` (Numerical
/// Recipes `erfcc`), taken in logs so a tail far below `f64`'s range does not
/// underflow to `ln 0`. Below the mean it is `ln(1 − P(Z ≥ |z|))`, through
/// `ln_1p` so a tail near 1 keeps its digits.
fn ln_upper_tail(z: f64) -> f64 {
    let x = z / std::f64::consts::SQRT_2;
    let a = x.abs();
    let t = 1.0 / (1.0 + 0.5 * a);
    let poly = -1.265_512_23
        + t * (1.000_023_68
            + t * (0.374_091_96
                + t * (0.096_784_18
                    + t * (-0.186_288_06
                        + t * (0.278_868_07
                            + t * (-1.135_203_98
                                + t * (1.488_515_87 + t * (-0.822_152_23 + t * 0.170_872_77))))))));
    // ln erfc(|x|) = ln t − x² + poly; P(Z ≥ |z|) = erfc(|x|) / 2.
    let ln_tail_abs = t.ln() - a * a + poly - std::f64::consts::LN_2;
    if x >= 0.0 {
        ln_tail_abs
    } else {
        (-ln_tail_abs.exp()).ln_1p()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A kind is the file name's extension**; a name with none, or a
    /// dotfile, is the empty kind, whatever the folders are called.
    #[test]
    fn a_kind_is_the_extension() {
        assert_eq!(path_kind("candle/docs/x.md"), "md");
        assert_eq!(path_kind("candle/zend/src/lib.rs"), "rs");
        assert_eq!(path_kind("candle/Makefile"), "");
        assert_eq!(path_kind("candle/.github/x"), "");
        assert_eq!(path_kind("candle/.gitignore"), "");
        assert_eq!(path_kind("a.b/c"), "");
    }

    /// **The band reads a z-score as the background normalizer reads a
    /// tail**: z 0 is a 1-in-2 tail, `1000 × ln 2 / ln 100` = 150.5; z 2.326 is
    /// 1 in 100, 1000.
    #[test]
    fn the_band_reads_z_as_a_tail() {
        assert_eq!(band_of_z(0.0).round(), 151.0);
        assert!((band_of_z(2.326_348).round() - 1000.0).abs() <= 1.0);
        assert!(band_of_z(-3.0) > 0.0 && band_of_z(-3.0) < 1.0);
        // Far out the tail stays finite and keeps rising.
        assert!(band_of_z(40.0).is_finite() && band_of_z(40.0) > band_of_z(39.0));
    }

    /// **The band rises with z**, from twelve spreads below the mean — past
    /// any file's z among a kind's few thousand — across the sign change and
    /// far into the tail; below that a scoring file still reads above 0.
    #[test]
    fn the_band_is_monotone() {
        let mut last = f32::NEG_INFINITY;
        for i in -240..=800 {
            let b = band_of_z(i as f64 / 20.0);
            assert!(b > last, "band({}) = {b} after {last}", i as f64 / 20.0);
            last = b;
        }
        assert!(band_of_z(-80.0) > 0.0);
    }

    /// **A kind of at least the sample is read against itself**: eight
    /// documents at 10 and 30 alternately and eight code files at 2 and 6 meet
    /// at the same band — z −1 and +1 in each — and a score of no evidence
    /// reads 0.
    #[test]
    fn a_large_kind_is_read_against_its_own() {
        let mut raw = Vec::new();
        let mut kinds = Vec::new();
        for i in 0..MIN_KIND_SAMPLE {
            raw.push(if i % 2 == 0 { 10.0 } else { 30.0 });
            kinds.push("md");
            raw.push(if i % 2 == 0 { 2.0 } else { 6.0 });
            kinds.push("rs");
        }
        raw.push(0.0);
        kinds.push("rs");
        let got = kind_z_band(&raw, &kinds);
        assert_eq!(got[0], got[1], "z −1 in each kind");
        assert_eq!(got[2], got[3], "z +1 in each kind");
        assert_eq!(got[0], band_of_z(-1.0));
        assert_eq!(got[2], band_of_z(1.0));
        assert_eq!(*got.last().unwrap(), 0.0);
    }

    /// **A kind under the sample is read against every kind's**: a lone
    /// `Makefile` at 30 among scores 10, 10, 30 and 30 — mean 22, spread
    /// 9.80 — reads at z 0.816, not at its own mean; a kind of two at 10 and
    /// 30 reads −1.22 and +0.816 there, not ±1.
    #[test]
    fn a_small_kind_is_read_against_every_kind() {
        let raw = [10.0, 30.0, 10.0, 30.0, 30.0];
        let kinds = ["md", "md", "toml", "toml", ""];
        let got = kind_z_band(&raw, &kinds);
        let (mean, sd) = (22.0_f64, (96.0_f64).sqrt());
        assert_eq!(got[4], band_of_z((30.0 - mean) / sd), "the Makefile");
        assert_eq!(got[0], band_of_z((10.0 - mean) / sd));
        assert_eq!(got[1], got[4], "same score, same pool");
        assert!(got[4] > band_of_z(0.0), "above where its own mean held it");
    }
}
