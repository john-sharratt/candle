//! A file's background: what it scores on questions that are not about it,
//! and how surprising a score is against it.
//!
//! Every calibration question probes the gallery; a file's scores over the
//! questions that are not its own are its background. A score is then read as
//! surprise, `-ln P(background ≥ score)`: under the null — a question not about
//! the file — every file's surprise follows the same distribution, so none
//! wins by chance more often than another, while a genuine hit sits in its
//! file's far tail.

use std::f64::consts::SQRT_2;

use candle_conversation::provenance::WideQSig;

use crate::corpus::File;
use crate::remote_gallery::{Questions, RemoteGallery};
use crate::sweep::summed;

/// Probes per launch.
const BATCH: usize = 256;

/// The probe length scores are restated at, as production's normalization
/// does (`NormConfig::probe_t_ref`).
pub const T_REF: f32 = 256.0;

/// `raw`, scored over a probe of `tokens` tokens, restated at [`T_REF`].
pub fn to_ref(raw: f32, tokens: usize) -> f32 {
    raw * T_REF / tokens.max(1) as f32
}

/// One file's background, from its reference-scaled scores.
pub struct Background {
    mean: f32,
    sd: f32,
    /// The share of probes it scored above zero on.
    p_nonzero: f32,
    /// Mean and spread of `ln score` over the probes it scored on.
    ln_mean: f32,
    ln_sd: f32,
    /// Every score, ascending.
    sorted: Vec<f32>,
}

impl Background {
    pub fn of(scores: &[f32]) -> Background {
        let n = scores.len().max(1) as f32;
        let mean = scores.iter().sum::<f32>() / n;
        let sd = (scores.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n).sqrt();
        let lns: Vec<f32> = scores
            .iter()
            .filter(|&&x| x > 0.0)
            .map(|x| x.ln())
            .collect();
        let m = lns.len().max(1) as f32;
        let ln_mean = lns.iter().sum::<f32>() / m;
        let ln_sd = (lns.iter().map(|x| (x - ln_mean).powi(2)).sum::<f32>() / m).sqrt();
        let mut sorted = scores.to_vec();
        sorted.sort_by(f32::total_cmp);
        Background {
            mean,
            sd,
            p_nonzero: lns.len() as f32 / n,
            ln_mean,
            ln_sd,
            sorted,
        }
    }

    pub fn z(&self, x: f32) -> f32 {
        (x - self.mean) / self.sd.max(1e-6)
    }

    /// `-ln P(score ≥ x)` under a zero-inflated log-normal; a zero is no
    /// evidence.
    pub fn log_normal_surprise(&self, x: f32) -> f32 {
        if x <= 0.0 {
            return 0.0;
        }
        let z = (x.ln() - self.ln_mean) / self.ln_sd.max(1e-3);
        let tail = self.p_nonzero.max(1.0 / self.sorted.len().max(1) as f32) * upper_tail(z);
        -tail.max(1e-30).ln()
    }

    /// `-ln P(score ≥ x)` over the scores seen, add-one smoothed; a zero is
    /// no evidence.
    pub fn empirical_surprise(&self, x: f32) -> f32 {
        if x <= 0.0 {
            return 0.0;
        }
        let below = self.sorted.partition_point(|&v| v < x);
        let at_least = self.sorted.len() - below;
        -((at_least as f32 + 1.0) / (self.sorted.len() as f32 + 1.0)).ln()
    }

    /// [`Self::empirical_surprise`] plus `blend × ln(1 + x)`: the tail,
    /// keeping some of the score's size.
    pub fn blended(&self, x: f32, blend: f32) -> f32 {
        self.empirical_surprise(x) + blend * x.max(0.0).ln_1p()
    }
}

/// `P(Z ≥ z)` for a standard normal — Abramowitz & Stegun 7.1.26 on `erfc`.
pub fn upper_tail(z: f32) -> f32 {
    let x = (z as f64) / SQRT_2;
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let erfc_abs = poly * (-x * x).exp();
    let erfc = if x >= 0.0 { erfc_abs } else { 2.0 - erfc_abs };
    (erfc / 2.0) as f32
}

/// Every calibration question, as `(owner, signatures)`.
pub fn question_probes(files: &[File]) -> Vec<(usize, &[WideQSig])> {
    files
        .iter()
        .enumerate()
        .flat_map(|(fi, f)| f.questions.iter().map(move |q| (fi, q.sigs.as_slice())))
        .collect()
}

/// Every probe's reference-scaled score per file, today's scorer, against
/// the gallery `questions` names.
pub fn scaled_scores(
    gpu: &RemoteGallery,
    probes: &[(usize, &[WideQSig])],
    questions: Questions,
    n_files: usize,
) -> anyhow::Result<Vec<Vec<f32>>> {
    let mut out: Vec<Vec<f32>> = Vec::with_capacity(probes.len());
    for chunk in probes.chunks(BATCH) {
        let sigs: Vec<&[WideQSig]> = chunk.iter().map(|(_, s)| *s).collect();
        let votes = gpu.votes(&sigs, &|_, _| false, questions, &|_, _| false)?;
        for ((_, s), v) in chunk.iter().zip(&votes) {
            let raw = summed(v, &[1.0, 1.0, 1.0], 0.25, n_files);
            out.push(raw.into_iter().map(|x| to_ref(x, s.len())).collect());
        }
    }
    Ok(out)
}

/// Each file's background over `scaled` — the probes that are not its own.
pub fn learn(
    probes: &[(usize, &[WideQSig])],
    scaled: &[Vec<f32>],
    n_files: usize,
) -> Vec<Background> {
    (0..n_files)
        .map(|f| {
            let scores: Vec<f32> = probes
                .iter()
                .zip(scaled)
                .filter(|((owner, _), _)| *owner != f)
                .map(|(_, s)| s[f])
                .collect();
            Background::of(&scores)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_normal_upper_tail_matches_its_table() {
        assert!((upper_tail(0.0) - 0.5).abs() < 1e-6);
        assert!((upper_tail(1.0) - 0.158_655).abs() < 1e-6);
        assert!((upper_tail(-1.0) - 0.841_345).abs() < 1e-6);
        assert!((upper_tail(3.0) - 0.001_350).abs() < 1e-6);
    }

    #[test]
    fn a_zero_is_no_evidence_and_a_rare_score_is_surprising() {
        let b = Background::of(&[0.0, 0.0, 0.0, 10.0, 20.0, 40.0]);
        assert_eq!(b.empirical_surprise(0.0), 0.0);
        // Two of six scores are at least 20: (2 + 1) / 7.
        assert!((b.empirical_surprise(20.0) - (7.0f32 / 3.0).ln()).abs() < 1e-6);
        assert!(b.empirical_surprise(40.0) > b.empirical_surprise(10.0));
        assert!(b.log_normal_surprise(40.0) > b.log_normal_surprise(10.0));
        assert!(
            (b.blended(20.0, 0.5) - (b.empirical_surprise(20.0) + 0.5 * 21.0f32.ln())).abs() < 1e-6
        );
    }

    #[test]
    fn a_score_is_restated_at_the_reference_length() {
        assert_eq!(to_ref(10.0, 64), 40.0);
        assert_eq!(to_ref(10.0, 0), 2560.0);
    }
}
