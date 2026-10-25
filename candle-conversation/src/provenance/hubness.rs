//! Hubness correction for file scores.
//!
//! Some files win a share of almost every probe's votes, whatever the probe
//! asks — a hub: a file of generic phrasing, or one whose signatures sit near
//! the centre of the gallery. Ranked by raw votes, the same hubs top every
//! question. A file's **background** is the mean share of the votes it wins
//! over probes it is not the answer to; a probe's evidence for a file is how
//! far its share for that file stands above the file's background — the
//! cross-domain similarity local scaling of nearest-neighbour retrieval,
//! applied to vote shares.

/// How a file's share for a probe is set against its background.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum HubCorrection {
    /// The raw share: no correction.
    None,
    /// The share less the background.
    Less,
    /// The share over the background, the background floored at `floor` so a
    /// file that never wins a background vote is not divided by nothing.
    Over { floor: f32 },
}

impl HubCorrection {
    pub fn name(self) -> String {
        match self {
            HubCorrection::None => "raw".into(),
            HubCorrection::Less => "less-background".into(),
            HubCorrection::Over { floor } => format!("over-background({floor})"),
        }
    }
}

/// `scores` as shares of their total — zero shares when nothing scored.
pub fn shares(scores: &[f32]) -> Vec<f32> {
    let total: f32 = scores.iter().map(|s| s.max(0.0)).sum();
    if total <= 0.0 {
        return vec![0.0; scores.len()];
    }
    scores.iter().map(|s| s.max(0.0) / total).collect()
}

/// Each file's mean share over `probes`, every probe's scores one vector of
/// per-file scores.
pub fn background(probes: &[Vec<f32>], n_files: usize) -> Vec<f32> {
    let mut mean = vec![0.0f32; n_files];
    if probes.is_empty() {
        return mean;
    }
    for p in probes {
        for (m, s) in mean.iter_mut().zip(shares(p)) {
            *m += s;
        }
    }
    let n = probes.len() as f32;
    mean.iter_mut().for_each(|m| *m /= n);
    mean
}

/// `scores` corrected for each file's `background` under `how`.
pub fn corrected(scores: &[f32], background: &[f32], how: HubCorrection) -> Vec<f32> {
    let p = shares(scores);
    match how {
        HubCorrection::None => p,
        HubCorrection::Less => p.iter().zip(background).map(|(s, b)| s - b).collect(),
        HubCorrection::Over { floor } => p
            .iter()
            .zip(background)
            .map(|(s, b)| s / b.max(floor))
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shares_sum_to_one_and_ignore_negatives() {
        assert_eq!(shares(&[1.0, 3.0, -2.0]), vec![0.25, 0.75, 0.0]);
        assert_eq!(shares(&[0.0, 0.0]), vec![0.0, 0.0]);
    }

    #[test]
    fn the_background_is_the_mean_share() {
        let b = background(&[vec![1.0, 1.0], vec![3.0, 1.0]], 2);
        assert_eq!(b, vec![0.625, 0.375]);
        assert_eq!(background(&[], 3), vec![0.0; 3]);
    }

    /// **A hub loses its lead.** File 0 takes most of every probe's votes;
    /// file 1 rarely scores. A probe giving file 0 its usual share and file 1
    /// far more than its usual share ranks file 1 first once corrected.
    #[test]
    fn a_hub_at_its_usual_share_ranks_below_a_file_above_its_own() {
        let bg = background(&[vec![8.0, 1.0, 1.0], vec![9.0, 0.5, 0.5]], 3);
        let probe = [7.0, 3.0, 0.0];
        let raw = corrected(&probe, &bg, HubCorrection::None);
        assert!(raw[0] > raw[1]);
        let less = corrected(&probe, &bg, HubCorrection::Less);
        assert!(less[1] > less[0], "{less:?}");
        let over = corrected(&probe, &bg, HubCorrection::Over { floor: 0.01 });
        assert!(over[1] > over[0], "{over:?}");
    }
}
