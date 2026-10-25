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
    /// The raw score over the file's background **peak** — the most it scored
    /// on any background probe, both scaled to a common probe length — floored
    /// at `floor`: the denominator production's normalization reaches when its
    /// levels have seen background traffic (`normalization::ScopeState`'s
    /// traffic-peak path).
    OverPeak { floor: f32 },
}

impl HubCorrection {
    pub fn name(self) -> String {
        match self {
            HubCorrection::None => "raw".into(),
            HubCorrection::Less => "less-background".into(),
            HubCorrection::Over { floor } => format!("over-background({floor})"),
            HubCorrection::OverPeak { floor } => format!("over-peak({floor})"),
        }
    }
}

/// Each file's peak raw score over `probes`, each probe's scores scaled from
/// its length to `t_ref` tokens.
pub fn peak(probes: &[(Vec<f32>, usize)], n_files: usize, t_ref: usize) -> Vec<f32> {
    let mut peak = vec![0.0f32; n_files];
    for (scores, len) in probes {
        let scale = t_ref as f32 / (*len).max(1) as f32;
        for (p, s) in peak.iter_mut().zip(scores) {
            *p = p.max(s * scale);
        }
    }
    peak
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

/// `scores` corrected for each file's `background` under `how` — its mean
/// share ([`background`]), or for [`HubCorrection::OverPeak`] its peak
/// ([`peak`]), against which `scores` are scaled from `probe_len` to `t_ref`.
pub fn corrected(
    scores: &[f32],
    background: &[f32],
    how: HubCorrection,
    probe_len: usize,
    t_ref: usize,
) -> Vec<f32> {
    if let HubCorrection::OverPeak { floor } = how {
        let scale = t_ref as f32 / probe_len.max(1) as f32;
        return scores
            .iter()
            .zip(background)
            .map(|(s, b)| s * scale / b.max(floor))
            .collect();
    }
    let p = shares(scores);
    match how {
        HubCorrection::None | HubCorrection::OverPeak { .. } => p,
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
        let fix = |how| corrected(&probe, &bg, how, 10, 10);
        let raw = fix(HubCorrection::None);
        assert!(raw[0] > raw[1]);
        let less = fix(HubCorrection::Less);
        assert!(less[1] > less[0], "{less:?}");
        let over = fix(HubCorrection::Over { floor: 0.01 });
        assert!(over[1] > over[0], "{over:?}");
    }

    /// The peak is each file's most over the background probes, at a common
    /// length; over it, a hub scoring its usual peak reads 1 and a quiet file
    /// scoring far past its own reads far more.
    #[test]
    fn over_the_peak_a_quiet_files_rare_hit_outranks_a_hubs_usual_one() {
        // Probe lengths 10 and 20, scaled to 20: the first probe's scores double.
        let bg = peak(&[(vec![4.0, 0.5], 10), (vec![9.0, 0.2], 20)], 2, 20);
        assert_eq!(bg, vec![9.0, 1.0]);
        let fixed = corrected(
            &[9.0, 3.0],
            &bg,
            HubCorrection::OverPeak { floor: 0.1 },
            20,
            20,
        );
        assert_eq!(fixed, vec![1.0, 3.0]);
        // A file never seen in the background is divided by the floor.
        let unseen = corrected(
            &[1.0],
            &[0.0],
            HubCorrection::OverPeak { floor: 0.5 },
            20,
            20,
        );
        assert_eq!(unseen, vec![2.0]);
    }
}
