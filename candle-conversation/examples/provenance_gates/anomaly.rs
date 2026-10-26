//! Anomalous-signal exclusion: chunks that win everything get a factor of 0.
//!
//! A chunk's background score is what it wins on questions about other files.
//! Most chunks win little or nothing; boilerplate — the request template, the
//! tool-call JSON, a pager's footer — wins on every file's questions and sits
//! orders of magnitude above the rest. A chunk is anomalous when its score is
//! an outlier of the background distribution: in log space, more than `k`
//! robust standard deviations (`1.4826 × MAD`) above the median of the chunks
//! that won anything.
//!
//! Exclusion never silences a file. Each file keeps at least `keep` of its
//! chunks; when more of them are anomalous than that allows, only its most
//! anomalous ones are excluded.

/// The robust spread of `ln(score)` over chunks that won anything:
/// `(median, 1.4826 × median absolute deviation)`.
pub fn log_spread(score: &[f32]) -> (f32, f32) {
    let mut logs: Vec<f32> = score.iter().filter(|&&s| s > 0.0).map(|s| s.ln()).collect();
    if logs.is_empty() {
        return (0.0, 0.0);
    }
    let median = |v: &mut Vec<f32>| {
        v.sort_by(f32::total_cmp);
        let n = v.len();
        if n % 2 == 1 {
            v[n / 2]
        } else {
            (v[n / 2 - 1] + v[n / 2]) / 2.0
        }
    };
    let med = median(&mut logs);
    let mut dev: Vec<f32> = logs.iter().map(|l| (l - med).abs()).collect();
    (med, 1.4826 * median(&mut dev))
}

/// The chunks excluded: outliers more than `k` robust deviations above the
/// median log score, at most `1 - keep` of each file's chunks, the most
/// anomalous first. `file_of[c]` is chunk `c`'s file.
pub fn excluded(score: &[f32], file_of: &[usize], k: f32, keep: f32) -> Vec<usize> {
    let (med, spread) = log_spread(score);
    let floor = med + k * spread;
    let n_files = file_of.iter().max().map_or(0, |&f| f + 1);
    let mut chunks_of = vec![0usize; n_files];
    let mut outliers: Vec<Vec<usize>> = vec![Vec::new(); n_files];
    for (c, &f) in file_of.iter().enumerate() {
        chunks_of[f] += 1;
        if score[c] > 0.0 && score[c].ln() > floor {
            outliers[f].push(c);
        }
    }
    let mut out = Vec::new();
    for (f, mut cands) in outliers.into_iter().enumerate() {
        let room = ((chunks_of[f] as f32) * (1.0 - keep)).floor() as usize;
        cands.sort_by(|&a, &b| score[b].total_cmp(&score[a]).then(a.cmp(&b)));
        out.extend(cands.into_iter().take(room));
    }
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Median and MAD of `ln` over the chunks that won anything: scores
    /// `e^0, e^1, e^2` and a zero give median 1, MAD 1.
    #[test]
    fn the_spread_is_taken_in_log_space_over_chunks_that_won() {
        let e = std::f32::consts::E;
        let (med, spread) = log_spread(&[1.0, e, e * e, 0.0]);
        assert!((med - 1.0).abs() < 1e-6);
        assert!((spread - 1.4826).abs() < 1e-5);
        assert_eq!(log_spread(&[0.0, 0.0]), (0.0, 0.0));
    }

    /// **Outliers are excluded; a file keeps its share.** Chunks 0–7 score
    /// around 10; chunks 8 and 9 (file 1) and 10 (file 2) score 10⁶. File 1
    /// has 4 chunks and keeps half — both its outliers go. File 2 has 2
    /// chunks and keeps half — its one outlier goes. With `keep = 0.9` file
    /// 1 may lose none of its 4 (`floor(0.4) = 0`).
    #[test]
    fn outliers_are_excluded_but_every_file_keeps_its_share() {
        let score = [
            10.0, 11.0, 9.0, 12.0, 10.0, 8.0, 10.0, 11.0, 1e6, 2e6, 1e6, 0.0,
        ];
        let file_of = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2];
        assert_eq!(excluded(&score, &file_of, 4.0, 0.5), vec![8, 9, 10]);
        assert_eq!(excluded(&score, &file_of, 4.0, 0.9), Vec::<usize>::new());
    }

    /// When a file's room is smaller than its outliers, the most anomalous go
    /// first: file 0 has 4 chunks, three of them outliers, and room for 2.
    #[test]
    fn the_most_anomalous_go_first_when_room_is_short() {
        let score = [5e6, 1e6, 3e6, 10.0, 10.0, 11.0, 9.0, 10.0, 12.0, 10.0];
        let file_of = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        assert_eq!(excluded(&score, &file_of, 4.0, 0.5), vec![0, 2]);
    }
}
