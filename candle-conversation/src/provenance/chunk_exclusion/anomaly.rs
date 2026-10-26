//! The anomalous-signal rule: which chunks win so much of everything that
//! they get a factor of 0.
//!
//! A chunk's background mass is what it wins on questions about other files.
//! Most chunks win little or nothing; boilerplate — the request template, the
//! tool-call JSON, a pager's footer — wins on every file's questions and sits
//! orders of magnitude above the rest. A chunk is anomalous when its mass is
//! an outlier of that distribution: in log space, more than `k` robust
//! standard deviations (`1.4826 × MAD`) above the median of the chunks that
//! won anything.
//!
//! The rule never silences a file. Each file keeps at least `keep` of its
//! chunks; when more of them are anomalous than that allows, only its most
//! anomalous ones go.

/// Robust deviations above the median log mass a chunk is cut at. Measured on
/// held-out calibration questions: k = 5 cuts ~130 of ~750,000 chunks and
/// lifts MRR 0.184 → 0.200; k = 4 cuts 544 and gains half that, k = 3 nothing
/// (`docs/results/provenance_gates_2026-10-26.md`).
pub const K: f32 = 5.0;

/// The threshold in a calibration competition: its chunks are few per file
/// and alike — questions and the thinking they start — so its outliers sit
/// nearer the median. Measured on held-out questions over the files first
/// calibrated with thinking: k = 3 cuts 17 of ~8,000 chunks and lifts MRR
/// 0.333 → 0.401 on question probes and 0.424 → 0.463 on thinking probes;
/// k = 4 and 5 cut none.
pub const K_CALIBRATION: f32 = 3.0;

/// The share of each file's chunks the cut leaves it.
pub const KEEP: f32 = 0.5;

/// The robust spread of `ln(mass)` over chunks that won anything:
/// `(median, 1.4826 × median absolute deviation)`; `(0, 0)` when none did.
pub fn log_spread(mass: &[f32]) -> (f32, f32) {
    let mut logs: Vec<f32> = mass.iter().filter(|&&m| m > 0.0).map(|m| m.ln()).collect();
    if logs.is_empty() {
        return (0.0, 0.0);
    }
    let med = median(&mut logs);
    let mut dev: Vec<f32> = logs.iter().map(|l| (l - med).abs()).collect();
    (med, 1.4826 * median(&mut dev))
}

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// The chunks cut, ascending: outliers more than `k` robust deviations above
/// the median log mass, at most `1 - keep` of each file's chunks, the most
/// anomalous first. `file_of[c]` is chunk `c`'s file.
pub fn outliers(mass: &[f32], file_of: &[usize], k: f32, keep: f32) -> Vec<usize> {
    let (med, spread) = log_spread(mass);
    let floor = med + k * spread;
    let n_files = file_of.iter().max().map_or(0, |&f| f + 1);
    let mut chunks_of = vec![0usize; n_files];
    let mut over: Vec<Vec<usize>> = vec![Vec::new(); n_files];
    for (c, &f) in file_of.iter().enumerate() {
        chunks_of[f] += 1;
        if mass[c] > 0.0 && mass[c].ln() > floor {
            over[f].push(c);
        }
    }
    let mut out = Vec::new();
    for (f, mut cands) in over.into_iter().enumerate() {
        let room = ((chunks_of[f] as f32) * (1.0 - keep)).floor() as usize;
        cands.sort_by(|&a, &b| mass[b].total_cmp(&mass[a]).then(a.cmp(&b)));
        out.extend(cands.into_iter().take(room));
    }
    out.sort_unstable();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Median and MAD of `ln` over the chunks that won anything: masses
    /// `e^0, e^1, e^2` and a zero give median 1, MAD 1.
    #[test]
    fn the_spread_is_taken_in_log_space_over_chunks_that_won() {
        let e = std::f32::consts::E;
        let (med, spread) = log_spread(&[1.0, e, e * e, 0.0]);
        assert!((med - 1.0).abs() < 1e-6);
        assert!((spread - 1.4826).abs() < 1e-5);
        assert_eq!(log_spread(&[0.0, 0.0]), (0.0, 0.0));
    }

    /// **Outliers are cut; a file keeps its share.** Chunks 0–7 win around
    /// 10; chunks 8 and 9 (file 1) and 10 (file 2) win 10⁶. File 1 has 4
    /// chunks and keeps half — both its outliers go. File 2 has 2 chunks and
    /// keeps half — its one outlier goes. With `keep = 0.9` neither may lose
    /// any (`floor(0.4) = floor(0.2) = 0`).
    #[test]
    fn outliers_are_cut_but_every_file_keeps_its_share() {
        let mass = [
            10.0, 11.0, 9.0, 12.0, 10.0, 8.0, 10.0, 11.0, 1e6, 2e6, 1e6, 0.0,
        ];
        let file_of = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2];
        assert_eq!(outliers(&mass, &file_of, 4.0, 0.5), vec![8, 9, 10]);
        assert_eq!(outliers(&mass, &file_of, 4.0, 0.9), Vec::<usize>::new());
    }

    /// When a file's room is smaller than its outliers, the most anomalous go
    /// first: file 0 has 4 chunks, three of them outliers, and room for 2.
    #[test]
    fn the_most_anomalous_go_first_when_room_is_short() {
        let mass = [5e6, 1e6, 3e6, 10.0, 10.0, 11.0, 9.0, 10.0, 12.0, 10.0];
        let file_of = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1];
        assert_eq!(outliers(&mass, &file_of, 4.0, 0.5), vec![0, 2]);
    }
}
