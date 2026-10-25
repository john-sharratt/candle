//! Windowed provenance scoring (`docs/provenance_scoped_signatures.md` §2.3).
//!
//! Late fusion gives every probe token, per fold group, to one file — the file
//! holding the best-agreeing gallery token — weighted by `z × margin`, the
//! margin over the best token of any other file; a token every file matches
//! alike votes for nothing. Summed per file, a long file collects votes over
//! all its tokens, so size alone ranks it up. Here each vote is kept with the
//! gallery token it was won at and credited to the windows holding that token
//! — windows cut to the probe's own shape (`probe_windows`) — so a window
//! scores by how many of the probe's tokens it wins **together**, and a file by
//! one of [`FileAggregate`]'s rules over its windows. Measured against today's
//! per-file sum in `docs/results/provenance_gates_2026-10-25.md`.

use std::ops::Range;

use rayon::prelude::*;

use super::popcount::group_agreement;
use super::scan::NEEDLE_KEEP_FRAC;
use super::{heads_per_group, WideQSig, FOLD_GROUPS};

/// One probe token's late-fusion vote: the gallery token that won it, the file
/// that holds it, and its weight.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vote {
    pub token: usize,
    pub file: usize,
    pub weight: f32,
}

/// Late fusion's votes, kept with the token each was won at: per probe token
/// and fold group, the file holding the best-agreeing gallery token wins a
/// `z × margin` vote, the margin over the best token of any **other** file;
/// then the needle gate keeps the top quarter of probe tokens by vote
/// magnitude. Summed per file these are today's scores. `token_file[t]` is
/// gallery token `t`'s file; a token whose file is `n_files` or past it takes
/// no part. `group_weights` weighs the fold groups (empty: uniform).
pub fn late_fusion_votes(
    probe: &[WideQSig],
    tokens: &[WideQSig],
    token_file: &[usize],
    n_files: usize,
    group_weights: &[f32],
) -> Vec<Vote> {
    let Some(shape) = probe.first() else {
        return Vec::new();
    };
    let wph = shape.words_per_head();
    let hpg = heads_per_group(shape.n_heads as usize);
    if wph == 0 || hpg == 0 || tokens.is_empty() || tokens.len() != token_file.len() {
        return Vec::new();
    }
    let gw = hpg * wph;
    let need = FOLD_GROUPS * gw;
    let n = tokens.len() as f32;
    let per_query: Vec<Vec<Vote>> = probe
        .par_iter()
        .filter(|q| q.words.len() >= need)
        .map(|q| {
            let mut best = vec![(0u32, usize::MAX); n_files];
            let mut out = Vec::with_capacity(FOLD_GROUPS);
            for g in 0..FOLD_GROUPS {
                let base = g * gw;
                let qg = &q.words[base..base + gw];
                best.iter_mut().for_each(|b| *b = (0, usize::MAX));
                let (mut sum, mut sumsq) = (0u64, 0u64);
                for (t, cand) in tokens.iter().enumerate() {
                    let Some(cg) = cand.words.get(base..base + gw) else {
                        continue;
                    };
                    let ag = group_agreement(qg, cg);
                    let f = token_file[t];
                    if f < n_files && (best[f].1 == usize::MAX || ag > best[f].0) {
                        best[f] = (ag, t);
                    }
                    sum += ag as u64;
                    sumsq += (ag as u64).pow(2);
                }
                let (mut top1, mut top1f, mut top2) = (0u32, usize::MAX, 0u32);
                for (f, &(m, t)) in best.iter().enumerate() {
                    if t == usize::MAX {
                        continue;
                    }
                    if top1f == usize::MAX || m > top1 {
                        top2 = top1;
                        top1 = m;
                        top1f = f;
                    } else if m > top2 {
                        top2 = m;
                    }
                }
                if top1f != usize::MAX {
                    let mean = sum as f32 / n;
                    let var = (sumsq as f32 / n - mean * mean).max(1e-6);
                    let z = ((top1 as f32 - mean) / var.sqrt()).max(0.0);
                    let margin = top1.saturating_sub(top2) as f32;
                    let w = group_weights.get(g).copied().unwrap_or(1.0);
                    out.push(Vote {
                        token: best[top1f].1,
                        file: top1f,
                        weight: z * margin * w,
                    });
                }
            }
            out
        })
        .collect();
    // The needle gate, as late fusion keeps it.
    let mags: Vec<f32> = per_query
        .iter()
        .map(|v| v.iter().map(|x| x.weight).sum())
        .collect();
    if mags.is_empty() {
        return Vec::new();
    }
    let keep = ((NEEDLE_KEEP_FRAC * mags.len() as f32).ceil() as usize).clamp(1, mags.len());
    let mut sorted = mags.clone();
    sorted.sort_unstable_by(|a, b| b.total_cmp(a));
    let thresh = sorted[keep - 1];
    per_query
        .into_iter()
        .zip(mags)
        .filter(|(_, m)| *m >= thresh)
        .flat_map(|(v, _)| v)
        .collect()
}

/// Each window's tally of `votes`: a vote counts for every window holding the
/// token it was won at. `windows[w]` are window `w`'s ranges of a gallery of
/// `n_tokens` tokens.
pub fn window_tally(votes: &[Vote], windows: &[Vec<Range<usize>>], n_tokens: usize) -> Vec<f32> {
    let mut at = vec![0.0f32; n_tokens];
    for v in votes {
        if v.token < n_tokens {
            at[v.token] += v.weight;
        }
    }
    // Prefix sums turn each range's tally into two reads.
    let mut prefix = Vec::with_capacity(n_tokens + 1);
    let mut run = 0.0f64;
    prefix.push(run);
    for x in &at {
        run += *x as f64;
        prefix.push(run);
    }
    windows
        .iter()
        .map(|ranges| {
            ranges
                .iter()
                .map(|r| {
                    let e = r.end.min(n_tokens);
                    let s = r.start.min(e);
                    (prefix[e] - prefix[s]) as f32
                })
                .sum()
        })
        .collect()
}

/// How a file's windows make its one score.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FileAggregate {
    /// Its best window.
    BestWindow,
    /// Its best window, less what the best of that many windows reaches by
    /// chance: `σ · √(2 ln W)` for `W` windows, `σ` the spread of every
    /// window's score for this probe.
    CountNormalised,
    /// Its best identity window — its first turn's request and its last turn's
    /// closing summary.
    Identity,
}

impl FileAggregate {
    pub const ALL: [FileAggregate; 3] = [
        FileAggregate::BestWindow,
        FileAggregate::CountNormalised,
        FileAggregate::Identity,
    ];

    pub fn name(self) -> &'static str {
        match self {
            FileAggregate::BestWindow => "best-window",
            FileAggregate::CountNormalised => "count-normalised",
            FileAggregate::Identity => "identity",
        }
    }
}

/// Each file's score under `mode`: `scores[w]` is window `w`'s, `file[w]` its
/// file, `identity[w]` whether it is one of its file's identity windows, and
/// `live[w]` whether it takes part. A file with no window taking part scores
/// `f32::NEG_INFINITY`.
pub fn file_scores(
    scores: &[f32],
    file: &[usize],
    identity: &[bool],
    live: &[bool],
    n_files: usize,
    mode: FileAggregate,
) -> Vec<f32> {
    let mut best = vec![f32::NEG_INFINITY; n_files];
    let mut count = vec![0usize; n_files];
    let (mut sum, mut sumsq, mut n) = (0f64, 0f64, 0f64);
    for w in 0..scores.len() {
        if !live[w] {
            continue;
        }
        sum += scores[w] as f64;
        sumsq += (scores[w] as f64).powi(2);
        n += 1.0;
        if mode == FileAggregate::Identity && !identity[w] {
            continue;
        }
        let f = file[w];
        count[f] += 1;
        best[f] = best[f].max(scores[w]);
    }
    if mode != FileAggregate::CountNormalised || n == 0.0 {
        return best;
    }
    let sd = (sumsq / n - (sum / n).powi(2)).max(0.0).sqrt() as f32;
    best.iter()
        .zip(&count)
        .map(|(&b, &c)| match c {
            0 => f32::NEG_INFINITY,
            c => b - sd * (2.0 * (c as f32).ln()).max(0.0).sqrt(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provenance::score_slots_weighted;

    /// A deterministic pseudo-random folded signature: 3 groups × 4 heads ×
    /// 2 words.
    fn sig(seed: u64) -> WideQSig {
        let mut x = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
        let words = (0..24)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x
            })
            .collect();
        WideQSig { n_heads: 12, words }
    }

    /// **Summed per file, the votes are today's late fusion.** Same gallery,
    /// one case a file: the per-file sums equal `score_slots_weighted`'s.
    #[test]
    fn votes_summed_per_file_are_late_fusion() {
        let tokens: Vec<WideQSig> = (0..300).map(sig).collect();
        let token_file: Vec<usize> = (0..300).map(|t| t / 60).collect();
        let probe: Vec<WideQSig> = tokens[130..170]
            .iter()
            .cloned()
            .chain((900..920).map(sig))
            .collect();
        let votes = late_fusion_votes(&probe, &tokens, &token_file, 5, &[]);
        let mut per_file = vec![0.0f32; 5];
        for v in &votes {
            assert_eq!(token_file[v.token], v.file);
            per_file[v.file] += v.weight;
        }
        let windows: Vec<&[WideQSig]> = (0..5).map(|f| &tokens[f * 60..f * 60 + 60]).collect();
        let slots: Vec<usize> = (0..5).collect();
        let want = score_slots_weighted(&probe, &windows, &slots, 5, &[]);
        for (a, b) in per_file.iter().zip(&want) {
            assert!(
                (a - b).abs() <= 1e-3 * (1.0 + b.abs()),
                "{per_file:?} vs {want:?}"
            );
        }
    }

    /// A token whose file is past the last takes no part: the votes go to the
    /// files that remain.
    #[test]
    fn a_token_of_no_file_wins_nothing() {
        let tokens: Vec<WideQSig> = (0..120).map(sig).collect();
        let token_file: Vec<usize> = (0..120).map(|t| if t < 60 { 2 } else { 1 }).collect();
        let probe: Vec<WideQSig> = tokens[10..30].to_vec();
        let votes = late_fusion_votes(&probe, &tokens, &token_file, 2, &[]);
        assert!(!votes.is_empty());
        assert!(votes.iter().all(|v| v.file == 1 && v.token >= 60));
    }

    /// A vote counts for every window holding its token, and for no other.
    #[test]
    fn a_window_tallies_the_votes_won_inside_it() {
        let votes = [
            Vote {
                token: 3,
                file: 0,
                weight: 2.0,
            },
            Vote {
                token: 10,
                file: 0,
                weight: 5.0,
            },
            Vote {
                token: 3,
                file: 0,
                weight: 1.0,
            },
        ];
        let windows = vec![vec![0..5], vec![2..12], vec![0..2, 9..11], vec![12..20]];
        assert_eq!(window_tally(&votes, &windows, 20), vec![3.0, 8.0, 5.0, 0.0]);
    }

    /// Best window takes the maximum; count normalisation charges a file for
    /// how many windows it drew; identity reads only identity windows; a
    /// window not taking part counts for nothing.
    #[test]
    fn each_aggregate_reads_the_windows_its_rule_names() {
        // File 0: one window of 5. File 1: four windows, best 5.5.
        let scores = [5.0, 1.0, 2.0, 5.5, 3.0];
        let file = [0, 1, 1, 1, 1];
        let identity = [true, true, false, false, false];
        let live = [true; 5];
        let agg = |mode, live: &[bool]| file_scores(&scores, &file, &identity, live, 2, mode);
        assert_eq!(agg(FileAggregate::BestWindow, &live), vec![5.0, 5.5]);
        assert_eq!(agg(FileAggregate::Identity, &live), vec![5.0, 1.0]);
        let norm = agg(FileAggregate::CountNormalised, &live);
        assert_eq!(norm[0], 5.0, "one window draws no extra chance");
        assert!(norm[1] < 5.0, "four windows pay for their draws: {norm:?}");
        let masked = [true, true, true, false, true];
        assert_eq!(agg(FileAggregate::BestWindow, &masked), vec![5.0, 3.0]);
        let none = agg(FileAggregate::BestWindow, &[false; 5]);
        assert!(none.iter().all(|s| *s == f32::NEG_INFINITY));
    }
}
