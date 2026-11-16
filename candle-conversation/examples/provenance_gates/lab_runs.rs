//! Token runs: consecutive probe tokens that agree with nearby tokens of one
//! file.
//!
//! Each probe token's ranking holds, per fold group, its best files and the
//! gallery token each matched through. Two neighbouring probe tokens *link*
//! on a file when both rank it in the same group and their matched gallery
//! tokens lie close — the second within `slack` tokens after the first
//! ([`RunOrder::Forward`]), or within `slack` either way
//! ([`RunOrder::Either`]). A run is a chain of links; a vote's run length is
//! the longest chain through it. A single spurious match is a run of one; a
//! phrase the file actually holds is a run of several.

use candle_conversation::provenance::window_score::Ranked;
use candle_conversation::provenance::FOLD_GROUPS;
use serde::Deserialize;

/// Which way a run's matched gallery tokens may step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunOrder {
    /// Forward, as the probe's own tokens do: `0 < next − this ≤ slack`.
    Forward,
    /// Either way: `0 < |next − this| ≤ slack`.
    Either,
}

/// Whether gallery token `next` follows `this` closely enough to link.
fn links(this: u32, next: u32, slack: u32, order: RunOrder) -> bool {
    let (a, b) = (this as i64, next as i64);
    let d = b - a;
    match order {
        RunOrder::Forward => d > 0 && d <= slack as i64,
        RunOrder::Either => d != 0 && d.abs() <= slack as i64,
    }
}

/// For every probe token and every one of its first `k` ranked files in
/// group `g`: the length of the longest run through that vote.
/// `out[token][rank]`.
pub fn run_lengths(
    ranked: &[[Ranked; FOLD_GROUPS]],
    g: usize,
    k: usize,
    slack: u32,
    order: RunOrder,
) -> Vec<Vec<usize>> {
    let tops: Vec<&[(usize, u32, u32)]> = ranked
        .iter()
        .map(|r| &r[g].top[..r[g].top.len().min(k)])
        .collect();
    // The longest chain ending at each vote, then the longest starting there.
    let chain = |forward: bool| -> Vec<Vec<usize>> {
        let n = tops.len();
        let mut len: Vec<Vec<usize>> = tops.iter().map(|t| vec![1; t.len()]).collect();
        let order_of: Vec<usize> = if forward {
            (0..n).collect()
        } else {
            (0..n).rev().collect()
        };
        for w in order_of.windows(2) {
            let (prev, cur) = (w[0], w[1]);
            for (ci, &(f, _, at)) in tops[cur].iter().enumerate() {
                let best = tops[prev]
                    .iter()
                    .enumerate()
                    .filter(|&(_, &(pf, _, pat))| {
                        pf == f
                            && if forward {
                                links(pat, at, slack, order)
                            } else {
                                links(at, pat, slack, order)
                            }
                    })
                    .map(|(pi, _)| len[prev][pi])
                    .max();
                if let Some(b) = best {
                    len[cur][ci] = b + 1;
                }
            }
        }
        len
    };
    let ending = chain(true);
    let starting = chain(false);
    ending
        .iter()
        .zip(&starting)
        .map(|(e, s)| e.iter().zip(s).map(|(a, b)| a + b - 1).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tokens ranked in group 0 only, each `(file, gallery token)` at one
    /// agreement.
    fn ranked(tokens: &[&[(usize, u32)]]) -> Vec<[Ranked; FOLD_GROUPS]> {
        tokens
            .iter()
            .map(|top| {
                std::array::from_fn(|g| Ranked {
                    mean: 0.0,
                    std: 1.0,
                    top: if g == 0 {
                        top.iter().map(|&(f, t)| (f, 50, t)).collect()
                    } else {
                        Vec::new()
                    },
                })
            })
            .collect()
    }

    /// **Three probe tokens matching file 1 at gallery tokens 10, 11, 12 are
    /// one run of three**; file 2 matched at 40 then 90 is two runs of one,
    /// the second token's gap too wide.
    #[test]
    fn consecutive_matches_in_one_file_are_one_run() {
        let r = ranked(&[&[(1, 10), (2, 40)], &[(1, 11), (2, 90)], &[(1, 12)]]);
        assert_eq!(
            run_lengths(&r, 0, 8, 1, RunOrder::Forward),
            vec![vec![3, 1], vec![3, 1], vec![3]]
        );
    }

    /// **Slack lets a run skip gallery tokens; order decides direction.**
    /// 10 → 13 links at slack 3, not at slack 2; 13 → 10 links only either
    /// way.
    #[test]
    fn slack_and_order_bound_a_link() {
        let fwd = ranked(&[&[(1, 10)], &[(1, 13)]]);
        assert_eq!(
            run_lengths(&fwd, 0, 8, 3, RunOrder::Forward),
            vec![vec![2], vec![2]]
        );
        assert_eq!(
            run_lengths(&fwd, 0, 8, 2, RunOrder::Forward),
            vec![vec![1], vec![1]]
        );
        let back = ranked(&[&[(1, 13)], &[(1, 10)]]);
        assert_eq!(
            run_lengths(&back, 0, 8, 3, RunOrder::Forward),
            vec![vec![1], vec![1]]
        );
        assert_eq!(
            run_lengths(&back, 0, 8, 3, RunOrder::Either),
            vec![vec![2], vec![2]]
        );
    }

    /// **A run needs the same file**, and only the first `k` ranked files
    /// take part.
    #[test]
    fn a_run_is_one_file_within_the_first_k() {
        let r = ranked(&[&[(1, 10), (2, 20)], &[(3, 11), (2, 21)]]);
        assert_eq!(
            run_lengths(&r, 0, 8, 1, RunOrder::Forward),
            vec![vec![1, 2], vec![1, 2]]
        );
        assert_eq!(
            run_lengths(&r, 0, 1, 1, RunOrder::Forward),
            vec![vec![1], vec![1]]
        );
    }

    /// **A run through the middle counts both sides**: the middle token of a
    /// chain of five reads five.
    #[test]
    fn a_vote_reads_the_whole_run_through_it() {
        let r = ranked(&[&[(1, 1)], &[(1, 2)], &[(1, 3)], &[(1, 4)], &[(1, 5)]]);
        assert_eq!(
            run_lengths(&r, 0, 8, 1, RunOrder::Forward),
            vec![vec![5]; 5]
        );
    }
}
