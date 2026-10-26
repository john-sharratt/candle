//! Boilerplate by file frequency: token runs that many files share verbatim.
//!
//! The request template, the tool-call JSON, a pager's footer and the chat
//! format's markers recur word for word in every file's turns, and the chunks
//! holding them win votes on every file's questions. A token is boilerplate
//! when a run of [`RUN`] tokens covering it occurs in at least a given number
//! of files — document frequency, counted over the gallery's own tokens, with
//! no scan and no background questions.

use std::collections::HashSet;

use candle_conversation::persistence::content_hash::turn_stream_id;

use crate::corpus::File;

/// Tokens per run.
pub const RUN: usize = 8;

/// A run's identity: its tokens folded through a 64-bit mix.
fn run_hash(run: &[u32]) -> u64 {
    run.iter().fold(0x9e37_79b9_7f4a_7c15u64, |h, &t| {
        let x = (h ^ t as u64).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x ^ (x >> 31)
    })
}

/// `(stream, start, end)` for every span of `turns` covered by a run of `run`
/// tokens that occurs in at least `min_files` distinct files. Each turn is
/// `(file, stream, tokens)`; a run repeated within one file counts once.
pub fn frequent_ranges(
    turns: &[(usize, u64, &[u32])],
    run: usize,
    min_files: usize,
) -> Vec<(u64, usize, usize)> {
    let hashes = |tokens: &[u32]| -> Vec<u64> {
        if tokens.len() < run {
            return Vec::new();
        }
        tokens.windows(run).map(run_hash).collect()
    };
    let mut seen: Vec<(u64, usize)> = turns
        .iter()
        .flat_map(|&(file, _, tokens)| hashes(tokens).into_iter().map(move |h| (h, file)))
        .collect();
    seen.sort_unstable();
    seen.dedup();
    let mut frequent = HashSet::new();
    for group in seen.chunk_by(|a, b| a.0 == b.0) {
        if group.len() >= min_files {
            frequent.insert(group[0].0);
        }
    }
    let mut out = Vec::new();
    for &(_, stream, tokens) in turns {
        let mut masked = vec![false; tokens.len()];
        for (i, h) in hashes(tokens).into_iter().enumerate() {
            if frequent.contains(&h) {
                masked[i..i + run].fill(true);
            }
        }
        let mut i = 0;
        while i < masked.len() {
            if masked[i] {
                let start = i;
                while i < masked.len() && masked[i] {
                    i += 1;
                }
                out.push((stream, start, i));
            } else {
                i += 1;
            }
        }
    }
    out
}

/// Every covered turn's boilerplate: runs found in at least `min_files` files.
pub fn ranges(files: &[File], min_files: usize) -> Vec<(u64, usize, usize)> {
    let turns: Vec<(usize, u64, &[u32])> = files
        .iter()
        .enumerate()
        .flat_map(|(fi, f)| {
            f.turns
                .iter()
                .enumerate()
                .filter(|(_, t)| t.covered())
                .map(move |(ti, t)| {
                    (
                        fi,
                        turn_stream_id(f.timeline, ti as u32).0,
                        t.tokens.as_slice(),
                    )
                })
        })
        .collect();
    frequent_ranges(&turns, RUN, min_files)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **A run shared by enough files is boilerplate wherever it sits.**
    /// `[7, 8, 9]` is in files 0, 1 and 2 at different offsets; `[4, 5, 6]`
    /// only in file 0, twice — once per file, so it is not frequent.
    #[test]
    fn a_run_in_enough_files_is_masked_wherever_it_sits() {
        let a = [7, 8, 9, 1, 4, 5, 6, 4, 5, 6];
        let b = [2, 7, 8, 9];
        let c = [3, 3, 7, 8, 9, 3];
        let turns = [(0, 10, &a[..]), (1, 11, &b[..]), (2, 12, &c[..])];
        assert_eq!(
            frequent_ranges(&turns, 3, 2),
            vec![(10, 0, 3), (11, 1, 4), (12, 2, 5)]
        );
        assert_eq!(frequent_ranges(&turns, 3, 4), vec![]);
    }

    /// Overlapping frequent runs merge into one span; a turn shorter than a
    /// run has none.
    #[test]
    fn overlapping_runs_merge_and_short_turns_have_none() {
        let a = [1, 2, 3, 4, 9];
        let b = [1, 2, 3, 4, 8];
        let short = [1, 2];
        let turns = [(0, 1, &a[..]), (1, 2, &b[..]), (2, 3, &short[..])];
        assert_eq!(frequent_ranges(&turns, 3, 2), vec![(1, 0, 4), (2, 0, 4)]);
    }
}
