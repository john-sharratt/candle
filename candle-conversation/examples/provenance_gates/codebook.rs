//! Near-duplicate signatures by a learned binary codebook.
//!
//! A gallery token's signature in a fold group is a sign vector; two tokens
//! whose vectors differ in a few bits vote alike, and the scan's own measure of
//! "alike" is Hamming agreement. A codebook of `c` codes — binary k-means under
//! Hamming distance, each code the per-bit majority of its members — names
//! every token by its nearest code, and the code is the token's identity for
//! the two reductions measured here:
//!
//! **within a file**, tokens sharing a code are near duplicates, and only the
//! first few of each to appear are kept.
//!
//! Dropping a code found in many files, as carrying no signal, was measured
//! and refused: at every cutoff it cost most of the retrieval (a token sits
//! ~22% of its bits from its code, so codes are regions every file reaches).
//!
//! Signatures are already bits: nothing is hashed.

use std::cmp::Reverse;
use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

/// One token's signature over the fold groups a scope weighs: their words,
/// concatenated.
pub type Slice = Vec<u64>;

/// Hamming distance between two slices of one width.
pub fn distance(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// The index of the code nearest `point`, the first on a tie.
pub fn nearest(point: &[u64], codes: &[Slice]) -> u32 {
    let mut best = (u32::MAX, 0u32);
    for (i, c) in codes.iter().enumerate() {
        let d = distance(point, c);
        if d < best.0 {
            best = (d, i as u32);
        }
    }
    best.1
}

/// Every point's nearest code.
pub fn assign(points: &[Slice], codes: &[Slice]) -> Vec<u32> {
    points.par_iter().map(|p| nearest(p, codes)).collect()
}

/// A deterministic stream of indices below `n`, from `seed`.
fn picks(seed: u64, n: usize) -> impl Iterator<Item = usize> {
    let mut x = seed | 1;
    std::iter::repeat_with(move || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        (x % n as u64) as usize
    })
}

/// Points a codebook's starting codes are chosen among, per code.
const START_CANDIDATES: usize = 4;

/// A codebook's starting codes: from a seeded sample of `c ×`
/// [`START_CANDIDATES`] of `points`, the first, then repeatedly the sampled
/// point farthest from every code chosen so far — so clusters far apart each
/// start with a code of their own, which a random start does not promise.
fn starts(points: &[Slice], c: usize, seed: u64) -> Vec<Slice> {
    let mut seen: HashSet<&Slice> = HashSet::new();
    let pool: Vec<&Slice> = picks(seed, points.len())
        .take(points.len().saturating_mul(4))
        .map(|i| &points[i])
        .filter(|p| seen.insert(*p))
        .take(c.saturating_mul(START_CANDIDATES))
        .collect();
    let Some(&first) = pool.first() else {
        return Vec::new();
    };
    let mut codes: Vec<Slice> = vec![first.clone()];
    let mut nearest_code: Vec<u32> = pool.iter().map(|p| distance(p, first)).collect();
    while codes.len() < c.min(pool.len()) {
        let (far, _) = nearest_code
            .iter()
            .enumerate()
            .max_by_key(|&(i, &d)| (d, Reverse(i)))
            .expect("the pool is not empty");
        if nearest_code[far] == 0 {
            break;
        }
        let code = pool[far].clone();
        for (d, p) in nearest_code.iter_mut().zip(&pool) {
            *d = (*d).min(distance(p, &code));
        }
        codes.push(code);
    }
    codes
}

/// A codebook of up to `c` codes over `points`: farthest-point starting codes
/// ([`starts`]), then `iterations` rounds of assigning every point its nearest
/// code and replacing each code with the per-bit majority of its members. A
/// code no point chose keeps its bits; a bit its members split evenly keeps
/// the code's.
pub fn train(points: &[Slice], c: usize, iterations: usize, seed: u64) -> Vec<Slice> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut codes = starts(points, c, seed);
    let words = points[0].len();
    for _ in 0..iterations {
        let of = assign(points, &codes);
        let mut ones = vec![vec![0u32; words * 64]; codes.len()];
        let mut members = vec![0u32; codes.len()];
        for (p, &k) in points.iter().zip(&of) {
            let k = k as usize;
            members[k] += 1;
            for (w, &word) in p.iter().enumerate() {
                for b in 0..64 {
                    ones[k][w * 64 + b] += ((word >> b) & 1) as u32;
                }
            }
        }
        for (k, code) in codes.iter_mut().enumerate() {
            let n = members[k];
            if n == 0 {
                continue;
            }
            for (w, word) in code.iter_mut().enumerate() {
                for b in 0..64 {
                    let set = 2 * ones[k][w * 64 + b];
                    if set > n {
                        *word |= 1 << b;
                    } else if set < n {
                        *word &= !(1 << b);
                    }
                }
            }
        }
    }
    codes
}

/// A two-level codebook: `coarse` codes, each split into up to `fine` codes
/// over the points nearest it. A point's code is its nearest coarse code's
/// nearest fine one — `coarse + fine` comparisons for `coarse × fine` codes.
pub struct Tree {
    pub coarse: Vec<Slice>,
    pub fine: Vec<Vec<Slice>>,
    pub fanout: usize,
}

impl Tree {
    pub fn train(
        points: &[Slice],
        coarse: usize,
        fine: usize,
        iterations: usize,
        seed: u64,
    ) -> Tree {
        let top = train(points, coarse, iterations, seed);
        let of = assign(points, &top);
        let mut members: Vec<Vec<Slice>> = vec![Vec::new(); top.len()];
        for (p, &c) in points.iter().zip(&of) {
            members[c as usize].push(p.clone());
        }
        let fine_codes: Vec<Vec<Slice>> = members
            .par_iter()
            .enumerate()
            .map(|(c, m)| train(m, fine, iterations, seed ^ c as u64))
            .collect();
        Tree {
            coarse: top,
            fine: fine_codes,
            fanout: fine,
        }
    }

    /// `point`'s code: coarse × fanout + fine.
    pub fn code(&self, point: &[u64]) -> u32 {
        let c = nearest(point, &self.coarse) as usize;
        let f = if self.fine[c].is_empty() {
            0
        } else {
            nearest(point, &self.fine[c])
        };
        (c * self.fanout) as u32 + f
    }

    /// Every point's code.
    pub fn assign(&self, points: &[Slice]) -> Vec<u32> {
        points.par_iter().map(|p| self.code(p)).collect()
    }
}

/// Which of one file's tokens a reduction keeps: the first `per_code` to
/// appear of each code. `codes[t]` is token `t`'s code. Returns `keep[t]`.
///
/// The first, not the nearest the code: the nearest are a code's most average
/// members, and kept instead they cost retrieval in every setting measured.
pub fn kept(codes: &[u32], per_code: usize) -> Vec<bool> {
    let mut taken: HashMap<u32, usize> = HashMap::new();
    codes
        .iter()
        .map(|&c| {
            let n = taken.entry(c).or_default();
            *n += 1;
            *n <= per_code
        })
        .collect()
}

/// How many files each code appears in, for every code some file holds.
pub fn file_counts(files: &[Vec<u32>]) -> Vec<usize> {
    let mut in_files = files_holding(files);
    in_files.retain(|&n| n > 0);
    in_files
}

/// How many of `files` hold each code, indexed by code.
fn files_holding(files: &[Vec<u32>]) -> Vec<usize> {
    let codes = files.iter().flatten().max().map_or(0, |&m| m as usize + 1);
    let mut in_files = vec![0usize; codes];
    for f in files {
        let distinct: HashSet<u32> = f.iter().copied().collect();
        for c in distinct {
            in_files[c as usize] += 1;
        }
    }
    in_files
}

/// The control with no codebook: every `stride`-th of `n` tokens, from the
/// first.
pub fn every(n: usize, stride: usize) -> Vec<bool> {
    (0..n).map(|t| t % stride.max(1) == 0).collect()
}

/// The control with no codebook and no order: each of `n` tokens kept with
/// probability `1/stride`, seeded.
pub fn sampled(n: usize, stride: usize, seed: u64) -> Vec<bool> {
    picks(seed, stride.max(1)).take(n).map(|r| r == 0).collect()
}

/// The runs of `false` in `keep`, as `[start, end)`.
pub fn dropped_runs(keep: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < keep.len() {
        if keep[i] {
            i += 1;
            continue;
        }
        let start = i;
        while i < keep.len() && !keep[i] {
            i += 1;
        }
        out.push((start, i));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Distance is bits that differ; the nearest code wins, the first on a
    /// tie.**
    #[test]
    fn the_nearest_code_is_the_one_fewest_bits_away() {
        assert_eq!(distance(&[0b1011, 0], &[0b0001, 1]), 3);
        let codes = vec![vec![0b1111u64], vec![0b0000], vec![0b0011]];
        assert_eq!(
            nearest(&[0b0001], &codes),
            1,
            "1 bit from 0000, 1 from 0011"
        );
        assert_eq!(nearest(&[0b0111], &codes), 0);
        assert_eq!(assign(&[vec![0b1110], vec![0b0011]], &codes), vec![0, 2]);
    }

    /// **Two well-separated clusters train to their majorities**: three points
    /// near all-zeros, three near all-ones, two codes.
    #[test]
    fn two_clusters_train_to_their_majorities() {
        let zeros = [0b0000_0001u64, 0b0000_0010, 0b0000_0000];
        let ones = [0xFFu64, 0xFE, 0xFD];
        let points: Vec<Slice> = zeros.iter().chain(&ones).map(|&w| vec![w]).collect();
        let mut codes = train(&points, 2, 4, 7);
        codes.sort();
        assert_eq!(codes, vec![vec![0x00], vec![0xFF]]);
    }

    /// **The first `per_code` of each code are kept.** Code 1 has three tokens.
    #[test]
    fn a_reduction_keeps_the_first_few_of_each_code() {
        let codes = [1, 0, 1, 1];
        assert_eq!(kept(&codes, 1), vec![true, true, false, false]);
        assert_eq!(kept(&codes, 2), vec![true, true, true, false]);
        assert_eq!(kept(&codes, 4), vec![true; 4]);
        let mut counts = file_counts(&[vec![1, 0, 1], vec![0, 3], vec![2, 0]]);
        counts.sort();
        assert_eq!(counts, vec![1, 1, 1, 3]);
    }

    /// **A two-level code is the coarse code's slot plus the fine code within
    /// it**: two clusters, each split in two.
    #[test]
    fn a_tree_code_is_its_coarse_slot_and_fine_code() {
        let points: Vec<Slice> = [0x00u64, 0x01, 0x10, 0x11, 0xF0F0, 0xF0F1, 0xFFF0, 0xFFF1]
            .iter()
            .map(|&w| vec![w])
            .collect();
        let tree = Tree::train(&points, 2, 2, 4, 3);
        let codes = tree.assign(&points);
        let low = codes[0] / 2;
        assert!(codes[..4].iter().all(|c| c / 2 == low), "{codes:?}");
        assert!(codes[4..].iter().all(|c| c / 2 != low), "{codes:?}");
        assert_eq!(tree.code(&points[5]), codes[5]);
    }

    /// The controls keep one in `stride`: exactly, by position, or in
    /// expectation, by a seeded draw.
    #[test]
    fn the_controls_keep_one_in_stride() {
        assert_eq!(
            every(7, 3),
            vec![true, false, false, true, false, false, true]
        );
        let s = sampled(40_000, 4, 9);
        let n = s.iter().filter(|&&k| k).count();
        assert!((9_000..11_000).contains(&n), "{n}");
        assert_eq!(s, sampled(40_000, 4, 9), "seeded");
    }

    /// The dropped tokens as ranges, adjacent ones merged.
    #[test]
    fn dropped_tokens_come_out_as_runs() {
        assert_eq!(
            dropped_runs(&[true, false, false, true, false]),
            vec![(1, 3), (4, 5)]
        );
        assert!(dropped_runs(&[true, true]).is_empty());
    }
}
