//! A two-level binary codebook over gallery signatures.
//!
//! A token's folded signature is a sign vector, and the scan's measure of two
//! tokens voting alike is Hamming agreement. Binary k-means under Hamming
//! distance — each code the per-bit majority of its members — names every token
//! by its nearest code. Two levels: `coarse` codes, each split into up to
//! `fine` codes over the points nearest it, so naming a token takes `coarse +
//! fine` comparisons for `coarse × fine` codes.
//!
//! Signatures are already bits: nothing is hashed. Points and codes are held
//! flat, `width` words apiece.

use std::cmp::Reverse;
use std::collections::HashSet;
use std::hash::{DefaultHasher, Hash, Hasher};

use rayon::prelude::*;

/// Hamming distance between two slices of one width.
#[inline]
fn distance(a: &[u64], b: &[u64]) -> u32 {
    a.iter().zip(b).map(|(x, y)| (x ^ y).count_ones()).sum()
}

/// The index of the code in `codes` (flat, `width` words each) nearest
/// `point`, the first on a tie.
fn nearest(point: &[u64], codes: &[u64], width: usize) -> u32 {
    let mut best = (u32::MAX, 0u32);
    for (i, c) in codes.chunks_exact(width).enumerate() {
        let d = distance(point, c);
        if d < best.0 {
            best = (d, i as u32);
        }
    }
    best.1
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

/// Points a level's starting codes are chosen among, per code.
const START_CANDIDATES: usize = 4;

/// A level's starting codes: from a seeded sample of `c ×` [`START_CANDIDATES`]
/// distinct points, the first, then repeatedly the sampled point farthest from
/// every code chosen so far — so clusters far apart each start with a code of
/// their own, which a random start does not promise. Fewer than `c` when the
/// points hold fewer distinct values.
fn starts(points: &[u64], width: usize, c: usize, seed: u64) -> Vec<u64> {
    let n = points.len() / width;
    let point = |i: usize| &points[i * width..(i + 1) * width];
    let mut seen: HashSet<&[u64]> = HashSet::new();
    let pool: Vec<&[u64]> = picks(seed, n)
        .take(n.saturating_mul(4))
        .map(point)
        .filter(|p| seen.insert(*p))
        .take(c.saturating_mul(START_CANDIDATES))
        .collect();
    let Some(&first) = pool.first() else {
        return Vec::new();
    };
    let mut codes: Vec<u64> = first.to_vec();
    let mut nearest_code: Vec<u32> = pool.iter().map(|p| distance(p, first)).collect();
    while codes.len() / width < c.min(pool.len()) {
        let (far, _) = nearest_code
            .iter()
            .enumerate()
            .max_by_key(|&(i, &d)| (d, Reverse(i)))
            .expect("the pool is not empty");
        if nearest_code[far] == 0 {
            break;
        }
        let code = pool[far];
        for (d, p) in nearest_code.iter_mut().zip(&pool) {
            *d = (*d).min(distance(p, code));
        }
        codes.extend_from_slice(code);
    }
    codes
}

/// One level: up to `c` codes over `points` — [`starts`], then `iterations`
/// rounds of assigning every point its nearest code and replacing each code
/// with the per-bit majority of its members. A code no point chose keeps its
/// bits; a bit its members split evenly keeps the code's.
fn train_level(points: &[u64], width: usize, c: usize, iterations: usize, seed: u64) -> Vec<u64> {
    if points.is_empty() {
        return Vec::new();
    }
    let mut codes = starts(points, width, c, seed);
    let k = codes.len() / width;
    for _ in 0..iterations {
        let of: Vec<u32> = points
            .par_chunks_exact(width)
            .map(|p| nearest(p, &codes, width))
            .collect();
        let mut ones = vec![0u32; k * width * 64];
        let mut members = vec![0u32; k];
        for (p, &code) in points.chunks_exact(width).zip(&of) {
            let code = code as usize;
            members[code] += 1;
            let ones = &mut ones[code * width * 64..(code + 1) * width * 64];
            for (w, &word) in p.iter().enumerate() {
                for b in 0..64 {
                    ones[w * 64 + b] += ((word >> b) & 1) as u32;
                }
            }
        }
        for (code, words) in codes.chunks_exact_mut(width).enumerate() {
            let n = members[code];
            if n == 0 {
                continue;
            }
            let ones = &ones[code * width * 64..(code + 1) * width * 64];
            for (w, word) in words.iter_mut().enumerate() {
                for b in 0..64 {
                    let set = 2 * ones[w * 64 + b];
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

/// A two-level codebook over `width`-word signatures.
pub struct Codebook {
    width: usize,
    fanout: usize,
    coarse: Vec<u64>,
    /// Each coarse code's fine codes; empty for a coarse code no point chose.
    fine: Vec<Vec<u64>>,
    id: u64,
}

impl Codebook {
    /// Train on `points`, `width` words each: `coarse` codes, each split into
    /// up to `fine` over its members, `iterations` rounds of k-means a level.
    /// Deterministic in `seed`.
    pub fn train(
        points: &[u64],
        width: usize,
        coarse: usize,
        fine: usize,
        iterations: usize,
        seed: u64,
    ) -> Codebook {
        assert!(width > 0 && points.len().is_multiple_of(width));
        let top = train_level(points, width, coarse, iterations, seed);
        let k = top.len() / width;
        let mut members: Vec<Vec<u64>> = vec![Vec::new(); k];
        for p in points.chunks_exact(width) {
            members[nearest(p, &top, width) as usize].extend_from_slice(p);
        }
        let fine_codes: Vec<Vec<u64>> = members
            .par_iter()
            .enumerate()
            .map(|(c, m)| train_level(m, width, fine, iterations, seed ^ c as u64))
            .collect();
        let mut h = DefaultHasher::new();
        width.hash(&mut h);
        fine.hash(&mut h);
        top.hash(&mut h);
        fine_codes.hash(&mut h);
        Codebook {
            width,
            fanout: fine,
            coarse: top,
            fine: fine_codes,
            id: h.finish(),
        }
    }

    /// Words a signature it names holds.
    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    /// Identifies the codes: two codebooks with one id name every point alike.
    #[inline]
    pub fn id(&self) -> u64 {
        self.id
    }

    /// `point`'s code: its nearest coarse code's slot, `× fanout`, plus its
    /// nearest fine code within it.
    pub fn code(&self, point: &[u64]) -> u32 {
        let c = nearest(point, &self.coarse, self.width) as usize;
        let f = if self.fine[c].is_empty() {
            0
        } else {
            nearest(point, &self.fine[c], self.width)
        };
        (c * self.fanout) as u32 + f
    }

    /// Every point's code, `points` flat at [`Self::width`] words apiece.
    pub fn codes(&self, points: &[u64]) -> Vec<u32> {
        points
            .par_chunks_exact(self.width)
            .map(|p| self.code(p))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Distance is bits that differ; the nearest code wins, the first on a
    /// tie.**
    #[test]
    fn the_nearest_code_is_the_one_fewest_bits_away() {
        assert_eq!(distance(&[0b1011, 0], &[0b0001, 1]), 3);
        let codes = [0b1111u64, 0b0000, 0b0011];
        assert_eq!(
            nearest(&[0b0001], &codes, 1),
            1,
            "1 bit from 0000, 1 from 0011"
        );
        assert_eq!(nearest(&[0b0111], &codes, 1), 0);
        assert_eq!(
            nearest(&[0b0011, 0], &[0, 0, 0b0011, 1], 2),
            1,
            "2 bits from the first code, 1 from the second"
        );
    }

    /// **Two well-separated clusters train to their majorities**: three points
    /// near all-zeros, three near all-ones, two codes.
    #[test]
    fn two_clusters_train_to_their_majorities() {
        let points = [0b0000_0001u64, 0b0000_0010, 0b0000_0000, 0xFF, 0xFE, 0xFD];
        let mut codes = train_level(&points, 1, 2, 4, 7);
        codes.sort();
        assert_eq!(codes, vec![0x00, 0xFF]);
    }

    /// Starting codes are distinct points; a pool of one value gives one code.
    #[test]
    fn starts_are_distinct_points() {
        assert_eq!(starts(&[5, 5, 5, 5], 1, 3, 1), vec![5]);
        let mut s = starts(&[0, 0xFF, 0, 0xFF], 1, 2, 1);
        s.sort();
        assert_eq!(s, vec![0, 0xFF]);
    }

    /// **A two-level code is the coarse code's slot plus the fine code within
    /// it**: two clusters, each split in two.
    #[test]
    fn a_code_is_its_coarse_slot_and_fine_code() {
        let points = [0x00u64, 0x01, 0x10, 0x11, 0xF0F0, 0xF0F1, 0xFFF0, 0xFFF1];
        let book = Codebook::train(&points, 1, 2, 2, 4, 3);
        let codes = book.codes(&points);
        let low = codes[0] / 2;
        assert!(codes[..4].iter().all(|c| c / 2 == low), "{codes:?}");
        assert!(codes[4..].iter().all(|c| c / 2 != low), "{codes:?}");
        assert_eq!(book.code(&points[5..6]), codes[5]);
        assert_eq!(book.width(), 1);
    }

    /// **Training is deterministic in its seed**, and the id names the codes.
    #[test]
    fn training_is_seeded_and_the_id_names_the_codes() {
        let points: Vec<u64> = (0..64u64)
            .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .collect();
        let a = Codebook::train(&points, 2, 4, 4, 3, 11);
        let b = Codebook::train(&points, 2, 4, 4, 3, 11);
        assert_eq!(a.id(), b.id());
        assert_eq!(a.codes(&points), b.codes(&points));
        let c = Codebook::train(&points, 2, 2, 4, 3, 11);
        assert_ne!(a.id(), c.id(), "a different shape is a different codebook");
    }
}
