//! Which of a turn's tokens the gallery holds on the card, and where each kept
//! one sits among them.

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};

/// A turn's kept tokens: every one, or a bit per token. A kept token's place
/// on the card is its [`rank`](Self::rank) — how many kept tokens precede it —
/// so the turn's pages hold its kept tokens packed in order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeepMask {
    /// Every one of `n` tokens.
    All(usize),
    Bits {
        n: usize,
        words: Vec<u64>,
        /// Kept tokens before each word.
        before: Vec<u32>,
        kept: usize,
        id: u64,
    },
}

impl KeepMask {
    /// Every one of `n` tokens.
    pub fn all(n: usize) -> Self {
        Self::All(n)
    }

    /// The tokens `keep` names, one per token.
    pub fn from_bools(keep: &[bool]) -> Self {
        let mut words = vec![0u64; keep.len().div_ceil(64)];
        for (t, _) in keep.iter().enumerate().filter(|(_, &k)| k) {
            words[t / 64] |= 1 << (t % 64);
        }
        let mut before = Vec::with_capacity(words.len());
        let mut kept = 0u32;
        for w in &words {
            before.push(kept);
            kept += w.count_ones();
        }
        let mut h = DefaultHasher::new();
        keep.len().hash(&mut h);
        words.hash(&mut h);
        Self::Bits {
            n: keep.len(),
            words,
            before,
            kept: kept as usize,
            id: h.finish(),
        }
    }

    /// The first `per_code` of each code in `codes` (token `t`'s code at
    /// `codes[t]`) to appear.
    ///
    /// The first, not the nearest their code: the nearest are a code's most
    /// average members, and kept instead they cost retrieval in every setting
    /// measured (`docs/gallery_dedup.md` §2).
    pub fn first_per_code(codes: &[u32], per_code: usize) -> Self {
        let mut taken: HashMap<u32, usize> = HashMap::new();
        let keep: Vec<bool> = codes
            .iter()
            .map(|&c| {
                let n = taken.entry(c).or_default();
                *n += 1;
                *n <= per_code
            })
            .collect();
        Self::from_bools(&keep)
    }

    /// Tokens in the turn.
    pub fn len(&self) -> usize {
        match self {
            Self::All(n) | Self::Bits { n, .. } => *n,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Tokens kept.
    pub fn kept(&self) -> usize {
        match self {
            Self::All(n) => *n,
            Self::Bits { kept, .. } => *kept,
        }
    }

    /// Whether token `t` is kept.
    #[inline]
    pub fn keeps(&self, t: usize) -> bool {
        match self {
            Self::All(n) => t < *n,
            Self::Bits { n, words, .. } => t < *n && words[t / 64] >> (t % 64) & 1 == 1,
        }
    }

    /// Kept tokens before token `t` — a kept token's place among them.
    #[inline]
    pub fn rank(&self, t: usize) -> usize {
        match self {
            Self::All(n) => t.min(*n),
            Self::Bits {
                n, words, before, ..
            } => {
                let t = t.min(*n);
                let (w, b) = (t / 64, t % 64);
                if w == words.len() {
                    return before
                        .last()
                        .map_or(0, |&k| k as usize + words[w - 1].count_ones() as usize);
                }
                before[w] as usize + (words[w] & ((1u64 << b) - 1)).count_ones() as usize
            }
        }
    }

    /// Identifies the kept tokens: masks with one id keep the same ones.
    pub fn id(&self) -> u64 {
        match self {
            Self::All(n) => *n as u64,
            Self::Bits { id, .. } => *id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **The first `per_code` of each code are kept.** Code 1 has three tokens.
    #[test]
    fn the_first_few_of_each_code_are_kept() {
        let codes = [1, 0, 1, 1];
        let one = KeepMask::first_per_code(&codes, 1);
        assert_eq!(
            (0..4).map(|t| one.keeps(t)).collect::<Vec<_>>(),
            vec![true, true, false, false]
        );
        assert_eq!(one.kept(), 2);
        assert_eq!(KeepMask::first_per_code(&codes, 2).kept(), 3);
        assert_eq!(KeepMask::first_per_code(&codes, 4).kept(), 4);
    }

    /// **A kept token's rank is how many kept tokens precede it**, across word
    /// boundaries, and the rank past the end is the kept count.
    #[test]
    fn rank_counts_the_kept_tokens_before() {
        let keep: Vec<bool> = (0..130).map(|t| t % 3 == 0).collect();
        let m = KeepMask::from_bools(&keep);
        assert_eq!(m.len(), 130);
        assert_eq!(m.kept(), 44);
        assert_eq!(m.rank(0), 0);
        assert_eq!(m.rank(3), 1);
        assert_eq!(m.rank(4), 2);
        assert_eq!(m.rank(63), 21);
        assert_eq!(m.rank(64), 22, "0,3,..,63 is 22 tokens");
        assert_eq!(m.rank(129), 43);
        assert_eq!(m.rank(130), 44);
        assert_eq!(m.rank(500), 44, "clamped to the turn");
        assert!(m.keeps(129) && !m.keeps(128) && !m.keeps(130));
        let full = KeepMask::from_bools(&[true; 128]);
        assert_eq!(full.rank(128), 128, "past the end on a word boundary");
    }

    /// Every token: rank is position, everything kept.
    #[test]
    fn all_keeps_every_token() {
        let m = KeepMask::all(5);
        assert_eq!((m.len(), m.kept(), m.rank(3), m.rank(9)), (5, 5, 3, 5));
        assert!(m.keeps(4) && !m.keeps(5));
    }

    /// One id per kept set.
    #[test]
    fn the_id_names_the_kept_tokens() {
        let a = KeepMask::from_bools(&[true, false, true]);
        assert_eq!(a.id(), KeepMask::from_bools(&[true, false, true]).id());
        assert_ne!(a.id(), KeepMask::from_bools(&[true, true, false]).id());
        assert!(KeepMask::from_bools(&[]).is_empty());
    }
}
