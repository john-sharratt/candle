//! A fixed-width bitset over a model's vocabulary.
//!
//! One bit per token id, packed 32 to a word in ascending id order. The layout
//! is the kernel's, not a convenience: a warp's lanes take consecutive token
//! ids, so all 32 of them read the SAME word and a membership test costs one
//! coalesced load plus a shift and a mask — `(w[t >> 5] >> (t & 31)) & 1`. The
//! sampler already does exactly this for its recent-token set.
//!
//! Sizing follows from that and nothing else: a 248,320-token vocabulary is
//! 30.3 KiB, which is a rounding error against an L2 and is built once at load
//! rather than uploaded per step.

/// A set of token ids, one bit each.
#[derive(Clone, PartialEq, Eq)]
pub struct TokenBitset {
    words: Vec<u32>,
    vocab: usize,
}

impl TokenBitset {
    /// An empty set over `vocab` token ids.
    pub fn new(vocab: usize) -> Self {
        Self {
            words: vec![0; vocab.div_ceil(32)],
            vocab,
        }
    }

    /// Token ids this set is defined over. Ids at or past this are not
    /// representable and are rejected rather than silently dropped — a set
    /// built against the wrong vocabulary would suppress unrelated tokens.
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Add `token`. Returns `false` if it lies outside the vocabulary.
    pub fn insert(&mut self, token: u32) -> bool {
        let t = token as usize;
        if t >= self.vocab {
            return false;
        }
        self.words[t >> 5] |= 1u32 << (t & 31);
        true
    }

    /// Is `token` in the set? Out-of-range ids are not members.
    pub fn contains(&self, token: u32) -> bool {
        let t = token as usize;
        if t >= self.vocab {
            return false;
        }
        (self.words[t >> 5] >> (t & 31)) & 1 == 1
    }

    /// `self |= other`.
    ///
    /// The union is what builds a suppression set out of per-language
    /// blacklists, so it is the operation this type exists for.
    pub fn union_with(&mut self, other: &Self) {
        debug_assert_eq!(self.vocab, other.vocab, "union across vocabularies");
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a |= *b;
        }
    }

    /// `self &= !other`.
    pub fn subtract(&mut self, other: &Self) {
        debug_assert_eq!(self.vocab, other.vocab, "subtract across vocabularies");
        for (a, b) in self.words.iter_mut().zip(other.words.iter()) {
            *a &= !*b;
        }
    }

    /// Members.
    pub fn len(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.words.iter().all(|w| *w == 0)
    }

    /// The packed words, for upload. Ascending token id, 32 per word, LSB
    /// first — the layout the kernel indexes.
    pub fn words(&self) -> &[u32] {
        &self.words
    }

    /// Members in ascending order. Diagnostics and tests; the kernel reads
    /// [`Self::words`].
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.vocab as u32).filter(move |t| self.contains(*t))
    }
}

impl std::fmt::Debug for TokenBitset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "TokenBitset({} of {})", self.len(), self.vocab)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_contains_round_trip() {
        let mut b = TokenBitset::new(100);
        assert!(b.is_empty());
        for t in [0u32, 1, 31, 32, 63, 64, 99] {
            assert!(b.insert(t), "{t} should be in range");
        }
        for t in 0..100u32 {
            let want = matches!(t, 0 | 1 | 31 | 32 | 63 | 64 | 99);
            assert_eq!(b.contains(t), want, "token {t}");
        }
        assert_eq!(b.len(), 7);
    }

    /// The word boundary is where a hand-rolled shift goes wrong, so it is
    /// asserted against the packed representation directly rather than through
    /// `contains`.
    #[test]
    fn packing_is_ascending_id_lsb_first() {
        let mut b = TokenBitset::new(64);
        b.insert(0);
        b.insert(5);
        b.insert(31);
        b.insert(32);
        assert_eq!(b.words()[0], (1 << 0) | (1 << 5) | (1 << 31));
        assert_eq!(b.words()[1], 1 << 0);
    }

    #[test]
    fn out_of_range_is_refused_not_wrapped() {
        let mut b = TokenBitset::new(40);
        assert!(!b.insert(40), "id at vocab must be refused");
        assert!(!b.insert(u32::MAX));
        assert!(!b.contains(40));
        assert!(
            b.is_empty(),
            "a refused insert must not land somewhere else"
        );
    }

    #[test]
    fn vocab_not_a_multiple_of_32_still_addresses_every_id() {
        let mut b = TokenBitset::new(70);
        assert_eq!(b.words().len(), 3, "70 ids need ceil(70/32) = 3 words");
        assert!(b.insert(69));
        assert!(b.contains(69));
        assert!(!b.insert(70));
    }

    #[test]
    fn union_is_set_union() {
        let mut a = TokenBitset::new(100);
        let mut b = TokenBitset::new(100);
        a.insert(1);
        a.insert(50);
        b.insert(50);
        b.insert(70);
        a.union_with(&b);
        assert_eq!(a.iter().collect::<Vec<_>>(), vec![1, 50, 70]);
    }

    #[test]
    fn subtract_removes_only_the_other_set() {
        let mut a = TokenBitset::new(100);
        for t in [1u32, 2, 3, 4] {
            a.insert(t);
        }
        let mut b = TokenBitset::new(100);
        b.insert(2);
        b.insert(99);
        a.subtract(&b);
        assert_eq!(a.iter().collect::<Vec<_>>(), vec![1, 3, 4]);
    }
}
