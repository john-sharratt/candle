//! The sampling stencil — restricting the sampler to an allowed set, and the
//! free-text close-token boost.
//!
//! Standalone, these operate on a `&mut [f32]` logits slice; the integration
//! applies the same operations per-row in the wave batch.

use super::vocab::TokenId;

const NEG_INF: f32 = f32::NEG_INFINITY;

/// A sorted, deduped set of permitted tokens — a branch frontier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllowedSet {
    tokens: Vec<TokenId>,
}

impl AllowedSet {
    pub fn from_tokens(mut tokens: Vec<TokenId>) -> Self {
        tokens.sort_unstable();
        tokens.dedup();
        AllowedSet { tokens }
    }

    pub fn tokens(&self) -> &[TokenId] {
        &self.tokens
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    pub fn contains(&self, t: TokenId) -> bool {
        self.tokens.binary_search(&t).is_ok()
    }

    /// Set every disallowed logit to `-inf`.  Applied *after* penalties so a
    /// penalty can never resurrect a disallowed token.
    pub fn apply(&self, logits: &mut [f32]) {
        for (i, l) in logits.iter_mut().enumerate() {
            if !self.contains(i as TokenId) {
                *l = NEG_INF;
            }
        }
    }
}

/// Ban a single token (set its logit to `-inf`) — used to forbid EOS inside a
/// non-EOS-terminated free-text span.
pub fn ban(logits: &mut [f32], token: TokenId) {
    if let Some(l) = logits.get_mut(token as usize) {
        *l = NEG_INF;
    }
}

/// Add `amount` to a token's logit — the soft close-token ramp in a free-text
/// span.  A no-op for `amount == 0.0`.
pub fn boost(logits: &mut [f32], token: TokenId, amount: f32) {
    if amount != 0.0 {
        if let Some(l) = logits.get_mut(token as usize) {
            *l += amount;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_set_dedups_and_sorts() {
        let s = AllowedSet::from_tokens(vec![5, 1, 5, 3]);
        assert_eq!(s.tokens(), &[1, 3, 5]);
        assert_eq!(s.len(), 3);
        assert!(s.contains(3));
        assert!(!s.contains(4));
    }

    #[test]
    fn apply_masks_disallowed() {
        let s = AllowedSet::from_tokens(vec![1, 3]);
        let mut logits = vec![0.0f32; 5];
        s.apply(&mut logits);
        assert_eq!(logits[0], NEG_INF);
        assert_eq!(logits[1], 0.0);
        assert_eq!(logits[2], NEG_INF);
        assert_eq!(logits[3], 0.0);
        assert_eq!(logits[4], NEG_INF);
    }

    #[test]
    fn ban_one() {
        let mut logits = vec![1.0f32; 3];
        ban(&mut logits, 1);
        assert_eq!(logits, vec![1.0, NEG_INF, 1.0]);
        ban(&mut logits, 99); // out of range: no-op
    }

    #[test]
    fn boost_one() {
        let mut logits = vec![1.0f32; 3];
        boost(&mut logits, 2, 4.0);
        assert_eq!(logits[2], 5.0);
        boost(&mut logits, 0, 0.0); // no-op
        assert_eq!(logits[0], 1.0);
    }
}
