//! [`TokenBuffer`] — a strongly-typed, growable sequence of token IDs.
//!
//! Wraps `Vec<u32>` with [`Deref`]/[`DerefMut`] so every `Vec` and slice
//! method is available directly on the buffer.  The newtype makes call
//! sites self-documenting and adds a small set of token-domain helpers.

use std::ops::{Deref, DerefMut};

// ────────────────────────────────────────────────────────────────────────────
// TokenBuffer
// ────────────────────────────────────────────────────────────────────────────

/// A growable, ordered sequence of token IDs.
///
/// `TokenBuffer` wraps `Vec<u32>` transparently: every `Vec` method
/// (`push`, `len`, `is_empty`, `iter`, `chunks`, `last`, …) and every slice
/// method is accessible directly via [`Deref`]/[`DerefMut`].  A reference to
/// a `TokenBuffer` also coerces automatically to `&[u32]`, so it can be
/// passed wherever a slice is expected without any explicit conversion.
///
/// # Construction
///
/// ```rust,ignore
/// let buf = TokenBuffer::new();
/// let buf = TokenBuffer::with_capacity(512);
/// let buf = TokenBuffer::from(vec![1_u32, 2, 3]);
/// let buf: TokenBuffer = enc.get_ids().into();
/// ```
///
/// # Token-domain helpers
///
/// Beyond standard `Vec`/slice methods, `TokenBuffer` adds:
/// - [`last_token`](Self::last_token) — copy the last token out (vs `last() → Option<&u32>`)
/// - [`token_count`](Self::token_count) — explicit alias for `len()`
/// - [`contains_token`](Self::contains_token) — search by value
/// - [`ends_with_any`](Self::ends_with_any) — end-of-sequence / EOS check
/// - [`into_vec`](Self::into_vec) — consume and unwrap to `Vec<u32>`
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct TokenBuffer(Vec<u32>);

impl TokenBuffer {
    // ── Constructors ──────────────────────────────────────────────────────

    /// Create an empty buffer.
    #[inline]
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// Create a buffer with pre-allocated storage for `cap` tokens.
    #[inline]
    pub fn with_capacity(cap: usize) -> Self {
        Self(Vec::with_capacity(cap))
    }

    // ── Token-domain helpers ──────────────────────────────────────────────

    /// Number of tokens in the buffer.
    ///
    /// Provided for expressiveness alongside the identical `.len()` via
    /// [`Deref`].
    #[inline]
    pub fn token_count(&self) -> usize {
        self.0.len()
    }

    /// Copy the last token out of the buffer, if any.
    ///
    /// Unlike `.last()` (which returns `Option<&u32>`), this copies the
    /// value so the caller does not hold a borrow into the buffer.
    #[inline]
    pub fn last_token(&self) -> Option<u32> {
        self.0.last().copied()
    }

    /// Returns `true` if `token` appears anywhere in the buffer.
    #[inline]
    pub fn contains_token(&self, token: u32) -> bool {
        self.0.contains(&token)
    }

    /// Returns `true` if the last token is one of `candidates`.
    ///
    /// Useful for EOS / stop-sequence detection without borrowing the buffer
    /// for a full `.contains()` scan.
    #[inline]
    pub fn ends_with_any(&self, candidates: &[u32]) -> bool {
        self.0.last().is_some_and(|t| candidates.contains(t))
    }

    /// Consume the buffer, returning the underlying `Vec<u32>`.
    #[inline]
    pub fn into_vec(self) -> Vec<u32> {
        self.0
    }
}

// ── Deref / DerefMut ─────────────────────────────────────────────────────────

impl Deref for TokenBuffer {
    type Target = Vec<u32>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for TokenBuffer {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

// ── Conversions ───────────────────────────────────────────────────────────────

impl From<Vec<u32>> for TokenBuffer {
    #[inline]
    fn from(v: Vec<u32>) -> Self {
        Self(v)
    }
}

impl From<TokenBuffer> for Vec<u32> {
    #[inline]
    fn from(b: TokenBuffer) -> Self {
        b.0
    }
}

impl From<&[u32]> for TokenBuffer {
    #[inline]
    fn from(s: &[u32]) -> Self {
        Self(s.to_vec())
    }
}

impl FromIterator<u32> for TokenBuffer {
    fn from_iter<I: IntoIterator<Item = u32>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl IntoIterator for TokenBuffer {
    type Item = u32;
    type IntoIter = std::vec::IntoIter<u32>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a TokenBuffer {
    type Item = &'a u32;
    type IntoIter = std::slice::Iter<'a, u32>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

// Cross-type PartialEq so `assert_eq!(buf, vec![...])` works in tests.
impl PartialEq<Vec<u32>> for TokenBuffer {
    fn eq(&self, other: &Vec<u32>) -> bool {
        self.0 == *other
    }
}

impl PartialEq<TokenBuffer> for Vec<u32> {
    fn eq(&self, other: &TokenBuffer) -> bool {
        *self == other.0
    }
}

impl PartialEq<&[u32]> for TokenBuffer {
    fn eq(&self, other: &&[u32]) -> bool {
        self.0.as_slice() == *other
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deref_gives_vec_methods() {
        let mut buf = TokenBuffer::from(vec![1, 2, 3]);
        buf.push(4);
        assert_eq!(buf.len(), 4);
        assert_eq!(&buf[..], &[1, 2, 3, 4]);
    }

    #[test]
    fn last_token_copies_value() {
        let buf = TokenBuffer::from(vec![10, 20, 30]);
        assert_eq!(buf.last_token(), Some(30));
        // buffer is still available — no borrow conflict
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn ends_with_any_eos_check() {
        let eos = &[0u32, 99u32];
        let buf = TokenBuffer::from(vec![1, 2, 99]);
        assert!(buf.ends_with_any(eos));
        let buf2 = TokenBuffer::from(vec![1, 2, 3]);
        assert!(!buf2.ends_with_any(eos));
    }

    #[test]
    fn contains_token() {
        let buf = TokenBuffer::from(vec![5, 10, 15]);
        assert!(buf.contains_token(10));
        assert!(!buf.contains_token(7));
    }

    #[test]
    fn from_slice_and_into_vec_roundtrip() {
        let v = vec![1u32, 2, 3];
        let buf = TokenBuffer::from(v.as_slice());
        let out: Vec<u32> = buf.into_vec();
        assert_eq!(out, vec![1, 2, 3]);
    }

    #[test]
    fn coerces_to_slice() {
        fn takes_slice(s: &[u32]) -> usize { s.len() }
        let buf = TokenBuffer::from(vec![1, 2, 3]);
        assert_eq!(takes_slice(&buf), 3);
    }

    #[test]
    fn from_iterator() {
        let buf: TokenBuffer = (0u32..5).collect();
        assert_eq!(&buf[..], &[0, 1, 2, 3, 4]);
    }
}
