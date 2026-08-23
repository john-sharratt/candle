//! Tokenization abstraction.
//!
//! The stencil compiler needs to turn the strings of a grammar into tokens *in
//! context* (so boundary merges match what a real decode produces), and the
//! free-text terminator needs the *bytes* a token decodes to.  Both are hidden
//! behind [`Vocab`] so the whole module is testable without a model: tests use
//! [`TestVocab`] (a deterministic byte-level tokenizer with explicit merges),
//! while real use wraps a `tokenizers::Tokenizer` via [`HfVocab`].

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// A token id, as produced by the tokenizer.
pub type TokenId = u32;

/// Everything the stencil module needs from a tokenizer.
pub trait Vocab {
    /// Tokenize `text` from a clean left boundary.
    fn encode(&self, text: &str) -> Vec<TokenId>;

    /// The raw bytes `token` decodes to (used by the free-text terminator and
    /// by healing).  Special/control tokens with no textual content return `&[]`.
    fn token_bytes(&self, token: TokenId) -> Vec<u8>;

    /// The end-of-sequence token id.
    fn eos(&self) -> TokenId;

    /// A stable fingerprint of this vocabulary.  Stored on a compiled tree so a
    /// later mismatch (tree compiled against a different tokenizer) fails loudly.
    fn fingerprint(&self) -> u64;

    /// Decode a token run to bytes (default: concatenate `token_bytes`).
    fn decode(&self, tokens: &[TokenId]) -> Vec<u8> {
        tokens.iter().flat_map(|&t| self.token_bytes(t)).collect()
    }
}

// ── Deterministic test tokenizer ────────────────────────────────────────────

/// A byte-level tokenizer with explicit, controllable merges — deterministic and
/// dependency-free, so every stencil test is reproducible.
///
/// Token-id space:
/// - `0..256`  — one id per byte (id == byte value).
/// - `256..`   — registered "specials": a string that encodes to a single id,
///   matched **longest-first** (so registering `"{\""` makes `{"` one token).
///   Used both for true special tokens (`<tool_call>`) and to force the BPE-like
///   merges the boundary tests need.
#[derive(Clone)]
pub struct TestVocab {
    /// (bytes, id), sorted by descending byte-length for longest-match.
    specials: Vec<(Vec<u8>, TokenId)>,
    eos: TokenId,
}

impl Default for TestVocab {
    fn default() -> Self {
        Self::new()
    }
}

impl TestVocab {
    /// A byte-level vocab with `eos = 256` and no other specials.
    pub fn new() -> Self {
        TestVocab {
            specials: Vec::new(),
            eos: 256,
        }
    }

    /// Register a special string `s` as the single token `id` (`id` must be
    /// `>= 256`).  Re-sorts for longest-match.  Returns `self` for chaining.
    pub fn with_special(mut self, s: &str, id: TokenId) -> Self {
        assert!(id >= 256, "special token ids must be >= 256");
        self.specials.push((s.as_bytes().to_vec(), id));
        self.specials.sort_by_key(|s| std::cmp::Reverse(s.0.len()));
        self
    }

    /// Override the eos id.
    pub fn with_eos(mut self, id: TokenId) -> Self {
        self.eos = id;
        self
    }
}

impl Vocab for TestVocab {
    fn encode(&self, text: &str) -> Vec<TokenId> {
        let bytes = text.as_bytes();
        let mut out = Vec::new();
        let mut i = 0;
        'outer: while i < bytes.len() {
            for (s, id) in &self.specials {
                if bytes[i..].starts_with(s) {
                    out.push(*id);
                    i += s.len();
                    continue 'outer;
                }
            }
            out.push(bytes[i] as TokenId);
            i += 1;
        }
        out
    }

    fn token_bytes(&self, token: TokenId) -> Vec<u8> {
        if token < 256 {
            return vec![token as u8];
        }
        if token == self.eos {
            return Vec::new();
        }
        for (s, id) in &self.specials {
            if *id == token {
                return s.clone();
            }
        }
        Vec::new()
    }

    fn eos(&self) -> TokenId {
        self.eos
    }

    fn fingerprint(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.eos.hash(&mut h);
        for (s, id) in &self.specials {
            s.hash(&mut h);
            id.hash(&mut h);
        }
        h.finish()
    }
}

// ── Real tokenizer wrapper ──────────────────────────────────────────────────

/// Wraps a `tokenizers::Tokenizer` for production use.  Not exercised by the
/// unit suites (which use [`TestVocab`]); provided so a compiled tree is usable
/// against a real model.
pub struct HfVocab {
    tok: tokenizers::Tokenizer,
    eos: TokenId,
    fingerprint: u64,
}

impl HfVocab {
    pub fn new(tok: tokenizers::Tokenizer, eos: TokenId, fingerprint: u64) -> Self {
        HfVocab {
            tok,
            eos,
            fingerprint,
        }
    }
}

impl Vocab for HfVocab {
    fn encode(&self, text: &str) -> Vec<TokenId> {
        self.tok
            .encode(text, false)
            .map(|e| e.get_ids().to_vec())
            .unwrap_or_default()
    }

    fn token_bytes(&self, token: TokenId) -> Vec<u8> {
        self.tok
            .decode(&[token], false)
            .map(|s| s.into_bytes())
            .unwrap_or_default()
    }

    fn eos(&self) -> TokenId {
        self.eos
    }

    fn fingerprint(&self) -> u64 {
        self.fingerprint
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_level_round_trip() {
        let v = TestVocab::new();
        let toks = v.encode("ab{");
        assert_eq!(toks, vec![b'a' as u32, b'b' as u32, b'{' as u32]);
        assert_eq!(v.decode(&toks), b"ab{");
        assert_eq!(v.token_bytes(b'{' as u32), vec![b'{']);
    }

    #[test]
    fn longest_match_specials() {
        // "{\"" merges to one id; "<tool_call>" is a single special.
        let v = TestVocab::new()
            .with_special("{\"", 300)
            .with_special("<tool_call>", 1000);
        assert_eq!(v.encode("<tool_call>{\"x"), vec![1000, 300, b'x' as u32]);
        // Longest-match: a longer special wins over a shorter prefix.
        let v2 = v.with_special("{", 301);
        assert_eq!(v2.encode("{\""), vec![300]); // "{\"" beats "{"
        assert_eq!(v2.encode("{a"), vec![301, b'a' as u32]); // "{" alone
    }

    #[test]
    fn special_token_bytes() {
        let v = TestVocab::new().with_special("<tool_call>", 1000);
        assert_eq!(v.token_bytes(1000), b"<tool_call>");
        assert_eq!(v.token_bytes(v.eos()), Vec::<u8>::new());
    }

    #[test]
    fn fingerprint_changes_with_specials() {
        let a = TestVocab::new();
        let b = TestVocab::new().with_special("x", 300);
        assert_ne!(a.fingerprint(), b.fingerprint());
    }
}
