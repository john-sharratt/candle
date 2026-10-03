//! Tokenization the compiler does once per distinct string.
//!
//! Lowering a grammar tokenizes in context: every static run and every arm of
//! every branch is encoded after the text that precedes it. A tool-call loop
//! repeats its whole call grammar once per call it admits, so the same name
//! arms and the same key runs are encoded at every level — the same text in the
//! same left context, tokenized again to the same answer. Encoding is the
//! compiler's dominant cost against a real tokenizer, so each distinct string is
//! encoded once.

use std::cell::RefCell;
use std::collections::HashMap;

use super::vocab::{TokenId, Vocab};

/// A [`Vocab`] that remembers what it has encoded. Everything but `encode` is
/// the wrapped vocabulary's own.
pub(super) struct EncodeCache<'a> {
    inner: &'a dyn Vocab,
    encoded: RefCell<HashMap<String, Vec<TokenId>>>,
}

impl<'a> EncodeCache<'a> {
    pub(super) fn new(inner: &'a dyn Vocab) -> Self {
        Self {
            inner,
            encoded: RefCell::new(HashMap::new()),
        }
    }
}

impl Vocab for EncodeCache<'_> {
    fn encode(&self, text: &str) -> Vec<TokenId> {
        if let Some(tokens) = self.encoded.borrow().get(text) {
            return tokens.clone();
        }
        let tokens = self.inner.encode(text);
        self.encoded
            .borrow_mut()
            .insert(text.to_string(), tokens.clone());
        tokens
    }

    fn token_bytes(&self, token: TokenId) -> Vec<u8> {
        self.inner.token_bytes(token)
    }

    fn eos(&self) -> TokenId {
        self.inner.eos()
    }

    fn end_tokens(&self) -> Vec<TokenId> {
        self.inner.end_tokens()
    }

    fn fingerprint(&self) -> u64 {
        self.inner.fingerprint()
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::stencil::vocab::TestVocab;

    /// A vocabulary that counts how often it is asked to encode.
    struct Counting {
        inner: TestVocab,
        encodes: Cell<usize>,
    }

    impl Vocab for Counting {
        fn encode(&self, text: &str) -> Vec<TokenId> {
            self.encodes.set(self.encodes.get() + 1);
            self.inner.encode(text)
        }
        fn token_bytes(&self, token: TokenId) -> Vec<u8> {
            self.inner.token_bytes(token)
        }
        fn eos(&self) -> TokenId {
            self.inner.eos()
        }
        fn end_tokens(&self) -> Vec<TokenId> {
            self.inner.end_tokens()
        }
        fn fingerprint(&self) -> u64 {
            self.inner.fingerprint()
        }
    }

    fn counting() -> Counting {
        Counting {
            inner: TestVocab::new().with_special("{\"", 300),
            encodes: Cell::new(0),
        }
    }

    #[test]
    fn a_string_is_encoded_once_however_often_it_is_asked_for() {
        let vocab = counting();
        let cache = EncodeCache::new(&vocab);
        let first = cache.encode("{\"a");
        assert_eq!(first, vec![300, b'a' as TokenId]);
        assert_eq!(cache.encode("{\"a"), first);
        assert_eq!(cache.encode("{\"a"), first);
        assert_eq!(vocab.encodes.get(), 1);
        assert_eq!(cache.encode("b"), vec![b'b' as TokenId]);
        assert_eq!(vocab.encodes.get(), 2);
    }

    #[test]
    fn everything_but_encode_is_the_wrapped_vocabularys_own() {
        let vocab = counting();
        let cache = EncodeCache::new(&vocab);
        assert_eq!(cache.eos(), vocab.eos());
        assert_eq!(cache.end_tokens(), vocab.end_tokens());
        assert_eq!(cache.fingerprint(), vocab.fingerprint());
        assert_eq!(cache.token_bytes(300), b"{\"".to_vec());
    }
}
