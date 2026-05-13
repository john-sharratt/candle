//! [`TokenizedText`] — a text string with lazily-computed token ids.
//!
//! Lives inside the tree module because it is the bridge between
//! conversation semantics (role, text) and the KV-cache layout (token spans).
//! Every turn node and the system prompt hold a `TokenizedText`; the KV
//! scheduler uses `.tokens(tokenizer)` to obtain the token ids on first
//! access, after which the result is cached and the tokenizer is no longer
//! needed.

use std::sync::{Arc, OnceLock};

use crate::token_buffer::TokenBuffer;

/// Text paired with its lazily-computed token ids.
///
/// # Lifecycle
///
/// ```text
/// TokenizedText::plaintext("hello")   ← text set, no tokens yet
///     │
///     └─ .tokens(tokenizer)           ← tokenized once, cached in OnceLock
///            ║
///            ▼
///     TokenizedText { text: "hello", token_ids: [15339] }   (immutable)
/// ```
///
/// `TokenizedText` can also be constructed pre-tokenized via
/// [`TokenizedText::new`] when the token ids are already available (the
/// common case when the tokenization happens outside of a tree node).
///
/// # Cloning
///
/// The text portion is reference-counted ([`Arc<str>`]), so cloning is O(1)
/// for the text. If tokens have been computed, the clone copies the
/// `TokenBuffer`; otherwise the clone starts with an empty cache so tokens will
/// be computed again on first access in the clone.
///
/// Turn nodes in the tree are themselves `Arc`-wrapped, so explicit
/// `TokenizedText` clones are rare.
#[derive(Default)]
pub struct TokenizedText {
    text: Arc<str>,
    /// Lazily initialized. The `OnceLock` means tokens are computed at most
    /// once; concurrent readers block until the first writer finishes.
    token_ids: OnceLock<TokenBuffer>,
}

impl TokenizedText {
    // ── Constructors ─────────────────────────────────────────────────────

    /// Create a `TokenizedText` with both text and pre-computed token ids.
    pub fn new(text: impl Into<Arc<str>>, token_ids: TokenBuffer) -> Self {
        let lock = OnceLock::new();
        let _ = lock.set(token_ids);
        Self {
            text: text.into(),
            token_ids: lock,
        }
    }

    /// Create a `TokenizedText` with text only — tokens computed on first access.
    ///
    /// Useful in tests and metadata-only paths where token ids are not needed.
    pub fn plaintext(text: impl Into<Arc<str>>) -> Self {
        Self {
            text: text.into(),
            token_ids: OnceLock::new(),
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────

    /// Borrow the text content.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// True if the text is empty.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// True if token ids have already been computed or supplied.
    pub fn is_tokenized(&self) -> bool {
        self.token_ids.get().is_some()
    }

    /// Borrow the token ids if already available, without triggering
    /// tokenization.
    ///
    /// Returns an empty slice when tokens have not yet been computed.
    /// Call [`tokens`](Self::tokens) to trigger lazy tokenization.
    pub fn token_ids(&self) -> &[u32] {
        self.token_ids.get().map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// Borrow the token ids, **lazily tokenizing** on first call.
    ///
    /// After the first call the tokenizer is no longer accessed; subsequent
    /// calls return the cached slice immediately. If the text is empty,
    /// returns an empty slice without calling the tokenizer.
    pub fn tokens(&self, tokenizer: &tokenizers::Tokenizer) -> &[u32] {
        if self.text.is_empty() {
            return &[];
        }
        self.token_ids.get_or_init(|| {
            tokenizer
                .encode(self.text.as_ref(), false)
                .map(|enc| TokenBuffer::from(enc.get_ids()))
                .unwrap_or_default()
        })
    }

    /// Number of tokens currently stored (0 if not yet tokenized).
    pub fn token_count(&self) -> usize {
        self.token_ids.get().map(|v| v.len()).unwrap_or(0)
    }

    // ── Mutation ──────────────────────────────────────────────────────────

    /// Eagerly supply token ids without a tokenizer.
    ///
    /// If tokens have already been set (or computed via [`tokens`]), the new
    /// ids **replace** the stored ones. Use this path when the tokenizer has
    /// already been run externally (e.g., the conversation engine tokenizes
    /// text before submitting to the scheduler, then stores the result).
    pub fn set_tokens(&mut self, token_ids: TokenBuffer) {
        // Swap the OnceLock so the new ids win.
        let lock = OnceLock::new();
        let _ = lock.set(token_ids);
        self.token_ids = lock;
    }

    /// Builder: attach token ids and return `self` (for construction chains).
    pub fn with_tokens(mut self, token_ids: TokenBuffer) -> Self {
        self.set_tokens(token_ids);
        self
    }
}

// ── Trait implementations ────────────────────────────────────────────────────

impl Clone for TokenizedText {
    fn clone(&self) -> Self {
        // Copy computed tokens into the clone so they don't have to be
        // recomputed; if not yet computed the clone starts fresh.
        let lock = OnceLock::new();
        if let Some(ids) = self.token_ids.get() {
            let _ = lock.set(ids.clone());
        }
        Self {
            text: Arc::clone(&self.text),
            token_ids: lock,
        }
    }
}

impl std::fmt::Debug for TokenizedText {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TokenizedText")
            .field("text", &self.text)
            .field(
                "token_ids",
                &self
                    .token_ids
                    .get()
                    .map(|v| format!("[{}; {} tokens]", v.first().unwrap_or(&0), v.len()))
                    .unwrap_or_else(|| "<not tokenized>".into()),
            )
            .finish()
    }
}

// ── From conversions ─────────────────────────────────────────────────────────

impl From<&'static str> for TokenizedText {
    fn from(s: &'static str) -> Self {
        Self::plaintext(s)
    }
}

impl From<String> for TokenizedText {
    fn from(s: String) -> Self {
        Self::plaintext(s)
    }
}

impl From<Arc<str>> for TokenizedText {
    fn from(s: Arc<str>) -> Self {
        Self::plaintext(s)
    }
}
