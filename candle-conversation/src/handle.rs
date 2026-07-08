use crate::error::ConversationError;
use crate::token_buffer::TokenBuffer;
use crate::TurnStats;
use std::sync::Arc;

/// Handle to an in-flight inference turn. Returned by [`crate::Sequence::submit_turn`].
///
/// The caller can block for the full response, stream token-by-token, or poll
/// non-blocking. If the handle is dropped before `Done` is received, the
/// scheduler detects the closed channel and stops decode at the next step.
///
/// The view sequence backing this turn is owned and auto-finalized by the
/// scheduler.  The caller never sees the view's `SequenceId`.
pub struct TurnHandle {
    rx: crossbeam::channel::Receiver<TurnEvent>,
}

impl TurnHandle {
    pub(crate) fn new(rx: crossbeam::channel::Receiver<TurnEvent>) -> Self {
        Self { rx }
    }

    /// Block until the turn completes. Returns the full response.
    ///
    /// Token and PrefillProgress events are consumed silently.
    /// Does **not** consume the handle — pass it to `finish_turn` afterwards.
    pub fn wait(&self) -> crate::Result<TurnResponse> {
        loop {
            match self.rx.recv() {
                Ok(TurnEvent::Done(response)) => return Ok(response),
                Ok(TurnEvent::Error(e)) => return Err(e),
                Ok(_) => {}
                Err(_) => return Err(ConversationError::SchedulerGone),
            }
        }
    }

    /// Iterate over events as they arrive (blocking iterator).
    ///
    /// Yields `PrefillProgress`, `Token`, `AttentionStats`, and finally
    /// `Done` or `Error`. The iterator ends after `Done`/`Error` or if
    /// the scheduler drops the sender.
    ///
    /// Does **not** consume the handle — pass it to `finish_turn` afterwards.
    pub fn stream(&self) -> impl Iterator<Item = TurnEvent> + '_ {
        let rx = &self.rx;
        std::iter::from_fn(move || rx.recv().ok())
    }

    /// Non-blocking poll. Returns `None` if no event is ready yet.
    pub fn try_recv(&self) -> Option<TurnEvent> {
        self.rx.try_recv().ok()
    }
}

/// Events sent from the scheduler to the caller during a turn.
pub enum TurnEvent {
    /// The formatted text that was actually submitted for prefill.
    /// Includes user turn markup and the (clean) assistant start — no think
    /// block is baked in; under `/no_think` the model decodes its own empty
    /// `<think></think>`. This is the exact string tokenized and sent to the model.
    Prefill(String),

    /// Prefill progress (for visibility into long prefills).
    PrefillProgress {
        /// Tokens processed so far.
        tokens_done: usize,
        /// Total tokens to prefill.
        tokens_total: usize,
    },

    /// A raw token ID (streamed during generation).
    ///
    /// The caller is responsible for decoding tokens into text using a
    /// [`TokenDecoder`]. This allows the caller to accumulate tokens
    /// and re-decode the full buffer on each arrival, which correctly
    /// handles multi-byte sequences (emoji, flag sequences, CJK) that
    /// BPE byte-fallback tokenizers produce as individual byte tokens.
    Token(u32),

    /// Generation complete. Contains the full response.
    Done(TurnResponse),

    /// A projection event: emitted once at each mid-decode reprojection, when
    /// the scheduler rebuilds the view against fresh provenance scores. Carries
    /// the materialized-context composition that reprojection selected plus the
    /// decode throughput of the span that just completed — the GUI drops a
    /// timeline dot per event (docs/zend_ui_redesign.md §2.3).
    Projection(crate::projection::ProjectionEvent),

    /// Something went wrong.
    Error(ConversationError),

    /// A decode health check triggered and the sequence was aborted early.
    ///
    /// Contains a human-readable description of the degradation that was
    /// detected. Emitted just before the final [`TurnEvent::Done`] event.
    ///
    /// Only generated when the `decode-health` feature is enabled and
    /// [`DecodeHealthConfig::enabled`](crate::config::DecodeHealthConfig::enabled)
    /// is `true`.
    HealthWarning(String),
}

/// Per-seal payload attached to [`TurnResponse`] when the scheduler
/// completes a turn or section that wrote into the substrate.
///
/// `None` for paths that don't seal-and-write (RULER eval,
/// summarisation).  When `Some`, the substrate already holds the new
/// turn or section by the time the conversation receives `Done`; the
/// payload exists so the conversation can run its post-seal
/// follow-ups (cold-store persistence) without a second round-trip to
/// the scheduler.
pub struct SealResult {
    /// Total sealed-block count for the parent slot **after** the seal
    /// advance.
    pub block_count: usize,
    /// First block index of this turn or section in the parent's
    /// block table.
    pub block_from: usize,
    /// One-past-last block index of this turn or section.
    pub block_to: usize,
    /// Total tokens in this turn or section
    /// (`parent.chunks[block_from..block_to].iter().map(.token_count).sum()`).
    pub turn_token_count: usize,
    /// Chunk size in tokens (mirrors the scheduler's chunk_size).
    pub chunk_size: usize,
}

/// Complete response from a turn.
pub struct TurnResponse {
    /// The assistant's generated text.
    pub text: String,

    /// Token IDs generated.
    pub token_ids: TokenBuffer,

    /// Generation statistics.
    pub stats: TurnStats,

    /// Seal payload — present when the scheduler did a substrate
    /// write for this turn / section.  `None` for raw paths
    /// (RULER, summarisation).
    pub seal: Option<SealResult>,
}

// ────────────────────────────────────────────────────────────────────────────
// TokenDecoder — public utility for callers to decode token IDs
// ────────────────────────────────────────────────────────────────────────────

/// Decodes token IDs into text strings.
///
/// Wraps the tokenizer and abstracts away BPE details. Callers collect
/// tokens into a `Vec<u32>` and call [`decode`](TokenDecoder::decode) to get
/// the text. This is cheap (~microseconds for thousands of tokens).
///
/// # Example — line-buffered streaming
///
/// ```ignore
/// let decoder = engine.token_decoder();
/// let mut line_tokens: Vec<u32> = Vec::new();
/// for event in handle.stream() {
///     match event {
///         TurnEvent::Token(id) => {
///             line_tokens.push(id);
///             let text = decoder.decode(&line_tokens);
///             // rewrite current line with `text`
///         }
///         _ => {}
///     }
/// }
/// ```
#[derive(Clone)]
pub struct TokenDecoder {
    tokenizer: Arc<tokenizers::Tokenizer>,
}

impl TokenDecoder {
    /// Create a new decoder from a shared tokenizer.
    pub fn new(tokenizer: Arc<tokenizers::Tokenizer>) -> Self {
        Self { tokenizer }
    }

    /// Decode token IDs into text, stripping special tokens.
    pub fn decode(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens, true).unwrap_or_default()
    }

    /// Decode token IDs into text, including special tokens verbatim.
    pub fn decode_with_special(&self, tokens: &[u32]) -> String {
        self.tokenizer.decode(tokens, false).unwrap_or_default()
    }
}
