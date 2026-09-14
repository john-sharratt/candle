use crate::error::ConversationError;
use crate::token_buffer::TokenBuffer;
use crate::TurnStats;
use flume::{Receiver, RecvTimeoutError};
use futures_core::Stream;
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
    rx: Receiver<TurnEvent>,
}

impl TurnHandle {
    pub(crate) fn new(rx: Receiver<TurnEvent>) -> Self {
        Self { rx }
    }

    /// Classify one event on the way to completion: `Some` settles the wait —
    /// the turn's response or its error — and `None` is an intermediate event
    /// the waiters consume silently. Every wait (`wait`, `wait_cancellable`,
    /// `wait_async`) reads events through this, so a change to what ends a
    /// turn reaches the blocking and async callers together or not at all.
    fn settle(event: TurnEvent) -> Option<crate::Result<TurnResponse>> {
        match event {
            TurnEvent::Done(response) => Some(Ok(response)),
            TurnEvent::Error(e) => Some(Err(e)),
            _ => None,
        }
    }

    /// Block until the turn completes. Returns the full response.
    ///
    /// Token and PrefillProgress events are consumed silently.
    /// Does **not** consume the handle — pass it to `finish_turn` afterwards.
    pub fn wait(&self) -> crate::Result<TurnResponse> {
        loop {
            match self.rx.recv() {
                Ok(event) => {
                    if let Some(settled) = Self::settle(event) {
                        return settled;
                    }
                }
                Err(_) => return Err(ConversationError::SchedulerGone),
            }
        }
    }

    /// Block until the turn completes, but abandon the wait if a graceful
    /// shutdown latches [`crate::ingest_cancelled`]. Used by the ingest
    /// decode-waits ([`crate::Sequence::ingest_scope_roundtrip_indices`]) so a
    /// Ctrl-C mid-ingest doesn't have to wait out the in-flight summary decode.
    ///
    /// On cancel this returns [`ConversationError::IngestCancelled`] WITHOUT
    /// consuming the handle; the caller drops it, and the scheduler — seeing the
    /// closed channel — stops decode at its next step and auto-finalizes the view
    /// slot (see the type-level docs). The poll cadence only bounds how long
    /// after the flag flips the wait returns; it does not busy-spin, since
    /// `recv_timeout` still delivers `Done`/`Token`/etc. events the moment they
    /// arrive.
    pub fn wait_cancellable(&self) -> crate::Result<TurnResponse> {
        let poll = std::time::Duration::from_millis(100);
        loop {
            match self.rx.recv_timeout(poll) {
                Ok(event) => {
                    if let Some(settled) = Self::settle(event) {
                        return settled;
                    }
                }
                Err(RecvTimeoutError::Timeout) => {
                    if crate::ingest_cancelled() {
                        return Err(ConversationError::IngestCancelled);
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(ConversationError::SchedulerGone)
                }
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
        self.rx.iter()
    }

    /// Non-blocking poll. Returns `None` if no event is ready yet.
    pub fn try_recv(&self) -> Option<TurnEvent> {
        self.rx.try_recv().ok()
    }

    /// Await the turn's completion without holding a thread — [`Self::wait`]'s
    /// async counterpart. Identical event handling, identical result: the same
    /// channel serves both, so which one a caller uses is a property of the
    /// caller, not of the turn.
    ///
    /// Cancelling one turn is dropping the [`TurnHandle`]: a task that owns
    /// the handle and is aborted drops it, the scheduler sees the closed
    /// channel, and decode stops at its next step (see the type-level docs).
    /// Dropping only this future while the handle lives abandons nothing —
    /// exactly as returning early from a blocking `wait` would not. The
    /// process-wide ingest shutdown latch is a different cancellation, and
    /// [`Self::wait_cancellable_async`] is the wait that honours it.
    pub async fn wait_async(&self) -> crate::Result<TurnResponse> {
        loop {
            match self.rx.recv_async().await {
                Ok(event) => {
                    if let Some(settled) = Self::settle(event) {
                        return settled;
                    }
                }
                Err(_) => return Err(ConversationError::SchedulerGone),
            }
        }
    }

    /// The event stream as an async [`Stream`] — [`Self::stream`]'s async
    /// counterpart. Yields the same events in the same order; the stream ends
    /// when the scheduler drops its sender, which it does after `Done`/`Error`
    /// when the turn's view is finalized.
    ///
    /// Does **not** consume the handle — pass it to `finish_turn` afterwards.
    pub fn stream_async(&self) -> impl Stream<Item = TurnEvent> + '_ {
        self.rx.stream()
    }

    /// Await this turn's next event — one step of [`Self::stream_async`], for a
    /// caller holding several handles who wants whichever of them speaks first
    /// (`futures::future::select_all` over these). `None` when the scheduler
    /// has dropped its sender and the turn has nothing more to say.
    pub async fn next_event_async(&self) -> Option<TurnEvent> {
        self.rx.recv_async().await.ok()
    }

    /// [`Self::wait_cancellable`]'s async counterpart: await the turn, but
    /// resolve with [`ConversationError::IngestCancelled`] the moment a
    /// graceful shutdown latches [`crate::ingest_cancelled`] — woken by the
    /// latch itself rather than found on a poll cadence, because the executors
    /// that drive the ingest waits (`futures::executor::block_on` on a loader
    /// or worker thread) have no timer to poll it on.
    ///
    /// On cancel the handle is NOT consumed; the caller drops it, and the
    /// scheduler — seeing the closed channel — stops decode at its next step.
    pub async fn wait_cancellable_async(&self) -> crate::Result<TurnResponse> {
        use std::future::Future;
        use std::task::Poll;
        loop {
            let mut recv = std::pin::pin!(self.rx.recv_async());
            let mut cancel = std::pin::pin!(crate::cancel::ingest_cancel_wait());
            // The event wins a tie: a decode that finished as the shutdown
            // arrived is a real response, and taking it leaves nothing in
            // flight to unwind.
            let raced = std::future::poll_fn(|cx| {
                if let Poll::Ready(received) = recv.as_mut().poll(cx) {
                    return Poll::Ready(Some(received));
                }
                if cancel.as_mut().poll(cx).is_ready() {
                    return Poll::Ready(None);
                }
                Poll::Pending
            })
            .await;
            match raced {
                None => return Err(ConversationError::IngestCancelled),
                Some(Err(_)) => return Err(ConversationError::SchedulerGone),
                Some(Ok(event)) => {
                    if let Some(settled) = Self::settle(event) {
                        return settled;
                    }
                }
            }
        }
    }
}

/// Events sent from the scheduler to the caller during a turn.
pub enum TurnEvent {
    /// The formatted text that was actually submitted for prefill.
    /// Includes user turn markup and the assistant start. On a suppressed turn
    /// it also carries whichever half of `Dialect::thinking_suppression` that
    /// family uses — the `/no_think` switch in the user turn, or an already-closed
    /// `<think></think>` block after the assistant header. This is the exact
    /// string tokenized and sent to the model.
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
    /// The substrate `TurnIndex` this seal recorded (`SealAction::Turn`
    /// only; `None` for section seals).  Callers keying per-turn persists
    /// (staged projection events) MUST use this rather than
    /// `turn_count - 1`: the async summariser appends its node turns to
    /// the same timeline, so counting races.
    pub turn_index: Option<u32>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use candle_transformers::models::batched_inference::SequenceStats;
    use futures::executor::block_on;
    use futures::StreamExt;

    fn response(text: &str) -> TurnResponse {
        TurnResponse {
            text: text.to_string(),
            token_ids: TokenBuffer::new(),
            stats: TurnStats {
                prefill_ms: 0.0,
                decode_ms: 0.0,
                total_ms: 0.0,
                tokens_generated: 0,
                tokens_per_second: 0.0,
                prefill_token_count: 0,
                context_tokens: 0,
                sequence: SequenceStats::default(),
            },
            seal: None,
        }
    }

    /// A handle whose channel already holds `events` with the sender dropped —
    /// the state a finished turn leaves for a late reader. Every test that
    /// does not need a live sender builds its handle here, so the event shape
    /// is written once.
    fn handle_with(events: Vec<TurnEvent>) -> TurnHandle {
        let (tx, rx) = flume::unbounded();
        for event in events {
            tx.send(event).unwrap();
        }
        TurnHandle::new(rx)
    }

    /// The events one completed turn produces, in order.
    fn one_turn() -> Vec<TurnEvent> {
        vec![
            TurnEvent::Prefill("prompt".into()),
            TurnEvent::PrefillProgress {
                tokens_done: 1,
                tokens_total: 2,
            },
            TurnEvent::Token(7),
            TurnEvent::Done(response("the answer")),
        ]
    }

    /// §5 of `docs/async_wave_submission.md`: a turn awaited with `wait_async`
    /// on a current-thread executor completes with the same response a
    /// blocking `wait` gives — same event handling, same consumption of the
    /// intermediate events.
    #[test]
    fn wait_async_returns_what_a_blocking_wait_returns() {
        let blocking = handle_with(one_turn()).wait().unwrap();
        let awaited = block_on(handle_with(one_turn()).wait_async()).unwrap();
        assert_eq!(blocking.text, "the answer");
        assert_eq!(awaited.text, blocking.text);
    }

    /// The error paths agree too: an `Error` event surfaces as `Err` from
    /// both, and a scheduler that is gone (sender dropped with no `Done`)
    /// reads as `SchedulerGone` from both.
    #[test]
    fn wait_async_agrees_with_wait_on_the_error_paths() {
        let failed = |handle_err: crate::Result<TurnResponse>| match handle_err {
            Err(ConversationError::Channel(msg)) => msg,
            Err(e) => panic!("expected the turn's own error, got {e:?}"),
            Ok(_) => panic!("expected the turn's own error, got a response"),
        };
        let boom = || vec![TurnEvent::Error(ConversationError::Channel("boom".into()))];
        let blocking = failed(handle_with(boom()).wait());
        let awaited = failed(block_on(handle_with(boom()).wait_async()));
        assert_eq!(blocking, "boom");
        assert_eq!(awaited, blocking);

        assert!(matches!(
            handle_with(Vec::new()).wait(),
            Err(ConversationError::SchedulerGone)
        ));
        assert!(matches!(
            block_on(handle_with(Vec::new()).wait_async()),
            Err(ConversationError::SchedulerGone)
        ));
    }

    /// §5: a blocking and an async receiver on the same kind of channel both
    /// see every event, in order.
    #[test]
    fn blocking_and_async_streams_see_every_event_in_order() {
        let ids: Vec<u32> = (0..100).collect();
        let tokens = || ids.iter().map(|&id| TurnEvent::Token(id)).collect();

        let handle = handle_with(tokens());
        let got_blocking: Vec<u32> = handle
            .stream()
            .filter_map(|e| match e {
                TurnEvent::Token(id) => Some(id),
                _ => None,
            })
            .collect();

        let handle = handle_with(tokens());
        let got_async: Vec<u32> = block_on(
            handle
                .stream_async()
                .filter_map(|e| async move {
                    match e {
                        TurnEvent::Token(id) => Some(id),
                        _ => None,
                    }
                })
                .collect(),
        );

        assert_eq!(got_blocking, ids);
        assert_eq!(got_async, ids);
    }

    /// The cancellation signal is the HANDLE's drop, not the future's: the
    /// scheduler stops a decode when its event send fails. Dropping an
    /// unfinished `wait_async` future while the handle lives abandons nothing;
    /// dropping the handle is what closes the channel the scheduler watches.
    /// (`send_turn_with_options_async`'s future owns its handle, so aborting a
    /// task that is awaiting it drops both — the drop IS the cancel.)
    #[test]
    fn dropping_the_handle_not_the_future_closes_the_channel() {
        let (tx, rx) = flume::unbounded();
        let handle = TurnHandle::new(rx);
        {
            let fut = handle.wait_async();
            drop(fut);
        }
        tx.send(TurnEvent::Token(1))
            .expect("the handle is alive, so the scheduler's send still lands");
        drop(handle);
        assert!(
            tx.send(TurnEvent::Token(2)).is_err(),
            "with the handle gone the send must fail — that failure is the \
             scheduler's stop signal"
        );
    }
}
