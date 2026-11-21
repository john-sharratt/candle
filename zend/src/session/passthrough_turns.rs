//! Runs one OpenAI-passthrough call: the client's own context is decoded as
//! sent, and its reply is sealed.
//!
//! A call reuses the live conversation when it extends what that conversation
//! holds. Otherwise the conversation is resumed or opened from the substrate,
//! and one that a failed call left in an unknown state is dropped first. Once
//! the model opens a tool call it is held to the client's declared tools and
//! their arguments, and the reasoning block is steered so a reply cannot end
//! with its reasoning still open. A client that goes away before a token is due
//! stops the decode, and the turn is never sealed.

use super::*;

/// One OpenAI-passthrough call, holding its conversation's lock for the whole
/// turn — see [`crate::passthrough`]. Reuses the live conversation when the call
/// extends what it holds; otherwise resumes or opens one. A conversation left
/// in an unknown state by a failed call is dropped, so the next call reopens it
/// from the substrate.
pub(super) async fn run_passthrough(
    state: Arc<InferenceState>,
    key: String,
    mut conv: OwnedMutexGuard<Option<LiveConv>>,
    transcript: Transcript,
    max_tokens: Option<usize>,
    tx: tokio::sync::mpsc::Sender<anyhow::Result<StreamItem>>,
) {
    let reusable = conv.as_ref().is_some_and(|live| {
        live.system == transcript.system && passthrough::extends(&live.history, &transcript.history)
    });
    if !reusable {
        // Free the stale conversation's slot before opening another.
        *conv = None;
        let opened = {
            let engine = &state.engine;
            passthrough::open(engine, state.refresh_config.clone(), &transcript)
        };
        match opened {
            Ok(live) => *conv = Some(live),
            Err(e) => {
                tracing::error!(key, "passthrough: {e:#}");
                let _ = tx.send(Err(e)).await;
                return;
            }
        }
    }
    let healthy = match conv.as_mut() {
        Some(live) => passthrough_turn(&state, &key, live, transcript, max_tokens, &tx).await,
        None => false,
    };
    if !healthy {
        *conv = None;
    }
    state.passthrough.touch(&key);
}

/// The answer room a passthrough reply gets past its think block — the
/// response-length dial's widest rung. A client's reply can be a whole file
/// written through a tool call, and the client's own `max_tokens` still caps
/// the turn; this only places the failsafe that ends a runaway.
pub(super) const PASSTHROUGH_RESPONSE_TOKENS: i32 = 7168;

/// Prefill the history `live` does not hold, decode the reply to the new user
/// half, seal it and commit the redo log. Returns whether `live` still matches
/// what the substrate holds.
pub(super) async fn passthrough_turn(
    state: &InferenceState,
    key: &str,
    live: &mut LiveConv,
    transcript: Transcript,
    max_tokens: Option<usize>,
    tx: &tokio::sync::mpsc::Sender<anyhow::Result<StreamItem>>,
) -> bool {
    let held = live.history.len();
    tracing::info!(
        key,
        timeline = live.seq.timeline_id().raw(),
        held,
        sent = transcript.history.len(),
        "passthrough turn",
    );
    for exchange in &transcript.history[held..] {
        if let Err(e) = live
            .seq
            .insert_turn(exchange.user.clone(), &exchange.assistant)
        {
            tracing::error!(key, "passthrough: prefilling history failed: {e}");
            let _ = tx.send(Err(anyhow::anyhow!("{e}"))).await;
            return false;
        }
        live.history.push(exchange.clone());
    }

    // Once the model opens a call it is held to the client's tools and their
    // declared arguments — see `LiveConv::tool_stencil`. The `<think>` block is
    // steered atop that, exactly as on a dialogue turn: its tree takes an EOS
    // sampled inside the block as the block's close, so a reply cannot end
    // with its reasoning still open. Measured on a Cline turn without it: the
    // model opened a block, planned in it, and stopped — the client received
    // only reasoning, no answer and no call, and ended the task.
    let triggers = match &state.think_steering {
        Some(ts) => ts.registry_for(&live.tool_stencil, ThinkMode::Balanced),
        None => Arc::clone(&live.tool_stencil),
    };
    // The balanced dial's budget, as a dialogue turn gets it — not the
    // preset's fallback, which cut a Cline plan mid-sentence at 301 thinking
    // tokens and force-ends a reply at 1000 in all.
    let mut sampling = live.seq.default_sampling();
    think_budget::steer(
        &mut sampling,
        ThinkMode::Balanced,
        PASSTHROUGH_RESPONSE_TOKENS,
        &state.think_closer_phrase,
    );
    let options = candle_conversation::TurnOptions {
        max_tokens,
        sampling: Some(sampling),
        triggers,
        ..Default::default()
    };
    let handle = match live
        .seq
        .submit_turn_with_options(transcript.message.clone(), options)
    {
        Ok(h) => h,
        Err(e) => {
            tracing::error!(key, "passthrough: submit failed: {e}");
            let _ = tx.send(Err(anyhow::anyhow!("{e}"))).await;
            return false;
        }
    };

    // Deltas are cut at byte offsets into the text, decoded token by token, and
    // a fragment that still holds U+FFFD waits for the rest of its byte-fallback
    // sequence.
    let mut text_stream = state.decoder.stream();
    let mut emitted_len = 0usize;
    let mut done = None;
    let mut deletions = state.deletions.watch();
    let timeline = live.seq.timeline_id();
    {
        let mut stream = std::pin::pin!(handle.stream_async());
        loop {
            let event = tokio::select! {
                event = stream.next() => event,
                Ok(()) = deletions.changed() => {
                    if state.engine.is_timeline_tombstoned(timeline) {
                        // Deleted under the reply: dropping the handle withdraws it
                        // from the engine, and the turn never seals.
                        tracing::info!(key, "passthrough: conversation deleted mid-reply");
                        return false;
                    }
                    continue;
                }
                // The client went away before a token was due: a reply still
                // queued for its prefill would otherwise run it, and seal, for
                // nobody — the next send is what used to notice.
                () = tx.closed() => {
                    tracing::info!(key, "passthrough: client disconnected before the reply");
                    return false;
                }
            };
            let Some(event) = event else {
                break;
            };
            match event {
                TurnEvent::Token(id) => {
                    text_stream.push(id);
                    let text = text_stream.text();
                    if text.len() > emitted_len && text.is_char_boundary(emitted_len) {
                        let part = &text[emitted_len..];
                        if !part.contains('\u{FFFD}') {
                            if tx
                                .send(Ok(StreamItem::Token(part.to_string())))
                                .await
                                .is_err()
                            {
                                // The client went away: dropping the handle stops the
                                // decode, and the turn never seals.
                                tracing::info!(key, "passthrough: client disconnected mid-stream");
                                return false;
                            }
                            emitted_len = text.len();
                        }
                    }
                }
                TurnEvent::Done(resp) => done = Some(resp),
                TurnEvent::Error(e) => {
                    tracing::error!(key, "passthrough: scheduler error: {e}");
                    let _ = tx.send(Err(anyhow::anyhow!("{e}"))).await;
                }
                _ => {}
            }
        }
    }
    // `handle` is free here: the stream that borrowed it was scoped to the block
    // above, so `finish_turn` below can take it.
    let Some(resp) = done else {
        return false;
    };
    if resp.text.len() > emitted_len && resp.text.is_char_boundary(emitted_len) {
        let tail = &resp.text[emitted_len..];
        if !tail.is_empty() {
            let _ = tx.send(Ok(StreamItem::Token(tail.to_string()))).await;
        }
    }
    let _ = tx
        .send(Ok(StreamItem::TurnEnd {
            usage: Usage::for_turn(resp.stats.context_tokens, resp.stats.tokens_generated),
            finish: resp.stats.finish,
        }))
        .await;
    if let Err(e) = live.seq.finish_turn(handle, &resp) {
        tracing::warn!(key, "passthrough: finish_turn: {e}");
    }
    // See the dialogue path's commit.
    let conversation = state.engine.conversation();
    if let Err(e) = conversation.commit_persistence() {
        tracing::warn!(key, "passthrough: persistence commit: {e}");
    }
    live.history.push(Exchange {
        user: transcript.message,
        assistant: resp.text,
    });
    true
}

impl ZendSession {
    /// Run an OpenAI-passthrough call — the client's own context, as-is — and
    /// return a stream of its reply tokens. See [`crate::passthrough`].
    ///
    /// Waits for the engine, then for the conversation's own lock: a call on a
    /// conversation another call is running queues behind it, while calls on
    /// other conversations go ahead.
    pub async fn submit_passthrough(
        &self,
        messages: Vec<ChatMessage>,
        tools: Vec<Value>,
        max_tokens: Option<usize>,
    ) -> Pin<Box<dyn Stream<Item = anyhow::Result<StreamItem>> + Send + 'static>> {
        let (tx, rx) = tokio::sync::mpsc::channel::<anyhow::Result<StreamItem>>(64);
        let transcript = match Transcript::from_messages(&messages, &tools) {
            Ok(t) => t,
            Err(e) => {
                let _ = tx.send(Err(anyhow::anyhow!("passthrough: {e}"))).await;
                return Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx));
            }
        };
        let inference = Arc::clone(&self.inference);
        let mut ready_rx = self.ready_tx.subscribe();
        tokio::spawn(async move {
            loop {
                // Loaded, or the load is over without an engine (a stop): the
                // read below tells the two apart.
                let loaded = inference.load().is_some();
                if loaded || *ready_rx.borrow_and_update() {
                    break;
                }
                if ready_rx.changed().await.is_err() {
                    return;
                }
            }
            let state = { inference.load_full() };
            let Some(state) = state else {
                return;
            };
            state.passthrough.ensure_sweeper();
            let key = transcript.key();
            // The conversation stays this member's until the reply is sealed.
            let claim = turn_claims::hold(&key);
            let conv = state.passthrough.conversation(&key).lock_owned().await;
            run_passthrough(state, key.clone(), conv, transcript, max_tokens, tx).await;
            turn_claims::release(claim, &key).await;
        });
        Box::pin(tokio_stream::wrappers::ReceiverStream::new(rx))
    }
}
