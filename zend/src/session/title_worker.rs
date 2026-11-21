//! Generates and rewrites a conversation's sidebar title on one background task,
//! never on the request path.
//!
//! Jobs queue behind a bound, and a backlog beyond it is dropped, since labels
//! are best-effort and the title then stays as it was. Each title runs on a
//! conversation of its own, retired afterwards. The soft switch cannot stop a
//! title from reasoning on this model, so the prompt and the sampler are both
//! steered out of the reasoning block. A re-title that keeps the same title
//! writes nothing, and the worker stops draining jobs once shutdown begins.

use super::*;

/// Bound on the titler queue. A backlog beyond this many un-started titles
/// means the worker can't keep up; further jobs are dropped (labels are
/// best-effort), which is preferable to unbounded growth under a burst.
pub(super) const TITLER_QUEUE_DEPTH: usize = 256;

/// Work item for the titler worker thread.
pub(super) enum TitleJob {
    /// Generate a sidebar title for `timeline` from `message` and write it.
    Title {
        timeline: TimelineId,
        message: String,
    },
    /// Title `timeline` again from its latest finished exchange, which waits
    /// in [`InferenceState::retitles`] — taken when this job starts, so a turn
    /// that finished since the job was queued is the one it reads.
    Retitle { timeline: TimelineId },
    /// Stop draining and exit (sent during shutdown to wake an idle worker).
    Shutdown,
}

// Room for the title plus the empty `<think></think>` Qwen3 still decodes under
// /no_think (no longer prefilled for free) — the block is stripped afterwards.
pub(super) const TITLER_MAX_TOKENS: usize = 24;

/// Queue a re-title of `timeline` from the request turn that just finished:
/// `request` asked, `answer` is the reply that ended it. Coalesced — a
/// conversation already waiting for one has it read this exchange instead —
/// and non-blocking: a full queue or a shutdown drops it, and the title stays
/// as it was. A turn resumed from a sealed call has no request text of its
/// own; its answer alone carries the topic.
pub(super) fn request_retitle(
    state: &InferenceState,
    timeline: TimelineId,
    request: &str,
    answer: &str,
) {
    if answer.trim().is_empty() || state.shutting_down.load(Ordering::Relaxed) {
        return;
    }
    let exchange = Finished {
        request: request.to_string(),
        answer: answer.to_string(),
    };
    if !state.retitles.offer(timeline, exchange) {
        return;
    }
    if let Err(e) = state.titler_tx.try_send(TitleJob::Retitle { timeline }) {
        // Nothing will take it: let the next finished turn queue afresh.
        state.retitles.take(timeline);
        tracing::debug!("titler queue full or closed — skipping re-title: {e}");
    }
}

/// The titler task. Drains title jobs one at a time off `rx`, each on a
/// conversation of its own ([`Titler`]). Each title-gen overlaps the main
/// decode (the scheduler batches both sequences), but the task never blocks
/// the request path and never piles up work per submit. Between jobs it
/// checks the shutdown flag, so once shutdown begins it stops draining queued
/// jobs immediately.
pub(super) async fn titler_task(
    state: Arc<InferenceState>,
    titler: Titler,
    mut rx: mpsc::Receiver<TitleJob>,
) {
    while let Some(job) = rx.recv().await {
        if state.shutting_down.load(Ordering::Relaxed) {
            break;
        }
        match job {
            TitleJob::Shutdown => break,
            TitleJob::Title { timeline, message } => {
                let prompt = head_tail_truncate(
                    &message,
                    &state.tokenizer,
                    FIRST_HEAD_TOKENS,
                    FIRST_TAIL_TOKENS,
                );
                generate_one_title(&state, &titler, timeline, &prompt).await;
            }
            TitleJob::Retitle { timeline } => {
                // Read the title as it stands when the re-title starts — the
                // first title, queued ahead of it, has been written by now.
                let Some(exchange) = state.retitles.take(timeline) else {
                    continue;
                };
                let current = state.engine.conversation_label_of(timeline);
                let cut = |text: &str, head: usize, tail: usize| {
                    head_tail_truncate(text, &state.tokenizer, head, tail)
                };
                let prompt = retitle_prompt(current.as_deref(), &exchange, &cut);
                generate_one_title(&state, &titler, timeline, &prompt).await;
            }
        }
    }
}

/// Decode a title for `prompt` — a first message, or a re-title's current
/// title and latest exchange ([`titles`]), already bounded — on a conversation
/// opened for it, and write the result as the main conversation's sidebar
/// label. Errors are logged and dropped — the worst case is a stale sidebar
/// label, never a failed response. The conversation is retired however the
/// title ends ([`Titler`]).
pub(super) async fn generate_one_title(
    state: &Arc<InferenceState>,
    titler: &Titler,
    timeline: TimelineId,
    prompt: &str,
) {
    use candle_conversation::TurnOptions;

    // **The titler must never reason, and the soft switch cannot stop it.**
    //
    // `NO_THINK_SELECTOR` below emits the scheduler's `no_think_current` glue,
    // which renders `Dialect::no_think` — and on the Qwen3.5/3.8 family that is
    // the empty string, because those models suppress by *prefilling* a closed
    // block rather than by a marker (`Dialect::thinking_suppression`). So the
    // glue was a zero-token segment and the `/no_think` in the system prompt was
    // prose: the titler had two suppression mechanisms and, on the model this
    // daemon actually runs, neither did anything.
    //
    // What that cost: the model reasoned into a 24-token budget and the label
    // got whatever survived. A block that never closed stripped to nothing
    // (`strip_think_blocks` drops an unterminated block to end) and the title
    // was skipped; a block that closed just in time left a fragment, which is
    // how "Cas", "The KV", "A" and "P" reached the sidebar — and, since a
    // one-character `String` is not empty, they were written without a warning.
    // Twelve identical prompts produced twelve different lengths, because what
    // varied was the reasoning, not the budget.
    //
    // The trigger below is the layer that cannot be ignored: it binds `<think>`
    // to `ThinkMode::Off`'s tree, whose whole body is the static run
    // `"\n\n</think>"` with no free-decode span, so the block closes on the
    // token after it opens and the budget goes to the title. `apply_think_mode`
    // programs the sampler to match. This is the same three-layer arrangement
    // the ingest summariser settled on after the same defect
    // (`Conversation::submit_ingest_summary`) and the tree compressor's probe
    // after it (`SubmitSummaryProbe`) — the titler was the one left behind.
    let triggers = {
        let en = &state.engine;
        match en.compile_think_steering() {
            // No `<think>`/`</think>` token in this vocabulary, so there is no
            // block to steer and nothing to bind to; `compile_think_steering`
            // has already warned, and the sampler side below still runs.
            Ok(Some(ts)) => ts.registry_for(&TriggerRegistry::new(), ThinkMode::Off),
            Ok(None) => Arc::new(TriggerRegistry::new()),
            Err(e) => {
                tracing::warn!("titler think steering unavailable: {e}");
                Arc::new(TriggerRegistry::new())
            }
        }
    };

    // A conversation of its own, so the title reads the system prompt and the
    // text it names, and nothing else. Opening it is a scheduler round trip.
    let opening = (Arc::clone(state), titler.clone());
    let mut job = match tokio::task::spawn_blocking(move || opening.1.open(&opening.0)).await {
        Ok(Ok(job)) => job,
        Ok(Err(e)) => {
            tracing::warn!("titler conversation did not open: {e}");
            return;
        }
        Err(e) => {
            tracing::warn!("titler conversation open panicked: {e}");
            return;
        }
    };
    let titler = &mut job.sequence;
    // The titler generates a short label — never reason. Suppress thinking
    // explicitly (the model thinks by default).
    let mut titler_sel = candle_conversation::SelectionState::new();
    titler_sel.set_optional(
        candle_conversation::NO_THINK_SELECTOR,
        candle_conversation::OptionalState::Present,
    );
    // The sampler half of the same suppression: `Off` at this budget collapses
    // the segment-close budget to a forced empty block, so the model is steered
    // out of the block it opened as well as being structurally closed out of it.
    let mut titler_sampling = titler.default_sampling();
    titler_sampling.apply_think_mode(ThinkMode::Off, &state.tokenizer, TITLER_MAX_TOKENS);
    let opts = TurnOptions {
        max_tokens: Some(TITLER_MAX_TOKENS),
        selection: titler_sel,
        sampling: Some(titler_sampling),
        triggers,
        ..Default::default()
    };
    let handle = match titler.submit_turn_with_options(prompt, opts) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!("titler submit failed: {e}");
            return;
        }
    };
    let resp = match handle.wait_async().await {
        Ok(r) => r,
        Err(e) => {
            // No response: scheduler error or shutdown mid-decode. The
            // conversation is retired with `job` either way.
            tracing::warn!("titler ended without a response: {e}");
            return;
        }
    };
    let title = clean_title(&resp.text);
    drop(job);

    if title.is_empty() {
        tracing::warn!("titler produced an empty title — skipping label write");
        return;
    }
    // Don't touch the engine once shutdown has begun — it's being torn down.
    if state.shutting_down.load(Ordering::Relaxed) {
        return;
    }
    // A re-title that kept the title writes nothing: the label record would
    // replicate a value every member already holds.
    if state.engine.conversation_label_of(timeline).as_deref() == Some(title.as_str()) {
        return;
    }

    // Write the label via the engine's substrate-shared handle.
    let result = state.engine.set_conversation_label(timeline, &title);
    match result {
        Ok(()) => tracing::info!("conversation labelled: \"{title}\""),
        Err(e) => tracing::warn!("set_conversation_label failed: {e}"),
    }
}
