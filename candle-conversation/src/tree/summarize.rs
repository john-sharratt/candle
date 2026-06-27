//! Internal summarization types: [`SummarizationReason`],
//! [`SummarizationTurnEntry`], [`SummarizationSnapshot`], and
//! [`SummarizationTask`].
//!
//! All types are `pub(crate)` — not exposed in the public API.
//!
//! # Design
//!
//! `build_snapshot()` produces a [`SummarizationSnapshot`]. `run_summarize()`
//! (on `ConversationTree`) immediately calls [`SummarizationTask::launch()`]
//! to start inference, then pushes the resulting task handle onto the tree's
//! `pending_tasks` queue. `Sequence` drains that queue after each turn
//! and spin-polls until all tasks resolve.
//!
//! The inference backend (`scheduler_tx` + `tokenizer`) must be injected onto
//! the tree via `ConversationTree::set_inference_backend()` before any turns
//! are submitted. In tests and forks where no backend is set, `run_summarize()`
//! logs at `debug` and skips.
//!
//! See [`CognitiveTask`](super::task::CognitiveTask) for the consumer trait.

use std::ops::RangeInclusive;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::time::Duration;

use crossbeam::channel::{Receiver, TryRecvError};

use crate::sequence_handle::SequenceId;
use crate::stencil::TriggerRegistry;
use crate::token_buffer::TokenBuffer;

use super::config::ConversationTreeConfig;
use super::node::{ConversationNode, ConversationSegment};
use super::patch::TreePatch;
use super::task::{CognitiveTask, TaskKind, TaskPoll};
use super::types::{NodeId, SegmentId, TurnId, TurnType};

// ────────────────────────────────────────────────────────────────────────────
// SummarizationReason
// ────────────────────────────────────────────────────────────────────────────

/// Reason that a summarization trigger fired.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) enum SummarizationReason {
    /// N turn-pairs have elapsed since the last summarization.
    TurnCountReached { count: u32 },
    /// The UTC calendar day changed between the previous turn and this one.
    DayBoundary { previous_day: i32, new_day: i32 },
    /// N top-level segment nodes have accumulated since the last higher-level
    /// summarization, triggering a recursive segment-of-segments compression.
    SegmentCountReached { count: u32 },
}

// ────────────────────────────────────────────────────────────────────────────
// SummarizationTurnEntry
// ────────────────────────────────────────────────────────────────────────────

/// A single turn's contribution to a summarization snapshot.
///
/// Each entry is extracted from a [`ConversationTurnInner`](super::node::ConversationTurnInner)
/// at snapshot time. TODO: these entries will be assembled into the
/// summarization inference prompt that the async worker submits to the model.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct SummarizationTurnEntry {
    pub node_id: NodeId,
    pub turn_type: TurnType,
    /// Formatted `"[T-3.47]"` or `""` if markers are disabled.
    pub temporal_marker_text: String,
    pub user_text: String,
    pub assistant_text: String,
}

// ────────────────────────────────────────────────────────────────────────────
// SummarizationSegmentEntry
// ────────────────────────────────────────────────────────────────────────────

/// A single segment's contribution to a higher-level summarization snapshot.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct SummarizationSegmentEntry {
    pub node_id: NodeId,
    /// The compressed text already stored in the segment.
    pub summary_text: String,
}

// ────────────────────────────────────────────────────────────────────────────
// SummarizationContent
// ────────────────────────────────────────────────────────────────────────────

/// The payload of a [`SummarizationSnapshot`]: either raw turns (level-1
/// summarization) or previously-created segment summaries (level 2+).
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) enum SummarizationContent {
    /// Summarising a window of raw user↔assistant turns.
    Turns(Vec<SummarizationTurnEntry>),
    /// Summarising a collection of existing segment summaries (recursive).
    Segments(Vec<SummarizationSegmentEntry>),
}

// ────────────────────────────────────────────────────────────────────────────
// SummarizationSnapshot
// ────────────────────────────────────────────────────────────────────────────

/// Complete snapshot of the state at the moment a summarization trigger fired.
///
/// Produced by `build_snapshot()` or `build_segment_snapshot()` inside
/// `ConversationTree` and immediately consumed by `run_summarize()`, which
/// calls [`SummarizationTask::launch()`] and pushes the resulting handle
/// onto the tree's `pending_tasks`.
///
/// # Snapshot → Segment mapping
///
/// When the task completes, it creates a
/// [`ConversationSegment`](super::node::ConversationSegment) with:
/// ```ignore
/// SegmentId {
///     start_turn: self.start_turn_id,
///     end_turn:   self.end_turn_id,
/// }
/// ```
/// The `node_range` field records the vec-index range at snapshot time; it
/// goes stale as new turns are appended and is treated as diagnostic-only.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct SummarizationSnapshot {
    pub reason: SummarizationReason,
    /// Index-range into `ConversationTree::nodes` covered by this snapshot
    /// (diagnostic only — stale once new nodes are appended).
    pub node_range: std::ops::Range<usize>,
    /// TurnId of the first original turn summarised (becomes `SegmentId::start_turn`).
    pub start_turn_id: TurnId,
    /// TurnId of the last original turn summarised (becomes `SegmentId::end_turn`).
    pub end_turn_id: TurnId,
    pub elapsed_since_last: Duration,
    /// The items being compressed — either raw turns or existing segment summaries.
    pub content: SummarizationContent,
}

// ────────────────────────────────────────────────────────────────────────────
// SummarizationTask
// ────────────────────────────────────────────────────────────────────────────

/// A running summarization inference task, created from a
/// [`SummarizationSnapshot`] by [`SummarizationTask::launch()`].
///
/// Implements [`CognitiveTask`]: holds a live event channel from the
/// scheduler and accumulates token events until `Done` arrives, then
/// packages the result as a [`TreePatch`] containing a
/// [`ConversationSegment`](super::node::ConversationSegment) node.
///
/// No background thread is spawned. The scheduler's thread sends events;
/// [`poll()`](CognitiveTask::poll) drains them non-blockingly. The task holds
/// a clone of `scheduler_tx` solely to send `FreeSequence` when done or
/// aborted.
pub(crate) struct SummarizationTask {
    /// The temporary sequence slot allocated for this summarization.
    seq_id: SequenceId,
    /// True once `FreeSequence` has been sent, to avoid double-free.
    freed: bool,
    /// Set by `abort()`. Checked at the top of `poll()`.
    cancelled: Arc<AtomicBool>,
    /// Live event stream from the scheduler for this sequence.
    event_rx: Receiver<crate::handle::TurnEvent>,
    /// Used to send `FreeSequence` when done or aborted.
    scheduler_tx: crossbeam::channel::Sender<crate::scheduler::SchedulerRequest>,
    /// `(start_turn_id, end_turn_id)` of the summarized window.
    span: (TurnId, TurnId),
    /// Optional observer channel: when set, streaming events (Token,
    /// Prefill, PrefillProgress, HealthWarning) are forwarded here before
    /// being consumed by `poll()`. This allows callers to monitor
    /// summarization inference in real time.
    event_observer: Option<crossbeam::channel::Sender<crate::handle::TurnEvent>>,
}

impl SummarizationTask {
    /// Allocate a scheduler slot and submit the summarization turn.
    ///
    /// Synchronous steps (fast, no inference):
    /// 1. Tokenize the system prompt and formatted window text.
    /// 2. Send [`NewConversation`](crate::scheduler::SchedulerRequest::NewConversation)
    ///    and block until the scheduler returns a `seq_id`.
    /// 3. Send [`SubmitTurn`](crate::scheduler::SchedulerRequest::SubmitTurn)
    ///    (fire-and-forget; inference starts on the scheduler's thread).
    /// 4. Return a handle holding the `event_rx` from step 3.
    pub(crate) fn launch(
        snapshot: &SummarizationSnapshot,
        config: &ConversationTreeConfig,
        scheduler_tx: crossbeam::channel::Sender<crate::scheduler::SchedulerRequest>,
        tokenizer: &tokenizers::Tokenizer,
        event_observer: Option<crossbeam::channel::Sender<crate::handle::TurnEvent>>,
    ) -> crate::Result<Self> {
        use crate::config::SamplingConfig;
        use crate::error::ConversationError;
        use crate::scheduler::SchedulerRequest;
        use crate::substrate::TurnContentBounds;

        // Tokenize the system prompt and window text together — the
        // summarisation slot is its own short-lived workspace; we
        // prefill `(system + window)` in a single `SubmitTurn` so
        // the model attends over its task framing while reading the
        // window.
        let system_tokens = tokenize_text(tokenizer, &config.summarization_system_prompt)?;
        let window_text = format_window(snapshot);
        let window_tokens = tokenize_text(tokenizer, &window_text)?;
        let mut prefill_tokens = TokenBuffer::new();
        for &tok in &system_tokens[..] {
            prefill_tokens.push(tok);
        }
        for &tok in &window_tokens[..] {
            prefill_tokens.push(tok);
        }

        // NewSequence — one blocking round-trip; no inference happens here.
        // The summarisation task gets its own fresh `Conversation`
        // workspace; nothing shared with the caller's substrate.
        let (resp_tx, resp_rx) = crossbeam::channel::bounded(1);
        scheduler_tx
            .send(SchedulerRequest::NewSequence {
                conversation: crate::projection::Conversation::new(),
                // Summarisation task: raw prefill+decode, no projection,
                // no substrate write — no target binding needed.
                target: None,
                response_tx: resp_tx,
            })
            .map_err(|_| ConversationError::SchedulerGone)?;
        let seq_id = resp_rx
            .recv()
            .map_err(|_| ConversationError::SchedulerGone)??;

        // SubmitTurn — fire-and-forget; inference starts on scheduler thread.
        // Use argmax (temperature=0) for deterministic summarization output,
        // but add a repeat penalty so greedy decoding cannot enter a phrase loop.
        // Pure argmax with no penalties is vulnerable to looping on long contexts
        // because the model is forced to repeat the single highest-probability token
        // at each step once it falls into a local attractor.  A modest repeat penalty
        // breaks these attractors while keeping output deterministic overall.
        let summarization_sampling = SamplingConfig {
            temperature: 0.0,
            repeat_penalty: 1.1,
            repeat_last_n: 128,
            ..SamplingConfig::default()
        };
        let (event_tx, event_rx) = crossbeam::channel::unbounded();
        if scheduler_tx
            .send(SchedulerRequest::SubmitTurn {
                sequence_id: seq_id,
                projection_inputs: None,
                prefill_tokens,
                prefill_text: window_text,
                user_text: String::new(),
                content_bounds: TurnContentBounds::default(),
                prefill_assistant_text: String::new(),
                post_decode_tokens: TokenBuffer::new(),
                max_decode_tokens: config.summarization_max_tokens as usize,
                sampling: summarization_sampling,
                event_tx,
                reprojection: None,
                disable_reprojection: false,
                // Summarization decodes free text only — no tool stencils.
                triggers: Arc::new(TriggerRegistry::new()),
            })
            .is_err()
        {
            // Free the slot we just allocated before propagating the error.
            let _ = scheduler_tx.send(SchedulerRequest::FreeSequence {
                sequence_id: seq_id,
            });
            return Err(ConversationError::SchedulerGone);
        }

        Ok(Self {
            seq_id,
            freed: false,
            cancelled: Arc::new(AtomicBool::new(false)),
            event_rx,
            scheduler_tx,
            span: (snapshot.start_turn_id, snapshot.end_turn_id),
            event_observer,
        })
    }

    fn free_sequence(&mut self) {
        if !self.freed {
            self.freed = true;
            self.scheduler_tx
                .send(crate::scheduler::SchedulerRequest::FreeSequence {
                    sequence_id: self.seq_id,
                })
                .ok();
        }
    }
}

impl Drop for SummarizationTask {
    /// Safety net: if a task is dropped before `poll()` returns `Ready` (e.g.
    /// because the `Sequence` was dropped while it still had pending tasks),
    /// this ensures the scheduler sequence slot is still freed. Without this,
    /// the slot leaks until the scheduler exits, which can cause the CUDA
    /// driver to encounter an allocated-but-orphaned sequence during its atexit
    /// cleanup → `STATUS_ACCESS_VIOLATION`.
    fn drop(&mut self) {
        self.free_sequence();
    }
}

impl CognitiveTask for SummarizationTask {
    fn kind(&self) -> TaskKind {
        TaskKind::Summarization
    }

    fn relevant_turns(&self) -> Option<RangeInclusive<TurnId>> {
        Some(self.span.0..=self.span.1)
    }

    fn poll(&mut self) -> TaskPoll {
        use crate::handle::TurnEvent;

        // Check abort flag first.
        if self.cancelled.load(Ordering::Relaxed) {
            self.free_sequence();
            return TaskPoll::Aborted;
        }
        // Drain all currently available events without blocking.
        loop {
            match self.event_rx.try_recv() {
                // Forward streaming events to observer, then continue.
                Ok(evt @ TurnEvent::Token(_))
                | Ok(evt @ TurnEvent::Prefill(_))
                | Ok(evt @ TurnEvent::PrefillProgress { .. })
                | Ok(evt @ TurnEvent::HealthWarning(_)) => {
                    if let Some(ref tx) = self.event_observer {
                        let _ = tx.send(evt);
                    }
                }
                // Projection events are timeline telemetry for the live UI;
                // the summariser has no use for them.
                Ok(TurnEvent::Projection(_)) => {}
                Ok(TurnEvent::Done(response)) => {
                    self.free_sequence();
                    let summary_text = crate::think_strip::strip_think_blocks(&response.text);
                    let segment = ConversationSegment::new(
                        SegmentId {
                            start_turn: self.span.0,
                            end_turn: self.span.1,
                        },
                        summary_text,
                    );
                    let patch = TreePatch {
                        appended: vec![ConversationNode::Segment(segment)],
                        metadata: None,
                    };
                    return TaskPoll::Ready(patch);
                }
                Ok(TurnEvent::Error(e)) => {
                    self.free_sequence();
                    return TaskPoll::Failed(e);
                }
                Err(TryRecvError::Empty) => return TaskPoll::Pending,
                Err(TryRecvError::Disconnected) => {
                    // Scheduler dropped the sender — treat as abort.
                    self.freed = true; // slot already gone
                    return TaskPoll::Aborted;
                }
            }
        }
    }

    fn abort(&self) {
        // Set the flag; FreeSequence is sent on the next poll() call to avoid
        // a race where poll() is mid-drain when abort() is called.
        self.cancelled.store(true, Ordering::Relaxed);
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Prompt formatting helpers
// ────────────────────────────────────────────────────────────────────────────

/// Format the snapshot window into the text submitted as the summarization
/// turn.
///
/// **Turn-level (level-1):** each entry becomes:
/// ```text
/// [T-day.seq]          ← only when temporal markers enabled
/// User: {user_text}
/// Character: {assistant_text}
/// ```
///
/// **Segment-level (level 2+):** each entry becomes:
/// ```text
/// Summary {n} (turns {start}–{end}):
/// {summary_text}
/// ```
fn format_window(snapshot: &SummarizationSnapshot) -> String {
    let mut out = String::new();
    match &snapshot.content {
        SummarizationContent::Turns(turns) => {
            for entry in turns {
                if !entry.temporal_marker_text.is_empty() {
                    out.push_str(&entry.temporal_marker_text);
                    out.push('\n');
                }
                out.push_str("User: ");
                out.push_str(&entry.user_text);
                out.push('\n');
                out.push_str("Character: ");
                out.push_str(&entry.assistant_text);
                out.push_str("\n\n");
            }
        }
        SummarizationContent::Segments(segments) => {
            for (i, entry) in segments.iter().enumerate() {
                let (start_seq, end_seq) = if let NodeId::Segment(sid) = entry.node_id {
                    (sid.start_turn.seq, sid.end_turn.seq)
                } else {
                    (0, 0)
                };
                out.push_str(&format!(
                    "Summary {} (turns {}-{}):\n{}\n\n",
                    i + 1,
                    start_seq,
                    end_seq,
                    entry.summary_text
                ));
            }
        }
    }
    out
}

fn tokenize_text(tokenizer: &tokenizers::Tokenizer, text: &str) -> crate::Result<TokenBuffer> {
    tokenizer
        .encode(text, false)
        .map(|enc| TokenBuffer::from(enc.get_ids()))
        .map_err(|e| crate::error::ConversationError::Tokenizer(e.to_string()))
}
