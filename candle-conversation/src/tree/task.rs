//! [`CognitiveTask`] trait and supporting types.
//!
//! A `CognitiveTask` is a **consumer-side handle** to already-running
//! background inference work. Once the tree has launched a task (allocated a
//! scheduler slot, submitted the inference turn), it pushes the handle onto
//! `pending_tasks`; `Sequence` drains and spin-polls each handle until it
//! resolves.
//!
//! The trait exposes only what the consumer needs:
//! - [`kind()`](CognitiveTask::kind) — what type of work this is
//! - [`relevant_turns()`](CognitiveTask::relevant_turns) — which turn range
//!   the task operates over (used for dedup before launching a new task)
//! - [`poll()`](CognitiveTask::poll) — non-blocking check for completion
//! - [`abort()`](CognitiveTask::abort) — signal early termination

use std::ops::RangeInclusive;

use super::patch::TreePatch;
use super::types::TurnId;
use crate::error::ConversationError;

// ────────────────────────────────────────────────────────────────────────────
// TaskKind
// ────────────────────────────────────────────────────────────────────────────

/// Identifies the kind of cognitive work a task represents.
///
/// Used for logging, metrics, and inspecting the `pending_tasks` queue.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Daydream/Sleep/Reason variants are stubs for future task types.
pub(crate) enum TaskKind {
    /// Compress a window of past turns into a [`ConversationSegment`](super::node::ConversationSegment).
    Summarization,
    /// Associative daydream triggered by resonance with a cold node.
    Daydream,
    /// End-of-day prospective sleep simulation.
    Sleep,
    /// Executive self-dialogue producing an updated plan.
    Reason,
}

// ────────────────────────────────────────────────────────────────────────────
// TaskPoll
// ────────────────────────────────────────────────────────────────────────────

/// Result of a non-blocking [`CognitiveTask::poll()`] call.
pub(crate) enum TaskPoll {
    /// Background work is still running; no result yet.
    Pending,
    /// Work is complete. The contained [`TreePatch`] should be applied to the
    /// conversation tree via [`ConversationTree::apply_patch`](super::conversation_tree::ConversationTree::apply_patch).
    Ready(TreePatch),
    /// The task was aborted before it produced a result — either via
    /// [`CognitiveTask::abort()`] or because the scheduler channel closed.
    /// No patch should be applied.
    Aborted,
    /// The task failed during execution. The error is logged by the caller;
    /// no patch is applied.
    Failed(ConversationError),
}

// ────────────────────────────────────────────────────────────────────────────
// CognitiveTask
// ────────────────────────────────────────────────────────────────────────────

/// A consumer-side handle to a running background cognitive task.
///
/// Implemented by [`SummarizationTask`](super::summarize::SummarizationTask)
/// and, in future, by Daydream, Sleep, and Reason task types.
///
/// Tasks are created when a trigger fires inside `ConversationTree` (e.g.
/// turn count threshold reached). The tree records the trigger as a
/// [`SummarizationSnapshot`](super::summarize::SummarizationSnapshot); the
/// `Sequence` layer converts that snapshot into a running task with
/// access to the scheduler channel and tokenizer.
///
/// # Contract
///
/// - `poll()` is non-blocking: call it on the main thread at turn boundaries.
/// - `abort()` is idempotent: safe to call multiple times.
/// - Once `poll()` returns `Ready`, `Aborted`, or `Failed`, the task is done.
///   Further calls to `poll()` are unspecified but safe.
pub(crate) trait CognitiveTask: Send {
    /// What kind of background work this task represents.
    #[allow(dead_code)]
    fn kind(&self) -> TaskKind;

    /// The turn range this task operates over, if applicable.
    ///
    /// Used by `ConversationTree` to detect duplicate tasks before launching a
    /// new one: if any pending task's range covers the candidate window,
    /// launch is skipped. Tasks with no specific turn scope (e.g. `Sleep`)
    /// return `None`.
    fn relevant_turns(&self) -> Option<RangeInclusive<TurnId>>;

    /// Non-blocking poll. Returns `Ready`/`Aborted`/`Failed` once the
    /// background work finishes; `Pending` otherwise.
    fn poll(&mut self) -> TaskPoll;

    /// Signal the running task to stop. Idempotent.
    ///
    /// The next `poll()` call returns `Aborted` once the task has noticed
    /// the signal and released the scheduler slot.
    #[allow(dead_code)]
    fn abort(&self);
}
