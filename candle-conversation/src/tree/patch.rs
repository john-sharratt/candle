//! [`TreePatch`] and [`TreeMetadataDelta`] — delta types produced by a
//! background fork and applied to the main tree via
//! [`ConversationTree::apply_patch`](super::conversation_tree::ConversationTree::apply_patch).
//!
//! # How a patch reaches the tree
//!
//! Summarization is the only producer today. `ConversationTree::run_summarize`
//! launches a `SummarizationTask` — a `CognitiveTask` whose inference runs on
//! the scheduler — and pushes the handle onto the tree's `pending_tasks`.
//! `Sequence::finish_turn` drains that queue via `drain_pending_tasks()` and
//! spin-polls each handle; on `TaskPoll::Ready(patch)` it calls
//! [`ConversationTree::apply_patch`](super::conversation_tree::ConversationTree::apply_patch)
//! and then re-checks whether a recursive segment-of-segments summarization
//! should fire.
//!
//! So patch application still happens on the main thread at a turn boundary,
//! which is the property the design wanted — but by polling a task handle, not
//! by receiving on a channel. The spin-poll is deliberately crude (see
//! `Conversation::run_task_blocking_inner`): summarization is infrequent enough
//! that blocking a turn boundary on it is acceptable for now.
//!
//! [`ConversationTree::fork`](super::conversation_tree::ConversationTree::fork)
//! and [`ConversationTreeFork`](super::conversation_tree::ConversationTreeFork)
//! offer the alternative — a `Send`-able snapshot plus a one-shot
//! [`Receiver<TreePatch>`](flume::Receiver) for genuinely
//! off-thread work. Nothing calls them yet; they are the seam for moving
//! summarization off the turn boundary.

use super::node::ConversationNode;

// ────────────────────────────────────────────────────────────────────────────
// TreePatch / TreeMetadataDelta
// ────────────────────────────────────────────────────────────────────────────

/// Delta of mutations produced by a background fork, to be applied to the
/// main tree via
/// [`ConversationTree::apply_patch`](super::conversation_tree::ConversationTree::apply_patch).
///
/// Produced by a completed `SummarizationTask` and applied at the next turn
/// boundary — see the module docs for the path it takes.
#[derive(Debug)]
pub struct TreePatch {
    /// Nodes appended by the fork (e.g. new segment nodes from summarization).
    pub appended: Vec<ConversationNode>,
    /// Optional scalar-state changes.
    pub metadata: Option<TreeMetadataDelta>,
}

/// Scalar-state changes carried in a [`TreePatch`].
#[derive(Debug)]
pub struct TreeMetadataDelta {
    /// Override `turns_since_summarize` on the main tree after applying this
    /// patch (e.g. reset to 0 after a segment is inserted).
    pub turns_since_summarize: Option<u32>,
}
