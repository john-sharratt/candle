//! [`TreePatch`] and [`TreeMetadataDelta`] — delta types produced by a
//! background fork and applied to the main tree via
//! [`ConversationTree::apply_patch`](super::conversation_tree::ConversationTree::apply_patch).
//!
//! # Fork / patch pattern
//!
//! When the engine needs background work (e.g. async summarization), it calls
//! [`ConversationTree::fork`] to get a `Send`-able
//! [`ConversationTreeFork`](super::conversation_tree::ConversationTreeFork)
//! plus a one-shot [`Receiver<TreePatch>`](crossbeam::channel::Receiver).
//! The fork is sent to a background thread; when it finishes it calls
//! `ConversationTreeFork::finish(patch)` to send the result back.
//!
//! On the main thread, the engine stores the receiver as
//! `pending_fork_rx: Option<Receiver<TreePatch>>`. At the start of each new
//! turn submission a `drain_pending_patch()` helper calls `try_recv()` on the
//! channel — non-blocking, so the turn proceeds immediately if no patch is
//! ready. This guarantees patch application happens on the main thread in a
//! quiescent moment, with no locking or blocking.
//!
//! TODO: wire `drain_pending_patch()` into the turn-loop once `run_summarize()`
//! spawns a real background thread.

use super::node::ConversationNode;

// ────────────────────────────────────────────────────────────────────────────
// TreePatch / TreeMetadataDelta
// ────────────────────────────────────────────────────────────────────────────

/// Delta of mutations produced by a background fork, to be applied to the
/// main tree via
/// [`ConversationTree::apply_patch`](super::conversation_tree::ConversationTree::apply_patch).
///
/// The `run_summarize()` stub currently never sends a real patch (it logs and
/// returns immediately). The `TreePatch` and channel infrastructure are
/// present so the real async worker can slot in without struct changes.
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
