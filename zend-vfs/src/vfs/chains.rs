//! The per-path deltas a session holds, and the stale-base check on a change.
//!
//! The session layer keeps the changes each path went through, never a copy of
//! its result, so a file's size and whether it is deleted can be read without
//! replaying the chain. A replacement or deletion supersedes everything before
//! it, so a chain is at most one of those followed by edits. A change is
//! recorded only if the base and the path's chain it was computed against still
//! stand when the write lock is taken; otherwise it is computed again against
//! what is there now.

use super::*;

/// The session layer: for each path the session has changed, the deltas that
/// took it from where it started to where it stands now. What is held is the
/// changes, never a copy of the result — see [`file_delta`](super::file_delta).
#[derive(Default)]
pub(super) struct Upper {
    pub(super) chains: HashMap<String, Chain>,
    /// Over a branch, what the chains are made on — `None` until the store
    /// first reads the branch and takes it.
    pub(super) base: Option<Base>,
}

impl Upper {
    pub(super) fn bytes(&self) -> usize {
        self.chains.values().map(Chain::bytes).sum()
    }

    /// Whether the session has `norm` deleted: it starts with a
    /// [`FileDelta::Delete`] — a whiteout over the workspace's copy.
    pub(super) fn deleted(&self, norm: &str) -> bool {
        self.chains.get(norm).is_some_and(|c| c.size.is_none())
    }
}

/// One path's deltas, in order.
///
/// A [`FileDelta::Replace`] or [`FileDelta::Delete`] supersedes everything
/// before it, so a chain is at most one of those followed by edits. A chain
/// that starts with an edit changes the workspace's copy of the file, and is
/// replayed onto it.
#[derive(Clone, PartialEq)]
pub(super) struct Chain {
    /// Each with the moment its operation executed.
    pub(super) deltas: Vec<TimedDelta>,
    /// The file's size after the last delta, `None` once deleted — kept so a
    /// listing never replays a chain to report a size.
    pub(super) size: Option<usize>,
    /// Left in conflict by a move of the base, until a change of the
    /// session's own leaves it without a conflict's markers.
    pub(super) conflict: bool,
}

impl Chain {
    pub(super) fn bytes(&self) -> usize {
        self.deltas.iter().map(|t| t.delta.bytes()).sum()
    }
}

/// What a change to one path was computed against: the base and the path's
/// chain as they stood when the file was read. A change is recorded only if
/// both still stand when the write lock is taken — otherwise a move of the
/// base, or another change to the path, landed in between, and the change is
/// computed again against what is there now.
#[derive(PartialEq)]
pub(super) struct Seen {
    pub(super) base: Option<Base>,
    pub(super) chain: Option<Chain>,
}

impl Seen {
    pub(super) fn of(upper: &Upper, norm: &str) -> Self {
        Self {
            base: upper.base.clone(),
            chain: upper.chains.get(norm).cloned(),
        }
    }

    pub(super) fn stands(&self, upper: &Upper, norm: &str) -> bool {
        self.base == upper.base && self.chain.as_ref() == upper.chains.get(norm)
    }
}
