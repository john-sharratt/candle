//! Strongly-typed sequence handle for scheduler slots.
//!
//! [`SequenceId`] is an opaque token identifying a scheduler-owned
//! GPU sequence slot.

// ────────────────────────────────────────────────────────────────────────────
// SequenceId
// ────────────────────────────────────────────────────────────────────────────

/// An opaque handle to a scheduler sequence slot.
///
/// Publicly visible but **not publicly constructible** — obtainable only
/// from the scheduler (via
/// [`NewSequence`](crate::scheduler::SchedulerRequest::NewSequence)).
/// This prevents callers from forging IDs and accidentally referencing
/// the wrong scheduler slot.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct SequenceId(pub(crate) usize);

impl std::fmt::Display for SequenceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BlockCount
// ────────────────────────────────────────────────────────────────────────────

/// The total number of sealed KV blocks in a sequence at a given point.
///
/// Carried alongside [`SealResult`](crate::handle::SealResult)
/// and stored per-turn in [`Sequence`](crate::conversation::Sequence).
/// Carried as a typed value so callers don't need to infer what the `usize`
/// represents from context.
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Default)]
pub(crate) struct BlockCount(pub(crate) usize);

impl std::fmt::Display for BlockCount {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// BlockRange
// ────────────────────────────────────────────────────────────────────────────

/// A half-open range of KV block indices `[start, end)` passed to
/// [`SubmitTurn`](crate::scheduler::SchedulerRequest::SubmitTurn) to
/// describe which parent blocks the carved view should borrow.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub(crate) struct BlockRange {
    pub(crate) start: usize,
    pub(crate) end: usize,
}

impl BlockRange {
    pub(crate) fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }

    /// Convert to the raw `(start, end)` tuple expected by the session API.
    pub(crate) fn to_raw(self) -> (usize, usize) {
        (self.start, self.end)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sequence_id_display() {
        let id = SequenceId(99);
        assert_eq!(id.to_string(), "99");
    }

    #[test]
    fn sequence_id_copy_and_eq() {
        let id = SequenceId(5);
        let id2 = id; // Copy
        assert_eq!(id, id2);
        assert_ne!(id, SequenceId(6));
    }
}
