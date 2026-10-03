//! The unit a guardian is assembled from.

use crate::engine::guardian::view::{NpcView, Question, Verdict};

/// One concern, watched one way.
///
/// A module is two pure halves around the one thing the guardian does for it:
/// [`Self::question`] says what to ask the character (or nothing, when the
/// view alone decides), and [`Self::judge`] reads the answer. Splitting it so
/// keeps every module testable without a running engine, and lets the runner
/// own the asking.
pub trait Module: Send + Sync {
    /// The module's name in the configuration and the log.
    fn name(&self) -> &'static str;

    /// What to put to the character this scan, if anything.
    fn question(&self, view: &NpcView) -> Option<Question>;

    /// The reading, given the character's answer to [`Self::question`] —
    /// `None` when nothing was asked or the character did not answer.
    fn judge(&self, view: &NpcView, answer: Option<&str>) -> Verdict;
}
