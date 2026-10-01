//! What a sandbox run hands back.

use crate::file_delta::TimedDelta;

/// What the command printed, as its sink took it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Output {
    /// How many bytes the command wrote to its two streams together.
    pub bytes: u64,
    /// Whether the sink holds less than that — the output ran past the cap,
    /// or the sink stopped taking writes.
    pub truncated: bool,
}

/// A file the command changed, as the delta the conversation's store now
/// holds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChangedFile {
    /// Repository-relative, `/`-separated.
    pub path: String,
    pub delta: TimedDelta,
}

/// A file the command changed that the conversation's store could not take —
/// a binary file, or one that would put the store over its size cap. The
/// change is reported here rather than lost.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unrecorded {
    pub path: String,
    pub delta: TimedDelta,
    /// Why the store refused it.
    pub why: String,
}

/// Everything one run produced besides its output, which went to the sink.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutcome {
    /// The command's exit code; `None` when it was killed or ended by a
    /// signal.
    pub exit_code: Option<i32>,
    /// Whether it was killed for running past its timeout.
    pub timed_out: bool,
    pub output: Output,
    /// What the command changed, now in the conversation's store, in path
    /// order.
    pub changed: Vec<ChangedFile>,
    /// What the command changed that the store could not take.
    pub unrecorded: Vec<Unrecorded>,
}
