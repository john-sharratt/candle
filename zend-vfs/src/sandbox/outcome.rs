//! What a sandbox run hands back.

use crate::file_delta::TimedDelta;

/// One output stream of the command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Stream {
    /// What the stream held, up to the cap, with any invalid UTF-8 replaced.
    pub text: String,
    /// How many bytes the command wrote to it in all.
    pub bytes: u64,
    /// Whether `text` stops short of everything the command wrote.
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
/// change is reported here rather than lost; the checkout no longer holds it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Unrecorded {
    pub path: String,
    pub delta: TimedDelta,
    /// Why the store refused it.
    pub why: String,
}

/// Everything one run produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunOutcome {
    /// The command's exit code; `None` when it was killed or ended by a
    /// signal.
    pub exit_code: Option<i32>,
    /// Whether it was killed for running past its timeout.
    pub timed_out: bool,
    pub stdout: Stream,
    pub stderr: Stream,
    /// What the command changed, now in the conversation's store, in path
    /// order.
    pub changed: Vec<ChangedFile>,
    /// What the command changed that the store could not take.
    pub unrecorded: Vec<Unrecorded>,
}
