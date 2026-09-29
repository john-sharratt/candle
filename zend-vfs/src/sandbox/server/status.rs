//! Where a job stands, and what asking after one returns.

use std::path::PathBuf;

use thiserror::Error;

use super::job_id::JobId;
use crate::sandbox::{RunOutcome, SandboxError};

/// Where a job stands.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JobStatus {
    /// Waiting for the checkout, which another job holds.
    Queued,
    /// Holding the checkout: being put in place, running, or read back.
    Running,
    /// Ran to its end; the exit code, `None` when a signal ended it.
    Exited { code: Option<i32> },
    /// Killed for running past its timeout.
    TimedOut,
    /// Its handle was dropped before it finished; its command was killed.
    Cancelled,
    /// The security check refused the command, which never ran.
    Refused { why: String },
    /// A step around the command failed.
    Failed { why: String },
}

impl JobStatus {
    /// Whether the job has ended, one way or another.
    pub fn is_finished(&self) -> bool {
        !matches!(self, JobStatus::Queued | JobStatus::Running)
    }

    /// How a run that has ended stands.
    pub(super) fn of(result: &Result<RunOutcome, SandboxError>) -> Self {
        match result {
            Ok(outcome) if outcome.timed_out => JobStatus::TimedOut,
            Ok(outcome) => JobStatus::Exited {
                code: outcome.exit_code,
            },
            Err(SandboxError::Refused(why)) => JobStatus::Refused {
                why: why.to_string(),
            },
            Err(e) => JobStatus::Failed { why: e.to_string() },
        }
    }
}

/// What a job's query returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JobInfo {
    pub status: JobStatus,
    /// Lines written to its log so far — every `\n`.
    pub lines: u64,
    /// Bytes written to its log so far.
    pub bytes: u64,
    /// Its log file.
    pub log: PathBuf,
}

/// A job the server does not know: it never ran here, or it is older than
/// the jobs a server keeps.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("no job {0}: it never ran here, or it is older than the jobs kept")]
pub struct JobNotFound(pub JobId);
