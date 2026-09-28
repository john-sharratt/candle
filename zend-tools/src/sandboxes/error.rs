//! Why a sandbox job could not be run or read.

use std::io;

use thiserror::Error;
use zend_vfs::{JobNotFound, SandboxError};

#[derive(Debug, Error)]
pub enum SandboxesError {
    /// `repo` is not a git repository of the workspace.
    #[error(
        "{repo} has no sandbox — commands run only in the workspace's git repositories: {known}"
    )]
    NoSandbox { repo: String, known: String },
    /// The job's log could not be made or read.
    #[error("the job's log could not be written or read: {0}")]
    Log(io::Error),
    /// The job itself failed or was refused.
    #[error(transparent)]
    Job(#[from] SandboxError),
    #[error(transparent)]
    NotFound(#[from] JobNotFound),
}
