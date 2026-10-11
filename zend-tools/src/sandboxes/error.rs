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
    /// With `--cluster`, no agent can be reached, or none holds the
    /// repository and the program (§11.5).
    #[error("no agent can run this: {0}")]
    NoAgent(String),
    /// The agent refused the command, which never ran.
    #[error("the agent refused the command: {0}")]
    AgentRefused(String),
    /// The job failed on the agent, or on its way there and back.
    #[error("the job failed on its agent: {0}")]
    AgentFailed(String),
}
