//! Every way a sandbox run fails.

use std::path::PathBuf;

use thiserror::Error;

use super::policy::Refused;
use crate::checkout::CheckoutError;

#[derive(Debug, Error)]
pub enum SandboxError {
    /// The conversation's store is over a different folder than this
    /// sandbox's repository — or over none.
    #[error("the conversation's files are for {store:?}, not this sandbox's repository {repo}")]
    WrongStore {
        store: Option<PathBuf>,
        repo: PathBuf,
    },
    /// The conversation's store reads another branch than the job's — or the
    /// repository's folder rather than a branch at all.
    #[error("the conversation's files read {reads:?}, not the job's branch {branch}")]
    WrongBranch {
        branch: String,
        reads: Option<String>,
    },
    /// The conversation's changes are made on another commit than the job's
    /// branch holds — it has not merged what the branch has gained, or is
    /// finishing a merge — and a checkout of the branch is not what they
    /// were made on.
    #[error(
        "the conversation's files are made on {base}, but {branch} holds {tip}; merge the \
         branch into them, or finish the merge under way, before running a command"
    )]
    BaseNotBranch {
        branch: String,
        base: String,
        tip: String,
    },
    /// The conversation's store could not give its changes.
    #[error("the conversation's files could not be read: {0}")]
    Store(String),
    /// Putting the checkout in place, or reading back what the command
    /// changed, failed.
    #[error(transparent)]
    Checkout(#[from] CheckoutError),
    /// The security check refused the command.
    #[error("refused: {0}")]
    Refused(#[from] Refused),
    /// The command could not be started.
    #[error("{program} could not be started: {source}")]
    Start {
        program: String,
        #[source]
        source: std::io::Error,
    },
    /// A step running off the async runtime was lost — it panicked, or the
    /// runtime is shutting down.
    #[error("a sandbox step was interrupted: {0}")]
    Interrupted(String),
}
