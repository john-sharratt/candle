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
    /// The job's branch holds commits the conversation's files are not made
    /// on: a checkout of the branch is not what its changes were made on.
    #[error(
        "{branch} holds commits your files do not have: it is at {tip}, and your files are \
         made on {base}. git_merge brings them in; then run the command"
    )]
    Behind {
        branch: String,
        base: String,
        tip: String,
    },
    /// The conversation's files are made on a commit the job's branch does
    /// not hold yet — a merge fast-forwarded them past it, or the branch has
    /// no commit — so a checkout of the branch is not what they read.
    #[error(
        "your files are made on {base}, which {branch} does not hold yet (it is at {tip}) — \
         what a merge brought in is published by git_commit with `from: all_changes`, even with \
         no change of your own; then run the command"
    )]
    Ahead {
        branch: String,
        base: String,
        tip: String,
    },
    /// The conversation is finishing a merge: its files are made on a tree
    /// no commit holds, and no checkout of the branch is that tree.
    #[error(
        "you are finishing a merge on {branch}, which no checkout holds until it is committed: \
         settle any conflicts and git_commit with `from: all_changes`; then run the command"
    )]
    Merging { branch: String },
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
