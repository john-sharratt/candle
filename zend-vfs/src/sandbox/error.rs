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
    /// The conversation's store writes the disk directly, so its changes are
    /// on disk already and a run's reset would destroy them.
    #[error(
        "the conversation's files are written to disk directly; a sandbox runs over an overlay"
    )]
    DirectStore,
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
    /// The checkout could not be reset after the run. `run` is how the run
    /// itself ended, when it failed first. Nothing was recorded in the
    /// conversation's store.
    #[error("the checkout could not be reset after the run: {reset}")]
    Reset {
        reset: Box<SandboxError>,
        run: Option<Box<SandboxError>>,
    },
}
