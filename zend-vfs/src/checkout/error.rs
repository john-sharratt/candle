//! Every way materialising or capturing a checkout fails.

use thiserror::Error;

use crate::GitError;

#[derive(Debug, Error)]
pub enum CheckoutError {
    #[error(transparent)]
    Git(#[from] GitError),
    /// Reading, writing or removing a file in the checkout failed.
    #[error("{path}: {source}")]
    Io {
        path: String,
        #[source]
        source: std::io::Error,
    },
    /// The conversation's changes to `path` do not fit the file at the base
    /// commit — made against a different version of it.
    #[error(
        "{path}: the conversation's changes do not fit the file at the base commit, so \
         they were made against a different version of it"
    )]
    Diverged { path: String },
    /// A commit landed on `branch` between reading it and checking it out,
    /// so the checkout is not at the commit the conversation's changes were
    /// laid over. Nothing of the conversation's has been written.
    #[error("{branch} moved while the checkout was being put on it; run again")]
    BranchMoved { branch: String },
    /// The checkout's own state, set aside while a run used it, could not be
    /// put back. It is kept whole — the journal in `journal` names the
    /// commits under `refs/zend/preserved/` and the files moved aside beside
    /// it — and the next run puts it back before doing anything else.
    #[error(
        "the checkout's own state could not be put back yet; it is kept whole, journalled in \
         {journal}, and the next run finishes putting it back before anything else: {detail}"
    )]
    NotPutBack { journal: String, detail: String },
    /// A preservation's journal could not be written or read.
    #[error("the checkout's preservation journal {0}")]
    Journal(String),
    /// `path` could lead outside the checkout, or into its git database, or
    /// names something that is not a file.
    #[error("{path}: {why}")]
    UnsafePath { path: String, why: String },
}

impl CheckoutError {
    pub(crate) fn io(path: &str, source: std::io::Error) -> Self {
        CheckoutError::Io {
            path: path.to_string(),
            source,
        }
    }

    pub(crate) fn journal(detail: impl Into<String>) -> Self {
        CheckoutError::Journal(detail.into())
    }

    pub(crate) fn unsafe_path(path: &str, why: &str) -> Self {
        CheckoutError::UnsafePath {
            path: path.to_string(),
            why: why.to_string(),
        }
    }
}
