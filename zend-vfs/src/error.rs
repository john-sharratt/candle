//! Every way a git operation fails, as data.

use std::io;
use std::path::PathBuf;
use std::time::Duration;

use thiserror::Error;

use crate::types::{BranchName, RefName, RemoteName};
use crate::version::GitVersion;

#[derive(Debug, Error)]
pub enum GitError {
    #[error("git is not installed or not on PATH: {0}")]
    GitMissing(io::Error),
    #[error("git {found} is installed; the git layer needs {need} or newer")]
    GitTooOld { found: GitVersion, need: GitVersion },
    #[error("{} is not the top level of a git working tree", dir.display())]
    NotARepository { dir: PathBuf },
    #[error("unknown revision {rev}")]
    UnknownRevision { rev: String },
    #[error("{object} is a {kind}, not a blob")]
    NotABlob { object: String, kind: String },
    #[error("{object} is {size} bytes, over the {limit}-byte read limit")]
    BlobTooLarge {
        object: String,
        size: u64,
        limit: u64,
    },
    #[error("ref {name} is locked by another process")]
    RefLocked { name: RefName },
    #[error("ref {name} did not hold its expected value: {detail}")]
    StaleRef { name: RefName, detail: String },
    #[error("branch {branch} is checked out; moving it would change the working tree")]
    CheckedOutBranch { branch: BranchName },
    #[error("no remote named {remote}")]
    UnknownRemote { remote: RemoteName },
    #[error("a remote named {remote} already exists")]
    RemoteExists { remote: RemoteName },
    #[error("authentication to remote {remote} failed: {detail}")]
    AuthFailed { remote: RemoteName, detail: String },
    #[error("remote {remote} could not be reached: {detail}")]
    RemoteUnreachable { remote: RemoteName, detail: String },
    #[error("invalid input: {0}")]
    InvalidInput(String),
    #[error("git {command} printed output the layer could not parse: {detail}")]
    Malformed {
        command: &'static str,
        detail: String,
    },
    #[error("git {} did not finish within {after:?}", args.join(" "))]
    Timeout { args: Vec<String>, after: Duration },
    #[error("git could not be run: {0}")]
    Io(#[from] io::Error),
    #[error("git {} failed (exit {status:?}): {stderr}", args.join(" "))]
    Unclassified {
        args: Vec<String>,
        status: Option<i32>,
        stderr: String,
    },
}

impl GitError {
    /// A value a type constructor refused.
    ///
    /// Public because a caller layering its own argument rules on top of this
    /// one — the `git_*` tools do — needs to report them as the same kind of
    /// failure, so `invalid_arguments` means one thing whether the value was
    /// refused here or above.
    pub fn invalid(detail: impl Into<String>) -> Self {
        Self::InvalidInput(detail.into())
    }

    /// Output a parser could not read.
    pub(crate) fn malformed(command: &'static str, detail: impl Into<String>) -> Self {
        Self::Malformed {
            command,
            detail: detail.into(),
        }
    }
}
