//! Running a program on the machine: `run_command`, `run_output`.
//!
//! A command runs in the repository's sandbox (`crate::sandboxes`): a checkout
//! of the conversation's branch at the commit its files are based on, with its
//! uncommitted changes laid down, the repository's own folder set aside first
//! and put back after (`docs/zend_workspace_execution.md` §7.4). What the
//! program changes — a formatter's rewrite, a generated file — comes back into
//! the conversation's changes, as if the conversation had made the edits.
//!
//! The program is started with no shell in between: `program` is one program
//! and `args` its arguments, each passed as written. Which programs may start
//! is the sandbox's policy, an allow-list; git run directly is refused towards
//! the git tools. The call waits for the program to end — killed with
//! everything it started at its timeout — and returns the first page of what
//! it printed; `run_output` reads the rest a page at a time.
//!
//! # Error codes
//!
//! | Code | Cause |
//! |------|-------|
//! | `no_sandbox` | The repository is not a git repository, or the session has no sandboxes |
//! | `no_branch` | The conversation is on no branch in the repository |
//! | `behind` | The branch holds commits the conversation's files are not based on — git_merge first |
//! | `unpublished` | The conversation's files are based on a commit its branch does not hold yet — a merge fast-forwarded them — git_commit first |
//! | `merging` | The conversation is finishing a merge — git_commit it first |
//! | `refused` | The command is not one the policy runs: a program not listed, git run directly, a path out of the repository |
//! | `cannot_start` | The program could not be started — not installed, or not on the `PATH` |
//! | `sandbox_failed` | A step around the command failed: the checkout, the read-back, putting the folder back |
//! | `not_found` | `run_output` names a job the sandbox does not keep |
//! | `invalid_arguments` | A command line in `program`, shell syntax in `args`, or a `job` that is not a job id |
//! | `io_error` | The job's log could not be written or read |
//! | `not_permitted` | The context may not run programs on this host |
//! | `unknown_repo` | `repo` names no repository in the workspace |

pub mod command;
pub mod output;

pub use command::RUN_COMMAND;
pub use output::RUN_OUTPUT;

use thiserror::Error;
use zend_vfs::{SandboxError, UnknownRepo};

use crate::sandboxes::SandboxesError;
use crate::tools::code::UNKNOWN_REPO;
use crate::{NotPermitted, ToolError};

#[derive(Debug, Error)]
pub enum RunError {
    #[error("this session has no command sandboxes, so there is nowhere to run a program")]
    NoSandboxes,
    #[error("this conversation is on no branch in {0}; switch to one with git_switch first")]
    NoBranch(String),
    /// `run_output` was given something that is not a job id.
    #[error("{0:?} is not a job id — pass the `job` a run_command result gave")]
    BadJob(String),
    /// `program` held a whole command line.
    #[error(
        "`{given}` is a program and its arguments together; run_command takes them apart — \
         call it again with program {program:?} and args {args:?}"
    )]
    CommandLine {
        given: String,
        program: String,
        args: Vec<String>,
    },
    /// An argument is shell syntax, which reaches the program as it is.
    #[error("{0}")]
    Shell(String),
    #[error(transparent)]
    Sandboxes(#[from] SandboxesError),
    #[error(transparent)]
    NotPermitted(#[from] NotPermitted),
    #[error(transparent)]
    UnknownRepo(#[from] UnknownRepo),
}

impl ToolError for RunError {
    fn code(&self) -> &'static str {
        match self {
            RunError::NoSandboxes => "no_sandbox",
            RunError::NoBranch(_) => "no_branch",
            RunError::BadJob(_) | RunError::CommandLine { .. } | RunError::Shell(_) => {
                "invalid_arguments"
            }
            RunError::NotPermitted(_) => NotPermitted::CODE,
            RunError::UnknownRepo(_) => UNKNOWN_REPO,
            RunError::Sandboxes(e) => match e {
                SandboxesError::NoSandbox { .. } => "no_sandbox",
                SandboxesError::Log(_) => "io_error",
                SandboxesError::NotFound(_) => "not_found",
                SandboxesError::Job(job) => match job {
                    SandboxError::Refused(_) => "refused",
                    SandboxError::Behind { .. } => "behind",
                    SandboxError::Ahead { .. } => "unpublished",
                    SandboxError::Merging { .. } => "merging",
                    SandboxError::Start { .. } => "cannot_start",
                    _ => "sandbox_failed",
                },
            },
        }
    }
}
