//! git_status tool — what this conversation has not committed, and where its
//! branch stands.
//!
//! A conversation's uncommitted work is its own file store's changes over its
//! base on the branch it is on, never a working tree: the repository's folder
//! belongs to the sandbox. So this reads the store for the base, the changes
//! and the conflicts, and the local refs — as current as origin was at the
//! last fetch — for what the branch holds beyond the base.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_vfs::{FileState, LogRange, Rev};

use super::wire::{WireChange, WireUpstream};
use super::{open, GitToolError};
use crate::tools::file::Paging;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// About 27 tokens a path — 80 measured at 2,151 — so a page costs what a
/// `file_read` page does.
const PER_PAGE: usize = 80;

/// The most incoming commits counted; more reads as this many.
const MAX_INCOMING: usize = 100;

#[derive(Deserialize, JsonSchema, Validate)]
#[serde(deny_unknown_fields)]
pub struct StatusRequest {
    /// The repository to inspect. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// Zero-based page of changed paths, 80 a page. Required — pass 0 to
    /// start. A page past the end clamps to the last one.
    pub page: u32,
}

#[derive(Serialize)]
pub struct Counts {
    pub added: usize,
    pub modified: usize,
    pub deleted: usize,
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub repo: String,
    /// The branch this conversation is on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    /// The commit your files are based on — where the branch stood when you
    /// last committed, merged, switched or reset. Absent before the branch's
    /// first commit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head: Option<String>,
    /// While a merge is being finished: the commit it brings in. Your next
    /// commit records the merge.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merging: Option<String>,
    /// Commits the branch holds — as of the last fetch — that your files do
    /// not have yet; git_merge brings them in, and a commit waits for them.
    #[serde(skip_serializing_if = "is_zero")]
    pub incoming: usize,
    /// Files still in conflict from a merge, to settle before committing.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub conflicts: Vec<String>,
    /// How far the branch is ahead of and behind origin's, as of the last
    /// fetch. "How far ahead am I" is asked of this tool first, so the answer
    /// is here rather than one call away.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream: Option<WireUpstream>,
    /// Set when the branch has no copy on origin to compare against, saying
    /// why.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_upstream: Option<&'static str>,
    /// Totals over every changed path, not just this page — so "how many"
    /// is read rather than counted.
    pub counts: Counts,
    pub paging: Paging,
    pub changes: Vec<WireChange>,
    /// Nothing uncommitted, and no merge being finished.
    pub clean: bool,
}

pub struct GitStatus;

impl Tool for GitStatus {
    const NAME: &'static str = "git_status";
    const DESCRIPTION: &'static str =
        "Report what you have not committed in a repository — every file you added, \
         modified or deleted since your files' base commit — and which branch you are on, \
         that commit, how many commits the branch has that you do not (`incoming`, which \
         git_merge brings in), files still in conflict from a merge, and how far ahead and \
         behind origin's copy the local branch is, with totals so \"how many\" needs no \
         counting. Use for \"what have I changed\", \"is this clean\", \"what branch am I \
         on\", \"am I up to date\", \"what is still in conflict\", or to see what a commit \
         of your changes would hold. `page` is required — pass 0 to start, and \
         `paging.next_page` for the next. Reads only.";

    type Request = StatusRequest;
    type Response = StatusResponse;
    type Error = GitToolError;

    /// Reads the conversation's files and the local refs; writes nothing.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: StatusRequest) -> Result<StatusResponse, GitToolError> {
        let repo = open(ctx, &req.repo)?;
        let branch = repo.branch();
        let base = repo.store().and_then(|s| s.base().ok().flatten());
        let head = base.as_ref().and_then(|b| b.commit().cloned());
        let merging = base.as_ref().and_then(|b| b.merging().cloned());
        let conflicts = repo.store().map(|s| s.conflicts()).unwrap_or_default();
        let origin = repo.origin()?;
        let has_origin = origin.is_some();
        // What the branch holds as last fetched: origin's copy, or with no
        // origin the local branch — counted beyond what the base has.
        let record = match (&branch, &origin) {
            (Some(b), Some(origin)) => repo.ref_target(&origin.tracking(b))?,
            (Some(b), None) => repo.ref_target(&b.to_ref())?,
            (None, _) => None,
        };
        // Every parent of the base is had: a merge being finished has them
        // all.
        let have: Vec<Rev> = base
            .as_ref()
            .map(|b| b.parents.iter().cloned().map(Rev::Oid).collect())
            .unwrap_or_default();
        let incoming = match &record {
            Some(record) if !have.is_empty() => repo
                .log(
                    &LogRange {
                        exclude: have,
                        ..LogRange::of(Rev::Oid(record.clone()))
                    },
                    MAX_INCOMING,
                )?
                .len(),
            _ => 0,
        };
        let tracked = match &branch {
            Some(current) => repo
                .branches()?
                .into_iter()
                .find(|b| &b.name == current)
                .and_then(|b| b.upstream),
            None => None,
        };
        let no_upstream = match (&branch, &tracked, has_origin) {
            (Some(_), None, true) => Some(
                "origin has no copy of this branch yet; its first commit or git_switch \
                 publishes it there",
            ),
            (Some(_), None, false) => {
                Some("this repository has no origin; its branches are kept on this machine only")
            }
            _ => None,
        };

        let changed = repo.store().map(|s| s.status()).unwrap_or_default();
        let count = |state: FileState| changed.iter().filter(|(_, s)| *s == state).count();
        let counts = Counts {
            added: count(FileState::Added),
            modified: count(FileState::Modified),
            deleted: count(FileState::Deleted),
        };
        let paging = Paging::of(changed.len(), req.page, PER_PAGE);
        let changes = changed
            .into_iter()
            .skip(paging.skipped())
            .take(paging.per_page)
            .map(|(path, state)| WireChange {
                status: match state {
                    FileState::Added => "added",
                    FileState::Modified => "modified",
                    FileState::Deleted => "deleted",
                },
                path,
                from_path: None,
            })
            .collect();

        let clean = paging.total == 0 && merging.is_none();
        Ok(StatusResponse {
            repo: req.repo,
            branch: branch.map(|b| b.as_str().to_string()),
            head: head.map(|o| o.as_str().to_string()),
            merging: merging.map(|o| o.as_str().to_string()),
            incoming,
            conflicts,
            upstream: tracked.as_ref().map(Into::into),
            no_upstream,
            clean,
            counts,
            paging,
            changes,
        })
    }
}

fn is_zero(n: &usize) -> bool {
    *n == 0
}

pub const GIT_STATUS: RegisteredTool = RegisteredTool::new::<GitStatus>();
