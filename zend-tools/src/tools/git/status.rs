//! git_status tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::wire::{WireStatus, WireUpstream};
use super::{open, GitToolError};
use crate::tools::file::Paging;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// About 27 tokens a path — 80 measured at 2,151 — so a page costs what a
/// `file_read` page does.
const PER_PAGE: usize = 80;

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
    pub staged: usize,
    pub unstaged: usize,
    pub untracked: usize,
    pub unmerged: usize,
}

#[derive(Serialize)]
pub struct StatusResponse {
    pub repo: String,
    /// The branch checked out, absent on a detached `HEAD`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    /// The commit `HEAD` resolves to, absent in a repository with no commits.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub head: Option<String>,
    /// What the checked-out branch tracks, and how far ahead and behind it
    /// is. "How far ahead am I" is asked of this tool first — measured live —
    /// so the answer is here rather than one call away. Absent when the branch
    /// tracks nothing, and then `no_upstream` says where to look instead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub upstream: Option<WireUpstream>,
    /// Set when the checked-out branch tracks nothing: the call that answers
    /// "how far ahead" without an upstream.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub no_upstream: Option<&'static str>,
    /// Totals over every changed path, not just this page — so "how many"
    /// is read rather than counted.
    pub counts: Counts,
    pub paging: Paging,
    pub changes: Vec<WireStatus>,
    /// No changes and no untracked files.
    pub clean: bool,
}

pub struct GitStatus;

impl Tool for GitStatus {
    const NAME: &'static str = "git_status";
    const DESCRIPTION: &'static str =
        "Report what is uncommitted in a repository's working tree: which files were \
         modified, added, deleted, renamed or left untracked, and which of those changes \
         are staged. Also names the branch checked out, the commit it sits on, and how far \
         ahead and behind its upstream it is, and gives totals so \"how many\" needs no \
         counting. A path can carry two states — `staged` (last commit vs index) and \
         `unstaged` (index vs disk) — because a file can be both at once; a side with no \
         change is left out. Use for \"what have I changed\", \"is this repo clean\", \
         \"what branch am I on\", \"how far ahead am I\", or to see what would go into the \
         next commit. `page` is required — pass 0 to start, and `paging.next_page` for the \
         next. Reads only.";

    type Request = StatusRequest;
    type Response = StatusResponse;
    type Error = GitToolError;

    /// Reads the working tree; writes nothing.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: StatusRequest) -> Result<StatusResponse, GitToolError> {
        let repo = open(ctx, &req.repo)?;
        let head = repo.head()?;
        let entries = repo.status()?;
        let all: Vec<WireStatus> = entries.iter().map(Into::into).collect();

        let tracked = match head.branch() {
            Some(current) => repo
                .branches()?
                .into_iter()
                .find(|b| &b.name == current)
                .and_then(|b| b.upstream),
            None => None,
        };
        let no_upstream = (head.branch().is_some() && tracked.is_none()).then_some(
            "this branch tracks no upstream; to count how far ahead of a remote it is, use \
             git_log with `since` {\"kind\":\"remote_branch\",\"name\":\"origin/main\"} and \
             read its `count`",
        );

        let counts = Counts {
            staged: all.iter().filter(|c| c.staged.is_some()).count(),
            unstaged: all.iter().filter(|c| c.unstaged.is_some()).count(),
            untracked: all.iter().filter(|c| c.state == "untracked").count(),
            unmerged: all.iter().filter(|c| c.state == "unmerged").count(),
        };
        let paging = Paging::of(all.len(), req.page, PER_PAGE);
        let changes = all
            .into_iter()
            .skip(paging.skipped())
            .take(paging.per_page)
            .collect();

        Ok(StatusResponse {
            repo: req.repo,
            branch: head.branch().map(|b| b.as_str().to_string()),
            head: head.oid().map(|o| o.as_str().to_string()),
            upstream: tracked.as_ref().map(Into::into),
            no_upstream,
            clean: paging.total == 0,
            counts,
            paging,
            changes,
        })
    }
}

pub const GIT_STATUS: RegisteredTool = RegisteredTool::new::<GitStatus>();
