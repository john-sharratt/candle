//! git_log tool.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_vfs::{GitError, LogRange};

use super::line_history::{line_history, LineRun, LineSpan};
use super::wire::WireCommit;
use super::{open, path_arg, path_args, rev_or_head, GitToolError, RevArg};
use crate::tools::file::Paging;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// About a hundred tokens a commit — two full ids, an author, a date and a
/// subject — so a page costs what a `file_read` page does.
const PER_PAGE: usize = 20;
/// How deep the walk goes before it stops counting. A repository's whole
/// history is not worth walking to answer "how many commits since".
const MAX_WALK: usize = 2000;

#[derive(Deserialize, JsonSchema, Validate)]
#[serde(deny_unknown_fields)]
pub struct LogRequest {
    /// The repository to read. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// Where to start walking back from. Defaults to the checked-out `HEAD`.
    /// Optional, never `null` — see [`RevArg`].
    #[serde(default)]
    #[schemars(with = "RevArg")]
    pub rev: Option<RevArg>,
    /// Stop at commits already reachable from this revision — how to ask
    /// "what is on my branch that main does not have". Omit for the whole
    /// history behind `rev`.
    #[serde(default)]
    #[schemars(with = "RevArg")]
    pub since: Option<RevArg>,
    /// Only commits that touched these repository-relative paths.
    #[serde(default)]
    #[schemars(with = "Vec<String>")]
    pub paths: Option<Vec<String>>,
    /// Follow one file back through its renames. Needs exactly one entry in
    /// `paths` and no `since`.
    #[serde(default)]
    #[schemars(with = "bool")]
    pub follow_renames: Option<bool>,
    /// Only the commits that last changed these lines of the one file in
    /// `paths` — who wrote them and when, rather than who last touched the
    /// file anywhere. 1-based and inclusive, at most 200 lines. Needs exactly
    /// one entry in `paths` and no `since`.
    #[serde(default)]
    #[schemars(with = "LineSpan")]
    pub lines: Option<LineSpan>,
    /// Zero-based page of commits, 20 a page. Required — pass 0 for the
    /// newest. A page past the end clamps to the last one.
    pub page: u32,
}

#[derive(Serialize)]
pub struct LogResponse {
    pub repo: String,
    /// How many commits the range holds, so "how many" is read rather than
    /// counted. `counted_all` says whether the walk reached the end.
    pub count: usize,
    pub counted_all: bool,
    pub paging: Paging,
    pub commits: Vec<WireCommit>,
    /// With `lines`: which commit last changed each run of those lines, in
    /// line order.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub line_runs: Vec<LineRun>,
}

pub struct GitLog;

impl Tool for GitLog {
    const NAME: &'static str = "git_log";
    const DESCRIPTION: &'static str =
        "Read a repository's commit history, newest first, 20 commits a page: each \
         commit's full object id, author, ISO date and subject line (a merge also lists \
         its parents as `merge_of`), plus a `count` of how \
         many commits the range holds. Narrow it with `rev` (start from a branch, tag or \
         commit instead of HEAD), `since` (exclude commits already reachable from another \
         revision — this is how to answer \"what is on this branch that main does not \
         have\" and \"how far ahead am I\"), and `paths`, with `follow_renames` to track \
         one file across renames, and `lines` to ask which commits last changed a span \
         of that one file's lines — who wrote them, not who last touched the file. Use \
         for \"what changed recently\", \
         \"when was this introduced\", or to get a commit id for git_show or git_commit; \
         a commit's full message is git_show's. `page` is required — pass 0 for the \
         newest, and `paging.next_page` for the next. Reads only.";

    type Request = LogRequest;
    type Response = LogResponse;
    type Error = GitToolError;

    /// Reads history; writes nothing.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: LogRequest) -> Result<LogResponse, GitToolError> {
        let repo = open(ctx, &req.repo)?;
        let to = rev_or_head(&req.rev, &repo)?;
        let paths = req.paths.as_deref().unwrap_or(&[]);

        if let Some(span) = req.lines {
            let [path] = paths else {
                return Err(GitError::invalid("lines needs exactly one entry in paths").into());
            };
            if req.since.is_some() || req.follow_renames.unwrap_or(false) {
                return Err(GitError::invalid(
                    "lines cannot be combined with since or follow_renames",
                )
                .into());
            }
            let history = line_history(&repo, &to, &path_arg(path)?, span)?;
            let paging = Paging::of(history.commits.len(), req.page, PER_PAGE);
            return Ok(LogResponse {
                repo: req.repo,
                count: history.commits.len(),
                counted_all: true,
                commits: history
                    .commits
                    .iter()
                    .skip(paging.skipped())
                    .take(paging.per_page)
                    .map(Into::into)
                    .collect(),
                paging,
                line_runs: history.runs,
            });
        }

        let all = if req.follow_renames.unwrap_or(false) {
            let [path] = paths else {
                return Err(
                    GitError::invalid("follow_renames needs exactly one entry in paths").into(),
                );
            };
            if req.since.is_some() {
                return Err(
                    GitError::invalid("follow_renames cannot be combined with since").into(),
                );
            }
            repo.file_history(&to, &path_arg(path)?, MAX_WALK)?
        } else {
            let range = LogRange {
                to,
                exclude: req.since.as_ref().map(|r| r.resolve(&repo)).transpose()?,
                paths: path_args(paths)?,
            };
            repo.log(&range, MAX_WALK)?
        };

        let paging = Paging::of(all.len(), req.page, PER_PAGE);
        Ok(LogResponse {
            repo: req.repo,
            count: all.len(),
            counted_all: all.len() < MAX_WALK,
            commits: all
                .iter()
                .skip(paging.skipped())
                .take(paging.per_page)
                .map(Into::into)
                .collect(),
            paging,
            line_runs: Vec::new(),
        })
    }
}

pub const GIT_LOG: RegisteredTool = RegisteredTool::new::<GitLog>();
