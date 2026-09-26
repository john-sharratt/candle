//! git_grep tool — search what a revision holds.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_git::GrepQuery;

use super::wire::WireGrepHit;
use super::{is_protected, open, path_args, rev_or_head, GitToolError, RevArg};
use crate::tools::file::Paging;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// A match is its path, line number and the line itself, ~40 tokens for
/// ordinary source — so 40 keeps a page to what a `file_read` page costs.
const PER_PAGE: usize = 40;

#[derive(Deserialize, JsonSchema, Validate)]
#[serde(deny_unknown_fields)]
pub struct GrepRequest {
    /// The repository to search. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// What to look for: a POSIX extended regular expression, or a literal
    /// string when `fixed` is set. Required.
    #[validate(length(min = 1))]
    pub pattern: String,
    /// The revision to search. Defaults to the checked-out `HEAD` — the last
    /// commit, not the files on disk. Optional, never `null` — see [`RevArg`].
    #[serde(default)]
    #[schemars(with = "RevArg")]
    pub rev: Option<RevArg>,
    /// Treat `pattern` as a literal string. Defaults to false.
    #[serde(default)]
    #[schemars(with = "bool")]
    pub fixed: Option<bool>,
    /// Match without regard to case. Defaults to false.
    #[serde(default)]
    #[schemars(with = "bool")]
    pub ignore_case: Option<bool>,
    /// Only search these repository-relative files or directories.
    #[serde(default)]
    #[schemars(with = "Vec<String>")]
    pub paths: Option<Vec<String>>,
    /// At most this many matches from any one file (1–100). Defaults to 20,
    /// so one generated file cannot crowd out every other hit.
    #[validate(range(min = 1, max = 100))]
    #[serde(default)]
    #[schemars(with = "u32")]
    pub max_per_file: Option<u32>,
    /// Zero-based page of matches, 40 a page. Required — pass 0 to start. A
    /// page past the end clamps to the last one.
    pub page: u32,
}

#[derive(Serialize)]
pub struct GrepResponse {
    pub repo: String,
    /// The commit the revision resolved to.
    pub commit: String,
    /// Total matches, and how many distinct files held them — read rather
    /// than counted.
    pub count: usize,
    pub files_matched: usize,
    pub paging: Paging,
    pub matches: Vec<WireGrepHit>,
}

pub struct GitGrep;

impl Tool for GitGrep {
    const NAME: &'static str = "git_grep";
    const DESCRIPTION: &'static str =
        "**Search what a COMMIT, BRANCH or TAG holds** — the committed content, not the \
         files on disk. This is the tool for \"search the committed files\", \"was this in \
         the release branch\", \"did that tag still have it\", \"what did this look like \
         before\". Searches every file the revision tracks without checking it out, \
         returning each matching file, line number and line, with a `count` of the total. \
         `pattern` is a POSIX extended regular expression, or a literal with `fixed`; \
         `rev` defaults to HEAD, the last commit. Binary files are skipped. **To search \
         the working tree as it stands right now, uncommitted edits and untracked files \
         included, use file_grep instead** — an untracked or since-edited file reads \
         differently through the two. `page` is required — pass 0 to start, and \
         `paging.next_page` for the next. Reads only.";

    type Request = GrepRequest;
    type Response = GrepResponse;
    type Error = GitToolError;

    /// Searches objects; writes nothing.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: GrepRequest) -> Result<GrepResponse, GitToolError> {
        let repo = open(ctx, &req.repo)?;
        let rev = rev_or_head(&req.rev, &repo)?;
        let commit = repo.resolve(&rev)?.as_str().to_string();
        let query = GrepQuery {
            pattern: req.pattern,
            fixed: req.fixed.unwrap_or(false),
            ignore_case: req.ignore_case.unwrap_or(false),
            paths: path_args(req.paths.as_deref().unwrap_or(&[]))?,
            max_per_file: Some(req.max_per_file.unwrap_or(20)),
        };
        // A search with no paths sweeps every tracked file, so a protected
        // one's matching LINES would come back unasked for.
        let hits: Vec<_> = repo
            .grep(&rev, &query)?
            .into_iter()
            .filter(|h| !is_protected(&h.path))
            .collect();

        let files_matched = hits
            .iter()
            .map(|h| h.path.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        let paging = Paging::of(hits.len(), req.page, PER_PAGE);
        Ok(GrepResponse {
            repo: req.repo,
            commit,
            count: hits.len(),
            files_matched,
            matches: hits
                .iter()
                .skip(paging.skipped())
                .take(paging.per_page)
                .map(Into::into)
                .collect(),
            paging,
        })
    }
}

pub const GIT_GREP: RegisteredTool = RegisteredTool::new::<GitGrep>();
