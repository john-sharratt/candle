//! git_refs tool — what branches, tags and remotes a repository has.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;

use super::wire::{WireBranch, WireRemote, WireTag};
use super::{open, GitToolError};
use crate::tools::file::Paging;
use crate::{RegisteredTool, Replay, Tool, ToolContext};

/// A branch with its upstream is ~55 tokens, a remote-tracking branch ~40 —
/// 60 of them measured 3,221 and 2,273 — so 40 keeps a page to what a
/// `file_read` page costs.
const PER_PAGE: usize = 40;

/// Which kind of ref to list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RefKind {
    /// Local branches, with upstream tracking where configured.
    Branches,
    /// Tags, annotated and lightweight.
    Tags,
    /// Configured remotes and their URLs.
    Remotes,
    /// Remote-tracking branches, as of the last fetch, with the URLs of the
    /// remotes they came from.
    RemoteBranches,
}

#[derive(Deserialize, JsonSchema, Validate)]
#[serde(deny_unknown_fields)]
pub struct RefsRequest {
    /// The repository to read. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// Which refs to list. Required.
    pub kind: RefKind,
    /// Zero-based page, 40 refs a page. Required — pass 0 to start. A page
    /// past the end clamps to the last one.
    pub page: u32,
}

#[derive(Serialize)]
pub struct WireRemoteBranch {
    pub remote: String,
    pub branch: String,
    pub id: String,
}

#[derive(Serialize)]
pub struct RefsResponse {
    pub repo: String,
    pub kind: &'static str,
    /// How many refs of this kind exist, so "how many" is read, not counted.
    pub count: usize,
    /// The branch this conversation is on, when listing branches.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub current: Option<String>,
    pub paging: Paging,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub branches: Option<Vec<WireBranch>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tags: Option<Vec<WireTag>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remotes: Option<Vec<WireRemote>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub remote_branches: Option<Vec<WireRemoteBranch>>,
}

pub struct GitRefs;

impl Tool for GitRefs {
    const NAME: &'static str = "git_refs";
    const DESCRIPTION: &'static str =
        "List a repository's refs. `kind` picks which: `branches` gives local branches \
         with the commit each points at, which one you are on, and how far ahead or \
         behind its upstream it is; `tags` gives tags with the commit each finally points \
         at and whether it is annotated; `remotes` gives configured remotes and their URLs \
         (credentials redacted); `remote_branches` gives the remote-tracking branches from \
         the last fetch, and those remotes' URLs too. A `count` comes back with every \
         listing, so \"how many branches \
         are there\" is read rather than counted. Use for \"what branches exist\", \"am I \
         up to date with origin\", \"do I have unpushed commits\", \"what versions have \
         been released\", \"where does this push to\". Contacts no server. `page` is \
         required — pass 0 to start, and `paging.next_page` for the next. Reads only.";

    type Request = RefsRequest;
    type Response = RefsResponse;
    type Error = GitToolError;

    /// Lists refs; writes nothing.
    fn replay(_req: &Self::Request) -> Replay {
        Replay::Safe
    }

    fn run(ctx: &ToolContext, req: RefsRequest) -> Result<RefsResponse, GitToolError> {
        let repo = open(ctx, &req.repo)?;
        let page = req.page;
        let mut out = RefsResponse {
            repo: req.repo,
            kind: "",
            count: 0,
            current: None,
            paging: Paging::of(0, 0, PER_PAGE),
            branches: None,
            tags: None,
            remotes: None,
            remote_branches: None,
        };

        match req.kind {
            RefKind::Branches => {
                let current = repo.branch().map(|b| b.as_str().to_string());
                let all = repo.branches()?;
                let paging = Paging::of(all.len(), page, PER_PAGE);
                out.kind = "branches";
                out.count = all.len();
                out.branches = Some(
                    all.iter()
                        .skip(paging.skipped())
                        .take(paging.per_page)
                        .map(|b| {
                            let is_head = current.as_deref() == Some(b.name.as_str());
                            WireBranch::new(b, is_head)
                        })
                        .collect(),
                );
                out.current = current;
                out.paging = paging;
            }
            RefKind::Tags => {
                let all = repo.tags()?;
                let paging = Paging::of(all.len(), page, PER_PAGE);
                out.kind = "tags";
                out.count = all.len();
                out.tags = Some(
                    all.iter()
                        .skip(paging.skipped())
                        .take(paging.per_page)
                        .map(Into::into)
                        .collect(),
                );
                out.paging = paging;
            }
            RefKind::Remotes => {
                let all = repo.remotes()?;
                let paging = Paging::of(all.len(), page, PER_PAGE);
                out.kind = "remotes";
                out.count = all.len();
                out.remotes = Some(
                    all.iter()
                        .skip(paging.skipped())
                        .take(paging.per_page)
                        .map(Into::into)
                        .collect(),
                );
                out.paging = paging;
            }
            RefKind::RemoteBranches => {
                let all = repo.remote_branches()?;
                let paging = Paging::of(all.len(), page, PER_PAGE);
                out.kind = "remote_branches";
                out.count = all.len();
                // The remotes those branches came from, URLs included. Asked
                // "where does this push to", the model listed remote branches
                // rather than remotes, found no URL, and answered from a typo
                // in the README. A repository has a remote or two, so carrying
                // them here costs little and makes either kind the answer.
                out.remotes = Some(repo.remotes()?.iter().map(Into::into).collect());
                out.remote_branches = Some(
                    all.iter()
                        .skip(paging.skipped())
                        .take(paging.per_page)
                        .map(|rb| WireRemoteBranch {
                            remote: rb.remote.as_str().to_string(),
                            branch: rb.branch.as_str().to_string(),
                            id: rb.oid.as_str().to_string(),
                        })
                        .collect(),
                );
                out.paging = paging;
            }
        }
        Ok(out)
    }
}

pub const GIT_REFS: RegisteredTool = RegisteredTool::new::<GitRefs>();
