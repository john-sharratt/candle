//! git_fetch tool — update remote-tracking refs from a remote.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_vfs::{BranchName, FetchFlag, FetchSpec, RemoteName};

use super::{open, GitToolError};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
#[serde(deny_unknown_fields)]
pub struct FetchRequest {
    /// The repository to fetch into. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// The remote to fetch from, as git_refs names it — usually `origin`.
    #[validate(length(min = 1))]
    pub remote: String,
    /// Fetch only this branch. Omit to fetch every branch and prune the
    /// tracking refs of branches deleted on the remote.
    #[serde(default)]
    #[schemars(with = "String")]
    pub branch: Option<String>,
}

#[derive(Serialize)]
pub struct WireRefUpdate {
    /// The remote-tracking ref that changed.
    pub ref_name: String,
    /// `new`, `fast_forward`, `forced` or `pruned`.
    pub change: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub old: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub new: Option<String>,
}

#[derive(Serialize)]
pub struct FetchResponse {
    pub repo: String,
    pub remote: String,
    pub count: usize,
    pub updates: Vec<WireRefUpdate>,
    /// Nothing on the remote had moved since the last fetch.
    pub up_to_date: bool,
}

fn flag(f: FetchFlag) -> &'static str {
    match f {
        FetchFlag::New => "new",
        FetchFlag::FastForward => "fast_forward",
        FetchFlag::Forced => "forced",
        FetchFlag::Pruned => "pruned",
    }
}

pub struct GitFetch;

impl Tool for GitFetch {
    const NAME: &'static str = "git_fetch";
    const DESCRIPTION: &'static str =
        "Contact a remote and update this repository's remote-tracking refs, reporting \
         which moved and how — new, fast-forwarded, force-moved, or pruned because the \
         branch was deleted upstream. This only refreshes what is known about the remote: \
         it moves no local branch, merges nothing and changes no file on disk, so it is \
         the safe way to find out whether there is new work before deciding anything. It \
         is also what makes git_push's default lease correct, so run it first. Follow with \
         git_refs to see how far ahead or behind each branch now is. Use for \"check for \
         new commits\", \"fetch origin\", \"is there anything upstream\". Reaches the \
         network and writes to the repository.";

    type Request = FetchRequest;
    type Response = FetchResponse;
    type Error = GitToolError;

    fn run(ctx: &ToolContext, req: FetchRequest) -> Result<FetchResponse, GitToolError> {
        let repo = open(ctx, &req.repo)?;
        let remote = RemoteName::parse(&req.remote)?;
        let spec = match &req.branch {
            Some(b) => FetchSpec::Branch(BranchName::parse(b)?),
            None => FetchSpec::AllBranches,
        };
        let updates = repo.fetch(&remote, &spec)?;
        Ok(FetchResponse {
            repo: req.repo,
            remote: req.remote,
            up_to_date: updates.is_empty(),
            count: updates.len(),
            updates: updates
                .iter()
                .map(|u| WireRefUpdate {
                    ref_name: u.local.as_str().to_string(),
                    change: flag(u.flag),
                    old: u.old.as_ref().map(|o| o.as_str().to_string()),
                    new: u.new.as_ref().map(|o| o.as_str().to_string()),
                })
                .collect(),
        })
    }
}

pub const GIT_FETCH: RegisteredTool = RegisteredTool::new::<GitFetch>();
