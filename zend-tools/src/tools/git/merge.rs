//! git_merge tool — bring commits into this conversation's own files.
//!
//! The way on from a refused commit. By default what comes in is the branch
//! the conversation is on, as origin now holds it: the commits other writers
//! pushed since this conversation's base. They are merged into the
//! conversation's own copy of the repository, never into a folder and never
//! onto origin — `git merge` into a working tree, with the conversation's
//! files as the working tree ([`merge_into`]). Where both changed the same
//! lines, both are kept between conflict markers and the file is flagged
//! until the conversation settles it; the next git_commit lands on top of
//! what came in, recording the merge when there was history on both sides.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_vfs::{merge_into, GitError, Merged};

use super::{open, GitToolError, RevArg};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
#[serde(deny_unknown_fields)]
pub struct MergeRequest {
    /// The repository. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// What to merge in. Leave it out for your branch as origin holds it now
    /// — what a refused commit is waiting for. Optional, never `null` — see
    /// [`RevArg`].
    #[serde(default)]
    #[schemars(with = "RevArg")]
    pub from: Option<RevArg>,
}

#[derive(Serialize)]
pub struct MergeResponse {
    pub repo: String,
    pub branch: String,
    /// The commit merged in.
    pub from: String,
    /// `up_to_date` — you already had it; `fast_forward` — your files now
    /// read it, with your uncommitted changes carried onto it; `merging` —
    /// both had commits of their own, and your next commit records the merge.
    pub merged: &'static str,
    /// Files left in conflict, for you to settle before committing.
    pub conflicts: Vec<String>,
    /// What to do next.
    pub next: &'static str,
}

pub struct GitMerge;

impl Tool for GitMerge {
    const NAME: &'static str = "git_merge";
    const DESCRIPTION: &'static str =
        "Bring other commits into your files in a repository — by default, what others \
         pushed to your branch on origin since you started, which is what a refused \
         git_commit is waiting for. The merge happens in your own files and nowhere else: \
         where both sides changed the same lines, both are kept between <<<<<<< and >>>>>>> \
         markers and the file is listed in `conflicts` until you settle it by writing it as \
         it should be. Then git_commit publishes the result; a new merge waits until every \
         conflict is settled. `from` merges another branch or commit instead. Use for \"pull in the latest\", \"merge main into this\", \"my \
         commit was refused\". Reads origin; writes only your files.";

    type Request = MergeRequest;
    type Response = MergeResponse;
    type Error = GitToolError;

    fn run(ctx: &ToolContext, req: MergeRequest) -> Result<MergeResponse, GitToolError> {
        let repo = open(ctx, &req.repo)?;
        let branch = repo.require_branch()?;
        let store = repo.store().cloned().ok_or_else(|| {
            GitError::invalid("this session holds no files for this repository to merge into")
        })?;
        // A file still holding a merge's markers would be merged again as
        // though the markers were your content.
        let conflicts = store.conflicts();
        if !conflicts.is_empty() {
            return Err(GitError::invalid(format!(
                "these files are still in conflict from the last merge: {}. Settle each — \
                 write it as it should be, without the markers — before merging again",
                conflicts.join(", ")
            ))
            .into());
        }
        let (theirs, label) = match &req.from {
            Some(from) => {
                let rev = from.resolve(&repo)?;
                let label = from.name.clone().unwrap_or_else(|| "theirs".to_string());
                (repo.resolve(&rev)?, label)
            }
            None => {
                let pulled = repo.pull_branch(&branch)?;
                let label = match &pulled.origin {
                    Some(origin) => format!("{origin}/{branch}"),
                    None => branch.to_string(),
                };
                let theirs = pulled
                    .record()
                    .cloned()
                    .ok_or_else(|| GitError::invalid(format!("{branch} has no commit to merge")))?;
                (theirs, label)
            }
        };
        let merged = merge_into(&repo, &store, &theirs, &label)?;
        let (state, next) = match &merged {
            Merged::UpToDate => ("up_to_date", "nothing came in; commit as you were"),
            Merged::FastForward { conflicts } if conflicts.is_empty() => (
                "fast_forward",
                "your files now include what came in; git_commit when ready",
            ),
            Merged::Merging { conflicts } if conflicts.is_empty() => (
                "merging",
                "merged cleanly; git_commit with `from: changes` records the merge",
            ),
            Merged::FastForward { .. } | Merged::Merging { .. } => (
                if matches!(merged, Merged::Merging { .. }) {
                    "merging"
                } else {
                    "fast_forward"
                },
                "settle each file in `conflicts` — read it, write it as it should be without \
                 the markers — then git_commit",
            ),
        };
        Ok(MergeResponse {
            repo: req.repo,
            branch: branch.to_string(),
            from: theirs.to_string(),
            merged: state,
            conflicts: merged.conflicts().to_vec(),
            next,
        })
    }
}

pub const GIT_MERGE: RegisteredTool = RegisteredTool::new::<GitMerge>();
