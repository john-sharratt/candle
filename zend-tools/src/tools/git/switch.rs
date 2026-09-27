//! git_switch tool — put this conversation on another branch.
//!
//! A conversation is on one branch per repository, and reads the repository
//! through it at its own base. Switching changes which branch that is, and
//! puts the base at the branch's commit — nothing on disk moves, since the
//! repository's folder is the sandbox's — and the conversation's uncommitted
//! changes come along, merged three ways onto the other branch's files: where
//! that branch changed the same lines, both are kept between conflict markers
//! for the conversation to settle. Making the branch first makes it on
//! origin. A merge being finished stays on its branch: it is committed or
//! discarded before a switch.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_vfs::vfs::Carried;
use zend_vfs::work::ThreeWay;
use zend_vfs::{Base, BranchName, GitError, MergeLabels, Rev};

use super::{landed, open, GitToolError, RevArg};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Deserialize, JsonSchema, Validate)]
#[serde(deny_unknown_fields)]
pub struct SwitchRequest {
    /// The repository. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// The branch to switch to, by short name. Required.
    #[validate(length(min = 1))]
    pub branch: String,
    /// Make the branch first, on origin. Defaults to false.
    #[serde(default)]
    #[schemars(with = "bool")]
    pub create: Option<bool>,
    /// For `create`: where the new branch starts. Defaults to the branch you
    /// are on. Optional, never `null` — see [`RevArg`].
    #[serde(default)]
    #[schemars(with = "RevArg")]
    pub from: Option<RevArg>,
}

#[derive(Serialize)]
pub struct SwitchResponse {
    pub repo: String,
    /// The branch you are on now.
    pub branch: String,
    /// The branch you were on.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous: Option<String>,
    /// The commit the branch holds.
    pub id: String,
    /// Whether the branch was made by this call.
    pub created: bool,
    /// Where a branch this call made is kept: `origin`, or `local` for a
    /// repository with no origin.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub on: Option<&'static str>,
    /// Uncommitted changes you brought with you.
    pub carried: usize,
    /// Files you changed that this branch changed too, on the same lines:
    /// both are kept between conflict markers, for you to settle.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub conflicts: Vec<String>,
}

pub struct GitSwitch;

impl Tool for GitSwitch {
    const NAME: &'static str = "git_switch";
    const DESCRIPTION: &'static str =
        "Switch to another branch in a repository, taking your uncommitted changes with \
         you — merged onto that branch's files, and where it changed the same lines, both \
         kept between markers and listed in `conflicts` to settle: from then on you read, \
         change and commit on that branch. A merge being finished must be committed or \
         discarded first, and files in conflict settled. The branch is brought up to date with origin first, so a branch \
         that only exists on origin works too. With `create`, the branch is made — on \
         origin — starting at `from`, which defaults to your own commit. Use for \"switch \
         to main\", \"start a branch for this\", \"work on feature/x\", \"go back to main\". \
         To make a branch without switching to it, use git_ref. Writes to origin when it \
         makes the branch.";

    type Request = SwitchRequest;
    type Response = SwitchResponse;
    type Error = GitToolError;

    fn run(ctx: &ToolContext, req: SwitchRequest) -> Result<SwitchResponse, GitToolError> {
        let repo = open(ctx, &req.repo)?;
        let branch = BranchName::parse(&req.branch)?;
        let store = repo.store().cloned().ok_or_else(|| {
            GitError::invalid("this session holds no files for this repository to switch")
        })?;
        let previous = repo.branch();
        let merging = store
            .base()
            .ok()
            .flatten()
            .and_then(|b| b.merging().cloned());
        if let Some(merging) = merging {
            return Err(GitError::invalid(format!(
                "a merge of {merging} is being finished on this branch: settle and commit it, \
                 or discard it with a hard git_reset, before switching"
            ))
            .into());
        }
        let conflicts = store.conflicts();
        if !conflicts.is_empty() {
            return Err(GitError::invalid(format!(
                "these files are still in conflict from a merge: {}. Settle each — write it \
                 as it should be, without the markers — or discard your changes with a hard \
                 git_reset, before switching",
                conflicts.join(", ")
            ))
            .into());
        }
        let start = match (&req.from, req.create.unwrap_or(false)) {
            (Some(from), true) => Some(repo.resolve(&from.resolve(&repo)?)?),
            (None, true) => Some(repo.resolve(&repo.head()?)?),
            (Some(_), false) => {
                return Err(GitError::invalid(
                    "`from` is where a new branch starts; set `create` to make one",
                )
                .into())
            }
            (None, false) => None,
        };
        let pulled = repo.pull_branch(&branch)?;
        let (id, on) = match start {
            Some(start) => {
                if pulled.tip.is_some() || pulled.on_origin.is_some() {
                    return Err(GitError::invalid(format!(
                        "a branch named {branch} already exists; switch to it without `create`"
                    ))
                    .into());
                }
                let on = landed(&branch.to_ref(), repo.publish_branch(&pulled, &start)?)?;
                (start, Some(on))
            }
            None => {
                let tip = pulled.tip.ok_or_else(|| {
                    GitError::invalid(format!("no branch named {branch}; set `create` to make it"))
                })?;
                (tip, None)
            }
        };
        let tree = repo
            .blobs()
            .commit_of(&Rev::Oid(id.clone()))?
            .map(|(_, tree)| tree)
            .ok_or_else(|| GitError::invalid(format!("no commit {id} to switch to")))?;
        let was = previous
            .as_ref()
            .map_or_else(|| "base".to_string(), |b| b.to_string());
        let three = ThreeWay::new(
            &repo,
            MergeLabels {
                ours: "yours",
                base: &was,
                theirs: branch.as_str(),
            },
        );
        let conflicts = store
            .move_base(
                Some(branch.clone()),
                Base::at(id.clone(), tree),
                &[],
                &mut |c: &Carried<'_>| three.carried(c),
            )
            .map_err(|e| GitError::invalid(e.to_string()))?;
        Ok(SwitchResponse {
            repo: req.repo,
            branch: branch.as_str().to_string(),
            previous: previous.map(|b| b.as_str().to_string()),
            id: id.as_str().to_string(),
            created: on.is_some(),
            on,
            carried: store.status().len(),
            conflicts,
        })
    }
}

pub const GIT_SWITCH: RegisteredTool = RegisteredTool::new::<GitSwitch>();
