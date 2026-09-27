//! git_ref tool — create, move or delete a branch or a tag.
//!
//! One tool for both kinds, because they are the same operation on the same
//! kind of pointer and the difference is an enum the grammar decides.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_vfs::{BranchName, GitError, Oid, Repo as GitRepo, TagAnnotation, TagName};

use super::{open, GitToolError, RevArg};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RefTarget {
    Branch,
    Tag,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RefAction {
    /// Make a ref that does not exist yet.
    Create,
    /// Point an existing branch somewhere else. Branches only — a tag is
    /// deleted and remade rather than moved.
    Move,
    /// Remove the ref.
    Delete,
}

#[derive(Deserialize, JsonSchema, Validate)]
#[serde(deny_unknown_fields)]
pub struct RefRequest {
    /// The repository to change. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// Whether this is a branch or a tag. Required.
    pub kind: RefTarget,
    /// What to do. Required.
    pub action: RefAction,
    /// The ref's short name — `feature/login`, `v1.4.0`. Required.
    #[validate(length(min = 1))]
    pub name: String,
    /// Where it points, for `create` and `move`. Defaults to the checked-out
    /// `HEAD`. Optional, never `null` — see [`RevArg`].
    #[serde(default)]
    #[schemars(with = "RevArg")]
    pub at: Option<RevArg>,
    /// What the ref must currently hold, for `move` and `delete`. Omit and
    /// the layer reads it, still swapping atomically; give it to refuse the
    /// change if the ref moved since you looked.
    #[serde(default)]
    #[schemars(with = "String")]
    pub expected: Option<String>,
    /// A tag message. Giving one makes an annotated tag, which records who
    /// tagged it and when; without one the tag is a bare pointer.
    #[serde(default)]
    #[schemars(with = "String")]
    pub message: Option<String>,
}

#[derive(Serialize)]
pub struct RefResponse {
    pub repo: String,
    pub kind: &'static str,
    pub name: String,
    /// `created`, `moved` or `deleted`.
    pub action: &'static str,
    /// What the ref holds now; absent after a deletion.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// The commit it points at — differs from `id` for an annotated tag.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// What it held before; absent after a creation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous: Option<String>,
}

/// What the ref holds now, either as the caller stated it or as read here.
/// Reading it is what lets `expected` be optional without giving up the
/// compare-and-swap: the swap still names a value, it is simply one the layer
/// established rather than one the model had to be holding.
fn current(
    repo: &GitRepo,
    name: &zend_vfs::RefName,
    expected: &Option<String>,
) -> Result<Oid, GitToolError> {
    let held = repo
        .ref_target(name)?
        .ok_or_else(|| GitError::invalid(format!("no ref named {name}")))?;
    match expected {
        None => Ok(held),
        Some(e) => {
            let e = Oid::parse(e)?;
            if e != held {
                return Err(GitError::StaleRef {
                    name: name.clone(),
                    detail: format!("expected {e}, but it holds {held}"),
                }
                .into());
            }
            Ok(held)
        }
    }
}

pub struct GitRef;

impl Tool for GitRef {
    const NAME: &'static str = "git_ref";
    const DESCRIPTION: &'static str =
        "Create, move or delete a branch or a tag. `kind` picks which and `action` picks \
         the operation. `create` takes `at`, the revision it starts from (defaulting to \
         HEAD); a tag created with a `message` is annotated, recording who tagged it and \
         when. `move` repoints a branch. `delete` removes either. `expected` is optional \
         throughout: omit it and the ref's current value is read here and swapped \
         atomically, or give it to refuse the change if the ref moved since you looked. \
         **Nothing is checked out and no file on disk changes** — this moves a pointer, it \
         does not switch the working tree — and a branch checked out in any worktree is \
         refused. Use for \"make a branch for this\", \"tag this as v2.0\", \"delete the \
         merged branch\", \"remove that bad tag\". Writes to the repository.";

    type Request = RefRequest;
    type Response = RefResponse;
    type Error = GitToolError;

    fn run(ctx: &ToolContext, req: RefRequest) -> Result<RefResponse, GitToolError> {
        let repo = open(ctx, &req.repo)?;
        let at = || -> Result<Oid, GitToolError> {
            let rev = match &req.at {
                Some(r) => r.resolve(&repo)?,
                None => zend_vfs::Rev::Head,
            };
            Ok(repo.resolve(&rev)?)
        };
        let mut out = RefResponse {
            repo: req.repo.clone(),
            kind: match req.kind {
                RefTarget::Branch => "branch",
                RefTarget::Tag => "tag",
            },
            name: req.name.clone(),
            action: "",
            id: None,
            target: None,
            previous: None,
        };

        match (req.kind, req.action) {
            (RefTarget::Branch, RefAction::Create) => {
                let branch = BranchName::parse(&req.name)?;
                let at = at()?;
                repo.create_branch(&branch, &at)?;
                out.action = "created";
                out.id = Some(at.as_str().to_string());
                out.target = Some(at.as_str().to_string());
            }
            (RefTarget::Branch, RefAction::Move) => {
                let branch = BranchName::parse(&req.name)?;
                let old = current(&repo, &branch.to_ref(), &req.expected)?;
                let new = at()?;
                repo.move_branch(&branch, &old, &new)?;
                out.action = "moved";
                out.id = Some(new.as_str().to_string());
                out.target = Some(new.as_str().to_string());
                out.previous = Some(old.as_str().to_string());
            }
            (RefTarget::Branch, RefAction::Delete) => {
                let branch = BranchName::parse(&req.name)?;
                let old = current(&repo, &branch.to_ref(), &req.expected)?;
                repo.delete_branch(&branch, &old)?;
                out.action = "deleted";
                out.previous = Some(old.as_str().to_string());
            }
            (RefTarget::Tag, RefAction::Create) => {
                let tag = TagName::parse(&req.name)?;
                let target = at()?;
                let annotation = req
                    .message
                    .as_ref()
                    .filter(|m| !m.is_empty())
                    .map(|message| -> Result<TagAnnotation, GitToolError> {
                        Ok(TagAnnotation {
                            message: message.clone(),
                            tagger: repo.identity()?,
                        })
                    })
                    .transpose()?;
                let id = repo.create_tag(&tag, &target, annotation.as_ref())?;
                out.action = "created";
                out.id = Some(id.as_str().to_string());
                out.target = Some(target.as_str().to_string());
            }
            (RefTarget::Tag, RefAction::Delete) => {
                let tag = TagName::parse(&req.name)?;
                let old = current(&repo, &tag.to_ref(), &req.expected)?;
                repo.delete_tag(&tag, &old)?;
                out.action = "deleted";
                out.previous = Some(old.as_str().to_string());
            }
            (RefTarget::Tag, RefAction::Move) => {
                return Err(GitError::invalid(
                    "a tag is not moved: delete it and create it again at the new target",
                )
                .into())
            }
        }
        Ok(out)
    }
}

pub const GIT_REF: RegisteredTool = RegisteredTool::new::<GitRef>();
