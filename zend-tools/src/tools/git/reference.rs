//! git_ref tool — create, move or delete a branch or a tag, on origin.
//!
//! One tool for both kinds, because they are the same operation on the same
//! kind of pointer and the difference is an enum the grammar decides. Every
//! change is made on origin first, under a lease on what origin held, and
//! locally once origin has it ([`zend_vfs::origin`]).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_vfs::{
    BranchName, GitError, Oid, Pulled, PushOutcome, PushSpec, RefName, Rejection, Repo as GitRepo,
    Rev, TagAnnotation, TagName,
};

use super::wire::refusal;
use super::{landed, open, ConvRepo, GitToolError, RevArg};
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
    /// Where it points, for `create` and `move`. Defaults to the branch you
    /// are on. Optional, never `null` — see [`RevArg`].
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
    /// Where the change is kept: `origin`, or `local` for a repository with
    /// no origin.
    pub on: &'static str,
}

/// `held`, checked against what the caller expected, when it said. Reading
/// the value here is what lets `expected` be optional without giving up the
/// compare-and-swap: the swap still names a value, it is simply one the layer
/// established rather than one the model had to be holding.
fn checked(name: &RefName, held: Oid, expected: &Option<String>) -> Result<Oid, GitToolError> {
    if let Some(e) = expected {
        let e = Oid::parse(e)?;
        if e != held {
            return Err(GitError::StaleRef {
                name: name.clone(),
                detail: format!("expected {e}, but it holds {held}"),
            }
            .into());
        }
    }
    Ok(held)
}

/// What a ref holds locally, checked against `expected`.
fn current(repo: &GitRepo, name: &RefName, expected: &Option<String>) -> Result<Oid, GitToolError> {
    let held = repo
        .ref_target(name)?
        .ok_or_else(|| GitError::invalid(format!("no ref named {name}")))?;
    checked(name, held, expected)
}

/// A branch as its record holds it — origin's copy, or with no origin the
/// local branch — or, for a branch only ever made here, the local branch;
/// an error when it is nowhere.
fn existing(pulled: &Pulled) -> Result<Oid, GitToolError> {
    pulled
        .record()
        .or(pulled.tip.as_ref())
        .cloned()
        .ok_or_else(|| GitError::invalid(format!("no branch named {}", pulled.branch)).into())
}

/// Push `spec` for a tag to origin, when there is one: where the tag is then
/// kept, or why origin refused it.
fn push_tag(
    repo: &ConvRepo,
    spec: PushSpec,
) -> Result<Result<&'static str, Rejection>, GitToolError> {
    let Some(origin) = repo.origin()? else {
        return Ok(Ok("local"));
    };
    let outcome = repo
        .push(&origin, &[spec])?
        .into_iter()
        .next()
        .map(|r| r.outcome)
        .ok_or_else(|| GitError::invalid("origin reported nothing for the tag"))?;
    Ok(match outcome {
        PushOutcome::Rejected(why) => Err(why),
        _ => Ok("origin"),
    })
}

pub struct GitRef;

impl Tool for GitRef {
    const NAME: &'static str = "git_ref";
    const DESCRIPTION: &'static str =
        "Create, move or delete a branch or a tag, on origin. `kind` picks which and \
         `action` picks the operation. `create` takes `at`, the revision it starts from \
         (defaulting to the branch you are on); a tag created with a `message` is \
         annotated, recording who tagged it and when. `move` repoints a branch other than \
         the one you are on (git_reset moves that one) — a move that would take commits off \
         it needs `expected`, naming the tip it takes them from. `delete` removes either — \
         never the branch you are on. `expected` is optional throughout: omit it and the \
         ref's current value is read — a branch's from origin, a tag's from this repository \
         — and swapped atomically, or give it to refuse the change if the ref moved since \
         you looked. This moves a \
         pointer and does not switch your branch — git_switch does that. Use for \"make a \
         branch for this\", \"tag this as v2.0\", \"delete the merged branch\", \"remove \
         that bad tag\". Writes to origin.";

    type Request = RefRequest;
    type Response = RefResponse;
    type Error = GitToolError;

    fn run(ctx: &ToolContext, req: RefRequest) -> Result<RefResponse, GitToolError> {
        let repo = open(ctx, &req.repo)?;
        let at = || -> Result<Oid, GitToolError> {
            let rev = match &req.at {
                Some(r) => r.resolve(&repo)?,
                None => repo.head()?,
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
            on: "",
        };

        match (req.kind, req.action) {
            (RefTarget::Branch, RefAction::Create) => {
                let branch = BranchName::parse(&req.name)?;
                let at = at()?;
                let pulled = repo.pull_branch(&branch)?;
                if pulled.tip.is_some() || pulled.on_origin.is_some() {
                    return Err(GitError::invalid(format!(
                        "a branch named {branch} already exists; move it, or pick another name"
                    ))
                    .into());
                }
                out.on = landed(&branch.to_ref(), repo.publish_branch(&pulled, &at)?)?;
                out.action = "created";
                out.id = Some(at.as_str().to_string());
                out.target = Some(at.as_str().to_string());
            }
            (RefTarget::Branch, RefAction::Move) => {
                let branch = BranchName::parse(&req.name)?;
                // Your files are based on the branch you are on: moving it
                // under them would leave them on a commit the branch no
                // longer holds, and your next commit would put it back.
                if repo.branch().as_ref() == Some(&branch) {
                    return Err(GitError::invalid(format!(
                        "you are on {branch}; git_reset moves it, and decides what becomes of \
                         your files as it does"
                    ))
                    .into());
                }
                let pulled = repo.pull_branch(&branch)?;
                let old = checked(&branch.to_ref(), existing(&pulled)?, &req.expected)?;
                let new = at()?;
                // A move that takes commits off the branch is made only on
                // purpose: `expected` names the tip whose commits go.
                let keeps_all = repo.is_ancestor(&Rev::Oid(old.clone()), &Rev::Oid(new.clone()))?;
                if !keeps_all && req.expected.is_none() {
                    return Err(GitError::invalid(format!(
                        "moving {branch} to {new} would take commits off it — {old}, what it \
                         holds now, is not in {new}'s history. If that is what you mean, give \
                         `expected` as {old}; to keep them, merge instead"
                    ))
                    .into());
                }
                out.on = landed(&branch.to_ref(), repo.publish_branch(&pulled, &new)?)?;
                out.action = "moved";
                out.id = Some(new.as_str().to_string());
                out.target = Some(new.as_str().to_string());
                out.previous = Some(old.as_str().to_string());
            }
            (RefTarget::Branch, RefAction::Delete) => {
                let branch = BranchName::parse(&req.name)?;
                if repo.branch().as_ref() == Some(&branch) {
                    return Err(GitError::invalid(format!(
                        "you are on {branch}; git_switch to another branch before deleting it"
                    ))
                    .into());
                }
                let pulled = repo.pull_branch(&branch)?;
                let old = checked(&branch.to_ref(), existing(&pulled)?, &req.expected)?;
                out.on = landed(&branch.to_ref(), repo.unpublish_branch(&pulled)?)?;
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
                // Origin not taking it — refused, or never reached — means it
                // is not kept here either.
                match push_tag(&repo, PushSpec::tag(id.clone(), tag.clone())) {
                    Ok(Ok(on)) => out.on = on,
                    Ok(Err(why)) => {
                        repo.delete_tag(&tag, &id)?;
                        return Err(GitError::invalid(refusal(&why)).into());
                    }
                    Err(e) => {
                        repo.delete_tag(&tag, &id)?;
                        return Err(e);
                    }
                }
                out.action = "created";
                out.id = Some(id.as_str().to_string());
                out.target = Some(target.as_str().to_string());
            }
            (RefTarget::Tag, RefAction::Delete) => {
                let tag = TagName::parse(&req.name)?;
                let old = current(&repo, &tag.to_ref(), &req.expected)?;
                match push_tag(&repo, PushSpec::delete_tag(tag.clone(), old.clone()))? {
                    Ok(on) => out.on = on,
                    Err(why) => return Err(GitError::invalid(refusal(&why)).into()),
                }
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
