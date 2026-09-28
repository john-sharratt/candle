//! git_push tool — publish branches and tags, under a lease.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_vfs::{
    BranchName, GitError, Lease, Oid, PushOutcome, PushSpec, PushTarget, Rejection, RemoteName,
    Repo as GitRepo, Rev, TagName,
};

use super::{open, GitToolError, RevArg};
use crate::tool::ConfirmationDetails;
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum PushAct {
    /// Set a remote branch to a revision.
    UpdateBranch,
    /// Delete a branch on the remote.
    DeleteBranch,
    /// Publish a tag. It must not already exist on the remote.
    UpdateTag,
    /// Delete a tag on the remote.
    DeleteTag,
}

/// One ref to change on the remote.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct PushItem {
    pub action: PushAct,
    /// The ref's name on the remote — a branch name or a tag name to match
    /// `action`. Required.
    pub name: String,
    /// What to publish, for the update actions. Defaults to the local branch
    /// or tag of the same name. Optional, never `null` — see [`RevArg`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "RevArg")]
    pub source: Option<RevArg>,
    /// What the remote ref must currently hold.
    ///
    /// Optional, and omitting it is the ordinary case: the lease then comes
    /// from this repository's remote-tracking ref, which is what
    /// `--force-with-lease` does by default and is the value that actually
    /// protects a concurrent push. Set `new: true` instead to require that
    /// the ref does not exist on the remote yet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub expected: Option<String>,
    /// Require that the ref does not exist on the remote yet — publishing
    /// something for the first time.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "bool")]
    pub new: Option<bool>,
}

#[derive(Deserialize, JsonSchema, Validate)]
#[serde(deny_unknown_fields)]
pub struct PushRequest {
    /// The repository to push from. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// The remote to push to, as git_refs reports it. Required.
    #[validate(length(min = 1))]
    pub remote: String,
    /// The refs to change, at least one. They go in one atomic push: either
    /// every one is accepted or none is.
    #[validate(length(min = 1, max = 50))]
    pub pushes: Vec<PushItem>,
}

#[derive(Serialize)]
pub struct WirePushResult {
    /// `branch` or `tag`.
    pub kind: &'static str,
    pub name: String,
    /// `created`, `fast_forward`, `forced`, `deleted`, `up_to_date` or
    /// `rejected`.
    pub outcome: &'static str,
    pub accepted: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Serialize)]
pub struct PushResponse {
    pub repo: String,
    pub remote: String,
    pub results: Vec<WirePushResult>,
    /// Every ref was accepted. A push is atomic, so false means nothing
    /// changed on the remote at all.
    pub accepted: bool,
}

fn outcome(o: &PushOutcome) -> (&'static str, Option<String>) {
    match o {
        PushOutcome::Created => ("created", None),
        PushOutcome::FastForward => ("fast_forward", None),
        PushOutcome::Forced => ("forced", None),
        PushOutcome::Deleted => ("deleted", None),
        PushOutcome::UpToDate => ("up_to_date", None),
        PushOutcome::Rejected(r) => (
            "rejected",
            Some(match r {
                Rejection::Stale => {
                    "the remote is not where the lease expected; run git_fetch and look again"
                        .to_string()
                }
                Rejection::AtomicAborted => {
                    "another ref in the same atomic push was rejected".to_string()
                }
                Rejection::Remote(detail) => format!("the remote refused it: {detail}"),
                Rejection::Other(detail) => detail.clone(),
            }),
        ),
    }
}

/// The lease for a ref the caller did not pin: what this repository last saw
/// the remote holding. Absent tracking ref means the branch is new there.
fn lease_from_tracking(
    repo: &GitRepo,
    remote: &RemoteName,
    branch: &BranchName,
) -> Result<Lease, GitToolError> {
    Ok(match repo.ref_target(&remote.tracking(branch))? {
        Some(oid) => Lease::Expect(oid),
        None => Lease::Absent,
    })
}

/// `update_branch` naming a tag of this repository that is no branch here or
/// on the remote is refused towards `update_tag`. Measured live: asked to push
/// the tag `v0.1.0`, a model sent `update_branch` with the tag as its source,
/// and origin gained a branch `v0.1.0` beside the tag — a name that then
/// means two things to every git command that reads it.
fn refuse_a_tag_as_a_branch(
    repo: &GitRepo,
    remote: &RemoteName,
    branch: &BranchName,
) -> Result<(), GitToolError> {
    let Ok(tag) = TagName::parse(branch.as_str()) else {
        return Ok(());
    };
    let is_tag = repo.ref_target(&tag.to_ref())?.is_some();
    let is_branch = repo.ref_target(&branch.to_ref())?.is_some()
        || repo.ref_target(&remote.tracking(branch))?.is_some();
    if is_tag && !is_branch {
        return Err(GitError::invalid(format!(
            "{branch} is a tag here, not a branch: publish it with `update_tag`. A branch of \
             the same name would make {branch} mean two things"
        ))
        .into());
    }
    Ok(())
}

pub struct GitPush;

impl Tool for GitPush {
    const NAME: &'static str = "git_push";
    const DESCRIPTION: &'static str =
        "Publish branches or tags to a remote. Every change carries a lease, so a push can \
         never silently discard work someone else pushed in between; a lease that no \
         longer matches is rejected and nothing changes. **You do not normally supply the \
         lease** — omit `expected` and it comes from this repository's remote-tracking \
         ref, which is the value that actually protects a concurrent push. Set `new: true` \
         to require the ref not exist on the remote yet. Several refs go in one atomic \
         push: all accepted or none. Run git_fetch first so the tracking refs reflect the \
         remote. Use for \"push this branch\", \"publish the tag\", \"delete that branch on \
         the remote\". This is the one git tool whose effect leaves this machine.";

    type Request = PushRequest;
    type Response = PushResponse;
    type Error = GitToolError;

    fn confirmation(req: &Self::Request) -> Option<ConfirmationDetails> {
        let refs = req
            .pushes
            .iter()
            .map(|p| match p.action {
                PushAct::UpdateBranch => format!("update {}", p.name),
                PushAct::DeleteBranch => format!("DELETE branch {}", p.name),
                PushAct::UpdateTag => format!("publish tag {}", p.name),
                PushAct::DeleteTag => format!("DELETE tag {}", p.name),
            })
            .collect::<Vec<_>>()
            .join(", ");
        Some(
            ConfirmationDetails::new(format!("Push to {} in {}: {refs}", req.remote, req.repo))
                .with_field("repository", req.repo.clone())
                .with_field("remote", req.remote.clone())
                .with_field("refs", refs),
        )
    }

    fn run(ctx: &ToolContext, req: PushRequest) -> Result<PushResponse, GitToolError> {
        let repo = open(ctx, &req.repo)?;
        let remote = RemoteName::parse(&req.remote)?;

        let mut specs = Vec::with_capacity(req.pushes.len());
        for item in &req.pushes {
            let lease = |repo: &GitRepo, branch: &BranchName| -> Result<Lease, GitToolError> {
                if item.new.unwrap_or(false) {
                    return Ok(Lease::Absent);
                }
                match &item.expected {
                    Some(e) => Ok(Lease::Expect(Oid::parse(e)?)),
                    None => lease_from_tracking(repo, &remote, branch),
                }
            };
            specs.push(match item.action {
                PushAct::UpdateBranch => {
                    let branch = BranchName::parse(&item.name)?;
                    refuse_a_tag_as_a_branch(&repo, &remote, &branch)?;
                    let rev = match &item.source {
                        Some(r) => r.resolve(&repo)?,
                        None => Rev::Branch(branch.clone()),
                    };
                    let source = repo.resolve(&rev)?;
                    PushSpec::branch(source, branch.clone(), lease(&repo, &branch)?)
                }
                // A tag is published as the object it names, unpeeled, so an
                // annotated tag reaches the remote with its message and tagger.
                PushAct::UpdateTag => {
                    let tag = TagName::parse(&item.name)?;
                    let rev = match &item.source {
                        Some(r) => r.resolve(&repo)?,
                        None => Rev::Tag(tag.clone()),
                    };
                    PushSpec::tag(repo.resolve_object(&rev)?, tag)
                }
                PushAct::DeleteBranch => {
                    let branch = BranchName::parse(&item.name)?;
                    let held = match &item.expected {
                        Some(e) => Oid::parse(e)?,
                        None => match lease_from_tracking(&repo, &remote, &branch)? {
                            Lease::Expect(oid) => oid,
                            Lease::Absent => {
                                return Err(GitError::invalid(format!(
                                    "nothing is known about {} on {}: run git_fetch, or \
                                     name what it holds in `expected`",
                                    item.name, req.remote
                                ))
                                .into())
                            }
                        },
                    };
                    PushSpec::delete_branch(branch, held)
                }
                PushAct::DeleteTag => {
                    let tag = TagName::parse(&item.name)?;
                    let held = match &item.expected {
                        Some(e) => Oid::parse(e)?,
                        None => repo.resolve_object(&Rev::Tag(tag.clone()))?,
                    };
                    PushSpec::delete_tag(tag, held)
                }
            });
        }

        let results = repo.push(&remote, &specs)?;
        let accepted = results.iter().all(|r| r.accepted());
        Ok(PushResponse {
            repo: req.repo,
            remote: req.remote,
            accepted,
            results: results
                .iter()
                .map(|r| {
                    let (outcome, reason) = outcome(&r.outcome);
                    let (kind, name) = match &r.target {
                        PushTarget::Branch(b) => ("branch", b.as_str().to_string()),
                        PushTarget::Tag(t) => ("tag", t.as_str().to_string()),
                    };
                    WirePushResult {
                        kind,
                        name,
                        outcome,
                        accepted: r.accepted(),
                        reason,
                    }
                })
                .collect(),
        })
    }
}

pub const GIT_PUSH: RegisteredTool = RegisteredTool::new::<GitPush>();
