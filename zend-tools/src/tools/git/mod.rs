//! `git_*` tools — the typed git layer ([`zend_git`]) over the workspace's
//! repositories.
//!
//! # Nine tools, split by capability
//!
//! - **Readers** — [`GIT_STATUS`], [`GIT_LOG`], [`GIT_SHOW`], [`GIT_GREP`],
//!   [`GIT_REFS`] — answer questions and change nothing. They declare no
//!   capability.
//! - **Writers** — [`GIT_COMMIT`], [`GIT_REF`], [`GIT_FETCH`], [`GIT_PUSH`] —
//!   declare [`Capability::DiskWrite`](crate::grants::Capability::DiskWrite),
//!   which the Comprehensive tools mode's grants withhold. Changing a
//!   repository is Mutable's alone, and a call that arrives anywhere else is
//!   refused before its arguments are parsed.
//!
//! Reads and writes stay separate **tools**, never modes of one tool, because
//! that split is what the capability check binds to.
//!
//! # Why so few tools, with modes inside them
//!
//! The constrained decoder guarantees a call's *structure*: an enum field can
//! only ever decode to one of its listed values. What it cannot guarantee is
//! that the model picked the right **tool** — that choice happens earlier, in
//! the projection's top-k, where nothing is enforced.
//!
//! So every distinction that was costing a wrong tool choice has been moved
//! *inside* a tool, where the grammar decides it. `git_show`'s `what` covers
//! what were four tools (a diff, a patch, a file's contents, a tree listing)
//! plus blame; measured live, the file-versus-patch distinction was the single
//! most common routing mistake, and as an enum it cannot be made at all.
//!
//! # Every argument has a satisfiable form
//!
//! The other live failure was a *dead end*: the grammar committing the model
//! to a field it had no way to fill. Having chosen a commit revision it had to
//! produce a 40-character object id it did not hold, could not revise, and so
//! invented. Three rules follow, and they shape the whole surface:
//!
//! - **A revision always has an arm the model can satisfy.** [`RevKind::Parent`]
//!   exists so "the previous commit" needs no id.
//! - **Nothing that only a prior call could supply is required.** Leases and
//!   expected values are optional; omitted, the layer reads the current value
//!   itself and still swaps atomically.
//! - **Results page rather than truncate.** A truncated reply is a dead end;
//!   `page` is a way out, and an over-shot page clamps to the last one rather
//!   than returning an empty list the model would read as "nothing there".

mod commit;
mod fetch;
mod grep;
mod line_history;
mod log;
mod push;
mod reference;
mod refs;
mod show;
mod status;
mod wire;

pub use commit::GIT_COMMIT;
pub use fetch::GIT_FETCH;
pub use grep::GIT_GREP;
pub use log::GIT_LOG;
pub use push::GIT_PUSH;
pub use reference::GIT_REF;
pub use refs::GIT_REFS;
pub use show::GIT_SHOW;
pub use status::GIT_STATUS;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use zend_git::{
    Ancestor, BranchName, GitError, Oid, RefName, Repo as GitRepo, RepoPath, Rev, TagName,
};

use crate::context::ToolContext;
use crate::tool::ToolError;

/// Every way a `git_*` call fails.
#[derive(Debug, Error)]
pub enum GitToolError {
    #[error("no repository named {name:?}; repo must be one of: {known}")]
    UnknownRepo { name: String, known: String },
    #[error("this session has no workspace, so there is no repository to work in")]
    NoWorkspace,
    #[error("{0}")]
    Git(#[from] GitError),
}

impl ToolError for GitToolError {
    /// The git layer's variants keep their identity rather than collapsing
    /// into one `git_failed`, because the model can act on several of them: a
    /// `stale_ref` wants a re-read, an `unknown_revision` a different
    /// argument, a `checked_out_branch` a different branch.
    fn code(&self) -> &'static str {
        match self {
            Self::UnknownRepo { .. } => "unknown_repo",
            Self::NoWorkspace => "no_workspace",
            Self::Git(e) => match e {
                GitError::GitMissing(_) | GitError::GitTooOld { .. } => "git_unavailable",
                GitError::NotARepository { .. } => "not_a_repository",
                GitError::UnknownRevision { .. } => "unknown_revision",
                GitError::NotABlob { .. } => "not_a_blob",
                GitError::BlobTooLarge { .. } => "blob_too_large",
                GitError::RefLocked { .. } => "ref_locked",
                GitError::StaleRef { .. } => "stale_ref",
                GitError::CheckedOutBranch { .. } => "checked_out_branch",
                GitError::UnknownRemote { .. } => "unknown_remote",
                GitError::RemoteExists { .. } => "remote_exists",
                GitError::AuthFailed { .. } => "auth_failed",
                GitError::RemoteUnreachable { .. } => "remote_unreachable",
                GitError::InvalidInput(_) => "invalid_arguments",
                GitError::Malformed { .. } => "git_output_unreadable",
                GitError::Timeout { .. } => "timeout",
                GitError::Io(_) => "io_error",
                GitError::Unclassified { .. } => "git_failed",
            },
        }
    }
}

/// Open the workspace repository called `name` as a git working tree.
pub fn open(ctx: &ToolContext, name: &str) -> Result<GitRepo, GitToolError> {
    let workspace = ctx.files.workspace().ok_or(GitToolError::NoWorkspace)?;
    let repo = workspace
        .repo(name)
        .ok_or_else(|| GitToolError::UnknownRepo {
            name: name.to_string(),
            known: workspace.names().join(", "),
        })?;
    Ok(GitRepo::open(&repo.dir)?)
}

/// Which form of revision a [`RevArg`] names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum RevKind {
    /// What the developer has checked out. Needs no `name`.
    Head,
    /// A local branch by short name, e.g. `main`.
    Branch,
    /// A tag by short name, e.g. `v1.2.0`.
    Tag,
    /// Any ref by full name, e.g. `refs/remotes/origin/main`.
    Ref,
    /// A commit by full object id.
    Commit,
    /// An ancestor of another revision — `back` commits before `name`, along
    /// first parents (git's `name~back`), so at a merge it stays on the
    /// branch the merge was made on. The way to say "the previous commit"
    /// without holding its id.
    Parent,
    /// A remote-tracking branch by the short name it is spoken of by:
    /// `origin/main`. Exists because that is how the model writes it, and
    /// because without it the only correct spelling was the full
    /// `refs/remotes/origin/main` — which, measured live, it did not reach
    /// for. It wrote the call it wanted in prose and emitted `null`.
    RemoteBranch,
    /// The branch `name` tracks — its upstream, or `HEAD`'s upstream when
    /// `name` is omitted. The form for "how far ahead of my upstream am I".
    Upstream,
}

/// A revision.
///
/// Flat rather than a tagged union, and that costs no typing: the constrained
/// decoder merges a `oneOf` into exactly this shape before the grammar sees
/// it, so the union's only extra promise — which field belongs to which tag —
/// was never enforced anyway. `kind` stays an enum the grammar decides, at
/// about a third of the tokens.
///
/// **Optional, never `null`** — here and on every field of the family that
/// takes a revision. A field the schema types as `T | null` hands the decoder
/// a one-token way to satisfy the grammar and say nothing, and measured live
/// that is the door the model took: `{"rev": null, "since": null}` where it
/// meant `origin/main`, three times over, until the repeat guard ended the
/// turn. `null` meant exactly what leaving the field out means, so it is not
/// offered: `#[serde(default)]` keeps the field optional and `schemars(with)`
/// types it as the value alone. The choice left is to leave it out or to
/// build the revision, and the grammar walks the second as far as it goes.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct RevArg {
    pub kind: RevKind,
    /// The branch or tag's short name, a full `refs/…` name, a commit id, or
    /// a remote branch as `origin/main` — whichever `kind` calls for. For
    /// `parent`, the revision to count back from, and for `upstream`, the
    /// branch whose upstream is meant; both default to `HEAD`. Omit for
    /// `head`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub name: Option<String>,
    /// For `parent` only: how many commits back. Defaults to 1.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "u32")]
    pub back: Option<u32>,
}

impl RevArg {
    /// This argument as the git layer's own typed revision.
    ///
    /// `parent` is resolved here to an object id, along first parents, so
    /// what comes out is a commit the layer has already found — and when the
    /// history is shorter than asked for, the error says how far back it
    /// goes.
    pub fn resolve(&self, repo: &GitRepo) -> Result<Rev, GitToolError> {
        let named = |what: &str| -> Result<&str, GitToolError> {
            self.name
                .as_deref()
                .filter(|s| !s.is_empty())
                .ok_or_else(|| GitError::invalid(format!("a {what} revision needs `name`")).into())
        };
        Ok(match self.kind {
            RevKind::Head => Rev::Head,
            RevKind::Branch => Rev::Branch(BranchName::parse(named("branch")?)?),
            RevKind::Tag => Rev::Tag(TagName::parse(named("tag")?)?),
            RevKind::Ref => Rev::Ref(RefName::parse(named("ref")?)?),
            RevKind::Commit => Rev::Oid(Oid::parse(named("commit")?)?),
            RevKind::RemoteBranch => {
                // `origin/main` is the name it is spoken of by; the ref lives
                // under refs/remotes/. Accept the full form too, so a caller
                // that already has it is not turned away.
                let name = named("remote branch")?;
                let full = if name.starts_with("refs/") {
                    name.to_string()
                } else {
                    format!("refs/remotes/{name}")
                };
                Rev::Ref(RefName::parse(&full)?)
            }
            RevKind::Upstream => {
                let of = match self.name.as_deref().filter(|s| !s.is_empty()) {
                    Some(name) => BranchName::parse(name)?,
                    None => repo.head()?.branch().cloned().ok_or_else(|| {
                        GitError::invalid(
                            "HEAD is detached, so it tracks nothing; name the branch whose \
                             upstream you mean",
                        )
                    })?,
                };
                let branch = repo
                    .branches()?
                    .into_iter()
                    .find(|b| b.name == of)
                    .ok_or_else(|| GitError::invalid(format!("no branch named {of}")))?;
                // No upstream is the common case on a local topic branch, and
                // the error names the form that does work rather than leaving
                // the caller to guess.
                let up = branch.upstream.ok_or_else(|| {
                    GitError::invalid(format!(
                        "{of} has no upstream configured, so it tracks nothing; compare \
                         against a remote branch instead, as \
                         {{\"kind\":\"remote_branch\",\"name\":\"origin/main\"}}"
                    ))
                })?;
                Rev::Ref(up.tracking_ref())
            }
            RevKind::Parent => {
                let from = self
                    .name
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .unwrap_or("HEAD");
                let base = branchish(from)?;
                let back = self.back.unwrap_or(1);
                match repo.first_parent_ancestor(&base, back)? {
                    Ancestor::Found(oid) => Rev::Oid(oid),
                    Ancestor::PastRoot { depth } => {
                        return Err(GitError::invalid(format!(
                            "{from} has only {depth} commit(s) behind it, so there is no \
                             ancestor {back} back from it"
                        ))
                        .into())
                    }
                }
            }
        })
    }
}

/// A `parent`'s base, read as whichever form the name fits: a full object id,
/// a full ref, or a branch. The model writes `HEAD`, `main` or a sha here and
/// all three work, because guessing wrong would be another dead end.
fn branchish(name: &str) -> Result<Rev, GitToolError> {
    if name == "HEAD" {
        return Ok(Rev::Head);
    }
    if let Ok(oid) = Oid::parse(name) {
        return Ok(Rev::Oid(oid));
    }
    if name.starts_with("refs/") {
        return Ok(Rev::Ref(RefName::parse(name)?));
    }
    Ok(Rev::Branch(BranchName::parse(name)?))
}

/// A revision argument that defaults to `HEAD` when the call omits it.
pub fn rev_or_head(rev: &Option<RevArg>, repo: &GitRepo) -> Result<Rev, GitToolError> {
    match rev {
        None => Ok(Rev::Head),
        Some(r) => r.resolve(repo),
    }
}

/// Parse a repository-relative path argument, refusing a protected one.
///
/// **A `secrets` path segment is refused here exactly as `VfsStore` refuses it
/// for the `file_*` tools.** A key committed once stays in the object store
/// forever, so reading it out of a past commit would serve precisely what the
/// live path is guarded against.
pub fn path_arg(path: &str) -> Result<RepoPath, GitToolError> {
    let parsed = RepoPath::parse(path)?;
    if parsed.is_protected() {
        return Err(
            GitError::invalid(format!("{path} is under a protected `secrets` folder")).into(),
        );
    }
    Ok(parsed)
}

/// Parse several repository-relative paths.
pub fn path_args(paths: &[String]) -> Result<Vec<RepoPath>, GitToolError> {
    paths.iter().map(|p| path_arg(p)).collect()
}

/// Whether a path the repository holds must be kept out of a result.
///
/// [`path_arg`] covers a path the call named; this covers everything a
/// wildcard read would sweep up — a grep with no paths searches every tracked
/// file, and a patch carries file contents.
pub fn is_protected(path: &RepoPath) -> bool {
    path.is_protected()
}
