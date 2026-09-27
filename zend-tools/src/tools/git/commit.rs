//! git_commit tool — produce one commit on one branch, on origin.
//!
//! Five sources behind one tool: your uncommitted changes, explicit file
//! contents, a patch, a commit replayed forward (cherry-pick) or backwards
//! (revert). All five answer the same question — "put a commit on this
//! branch" — and differ only in where the content comes from, which is what
//! `from` says.
//!
//! A commit is one attempt, published whole or not at all. On the branch the
//! conversation is on it is built on the conversation's own base
//! ([`Committing`]) and pushed to origin under a lease; it is refused — with
//! nothing written anywhere, the conversation's files untouched — while a
//! merge's conflicts are unsettled, when origin holds commits the
//! conversation does not have, or when origin moves during the push. The way
//! on is git_merge, then the commit again. Once it lands, the committed
//! changes read from the branch and leave the conversation's uncommitted
//! work; a commit made while a merge is being finished records the merge.
//!
//! On another branch the commit is built on origin's copy of that branch,
//! and published the same way.

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_vfs::{
    ApplyOutcome, BranchName, ChangeSet, Committing, FileMode, FileState, GitError, Landed,
    NotCommitted, Oid, PickOutcome, Published, Rejection, RepoPath, Rev, Signature, TreeEntry,
    VfsStore,
};

use super::wire::{landed_on, refusal};
use super::{open, path_arg, ConvRepo, GitToolError};
use crate::{RegisteredTool, Tool, ToolContext};

/// Where the commit's content comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CommitFrom {
    /// Every change you have not committed, as git_status lists it — and,
    /// while a merge is being finished, the merge itself, even with no
    /// change of your own.
    Changes,
    /// The `changes` list: files written, taken as you hold them, or deleted.
    Files,
    /// A unified-diff `patch` applied to the branch.
    Patch,
    /// Replay `commit`'s change onto the branch, keeping its author and
    /// message.
    CherryPick,
    /// Apply `commit`'s change backwards, undoing it.
    Revert,
}

/// What to do to one file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ChangeAction {
    /// Commit the file exactly as you hold it now. **Prefer this whenever the
    /// edit is already written** — it needs no `content`, so there is nothing
    /// to reproduce from memory and nothing to get wrong.
    Take,
    /// Replace the file with `content`, which must be its complete new text.
    Write,
    /// Remove the file.
    Delete,
}

/// One file's change.
///
/// Flat, with `action` an enum the grammar decides. `Take` exists because the
/// alternative — requiring the model to reproduce a file it may not have read
/// — is the one dead end in this family whose invented output would be
/// *committed*.
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct FileChange {
    pub action: ChangeAction,
    /// Repository-relative path.
    pub path: String,
    /// The file's complete new text. Required for `write`, ignored otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "String")]
    pub content: Option<String>,
    /// Mark the file executable. Defaults to the mode it already has.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[schemars(with = "bool")]
    pub executable: Option<bool>,
}

#[derive(Deserialize, JsonSchema, Validate)]
#[serde(deny_unknown_fields)]
pub struct CommitRequest {
    /// The repository to commit in. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// The branch to commit on. Defaults to the branch you are on.
    #[serde(default)]
    #[schemars(with = "String")]
    pub branch: Option<String>,
    /// Where the content comes from. Required.
    pub from: CommitFrom,
    /// The commit message. Required for `changes`, `files` and `patch`;
    /// `cherry_pick` keeps the original's and `revert` writes its own.
    #[serde(default)]
    #[schemars(with = "String")]
    pub message: Option<String>,
    /// For `from: files` — the files this commit changes.
    #[validate(length(max = 200))]
    #[serde(default)]
    #[schemars(with = "Vec<FileChange>")]
    pub changes: Option<Vec<FileChange>>,
    /// For `from: patch` — a unified diff, as `git diff` produces it.
    #[serde(default)]
    #[schemars(with = "String")]
    pub patch: Option<String>,
    /// For `cherry_pick` and `revert` — the commit to replay, by full id.
    #[serde(default)]
    #[schemars(with = "String")]
    pub commit: Option<String>,
    /// The commit the new one must be made on — on your branch, git_status's
    /// `head`. Omit it and the commit is made on whatever that is; give it to
    /// refuse the commit if it moved since you looked.
    #[serde(default)]
    #[schemars(with = "String")]
    pub expected_head: Option<String>,
}

#[derive(Serialize)]
pub struct CommitResponse {
    pub repo: String,
    pub branch: String,
    /// False when nothing was committed — see `reason`. Nothing was then
    /// written anywhere, and your files are exactly as they were.
    pub applied: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    /// The commit a merge brought in, when this commit finished the merge.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merged: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// Where the commit is kept: `origin`, or `local` for a repository with
    /// no origin.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub on: Option<&'static str>,
    /// Set when the commit landed but your files could not be moved onto it:
    /// they still read from the commit before it, and git_merge brings them
    /// up to it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// Why nothing was committed, and what to do next.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The paths in the way: files still in conflict from a merge, or those
    /// a cherry-pick or revert could not apply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conflicts: Option<Vec<String>>,
}

/// Content that does not apply to the commit it is built on.
struct Refusal {
    reason: String,
    conflicts: Option<Vec<String>>,
}

/// The change set `changes` describe. `take` reads through `files`, the
/// conversation's store — the same guarded route `file_read` takes — so a
/// path that store would not open, one leaving the repository or under a
/// protected folder, cannot be committed either.
fn build(files: &VfsStore, changes: &[FileChange]) -> Result<ChangeSet, GitToolError> {
    let mut set = ChangeSet::new();
    for change in changes {
        let path = path_arg(&change.path)?;
        let mode = change.executable.map(|x| {
            if x {
                FileMode::Executable
            } else {
                FileMode::Regular
            }
        });
        match change.action {
            ChangeAction::Delete => set.delete(path)?,
            ChangeAction::Write => {
                let content = change.content.as_deref().ok_or_else(|| {
                    GitError::invalid(format!(
                        "{}: `write` needs `content` — or use `take` to commit the file \
                         as you hold it",
                        change.path
                    ))
                })?;
                set.write(path, content.as_bytes().to_vec(), mode)?;
            }
            ChangeAction::Take => {
                let bytes = take(files, &change.path)?;
                set.write(path, bytes, mode)?;
            }
        }
    }
    Ok(set)
}

/// Every change `files` holds, as a change set. None is refused, unless the
/// commit finishes a merge — which is a commit of its own.
fn uncommitted(files: &VfsStore, finishing_merge: bool) -> Result<ChangeSet, GitToolError> {
    let changed = files.status();
    if changed.is_empty() && !finishing_merge {
        return Err(GitError::invalid(
            "you have no uncommitted changes in this repository; git_status lists them",
        )
        .into());
    }
    let mut set = ChangeSet::new();
    for (path, state) in changed {
        let at = path_arg(&path)?;
        match state {
            FileState::Deleted => set.delete(at)?,
            FileState::Added | FileState::Modified => set.write(at, take(files, &path)?, None)?,
        }
    }
    Ok(set)
}

/// The file at `path`, as the conversation holds it.
fn take(files: &VfsStore, path: &str) -> Result<Vec<u8>, GitToolError> {
    let cannot = |why: String| GitError::invalid(format!("{path}: cannot take it: {why}"));
    let bytes = files
        .read_bytes(path)
        .map_err(|e| cannot(e.to_string()))?
        .ok_or_else(|| cannot("no such file in the repository".into()))?;
    Ok(bytes)
}

/// The conversation's store, which `take` and `changes` read.
fn store(repo: &ConvRepo) -> Result<&VfsStore, GitToolError> {
    repo.store().map(|s| s.as_ref()).ok_or_else(|| {
        GitError::invalid("this session holds no files for this repository to commit").into()
    })
}

/// What a commit is made of, gathered from the request before anything is
/// written: a bad argument writes nothing anywhere.
struct Content<'r> {
    from: CommitFrom,
    set: Option<ChangeSet>,
    patch: Option<&'r str>,
    subject: Option<Oid>,
    message: Option<String>,
}

impl Content<'_> {
    /// The commit, built on `onto` — or why its content does not apply there.
    fn build_on(
        &self,
        repo: &ConvRepo,
        onto: &Oid,
        identity: &Signature,
    ) -> Result<Result<Oid, Refusal>, GitToolError> {
        let message = self.message.as_deref().unwrap_or_default();
        let refused =
            |reason: String, conflicts: Option<Vec<String>>| Ok(Err(Refusal { reason, conflicts }));
        match self.from {
            CommitFrom::Changes | CommitFrom::Files => {
                let set = self.set.as_ref().expect("gathered for changes and files");
                Ok(Ok(
                    repo.commit_changes(onto, set, message, identity, identity)?
                ))
            }
            CommitFrom::Patch => {
                let patch = self.patch.expect("gathered for a patch");
                match repo.apply_patch(onto, patch.as_bytes())? {
                    ApplyOutcome::Rejected { detail } => refused(detail, None),
                    ApplyOutcome::Applied { tree } => Ok(Ok(repo.commit_tree(
                        &tree,
                        &[onto],
                        message,
                        identity,
                        identity,
                    )?)),
                }
            }
            CommitFrom::CherryPick | CommitFrom::Revert => {
                let subject = self.subject.as_ref().expect("gathered for a replay");
                let outcome = if self.from == CommitFrom::CherryPick {
                    repo.cherry_pick(subject, onto, identity)?
                } else {
                    repo.revert(subject, onto, identity, identity)?
                };
                match outcome {
                    PickOutcome::Clean(commit) => Ok(Ok(commit)),
                    PickOutcome::Conflicted { paths } => refused(
                        "the change does not apply cleanly here".to_string(),
                        Some(paths.iter().map(|p| p.as_str().to_string()).collect()),
                    ),
                }
            }
        }
    }
}

impl CommitResponse {
    fn new(repo: &str, branch: &BranchName) -> Self {
        Self {
            repo: repo.to_string(),
            branch: branch.as_str().to_string(),
            applied: false,
            commit: None,
            parent: None,
            merged: None,
            author: None,
            on: None,
            note: None,
            reason: None,
            conflicts: None,
        }
    }

    fn landed(mut self, landed: Landed, identity: &Signature) -> Self {
        self.applied = true;
        self.commit = Some(landed.commit.as_str().to_string());
        self.parent = landed.parents.first().map(|p| p.as_str().to_string());
        self.merged = landed.parents.get(1).map(|p| p.as_str().to_string());
        self.author = Some(format!("{} <{}>", identity.name(), identity.email()));
        self.on = Some(landed_on(&landed.published));
        self.note = landed.behind.map(|why| {
            format!(
                "the commit landed, but your files could not be moved onto it ({why}); they \
                 still read from the commit before — git_merge brings them up to it"
            )
        });
        self
    }

    fn refused(mut self, refusal: Refusal) -> Self {
        self.reason = Some(refusal.reason);
        self.conflicts = refusal.conflicts;
        self
    }

    /// Why a commit was not made, in words that say what to do next.
    fn not_committed(self, branch: &BranchName, why: NotCommitted) -> Self {
        let refusal = match why {
            NotCommitted::Conflicts(paths) => Refusal {
                reason: "these files are still in conflict from a merge, so nothing was \
                         committed. Each holds both sides between <<<<<<< and >>>>>>> \
                         markers, or is a change one side made to a file the other deleted: \
                         write each as it should be — or delete it — then commit again"
                    .to_string(),
                conflicts: Some(paths),
            },
            NotCommitted::Behind { record } => Refusal {
                reason: format!(
                    "{branch} holds commits you do not have (it is at {record}), so nothing \
                     was committed and your files are as they were. git_merge brings them \
                     into your files; then commit again"
                ),
                conflicts: None,
            },
            NotCommitted::Refused(Rejection::Stale) => Refusal {
                reason: format!(
                    "{branch} on origin moved while this commit was being pushed, so nothing \
                     was written and your files are as they were. git_merge brings in what \
                     arrived; then commit again"
                ),
                conflicts: None,
            },
            NotCommitted::Refused(why) => Refusal {
                reason: refusal(&why),
                conflicts: None,
            },
        };
        self.refused(refusal)
    }

    /// Why a commit onto another branch — not the one the conversation is
    /// on — was not made. Nothing of the conversation's moves either way, so
    /// the way on is to commit again, onto the branch as it then stands.
    fn not_committed_there(self, branch: &BranchName, why: Rejection) -> Self {
        let reason = match why {
            Rejection::Stale => format!(
                "{branch} on origin moved while this commit was being made, so nothing was \
                 written. Commit again: it is made on {branch} as it then stands"
            ),
            other => refusal(&other),
        };
        self.refused(Refusal {
            reason,
            conflicts: None,
        })
    }
}

pub struct GitCommit;

impl Tool for GitCommit {
    const NAME: &'static str = "git_commit";
    const DESCRIPTION: &'static str =
        "Record one commit on a branch — the branch you are on unless you name another — \
         and publish it to origin, all at once or not at all. `from` says where the content \
         comes from: `changes` commits every change you have not committed (what git_status \
         lists); `files` takes a list — `take` commits a file exactly as you hold it (prefer \
         this whenever the edit is already written), `write` replaces it with `content`, \
         `delete` removes it; `patch` applies a unified diff; `cherry_pick` replays another \
         commit onto the branch; `revert` applies one backwards. Once committed, those \
         changes are no longer uncommitted. If the branch has commits you do not have, or \
         gets one while this is pushed, nothing is committed and your files are untouched: \
         git_merge them in, then commit again. Files still in conflict from a merge must be \
         settled first, and a merge is committed whole, with `changes`. Committing `files` to \
         another branch that has changed them since your work started is refused unless \
         `expected_head` names that branch's tip. A patch that does not apply, or a \
         replay that conflicts, reports why and commits nothing. The branch must already \
         exist — make one with git_switch. Writes to origin.";

    type Request = CommitRequest;
    type Response = CommitResponse;
    type Error = GitToolError;

    fn run(ctx: &ToolContext, req: CommitRequest) -> Result<CommitResponse, GitToolError> {
        let repo = open(ctx, &req.repo)?;
        let current = repo.branch();
        let branch = match &req.branch {
            Some(name) => BranchName::parse(name)?,
            None => repo.require_branch()?,
        };
        let own = current.as_ref() == Some(&branch);
        let identity = repo.identity()?;
        let expected = req.expected_head.as_deref().map(Oid::parse).transpose()?;
        let message = req.message.clone().filter(|m| !m.is_empty());
        if message.is_none()
            && matches!(
                req.from,
                CommitFrom::Changes | CommitFrom::Files | CommitFrom::Patch
            )
        {
            return Err(GitError::invalid("this commit needs a `message`").into());
        }
        let patch = match req.from {
            CommitFrom::Patch => Some(
                req.patch
                    .as_deref()
                    .filter(|p| !p.is_empty())
                    .ok_or_else(|| GitError::invalid("`patch` needs a unified diff in `patch`"))?,
            ),
            _ => None,
        };
        let subject = match req.from {
            CommitFrom::CherryPick | CommitFrom::Revert => {
                let id = req
                    .commit
                    .as_deref()
                    .filter(|c| !c.is_empty())
                    .ok_or_else(|| {
                        GitError::invalid("this needs the id of the commit to replay, in `commit`")
                    })?;
                Some(Oid::parse(id)?)
            }
            _ => None,
        };
        let files = || -> Result<ChangeSet, GitToolError> {
            let changes = req.changes.as_deref().unwrap_or(&[]);
            if changes.is_empty() {
                return Err(
                    GitError::invalid("`files` needs at least one entry in `changes`").into(),
                );
            }
            build(store(&repo)?, changes)
        };
        let out = CommitResponse::new(&req.repo, &branch);
        let stale = |at: &Oid, expected: &Oid| -> GitToolError {
            GitError::StaleRef {
                name: branch.to_ref(),
                detail: format!("expected {expected}, but the commit would be made on {at}"),
            }
            .into()
        };

        if own {
            let store = store(&repo)?;
            let committing = match Committing::begin(&repo, store, &branch)? {
                Ok(committing) => committing,
                Err(why) => return Ok(out.not_committed(&branch, why)),
            };
            if let (Some(expected), Some(base)) = (&expected, store.base().ok().flatten()) {
                let at = base.commit().cloned().unwrap_or_else(|| expected.clone());
                if &at != expected {
                    return Err(stale(&at, expected));
                }
            }
            let finishing_merge = committing.finishes_merge();
            // A merge is recorded with every change the conversation settled
            // it with: committing some of them, or a patch, would record the
            // merge with the other side's changes to the rest dropped.
            if finishing_merge && req.from != CommitFrom::Changes {
                return Err(GitError::invalid(
                    "a merge is being finished here, and it is committed whole: use \
                     `from: changes`, which records the merge with everything you settled it \
                     with — then make any other commit",
                )
                .into());
            }
            let set = match req.from {
                CommitFrom::Changes => Some(uncommitted(store, finishing_merge)?),
                CommitFrom::Files => Some(files()?),
                _ => None,
            };
            let content = Content {
                from: req.from,
                set,
                patch,
                subject,
                message,
            };
            let built = match content.build_on(&repo, committing.onto(), &identity)? {
                Ok(built) => built,
                Err(refusal) => return Ok(out.refused(refusal)),
            };
            let message = content.message.as_deref().unwrap_or_default();
            return Ok(
                match committing.land(&repo, store, &built, message, &identity, &identity)? {
                    Ok(landed) => out.landed(landed, &identity),
                    Err(why) => out.not_committed(&branch, why),
                },
            );
        }

        if req.from == CommitFrom::Changes {
            let on = current.map_or_else(|| "no branch".to_string(), |b| b.to_string());
            return Err(GitError::invalid(format!(
                "your uncommitted changes are made on {on}, not {branch}: switch to {branch} \
                 with git_switch to commit them there, or name the files with `from: files`"
            ))
            .into());
        }
        let pulled = repo.pull_branch(&branch)?;
        let onto = pulled
            .record()
            .or(pulled.tip.as_ref())
            .cloned()
            .ok_or_else(|| {
                GitError::invalid(format!(
                    "no branch named {branch}; make it with git_switch before committing on it"
                ))
            })?;
        if let Some(expected) = &expected {
            if &onto != expected {
                return Err(stale(&onto, expected));
            }
        }
        let set = match req.from {
            CommitFrom::Files => Some(files()?),
            _ => None,
        };
        let named = match req.from {
            CommitFrom::Files => req.changes.as_deref().unwrap_or(&[]),
            _ => &[],
        };
        // A file still in conflict holds a merge's markers, not content: it
        // goes nowhere until it is settled, on any branch.
        if let Some(store) = repo.store() {
            let conflicts = store.conflicts();
            let unsettled: Vec<String> = named
                .iter()
                .filter_map(|c| path_arg(&c.path).ok())
                .map(|p| p.as_str().to_string())
                .filter(|p| conflicts.contains(p))
                .collect();
            if !unsettled.is_empty() {
                return Ok(out.not_committed(&branch, NotCommitted::Conflicts(unsettled)));
            }
        }
        // A file named whole onto another branch replaces that branch's copy.
        // When the branch has changed it since the version your work started
        // from, that would throw its change away: refused, unless
        // `expected_head` says you have seen the branch as it stands.
        if !named.is_empty() && expected.is_none() {
            let changed = changed_since_your_base(&repo, &onto, named)?;
            if !changed.is_empty() {
                return Ok(out.refused(Refusal {
                    reason: format!(
                        "{branch} has changed these files since the version your work started \
                         from, so committing yours would throw its changes away. Read them there \
                         (git_show on {branch}), make your content account for them, and commit \
                         again with `expected_head` set to {onto}"
                    ),
                    conflicts: Some(changed),
                }));
            }
        }
        let content = Content {
            from: req.from,
            set,
            patch,
            subject,
            message,
        };
        let built = match content.build_on(&repo, &onto, &identity)? {
            Ok(built) => built,
            Err(refusal) => return Ok(out.refused(refusal)),
        };
        Ok(match repo.publish_commit(&pulled, &built)? {
            Published::Behind => out.not_committed_there(&branch, Rejection::Stale),
            Published::Refused(why) => out.not_committed_there(&branch, why),
            published => out.landed(
                Landed {
                    commit: built,
                    parents: vec![onto],
                    published,
                    behind: None,
                },
                &identity,
            ),
        })
    }
}

/// The paths of `changes` whose file at `onto` differs from the one the
/// conversation's work started from — its base commit.
fn changed_since_your_base(
    repo: &ConvRepo,
    onto: &Oid,
    changes: &[FileChange],
) -> Result<Vec<String>, GitToolError> {
    let Some(base) = repo
        .store()
        .and_then(|s| s.base().ok().flatten())
        .and_then(|b| b.commit().cloned())
    else {
        return Ok(Vec::new());
    };
    let paths: Vec<RepoPath> = changes
        .iter()
        .map(|c| path_arg(&c.path))
        .collect::<Result<_, _>>()?;
    let refs: Vec<&RepoPath> = paths.iter().collect();
    let blobs = |entries: Vec<TreeEntry>| -> HashMap<String, Oid> {
        entries
            .into_iter()
            .map(|e| (e.path.as_str().to_string(), e.oid))
            .collect()
    };
    let there = blobs(repo.tree_entries(&Rev::Oid(onto.clone()), &refs)?);
    let yours = blobs(repo.tree_entries(&Rev::Oid(base), &refs)?);
    Ok(paths
        .iter()
        .map(|p| p.as_str())
        .filter(|p| there.get(*p) != yours.get(*p))
        .map(str::to_string)
        .collect())
}

pub const GIT_COMMIT: RegisteredTool = RegisteredTool::new::<GitCommit>();
