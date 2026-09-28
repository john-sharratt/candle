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
//! work; a commit made while a merge is being finished records the merge,
//! and one after a merge fast-forwarded the files past the branch, with no
//! change of the conversation's own, moves the branch onto them.
//!
//! On another branch the commit is built on origin's copy of that branch,
//! and published the same way.

use std::collections::HashMap;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_vfs::{
    ApplyOutcome, BranchName, ChangeSet, Committing, FileMode, FileState, GitError, Landed,
    LogRange, MergeLabels, NotCommitted, Oid, PickOutcome, Published, Rejection, RepoPath, Rev,
    Signature, TreeEntry, VfsStore,
};

use super::wire::{landed_on, refusal};
use super::{open, path_arg, ConvRepo, GitToolError};
use crate::{RegisteredTool, Tool, ToolContext};

/// Where the commit's content comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CommitFrom {
    /// Every change you have not committed, as git_status lists it — and,
    /// while a merge is being finished or after one fast-forwarded your
    /// files past the branch, what the merge brought in, even with no change
    /// of your own.
    ///
    /// Named for what it takes — all of it. As `changes` it read as "my
    /// changes": measured live, a model whose reasoning said "commit only
    /// README.md" wrote `from: changes` as its first constrained choice, and
    /// committed the file it had been told to leave out as well.
    AllChanges,
    /// The `changes` list: files written, taken as you hold them, or deleted —
    /// some of your changes and not others.
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
    /// The commit message. Required for `all_changes`, `files` and `patch`;
    /// `cherry_pick` keeps the original's and `revert` writes its own.
    #[serde(default)]
    #[schemars(with = "String")]
    pub message: Option<String>,
    /// For `from: files` — the files this commit changes. Emptiness is
    /// refused where it matters — the `files()` closure below, scoped to
    /// `from: files` — not here: a field-level `Validate` on `Option<Vec<_>>`
    /// applies to the inner value whenever it is `Some`, so a `min` here would
    /// also refuse a `from: patch`/`cherry_pick`/`revert` call that happened to
    /// carry `changes: []`, which is none of this field's business under those.
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
    /// up to it. Also set when no commit was needed — the branch was moved
    /// onto the commit a merge brought your files onto.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<String>,
    /// Why nothing was committed, and what to do next.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The paths in the way: files still in conflict from a merge, or those
    /// a cherry-pick or revert could not apply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conflicts: Option<Vec<String>>,
    /// For a cherry-pick or revert that did not apply: each file in
    /// `conflicts` with the commit's change merged in as far as it goes, and
    /// both sides of every overlap between conflict markers — to write as it
    /// should be. A file that is not text is left out.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub drafts: Option<Vec<Draft>>,
}

/// One conflicting file of a replay, merged three ways.
#[derive(Serialize)]
pub struct Draft {
    pub path: String,
    /// The merge, overlaps between `<<<<<<< your branch` and `>>>>>>> the
    /// commit` markers. Cut past [`DRAFT_SHOWN_BYTES`].
    pub content: String,
}

/// How much of one draft a refused replay's answer carries.
const DRAFT_SHOWN_BYTES: usize = 16 * 1024;

/// Content that does not apply to the commit it is built on.
struct Refusal {
    reason: String,
    conflicts: Option<Vec<String>>,
    /// The conflicting files merged three ways, when a replay did not apply.
    drafts: Option<Vec<Draft>>,
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
        let refused = |reason: String, conflicts: Option<Vec<String>>| {
            Ok(Err(Refusal {
                reason,
                conflicts,
                drafts: None,
            }))
        };
        match self.from {
            CommitFrom::AllChanges | CommitFrom::Files => {
                let set = self
                    .set
                    .as_ref()
                    .expect("gathered for all_changes and files");
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
                // A replay must have something to do: a cherry-pick brings a
                // commit the branch lacks, a revert undoes one it holds.
                // Measured live: a model cherry-picked a commit already in the
                // branch's history, took the conflict it made for the change it
                // wanted, and "settled" it by undoing the revert it had just
                // committed.
                let held = repo.is_ancestor(&Rev::Oid(subject.clone()), &Rev::Oid(onto.clone()))?;
                match (self.from, held) {
                    (CommitFrom::CherryPick, true) => {
                        return refused(already_held(subject), None);
                    }
                    (CommitFrom::Revert, false) => {
                        return refused(not_held(subject), None);
                    }
                    _ => {}
                }
                let outcome = if self.from == CommitFrom::CherryPick {
                    repo.cherry_pick(subject, onto, identity)?
                } else {
                    repo.revert(subject, onto, identity, identity)?
                };
                match outcome {
                    PickOutcome::Clean(commit) => Ok(Ok(commit)),
                    PickOutcome::Conflicted { paths } => Ok(Err(Refusal {
                        reason: replay_conflict(self.from, subject, onto),
                        conflicts: Some(paths.iter().map(|p| p.as_str().to_string()).collect()),
                        drafts: Some(drafts(repo, self.from, subject, onto, &paths)?),
                    })),
                }
            }
        }
    }
}

/// Why a cherry-pick of `commit` has nothing to do: the branch holds it.
fn already_held(commit: &Oid) -> String {
    format!(
        "{commit} is already in this branch's history, so there is nothing to replay and \
         nothing was committed. To bring over a commit from another branch, git_log that branch \
         with `since` set to this one: what it lists is what this branch lacks"
    )
}

/// Why a revert of `commit` has nothing to do: the branch does not hold it.
fn not_held(commit: &Oid) -> String {
    format!(
        "{commit} is not in this branch's history, so there is nothing of it to undo and \
         nothing was committed. git_log this branch for the commit to revert"
    )
}

/// Why `commit` cannot be replayed in `repo`: it is not there.
fn no_such_commit(repo: &str, commit: &Oid) -> GitError {
    GitError::invalid(format!(
        "{repo} holds no commit {commit} — an id from another repository's history is not in \
         this one. git_log in {repo} on the branch that holds the change gives its id; for a \
         branch on origin, `rev: {{\"kind\": \"remote_branch\", \"name\": \"origin/<branch>\"}}`"
    ))
}

/// Why a replay of `subject` onto `onto` was refused, and the way on: the
/// `drafts` the answer carries, written as they should be. A replay is one
/// attempt, whole or not at all, so what is left is to make the change by
/// hand. Measured live: told where to read the commit's change, a model read
/// its own branch's latest patch instead and made that; handed the commit's
/// patch alone, another rebuilt the file from it and dropped the branch's
/// own line beside it. A merged file shows both at once.
fn replay_conflict(from: CommitFrom, subject: &Oid, onto: &Oid) -> String {
    let what = match from {
        CommitFrom::Revert => "undoing",
        _ => "replaying",
    };
    format!(
        "{what} {subject} does not apply cleanly onto {onto}, where the files in `conflicts` \
         have changed too, so nothing was committed. `drafts` holds each of them with the \
         change merged in as far as it goes, and both sides of every overlap between \
         <<<<<<< your branch and >>>>>>> the commit markers: write each file as it should be — \
         keeping the lines of both sides that belong — then commit them with git_commit"
    )
}

/// Each of `paths` with `subject`'s change — undone, for a revert — merged
/// three ways into `onto`'s copy: its parent (or the empty tree) as the base.
/// A side that is not text leaves its file out.
fn drafts(
    repo: &ConvRepo,
    from: CommitFrom,
    subject: &Oid,
    onto: &Oid,
    paths: &[RepoPath],
) -> Result<Vec<Draft>, GitToolError> {
    let info = repo.log(&LogRange::of(Rev::Oid(subject.clone())), 1)?;
    let parent = match info.first().and_then(|c| c.parents.first()) {
        Some(parent) => parent.clone(),
        None => repo.format().empty_tree(),
    };
    let (base, theirs, label) = match from {
        CommitFrom::Revert => (subject, &parent, "the commit undone"),
        _ => (&parent, subject, "the commit"),
    };
    let blobs = repo.blobs();
    let text_at = |rev: &Oid, path: &RepoPath| -> Result<Option<String>, GitToolError> {
        Ok(match blobs.read_at(&Rev::Oid(rev.clone()), path) {
            Ok(Some(bytes)) => String::from_utf8(bytes).ok(),
            Ok(None) => Some(String::new()),
            Err(GitError::BlobTooLarge { .. }) | Err(GitError::NotABlob { .. }) => None,
            Err(e) => return Err(e.into()),
        })
    };
    let mut drafts = Vec::new();
    for path in paths {
        let (Some(was), Some(yours), Some(incoming)) = (
            text_at(base, path)?,
            text_at(onto, path)?,
            text_at(theirs, path)?,
        ) else {
            continue;
        };
        let labels = MergeLabels {
            ours: "your branch",
            base: "before the commit",
            theirs: label,
        };
        let merged = repo.merge_text(&was, &yours, &incoming, labels)?;
        drafts.push(Draft {
            path: path.as_str().to_string(),
            content: cut(merged.text, subject),
        });
    }
    Ok(drafts)
}

/// `text`, cut at [`DRAFT_SHOWN_BYTES`] on a character boundary with a note.
fn cut(text: String, subject: &Oid) -> String {
    if text.len() <= DRAFT_SHOWN_BYTES {
        return text;
    }
    let at = (0..=DRAFT_SHOWN_BYTES)
        .rev()
        .find(|&at| text.is_char_boundary(at))
        .unwrap_or(0);
    format!(
        "{}\n… (cut here; git_show {{\"what\": \"patch\", \"rev\": {{\"kind\": \"commit\", \
         \"name\": \"{subject}\"}}}} reads the commit's change whole)",
        &text[..at]
    )
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
            drafts: None,
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

    /// The branch moved onto the commit a merge brought the files onto, with
    /// no commit made.
    fn fast_forwarded(mut self, landed: Landed, from: Option<Oid>) -> Self {
        let from = from.map_or_else(|| "no commit".to_string(), |o| o.to_string());
        self.note = Some(format!(
            "no new commit was needed: {} moved from {from} to {}, the commit a merge brought \
             your files onto",
            self.branch, landed.commit
        ));
        self.applied = true;
        self.commit = Some(landed.commit.as_str().to_string());
        self.on = Some(landed_on(&landed.published));
        self
    }

    fn refused(mut self, refusal: Refusal) -> Self {
        self.reason = Some(refusal.reason);
        self.conflicts = refusal.conflicts;
        self.drafts = refusal.drafts;
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
                drafts: None,
            },
            NotCommitted::Behind { record } => Refusal {
                reason: format!(
                    "{branch} holds commits you do not have (it is at {record}), so nothing \
                     was committed and your files are as they were. git_merge brings them \
                     into your files; then commit again"
                ),
                conflicts: None,
                drafts: None,
            },
            NotCommitted::Refused(Rejection::Stale) => Refusal {
                reason: format!(
                    "{branch} on origin moved while this commit was being pushed, so nothing \
                     was written and your files are as they were. git_merge brings in what \
                     arrived; then commit again"
                ),
                conflicts: None,
                drafts: None,
            },
            NotCommitted::Refused(why) => Refusal {
                reason: refusal(&why),
                conflicts: None,
                drafts: None,
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
            drafts: None,
        })
    }
}

pub struct GitCommit;

impl Tool for GitCommit {
    const NAME: &'static str = "git_commit";
    const DESCRIPTION: &'static str =
        "Record one commit on a branch — the branch you are on unless you name another — \
         and publish it to origin, all at once or not at all. `from` says where the content \
         comes from: `all_changes` commits every change you have not committed (what \
         git_status lists), and publishes what a git_merge brought in even with no change of \
         your own; `files` commits only the files it lists — `take` commits a file exactly \
         as you hold it (prefer this whenever the edit is already written), `write` replaces \
         it with `content`, `delete` removes it — and leaves every other change uncommitted; \
         `patch` applies a unified diff; `cherry_pick` replays another \
         commit onto the branch; `revert` applies one backwards. Once committed, those \
         changes are no longer uncommitted. If the branch has commits you do not have, or \
         gets one while this is pushed, nothing is committed and your files are untouched: \
         git_merge them in, then commit again. Files still in conflict from a merge must be \
         settled first, and a merge is committed whole, with `all_changes`. Committing `files` to \
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
                CommitFrom::AllChanges | CommitFrom::Files | CommitFrom::Patch
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
                let oid = Oid::parse(id)?;
                // A commit this repository does not hold is named as such, with
                // where its id comes from — not git's `bad object`. Measured
                // live: a model looked the change up in another repository of
                // the workspace, replayed that commit's id here, and read the
                // raw git failure as a reason to go looking somewhere else.
                if repo.blobs().commit_of(&Rev::Oid(oid.clone()))?.is_none() {
                    return Err(no_such_commit(&req.repo, &oid).into());
                }
                Some(oid)
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
            if finishing_merge && req.from != CommitFrom::AllChanges {
                return Err(GitError::invalid(
                    "a merge is being finished here, and it is committed whole: use \
                     `from: all_changes`, which records the merge with everything you settled it \
                     with — then make any other commit",
                )
                .into());
            }
            // A merge that fast-forwarded the files past the branch leaves no
            // change to commit: publishing it is moving the branch onto them.
            if req.from == CommitFrom::AllChanges
                && !finishing_merge
                && committing.ahead()
                && store.status().is_empty()
            {
                let from = committing.record().cloned();
                return Ok(match committing.publish_base(&repo)? {
                    Ok(landed) => out.fast_forwarded(landed, from),
                    Err(why) => out.not_committed(&branch, why),
                });
            }
            let set = match req.from {
                CommitFrom::AllChanges => Some(uncommitted(store, finishing_merge)?),
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

        if req.from == CommitFrom::AllChanges {
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
                    drafts: None,
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

#[cfg(test)]
mod validation_tests {
    use validator::Validate;

    use super::{ChangeAction, CommitFrom, CommitRequest, FileChange};

    fn file(path: &str) -> FileChange {
        FileChange {
            action: ChangeAction::Take,
            path: path.to_string(),
            content: None,
            executable: None,
        }
    }

    /// `changes` has no business being validated for a `from` that never
    /// reads it. A field-level `Validate` on `Option<Vec<_>>` applies to the
    /// inner value whenever it is `Some` — regardless of `from` — so this is
    /// the regression a blanket `min` on the field reintroduces.
    #[test]
    fn an_empty_changes_list_does_not_refuse_a_patch_commit() {
        let req = CommitRequest {
            repo: "r".to_string(),
            branch: None,
            from: CommitFrom::Patch,
            message: Some("m".to_string()),
            changes: Some(vec![]),
            patch: Some("diff".to_string()),
            commit: None,
            expected_head: None,
        };
        assert!(req.validate().is_ok(), "{:?}", req.validate());
    }

    /// The upper bound is still real — `changes` genuinely belongs to
    /// `from: files`, so a request that grossly overruns it is still refused
    /// at the field, whichever `from` carried it.
    #[test]
    fn an_oversized_changes_list_is_still_refused() {
        let req = CommitRequest {
            repo: "r".to_string(),
            branch: None,
            from: CommitFrom::Files,
            message: Some("m".to_string()),
            changes: Some((0..201).map(|i| file(&format!("f{i}"))).collect()),
            patch: None,
            commit: None,
            expected_head: None,
        };
        assert!(req.validate().is_err());
    }

    /// The real protection against an empty `changes` under `from: files`
    /// lives in the `files()` closure, scoped to that `from` — not the
    /// struct-level validator. Covered here as the sibling fact to the two
    /// tests above: this crate's own `git_tools` integration tests exercise
    /// the closure's refusal through a live repo.
    #[test]
    fn an_empty_changes_list_passes_struct_validation_regardless_of_from() {
        let req = CommitRequest {
            repo: "r".to_string(),
            branch: None,
            from: CommitFrom::Files,
            message: Some("m".to_string()),
            changes: Some(vec![]),
            patch: None,
            commit: None,
            expected_head: None,
        };
        assert!(req.validate().is_ok(), "{:?}", req.validate());
    }
}
