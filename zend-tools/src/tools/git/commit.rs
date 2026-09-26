//! git_commit tool — produce one commit on one branch.
//!
//! Four sources behind one tool: explicit file contents, a patch, a commit
//! replayed forward (cherry-pick) or backwards (revert). They were three
//! tools; all three answer the same question — "put a commit on this branch"
//! — and differ only in where the content comes from, which is what `from`
//! now says.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_git::{
    ApplyOutcome, BranchName, ChangeSet, FileMode, GitError, Oid, PickOutcome, Repo as GitRepo,
};

use super::{open, path_arg, GitToolError};
use crate::state::VfsStore;
use crate::{RegisteredTool, Tool, ToolContext};

/// Where the commit's content comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum CommitFrom {
    /// The `changes` list: files written, taken from disk, or deleted.
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
    /// Commit the file exactly as it stands in the working tree. **Prefer
    /// this whenever the edit is already on disk** — it needs no `content`,
    /// so there is nothing to reproduce from memory and nothing to get wrong.
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
    /// The branch to commit on. It must exist and must not be checked out.
    #[validate(length(min = 1))]
    pub branch: String,
    /// Where the content comes from. Required.
    pub from: CommitFrom,
    /// The commit message. Required for `files` and `patch`; `cherry_pick`
    /// keeps the original's and `revert` writes its own.
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
    /// The commit the branch must currently point at. Omit and the layer
    /// reads it, still swapping atomically; give it to refuse the commit if
    /// the branch moved since you looked.
    #[serde(default)]
    #[schemars(with = "String")]
    pub expected_head: Option<String>,
}

#[derive(Serialize)]
pub struct CommitResponse {
    pub repo: String,
    pub branch: String,
    /// False when the content did not apply; nothing was committed and the
    /// branch did not move.
    pub applied: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commit: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub author: Option<String>,
    /// Why it did not apply — git's reason for a patch, the conflicting
    /// paths for a cherry-pick or revert.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub conflicts: Option<Vec<String>>,
}

/// The branch's tip, refusing a branch that does not exist, and checking the
/// caller's expectation when it gave one.
fn base(
    repo: &GitRepo,
    branch: &BranchName,
    expected: &Option<String>,
) -> Result<Oid, GitToolError> {
    let tip = repo.ref_target(&branch.to_ref())?.ok_or_else(|| {
        GitError::invalid(format!(
            "no branch named {branch}; create it with git_ref before committing on it"
        ))
    })?;
    if let Some(expected) = expected {
        let expected = Oid::parse(expected)?;
        if expected != tip {
            return Err(GitError::StaleRef {
                name: branch.to_ref(),
                detail: format!("expected {expected}, but the branch holds {tip}"),
            }
            .into());
        }
    }
    Ok(tip)
}

/// The change set `changes` describe. `take` reads through `files`, the
/// repository's file store — the same guarded route `file_read` takes — so a
/// path that store would not open, one leaving the repository or under a
/// protected folder, cannot be committed from disk either.
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
                         as it stands on disk",
                        change.path
                    ))
                })?;
                set.write(path, content.as_bytes().to_vec(), mode)?;
            }
            ChangeAction::Take => {
                let cannot = |why: String| {
                    GitError::invalid(format!(
                        "{}: cannot take it from the working tree: {why}",
                        change.path
                    ))
                };
                let bytes = files
                    .read_bytes(path.as_str())
                    .map_err(|e| cannot(e.to_string()))?
                    .ok_or_else(|| cannot("no such file inside the repository".into()))?;
                set.write(path, bytes, mode)?;
            }
        }
    }
    Ok(set)
}

pub struct GitCommit;

impl Tool for GitCommit {
    const NAME: &'static str = "git_commit";
    const DESCRIPTION: &'static str =
        "Record one commit on a branch. `from` says where the content comes from: `files` \
         takes a list of changes — `take` commits a file exactly as it stands on disk \
         (prefer this whenever the edit is already written), `write` replaces it with \
         `content`, `delete` removes it; `patch` applies a unified diff; `cherry_pick` \
         replays another commit onto the branch; `revert` applies one backwards. The \
         commit is built as objects and the branch moved onto it — the working tree, the \
         index and the checked-out files are never touched, and a branch checked out in \
         any worktree is refused. A patch that does not apply, or a replay that conflicts, \
         reports why and commits nothing. `expected_head` is optional: omit it and the \
         branch tip is read here, still swapped atomically. The branch must already exist \
         — create one with git_ref. Writes to the repository.";

    type Request = CommitRequest;
    type Response = CommitResponse;
    type Error = GitToolError;

    fn run(ctx: &ToolContext, req: CommitRequest) -> Result<CommitResponse, GitToolError> {
        let repo = open(ctx, &req.repo)?;
        let branch = BranchName::parse(&req.branch)?;
        let onto = base(&repo, &branch, &req.expected_head)?;
        let identity = repo.identity()?;
        let done = |commit: Oid| -> Result<CommitResponse, GitToolError> {
            repo.move_branch(&branch, &onto, &commit)?;
            Ok(CommitResponse {
                repo: req.repo.clone(),
                branch: req.branch.clone(),
                applied: true,
                commit: Some(commit.as_str().to_string()),
                parent: Some(onto.as_str().to_string()),
                author: Some(format!("{} <{}>", identity.name(), identity.email())),
                reason: None,
                conflicts: None,
            })
        };
        let refused = |reason: Option<String>, conflicts: Option<Vec<String>>| CommitResponse {
            repo: req.repo.clone(),
            branch: req.branch.clone(),
            applied: false,
            commit: None,
            parent: None,
            author: None,
            reason,
            conflicts,
        };
        let message = || -> Result<String, GitToolError> {
            req.message
                .clone()
                .filter(|m| !m.is_empty())
                .ok_or_else(|| GitError::invalid("this commit needs a `message`").into())
        };

        match req.from {
            CommitFrom::Files => {
                let changes = req.changes.as_deref().unwrap_or(&[]);
                if changes.is_empty() {
                    return Err(
                        GitError::invalid("`files` needs at least one entry in `changes`").into(),
                    );
                }
                let files = ctx
                    .files
                    .repo(&req.repo)
                    .map_err(|e| GitToolError::UnknownRepo {
                        name: e.name,
                        known: e.known.join(", "),
                    })?;
                let set = build(&files, changes)?;
                let commit = repo.commit_changes(&onto, &set, &message()?, &identity, &identity)?;
                done(commit)
            }
            CommitFrom::Patch => {
                let patch = req
                    .patch
                    .as_deref()
                    .filter(|p| !p.is_empty())
                    .ok_or_else(|| GitError::invalid("`patch` needs a unified diff in `patch`"))?;
                match repo.apply_patch(&onto, patch.as_bytes())? {
                    ApplyOutcome::Rejected { detail } => Ok(refused(Some(detail), None)),
                    ApplyOutcome::Applied { tree } => {
                        let commit =
                            repo.commit_tree(&tree, &[&onto], &message()?, &identity, &identity)?;
                        done(commit)
                    }
                }
            }
            CommitFrom::CherryPick | CommitFrom::Revert => {
                let id = req
                    .commit
                    .as_deref()
                    .filter(|c| !c.is_empty())
                    .ok_or_else(|| {
                        GitError::invalid("this needs the id of the commit to replay, in `commit`")
                    })?;
                let subject = Oid::parse(id)?;
                let outcome = if req.from == CommitFrom::CherryPick {
                    repo.cherry_pick(&subject, &onto, &identity)?
                } else {
                    repo.revert(&subject, &onto, &identity, &identity)?
                };
                match outcome {
                    PickOutcome::Clean(commit) => done(commit),
                    PickOutcome::Conflicted { paths } => Ok(refused(
                        Some("the change does not apply cleanly here".to_string()),
                        Some(paths.iter().map(|p| p.as_str().to_string()).collect()),
                    )),
                }
            }
        }
    }
}

pub const GIT_COMMIT: RegisteredTool = RegisteredTool::new::<GitCommit>();
