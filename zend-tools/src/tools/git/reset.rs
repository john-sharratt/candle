//! git_reset tool — move this conversation's branch, on origin, and decide
//! what becomes of the work it moved past.
//!
//! The branch moves from the conversation's own `HEAD` — its base — and only
//! when the branch still holds exactly that: moving a branch that has gained
//! commits the conversation never saw would take them off it, so that is
//! refused until they are merged in. The move goes to origin first, under a
//! lease on what origin held, and locally after. Then, as `git reset` does to
//! a working tree:
//!
//! - **`soft`** keeps what you see exactly as it was: every file the move
//!   changed becomes one of your uncommitted changes, holding what the
//!   branch held before, and your own uncommitted changes stay.
//! - **`hard`** discards every uncommitted change, and any merge being
//!   finished: you see the branch as it now stands. With no `to`, it moves
//!   nothing and only discards.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use validator::Validate;
use zend_vfs::vfs::Carried;
use zend_vfs::work::{held, keep_ours};
use zend_vfs::{Base, GitError, Oid, RepoPath, Rev, VfsError, VfsStore};

use super::{landed, open, ConvRepo, GitToolError, RevArg};
use crate::{RegisteredTool, Tool, ToolContext};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum ResetMode {
    /// Keep what you see: what the move changes becomes uncommitted.
    Soft,
    /// Discard every uncommitted change.
    Hard,
}

#[derive(Deserialize, JsonSchema, Validate)]
#[serde(deny_unknown_fields)]
pub struct ResetRequest {
    /// The repository. Required.
    #[validate(length(min = 1))]
    pub repo: String,
    /// Where to move your branch. Defaults to where it is — with `hard`,
    /// that discards your uncommitted changes and moves nothing. Optional,
    /// never `null` — see [`RevArg`].
    #[serde(default)]
    #[schemars(with = "RevArg")]
    pub to: Option<RevArg>,
    /// What becomes of the work: `soft` or `hard`. Required.
    pub mode: ResetMode,
    /// The commit you must be at — git_status's `head`. Omit it and the reset
    /// starts from wherever that is; give it to refuse the reset if it moved
    /// since you looked.
    #[serde(default)]
    #[schemars(with = "String")]
    pub expected_head: Option<String>,
}

#[derive(Serialize)]
pub struct ResetResponse {
    pub repo: String,
    pub branch: String,
    /// The commit the branch holds now.
    pub id: String,
    /// What it held before.
    pub previous: String,
    /// Where the move is kept: `origin`, or `local` for a repository with no
    /// origin. Absent when the branch did not move.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub on: Option<&'static str>,
    /// Uncommitted changes discarded — `hard` only.
    pub discarded: usize,
    /// Files the move changed that are now uncommitted changes — `soft` only.
    pub kept: usize,
}

/// Every file that differs between `was` and `now` and `files` holds no
/// change of its own at: what a soft reset writes back.
fn moved_past(
    repo: &ConvRepo,
    files: &VfsStore,
    was: &Oid,
    now: &Oid,
) -> Result<Vec<RepoPath>, GitToolError> {
    let mut paths: Vec<RepoPath> = Vec::new();
    for entry in repo.diff(&Rev::Oid(now.clone()), &Rev::Oid(was.clone()), &[])? {
        for side in [&entry.old, &entry.new].into_iter().flatten() {
            let path = &side.path;
            if !paths.contains(path) && !path.is_protected() && !files.is_modified(path.as_str()) {
                paths.push(path.clone());
            }
        }
    }
    Ok(paths)
}

/// The content of each of `paths` at `was`: its text, or `None` where
/// `was` has no file. Refused, naming them, when any is not text — a soft
/// reset could not keep it as an uncommitted change, and must not move the
/// branch past it.
fn texts_at(
    repo: &ConvRepo,
    was: &Oid,
    paths: &[RepoPath],
) -> Result<Vec<(RepoPath, Option<String>)>, GitToolError> {
    let blobs = repo.blobs();
    let rev = Rev::Oid(was.clone());
    let (mut texts, mut not_text) = (Vec::new(), Vec::new());
    for path in paths {
        match blobs.read_at(&rev, path) {
            Ok(Some(bytes)) => match String::from_utf8(bytes) {
                Ok(text) => texts.push((path.clone(), Some(text))),
                Err(_) => not_text.push(path.as_str().to_string()),
            },
            Ok(None) => texts.push((path.clone(), None)),
            Err(GitError::BlobTooLarge { .. }) => not_text.push(path.as_str().to_string()),
            Err(e) => return Err(e.into()),
        }
    }
    if !not_text.is_empty() {
        return Err(GitError::invalid(format!(
            "a soft reset keeps what the commits changed as your uncommitted changes, and these \
             are not text, so they could not be kept: {}. Nothing was moved. A hard reset \
             drops them with the rest; git_commit's `revert` undoes the commits instead",
            not_text.join(", ")
        ))
        .into());
    }
    Ok(texts)
}

/// Make each of `texts` read in `files` as it was. Returns how many.
fn keep_as_it_was(
    files: &VfsStore,
    texts: Vec<(RepoPath, Option<String>)>,
) -> Result<usize, GitToolError> {
    let mut kept = 0;
    for (path, text) in texts {
        match text {
            Some(text) => {
                files
                    .write(path.as_str(), text)
                    .map_err(|e| GitError::invalid(e.to_string()))?;
                kept += 1;
            }
            None => {
                if files.delete(path.as_str()) {
                    kept += 1;
                }
            }
        }
    }
    Ok(kept)
}

pub struct GitReset;

impl Tool for GitReset {
    const NAME: &'static str = "git_reset";
    const DESCRIPTION: &'static str =
        "Move the branch you are on to another commit — back to undo commits, or anywhere \
         else — on origin, and decide what becomes of your work. `soft` keeps what you see: \
         what the commits you moved past changed becomes your uncommitted changes, ready to \
         commit again. `hard` discards every uncommitted change, so you see the branch as \
         it now stands; with no `to` it only discards your changes, a merge being finished \
         included. Moving back rewrites origin's copy of the branch, so it is refused while \
         the branch has commits you do not have — git_merge them in first — or if someone \
         else pushes in between. A soft reset past a file that is not text is refused, \
         since it could not be kept, and so is a soft reset while a merge is being \
         finished — commit it or discard it with a hard reset. Use for \"undo the last commit\", \"throw away my \
         changes\", \"abandon this merge\", \"put the branch back to that commit\". To undo \
         a commit without rewriting history, git_commit's `revert` is the way. Writes to \
         origin.";

    type Request = ResetRequest;
    type Response = ResetResponse;
    type Error = GitToolError;

    fn run(ctx: &ToolContext, req: ResetRequest) -> Result<ResetResponse, GitToolError> {
        let repo = open(ctx, &req.repo)?;
        let branch = repo.require_branch()?;
        let files = repo.store().cloned().ok_or_else(|| {
            GitError::invalid("this session holds no files for this repository to reset")
        })?;
        let invalid = |e: VfsError| GitError::invalid(e.to_string());
        let base = files
            .base()
            .map_err(invalid)?
            .ok_or_else(|| GitError::invalid("these files are not read from a branch"))?;
        let was = base
            .commit()
            .cloned()
            .ok_or_else(|| GitError::invalid(format!("{branch} has no commit to reset from")))?;
        let blobs = repo.blobs();
        let tree_of = |commit: &Oid| -> Result<Option<Oid>, GitToolError> {
            Ok(blobs
                .commit_of(&Rev::Oid(commit.clone()))?
                .map(|(_, tree)| tree))
        };
        if let Some(expected) = &req.expected_head {
            let expected = Oid::parse(expected)?;
            if expected != was {
                return Err(GitError::StaleRef {
                    name: branch.to_ref(),
                    detail: format!("expected {expected}, but you are at {was}"),
                }
                .into());
            }
        }
        // A base the repository no longer holds leaves one way on: a hard
        // reset onto the branch as it stands, which needs nothing of it.
        if tree_of(&was)?.is_none() {
            if req.mode == ResetMode::Soft || req.to.is_some() {
                return Err(GitError::invalid(format!(
                    "the commit your files were based on, {was}, is gone from the repository; \
                     a hard reset with no `to` puts you on {branch} as it stands"
                ))
                .into());
            }
            let tip = repo
                .ref_target(&branch.to_ref())?
                .ok_or_else(|| GitError::invalid(format!("no branch named {branch}")))?;
            let tree =
                tree_of(&tip)?.ok_or_else(|| GitError::invalid(format!("no commit {tip}")))?;
            let discarded = files
                .reset_to(None, Base::at(tip.clone(), tree))
                .map_err(invalid)?;
            held::release(&repo, &base);
            return Ok(ResetResponse {
                repo: req.repo,
                branch: branch.as_str().to_string(),
                id: tip.as_str().to_string(),
                previous: was.as_str().to_string(),
                on: None,
                discarded,
                kept: 0,
            });
        }
        if base.merging().is_some() && req.mode == ResetMode::Soft {
            return Err(GitError::invalid(
                "a merge is being finished here: commit it, or discard it with a hard reset",
            )
            .into());
        }
        let now = match &req.to {
            Some(to) => repo.resolve(&to.resolve(&repo)?)?,
            None => was.clone(),
        };
        let tree = tree_of(&now)?.ok_or_else(|| GitError::invalid(format!("no commit {now}")))?;
        // What a soft reset keeps is read, and refused if it cannot be kept,
        // before the branch moves anywhere.
        let keeping = match req.mode {
            ResetMode::Soft if now != was => Some(texts_at(
                &repo,
                &was,
                &moved_past(&repo, &files, &was, &now)?,
            )?),
            _ => None,
        };
        let pulled = if now == was {
            None
        } else {
            let pulled = repo.pull_branch(&branch)?;
            match pulled.record() {
                // The branch is there already: only your files move.
                Some(record) if record == &now => None,
                Some(record) if record != &was => {
                    return Err(GitError::StaleRef {
                        name: branch.to_ref(),
                        detail: format!(
                            "{branch} is at {record}, which holds commits you do not have; \
                             moving it would take them off the branch — git_merge them in \
                             first"
                        ),
                    }
                    .into());
                }
                _ => Some(pulled),
            }
        };

        // The conversation's files move first, so that anything about them
        // that cannot be done — the work would not fit, a file could not be
        // read — refuses the reset before origin is touched; and should
        // origin then refuse the move, the files are put back as they were.
        let before = files.snapshot();
        let moved = (|| -> Result<(usize, usize), GitToolError> {
            Ok(match req.mode {
                // Nothing is carried, so nothing about the work can stand in
                // the way of dropping it.
                ResetMode::Hard => (
                    files
                        .reset_to(None, Base::at(now.clone(), tree.clone()))
                        .map_err(invalid)?,
                    0,
                ),
                ResetMode::Soft => {
                    files
                        .move_base(
                            None,
                            Base::at(now.clone(), tree.clone()),
                            &[],
                            &mut |c: &Carried<'_>| keep_ours(c),
                        )
                        .map_err(invalid)?;
                    let kept = match keeping {
                        Some(texts) => keep_as_it_was(&files, texts)?,
                        None => 0,
                    };
                    (0, kept)
                }
            })
        })();
        let (discarded, kept) = match moved {
            Ok(counts) => counts,
            Err(e) => {
                files
                    .roll_back(&files.snapshot(), before)
                    .map_err(invalid)?;
                return Err(e);
            }
        };
        let after = files.snapshot();
        let on = match &pulled {
            None => None,
            Some(pulled) => {
                match repo
                    .publish_branch(pulled, &now)
                    .map_err(GitToolError::from)
                    .and_then(|published| landed(&branch.to_ref(), published))
                {
                    Ok(on) => Some(on),
                    Err(e) => {
                        if !files.roll_back(&after, before).map_err(invalid)? {
                            tracing::warn!(
                                "{branch} did not move, and your files changed while it was \
                                 being moved, so they were left where the reset put them"
                            );
                        }
                        return Err(e);
                    }
                }
            }
        };
        if req.mode == ResetMode::Hard {
            held::release(&repo, &base);
        }
        Ok(ResetResponse {
            repo: req.repo,
            branch: branch.as_str().to_string(),
            id: now.as_str().to_string(),
            previous: was.as_str().to_string(),
            on,
            discarded,
            kept,
        })
    }
}

pub const GIT_RESET: RegisteredTool = RegisteredTool::new::<GitReset>();
