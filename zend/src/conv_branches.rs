//! The branch each conversation works on in each repository.
//!
//! A conversation keeps its own line of work in every repository of the
//! workspace, whatever the developer has checked out: a branch per repository,
//! held in the conversation's state (`ConvState`, one record written whole
//! and last-writer-wins on replay).
//!
//! At startup the daemon reads each repository's branches once and picks the
//! branch a conversation starts on there — its *base*: `main` when the
//! repository has one, else `master`, else whatever branch is checked out. A
//! folder that is not a git repository (the uploads folder, say) has none. A
//! conversation with no branch recorded for a repository is given the base —
//! every live conversation at startup, and each new one as it is created. A
//! branch once recorded is the conversation's own and is never replaced here.

use std::collections::{BTreeMap, HashSet};
use std::path::Path;

use candle_conversation::persistence::manifest::ConvState;
use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;
use zend_vfs::{GitError, Repo, Workspace};

/// The branch a conversation starts on in each repository, by repository name.
pub type BaseBranches = BTreeMap<String, String>;

/// The branches a repository's base is chosen from, most preferred first.
const PREFERRED: [&str; 2] = ["main", "master"];

/// Each git repository's base branch. A repository whose branches cannot be
/// read, or that has none, is left out — logged, never fatal, since the daemon
/// serves a workspace whether or not every folder in it is under git.
pub fn base_branches(workspace: &Workspace) -> BaseBranches {
    let mut bases = BaseBranches::new();
    for repo in workspace.repos() {
        match base_branch(&repo.dir) {
            Ok(Some(branch)) => {
                tracing::info!(repo = %repo.name, %branch, "conversations start on");
                bases.insert(repo.name.clone(), branch);
            }
            Ok(None) => {
                tracing::info!(repo = %repo.name, "no branch for conversations to start on");
            }
            Err(e) => {
                tracing::info!(repo = %repo.name, "no conversation branches here: {e}");
            }
        }
    }
    bases
}

/// `dir`'s base branch: `main`, else `master`, else the checked-out branch.
/// `None` when there is none of those — a detached `HEAD` in a repository
/// with neither preferred branch.
fn base_branch(dir: &Path) -> Result<Option<String>, GitError> {
    let repo = Repo::open(dir)?;
    let branches = repo.branches()?;
    for preferred in PREFERRED {
        if branches.iter().any(|b| b.name.as_str() == preferred) {
            return Ok(Some(preferred.to_string()));
        }
    }
    Ok(repo.head()?.branch().map(|b| b.as_str().to_string()))
}

/// The bases for the repositories `state` records no branch for yet.
fn unset(state: &ConvState, bases: &BaseBranches) -> BaseBranches {
    bases
        .iter()
        .filter(|(repo, _)| !state.branches.contains_key(*repo))
        .map(|(repo, branch)| (repo.clone(), branch.clone()))
        .collect()
}

/// Give `timeline` the base branch of every repository it has none for, in
/// one state write. Returns how many it was given; `0` for an unknown
/// timeline, or one that already has a branch everywhere.
pub fn seed(engine: &ConversationEngine, timeline: TimelineId, bases: &BaseBranches) -> usize {
    let Some(state) = engine.conversation_state(timeline) else {
        return 0;
    };
    let missing = unset(&state, bases);
    if !missing.is_empty() {
        engine.set_conversation_branches(timeline, &missing);
    }
    missing.len()
}

/// The startup pass: [`seed`] every conversation a user has — named, live,
/// not archived — except `hidden`, the daemon's own titler. An archived
/// conversation is seeded when it is next used instead: calibration archives
/// thousands of exemplars, none of which works on a branch. Returns how many
/// conversations were written.
pub fn seed_live(engine: &ConversationEngine, bases: &BaseBranches, hidden: TimelineId) -> usize {
    if bases.is_empty() {
        return 0;
    }
    let live: HashSet<TimelineId> = engine.live_conversations().into_iter().collect();
    engine
        .known_conversations()
        .into_iter()
        .filter(|(tl, _, _, archived, _)| !archived && *tl != hidden && live.contains(tl))
        .filter(|(tl, ..)| seed(engine, *tl, bases) > 0)
        .count()
}

#[cfg(test)]
mod tests {
    use std::process::Command;

    use zend_vfs::RepoSpec;

    use super::*;

    fn git(dir: &Path, args: &[&str]) {
        let out = Command::new("git")
            .arg("-C")
            .arg(dir)
            .args([
                "-c",
                "core.hooksPath=",
                "-c",
                "user.name=T",
                "-c",
                "user.email=t@x",
            ])
            .args(args)
            .output()
            .expect("git runs");
        assert!(
            out.status.success(),
            "git {args:?}: {}",
            String::from_utf8_lossy(&out.stderr)
        );
    }

    /// A repository at `dir` with one commit on `first`, then each of `more`
    /// created, and `checked_out` checked out.
    fn repo_with(dir: &Path, first: &str, more: &[&str], checked_out: &str) {
        std::fs::create_dir_all(dir).unwrap();
        git(dir, &["init", "-q"]);
        git(
            dir,
            &["symbolic-ref", "HEAD", &format!("refs/heads/{first}")],
        );
        git(dir, &["commit", "-q", "--allow-empty", "-m", "first"]);
        for b in more {
            git(dir, &["branch", b]);
        }
        git(dir, &["checkout", "-q", checked_out]);
    }

    fn base(dir: &Path) -> Option<String> {
        base_branch(dir).unwrap()
    }

    /// **`main` wins, then `master`, then what is checked out** — and what
    /// is checked out never outranks `main`.
    #[test]
    fn the_base_is_main_then_master_then_the_checkout() {
        let root = tempfile::tempdir().unwrap();
        let both = root.path().join("both");
        repo_with(&both, "master", &["main", "topic"], "topic");
        assert_eq!(base(&both).as_deref(), Some("main"));

        let master = root.path().join("master");
        repo_with(&master, "master", &["topic"], "topic");
        assert_eq!(base(&master).as_deref(), Some("master"));

        let neither = root.path().join("neither");
        repo_with(&neither, "trunk", &["topic"], "topic");
        assert_eq!(base(&neither).as_deref(), Some("topic"));
    }

    /// A folder that is not a repository is left out of the bases, and the
    /// rest of the workspace still gets them.
    #[test]
    fn a_folder_outside_git_has_no_base() {
        let root = tempfile::tempdir().unwrap();
        repo_with(&root.path().join("code"), "main", &[], "main");
        std::fs::create_dir_all(root.path().join("uploads")).unwrap();
        let workspace = Workspace::new(
            root.path(),
            vec![RepoSpec::named("code"), RepoSpec::named("uploads")],
        )
        .unwrap();
        assert_eq!(
            base_branches(&workspace),
            BaseBranches::from([("code".to_string(), "main".to_string())])
        );
    }

    /// **Only a repository with no branch yet is given one.** A branch the
    /// conversation already works on is its own, and stays.
    #[test]
    fn only_unset_repositories_are_given_their_base() {
        let bases = BaseBranches::from([
            ("candle".to_string(), "main".to_string()),
            ("mind".to_string(), "master".to_string()),
        ]);
        let state = ConvState {
            archived: false,
            branches: BTreeMap::from([("candle".to_string(), "zen/work".to_string())]),
            active: 0,
        };
        assert_eq!(
            unset(&state, &bases),
            BaseBranches::from([("mind".to_string(), "master".to_string())])
        );
        assert_eq!(unset(&ConvState::default(), &bases), bases);
        let full = ConvState {
            archived: true,
            branches: bases.clone(),
            active: 0,
        };
        assert!(unset(&full, &bases).is_empty());
    }
}
