//! Linked worktrees: more checkouts of the same repository, each in its own
//! folder, sharing one object store. A build can run in one without
//! disturbing the user's checkout.

use std::path::{Path, PathBuf};

use crate::error::GitError;
use crate::setup::require_empty_target;
use crate::types::{BranchName, Oid, RefName};
use crate::Repo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Worktree {
    pub path: PathBuf,
    /// `None` for a bare repository's entry, or a branch with no commits.
    pub head: Option<Oid>,
    /// The checked-out branch; `None` when detached.
    pub branch: Option<BranchName>,
    pub detached: bool,
    pub bare: bool,
    pub locked: bool,
    /// Its folder is gone; `git worktree prune` would remove the entry.
    pub prunable: bool,
}

/// What a new worktree checks out.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorktreeCheckout {
    /// A new branch at `at`, which must not exist.
    NewBranch { branch: BranchName, at: Oid },
    /// An existing branch, not checked out in any other worktree.
    Branch(BranchName),
    /// A commit, on no branch.
    Detached(Oid),
}

/// Parse `worktree list --porcelain`: one attribute per line, records
/// separated by a blank line. (`-z` needs 2.36; a worktree path with a
/// newline in it cannot be created through [`Repo::add_worktree`].)
pub(crate) fn parse_worktrees(out: &[u8]) -> Result<Vec<Worktree>, GitError> {
    let text =
        std::str::from_utf8(out).map_err(|e| GitError::malformed("worktree", e.to_string()))?;
    let mut trees = Vec::new();
    let mut current: Option<Worktree> = None;
    for attr in text.split('\n').map(|l| l.trim_end_matches('\r')) {
        if attr.is_empty() {
            if let Some(t) = current.take() {
                trees.push(t);
            }
            continue;
        }
        let (key, value) = attr.split_once(' ').unwrap_or((attr, ""));
        if key == "worktree" {
            if let Some(t) = current.take() {
                trees.push(t);
            }
            current = Some(Worktree {
                path: PathBuf::from(value),
                head: None,
                branch: None,
                detached: false,
                bare: false,
                locked: false,
                prunable: false,
            });
            continue;
        }
        let t = current
            .as_mut()
            .ok_or_else(|| GitError::malformed("worktree", attr.to_string()))?;
        match key {
            "HEAD" => t.head = Oid::parse_nonzero(value)?,
            "branch" => t.branch = RefName::parse(value)?.branch(),
            "detached" => t.detached = true,
            "bare" => t.bare = true,
            "locked" => t.locked = true,
            "prunable" => t.prunable = true,
            _ => {}
        }
    }
    if let Some(t) = current {
        trees.push(t);
    }
    Ok(trees)
}

impl Repo {
    /// Every worktree of this repository, the main one first.
    pub fn worktrees(&self) -> Result<Vec<Worktree>, GitError> {
        let out = self
            .git("worktree")
            .args(["list", "--porcelain"])
            .read_only()
            .run_ok()?;
        parse_worktrees(&out)
    }

    /// Add a worktree at `path` — absolute, free of control characters, and
    /// not an existing non-empty folder — and open it.
    pub fn add_worktree(&self, path: &Path, checkout: &WorktreeCheckout) -> Result<Repo, GitError> {
        require_empty_target(path)?;
        if path.to_string_lossy().chars().any(|c| c.is_control()) {
            return Err(GitError::invalid(format!(
                "worktree path {} contains a control character",
                path.display()
            )));
        }
        let _write = self.write_lock();
        let mut inv = self.git("worktree").arg("add");
        let target = match checkout {
            WorktreeCheckout::NewBranch { branch, at } => {
                inv = inv.arg("-b").arg(branch.as_str());
                at.to_string()
            }
            WorktreeCheckout::Branch(branch) => branch.to_string(),
            WorktreeCheckout::Detached(oid) => {
                inv = inv.arg("--detach");
                oid.to_string()
            }
        };
        inv.arg("--end-of-options").arg(path).arg(target).run_ok()?;
        Repo::open(path)
    }

    /// Remove the worktree at `path`. Without `force`, one with local changes
    /// is refused.
    pub fn remove_worktree(&self, path: &Path, force: bool) -> Result<(), GitError> {
        let _write = self.write_lock();
        let mut inv = self.git("worktree").arg("remove");
        if force {
            inv = inv.arg("--force");
        }
        inv.arg("--end-of-options").arg(path).run_ok()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::read::head::Head;
    use crate::testing::{git_in, scratch, TestRepo};
    use crate::types::Rev;

    #[test]
    fn porcelain_records_parse() {
        let a = "ce013625030ba8dba906f756967f9e9ca394464a";
        let out = format!(
            "worktree /r/main\nHEAD {a}\nbranch refs/heads/main\n\n\
             worktree /r/wt\nHEAD {a}\ndetached\nlocked\nprunable gitdir file points to non-existent location\n\n"
        );
        let w = parse_worktrees(out.as_bytes()).unwrap();
        assert_eq!(w.len(), 2);
        assert_eq!(w[0].path, PathBuf::from("/r/main"));
        assert_eq!(w[0].branch.as_ref().unwrap().as_str(), "main");
        assert!(!w[0].detached);
        assert!(w[1].detached && w[1].locked && w[1].prunable);
        assert_eq!(w[1].branch, None);
    }

    fn same(a: &Path, b: &Path) -> bool {
        a.canonicalize().unwrap() == b.canonicalize().unwrap()
    }

    #[test]
    fn worktrees_are_added_listed_used_and_removed() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let base = t.commit_all("base");
        let repo = t.repo();
        let dir = tempfile::Builder::new()
            .prefix("wt-")
            .tempdir_in(scratch())
            .unwrap();
        let wt_path = dir.path().join("build");
        let branch = BranchName::parse("zen/build").unwrap();

        let wt = repo
            .add_worktree(
                &wt_path,
                &WorktreeCheckout::NewBranch {
                    branch: branch.clone(),
                    at: base.clone(),
                },
            )
            .unwrap();
        assert_eq!(wt.head().unwrap().branch(), Some(&branch));
        assert_eq!(std::fs::read(wt_path.join("a.txt")).unwrap(), b"a\n");

        let list = repo.worktrees().unwrap();
        assert_eq!(list.len(), 2);
        assert!(same(&list[0].path, &t.path), "the main worktree is first");
        assert!(same(&list[1].path, &wt_path));
        assert_eq!(list[1].branch.as_ref(), Some(&branch));
        assert_eq!(list[1].head.as_ref(), Some(&base));

        // A commit made in the worktree moves the shared branch.
        std::fs::write(wt_path.join("b.txt"), b"b\n").unwrap();
        git_in(&wt_path, &["add", "b.txt"]);
        git_in(&wt_path, &["commit", "-q", "-m", "in worktree"]);
        let moved = repo.resolve(&Rev::Branch(branch.clone())).unwrap();
        assert_ne!(moved, base);
        // The user's checkout is untouched.
        assert!(!t.path.join("b.txt").exists());

        // A branch checked out in one worktree cannot be checked out again.
        assert!(repo
            .add_worktree(
                &dir.path().join("again"),
                &WorktreeCheckout::Branch(branch.clone())
            )
            .is_err());

        repo.remove_worktree(&wt_path, false).unwrap();
        assert_eq!(repo.worktrees().unwrap().len(), 1);
        assert!(!wt_path.exists());
    }

    #[test]
    fn a_detached_worktree_and_forced_removal() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let base = t.commit_all("base");
        let repo = t.repo();
        let dir = tempfile::Builder::new()
            .prefix("wt-")
            .tempdir_in(scratch())
            .unwrap();
        let wt_path = dir.path().join("detached");
        let wt = repo
            .add_worktree(&wt_path, &WorktreeCheckout::Detached(base.clone()))
            .unwrap();
        assert_eq!(wt.head().unwrap(), Head::Detached(base));

        std::fs::write(wt_path.join("a.txt"), b"dirty\n").unwrap();
        assert!(
            repo.remove_worktree(&wt_path, false).is_err(),
            "local changes refuse"
        );
        repo.remove_worktree(&wt_path, true).unwrap();
        assert!(!wt_path.exists());
    }
}
