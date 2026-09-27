//! Atomic, compare-and-swap reference updates, via `update-ref --stdin -z`.

use crate::error::GitError;
use crate::types::{BranchName, Oid, RefName};
use crate::Repo;

/// One reference change. Every variant names the value it expects to find,
/// so a concurrent change fails the whole transaction instead of being
/// overwritten.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefOp {
    /// `name` must not exist.
    Create { name: RefName, new: Oid },
    /// `name` must hold `old`.
    Update { name: RefName, new: Oid, old: Oid },
    /// `name` must hold `old`.
    Delete { name: RefName, old: Oid },
}

impl RefOp {
    fn name(&self) -> &RefName {
        match self {
            Self::Create { name, .. } | Self::Update { name, .. } | Self::Delete { name, .. } => {
                name
            }
        }
    }
}

/// Reference changes applied all together or not at all.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct RefTransaction {
    ops: Vec<RefOp>,
}

impl RefTransaction {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(mut self, op: RefOp) -> Self {
        self.ops.push(op);
        self
    }

    pub fn ops(&self) -> &[RefOp] {
        &self.ops
    }

    /// The `update-ref --stdin -z` script: one command per op. `update-ref`
    /// locks every ref and checks every expected value before changing any,
    /// so the script applies whole or not at all without the explicit
    /// `start`/`commit` verbs newer releases add.
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut s = Vec::new();
        let mut field = |text: &str| {
            s.extend_from_slice(text.as_bytes());
            s.push(0);
        };
        for op in &self.ops {
            match op {
                RefOp::Create { name, new } => {
                    field(&format!("create {name}"));
                    field(new.as_str());
                }
                RefOp::Update { name, new, old } => {
                    field(&format!("update {name}"));
                    field(new.as_str());
                    field(old.as_str());
                }
                RefOp::Delete { name, old } => {
                    field(&format!("delete {name}"));
                    field(old.as_str());
                }
            }
        }
        s
    }
}

impl Repo {
    /// Apply `txn` atomically. Fails as [`GitError::StaleRef`] when any ref
    /// does not hold its expected value, and then changes nothing.
    ///
    /// Refuses to move the branch `HEAD` is on: that would change the user's
    /// checkout under them.
    pub fn update_refs(&self, txn: &RefTransaction) -> Result<(), GitError> {
        if txn.ops.is_empty() {
            return Ok(());
        }
        let _write = self.write_lock();
        self.update_refs_locked(txn)
    }

    pub(crate) fn update_refs_locked(&self, txn: &RefTransaction) -> Result<(), GitError> {
        // Every worktree's branch, not just this one's: moving a branch a
        // linked worktree has checked out changes that checkout under it.
        if txn.ops.iter().any(|op| op.name().branch().is_some()) {
            for tree in self.worktrees()? {
                if let Some(branch) = tree.branch {
                    if txn.ops.iter().any(|op| op.name() == &branch.to_ref()) {
                        return Err(GitError::CheckedOutBranch { branch });
                    }
                }
            }
        }
        self.write_refs(txn)
    }

    /// Apply `txn` with no check on which branches are checked out — for the
    /// execution checkout, which owns its branch for the length of a run.
    /// The caller holds the write lock.
    pub(crate) fn write_refs(&self, txn: &RefTransaction) -> Result<(), GitError> {
        // `--no-deref`: an op changes the ref it names and never the one a
        // symbolic ref points at, so the check in `update_refs_locked` —
        // which compares names — cannot be walked around through an alias of
        // a checked-out branch.
        self.git("update-ref")
            .args(["--no-deref", "--stdin", "-z"])
            .stdin(txn.to_bytes())
            .run_ok()?;
        Ok(())
    }

    /// Create `branch` at `at`; it must not exist.
    pub fn create_branch(&self, branch: &BranchName, at: &Oid) -> Result<(), GitError> {
        self.update_refs(&RefTransaction::new().push(RefOp::Create {
            name: branch.to_ref(),
            new: at.clone(),
        }))
    }

    /// Move `branch` from `old` to `new`.
    pub fn move_branch(&self, branch: &BranchName, old: &Oid, new: &Oid) -> Result<(), GitError> {
        self.update_refs(&RefTransaction::new().push(RefOp::Update {
            name: branch.to_ref(),
            new: new.clone(),
            old: old.clone(),
        }))
    }

    /// Delete `branch`, which must hold `old`.
    pub fn delete_branch(&self, branch: &BranchName, old: &Oid) -> Result<(), GitError> {
        self.update_refs(&RefTransaction::new().push(RefOp::Delete {
            name: branch.to_ref(),
            old: old.clone(),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::{scratch, TestRepo};
    use crate::worktrees::WorktreeCheckout;

    fn oid(c: char) -> Oid {
        Oid::parse(&c.to_string().repeat(40)).unwrap()
    }

    #[test]
    fn the_script_is_exactly_these_bytes() {
        let txn = RefTransaction::new()
            .push(RefOp::Create {
                name: RefName::parse("refs/heads/a").unwrap(),
                new: oid('1'),
            })
            .push(RefOp::Update {
                name: RefName::parse("refs/heads/b").unwrap(),
                new: oid('2'),
                old: oid('3'),
            })
            .push(RefOp::Delete {
                name: RefName::parse("refs/heads/c").unwrap(),
                old: oid('4'),
            });
        let expected = format!(
            "create refs/heads/a\0{}\0update refs/heads/b\0{}\0{}\0delete refs/heads/c\0{}\0",
            "1".repeat(40),
            "2".repeat(40),
            "3".repeat(40),
            "4".repeat(40)
        );
        assert_eq!(txn.to_bytes(), expected.into_bytes());
    }

    fn two_commits(t: &TestRepo) -> (Oid, Oid) {
        t.write("a", b"1\n");
        let first = t.commit_all("first");
        t.write("a", b"2\n");
        let second = t.commit_all("second");
        (first, second)
    }

    #[test]
    fn branches_are_created_moved_and_deleted_with_their_old_values_checked() {
        let t = TestRepo::init();
        let (first, second) = two_commits(&t);
        let repo = t.repo();
        let b = BranchName::parse("zen/work").unwrap();

        repo.create_branch(&b, &first).unwrap();
        assert_eq!(repo.ref_target(&b.to_ref()).unwrap(), Some(first.clone()));
        assert!(matches!(
            repo.create_branch(&b, &second),
            Err(GitError::StaleRef { .. })
        ));

        assert!(matches!(
            repo.move_branch(&b, &second, &first),
            Err(GitError::StaleRef { .. })
        ));
        repo.move_branch(&b, &first, &second).unwrap();
        assert_eq!(repo.ref_target(&b.to_ref()).unwrap(), Some(second.clone()));

        assert!(matches!(
            repo.delete_branch(&b, &first),
            Err(GitError::StaleRef { .. })
        ));
        repo.delete_branch(&b, &second).unwrap();
        assert_eq!(repo.ref_target(&b.to_ref()).unwrap(), None);
    }

    /// **All or nothing.** One stale op fails the transaction and every other
    /// op in it is left unapplied.
    #[test]
    fn one_stale_op_applies_nothing() {
        let t = TestRepo::init();
        let (first, second) = two_commits(&t);
        let repo = t.repo();
        let a = BranchName::parse("a").unwrap();
        repo.create_branch(&a, &first).unwrap();

        let txn = RefTransaction::new()
            .push(RefOp::Create {
                name: BranchName::parse("fresh").unwrap().to_ref(),
                new: second.clone(),
            })
            .push(RefOp::Update {
                name: a.to_ref(),
                new: second.clone(),
                old: second.clone(), // wrong: `a` holds `first`
            });
        match repo.update_refs(&txn) {
            Err(GitError::StaleRef { name, .. }) => assert_eq!(name, a.to_ref()),
            other => panic!("{other:?}"),
        }
        assert_eq!(repo.ref_target(&a.to_ref()).unwrap(), Some(first));
        assert_eq!(
            repo.ref_target(&BranchName::parse("fresh").unwrap().to_ref())
                .unwrap(),
            None
        );
    }

    #[test]
    fn a_held_lock_is_ref_locked() {
        let t = TestRepo::init();
        let (first, second) = two_commits(&t);
        let repo = t.repo();
        let b = BranchName::parse("locked").unwrap();
        repo.create_branch(&b, &first).unwrap();
        std::fs::write(t.path.join(".git/refs/heads/locked.lock"), b"").unwrap();
        assert!(matches!(
            repo.move_branch(&b, &first, &second),
            Err(GitError::RefLocked { .. })
        ));
    }

    /// A branch checked out in a linked worktree is as off-limits as the
    /// main checkout's.
    #[test]
    fn a_branch_checked_out_in_another_worktree_is_never_moved() {
        let t = TestRepo::init();
        let (first, second) = two_commits(&t);
        let repo = t.repo();
        let dir = tempfile::Builder::new()
            .prefix("wt-")
            .tempdir_in(scratch())
            .unwrap();
        let build = BranchName::parse("zen/build").unwrap();
        repo.add_worktree(
            &dir.path().join("build"),
            &WorktreeCheckout::NewBranch {
                branch: build.clone(),
                at: first.clone(),
            },
        )
        .unwrap();
        assert!(matches!(
            repo.move_branch(&build, &first, &second),
            Err(GitError::CheckedOutBranch { .. })
        ));
        assert!(matches!(
            repo.delete_branch(&build, &first),
            Err(GitError::CheckedOutBranch { .. })
        ));
    }

    #[test]
    fn the_checked_out_branch_is_never_moved() {
        let t = TestRepo::init();
        let (first, second) = two_commits(&t);
        let main = BranchName::parse("main").unwrap();
        assert!(matches!(
            t.repo().move_branch(&main, &second, &first),
            Err(GitError::CheckedOutBranch { .. })
        ));
        assert_eq!(t.oid("HEAD"), second);
    }

    /// **A symbolic ref is no way round the checked-out guard.** `alias`
    /// points at the checked-out `main`; changing or deleting `alias` must
    /// leave `main` where it was.
    #[test]
    fn an_alias_of_the_checked_out_branch_never_moves_it() {
        let t = TestRepo::init();
        let (first, second) = two_commits(&t);
        t.git(&["symbolic-ref", "refs/heads/alias", "refs/heads/main"]);
        let repo = t.repo();
        let alias = BranchName::parse("alias").unwrap();
        let main = BranchName::parse("main").unwrap().to_ref();

        let _ = repo.move_branch(&alias, &second, &first);
        assert_eq!(repo.ref_target(&main).unwrap(), Some(second.clone()));
        let _ = repo.delete_branch(&alias, &second);
        assert_eq!(repo.ref_target(&main).unwrap(), Some(second));
    }
}
