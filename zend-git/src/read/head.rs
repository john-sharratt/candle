//! What the working tree's `HEAD` points at.

use crate::error::GitError;
use crate::runner::utf8;
use crate::types::{BranchName, Oid, RefName};
use crate::Repo;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Head {
    /// On a branch that has commits.
    Branch { branch: BranchName, oid: Oid },
    /// At a commit, on no branch.
    Detached(Oid),
    /// On a branch with no commits yet.
    Unborn(BranchName),
}

impl Head {
    /// The branch `HEAD` is on, if any.
    pub fn branch(&self) -> Option<&BranchName> {
        match self {
            Self::Branch { branch, .. } | Self::Unborn(branch) => Some(branch),
            Self::Detached(_) => None,
        }
    }

    /// The commit `HEAD` resolves to, if any.
    pub fn oid(&self) -> Option<&Oid> {
        match self {
            Self::Branch { oid, .. } | Self::Detached(oid) => Some(oid),
            Self::Unborn(_) => None,
        }
    }
}

impl Repo {
    pub fn head(&self) -> Result<Head, GitError> {
        let symbolic = self
            .git("symbolic-ref")
            .args(["-q", "HEAD"])
            .read_only()
            .run_accepting(&[0, 1])?;
        let branch = match symbolic.status {
            Some(0) => {
                let name = utf8("symbolic-ref", symbolic.stdout)?;
                let name = RefName::parse(name.trim_end())?;
                Some(name.branch().ok_or_else(|| {
                    GitError::malformed("symbolic-ref", format!("HEAD points at {name}"))
                })?)
            }
            _ => None,
        };
        let resolved = self
            .git("rev-parse")
            .args(["-q", "--verify", "HEAD^{commit}"])
            .read_only()
            .run_accepting(&[0, 1])?;
        let oid = match resolved.status {
            Some(0) => Some(Oid::parse(utf8("rev-parse", resolved.stdout)?.trim_end())?),
            _ => None,
        };
        match (branch, oid) {
            (Some(branch), Some(oid)) => Ok(Head::Branch { branch, oid }),
            (Some(branch), None) => Ok(Head::Unborn(branch)),
            (None, Some(oid)) => Ok(Head::Detached(oid)),
            (None, None) => Err(GitError::malformed(
                "rev-parse",
                "HEAD is neither a branch nor a commit",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;

    #[test]
    fn a_fresh_repository_is_unborn_on_main() {
        let t = TestRepo::init();
        assert_eq!(
            t.repo().head().unwrap(),
            Head::Unborn(BranchName::parse("main").unwrap())
        );
    }

    #[test]
    fn a_branch_with_commits_is_reported_with_its_commit() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let c = t.commit_all("first");
        let head = t.repo().head().unwrap();
        assert_eq!(
            head,
            Head::Branch {
                branch: BranchName::parse("main").unwrap(),
                oid: c.clone()
            }
        );
        assert_eq!(head.oid(), Some(&c));
    }

    #[test]
    fn a_detached_head_names_its_commit() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let c = t.commit_all("first");
        t.git(&["checkout", "-q", "--detach"]);
        assert_eq!(t.repo().head().unwrap(), Head::Detached(c));
    }
}
