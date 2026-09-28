//! A repository's branches as its record holds them.
//!
//! With an `origin` remote the record is origin's copy of each branch —
//! `refs/remotes/origin/<b>` as the last fetch left it — and with none it is
//! the local branches (`docs/zend_git.md` §7.8). Reading it touches no
//! network: the origin watcher keeps the tracking refs current.

use crate::error::GitError;
use crate::types::{BranchName, Oid, Rev};
use crate::Repo;

/// One branch on the record, and the commit the record holds for it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordBranch {
    pub name: BranchName,
    pub tip: Oid,
}

impl Repo {
    /// Every branch on the record, in name order: origin's branches when the
    /// repository has an origin, its local branches when it has none. A local
    /// branch origin has never had is not on the record, and neither is a
    /// branch of any other remote.
    pub fn record_branches(&self) -> Result<Vec<RecordBranch>, GitError> {
        let mut out: Vec<RecordBranch> = match self.origin()? {
            Some(origin) => self
                .remote_branches()?
                .into_iter()
                .filter(|b| b.remote == origin)
                .map(|b| RecordBranch {
                    name: b.branch,
                    tip: b.oid,
                })
                .collect(),
            None => self
                .branches()?
                .into_iter()
                .map(|b| RecordBranch {
                    name: b.name,
                    tip: b.oid,
                })
                .collect(),
        };
        out.sort_by(|a, b| a.name.as_str().cmp(b.name.as_str()));
        Ok(out)
    }

    /// Where the record holds `branch`: origin's copy when there is one, the
    /// local branch otherwise — what a conversation reading the branch pins.
    /// `None` when neither exists.
    pub fn record_tip(&self, branch: &BranchName) -> Result<Option<Oid>, GitError> {
        if let Some(origin) = self.origin()? {
            if let Some(tip) = self.ref_target(&origin.tracking(branch))? {
                return Ok(Some(tip));
            }
        }
        self.ref_target(&branch.to_ref())
    }

    /// Whether `commit` is on the record's copy of `branch` — the tip or
    /// something in its history.
    pub(crate) fn on_record(&self, branch: &BranchName, commit: &Oid) -> Result<bool, GitError> {
        match self.record_tip(branch)? {
            Some(tip) if &tip == commit => Ok(true),
            Some(tip) => self.is_ancestor(&Rev::Oid(commit.clone()), &Rev::Oid(tip)),
            None => Ok(false),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::fetch::FetchSpec;
    use crate::testing::TestRepo;
    use crate::RemoteName;

    fn names(branches: &[RecordBranch]) -> Vec<&str> {
        branches.iter().map(|b| b.name.as_str()).collect()
    }

    /// **With an origin, the record is origin's branches** — a local branch
    /// origin never had is not on it, and neither is another remote's.
    #[test]
    fn with_an_origin_the_record_is_origins_branches() {
        let origin = TestRepo::bare();
        let other = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a", b"a\n");
        let first = t.commit_all("first");
        t.git(&["remote", "add", "origin", &origin.url()]);
        t.git(&["remote", "add", "upstream", &other.url()]);
        t.git(&["push", "-q", "origin", "main", "main:zen/work"]);
        t.git(&["push", "-q", "upstream", "main:trunk"]);
        t.git(&["branch", "local-only"]);
        let repo = t.repo();
        for r in ["origin", "upstream"] {
            repo.fetch(&RemoteName::parse(r).unwrap(), &FetchSpec::AllBranches)
                .unwrap();
        }
        let record = repo.record_branches().unwrap();
        assert_eq!(names(&record), ["main", "zen/work"]);
        assert!(record.iter().all(|b| b.tip == first));
    }

    /// **With no origin, the record is the local branches.**
    #[test]
    fn with_no_origin_the_record_is_the_local_branches() {
        let t = TestRepo::init();
        t.write("a", b"a\n");
        t.commit_all("first");
        t.git(&["branch", "topic"]);
        assert_eq!(
            names(&t.repo().record_branches().unwrap()),
            ["main", "topic"]
        );
    }

    /// **The record's tip is origin's copy when origin has the branch**, even
    /// when the local branch is behind it or ahead of it; the local branch
    /// when origin does not; nothing when neither has it.
    #[test]
    fn the_record_tip_is_origins_copy_when_origin_has_the_branch() {
        let origin = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a", b"one\n");
        let first = t.commit_all("first");
        t.git(&["remote", "add", "origin", &origin.url()]);
        t.git(&["push", "-q", "origin", "main"]);
        t.write("a", b"two\n");
        let second = t.commit_all("second");
        t.git(&["push", "-q", "origin", "main"]);
        t.git(&["reset", "-q", "--hard", first.as_str()]);
        let repo = t.repo();
        let main = BranchName::parse("main").unwrap();
        assert_eq!(repo.record_tip(&main).unwrap(), Some(second.clone()));
        assert!(repo.on_record(&main, &first).unwrap(), "in its history");
        assert!(repo.on_record(&main, &second).unwrap(), "its tip");

        t.git(&["branch", "local-only"]);
        let local = BranchName::parse("local-only").unwrap();
        assert_eq!(repo.record_tip(&local).unwrap(), Some(first));
        let nowhere = BranchName::parse("nowhere").unwrap();
        assert_eq!(repo.record_tip(&nowhere).unwrap(), None);
        assert!(!repo.on_record(&nowhere, &second).unwrap());
    }
}
