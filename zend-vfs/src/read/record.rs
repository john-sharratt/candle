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

/// A local branch, and how much of it is on the record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LocalBranch {
    pub tip: Oid,
    /// The last commit of it the record holds: `tip` itself unless the
    /// branch holds commits origin's copy never had; then the commit those
    /// were made on, or `None` when it shares no history with origin's copy.
    pub on_record: Option<Oid>,
}

impl LocalBranch {
    /// Whether it holds commits origin's copy never had — made on this
    /// machine and never pushed.
    pub fn unpushed(&self) -> bool {
        self.on_record.as_ref() != Some(&self.tip)
    }
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

    /// The local `branch` and how much of it the record holds; `None` when
    /// there is no local branch. With no origin copy of it — no origin, or
    /// a branch origin does not have — the local branch is the record, and
    /// all of it is on it.
    pub(crate) fn local_branch(
        &self,
        branch: &BranchName,
    ) -> Result<Option<LocalBranch>, GitError> {
        let Some(tip) = self.ref_target(&branch.to_ref())? else {
            return Ok(None);
        };
        let on_origin = match self.origin()? {
            Some(origin) => self.ref_target(&origin.tracking(branch))?,
            None => None,
        };
        let (at, held) = (Rev::Oid(tip.clone()), on_origin.map(Rev::Oid));
        let on_record = match held {
            None => Some(tip.clone()),
            Some(held) if self.is_ancestor(&at, &held)? => Some(tip.clone()),
            Some(held) => self.merge_base(&at, &held)?,
        };
        Ok(Some(LocalBranch { tip, on_record }))
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

    /// **A local branch is on the record up to its own commits**: all of it
    /// when origin's copy holds it, level or ahead; up to where it left
    /// origin's copy when it holds commits origin never had, diverged or
    /// not; all of it when origin has no copy of the branch.
    #[test]
    fn a_local_branch_is_on_the_record_up_to_its_own_commits() {
        let origin = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a", b"one\n");
        let first = t.commit_all("first");
        t.git(&["remote", "add", "origin", &origin.url()]);
        t.git(&["push", "-q", "origin", "main"]);
        let repo = t.repo();
        let main = BranchName::parse("main").unwrap();
        let on_record = |b: &BranchName| repo.local_branch(b).unwrap().unwrap();

        let level = on_record(&main);
        assert_eq!(
            (level.on_record.as_ref(), level.unpushed()),
            (Some(&first), false)
        );

        t.write("a", b"two\n");
        let second = t.commit_all("second");
        t.git(&["push", "-q", "origin", "main"]);
        t.git(&["reset", "-q", "--hard", first.as_str()]);
        let behind = on_record(&main);
        assert_eq!(
            (behind.tip.clone(), behind.unpushed()),
            (first.clone(), false)
        );

        t.write("mine", b"mine\n");
        let mine = t.commit_all("mine");
        let diverged = on_record(&main);
        assert_eq!(diverged.tip, mine);
        assert_eq!(
            diverged.on_record,
            Some(first.clone()),
            "where it left origin's"
        );
        assert!(diverged.unpushed());

        t.git(&["reset", "-q", "--hard", second.as_str()]);
        t.write("mine", b"mine again\n");
        let ahead_tip = t.commit_all("mine again");
        let ahead = on_record(&main);
        assert_eq!(
            ahead,
            LocalBranch {
                tip: ahead_tip,
                on_record: Some(second)
            }
        );
        assert!(ahead.unpushed());

        t.git(&["branch", "local-only"]);
        let local = on_record(&BranchName::parse("local-only").unwrap());
        assert!(
            !local.unpushed(),
            "origin has no copy: the local branch is the record"
        );
        assert_eq!(
            repo.local_branch(&BranchName::parse("nowhere").unwrap())
                .unwrap(),
            None
        );
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
