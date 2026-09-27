//! Bringing a local branch up to what origin holds.

use crate::error::GitError;
use crate::remote::fetch::FetchSpec;
use crate::remote::push::Lease;
use crate::types::{BranchName, Oid, RemoteName, Rev};
use crate::Repo;

/// What a pull found, and left the local branch at.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pulled {
    pub branch: BranchName,
    /// The repository's origin; `None` when it has none, and its record is
    /// local.
    pub origin: Option<RemoteName>,
    /// What origin held for the branch when fetched; `None` when origin does
    /// not have it, or there is no origin.
    pub on_origin: Option<Oid>,
    /// What the local branch holds now. `None` when the branch exists
    /// nowhere yet.
    pub tip: Option<Oid>,
    /// The local branch holds commits origin does not, and origin commits
    /// the local branch does not: it was left as it was.
    pub diverged: bool,
}

impl Pulled {
    /// The lease a publish swaps against: origin still holding what this pull
    /// found there.
    pub fn lease(&self) -> Lease {
        match &self.on_origin {
            Some(oid) => Lease::Expect(oid.clone()),
            None => Lease::Absent,
        }
    }

    /// The branch as its record holds it: origin's copy when there is an
    /// origin, the local branch when there is none.
    pub fn record(&self) -> Option<&Oid> {
        match &self.origin {
            Some(_) => self.on_origin.as_ref(),
            None => self.tip.as_ref(),
        }
    }
}

impl Repo {
    /// Fetch `branch` from origin and bring the local branch up to it by
    /// fast-forward: made from origin's when it does not exist locally,
    /// moved when it is behind, left when it is level or ahead (the next
    /// publish carries its commits). A branch that has diverged from origin's
    /// is left as it is and reported — its commits and origin's meet only in
    /// a merge.
    pub fn pull_branch(&self, branch: &BranchName) -> Result<Pulled, GitError> {
        let local = self.ref_target(&branch.to_ref())?;
        let Some(origin) = self.origin()? else {
            return Ok(Pulled {
                branch: branch.clone(),
                origin: None,
                on_origin: None,
                tip: local,
                diverged: false,
            });
        };
        let on_origin = match self.fetch(&origin, &FetchSpec::Branch(branch.clone())) {
            Ok(_) => self.ref_target(&origin.tracking(branch))?,
            Err(GitError::UnknownRevision { .. }) => None,
            Err(e) => return Err(e),
        };
        let mut pulled = Pulled {
            branch: branch.clone(),
            origin: Some(origin),
            on_origin: on_origin.clone(),
            tip: local.clone(),
            diverged: false,
        };
        let (Some(remote), local) = (on_origin, local) else {
            return Ok(pulled);
        };
        let Some(local) = local else {
            self.force_branch_tip(branch, None, &remote)?;
            if let Some(origin) = &pulled.origin {
                self.set_upstream(branch, origin, branch)?;
            }
            pulled.tip = Some(remote);
            return Ok(pulled);
        };
        let (l, r) = (Rev::Oid(local.clone()), Rev::Oid(remote.clone()));
        if local == remote || self.is_ancestor(&r, &l)? {
            return Ok(pulled);
        }
        if self.is_ancestor(&l, &r)? {
            self.force_branch_tip(branch, Some(&local), &remote)?;
            pulled.tip = Some(remote);
            return Ok(pulled);
        }
        pulled.diverged = true;
        Ok(pulled)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;

    fn main() -> BranchName {
        BranchName::parse("main").unwrap()
    }

    /// A repository on `main` with one commit, pushed to a bare origin, and
    /// a second clone of that origin standing in for someone else.
    fn pair() -> (TestRepo, TestRepo, TestRepo, Oid) {
        let origin = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a.txt", b"one\n");
        let first = t.commit_all("first");
        t.git(&["remote", "add", "origin", &origin.url()]);
        t.git(&["push", "-q", "origin", "main"]);
        let other = TestRepo::init();
        other.git(&["remote", "add", "origin", &origin.url()]);
        other.git(&["fetch", "-q", "origin"]);
        other.git(&["reset", "-q", "--hard", "origin/main"]);
        (origin, t, other, first)
    }

    /// **With no origin, the local branch is the record.**
    #[test]
    fn without_origin_the_local_branch_is_read() {
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let first = t.commit_all("first");
        let pulled = t.repo().pull_branch(&main()).unwrap();
        assert_eq!(pulled.origin, None);
        assert_eq!(pulled.tip, Some(first.clone()));
        assert_eq!(pulled.record(), Some(&first));
        assert_eq!(pulled.lease(), Lease::Absent);
    }

    /// **Behind origin, the local branch fast-forwards**, and the lease is
    /// what origin holds.
    #[test]
    fn a_branch_behind_origin_fast_forwards() {
        let (_origin, t, other, _) = pair();
        other.write("a.txt", b"two\n");
        let theirs = other.commit_all("theirs");
        other.git(&["push", "-q", "origin", "main"]);

        let pulled = t.repo().pull_branch(&main()).unwrap();
        assert_eq!(pulled.tip, Some(theirs.clone()));
        assert_eq!(pulled.record(), Some(&theirs));
        assert_eq!(pulled.lease(), Lease::Expect(theirs.clone()));
        assert!(!pulled.diverged);
        assert_eq!(t.oid("refs/heads/main"), theirs);
    }

    /// **Ahead of origin, the local branch is left for the publish to
    /// carry**; a branch origin does not have is left as it is too.
    #[test]
    fn a_branch_ahead_or_unknown_to_origin_is_left() {
        let (_origin, t, _other, first) = pair();
        t.write("a.txt", b"mine\n");
        let mine = t.commit_all("mine");
        let pulled = t.repo().pull_branch(&main()).unwrap();
        assert_eq!(pulled.tip, Some(mine));
        assert_eq!(pulled.lease(), Lease::Expect(first));

        t.git(&["branch", "local-only"]);
        let local_only = BranchName::parse("local-only").unwrap();
        let pulled = t.repo().pull_branch(&local_only).unwrap();
        assert_eq!(pulled.on_origin, None);
        assert_eq!(pulled.lease(), Lease::Absent);
    }

    /// **A branch that exists only on origin is made locally.**
    #[test]
    fn a_branch_only_on_origin_is_made_locally() {
        let (_origin, t, other, _) = pair();
        other.git(&["checkout", "-q", "-b", "topic"]);
        other.write("t.txt", b"t\n");
        let topic = other.commit_all("topic");
        other.git(&["push", "-q", "origin", "topic"]);
        let pulled = t
            .repo()
            .pull_branch(&BranchName::parse("topic").unwrap())
            .unwrap();
        assert_eq!(pulled.tip, Some(topic.clone()));
        assert_eq!(t.oid("refs/heads/topic"), topic);
    }

    /// **A diverged branch is left exactly as it was** — no commit of either
    /// side rewritten — and reported, with origin's copy as its record.
    #[test]
    fn a_diverged_branch_is_left_and_reported() {
        let (_origin, t, other, _) = pair();
        other.write("a.txt", b"theirs\n");
        let theirs = other.commit_all("theirs");
        other.git(&["push", "-q", "origin", "main"]);
        t.write("a.txt", b"mine\n");
        let mine = t.commit_all("mine");

        let pulled = t.repo().pull_branch(&main()).unwrap();
        assert!(pulled.diverged);
        assert_eq!(pulled.tip, Some(mine.clone()));
        assert_eq!(pulled.record(), Some(&theirs));
        assert_eq!(
            t.oid("refs/heads/main"),
            mine,
            "the local branch is untouched"
        );
    }
}
