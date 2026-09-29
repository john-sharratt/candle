//! Bringing a local branch up to a commit the record already holds.
//!
//! A conversation pins its branch's record — origin's copy — so its base can
//! be ahead of the local branch when origin moved since the local branch was
//! last written. Before a checkout is put on the conversation's branch, the
//! local branch follows: it is a cache of origin's, and origin already has
//! the commit, so moving it forward loses nothing and publishes nothing.

use crate::error::GitError;
use crate::types::{BranchName, Oid, Rev};
use crate::Repo;

/// What [`Repo::follow_record`] did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Followed {
    /// The local branch already held the commit.
    Level,
    /// The local branch was moved forward to the commit, or made there.
    Moved { from: Option<Oid> },
    /// The local branch was left as it was: the commit is not on the record,
    /// or the local branch holds commits the commit does not.
    Refused,
}

impl Repo {
    /// Move the local `branch` forward to `to` when the record holds `to`
    /// and the local branch is in `to`'s history — creating it there, with
    /// origin as its upstream, when it does not exist. The move is a
    /// compare-and-swap on the local branch as read here. Anything else is
    /// [`Followed::Refused`] and changes nothing: a commit the record lacks
    /// is not the record's to hand out, and a local branch holding commits
    /// of its own is never moved off them.
    pub fn follow_record(&self, branch: &BranchName, to: &Oid) -> Result<Followed, GitError> {
        let local = self.ref_target(&branch.to_ref())?;
        if local.as_ref() == Some(to) {
            return Ok(Followed::Level);
        }
        if !self.on_record(branch, to)? {
            return Ok(Followed::Refused);
        }
        match local {
            None => {
                self.force_branch_tip(branch, None, to)?;
                if let Some(origin) = self.origin()? {
                    self.set_upstream(branch, &origin, branch)?;
                }
                Ok(Followed::Moved { from: None })
            }
            Some(from) => {
                if !self.is_ancestor(&Rev::Oid(from.clone()), &Rev::Oid(to.clone()))? {
                    return Ok(Followed::Refused);
                }
                self.force_branch_tip(branch, Some(&from), to)?;
                Ok(Followed::Moved { from: Some(from) })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;

    fn main() -> BranchName {
        BranchName::parse("main").unwrap()
    }

    /// A repository whose origin holds `first` then `second` on `main`, with
    /// the local `main` left at `first`.
    fn behind() -> (TestRepo, TestRepo, Oid, Oid) {
        let origin = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a", b"one\n");
        let first = t.commit_all("first");
        t.git(&["remote", "add", "origin", &origin.url()]);
        t.git(&["push", "-q", "origin", "main"]);
        t.write("a", b"two\n");
        let second = t.commit_all("second");
        t.git(&["push", "-q", "origin", "main"]);
        t.git(&["checkout", "-q", "--detach"]);
        t.git(&["branch", "-f", "main", first.as_str()]);
        (t, origin, first, second)
    }

    /// **A local branch behind the record is moved forward to it.**
    #[test]
    fn a_branch_behind_the_record_moves_forward() {
        let (t, _origin, first, second) = behind();
        let repo = t.repo();
        assert_eq!(
            repo.follow_record(&main(), &second).unwrap(),
            Followed::Moved { from: Some(first) }
        );
        assert_eq!(
            repo.ref_target(&main().to_ref()).unwrap(),
            Some(second.clone())
        );
        assert_eq!(
            repo.follow_record(&main(), &second).unwrap(),
            Followed::Level
        );
    }

    /// **A branch only origin has is made where the record holds it**, with
    /// origin as its upstream.
    #[test]
    fn a_branch_only_origin_has_is_made() {
        let (t, _origin, _first, second) = behind();
        // `HEAD` is detached at `second`; the local `main` was set back.
        t.git(&["push", "-q", "origin", "HEAD:refs/heads/topic"]);
        let repo = t.repo();
        let topic = BranchName::parse("topic").unwrap();
        assert_eq!(
            repo.follow_record(&topic, &second).unwrap(),
            Followed::Moved { from: None }
        );
        assert_eq!(repo.ref_target(&topic.to_ref()).unwrap(), Some(second));
        let upstream = t.git(&["rev-parse", "--abbrev-ref", "topic@{upstream}"]);
        assert_eq!(upstream.trim(), "origin/topic");
    }

    /// **A commit the record does not hold is refused**, and nothing moves.
    #[test]
    fn a_commit_off_the_record_is_refused() {
        let (t, _origin, first, _second) = behind();
        t.git(&["checkout", "-q", "-b", "elsewhere", first.as_str()]);
        t.write("b", b"b\n");
        let elsewhere = t.commit_all("not pushed");
        let repo = t.repo();
        assert_eq!(
            repo.follow_record(&main(), &elsewhere).unwrap(),
            Followed::Refused
        );
        assert_eq!(repo.ref_target(&main().to_ref()).unwrap(), Some(first));
    }

    /// **A local branch holding commits of its own is never moved off them.**
    #[test]
    fn a_diverged_local_branch_is_left_alone() {
        let (t, _origin, first, second) = behind();
        t.git(&["checkout", "-q", "main"]);
        t.write("c", b"mine\n");
        let mine = t.commit_all("local work");
        let repo = t.repo();
        assert_eq!(
            repo.follow_record(&main(), &second).unwrap(),
            Followed::Refused
        );
        assert_eq!(repo.ref_target(&main().to_ref()).unwrap(), Some(mine));
        assert_ne!(first, second);
    }
}
