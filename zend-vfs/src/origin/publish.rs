//! Publishing a branch's new tip: origin first, the local branch after.

use super::pull::Pulled;
use crate::error::GitError;
use crate::remote::push::{PushOutcome, PushSpec, Rejection};
use crate::types::{Oid, RemoteName, Rev};
use crate::Repo;

/// What publishing did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Published {
    /// Origin took it — created, fast-forwarded, or moved under the lease —
    /// and the local branch followed.
    Pushed(PushOutcome),
    /// There is no origin: the local branch moved, and is the record.
    Local,
    /// Origin refused it, and nothing changed locally.
    Refused(Rejection),
    /// A commit that does not descend from what the branch's record holds:
    /// publishing it would take someone else's commits off the branch, so
    /// nothing was written anywhere. The commits meet in a merge first.
    Behind,
}

impl Published {
    /// Whether the branch now holds what was published.
    pub fn landed(&self) -> bool {
        matches!(self, Published::Pushed(_) | Published::Local)
    }
}

impl Repo {
    /// Make `new` the tip of `pulled`'s branch, wherever it is: on origin,
    /// under the lease the pull found, then locally. This is how a branch is
    /// moved on purpose — made, rewound, put anywhere; a commit is published
    /// with [`Self::publish_commit`].
    ///
    /// With an origin, the local branch is a cache of it and follows only
    /// when that takes nothing of its own off it — a local branch holding
    /// commits origin never had keeps them, and stays where it is.
    pub fn publish_branch(&self, pulled: &Pulled, new: &Oid) -> Result<Published, GitError> {
        let published = self.push_tip(pulled, new)?;
        if !published.landed() {
            return Ok(published);
        }
        if pulled.origin.is_none() {
            // The local branch is the record: moving it is the write.
            self.follow_locally(pulled, new)?;
        } else if !self.holds_its_own(pulled)? {
            self.follow_cache(pulled, new);
        }
        Ok(published)
    }

    /// Publish `new` as the branch's next commit — the one step that makes a
    /// commit real. It must descend from what the branch's record holds, so
    /// that nothing anyone else published is lost ([`Published::Behind`]
    /// otherwise, with nothing written); then it goes to origin under the
    /// lease the pull found, and the local branch follows when that takes
    /// nothing off it either. A local branch holding commits `new` does not
    /// — made on this machine and never published — is left where it is,
    /// commits and all.
    pub fn publish_commit(&self, pulled: &Pulled, new: &Oid) -> Result<Published, GitError> {
        let descends =
            |from: &Oid| self.is_ancestor(&Rev::Oid(from.clone()), &Rev::Oid(new.clone()));
        if let Some(record) = pulled.record() {
            if !descends(record)? {
                return Ok(Published::Behind);
            }
        }
        let published = self.push_tip(pulled, new)?;
        if !published.landed() {
            return Ok(published);
        }
        if pulled.origin.is_none() {
            // The local branch is the record, and `new` descends from it.
            self.follow_locally(pulled, new)?;
            return Ok(published);
        }
        let takes_nothing = match &pulled.tip {
            Some(tip) => descends(tip)?,
            None => true,
        };
        if takes_nothing {
            self.follow_cache(pulled, new);
        }
        Ok(published)
    }

    /// Remove `pulled`'s branch: from origin, under the lease the pull
    /// found, then locally — unless the local branch holds commits origin
    /// never had, which keeps it.
    pub fn unpublish_branch(&self, pulled: &Pulled) -> Result<Published, GitError> {
        let published = match (&pulled.origin, &pulled.on_origin) {
            (Some(origin), Some(held)) => {
                let spec = PushSpec::delete_branch(pulled.branch.clone(), held.clone());
                match self.push_one(origin, spec)? {
                    PushOutcome::Rejected(why) => return Ok(Published::Refused(why)),
                    outcome => Published::Pushed(outcome),
                }
            }
            _ => Published::Local,
        };
        let Some(tip) = &pulled.tip else {
            return Ok(published);
        };
        if pulled.origin.is_none() {
            self.force_delete_branch(&pulled.branch, tip)?;
        } else if !self.holds_its_own(pulled)? {
            if let Err(e) = self.force_delete_branch(&pulled.branch, tip) {
                tracing::warn!(
                    "{} is gone from origin; its local copy could not be removed: {e}",
                    pulled.branch
                );
            }
        }
        Ok(published)
    }

    /// Whether the local branch holds commits origin's copy does not — made
    /// on this machine and never published.
    fn holds_its_own(&self, pulled: &Pulled) -> Result<bool, GitError> {
        match (&pulled.tip, &pulled.on_origin) {
            (None, _) => Ok(false),
            (Some(_), None) => Ok(true),
            (Some(tip), Some(held)) => {
                Ok(!self.is_ancestor(&Rev::Oid(tip.clone()), &Rev::Oid(held.clone()))?)
            }
        }
    }

    /// Bring the local branch — a cache of origin's, which has just taken
    /// `new` — up to it. Origin holding it is what counts: a local branch
    /// that could not follow is reported, never the write's failure, and the
    /// next pull brings it up.
    fn follow_cache(&self, pulled: &Pulled, new: &Oid) {
        if let Err(e) = self.follow_locally(pulled, new) {
            tracing::warn!(
                "origin holds {new} on {}; the local branch could not follow yet: {e}",
                pulled.branch
            );
        }
    }

    /// Put `new` on origin's copy of the branch under the pull's lease; with
    /// no origin, nothing to do.
    fn push_tip(&self, pulled: &Pulled, new: &Oid) -> Result<Published, GitError> {
        let Some(origin) = &pulled.origin else {
            return Ok(Published::Local);
        };
        let spec = PushSpec::branch(new.clone(), pulled.branch.clone(), pulled.lease());
        Ok(match self.push_one(origin, spec)? {
            PushOutcome::Rejected(why) => Published::Refused(why),
            outcome => Published::Pushed(outcome),
        })
    }

    /// The outcome of pushing `spec` alone.
    fn push_one(&self, origin: &RemoteName, spec: PushSpec) -> Result<PushOutcome, GitError> {
        Ok(self
            .push(origin, &[spec])?
            .into_iter()
            .next()
            .ok_or_else(|| GitError::malformed("push", "no result for the branch"))?
            .outcome)
    }

    /// Move the local branch to `new` from where the pull left it, tracking
    /// origin's copy so its distance from it reads from local refs alone.
    fn follow_locally(&self, pulled: &Pulled, new: &Oid) -> Result<(), GitError> {
        if pulled.tip.as_ref() != Some(new) {
            self.force_branch_tip(&pulled.branch, pulled.tip.as_ref(), new)?;
        }
        if let Some(origin) = &pulled.origin {
            self.set_upstream(&pulled.branch, origin, &pulled.branch)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestRepo;
    use crate::types::BranchName;
    use crate::{ChangeSet, RepoPath};

    fn main() -> BranchName {
        BranchName::parse("main").unwrap()
    }

    fn origin_holds(origin: &TestRepo, branch: &str) -> Option<String> {
        let out = origin.git(&[
            "for-each-ref",
            "--format=%(objectname)",
            &format!("refs/heads/{branch}"),
        ]);
        let out = out.trim();
        (!out.is_empty()).then(|| out.to_string())
    }

    /// A commit writing `path` as `content` on top of `onto`.
    fn commit_on(repo: &Repo, onto: &Oid, path: &str, content: &str) -> Oid {
        let mut set = ChangeSet::new();
        set.write(
            RepoPath::parse(path).unwrap(),
            content.as_bytes().to_vec(),
            None,
        )
        .unwrap();
        let me = repo.identity().unwrap();
        repo.commit_changes(onto, &set, "write", &me, &me).unwrap()
    }

    fn with_origin() -> (TestRepo, TestRepo) {
        let origin = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a.txt", b"one\n");
        t.commit_all("first");
        t.git(&["remote", "add", "origin", &origin.url()]);
        t.git(&["push", "-q", "origin", "main"]);
        (origin, t)
    }

    /// **A commit lands on origin first, and the local branch follows** —
    /// its tracking ref too.
    #[test]
    fn a_commit_lands_on_origin_and_the_local_branch_follows() {
        let (origin, t) = with_origin();
        let repo = t.repo();
        let pulled = repo.pull_branch(&main()).unwrap();
        let commit = commit_on(&repo, pulled.tip.as_ref().unwrap(), "b.txt", "b\n");
        let published = repo.publish_commit(&pulled, &commit).unwrap();
        assert_eq!(published, Published::Pushed(PushOutcome::FastForward));
        assert_eq!(
            origin_holds(&origin, "main").as_deref(),
            Some(commit.as_str())
        );
        assert_eq!(t.oid("refs/heads/main"), commit);
        assert_eq!(t.oid("refs/remotes/origin/main"), commit);
        assert_eq!(
            t.git(&["rev-parse", "--abbrev-ref", "main@{upstream}"])
                .trim(),
            "origin/main",
            "the local branch tracks origin's"
        );
    }

    /// **A commit that does not descend from origin's copy is refused, with
    /// nothing written anywhere** — someone else's commit is never taken off
    /// the branch.
    #[test]
    fn a_commit_behind_origin_is_refused_whole() {
        let (origin, t) = with_origin();
        let repo = t.repo();
        let first = t.oid("main");
        let theirs = commit_on(&repo, &first, "theirs.txt", "theirs\n");
        t.git(&["push", "-q", "origin", &format!("{theirs}:refs/heads/main")]);

        let pulled = repo.pull_branch(&main()).unwrap();
        let mine = commit_on(&repo, &first, "mine.txt", "mine\n");
        assert_eq!(
            repo.publish_commit(&pulled, &mine).unwrap(),
            Published::Behind
        );
        assert_eq!(
            origin_holds(&origin, "main").as_deref(),
            Some(theirs.as_str())
        );
        assert_eq!(t.oid("main"), theirs, "the pull's fast-forward only");
    }

    /// **Origin moving between the pull and the push refuses the commit**,
    /// with nothing changed locally.
    #[test]
    fn origin_moving_under_a_commit_refuses_it() {
        let (origin, t) = with_origin();
        let repo = t.repo();
        let first = t.oid("main");
        let pulled = repo.pull_branch(&main()).unwrap();
        let theirs = commit_on(&repo, &first, "theirs.txt", "theirs\n");
        t.git(&["push", "-q", "origin", &format!("{theirs}:refs/heads/main")]);
        let mine = commit_on(&repo, &first, "mine.txt", "mine\n");
        assert_eq!(
            repo.publish_commit(&pulled, &mine).unwrap(),
            Published::Refused(Rejection::Stale)
        );
        assert_eq!(t.oid("main"), first);
        assert_eq!(
            origin_holds(&origin, "main").as_deref(),
            Some(theirs.as_str())
        );
    }

    /// **A local branch holding commits origin never had keeps them**: the
    /// commit lands on origin, and the local branch is left where it was.
    #[test]
    fn local_commits_origin_never_had_are_kept() {
        let (origin, t) = with_origin();
        let repo = t.repo();
        let first = t.oid("main");
        t.write("local.txt", b"local\n");
        let local = t.commit_all("local only");
        let pulled = repo.pull_branch(&main()).unwrap();
        let mine = commit_on(&repo, &first, "mine.txt", "mine\n");
        assert!(repo.publish_commit(&pulled, &mine).unwrap().landed());
        assert_eq!(
            origin_holds(&origin, "main").as_deref(),
            Some(mine.as_str())
        );
        assert_eq!(t.oid("main"), local, "the local commit is still on it");
    }

    /// **A rewind is published under the lease** — and refused, with nothing
    /// changed locally, when origin holds something else.
    #[test]
    fn a_rewind_is_published_under_the_lease() {
        let (origin, t) = with_origin();
        let repo = t.repo();
        let first = t.oid("main");
        let pulled = repo.pull_branch(&main()).unwrap();
        let second = commit_on(&repo, &first, "b.txt", "b\n");
        assert!(repo.publish_commit(&pulled, &second).unwrap().landed());

        let pulled = repo.pull_branch(&main()).unwrap();
        assert_eq!(
            repo.publish_branch(&pulled, &first).unwrap(),
            Published::Pushed(PushOutcome::Forced)
        );
        assert_eq!(t.oid("main"), first);

        // A stale pull: origin moved since.
        let stale = repo.pull_branch(&main()).unwrap();
        let third = commit_on(&repo, &first, "c.txt", "c\n");
        t.git(&["push", "-q", "origin", &format!("{third}:refs/heads/main")]);
        assert_eq!(
            repo.publish_branch(&stale, &second).unwrap(),
            Published::Refused(Rejection::Stale)
        );
        assert_eq!(t.oid("main"), first, "nothing moved locally");
        assert_eq!(
            origin_holds(&origin, "main").as_deref(),
            Some(third.as_str())
        );
    }

    /// **Without origin, a commit moves the local branch alone** — and one
    /// that does not descend from it is refused.
    #[test]
    fn without_origin_a_commit_is_local() {
        let t = TestRepo::init();
        t.write("a.txt", b"one\n");
        let first = t.commit_all("first");
        let repo = t.repo();
        let pulled = repo.pull_branch(&main()).unwrap();
        let commit = commit_on(&repo, &first, "b.txt", "b\n");
        assert_eq!(
            repo.publish_commit(&pulled, &commit).unwrap(),
            Published::Local
        );
        assert_eq!(t.oid("main"), commit);

        let pulled = repo.pull_branch(&main()).unwrap();
        let sideways = commit_on(&repo, &first, "c.txt", "c\n");
        assert_eq!(
            repo.publish_commit(&pulled, &sideways).unwrap(),
            Published::Behind
        );
        assert_eq!(t.oid("main"), commit);
    }

    /// **A local branch holding commits origin never had keeps them through
    /// a rewind and a delete**: origin moves, the local branch stays.
    #[test]
    fn local_commits_survive_a_rewind_and_a_delete() {
        let (origin, t) = with_origin();
        let repo = t.repo();
        let first = t.oid("main");
        let pulled = repo.pull_branch(&main()).unwrap();
        let second = commit_on(&repo, &first, "b.txt", "b\n");
        assert!(repo.publish_commit(&pulled, &second).unwrap().landed());
        t.git(&["reset", "-q", "--hard", "main"]);
        t.write("local.txt", b"never pushed\n");
        let local = t.commit_all("local only");

        let pulled = repo.pull_branch(&main()).unwrap();
        assert!(repo.publish_branch(&pulled, &first).unwrap().landed());
        assert_eq!(
            origin_holds(&origin, "main").as_deref(),
            Some(first.as_str())
        );
        assert_eq!(t.oid("main"), local, "the local commit stays on its branch");

        let topic = BranchName::parse("topic").unwrap();
        let pulled = repo.pull_branch(&topic).unwrap();
        assert!(repo.publish_branch(&pulled, &first).unwrap().landed());
        t.git(&["branch", "-f", "topic", local.as_str()]);
        let pulled = repo.pull_branch(&topic).unwrap();
        assert!(repo.unpublish_branch(&pulled).unwrap().landed());
        assert_eq!(origin_holds(&origin, "topic"), None);
        assert_eq!(t.oid("topic"), local, "kept locally, commits and all");
    }

    /// **A local branch that cannot follow never fails a push origin took**:
    /// the commit has landed, and the next pull brings the branch up.
    #[test]
    fn a_local_branch_that_cannot_follow_does_not_fail_the_push() {
        let (origin, t) = with_origin();
        let repo = t.repo();
        let first = t.oid("main");
        let pulled = repo.pull_branch(&main()).unwrap();
        let commit = commit_on(&repo, &first, "b.txt", "b\n");
        // The local branch moves under the publish, to a commit origin has.
        let elsewhere = commit_on(&repo, &first, "c.txt", "c\n");
        t.git(&["update-ref", "refs/heads/main", elsewhere.as_str()]);
        assert_eq!(
            repo.publish_commit(&pulled, &commit).unwrap(),
            Published::Pushed(PushOutcome::FastForward)
        );
        assert_eq!(
            origin_holds(&origin, "main").as_deref(),
            Some(commit.as_str())
        );
    }

    /// **A branch is deleted from origin, then locally** — even while it is
    /// checked out, which here is only ever the sandbox's own checkout.
    #[test]
    fn a_branch_is_deleted_from_origin_then_locally() {
        let (origin, t) = with_origin();
        let repo = t.repo();
        let topic = BranchName::parse("topic").unwrap();
        let first = t.oid("main");
        let pulled = repo.pull_branch(&topic).unwrap();
        assert!(repo.publish_branch(&pulled, &first).unwrap().landed());
        assert!(origin_holds(&origin, "topic").is_some());

        t.git(&["checkout", "-q", "topic"]);
        let pulled = repo.pull_branch(&topic).unwrap();
        assert!(repo.unpublish_branch(&pulled).unwrap().landed());
        assert_eq!(origin_holds(&origin, "topic"), None);
        assert_eq!(repo.ref_target(&topic.to_ref()).unwrap(), None);
        assert!(repo.resolve(&Rev::Branch(main())).is_ok());
    }
}
