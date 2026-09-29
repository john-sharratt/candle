//! A branch's upstream: the remote branch it tracks.
//!
//! Written as the branch's own config — `branch.<name>.remote` and
//! `branch.<name>.merge` — rather than through `git branch
//! --set-upstream-to`, which refuses unless the remote-tracking ref already
//! exists. A branch zen is about to push, or one whose remote branch was
//! deleted, still gets its upstream; [`Repo::branches`] then reports it
//! `gone` until a fetch brings the tracking ref in.
//!
//! The two keys are written in the order that keeps the config coherent if
//! the second write fails: git treats a branch as having an upstream only
//! when `merge` is set, so `remote` is written first and removed last.

use crate::error::GitError;
use crate::types::{BranchName, RemoteName};
use crate::Repo;

impl Repo {
    fn require_branch(&self, branch: &BranchName) -> Result<(), GitError> {
        if self.ref_target(&branch.to_ref())?.is_none() {
            return Err(GitError::UnknownRevision {
                rev: branch.to_ref().to_string(),
            });
        }
        Ok(())
    }

    /// Make local `branch` track `remote_branch` on `remote` — what `git push
    /// -u` records — so the user's own `git status`, `pull` and `push` work on
    /// it, and [`Repo::branches`] reports how far apart the two are.
    pub fn set_upstream(
        &self,
        branch: &BranchName,
        remote: &RemoteName,
        remote_branch: &BranchName,
    ) -> Result<(), GitError> {
        let _write = self.write_lock();
        self.require_branch(branch)?;
        self.git("config")
            .arg("--end-of-options")
            .arg(format!("branch.{branch}.remote"))
            .arg(remote.as_str())
            .run_ok()?;
        self.git("config")
            .arg("--end-of-options")
            .arg(format!("branch.{branch}.merge"))
            .arg(remote_branch.to_ref().as_str())
            .run_ok()?;
        Ok(())
    }

    /// Stop `branch` tracking anything. A branch with no upstream is left as
    /// it is.
    pub fn unset_upstream(&self, branch: &BranchName) -> Result<(), GitError> {
        let _write = self.write_lock();
        self.require_branch(branch)?;
        // Exit 5: the key was not set.
        for key in ["merge", "remote"] {
            self.git("config")
                .args(["--unset", "--end-of-options"])
                .arg(format!("branch.{branch}.{key}"))
                .run_accepting(&[0, 5])?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::changeset::ChangeSet;
    use crate::error::GitError;
    use crate::read::refs::Upstream;
    use crate::remote::fetch::FetchSpec;
    use crate::remote::push::{Lease, PushOutcome, PushSpec};
    use crate::testing::TestRepo;
    use crate::types::{BranchName, GitTime, Oid, RemoteName, RepoPath, Signature};
    use crate::Repo;

    fn sig() -> Signature {
        Signature::new(
            "Ada Lovelace",
            "ada@example.com",
            GitTime::parse_raw("1700000000 +0000").unwrap(),
        )
        .unwrap()
    }

    fn upstream_of(repo: &Repo, branch: &BranchName) -> Option<Upstream> {
        repo.branches()
            .unwrap()
            .into_iter()
            .find(|b| &b.name == branch)
            .expect("the branch exists")
            .upstream
    }

    /// A commit on top of `base` changing one file, without the checkout.
    fn advance(repo: &Repo, base: &Oid, content: &[u8]) -> Oid {
        let mut c = ChangeSet::new();
        c.write(RepoPath::parse("work.txt").unwrap(), content.to_vec(), None)
            .unwrap();
        repo.commit_changes(base, &c, "advance", &sig(), &sig())
            .unwrap()
    }

    /// **A branch zen creates and pushes tracks its remote branch** exactly
    /// as one pushed with `git push -u` would: the config keys git reads, and
    /// ahead/behind counts that follow local commits and fetched remote ones.
    #[test]
    fn a_pushed_branch_tracks_its_remote_branch() {
        let origin = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let base = t.commit_all("base");
        t.git(&["remote", "add", "origin", &origin.url()]);
        let repo = t.repo();
        let o = RemoteName::parse("origin").unwrap();
        let work = BranchName::parse("zen/work").unwrap();

        repo.create_branch(&work, &base).unwrap();
        let pushed = repo
            .push(
                &o,
                &[PushSpec::branch(base.clone(), work.clone(), Lease::Absent)],
            )
            .unwrap();
        assert_eq!(pushed[0].outcome, PushOutcome::Created);
        repo.set_upstream(&work, &o, &work).unwrap();

        assert_eq!(
            t.git(&["config", "branch.zen/work.remote"]).trim(),
            "origin"
        );
        assert_eq!(
            t.git(&["config", "branch.zen/work.merge"]).trim(),
            "refs/heads/zen/work"
        );
        let tracked = |ahead, behind| Upstream {
            remote: Some(o.clone()),
            branch: work.clone(),
            ahead,
            behind,
            gone: false,
        };
        assert_eq!(upstream_of(&repo, &work), Some(tracked(0, 0)));

        // A local commit puts the branch one ahead.
        let local = advance(&repo, &base, b"local\n");
        repo.move_branch(&work, &base, &local).unwrap();
        assert_eq!(upstream_of(&repo, &work), Some(tracked(1, 0)));

        // Someone else pushes to the remote branch; after a fetch it is one
        // behind as well.
        let other = TestRepo::init();
        other.git(&["remote", "add", "origin", &origin.url()]);
        other.git(&["fetch", "-q", "origin"]);
        other.git(&["checkout", "-q", "-b", "theirs", "origin/zen/work"]);
        other.write("theirs.txt", b"theirs\n");
        other.commit_all("theirs");
        other.git(&["push", "-q", "origin", "theirs:zen/work"]);
        repo.fetch(&o, &FetchSpec::Branch(work.clone())).unwrap();
        assert_eq!(upstream_of(&repo, &work), Some(tracked(1, 1)));

        // Git itself agrees: the user's own tools see the same upstream.
        assert_eq!(
            t.git(&["rev-parse", "--abbrev-ref", "zen/work@{upstream}"])
                .trim(),
            "origin/zen/work"
        );
    }

    /// An upstream can be set before the remote branch exists. It reads as
    /// `gone` until a fetch brings the tracking ref in.
    #[test]
    fn an_upstream_set_before_the_remote_branch_exists_is_gone_until_fetched() {
        let origin = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let base = t.commit_all("base");
        t.git(&["remote", "add", "origin", &origin.url()]);
        let repo = t.repo();
        let o = RemoteName::parse("origin").unwrap();
        let work = BranchName::parse("feature").unwrap();
        repo.create_branch(&work, &base).unwrap();

        repo.set_upstream(&work, &o, &work).unwrap();
        assert!(upstream_of(&repo, &work).unwrap().gone);

        t.git(&["push", "-q", "origin", "feature"]);
        repo.fetch(&o, &FetchSpec::AllBranches).unwrap();
        let up = upstream_of(&repo, &work).unwrap();
        assert!(!up.gone);
        assert_eq!((up.ahead, up.behind), (0, 0));
    }

    /// A branch can track a differently named branch on another remote — the
    /// fork layout, where `upstream` is the shared repository.
    #[test]
    fn a_branch_can_track_another_remote_and_name() {
        let shared = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let base = t.commit_all("base");
        t.git(&["remote", "add", "upstream", &shared.url()]);
        t.git(&["push", "-q", "upstream", "main:trunk"]);
        let repo = t.repo();
        let up = RemoteName::parse("upstream").unwrap();
        repo.fetch(&up, &FetchSpec::AllBranches).unwrap();
        let local = BranchName::parse("mine").unwrap();
        repo.create_branch(&local, &base).unwrap();

        repo.set_upstream(&local, &up, &BranchName::parse("trunk").unwrap())
            .unwrap();
        let got = upstream_of(&repo, &local).unwrap();
        assert_eq!(got.remote, Some(up));
        assert_eq!(got.branch.as_str(), "trunk");
        assert!(!got.gone);
    }

    #[test]
    fn unsetting_removes_the_upstream_and_is_idempotent() {
        let origin = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        let base = t.commit_all("base");
        t.git(&["remote", "add", "origin", &origin.url()]);
        let repo = t.repo();
        let o = RemoteName::parse("origin").unwrap();
        let work = BranchName::parse("work").unwrap();
        repo.create_branch(&work, &base).unwrap();
        repo.set_upstream(&work, &o, &work).unwrap();
        assert!(upstream_of(&repo, &work).is_some());

        repo.unset_upstream(&work).unwrap();
        assert_eq!(upstream_of(&repo, &work), None);
        repo.unset_upstream(&work).unwrap();
        let config = t.git(&["config", "--local", "--list"]);
        assert!(
            !config.lines().any(|l| l.starts_with("branch.work.")),
            "{config}"
        );
    }

    /// Setting an upstream on the checked-out branch is allowed — it changes
    /// config, not the working tree — and a missing branch is refused.
    #[test]
    fn the_checked_out_branch_may_track_and_a_missing_branch_is_refused() {
        let origin = TestRepo::bare();
        let t = TestRepo::init();
        t.write("a.txt", b"a\n");
        t.commit_all("base");
        t.git(&["remote", "add", "origin", &origin.url()]);
        t.git(&["push", "-q", "origin", "main"]);
        let repo = t.repo();
        let o = RemoteName::parse("origin").unwrap();
        repo.fetch(&o, &FetchSpec::AllBranches).unwrap();
        let main = BranchName::parse("main").unwrap();
        repo.set_upstream(&main, &o, &main).unwrap();
        assert_eq!(upstream_of(&repo, &main).unwrap().branch, main);
        assert_eq!(repo.head().unwrap().branch(), Some(&main));

        let missing = BranchName::parse("nope").unwrap();
        assert!(matches!(
            repo.set_upstream(&missing, &o, &missing),
            Err(GitError::UnknownRevision { .. })
        ));
        assert!(matches!(
            repo.unset_upstream(&missing),
            Err(GitError::UnknownRevision { .. })
        ));
    }
}
