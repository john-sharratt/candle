//! A push's objects carried to another checkout of the same repository
//! (`docs/replicated_substrate.md` §11.4). A member builds commits in its own
//! repository and holds no push credentials; an agent holds them. The member
//! bundles what the push sets that origin does not already have
//! ([`Repo::bundle_for_push`]), the agent takes the bundle into its own
//! repository ([`Repo::take_bundle`]) and pushes the same specs from there.

use std::fs;

use crate::error::GitError;
use crate::remote::push::{PushAction, PushSpec};
use crate::types::RemoteName;
use crate::Repo;

/// The refs a bundle is cut from, which exist only while it is made: a bundle
/// names its tips by ref.
const PUSH_REFS: &str = "refs/zend/push/";

/// What `git bundle create` says when every object is already on the remote.
const EMPTY: &str = "Refusing to create empty bundle";

impl Repo {
    /// A bundle of every object `specs` set on `remote` that `remote`'s
    /// tracking refs do not already reach; `None` when there is none — a
    /// delete, or commits origin already holds.
    pub fn bundle_for_push(
        &self,
        remote: &RemoteName,
        specs: &[PushSpec],
    ) -> Result<Option<Vec<u8>>, GitError> {
        let tips: Vec<String> = specs
            .iter()
            .filter_map(|s| match &s.action {
                PushAction::Update(oid) => Some(oid.to_string()),
                PushAction::Delete => None,
            })
            .collect();
        if tips.is_empty() {
            return Ok(None);
        }
        self.require_remote(remote)?;
        let _write = self.write_lock();
        let refs: Vec<String> = (0..tips.len()).map(|i| format!("{PUSH_REFS}{i}")).collect();
        let create: String = refs
            .iter()
            .zip(&tips)
            .map(|(r, oid)| format!("update {r} {oid}\n"))
            .collect();
        self.git("update-ref")
            .arg("--stdin")
            .stdin(create.into_bytes())
            .run_ok()?;
        let made = self
            .git("bundle")
            .args(["create", "--quiet", "-"])
            .args(&refs)
            .arg("--not")
            .arg(format!("--remotes={}", remote.as_str()))
            .run_accepting(&[0, 128]);
        let delete: String = refs.iter().map(|r| format!("delete {r}\n")).collect();
        self.git("update-ref")
            .arg("--stdin")
            .stdin(delete.into_bytes())
            .run_ok()?;
        let made = made?;
        match made.status {
            Some(0) => Ok(Some(made.stdout)),
            _ if made.stderr.contains(EMPTY) => Ok(None),
            status => Err(GitError::malformed(
                "bundle",
                format!("exit {status:?}: {}", made.stderr.trim()),
            )),
        }
    }

    /// Take the objects of `bundle` into this repository, for a push of them.
    /// Its prerequisites — what origin already held — must be here, which a
    /// fetch from origin first makes sure of.
    pub fn take_bundle(&self, bundle: &[u8]) -> Result<(), GitError> {
        let _write = self.write_lock();
        let path = self.git_dir().join("zend-push.bundle");
        fs::write(&path, bundle).map_err(GitError::Io)?;
        let taken = self
            .git("bundle")
            .arg("unbundle")
            .arg(path.as_os_str())
            .run_ok();
        let _ = fs::remove_file(&path);
        taken.map(|_| ())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::remote::push::{Lease, PushOutcome};
    use crate::testing::TestRepo;
    use crate::types::BranchName;

    fn origin() -> RemoteName {
        RemoteName::parse("origin").unwrap()
    }

    /// **A commit made on one checkout is pushed from another** through a
    /// bundle of only what origin lacks, and lands on origin under its lease.
    #[test]
    fn a_push_is_carried_to_another_checkout_as_a_bundle() {
        let bare = TestRepo::bare();
        let member = TestRepo::init();
        member.write("a", b"1\n");
        let first = member.commit_all("first");
        member.git(&["remote", "add", "origin", &bare.url()]);
        member.git(&["push", "-q", "origin", &format!("{first}:refs/heads/main")]);
        member.git(&["fetch", "-q", "origin"]);
        member.write("a", b"2\n");
        let second = member.commit_all("second");

        let agent = TestRepo::init();
        agent.git(&["remote", "add", "origin", &bare.url()]);
        agent.git(&["fetch", "-q", "origin"]);

        let main = BranchName::parse("main").unwrap();
        let specs = [PushSpec::branch(
            second.clone(),
            main.clone(),
            Lease::Expect(first.clone()),
        )];
        let bundle = member
            .repo()
            .bundle_for_push(&origin(), &specs)
            .unwrap()
            .expect("origin lacks the second commit");
        agent.repo().take_bundle(&bundle).unwrap();
        let pushed = agent.repo().push_here(&origin(), &specs).unwrap();
        assert_eq!(pushed[0].outcome, PushOutcome::FastForward);
        let on_origin = bare.git(&["rev-parse", "refs/heads/main"]);
        assert_eq!(on_origin.trim(), second.to_string());
        // The bundle's refs were the member's for the making only.
        assert_eq!(member.git(&["for-each-ref", PUSH_REFS]).trim(), "");
    }

    /// Nothing to carry: a delete, and a commit origin already holds.
    #[test]
    fn a_push_origin_already_holds_needs_no_bundle() {
        let bare = TestRepo::bare();
        let member = TestRepo::init();
        member.write("a", b"1\n");
        let first = member.commit_all("first");
        member.git(&["remote", "add", "origin", &bare.url()]);
        member.git(&["push", "-q", "origin", &format!("{first}:refs/heads/main")]);
        member.git(&["fetch", "-q", "origin"]);
        let repo = member.repo();
        let main = BranchName::parse("main").unwrap();
        let other = BranchName::parse("other").unwrap();
        assert_eq!(
            repo.bundle_for_push(
                &origin(),
                &[PushSpec::branch(first.clone(), other, Lease::Absent)]
            )
            .unwrap(),
            None
        );
        assert_eq!(
            repo.bundle_for_push(&origin(), &[PushSpec::delete_branch(main, first)])
                .unwrap(),
            None
        );
    }
}
