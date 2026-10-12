//! A fetch carried from a checkout that reaches the remote to one that does
//! not (`docs/replicated_substrate.md` §10.3, §11.4). An agent fetches its
//! origin and answers with what origin holds for the branches asked —
//! [`Repo::bundle_for_fetch`]: each branch's commit, and a bundle of every
//! object of them the asker does not already hold — and the asker takes it in
//! as its own remote-tracking refs ([`Repo::take_fetch`]), with the same refs
//! and the same report a fetch from origin itself gives.

use std::collections::BTreeMap;

use crate::error::GitError;
use crate::remote::fetch::{FetchSpec, RefUpdate};
use crate::types::{BranchName, Oid, RemoteName, Rev};
use crate::write::ref_txn::{RefOp, RefTransaction};
use crate::Repo;

/// The refs a fetch's bundle is cut from, which exist only while it is made.
const FETCH_REFS: &str = "refs/zend/fetch/";

/// What `git bundle create` says when every object is already held.
const EMPTY: &str = "Refusing to create empty bundle";

/// What a remote holds for the branches a fetch asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchBundle {
    /// Each branch's commit on the remote.
    pub tips: BTreeMap<BranchName, Oid>,
    /// The objects of those commits the asker lacks, as a git bundle; `None`
    /// when it lacks none.
    pub bundle: Option<Vec<u8>>,
}

impl Repo {
    /// The commits `remote`'s tracking refs hold: what a fetch through an
    /// agent names as already held.
    pub fn tracking_tips(&self, remote: &RemoteName) -> Result<Vec<Oid>, GitError> {
        Ok(self
            .snapshot(&format!("refs/remotes/{remote}/"))?
            .into_values()
            .collect())
    }

    /// What `remote`'s tracking refs hold for `spec` — fetched first by the
    /// caller — and a bundle of every object they reach that `have` does not.
    /// A commit in `have` this repository does not hold is passed over.
    pub fn bundle_for_fetch(
        &self,
        remote: &RemoteName,
        spec: &FetchSpec,
        have: &[Oid],
    ) -> Result<FetchBundle, GitError> {
        let prefix = format!("refs/remotes/{remote}/");
        let tips: BTreeMap<BranchName, Oid> = self
            .snapshot(&spec.scope(remote))?
            .into_iter()
            .filter_map(|(name, oid)| {
                let branch = name.as_str().strip_prefix(&prefix)?;
                Some((BranchName::parse(branch).ok()?, oid))
            })
            .collect();
        if tips.is_empty() {
            return Ok(FetchBundle { tips, bundle: None });
        }
        let held: Vec<String> = have
            .iter()
            .filter(|oid| self.resolve(&Rev::Oid((*oid).clone())).is_ok())
            .map(|oid| format!("^{oid}"))
            .collect();
        let _write = self.write_lock();
        let refs: Vec<String> = (0..tips.len())
            .map(|i| format!("{FETCH_REFS}{i}"))
            .collect();
        let create: String = refs
            .iter()
            .zip(tips.values())
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
            .args(&held)
            .run_accepting(&[0, 128]);
        let delete: String = refs.iter().map(|r| format!("delete {r}\n")).collect();
        self.git("update-ref")
            .arg("--stdin")
            .stdin(delete.into_bytes())
            .run_ok()?;
        let made = made?;
        let bundle = match made.status {
            Some(0) => Some(made.stdout),
            _ if made.stderr.contains(EMPTY) => None,
            status => {
                return Err(GitError::malformed(
                    "bundle",
                    format!("exit {status:?}: {}", made.stderr.trim()),
                ))
            }
        };
        Ok(FetchBundle { tips, bundle })
    }

    /// Take `fetched` — what `remote` held for `spec` — into this repository:
    /// its objects, then each branch's tracking ref set to its commit, and for
    /// every branch the tracking refs of branches the remote no longer has
    /// removed. Reports what changed, as [`Repo::fetch_here`] does. A branch
    /// asked for alone that the remote does not have is an unknown revision.
    pub fn take_fetch(
        &self,
        remote: &RemoteName,
        spec: &FetchSpec,
        fetched: &FetchBundle,
    ) -> Result<Vec<RefUpdate>, GitError> {
        self.require_remote(remote)?;
        if let FetchSpec::Branch(branch) = spec {
            if !fetched.tips.contains_key(branch) {
                return Err(GitError::UnknownRevision {
                    rev: branch.to_ref().to_string(),
                });
            }
        }
        let _write = self.write_lock();
        let scope = spec.scope(remote);
        let before = self.snapshot(&scope)?;
        if let Some(bundle) = &fetched.bundle {
            self.take_bundle_locked(bundle)?;
        }
        let wanted: BTreeMap<_, _> = fetched
            .tips
            .iter()
            .filter(|(branch, _)| match spec {
                FetchSpec::Branch(b) => b == *branch,
                FetchSpec::AllBranches => true,
            })
            .map(|(branch, oid)| (remote.tracking(branch), oid.clone()))
            .collect();
        let mut txn = RefTransaction::new();
        for (name, new) in &wanted {
            txn = match before.get(name) {
                None => txn.push(RefOp::Create {
                    name: name.clone(),
                    new: new.clone(),
                }),
                Some(old) if old == new => txn,
                Some(old) => txn.push(RefOp::Update {
                    name: name.clone(),
                    new: new.clone(),
                    old: old.clone(),
                }),
            };
        }
        if *spec == FetchSpec::AllBranches {
            for (name, old) in &before {
                if !wanted.contains_key(name) {
                    txn = txn.push(RefOp::Delete {
                        name: name.clone(),
                        old: old.clone(),
                    });
                }
            }
        }
        if !txn.ops().is_empty() {
            self.write_refs(&txn)?;
        }
        let after = self.snapshot(&scope)?;
        self.changed(&before, &after)
    }
}

#[cfg(test)]
mod tests {
    use std::slice::from_ref;

    use super::*;
    use crate::remote::fetch::FetchFlag;
    use crate::testing::TestRepo;
    use crate::types::RemoteUrl;

    fn origin() -> RemoteName {
        RemoteName::parse("origin").unwrap()
    }

    fn main() -> BranchName {
        BranchName::parse("main").unwrap()
    }

    /// An origin with `main` at one commit; an agent's checkout of it; and a
    /// member whose `origin` names the agent, not a URL it can reach.
    fn three() -> (TestRepo, TestRepo, TestRepo, TestRepo, Oid) {
        let bare = TestRepo::bare();
        let upstream = TestRepo::init();
        upstream.write("projection.yaml", b"layers: []\n");
        let first = upstream.commit_all("first");
        upstream.git(&["remote", "add", "origin", &bare.url()]);
        upstream.git(&["push", "-q", "origin", "main"]);
        let agent = TestRepo::init();
        agent.git(&["remote", "add", "origin", &bare.url()]);
        let member = TestRepo::init();
        member
            .repo()
            .add_remote(&origin(), &RemoteUrl::parse("agent:mind").unwrap())
            .unwrap();
        (bare, upstream, agent, member, first)
    }

    /// The agent fetches origin and bundles a branch; the member takes it in.
    fn carry(agent: &TestRepo, member: &TestRepo, spec: &FetchSpec) -> Vec<RefUpdate> {
        agent
            .repo()
            .fetch_here(&origin(), &FetchSpec::AllBranches)
            .unwrap();
        let have = member.repo().tracking_tips(&origin()).unwrap();
        let fetched = agent
            .repo()
            .bundle_for_fetch(&origin(), spec, &have)
            .unwrap();
        member.repo().take_fetch(&origin(), spec, &fetched).unwrap()
    }

    /// **A member with no way to origin gets origin's branch through the
    /// agent**: the commit and its objects, under its own tracking ref, with
    /// the report a fetch from origin gives; then only what it lacks.
    #[test]
    fn a_branch_is_carried_to_a_member_that_cannot_reach_origin() {
        let (_bare, upstream, agent, member, first) = three();
        let updates = carry(&agent, &member, &FetchSpec::Branch(main()));
        assert_eq!(updates.len(), 1);
        assert_eq!(
            (updates[0].flag, updates[0].new.clone()),
            (FetchFlag::New, Some(first.clone()))
        );
        assert_eq!(member.oid("refs/remotes/origin/main"), first);
        assert_eq!(
            member.git(&["show", &format!("{first}:projection.yaml")]),
            "layers: []\n"
        );

        upstream.write("projection.yaml", b"layers: [eras]\n");
        let second = upstream.commit_all("second");
        upstream.git(&["push", "-q", "origin", "main"]);
        agent
            .repo()
            .fetch_here(&origin(), &FetchSpec::AllBranches)
            .unwrap();
        let fetched = agent
            .repo()
            .bundle_for_fetch(&origin(), &FetchSpec::Branch(main()), from_ref(&first))
            .unwrap();
        let updates = member
            .repo()
            .take_fetch(&origin(), &FetchSpec::Branch(main()), &fetched)
            .unwrap();
        assert_eq!(
            (
                updates[0].flag,
                updates[0].old.clone(),
                updates[0].new.clone()
            ),
            (FetchFlag::FastForward, Some(first), Some(second))
        );

        // Nothing new: no bundle, nothing changed.
        let again = agent
            .repo()
            .bundle_for_fetch(
                &origin(),
                &FetchSpec::Branch(main()),
                &[member.oid("refs/remotes/origin/main")],
            )
            .unwrap();
        assert_eq!(again.bundle, None);
        assert!(member
            .repo()
            .take_fetch(&origin(), &FetchSpec::Branch(main()), &again)
            .unwrap()
            .is_empty());
    }

    /// Every branch: a new one appears, a deleted one is pruned; a branch
    /// asked for alone that origin lacks is an unknown revision; and a commit
    /// the agent does not hold is passed over as `have`.
    #[test]
    fn every_branch_is_carried_and_pruned() {
        let (_bare, upstream, agent, member, _first) = three();
        upstream.git(&["push", "-q", "origin", "main:topic"]);
        carry(&agent, &member, &FetchSpec::AllBranches);
        assert!(member
            .repo()
            .snapshot("refs/remotes/origin/")
            .unwrap()
            .contains_key(&origin().tracking(&BranchName::parse("topic").unwrap())));
        upstream.git(&["push", "-q", "origin", ":topic"]);
        let updates = carry(&agent, &member, &FetchSpec::AllBranches);
        assert_eq!(updates.len(), 1);
        assert_eq!(updates[0].flag, FetchFlag::Pruned);

        let absent = BranchName::parse("absent").unwrap();
        let fetched = agent
            .repo()
            .bundle_for_fetch(&origin(), &FetchSpec::Branch(absent.clone()), &[])
            .unwrap();
        assert!(fetched.tips.is_empty());
        assert!(matches!(
            member
                .repo()
                .take_fetch(&origin(), &FetchSpec::Branch(absent), &fetched),
            Err(GitError::UnknownRevision { .. })
        ));

        let stranger = Oid::parse("1234567890123456789012345678901234567890").unwrap();
        let fresh = TestRepo::init();
        fresh
            .repo()
            .add_remote(&origin(), &RemoteUrl::parse("agent:mind").unwrap())
            .unwrap();
        let fetched = agent
            .repo()
            .bundle_for_fetch(&origin(), &FetchSpec::Branch(main()), &[stranger])
            .unwrap();
        assert!(fetched.bundle.is_some());
        fresh
            .repo()
            .take_fetch(&origin(), &FetchSpec::Branch(main()), &fetched)
            .unwrap();
        assert_eq!(
            fresh.oid("refs/remotes/origin/main"),
            agent.oid("refs/remotes/origin/main")
        );
    }
}
