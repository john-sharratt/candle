//! Branches whose record is origin: writes go there, reads stay local.
//!
//! The repositories' folders are the sandbox's working caches, and a
//! repository's local refs are a cache of origin's: a branch, a commit on it,
//! a tag is only kept once origin has it. So a write that changes a branch is
//! made in three steps:
//!
//! 1. **Pull** the branch ([`Repo::pull_branch`]): fetch what origin holds and
//!    fast-forward the local branch to it. A local branch that has diverged
//!    from origin's is left exactly as it is, and reported: no commit is ever
//!    rewritten or taken off a branch here.
//! 2. **Build** the change.
//! 3. **Publish** it: push to origin under a lease on what the pull found
//!    there, and only once origin has accepted, move the local branch to
//!    match. A commit is published with [`Repo::publish_commit`], which also
//!    refuses one that does not descend from origin's copy — publishing it
//!    would take someone else's commits off the branch. A push refused
//!    because origin moved in the meantime changes nothing, anywhere; the
//!    commits meet in a merge ([`crate::work`]). [`Repo::publish_branch`]
//!    moves a branch anywhere on purpose — a new branch, a rewind.
//!
//! Reads never touch the network: they read the local refs, which the last
//! write's pull left as current as origin was then.
//!
//! A repository with no `origin` remote keeps its record locally: the pull
//! reads the local branch, and a publish moves it.
//!
//! The local branch is moved whether or not a working tree has it checked
//! out: the only working trees here are the sandbox's own, and each job puts
//! its checkout on its branch as it finds it.

mod follow;
mod publish;
mod pull;

pub use follow::Followed;
pub use publish::Published;
pub use pull::Pulled;

use crate::error::GitError;
use crate::types::{BranchName, Oid, RemoteName};
use crate::{RefOp, RefTransaction, Repo};

/// The remote a repository's record is kept on.
pub const ORIGIN: &str = "origin";

impl Repo {
    /// This repository's [`ORIGIN`], when it has one configured.
    pub fn origin(&self) -> Result<Option<RemoteName>, GitError> {
        let name = RemoteName::parse(ORIGIN)?;
        Ok(self
            .remotes()?
            .iter()
            .any(|r| r.name == name)
            .then_some(name))
    }

    /// Remove `branch`, as a compare-and-swap against `old`, whether or not
    /// a working tree has it checked out — the local half of deleting a
    /// branch origin has already let go of.
    pub(crate) fn force_delete_branch(
        &self,
        branch: &BranchName,
        old: &Oid,
    ) -> Result<(), GitError> {
        let _write = self.write_lock();
        self.write_refs(&RefTransaction::new().push(RefOp::Delete {
            name: branch.to_ref(),
            old: old.clone(),
        }))
    }
}
