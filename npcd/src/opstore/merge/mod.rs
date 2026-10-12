//! The merge table's work on an operation's store (`docs/replicated_substrate.md`
//! §10.3): what the store changes against its base and whether it merges with
//! the record ([`status`]), landing it on the record ([`publish`]), and bringing
//! the record's newer commit into it with each overlap marked ([`sync`]).
//!
//! **The record is the mind repository's branch as `origin` holds it** — or,
//! with no `origin`, as the mind folder's own branch holds it. A store lands
//! there the way a conversation's work does in zend: one commit on the store's
//! base, published under a lease on what the record held when the commit
//! began, so a record that moves in between refuses it and nothing changes.
//! Its push goes through the [`Push`] it is opened with — this checkout, or an
//! agent holding the repository with push rights (§11.5) — and its reads of
//! the record through the [`Fetch`]: an agent when the mind is a mirror of the
//! repository the agents hold (§10.3).
//!
//! | Module | |
//! |---|---|
//! | [`status`] | what the store changes, and whether it merges cleanly |
//! | [`publish`] | the store landed on the record, or why not |
//! | [`sync`] | the record's newer commit brought into the store |
//! | [`follow`] | the mind folder brought up to the record |

pub mod follow;
pub mod publish;
pub mod status;
pub mod sync;

use std::path::Path;
use std::sync::Arc;

use zend_vfs::work::held;
use zend_vfs::{
    merge_into, BranchName, Fetch, GitError, Merged, Oid, Push, Repo, Rev, Snapshot, VfsStore,
};

/// What the merge marks the record's side of an overlap with.
pub const RECORD: &str = "the record";

/// An operation's store and the mind repository it lands in.
pub struct Merge<'a> {
    pub repo: Repo,
    pub store: &'a VfsStore,
    pub branch: BranchName,
}

impl<'a> Merge<'a> {
    /// The store over the repository at `dir`, its pushes made by `pusher`
    /// and its fetches of the record by `fetcher`. Refused when the store is
    /// not read through a branch: a mind folder that is not a git repository
    /// has no record to land anything on.
    pub fn open(
        dir: &Path,
        store: &'a VfsStore,
        pusher: Arc<dyn Push>,
        fetcher: Arc<dyn Fetch>,
    ) -> Result<Merge<'a>, String> {
        let Some(Rev::Branch(branch)) = store.rev() else {
            return Err(NOT_UNDER_GIT.to_string());
        };
        let repo = Repo::open(dir)
            .map_err(|e| format!("the mind repository cannot be opened: {e}"))?
            .pushing_through(pusher)
            .fetching_through(fetcher);
        Ok(Merge {
            repo,
            store,
            branch,
        })
    }

    /// The record's commit, as `origin` holds it now — fetched — or as the
    /// local branch holds it with no `origin`. `None` for a branch the record
    /// does not have yet.
    pub fn record(&self) -> Result<Option<Oid>, GitError> {
        Ok(self.repo.pull_branch(&self.branch)?.record().cloned())
    }

    /// The commit the store's work is made on.
    pub fn base(&self) -> Option<Oid> {
        self.store
            .base()
            .ok()
            .flatten()
            .and_then(|b| b.commit().cloned())
    }

    /// The paths merging `theirs` into the store would leave in conflict —
    /// found by merging, then putting the store back exactly as it was.
    pub fn trial(&self, theirs: &Oid) -> Result<Vec<String>, String> {
        let before = self.store.snapshot();
        let merged =
            merge_into(&self.repo, self.store, theirs, RECORD).map_err(|e| e.to_string())?;
        self.put_back(before, &merged)?;
        Ok(merged.conflicts().to_vec())
    }

    /// Undo `merged`: the store as `before` held it, and the hold the merge
    /// took on its settled tree let go.
    pub fn put_back(&self, before: Snapshot, merged: &Merged) -> Result<(), String> {
        if let (Merged::Merging { .. }, Ok(Some(base))) = (merged, self.store.base()) {
            held::release(&self.repo, &base);
        }
        self.store
            .restore(before)
            .map_err(|e| format!("the store could not be put back after a merge: {e}"))
    }
}

/// Why a store not read through git cannot be published.
pub const NOT_UNDER_GIT: &str = "The mind folder is not a git repository, so there is no record \
                                 to land the work on.";

/// Whether `e` means git could not be reached — `origin`, or the agent that
/// pushes to it — rather than that it refused: a publish that meets it waits
/// and is tried again (§10.3).
pub fn unreachable(e: &GitError) -> bool {
    matches!(
        e,
        GitError::RemoteUnreachable { .. } | GitError::Timeout { .. } | GitError::Io(_)
    )
}

#[cfg(test)]
mod tests;
