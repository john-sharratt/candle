//! The mind folder brought up to the record.
//!
//! The mind folder is the published mind: what a body with no operation reads,
//! what the corpus is drawn from, and what the watcher ingests into memory. A
//! publish moves the record — `origin`'s branch, and the local branch after it
//! — but not the files checked out, and a publish from another member moves
//! only `origin`. [`follow`] fetches the record and checks it out, so the
//! files on disk are what the record holds and the watcher takes in the
//! change (§10.3).
//!
//! **Only a clean checkout is moved.** The index holds what was last checked
//! out, whatever the branch has moved to since, so a file on disk that differs
//! from the index is somebody's work outside any operation: it is left as it
//! is, and said so, rather than overwritten.

use std::path::Path;
use std::sync::Arc;

use zend_vfs::{Fetch, GitError, Oid, Repo, StatusCode, StatusEntry};

/// What following the record did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Followed {
    /// The mind folder already holds the record.
    Level,
    /// The mind folder was checked out at `to`.
    Moved { to: Oid },
    /// The mind folder has changes of its own on disk, and was left.
    Dirty(Vec<String>),
    /// The local branch holds commits the record does not, and was left.
    Diverged,
}

/// Fetch the record of the branch checked out at `dir` through `fetcher` and
/// check it out, when the checkout is clean. `None` when `dir` is on no
/// branch.
pub fn follow(dir: &Path, fetcher: Arc<dyn Fetch>) -> Result<Option<Followed>, GitError> {
    let repo = Repo::open(dir)?.fetching_through(fetcher);
    let Some(branch) = repo.head()?.branch().cloned() else {
        return Ok(None);
    };
    let status = repo.status()?;
    let dirty: Vec<String> = status
        .iter()
        .filter_map(|e| match e {
            StatusEntry::Changed { xy, path, .. } | StatusEntry::Renamed { xy, path, .. }
                if xy.worktree != StatusCode::Unmodified =>
            {
                Some(path.as_str().to_string())
            }
            StatusEntry::Unmerged { path, .. } => Some(path.as_str().to_string()),
            _ => None,
        })
        .collect();
    if !dirty.is_empty() {
        return Ok(Some(Followed::Dirty(dirty)));
    }
    let pulled = repo.pull_branch(&branch)?;
    if pulled.diverged {
        return Ok(Some(Followed::Diverged));
    }
    let Some(tip) = pulled.tip else {
        return Ok(Some(Followed::Level));
    };
    // The index behind the branch — a publish moved it, or the pull did —
    // shows as changes staged against it; a checkout at one with none is
    // already at the record.
    let staged = repo.status()?.iter().any(|e| match e {
        StatusEntry::Changed { xy, .. } | StatusEntry::Renamed { xy, .. } => {
            xy.index != StatusCode::Unmodified
        }
        _ => false,
    });
    if !staged {
        return Ok(Some(Followed::Level));
    }
    repo.force_checkout_branch(&branch)?;
    Ok(Some(Followed::Moved { to: tip }))
}
