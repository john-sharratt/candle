//! One preservation of a checkout at a time, across every process.
//!
//! A sandbox keeps its own jobs off a checkout one at a time, but that lock
//! lives in one process. Two daemons — or two sandboxes — over the same
//! repository would each find the other's live preservation on disk and
//! take it for one a crash left, and put the checkout back under the other's
//! running job. The file lock held here, `zend-preserved/lock` in the git
//! folder, is what makes "left on disk" mean "left by a run that is gone":
//! whoever holds it is the only preservation there is, and the operating
//! system lets go of it when its holder exits, however it exits.

use std::fs::{File, OpenOptions, TryLockError};
use std::path::Path;

use super::journal::PRESERVED_DIR;
use crate::checkout::CheckoutError;
use crate::GitError;

const LOCK: &str = "lock";

/// The checkout's preservation lock, held until dropped.
#[derive(Debug)]
pub(super) struct CheckoutLock {
    _file: File,
}

impl CheckoutLock {
    /// Take the lock for the checkout whose git folder is `git_dir`, or say
    /// that another process holds it. Never waits: a holder that is stuck
    /// would otherwise hold every later run with it.
    pub(super) fn take(git_dir: &Path) -> Result<Self, CheckoutError> {
        let root = git_dir.join(PRESERVED_DIR);
        let fail = |e| CheckoutError::io(&root.to_string_lossy(), e);
        std::fs::create_dir_all(&root).map_err(fail)?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(root.join(LOCK))
            .map_err(fail)?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(TryLockError::WouldBlock) => Err(GitError::invalid(
                "another process is running a job in this checkout; nothing was touched",
            )
            .into()),
            Err(TryLockError::Error(e)) => Err(fail(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **Only one holder at a time**, and the lock is free again once it is
    /// dropped.
    #[test]
    fn one_holder_at_a_time() {
        let dir = tempfile::tempdir().unwrap();
        let first = CheckoutLock::take(dir.path()).unwrap();
        let second = CheckoutLock::take(dir.path()).unwrap_err();
        assert!(second.to_string().contains("another process"), "{second}");
        drop(first);
        CheckoutLock::take(dir.path()).unwrap();
    }
}
