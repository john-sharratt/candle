//! One writable handle per substrate, enforced.
//!
//! [`super::log_file::LogFile`] opens segments read-write and takes no lock of
//! its own, and a second writable [`super::SubstratePersistence`] over the same
//! directory is a second append cursor and a second record index — its
//! compaction carries forward only what its own walk saw, and every record the
//! other handle appended after it opened is lost (see [`super::SharedSubstrate`]).
//! With three binaries able to serve the same workspace — `zend`, and `tokerad`
//! hosting zend — that second handle can be another process.
//!
//! So a writable open takes an exclusive lock on `substrate/LOCK` until the
//! handle drops or releases it ([`super::SubstratePersistence::release_writer`],
//! which an engine's shutdown ends with and which leaves the handle refusing
//! writes), and a second writable open — in this process or another — is
//! refused meanwhile. A read-only open takes no lock: reading a workspace a
//! daemon is writing is what read-only handles are for.

use std::fs::{self, File, OpenOptions, TryLockError};
use std::path::Path;

use super::{PersistenceError, Result};

/// The lock file's name inside `substrate/`.
pub const LOCK_FILE: &str = "LOCK";

/// The exclusive lock a writable handle holds on its substrate directory.
/// Released when dropped.
#[derive(Debug)]
pub struct WriterLock {
    _file: File,
}

impl WriterLock {
    /// Take the lock on `dir` (the `substrate/` directory), creating the
    /// directory and the lock file as needed. Refused with
    /// [`PersistenceError::Locked`] while another writable handle holds it.
    pub fn acquire(dir: &Path) -> Result<Self> {
        fs::create_dir_all(dir)?;
        let file = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .open(dir.join(LOCK_FILE))?;
        match file.try_lock() {
            Ok(()) => Ok(Self { _file: file }),
            Err(TryLockError::WouldBlock) => Err(PersistenceError::Locked {
                dir: dir.to_path_buf(),
            }),
            Err(TryLockError::Error(e)) => Err(PersistenceError::Io(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_lock_on_one_directory_is_refused_until_the_first_drops() {
        let dir = tempfile::tempdir().unwrap();
        let first = WriterLock::acquire(dir.path()).unwrap();
        assert!(matches!(
            WriterLock::acquire(dir.path()),
            Err(PersistenceError::Locked { .. })
        ));
        drop(first);
        assert!(WriterLock::acquire(dir.path()).is_ok());
    }

    #[test]
    fn two_directories_lock_independently() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let _a = WriterLock::acquire(a.path()).unwrap();
        assert!(WriterLock::acquire(b.path()).is_ok());
    }
}
