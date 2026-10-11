//! A write that failed for lack of space (`docs/replicated_substrate.md` §7.7):
//! the error that sends a member into fault mode, latched where the log is
//! written, and the test append and commit that must succeed before it leaves.

use std::io::{self, ErrorKind};

use super::{PersistenceError, Result, SubstratePersistence};

/// The raw codes for a full volume: `ERROR_HANDLE_DISK_FULL` (39) and
/// `ERROR_DISK_FULL` (112) on Windows, `ENOSPC` (28) elsewhere.
#[cfg(windows)]
const FULL_CODES: [i32; 2] = [39, 112];
#[cfg(not(windows))]
const FULL_CODES: [i32; 1] = [28];

/// Whether `e` says the volume is full.
pub fn is_out_of_space(e: &io::Error) -> bool {
    e.kind() == ErrorKind::StorageFull || e.raw_os_error().is_some_and(|c| FULL_CODES.contains(&c))
}

impl SubstratePersistence {
    /// Pass `result` through, latching an out-of-space failure.
    pub(super) fn watch_space<T>(&mut self, result: Result<T>) -> Result<T> {
        if let Err(PersistenceError::Io(e)) = &result {
            if is_out_of_space(e) {
                self.out_of_space = true;
            }
        }
        result
    }

    /// Whether a write has failed for lack of space since the last
    /// [`Self::test_write`] succeeded.
    pub fn out_of_space(&self) -> bool {
        self.out_of_space
    }

    /// A test append and commit: the replica rewritten, made durable. Clears
    /// the latch when it succeeds.
    pub fn test_write(&mut self) -> Result<()> {
        self.replica_dirty = true;
        self.commit()?;
        self.out_of_space = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_volume_is_recognised_by_kind_and_by_code() {
        assert!(is_out_of_space(&io::Error::from(ErrorKind::StorageFull)));
        for code in FULL_CODES {
            assert!(is_out_of_space(&io::Error::from_raw_os_error(code)));
        }
        assert!(!is_out_of_space(&io::Error::from(ErrorKind::NotFound)));
    }

    /// A code meaning something else on this platform is not a full volume:
    /// 39 is `ENOTEMPTY` on Linux, 28 `ERROR_OUT_OF_PAPER` on Windows.
    #[test]
    fn another_platforms_code_is_not_a_full_volume() {
        #[cfg(windows)]
        assert!(!is_out_of_space(&io::Error::from_raw_os_error(28)));
        #[cfg(not(windows))]
        assert!(!is_out_of_space(&io::Error::from_raw_os_error(39)));
    }
}
