//! Opening a substrate read-only beside the daemon that writes it.
//!
//! The writer compacts as it runs: a segment is rewritten into a new one and
//! the old file deleted. A reader that listed the segments before the delete
//! and opens them after finds one gone — `NotFound` — and the open fails
//! halfway, its substrate half-built. Started over, the listing names the
//! segment that replaced it. A compaction finishes in moments, so a few tries
//! always find a stable listing; a directory that is genuinely missing still
//! fails, on the last try.

use std::io::ErrorKind;

use super::PersistenceError;

/// Opens a read-only handle tries before a vanished segment is an error.
pub(super) const ATTEMPTS: usize = 5;

/// Run `open` until it does not fail on a vanished file, at most
/// [`ATTEMPTS`] times, calling `reset` before each retry so it starts from
/// nothing. Any other outcome is returned as it is.
pub(super) fn retry_vanished<T>(
    mut open: impl FnMut() -> Result<T, PersistenceError>,
    mut reset: impl FnMut(),
) -> Result<T, PersistenceError> {
    let mut tried = 1;
    loop {
        match open() {
            Err(PersistenceError::Io(e)) if e.kind() == ErrorKind::NotFound && tried < ATTEMPTS => {
                tried += 1;
                reset();
            }
            out => return out,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io;

    use super::*;

    fn vanished() -> PersistenceError {
        PersistenceError::Io(io::Error::from(ErrorKind::NotFound))
    }

    /// **A segment gone mid-open is opened again, from nothing.** Two
    /// vanished segments, then a stable listing: three opens, two resets.
    #[test]
    fn a_vanished_segment_is_opened_again_from_nothing() {
        let (mut opens, mut resets) = (0, 0);
        let out = retry_vanished(
            || {
                opens += 1;
                if opens < 3 {
                    Err(vanished())
                } else {
                    Ok(opens)
                }
            },
            || resets += 1,
        );
        assert_eq!(out.unwrap(), 3);
        assert_eq!(resets, 2);
    }

    /// Any other failure is returned at once; a file that stays missing fails
    /// on the last try.
    #[test]
    fn other_failures_and_a_missing_directory_still_fail() {
        let mut opens = 0;
        let out: Result<(), _> = retry_vanished(
            || {
                opens += 1;
                Err(PersistenceError::Corrupt("bad".into()))
            },
            || {},
        );
        assert!(matches!(out, Err(PersistenceError::Corrupt(_))));
        assert_eq!(opens, 1);

        let mut opens = 0;
        let out: Result<(), _> = retry_vanished(
            || {
                opens += 1;
                Err(vanished())
            },
            || {},
        );
        assert!(matches!(out, Err(PersistenceError::Io(_))));
        assert_eq!(opens, ATTEMPTS);
    }
}
