//! What a file's metadata says about it — enough to tell, without reading it,
//! that it is not the file it was.
//!
//! A [`FileStamp`] is taken when the checkout writes or reads a file, and
//! compared with a fresh one later: a different size or modification time means
//! the file changed. Equal stamps mean it did not, provided the stamp is not
//! *racy* ([`FileStamp::is_racy`]) — taken so soon after a write that a second
//! write in the same timestamp tick would have left it unchanged. A racy stamp
//! is settled by comparing contents instead, as git settles its index.
//!
//! Both fields come from the standard library, the same on every platform. A
//! write that keeps a file's size and then sets its modification time back to
//! what it was is the one change a stamp cannot see.

use std::io;
use std::path::Path;
use std::time::UNIX_EPOCH;

use serde::{Deserialize, Serialize};

/// A file's size and modification time, nanoseconds since the Unix epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FileStamp {
    pub size: u64,
    pub mtime_ns: i64,
}

/// How far behind the moment of stamping a file's modification time must be
/// for its stamp to be trusted. Two seconds covers the coarsest filesystem in
/// use (FAT's modification time), and costs only a content comparison for files
/// written that recently.
pub const RACY_WINDOW_NS: i64 = 2_000_000_000;

impl FileStamp {
    /// `path`'s stamp, or `None` when nothing is there. A symbolic link is
    /// stamped as itself, never followed.
    pub fn of(path: &Path) -> io::Result<Option<FileStamp>> {
        let meta = match std::fs::symlink_metadata(path) {
            Ok(meta) => meta,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e),
        };
        let mtime_ns = match meta.modified()?.duration_since(UNIX_EPOCH) {
            Ok(after) => i64::try_from(after.as_nanos()).unwrap_or(i64::MAX),
            Err(before) => -i64::try_from(before.duration().as_nanos()).unwrap_or(i64::MAX),
        };
        Ok(Some(FileStamp {
            size: meta.len(),
            mtime_ns,
        }))
    }

    /// Whether this stamp cannot vouch for the file: its modification time
    /// falls within [`RACY_WINDOW_NS`] of `stamped_at_ns`, the moment it was
    /// taken, so a write in the same tick could have left it as it is.
    pub fn is_racy(&self, stamped_at_ns: i64) -> bool {
        self.mtime_ns >= stamped_at_ns.saturating_sub(RACY_WINDOW_NS)
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, SystemTime};

    use super::*;

    fn set_mtime(path: &Path, to: SystemTime) {
        std::fs::File::options()
            .write(true)
            .open(path)
            .unwrap()
            .set_modified(to)
            .unwrap();
    }

    #[test]
    fn a_missing_path_has_no_stamp() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(FileStamp::of(&dir.path().join("absent")).unwrap(), None);
    }

    /// An untouched file stamps the same twice — a read is not a change — and
    /// the stamp is its real size and modification time.
    #[test]
    fn an_untouched_file_stamps_the_same() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("a.txt");
        std::fs::write(&f, b"hello\n").unwrap();
        let at = SystemTime::UNIX_EPOCH + Duration::from_nanos(1_700_000_000_123_456_700);
        set_mtime(&f, at);
        let first = FileStamp::of(&f).unwrap().unwrap();
        let _ = std::fs::read(&f).unwrap();
        assert_eq!(FileStamp::of(&f).unwrap().unwrap(), first);
        assert_eq!(
            first,
            FileStamp {
                size: 6,
                mtime_ns: 1_700_000_000_123_456_700
            }
        );
    }

    /// A write moves the stamp even when the size stays the same.
    #[test]
    fn a_same_size_write_changes_the_stamp() {
        let dir = tempfile::tempdir().unwrap();
        let f = dir.path().join("a.txt");
        std::fs::write(&f, b"aaaa").unwrap();
        set_mtime(&f, SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000));
        let before = FileStamp::of(&f).unwrap().unwrap();
        std::fs::write(&f, b"bbbb").unwrap();
        let after = FileStamp::of(&f).unwrap().unwrap();
        assert_eq!(after.size, before.size);
        assert_ne!(after, before);
    }

    /// A stamp whose time is within the window of the moment it was taken is
    /// racy; one comfortably older is not; a time after it is.
    #[test]
    fn racy_is_judged_against_the_moment_of_stamping() {
        let at = 10_000_000_000;
        let stamp = |mtime_ns| FileStamp { size: 1, mtime_ns };
        assert!(!stamp(at - RACY_WINDOW_NS - 1).is_racy(at));
        assert!(stamp(at - RACY_WINDOW_NS).is_racy(at));
        assert!(stamp(at + 5).is_racy(at));
    }

    #[test]
    fn a_stamp_round_trips_its_wire_form() {
        let s = FileStamp {
            size: 3,
            mtime_ns: -5,
        };
        let wire = serde_json::to_string(&s).unwrap();
        assert_eq!(wire, r#"{"size":3,"mtime_ns":-5}"#);
        assert_eq!(serde_json::from_str::<FileStamp>(&wire).unwrap(), s);
    }
}
