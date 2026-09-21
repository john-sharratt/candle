//! Size-rotated file log sink for the daemon.
//!
//! A third tracing subscriber (alongside stdout and the WebSocket bus) writes
//! the full configured log stream to `<workspace>/.substrate/zend.log`. The
//! active file is truncated on every ORDINARY daemon start — and any archives
//! from the prior run are removed — so each run begins from a clean set. When
//! the active file would exceed [`MAX_BYTES`] it rotates: `zend.log.{N-1}` →
//! `.{N}` (oldest dropped), `zend.log` → `.1`, then reopens an empty active
//! file. On-disk log size is therefore bounded at roughly
//! `MAX_BYTES * (MAX_ARCHIVES + 1)`.
//!
//! The one exception is a **self-heal restart** (`self_heal.rs`): the whole
//! reason that relaunch happens is a fault worth diagnosing, and truncating
//! the log on the very next start would destroy the only record of it before
//! anyone could read it — which is exactly what made an earlier out-of-memory
//! stall invisible until the daemon was restarted by hand. [`RESTART_MARKER_NAME`]
//! is dropped next to the log right before that relaunch; its presence here
//! means "append, keep the archives" instead of the ordinary fresh start.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tracing_subscriber::fmt::MakeWriter;

/// Per-file size cap before the active log rotates (32 MiB).
const MAX_BYTES: u64 = 32 * 1024 * 1024;
/// Number of rotated archives kept (`zend.log.1` … `zend.log.N`).
const MAX_ARCHIVES: usize = 4;
/// Active log file name under `.substrate/`.
pub const LOG_NAME: &str = "zend.log";
/// Dropped next to the log by `self_heal` right before it relaunches the
/// process. Its presence on the NEXT start means "this run continues one that
/// just crashed" — see the module doc.
pub const RESTART_MARKER_NAME: &str = ".self_heal_restart";

/// Whether `dir` (the daemon's `.substrate`) carries the self-heal restart
/// marker — checked BEFORE [`RotatingFileLog::new`] (which consumes it), so
/// the caller can still tell whether this start is a resume after logging is
/// up and log the boundary itself.
pub fn is_resuming(dir: &Path) -> bool {
    dir.join(RESTART_MARKER_NAME).exists()
}

/// `…/zend.log` → `…/zend.log.{n}` (append, not extension-replace).
fn archive_path(base: &Path, n: usize) -> PathBuf {
    let mut s = base.to_path_buf().into_os_string();
    s.push(format!(".{n}"));
    PathBuf::from(s)
}

fn open_truncated(path: &Path) -> io::Result<File> {
    OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(path)
}

fn open_appending(path: &Path) -> io::Result<File> {
    OpenOptions::new().create(true).append(true).open(path)
}

struct Inner {
    /// Active log file (`None` only transiently mid-rotation).
    file: Option<File>,
    /// Bytes written to the active file so far.
    written: u64,
    /// Path of the active file (`…/zend.log`).
    base: PathBuf,
}

impl Inner {
    /// Rotate the active file out to `.1`, shifting existing archives up and
    /// dropping the oldest, then reopen a fresh empty active file.
    fn rotate(&mut self) -> io::Result<()> {
        // Close the active file before renaming: Windows can't rename an open
        // handle, and rename fails when the destination exists — so vacate each
        // slot from oldest to newest before moving into it.
        if let Some(mut f) = self.file.take() {
            let _ = f.flush();
        }
        let _ = fs::remove_file(archive_path(&self.base, MAX_ARCHIVES));
        for n in (1..MAX_ARCHIVES).rev() {
            let src = archive_path(&self.base, n);
            if src.exists() {
                let _ = fs::rename(&src, archive_path(&self.base, n + 1));
            }
        }
        let _ = fs::rename(&self.base, archive_path(&self.base, 1));
        self.file = Some(open_truncated(&self.base)?);
        self.written = 0;
        Ok(())
    }

    fn write_event(&mut self, buf: &[u8]) -> io::Result<()> {
        if self.written > 0 && self.written + buf.len() as u64 > MAX_BYTES {
            self.rotate()?;
        }
        if let Some(f) = self.file.as_mut() {
            f.write_all(buf)?;
            self.written += buf.len() as u64;
        }
        Ok(())
    }
}

/// Handle plugged into `tracing_subscriber::fmt::layer().with_writer(...)`.
///
/// The fmt layer writes one formatted event per `write_all`, so a single lock
/// per call keeps whole log lines intact under concurrency.
#[derive(Clone)]
pub struct RotatingFileLog(Arc<Mutex<Inner>>);

impl RotatingFileLog {
    /// Open the active log at `<dir>/zend.log`, creating `dir` if missing.
    /// Returns `None` (with a stderr note) if the directory or file can't be
    /// opened — boot then proceeds with the stdout + bus sinks only rather
    /// than aborting.
    ///
    /// Ordinarily this truncates the active file and clears previous archives
    /// so each daemon run starts from an empty set. When [`RESTART_MARKER_NAME`]
    /// is present next to the log (consumed here, so this happens at most
    /// once per relaunch), it instead appends to the existing active file and
    /// leaves the archives alone — see the module doc.
    pub fn new(dir: &Path) -> Option<Self> {
        if let Err(e) = fs::create_dir_all(dir) {
            eprintln!("zend: could not create log dir {}: {e}", dir.display());
            return None;
        }
        let base = dir.join(LOG_NAME);
        let resuming = is_resuming(dir);
        if resuming {
            let _ = fs::remove_file(dir.join(RESTART_MARKER_NAME));
        } else {
            for n in 1..=MAX_ARCHIVES {
                let _ = fs::remove_file(archive_path(&base, n));
            }
        }
        let open = if resuming {
            open_appending
        } else {
            open_truncated
        };
        let file = match open(&base) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("zend: could not open log file {}: {e}", base.display());
                return None;
            }
        };
        // Resuming: `written` must reflect what's already on disk, or
        // rotation waits for another full `MAX_BYTES` on top of it instead of
        // triggering where it should.
        let written = if resuming {
            fs::metadata(&base).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };
        Some(Self(Arc::new(Mutex::new(Inner {
            file: Some(file),
            written,
            base,
        }))))
    }
}

impl Write for RotatingFileLog {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if let Ok(mut inner) = self.0.lock() {
            let _ = inner.write_event(buf);
        }
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        if let Ok(mut inner) = self.0.lock() {
            if let Some(f) = inner.file.as_mut() {
                f.flush()?;
            }
        }
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for RotatingFileLog {
    type Writer = RotatingFileLog;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Read;

    fn read(path: &Path) -> String {
        let mut s = String::new();
        File::open(path).unwrap().read_to_string(&mut s).unwrap();
        s
    }

    #[test]
    fn truncates_active_and_clears_archives_on_new() {
        let dir = std::env::temp_dir().join(format!("zendlog-trunc-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let base = dir.join(LOG_NAME);
        fs::create_dir_all(&dir).unwrap();
        // Seed a stale active file and a stale archive from a "prior run".
        fs::write(&base, b"stale active\n").unwrap();
        fs::write(archive_path(&base, 1), b"stale archive\n").unwrap();

        let log = RotatingFileLog::new(&dir).unwrap();
        // Active file is empty (truncated), prior archive is gone.
        assert_eq!(read(&base), "");
        assert!(!archive_path(&base, 1).exists());
        drop(log);
        let _ = fs::remove_dir_all(&dir);
    }

    /// The self-heal restart path: the marker means append and keep the
    /// archives, not the ordinary fresh-start truncate.
    #[test]
    fn resume_marker_appends_and_keeps_archives_then_consumes_itself() {
        let dir = std::env::temp_dir().join(format!("zendlog-resume-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let base = dir.join(LOG_NAME);
        fs::create_dir_all(&dir).unwrap();
        fs::write(&base, b"before the restart\n").unwrap();
        fs::write(archive_path(&base, 1), b"an older archive\n").unwrap();
        fs::write(dir.join(RESTART_MARKER_NAME), b"").unwrap();

        assert!(
            is_resuming(&dir),
            "the marker must be visible before new() consumes it"
        );
        let mut log = RotatingFileLog::new(&dir).unwrap();
        // The prior content survived, and the archive was left alone.
        assert_eq!(read(&base), "before the restart\n");
        assert!(archive_path(&base, 1).exists());
        // New writes append after what was already there.
        log.write_all(b"after the restart\n").unwrap();
        assert_eq!(read(&base), "before the restart\nafter the restart\n");
        // The marker is consumed: a second `new()` on the same directory goes
        // back to the ordinary truncating start.
        assert!(!dir.join(RESTART_MARKER_NAME).exists());
        drop(log);
        let log2 = RotatingFileLog::new(&dir).unwrap();
        assert_eq!(read(&base), "");
        assert!(!archive_path(&base, 1).exists());
        drop(log2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn rotates_when_active_exceeds_cap() {
        let dir = std::env::temp_dir().join(format!("zendlog-rot-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let base = dir.join(LOG_NAME);
        let mut w = RotatingFileLog::new(&dir).unwrap();

        // Force a tiny cap by writing just over MAX_BYTES would be wasteful;
        // instead exercise the rotate() seam directly through the lock.
        {
            let mut inner = w.0.lock().unwrap();
            inner.written = MAX_BYTES; // pretend the active file is full
        }
        w.write_all(b"first line after full\n").unwrap();
        // The full active file moved to `.1`; the new active holds only the line.
        assert!(archive_path(&base, 1).exists());
        assert_eq!(read(&base), "first line after full\n");
        drop(w);
        let _ = fs::remove_dir_all(&dir);
    }

    /// A resumed log's `written` must reflect the real on-disk size, or
    /// rotation would wait for another full `MAX_BYTES` stacked on top of
    /// whatever the crashed run had already written before this one started.
    #[test]
    fn resumed_written_reflects_the_existing_file_size_for_rotation() {
        let dir = std::env::temp_dir().join(format!("zendlog-resume-rot-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let base = dir.join(LOG_NAME);
        fs::create_dir_all(&dir).unwrap();
        fs::write(&base, vec![b'x'; MAX_BYTES as usize]).unwrap();
        fs::write(dir.join(RESTART_MARKER_NAME), b"").unwrap();

        let mut w = RotatingFileLog::new(&dir).unwrap();
        // A single further write should already be past the cap and rotate,
        // exactly as `rotates_when_active_exceeds_cap` expects when `written`
        // starts at `MAX_BYTES` — here from the resumed file's real size.
        w.write_all(b"first line after resume\n").unwrap();
        assert!(archive_path(&base, 1).exists());
        assert_eq!(read(&base), "first line after resume\n");
        drop(w);
        let _ = fs::remove_dir_all(&dir);
    }
}
