//! Size-rotated file log sink for a process.
//!
//! A tracing sink beside stdout that writes the full configured log stream to
//! `<dir>/<name>` — `web/web.log` for the gateway, `substrate/zend.log` for
//! zend. It is what survives a process whose stdout nobody reads: `keep`
//! starts the gateway and `motherd` with theirs detached. When the active file
//! would exceed [`MAX_BYTES`] it rotates: `<name>.{N-1}` → `.{N}` (oldest
//! dropped), `<name>` → `.1`, then reopens an empty active file. On-disk log
//! size is therefore bounded at roughly `MAX_BYTES * (MAX_ARCHIVES + 1)`.
//!
//! An ordinary start rotates the previous run's file out the same way, so the
//! new run begins on an empty active file and the run before it is `.1`. The
//! run that just ended is the one most worth reading — it is the one that
//! failed, or was replaced by a deploy whose trial then failed and was rolled
//! back — and truncating on start destroyed exactly that record: two starts
//! in a row (a failed trial, then the stable relaunch) left no trace of either
//! the trial's fault or what the replaced process had been doing.
//!
//! A **self-heal restart** (`host::self_heal`) appends instead, so the fault
//! and the recovery read as one run. [`RESTART_MARKER_NAME`] is dropped next to
//! the log right before that relaunch; its presence here means "append".

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use tracing_subscriber::fmt::MakeWriter;

/// Per-file size cap before the active log rotates (32 MiB).
const MAX_BYTES: u64 = 32 * 1024 * 1024;
/// Number of rotated archives kept (`<name>.1` … `<name>.N`).
const MAX_ARCHIVES: usize = 4;
/// Dropped next to the log by `host::self_heal` right before it relaunches the
/// process. Its presence on the NEXT start means "this run continues one that
/// just crashed" — see the module doc.
pub const RESTART_MARKER_NAME: &str = ".self_heal_restart";

/// Whether `dir` (where the log lives) carries the self-heal restart marker —
/// checked BEFORE [`RotatingFileLog::new`] (which consumes it), so the caller
/// can still tell whether this start is a resume after logging is up and log
/// the boundary itself.
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
    /// Path of the active file (`<dir>/<name>`).
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
    /// Open the active log at `<dir>/<name>`, creating `dir` if missing.
    /// Returns `None` (with a stderr note) if the directory or file can't be
    /// opened — the process then proceeds with its other sinks only rather
    /// than aborting.
    ///
    /// Ordinarily the previous run's non-empty active file is rotated out to
    /// `.1` and this run starts on an empty one. When [`RESTART_MARKER_NAME`]
    /// is present next to the log (consumed here, so this happens at most once
    /// per relaunch), it instead appends to the existing active file — see the
    /// module doc.
    pub fn new(dir: &Path, name: &str) -> Option<Self> {
        if let Err(e) = fs::create_dir_all(dir) {
            eprintln!("could not create log dir {}: {e}", dir.display());
            return None;
        }
        let base = dir.join(name);
        let resuming = is_resuming(dir);
        if resuming {
            let _ = fs::remove_file(dir.join(RESTART_MARKER_NAME));
        }
        let file = match open_appending(&base) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("could not open log file {}: {e}", base.display());
                return None;
            }
        };
        // `written` reflects what is already on disk: a resumed run rotates
        // where it should rather than a full `MAX_BYTES` later, and an
        // ordinary start sees the previous run's bytes and rotates them out.
        let written = fs::metadata(&base).map(|m| m.len()).unwrap_or(0);
        let mut inner = Inner {
            file: Some(file),
            written,
            base,
        };
        if !resuming && inner.written > 0 {
            if let Err(e) = inner.rotate() {
                eprintln!(
                    "could not rotate the previous log {}: {e}",
                    inner.base.display()
                );
                return None;
            }
        }
        Some(Self(Arc::new(Mutex::new(inner))))
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

    use tempfile::TempDir;

    const NAME: &str = "zend.log";

    fn read(path: &Path) -> String {
        let mut s = String::new();
        File::open(path).unwrap().read_to_string(&mut s).unwrap();
        s
    }

    /// An ordinary start keeps the previous run: its file becomes `.1`, the
    /// older archives shift up, and the new run writes to an empty file.
    #[test]
    fn a_start_rotates_the_previous_run_out_instead_of_truncating_it() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();
        let base = dir.join(NAME);
        fs::write(&base, b"the run before\n").unwrap();
        fs::write(archive_path(&base, 1), b"two runs before\n").unwrap();

        let mut log = RotatingFileLog::new(dir, NAME).unwrap();
        assert_eq!(read(&base), "");
        assert_eq!(read(&archive_path(&base, 1)), "the run before\n");
        assert_eq!(read(&archive_path(&base, 2)), "two runs before\n");
        log.write_all(b"this run\n").unwrap();
        assert_eq!(read(&base), "this run\n");
    }

    /// A start over an empty file has nothing to keep and archives nothing.
    #[test]
    fn a_start_over_an_empty_log_archives_nothing() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join(NAME);
        fs::write(&base, b"").unwrap();

        let log = RotatingFileLog::new(tmp.path(), NAME).unwrap();
        assert_eq!(read(&base), "");
        assert!(!archive_path(&base, 1).exists());
        drop(log);
    }

    /// Starts past the archive limit drop the oldest run, so the set stays
    /// bounded however often the process restarts.
    #[test]
    fn repeated_starts_keep_at_most_the_archive_limit() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join(NAME);
        for run in 0..MAX_ARCHIVES + 3 {
            let mut log = RotatingFileLog::new(tmp.path(), NAME).unwrap();
            log.write_all(format!("run {run}\n").as_bytes()).unwrap();
        }
        let last = MAX_ARCHIVES + 2;
        assert_eq!(read(&base), format!("run {last}\n"));
        for n in 1..=MAX_ARCHIVES {
            assert_eq!(read(&archive_path(&base, n)), format!("run {}\n", last - n));
        }
        assert!(!archive_path(&base, MAX_ARCHIVES + 1).exists());
    }

    /// The name is the caller's: each binary writes its own file.
    #[test]
    fn the_file_takes_the_name_it_is_given() {
        let tmp = TempDir::new().unwrap();
        let mut log = RotatingFileLog::new(tmp.path(), "motherd.log").unwrap();
        log.write_all(b"up\n").unwrap();
        assert_eq!(read(&tmp.path().join("motherd.log")), "up\n");
    }

    /// The self-heal restart path: the marker means append and keep the
    /// archives, not the ordinary fresh-start truncate.
    #[test]
    fn resume_marker_appends_and_keeps_archives_then_consumes_itself() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();
        let base = dir.join(NAME);
        fs::write(&base, b"before the restart\n").unwrap();
        fs::write(archive_path(&base, 1), b"an older archive\n").unwrap();
        fs::write(dir.join(RESTART_MARKER_NAME), b"").unwrap();

        assert!(
            is_resuming(dir),
            "the marker must be visible before new() consumes it"
        );
        let mut log = RotatingFileLog::new(dir, NAME).unwrap();
        // The prior content survived, and the archive was left alone.
        assert_eq!(read(&base), "before the restart\n");
        assert!(archive_path(&base, 1).exists());
        // New writes append after what was already there.
        log.write_all(b"after the restart\n").unwrap();
        assert_eq!(read(&base), "before the restart\nafter the restart\n");
        // The marker is consumed: a second `new()` on the same directory goes
        // back to the ordinary start, which rotates the resumed run out.
        assert!(!dir.join(RESTART_MARKER_NAME).exists());
        drop(log);
        let log2 = RotatingFileLog::new(dir, NAME).unwrap();
        assert_eq!(read(&base), "");
        assert_eq!(
            read(&archive_path(&base, 1)),
            "before the restart\nafter the restart\n"
        );
        assert_eq!(read(&archive_path(&base, 2)), "an older archive\n");
        drop(log2);
    }

    #[test]
    fn rotates_when_active_exceeds_cap() {
        let tmp = TempDir::new().unwrap();
        let base = tmp.path().join(NAME);
        let mut w = RotatingFileLog::new(tmp.path(), NAME).unwrap();

        // Writing just over MAX_BYTES would be wasteful; exercise the rotate()
        // seam directly through the lock instead.
        {
            let mut inner = w.0.lock().unwrap();
            inner.written = MAX_BYTES; // pretend the active file is full
        }
        w.write_all(b"first line after full\n").unwrap();
        // The full active file moved to `.1`; the new active holds only the line.
        assert!(archive_path(&base, 1).exists());
        assert_eq!(read(&base), "first line after full\n");
        drop(w);
    }

    /// A resumed log's `written` must reflect the real on-disk size, or
    /// rotation would wait for another full `MAX_BYTES` stacked on top of
    /// whatever the crashed run had already written before this one started.
    #[test]
    fn resumed_written_reflects_the_existing_file_size_for_rotation() {
        let tmp = TempDir::new().unwrap();
        let dir = tmp.path();
        let base = dir.join(NAME);
        fs::write(&base, vec![b'x'; MAX_BYTES as usize]).unwrap();
        fs::write(dir.join(RESTART_MARKER_NAME), b"").unwrap();

        let mut w = RotatingFileLog::new(dir, NAME).unwrap();
        // A single further write should already be past the cap and rotate,
        // exactly as `rotates_when_active_exceeds_cap` expects when `written`
        // starts at `MAX_BYTES` — here from the resumed file's real size.
        w.write_all(b"first line after resume\n").unwrap();
        assert!(archive_path(&base, 1).exists());
        assert_eq!(read(&base), "first line after resume\n");
        drop(w);
    }
}
