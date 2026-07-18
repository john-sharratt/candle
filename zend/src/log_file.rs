//! Size-rotated file log sink for the daemon.
//!
//! A third tracing subscriber (alongside stdout and the WebSocket bus) writes
//! the full configured log stream to `<workspace>/.substrate/zend.log`. The
//! active file is truncated on every daemon start — and any archives from the
//! prior run are removed — so each run begins from a clean set. When the active
//! file would exceed [`MAX_BYTES`] it rotates: `zend.log.{N-1}` → `.{N}` (oldest
//! dropped), `zend.log` → `.1`, then reopens an empty active file. On-disk log
//! size is therefore bounded at roughly `MAX_BYTES * (MAX_ARCHIVES + 1)`.

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
    /// Open a fresh active log at `<dir>/zend.log`, creating `dir` if missing,
    /// truncating any prior active file, and clearing previous archives so each
    /// daemon run starts from an empty set. Returns `None` (with a stderr note)
    /// if the directory or file can't be opened — boot then proceeds with the
    /// stdout + bus sinks only rather than aborting.
    pub fn new(dir: &Path) -> Option<Self> {
        if let Err(e) = fs::create_dir_all(dir) {
            eprintln!("zend: could not create log dir {}: {e}", dir.display());
            return None;
        }
        let base = dir.join(LOG_NAME);
        for n in 1..=MAX_ARCHIVES {
            let _ = fs::remove_file(archive_path(&base, n));
        }
        let file = match open_truncated(&base) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("zend: could not open log file {}: {e}", base.display());
                return None;
            }
        };
        Some(Self(Arc::new(Mutex::new(Inner {
            file: Some(file),
            written: 0,
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
}
