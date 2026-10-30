//! Files the scheduler keeps for its next start — the published rates and the
//! rate model's learned state — written on a thread of their own.
//!
//! The scheduler publishes as it runs, and a write that waits on the disk on
//! its thread is a wait between two waves. It hands the text here and goes on.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crossbeam::channel::{self, Sender};

static WRITER: OnceLock<Sender<(PathBuf, String)>> = OnceLock::new();

/// Replace `path` with `text`, whole (written beside it, then renamed over
/// it), on the store's thread. Best effort: a store that cannot be written
/// costs the next start its head start, not this run anything.
pub fn write_behind(path: PathBuf, text: String) {
    let writer = WRITER.get_or_init(|| {
        let (tx, rx) = channel::unbounded::<(PathBuf, String)>();
        std::thread::Builder::new()
            .name("scheduler-store".into())
            .spawn(move || {
                for (path, text) in rx {
                    replace(&path, &text);
                }
            })
            .expect("spawn the scheduler's store thread");
        tx
    });
    let _ = writer.send((path, text));
}

fn replace(path: &Path, text: &str) {
    let tmp = path.with_extension("tmp");
    if std::fs::write(&tmp, text).is_ok() {
        let _ = std::fs::rename(&tmp, path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::time::{Duration, Instant};

    /// **A write lands whole, off the caller's thread**, and a later write to
    /// the same path replaces it.
    #[test]
    fn a_write_lands_and_a_later_one_replaces_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("rates");
        write_behind(path.clone(), "first\n".to_string());
        write_behind(path.clone(), "second\n".to_string());
        let deadline = Instant::now() + Duration::from_secs(10);
        while std::fs::read_to_string(&path).ok().as_deref() != Some("second\n") {
            assert!(Instant::now() < deadline, "the store never wrote the file");
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(!path.with_extension("tmp").exists(), "renamed into place");
    }
}
