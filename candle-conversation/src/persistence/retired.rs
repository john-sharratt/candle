//! Segment files a maintenance op took out of the log, still on disk.
//!
//! A drained segment leaves the log on its owner — its read handle closed,
//! its id no longer named — but its file is deleted afterwards, by whoever
//! holds the [`RetiredSegments`], off the owner. Unlinking a multi-GiB file
//! is filesystem work that can take seconds (measured: 14 s on one member),
//! and every write on the substrate waits for whatever the owner is doing.
//!
//! A crash before the unlink leaves the file behind, as a crash between the
//! relocation and the drop always could: every record carried out of it keeps
//! its identity, so a recovery that walks both copies applies each value once,
//! and the next maintenance pass reclaims the stale segment.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// Files no longer named by the log, to be unlinked off its owner.
#[derive(Debug, Default)]
#[must_use = "a retired segment's file stays on disk until it is unlinked"]
pub struct RetiredSegments {
    paths: Vec<PathBuf>,
}

/// What [`RetiredSegments::unlink`] did.
#[derive(Debug)]
pub struct Unlinked {
    /// How many files went.
    pub removed: usize,
    /// The files that would not go, with why: left for the next pass.
    pub failed: Vec<(PathBuf, io::Error)>,
    /// How long the unlinks took.
    pub took: Duration,
}

impl RetiredSegments {
    pub(super) fn push(&mut self, path: PathBuf) {
        self.paths.push(path);
    }

    /// The files, in the order they were retired.
    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    /// Unlink every file. A file already gone counts as removed.
    pub fn unlink(self) -> Unlinked {
        let started = Instant::now();
        let mut removed = 0;
        let mut failed = Vec::new();
        for path in self.paths {
            match remove(&path) {
                Ok(()) => removed += 1,
                Err(e) => failed.push((path, e)),
            }
        }
        Unlinked {
            removed,
            failed,
            took: started.elapsed(),
        }
    }
}

fn remove(path: &Path) -> io::Result<()> {
    match fs::remove_file(path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scratch(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("retired_{tag}_{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// **Retiring deletes nothing; unlinking deletes every file**, and a file
    /// already gone is not a failure.
    #[test]
    fn unlink_removes_every_retired_file() {
        let dir = scratch("unlink");
        let a = dir.join("seg-a.log");
        let b = dir.join("seg-b.log");
        fs::write(&a, b"a").unwrap();
        let mut retired = RetiredSegments::default();
        retired.push(a.clone());
        retired.push(b.clone());
        assert!(a.exists(), "retiring unlinked a file");
        let done = retired.unlink();
        assert_eq!(done.removed, 2, "{:?}", done.failed);
        assert!(done.failed.is_empty());
        assert!(!a.exists());
        fs::remove_dir_all(&dir).unwrap();
    }
}
