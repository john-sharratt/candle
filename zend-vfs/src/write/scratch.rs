//! Scratch files the write operations need: a private index, and the three
//! versions of a file `merge-file` merges.
//!
//! Both live in the repository's own git folder, never a shared temp folder
//! where another user could create the path first, and each is deleted when
//! dropped.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

/// A name no other scratch file in this process — or another — uses.
fn unique(git_dir: &Path, tag: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    git_dir.join(format!("zen-{tag}-{}-{nanos}-{n}", std::process::id()))
}

/// An index file for one operation, named by `GIT_INDEX_FILE`, so the
/// user's index is never read or written. Git creates the file itself; it
/// and its lock are deleted on drop.
pub(crate) struct PrivateIndex(pub(crate) PathBuf);

impl PrivateIndex {
    pub(crate) fn new(git_dir: &Path) -> Self {
        Self(unique(git_dir, "index"))
    }
}

impl Drop for PrivateIndex {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        let mut lock = self.0.clone().into_os_string();
        lock.push(".lock");
        let _ = std::fs::remove_file(lock);
    }
}

/// A file holding `bytes`, created exclusively — it fails rather than
/// open a path something else already made.
pub(crate) struct ScratchFile(pub(crate) PathBuf);

impl ScratchFile {
    pub(crate) fn new(git_dir: &Path, tag: &str, bytes: &[u8]) -> std::io::Result<Self> {
        let path = unique(git_dir, tag);
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        let scratch = Self(path);
        file.write_all(bytes)?;
        Ok(scratch)
    }
}

impl Drop for ScratchFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_are_unique_and_in_the_given_folder() {
        let dir = tempfile::tempdir().unwrap();
        let (a, b) = (PrivateIndex::new(dir.path()), PrivateIndex::new(dir.path()));
        assert_ne!(a.0, b.0);
        assert_eq!(a.0.parent().unwrap(), dir.path());
    }

    #[test]
    fn an_index_and_its_lock_are_removed_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let index = PrivateIndex::new(dir.path());
        let path = index.0.clone();
        let mut lock = path.clone().into_os_string();
        lock.push(".lock");
        std::fs::write(&path, b"index").unwrap();
        std::fs::write(&lock, b"lock").unwrap();
        drop(index);
        assert!(!path.exists());
        assert!(!PathBuf::from(lock).exists());
    }

    #[test]
    fn a_scratch_file_holds_its_bytes_and_is_removed_on_drop() {
        let dir = tempfile::tempdir().unwrap();
        let f = ScratchFile::new(dir.path(), "t", b"bytes").unwrap();
        let path = f.0.clone();
        assert_eq!(std::fs::read(&path).unwrap(), b"bytes");
        drop(f);
        assert!(!path.exists());
    }
}
