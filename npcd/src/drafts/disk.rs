//! The mind folder written directly, for a mind that is not a git repository
//! (`docs/replicated_substrate.md` §10.3).
//!
//! With no record to land a draft on, what an author writes is what stands.
//! Each document is written atomically: to a temporary beside it, made
//! durable, then renamed over it. A plain truncating write empties the file
//! before it writes, so an interruption in that window leaves the file gone
//! rather than merely unchanged — and on NTFS the size metadata can land while
//! the data does not, leaving a file of the right length full of `NUL`.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::mind::path::MindPath;
use crate::sim::bench::refuse_link;
use crate::sim::shelf::{Published, Shelf};

/// The mind folder at `root`, read and written as it stands.
pub struct Disk {
    root: PathBuf,
    pub folder: Published,
}

impl Disk {
    pub fn new(root: impl Into<PathBuf>) -> Disk {
        let root = root.into();
        Disk {
            folder: Published::folder(root.clone()),
            root,
        }
    }

    /// Where `key` is in the folder, refusing a path that leaves it or a link.
    fn file(&self, key: &str) -> Option<PathBuf> {
        let full = MindPath::parse(key).ok()?.resolve(&self.root).ok()?;
        refuse_link(&full, key).ok()?;
        Some(full)
    }
}

impl Shelf for Disk {
    fn read(&self, key: &str) -> Option<String> {
        self.folder.read(key)
    }

    fn list(&self, dir: &str) -> BTreeSet<String> {
        self.folder.list(dir)
    }

    fn size(&self, key: &str) -> Option<u64> {
        self.folder.size(key)
    }

    /// Every path is checked before anything is written, so a batch naming a
    /// path that leaves the mind, a link or a folder writes nothing. Past that,
    /// each document is replaced on its own: a disk cannot replace two files
    /// in one step.
    fn commit(&self, changes: &[(String, Option<String>)]) -> Result<(), String> {
        let mut planned = Vec::with_capacity(changes.len());
        for (key, now) in changes {
            let full = self
                .file(key)
                .ok_or_else(|| format!("{key} is not a document in the mind"))?;
            if full.is_dir() {
                return Err(format!("{key} is a folder, not a document"));
            }
            planned.push((key, full, now));
        }
        for (key, full, now) in planned {
            match now {
                Some(text) => write_atomic(&full, text.as_bytes())
                    .map_err(|e| format!("{key} could not be written: {e}"))?,
                None => match fs::remove_file(&full) {
                    Ok(()) => {}
                    Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
                    Err(e) => return Err(format!("{key} could not be removed: {e}")),
                },
            }
        }
        Ok(())
    }
}

/// Write `bytes` to `target` without ever leaving it partly written, creating
/// its folder first. The temporary lives in the same folder so the rename is
/// within one filesystem.
fn write_atomic(target: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    fs::create_dir_all(dir)?;
    let name = target
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("document");
    let tmp = dir.join(format!(".{name}.{}.tmp", std::process::id()));
    // Closed before the rename: Windows refuses to rename over a file that is
    // still open.
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        // Durable before it is visible, or the rename can be ordered ahead of
        // the data on a crash.
        f.sync_all()?;
    }
    fs::rename(&tmp, target).inspect_err(|_| {
        let _ = fs::remove_file(&tmp);
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// **What is committed is on disk**, new folders and all, and a removal
    /// of something already gone is not an error.
    #[test]
    fn a_commit_writes_and_removes_documents() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("old.md"), "old\n").unwrap();
        let disk = Disk::new(dir.path());
        disk.commit(&[
            ("worlds/w.yaml".into(), Some("name: W\n".into())),
            ("old.md".into(), None),
            ("never.md".into(), None),
        ])
        .unwrap();
        assert_eq!(
            fs::read_to_string(dir.path().join("worlds/w.yaml")).unwrap(),
            "name: W\n"
        );
        assert!(!dir.path().join("old.md").exists());
        assert_eq!(disk.read("worlds/w.yaml").as_deref(), Some("name: W\n"));
        assert_eq!(disk.size("worlds/w.yaml"), Some(8));
        let left: Vec<_> = fs::read_dir(dir.path().join("worlds"))
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .collect();
        assert_eq!(left, ["w.yaml"], "a temporary was left behind");
    }

    /// **A batch naming a path that cannot be written writes nothing.**
    #[test]
    fn a_batch_with_a_bad_path_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir_all(dir.path().join("worlds")).unwrap();
        let disk = Disk::new(dir.path());
        for bad in ["../escape.md", "worlds"] {
            assert!(disk
                .commit(&[
                    ("a.md".into(), Some("a\n".into())),
                    (bad.into(), Some("x\n".into())),
                ])
                .is_err());
            assert!(!dir.path().join("a.md").exists(), "{bad}");
        }
    }
}
