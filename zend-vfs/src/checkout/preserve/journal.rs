//! What a preservation has done so far, on disk, so that it can be undone
//! by whoever finds it — the run that made it, or the next one after a crash.
//!
//! Each preservation has a folder of its own under the repository's git
//! folder, `zend-preserved/<id>/`, holding the journal, the list of what was
//! ignored (`ignored`), and the files moved aside (`files/`). The journal is
//! rewritten, whole, through a temporary file flushed to disk and renamed
//! over it, before every step that changes what an undo must do — so a crash
//! at any moment leaves the old journal or the new one, never a torn one,
//! and never one that is behind what was done.

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use serde::{Deserialize, Serialize};

use super::ignored::Ignored;
use crate::checkout::CheckoutError;

/// The folder in the git folder that holds every preservation.
pub(super) const PRESERVED_DIR: &str = "zend-preserved";

const JOURNAL: &str = "journal.yaml";
const IGNORED: &str = "ignored";
const FILES: &str = "files";

/// How far a preservation got.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum Phase {
    /// The checkout's state is being read: nothing in it has been changed
    /// yet, except for files already moved aside.
    Capturing,
    /// Everything is captured; the run may change the checkout.
    Preserved,
    /// Putting the checkout back has begun: it is part run's, part its own.
    Restoring,
    /// The checkout is back as it was; only the refs and this folder are
    /// left to let go of.
    Restored,
}

/// Where `HEAD` was.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SavedHead {
    /// The branch it was on; absent when detached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub branch: Option<String>,
    pub commit: String,
}

/// A branch the run moved off commits origin never had, and the commit it
/// goes back to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SavedBranch {
    pub branch: String,
    pub commit: String,
}

/// A file git was told to leave alone in the working tree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Flag {
    pub path: String,
    #[serde(default)]
    pub skip_worktree: bool,
    #[serde(default)]
    pub assume_unchanged: bool,
}

/// One preservation's journal. See the module.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Journal {
    pub why: String,
    pub phase: Phase,
    pub head: SavedHead,
    /// The commit the run puts the checkout on; absent when the caller did
    /// not say. With `head`, what a checkout left mid-run can hold.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// A branch set aside for the run — moved off the commits it holds that
    /// origin never had — which the restore puts back at its own commit,
    /// held meanwhile by a ref of the preservation's own.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub set_aside: Option<SavedBranch>,
    /// The commit holding the index as it was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub index: Option<String>,
    /// The commit holding, byte for byte, every working-tree file that
    /// differed from the index or was untracked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub files: Option<String>,
    /// Paths that held no file — tracked ones deleted, or a folder where the
    /// index has a file.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deleted: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub flags: Vec<Flag>,
    /// Paths added with `git add -N`: in the index, with no content yet.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub intent_to_add: Vec<String>,
    /// Each captured file's permission bits, where the file system has them.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub perms: BTreeMap<String, u32>,
    /// Captured links that pointed at a folder.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub link_dirs: Vec<String>,
    /// Captured paths the checkout had not changed, taken only to keep
    /// their bytes.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub unchanged: Vec<String>,
    /// `.git/info/exclude` as it was, base64; absent when there was none.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub exclude: Option<String>,
    /// Paths the run may write where nothing stood.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub vacant: Vec<String>,
    /// Paths whose ignored file or folder is moved aside, into `files/` —
    /// each recorded before it is moved, so a path listed here whose copy
    /// is not in `files/` was never moved, or is back already.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub moved: Vec<String>,
}

impl Journal {
    pub(super) fn exclude_bytes(&self) -> Result<Option<Vec<u8>>, CheckoutError> {
        self.exclude
            .as_ref()
            .map(|e| {
                STANDARD
                    .decode(e)
                    .map_err(|e| CheckoutError::journal(format!("exclude is not base64: {e}")))
            })
            .transpose()
    }

    pub(super) fn set_exclude(&mut self, bytes: Option<&[u8]>) {
        self.exclude = bytes.map(|b| STANDARD.encode(b));
    }
}

/// One preservation's folder.
#[derive(Debug, Clone)]
pub(super) struct Place {
    pub dir: PathBuf,
}

impl Place {
    /// A folder no other preservation — in this process or another — uses,
    /// named so that names sort in the order they were made.
    pub(super) fn new(git_dir: &Path) -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let id = format!(
            "{nanos:020}-{}-{:06}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        );
        Self {
            dir: git_dir.join(PRESERVED_DIR).join(id),
        }
    }

    /// Every preservation a previous run left, oldest first.
    pub(super) fn left_in(git_dir: &Path) -> Result<Vec<Self>, CheckoutError> {
        let root = git_dir.join(PRESERVED_DIR);
        let entries = match std::fs::read_dir(&root) {
            Ok(entries) => entries,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(CheckoutError::io(PRESERVED_DIR, e)),
        };
        let mut found: Vec<Self> = Vec::new();
        for entry in entries {
            let entry = entry.map_err(|e| CheckoutError::io(PRESERVED_DIR, e))?;
            if entry.path().is_dir() {
                found.push(Self { dir: entry.path() });
            }
        }
        found.sort_by(|a, b| a.dir.cmp(&b.dir));
        Ok(found)
    }

    /// Its id: the folder's name, which also names its refs.
    pub(super) fn id(&self) -> String {
        self.dir
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    /// Where a file moved aside from `path` is kept.
    pub(super) fn kept(&self, path: &str) -> PathBuf {
        self.dir.join(FILES).join(path)
    }

    pub(super) fn save(&self, journal: &Journal) -> Result<(), CheckoutError> {
        let text = serde_yaml::to_string(journal)
            .map_err(|e| CheckoutError::journal(format!("could not be written: {e}")))?;
        self.write_durably(JOURNAL, text.as_bytes())
    }

    pub(super) fn load(&self) -> Result<Journal, CheckoutError> {
        let path = self.dir.join(JOURNAL);
        let text = std::fs::read_to_string(&path)
            .map_err(|e| CheckoutError::io(&path.to_string_lossy(), e))?;
        serde_yaml::from_str(&text)
            .map_err(|e| CheckoutError::journal(format!("{} cannot be read: {e}", path.display())))
    }

    pub(super) fn save_ignored(&self, ignored: &Ignored) -> Result<(), CheckoutError> {
        self.write_durably(IGNORED, &ignored.to_bytes())
    }

    /// What was ignored when this preservation was made; nothing, for one
    /// that crashed before it was listed.
    pub(super) fn load_ignored(&self) -> Result<Ignored, CheckoutError> {
        let path = self.dir.join(IGNORED);
        match std::fs::read(&path) {
            Ok(bytes) => Ok(Ignored::from_bytes(&bytes)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Ignored::default()),
            Err(e) => Err(CheckoutError::io(&path.to_string_lossy(), e)),
        }
    }

    /// `bytes` as `name` in the folder: written to a temporary file, flushed
    /// to disk, then renamed over the old one, and the rename flushed where
    /// the file system allows it.
    fn write_durably(&self, name: &str, bytes: &[u8]) -> Result<(), CheckoutError> {
        let fail = |e| CheckoutError::io(&self.dir.to_string_lossy(), e);
        std::fs::create_dir_all(&self.dir).map_err(fail)?;
        let temp = self.dir.join(format!("{name}.new"));
        File::create(&temp)
            .and_then(|mut f| f.write_all(bytes).and_then(|()| f.sync_all()))
            .map_err(fail)?;
        std::fs::rename(&temp, self.dir.join(name)).map_err(fail)?;
        #[cfg(unix)]
        File::open(&self.dir)
            .and_then(|d| d.sync_all())
            .map_err(fail)?;
        Ok(())
    }

    /// The folder, gone — once nothing in it is needed.
    pub(super) fn remove(&self) -> Result<(), CheckoutError> {
        match std::fs::remove_dir_all(&self.dir) {
            Ok(()) => Ok(()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(CheckoutError::io(&self.dir.to_string_lossy(), e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn journal() -> Journal {
        let mut journal = Journal {
            why: "a job ran".into(),
            phase: Phase::Preserved,
            head: SavedHead {
                branch: Some("main".into()),
                commit: "ce013625030ba8dba906f756967f9e9ca394464a".into(),
            },
            target: Some("7898192261b8b8d7ab18ee7faa5b2d26fd8b35cc".into()),
            set_aside: Some(SavedBranch {
                branch: "main".into(),
                commit: "ce013625030ba8dba906f756967f9e9ca394464a".into(),
            }),
            index: Some("4b825dc642cb6eb9a060e54bf8d69288fbee4904".into()),
            files: None,
            deleted: vec!["gone.txt".into()],
            flags: vec![Flag {
                path: "local.cfg".into(),
                skip_worktree: true,
                assume_unchanged: false,
            }],
            intent_to_add: vec!["later.txt".into()],
            perms: [("secret.key".to_string(), 0o600)].into(),
            link_dirs: vec!["linked".into()],
            unchanged: vec!["a.txt".into()],
            exclude: None,
            vacant: vec!["new.txt".into()],
            moved: vec![".env".into()],
        };
        journal.set_exclude(Some(b"*.local\n"));
        journal
    }

    /// **A journal reads back exactly what was saved**, the exclude file's
    /// bytes included; a second save replaces the first whole.
    #[test]
    fn a_journal_reads_back_as_saved() {
        let dir = tempfile::tempdir().unwrap();
        let place = Place::new(dir.path());
        let mut saved = journal();
        place.save(&saved).unwrap();
        assert_eq!(place.load().unwrap(), saved);
        assert_eq!(
            saved.exclude_bytes().unwrap().as_deref(),
            Some(&b"*.local\n"[..])
        );

        saved.moved.clear();
        saved.phase = Phase::Capturing;
        place.save(&saved).unwrap();
        assert_eq!(place.load().unwrap(), saved);
        assert!(!place.dir.join("journal.yaml.new").exists());
    }

    /// **What a previous run left is found**, in the order it was made, and
    /// a folder removed is gone.
    #[test]
    fn what_was_left_is_found_in_order() {
        let dir = tempfile::tempdir().unwrap();
        assert!(Place::left_in(dir.path()).unwrap().is_empty());
        let (a, b) = (Place::new(dir.path()), Place::new(dir.path()));
        a.save(&journal()).unwrap();
        b.save(&journal()).unwrap();
        let left: Vec<String> = Place::left_in(dir.path())
            .unwrap()
            .iter()
            .map(Place::id)
            .collect();
        assert_eq!(left, [a.id(), b.id()]);
        a.remove().unwrap();
        assert_eq!(Place::left_in(dir.path()).unwrap().len(), 1);
        a.remove().unwrap();
    }

    /// **Preservations are found oldest first whatever made them** — the
    /// time leads the name, so a longer process id never sorts first.
    #[test]
    fn the_oldest_is_found_first_across_processes() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join(PRESERVED_DIR);
        for id in [
            "00000000000000000020-7-000000",
            "00000000000000000010-99999-000000",
        ] {
            Place { dir: root.join(id) }.save(&journal()).unwrap();
        }
        let left: Vec<String> = Place::left_in(dir.path())
            .unwrap()
            .iter()
            .map(Place::id)
            .collect();
        assert_eq!(
            left,
            [
                "00000000000000000010-99999-000000",
                "00000000000000000020-7-000000"
            ]
        );
        let made = Place::new(dir.path()).id();
        assert!(made.as_str() > left[1].as_str(), "{made}");
    }

    /// **The ignored list reads back as saved**, and a preservation that
    /// never listed one has nothing ignored.
    #[test]
    fn the_ignored_list_reads_back() {
        let dir = tempfile::tempdir().unwrap();
        let place = Place::new(dir.path());
        assert_eq!(place.load_ignored().unwrap(), Ignored::default());
        let ignored = Ignored::from_bytes(b"target/\0.env\0");
        place.save_ignored(&ignored).unwrap();
        assert_eq!(place.load_ignored().unwrap(), ignored);
        assert!(!place.dir.join("ignored.new").exists());
    }
}
