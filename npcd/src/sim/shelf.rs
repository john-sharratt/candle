//! Where a body's documents stand, beneath its own uncommitted changes
//! (`docs/replicated_substrate.md` §10.3).
//!
//! A body carrying a mission that is a step of an operation works in that
//! operation's store: its reads fall through to the store, and its commit
//! records into it. A body with no such mission reads the published mind —
//! the mind folder as it stands — and has nowhere to commit: nothing reaches
//! the record but through an operation's publish step.

use std::collections::BTreeSet;
use std::path::PathBuf;

use crate::mind::path::MindPath;
use crate::sim::bench::refuse_link;

/// What a body's documents stand on. Shared, so an act may hold one across
/// its awaits.
pub trait Shelf: Send + Sync {
    /// What `key` holds as it stands, or `None` when nothing is there.
    fn read(&self, key: &str) -> Option<String>;

    /// What is directly in the folder `dir`: documents by name, folders with
    /// a trailing `/`.
    fn list(&self, dir: &str) -> BTreeSet<String>;

    /// Make `changes` stand, all at once — each document's new text, or
    /// `None` to take it out. `Err` refuses them, and nothing changed.
    fn commit(&self, changes: &[(String, Option<String>)]) -> Result<(), String>;

    /// How many bytes `key` holds, or `None` when nothing is there. What a
    /// listing shows beside each document, so a shelf that can answer it
    /// without reading the document does.
    fn size(&self, key: &str) -> Option<u64> {
        self.read(key).map(|text| text.len() as u64)
    }
}

/// The published mind, read from its folder; it takes no commit.
pub struct Published {
    pub root: PathBuf,
}

impl Published {
    /// Where `key` is in the folder, refusing a path that leaves it or a link.
    pub fn file(&self, key: &str) -> Option<PathBuf> {
        let full = MindPath::parse(key).ok()?.resolve(&self.root).ok()?;
        refuse_link(&full, key).ok()?;
        Some(full)
    }

    /// Whether something that is not a document stands at `key`: a folder, or
    /// a link, which is never followed.
    pub fn not_a_document(&self, key: &str) -> bool {
        let Some(full) = MindPath::parse(key)
            .ok()
            .and_then(|p| p.resolve(&self.root).ok())
        else {
            return false;
        };
        match full.symlink_metadata() {
            Ok(meta) => !meta.is_file() || refuse_link(&full, key).is_err(),
            Err(_) => false,
        }
    }
}

/// What a body is told when it commits with no operation to commit into.
pub const NO_OPERATION: &str = "There is nowhere for this to go: work reaches what stands only \
                                through an operation, and you are on no step of one.";

impl Shelf for Published {
    fn read(&self, key: &str) -> Option<String> {
        std::fs::read_to_string(self.file(key)?).ok()
    }

    fn size(&self, key: &str) -> Option<u64> {
        let meta = self.file(key)?.metadata().ok()?;
        meta.is_file().then_some(meta.len())
    }

    /// A folder that is a link lists as nothing: a junction would otherwise
    /// report documents from somewhere the character never named.
    fn list(&self, dir: &str) -> BTreeSet<String> {
        let Some(full) = MindPath::parse(dir)
            .ok()
            .and_then(|p| p.resolve(&self.root).ok())
            .filter(|full| refuse_link(full, dir).is_ok())
        else {
            return BTreeSet::new();
        };
        let Ok(entries) = std::fs::read_dir(full) else {
            return BTreeSet::new();
        };
        entries
            .flatten()
            .map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                match e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                    true => format!("{name}/"),
                    false => name,
                }
            })
            .collect()
    }

    fn commit(&self, _changes: &[(String, Option<String>)]) -> Result<(), String> {
        Err(NO_OPERATION.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The published mind reads its folder and refuses every commit.
    #[test]
    fn the_published_mind_reads_and_takes_no_commit() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("layers/eras")).unwrap();
        std::fs::write(dir.path().join("layers/eras/third.md"), "the third era\n").unwrap();
        let shelf = Published {
            root: dir.path().to_path_buf(),
        };
        assert_eq!(
            shelf.read("layers/eras/third.md").as_deref(),
            Some("the third era\n")
        );
        assert_eq!(shelf.read("layers/eras/fourth.md"), None);
        assert_eq!(shelf.size("layers/eras/third.md"), Some(14));
        assert_eq!(shelf.size("layers/eras"), None);
        assert_eq!(shelf.size("layers/eras/fourth.md"), None);
        assert!(shelf.not_a_document("layers/eras"));
        assert!(!shelf.not_a_document("layers/eras/third.md"));
        assert!(!shelf.not_a_document("layers/eras/fourth.md"));
        assert_eq!(
            shelf.list("layers").into_iter().collect::<Vec<_>>(),
            ["eras/"]
        );
        assert_eq!(
            shelf.commit(&[("layers/x.md".into(), Some("x".into()))]),
            Err(NO_OPERATION.to_string())
        );
    }
}
