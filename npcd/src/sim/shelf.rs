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
use std::sync::Arc;

use crate::mind::files::MindFiles;

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

/// The published mind — its files at the commit the log names, or a plain
/// folder ([`MindFiles`]); it takes no commit.
#[derive(Clone)]
pub struct Published {
    files: Arc<MindFiles>,
}

impl Published {
    /// The published mind as `files` hold it.
    pub fn of(files: Arc<MindFiles>) -> Self {
        Self { files }
    }

    /// The published mind as the folder `root` holds it.
    pub fn folder(root: impl Into<PathBuf>) -> Self {
        Self::of(Arc::new(MindFiles::folder(root)))
    }

    /// The files beneath.
    pub fn files(&self) -> &Arc<MindFiles> {
        &self.files
    }

    /// Whether something that is not a document stands at `key`: a folder, or
    /// a link, which is never followed.
    pub fn not_a_document(&self, key: &str) -> bool {
        self.files.not_a_document(key)
    }
}

/// What a body is told when it commits with no operation to commit into.
pub const NO_OPERATION: &str = "There is nowhere for this to go: work reaches what stands only \
                                through an operation, and you are on no step of one.";

impl Shelf for Published {
    fn read(&self, key: &str) -> Option<String> {
        self.files.read(key)
    }

    fn size(&self, key: &str) -> Option<u64> {
        self.files.size(key)
    }

    /// A folder that is a link lists as nothing: a junction would otherwise
    /// report documents from somewhere the character never named.
    fn list(&self, dir: &str) -> BTreeSet<String> {
        self.files.list(dir)
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
        let shelf = Published::folder(dir.path().to_path_buf());
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
