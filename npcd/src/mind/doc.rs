//! Reading, writing and removing one document of the mind, as a [`Shelf`]
//! holds it: the published mind, or an author's view with their drafts over it
//! (`docs/replicated_substrate.md` §10.3). Where a write lands — an author's
//! store, or the folder itself — is the shelf's.
//!
//! # Text, and only text
//!
//! Both formats here are text an author reads. A document that is not valid
//! UTF-8 is refused rather than replaced with lossy characters, because
//! `from_utf8_lossy` on a save would silently rewrite bytes the author never
//! typed — and it would do it to the one file that was already unusual enough
//! to be worth keeping intact.

use super::path::{MindPath, PathError};
use crate::sim::shelf::Shelf;

/// Largest document this editor will write.
///
/// The biggest file in a real mind is a 14 KB canon page, so this is two orders
/// of magnitude of headroom. It exists because the body arrives from a browser:
/// without a bound, one request could fill the disk the substrate is writing
/// its redo log to.
pub const MAX_BYTES: usize = 512 * 1024;

/// What went wrong.
#[derive(Debug)]
pub enum DocError {
    Path(PathError),
    /// No such document. Distinct from a folder: a caller asking to edit a
    /// folder has made a different mistake than one naming a file that is gone.
    NotFound,
    IsADirectory,
    /// The file is there and is not UTF-8, so there is no text to show.
    NotText,
    TooLarge(usize),
    /// A create was asked for and the document is already there.
    Exists,
    /// A part of a document could not be spliced back into it without
    /// rewriting the document, which would lose its comments. See
    /// [`crate::mind::parts`].
    CannotPatch,
    /// The write was not kept, for this reason — which can name the disk, so
    /// it is the operator's to read and not the caller's.
    Unkept(String),
}

impl std::fmt::Display for DocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DocError::Path(e) => write!(f, "{e}"),
            DocError::NotFound => write!(f, "no such document in the mind"),
            DocError::IsADirectory => write!(f, "that path is a folder, not a document"),
            DocError::NotText => write!(f, "that file is not UTF-8 text"),
            DocError::TooLarge(n) => {
                write!(f, "{n} bytes exceeds the {MAX_BYTES} byte limit")
            }
            DocError::Exists => write!(f, "a document already exists at that path"),
            DocError::CannotPatch => write!(
                f,
                "this could not be edited without rewriting the document it is part of, \
                 which would lose its comments — edit that document as text instead"
            ),
            DocError::Unkept(why) => write!(f, "{why}"),
        }
    }
}

/// A document's text and what is worth knowing about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Doc {
    pub path: String,
    pub text: String,
    pub bytes: u64,
    pub ext: String,
}

/// Why there is no document at `key` on `shelf`: a folder is there, a file
/// that is not text is, or nothing is.
fn absent(shelf: &dyn Shelf, key: &str) -> DocError {
    if shelf.size(key).is_some() {
        DocError::NotText
    } else if !shelf.list(key).is_empty() {
        DocError::IsADirectory
    } else {
        DocError::NotFound
    }
}

/// Read a document.
pub fn read(shelf: &dyn Shelf, path: &MindPath) -> Result<Doc, DocError> {
    path.check_editable().map_err(DocError::Path)?;
    let key = path.as_str();
    let text = shelf.read(&key).ok_or_else(|| absent(shelf, &key))?;
    Ok(Doc {
        bytes: text.len() as u64,
        ext: path.ext().unwrap_or_default(),
        path: key,
        text,
    })
}

/// What a write did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wrote {
    Created,
    Updated,
}

/// Write a document, creating it where there is none.
///
/// `must_be_new` is how "add an item" differs from "save an edit": a create
/// that silently overwrote a file somebody else had just added would lose it
/// with no error to say so.
pub fn write(
    shelf: &dyn Shelf,
    path: &MindPath,
    text: &str,
    must_be_new: bool,
) -> Result<Wrote, DocError> {
    path.check_editable().map_err(DocError::Path)?;
    if text.len() > MAX_BYTES {
        return Err(DocError::TooLarge(text.len()));
    }
    let key = path.as_str();
    if !shelf.list(&key).is_empty() {
        return Err(DocError::IsADirectory);
    }
    let existed = shelf.size(&key).is_some();
    if existed && must_be_new {
        return Err(DocError::Exists);
    }
    shelf
        .commit(&[(key, Some(text.to_string()))])
        .map_err(DocError::Unkept)?;
    Ok(if existed {
        Wrote::Updated
    } else {
        Wrote::Created
    })
}

/// Remove a document.
///
/// Documents only. Removing a folder would be a recursive delete behind a
/// single click, and the one thing this editor must never do is take more than
/// it was asked for — a folder is emptied one document at a time, visibly.
pub fn remove(shelf: &dyn Shelf, path: &MindPath) -> Result<(), DocError> {
    path.check_editable().map_err(DocError::Path)?;
    let key = path.as_str();
    if shelf.size(&key).is_none() {
        return Err(absent(shelf, &key));
    }
    shelf.commit(&[(key, None)]).map_err(DocError::Unkept)
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;
    use crate::drafts::disk::Disk;
    use crate::sim::shelf::Published;

    fn p(s: &str) -> MindPath {
        MindPath::parse(s).expect("test path parses")
    }

    #[test]
    fn a_document_round_trips_through_the_shelf() {
        let root = tempfile::tempdir().unwrap();
        let disk = Disk::new(root.path());
        let path = p("layers/a.md");

        assert_eq!(write(&disk, &path, "hello", false).unwrap(), Wrote::Created);
        let doc = read(&disk, &path).unwrap();
        assert_eq!(doc.text, "hello");
        assert_eq!(doc.path, "layers/a.md");
        assert_eq!(doc.ext, "md");
        assert_eq!(doc.bytes, 5);
        assert_eq!(
            fs::read_to_string(root.path().join("layers/a.md")).unwrap(),
            "hello"
        );

        assert_eq!(
            write(&disk, &path, "second", false).unwrap(),
            Wrote::Updated
        );
        assert_eq!(read(&disk, &path).unwrap().text, "second");
    }

    /// Unicode survives a round trip unchanged — the mind has a `protégés.md`
    /// and em-dashes throughout, and a save that mangled them would be worse
    /// than one that failed.
    #[test]
    fn unicode_content_and_names_survive_a_save() {
        let root = tempfile::tempdir().unwrap();
        let disk = Disk::new(root.path());
        let path = p("layers/protégés.md");
        let text = "Les protégés — “quoted”, ≤ 5, ✓\nsecond line\n";
        write(&disk, &path, text, true).unwrap();
        assert_eq!(read(&disk, &path).unwrap().text, text);
        assert_eq!(
            fs::read_to_string(root.path().join("layers/protégés.md")).unwrap(),
            text
        );
    }

    #[test]
    fn a_create_refuses_to_overwrite() {
        let root = tempfile::tempdir().unwrap();
        let disk = Disk::new(root.path());
        let path = p("a.md");
        write(&disk, &path, "first", true).unwrap();
        assert!(matches!(
            write(&disk, &path, "second", true),
            Err(DocError::Exists)
        ));
        assert_eq!(read(&disk, &path).unwrap().text, "first");
    }

    #[test]
    fn a_document_can_be_removed_and_then_is_gone() {
        let root = tempfile::tempdir().unwrap();
        let disk = Disk::new(root.path());
        let path = p("a.md");
        write(&disk, &path, "x", true).unwrap();
        remove(&disk, &path).unwrap();
        assert!(matches!(read(&disk, &path), Err(DocError::NotFound)));
        assert!(matches!(remove(&disk, &path), Err(DocError::NotFound)));
    }

    /// A folder is never removed or written over by this editor: one click
    /// must not become a recursive delete.
    #[test]
    fn a_folder_is_never_removed_or_written_over() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir_all(root.path().join("layers.md")).unwrap();
        fs::write(root.path().join("layers.md/inside.md"), "x").unwrap();
        let disk = Disk::new(root.path());
        assert!(matches!(
            remove(&disk, &p("layers.md")),
            Err(DocError::IsADirectory)
        ));
        assert!(matches!(
            write(&disk, &p("layers.md"), "x", false),
            Err(DocError::IsADirectory)
        ));
        assert!(
            root.path().join("layers.md").is_dir(),
            "the folder survived"
        );
    }

    #[test]
    fn only_text_documents_are_reachable() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("a.exe"), "x").unwrap();
        let disk = Disk::new(root.path());
        let path = p("a.exe");
        assert!(matches!(read(&disk, &path), Err(DocError::Path(_))));
        assert!(matches!(
            write(&disk, &path, "y", false),
            Err(DocError::Path(_))
        ));
        assert!(matches!(remove(&disk, &path), Err(DocError::Path(_))));
        assert_eq!(fs::read_to_string(root.path().join("a.exe")).unwrap(), "x");
    }

    #[test]
    fn a_body_over_the_limit_is_refused_and_writes_nothing() {
        let root = tempfile::tempdir().unwrap();
        let disk = Disk::new(root.path());
        let huge = "x".repeat(MAX_BYTES + 1);
        assert!(matches!(
            write(&disk, &p("a.md"), &huge, true),
            Err(DocError::TooLarge(_))
        ));
        assert!(!root.path().join("a.md").exists());
    }

    #[test]
    fn a_file_that_is_not_utf8_is_refused_rather_than_mangled() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("a.md"), [0xff, 0xfe, 0x00, 0x01]).unwrap();
        let disk = Disk::new(root.path());
        assert!(matches!(read(&disk, &p("a.md")), Err(DocError::NotText)));
    }

    /// The published mind takes no write: an author's goes through their own
    /// view.
    #[test]
    fn the_published_mind_takes_no_write() {
        let root = tempfile::tempdir().unwrap();
        let published = Published::folder(root.path().to_path_buf());
        assert!(matches!(
            write(&published, &p("a.md"), "x", true),
            Err(DocError::Unkept(_))
        ));
        assert!(!root.path().join("a.md").exists());
    }
}
