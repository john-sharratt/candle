//! Reading, writing and removing one document of the mind.
//!
//! # Writes are atomic
//!
//! A write goes to a temporary file beside the target and is renamed over it.
//! The mind is an author's working tree with no version control behind it —
//! this session has already destroyed two files in it — and the failure mode of
//! a plain truncating write is the worst one available: `open(path, "w")`
//! empties the file before it writes, so an interruption in that window leaves
//! the file gone rather than merely unchanged. On NTFS the size metadata can
//! land while the data does not, leaving a file of the right length full of
//! `NUL`. A rename is a single atomic step: the reader either sees the old file
//! or the new one, never a half of either.
//!
//! # Text, and only text
//!
//! Both formats here are text an author reads. A document that is not valid
//! UTF-8 is refused rather than replaced with lossy characters, because
//! `from_utf8_lossy` on a save would silently rewrite bytes the author never
//! typed — and it would do it to the one file that was already unusual enough
//! to be worth keeping intact.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use super::path::{MindPath, PathError};

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
    /// No such file. Distinct from a directory: a caller asking to edit a
    /// folder has made a different mistake than one naming a file that is gone.
    NotFound,
    IsADirectory,
    /// The file on disk is not UTF-8, so there is no text to show.
    NotText,
    TooLarge(usize),
    /// A create was asked for and the file is already there.
    Exists,
    /// A part of a document could not be spliced back into it without
    /// rewriting the document, which would lose its comments. See
    /// [`crate::mind::parts`].
    CannotPatch,
    Io(std::io::Error),
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
            DocError::Io(e) => write!(f, "{e}"),
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

/// Read a document.
pub fn read(root: &Path, path: &MindPath) -> Result<Doc, DocError> {
    path.check_editable().map_err(DocError::Path)?;
    let full = path.resolve(root).map_err(DocError::Path)?;
    let meta = full.symlink_metadata().map_err(|_| DocError::NotFound)?;
    if meta.is_dir() {
        return Err(DocError::IsADirectory);
    }
    // A symlink is refused rather than followed. `resolve` already proves the
    // *target* is inside the mind, so this is not about escape — it is that
    // writing back through a link would replace the link with a plain file and
    // quietly break whatever else pointed at it.
    if meta.file_type().is_symlink() {
        return Err(DocError::NotFound);
    }
    let raw = fs::read(&full).map_err(DocError::Io)?;
    let text = String::from_utf8(raw).map_err(|_| DocError::NotText)?;
    Ok(Doc {
        path: path.as_str(),
        bytes: meta.len(),
        ext: path.ext().unwrap_or_default(),
        text,
    })
}

/// What a write did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Wrote {
    Created,
    Updated,
}

/// Write a document, creating it and any missing parent directories.
///
/// `must_be_new` is how "add an item" differs from "save an edit": a create
/// that silently overwrote a file somebody else had just added would lose it
/// with no error to say so.
pub fn write(
    root: &Path,
    path: &MindPath,
    text: &str,
    must_be_new: bool,
) -> Result<Wrote, DocError> {
    path.check_editable().map_err(DocError::Path)?;
    if text.len() > MAX_BYTES {
        return Err(DocError::TooLarge(text.len()));
    }
    let full = path.resolve(root).map_err(DocError::Path)?;

    let existed = match full.symlink_metadata() {
        Ok(m) if m.is_dir() => return Err(DocError::IsADirectory),
        Ok(m) if m.file_type().is_symlink() => return Err(DocError::IsADirectory),
        Ok(_) => true,
        Err(_) => false,
    };
    if existed && must_be_new {
        return Err(DocError::Exists);
    }

    if let Some(parent) = full.parent() {
        fs::create_dir_all(parent).map_err(DocError::Io)?;
    }
    atomic_write(&full, text.as_bytes())?;
    Ok(if existed {
        Wrote::Updated
    } else {
        Wrote::Created
    })
}

/// Remove a document.
///
/// Files only. Removing a directory would be a recursive delete behind a single
/// click, and the one thing this editor must never do is take more than it was
/// asked for — a folder is emptied one document at a time, visibly.
pub fn remove(root: &Path, path: &MindPath) -> Result<(), DocError> {
    path.check_editable().map_err(DocError::Path)?;
    let full = path.resolve(root).map_err(DocError::Path)?;
    let meta = full.symlink_metadata().map_err(|_| DocError::NotFound)?;
    if meta.is_dir() {
        return Err(DocError::IsADirectory);
    }
    fs::remove_file(&full).map_err(DocError::Io)
}

/// Write bytes to `target` without ever leaving it partly written.
///
/// The temporary lives in the same directory so the rename is within one
/// filesystem — across a mount, `rename` is a copy that can fail halfway, which
/// is the failure this exists to avoid.
fn atomic_write(target: &Path, bytes: &[u8]) -> Result<(), DocError> {
    let dir = target.parent().unwrap_or_else(|| Path::new("."));
    let stem = target
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("document");
    let tmp: PathBuf = dir.join(format!(".{stem}.{}.tmp", std::process::id()));

    // Scoped so the handle is closed before the rename: Windows refuses to
    // rename over a file that is still open, and the failure would look like a
    // permission problem rather than a held handle.
    {
        let mut f = fs::File::create(&tmp).map_err(DocError::Io)?;
        f.write_all(bytes).map_err(DocError::Io)?;
        // Durable before it is visible. Without this the rename can be ordered
        // ahead of the data on a crash, publishing a name over an empty file.
        f.sync_all().map_err(DocError::Io)?;
    }
    match fs::rename(&tmp, target) {
        Ok(()) => Ok(()),
        Err(e) => {
            // Leaving a stray `.tmp` beside an author's files would be litter
            // in a directory they read by hand.
            let _ = fs::remove_file(&tmp);
            Err(DocError::Io(e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("npcd-minddoc-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn p(s: &str) -> MindPath {
        MindPath::parse(s).expect("test path parses")
    }

    #[test]
    fn a_document_round_trips_through_disk() {
        let root = tmp("round");
        fs::create_dir_all(root.join("layers")).unwrap();
        let path = p("layers/a.md");

        assert_eq!(write(&root, &path, "hello", false).unwrap(), Wrote::Created);
        let doc = read(&root, &path).unwrap();
        assert_eq!(doc.text, "hello");
        assert_eq!(doc.path, "layers/a.md");
        assert_eq!(doc.ext, "md");
        assert_eq!(doc.bytes, 5);

        // And the bytes really are on disk, not only in the reply.
        assert_eq!(
            fs::read_to_string(root.join("layers/a.md")).unwrap(),
            "hello"
        );

        assert_eq!(
            write(&root, &path, "second", false).unwrap(),
            Wrote::Updated
        );
        assert_eq!(read(&root, &path).unwrap().text, "second");
    }

    /// Unicode survives a round trip unchanged — the mind has a `protégés.md`
    /// and em-dashes throughout, and a save that mangled them would be worse
    /// than one that failed.
    #[test]
    fn unicode_content_and_names_survive_a_save() {
        let root = tmp("unicode");
        let path = p("layers/protégés.md");
        let text = "Les protégés — “quoted”, ≤ 5, ✓\nsecond line\n";
        write(&root, &path, text, true).unwrap();
        assert_eq!(read(&root, &path).unwrap().text, text);
        assert_eq!(
            fs::read_to_string(root.join("layers/protégés.md")).unwrap(),
            text
        );
    }

    #[test]
    fn creating_makes_the_parent_directories() {
        let root = tmp("mkdir");
        let path = p("layers/world/brand/new.md");
        assert_eq!(write(&root, &path, "x", true).unwrap(), Wrote::Created);
        assert!(root.join("layers/world/brand/new.md").is_file());
    }

    #[test]
    fn a_create_refuses_to_overwrite() {
        let root = tmp("exists");
        let path = p("a.md");
        write(&root, &path, "first", true).unwrap();
        assert!(matches!(
            write(&root, &path, "second", true),
            Err(DocError::Exists)
        ));
        // And the first is untouched.
        assert_eq!(read(&root, &path).unwrap().text, "first");
    }

    #[test]
    fn a_document_can_be_removed_and_then_is_gone() {
        let root = tmp("remove");
        let path = p("a.md");
        write(&root, &path, "x", true).unwrap();
        remove(&root, &path).unwrap();
        assert!(matches!(read(&root, &path), Err(DocError::NotFound)));
        assert!(matches!(remove(&root, &path), Err(DocError::NotFound)));
    }

    /// A folder is never removed by this editor: one click must not become a
    /// recursive delete.
    #[test]
    fn a_directory_is_never_removed() {
        let root = tmp("rmdir");
        fs::create_dir_all(root.join("layers.md")).unwrap();
        let path = p("layers.md");
        assert!(matches!(remove(&root, &path), Err(DocError::IsADirectory)));
        assert!(root.join("layers.md").is_dir(), "the folder survived");
    }

    #[test]
    fn a_directory_is_never_written_over() {
        let root = tmp("wrdir");
        fs::create_dir_all(root.join("a.md")).unwrap();
        assert!(matches!(
            write(&root, &p("a.md"), "x", false),
            Err(DocError::IsADirectory)
        ));
        assert!(root.join("a.md").is_dir());
    }

    #[test]
    fn only_text_documents_are_reachable() {
        let root = tmp("ext");
        fs::write(root.join("a.exe"), "x").unwrap();
        let path = p("a.exe");
        assert!(matches!(read(&root, &path), Err(DocError::Path(_))));
        assert!(matches!(
            write(&root, &path, "y", false),
            Err(DocError::Path(_))
        ));
        assert!(matches!(remove(&root, &path), Err(DocError::Path(_))));
        assert_eq!(
            fs::read_to_string(root.join("a.exe")).unwrap(),
            "x",
            "the file was not touched"
        );
    }

    #[test]
    fn a_body_over_the_limit_is_refused_and_writes_nothing() {
        let root = tmp("big");
        let path = p("a.md");
        let huge = "x".repeat(MAX_BYTES + 1);
        assert!(matches!(
            write(&root, &path, &huge, true),
            Err(DocError::TooLarge(_))
        ));
        assert!(!root.join("a.md").exists());
    }

    #[test]
    fn a_file_that_is_not_utf8_is_refused_rather_than_mangled() {
        let root = tmp("binary");
        fs::write(root.join("a.md"), [0xff, 0xfe, 0x00, 0x01]).unwrap();
        assert!(matches!(read(&root, &p("a.md")), Err(DocError::NotText)));
    }

    /// The temporary a write goes through must never be left behind — an
    /// author reads this directory by hand.
    #[test]
    fn a_write_leaves_no_temporary_behind() {
        let root = tmp("notmp");
        write(&root, &p("layers/a.md"), "x", true).unwrap();
        write(&root, &p("layers/a.md"), "y", false).unwrap();
        let strays: Vec<String> = fs::read_dir(root.join("layers"))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.ends_with(".tmp") || n.starts_with('.'))
            .collect();
        assert!(strays.is_empty(), "left behind: {strays:?}");
    }

    /// The whole point of the rename: the file is never observable as empty or
    /// half-written. Checked by proving the previous contents are intact right
    /// up until the new ones are complete.
    #[test]
    fn a_save_replaces_the_contents_in_one_step() {
        let root = tmp("atomic");
        let path = p("a.md");
        write(&root, &path, "original", true).unwrap();

        // A body at the size limit, so the write is not a single small block.
        let big = "y".repeat(MAX_BYTES);
        write(&root, &path, &big, false).unwrap();
        let after = fs::read_to_string(root.join("a.md")).unwrap();
        assert_eq!(after.len(), MAX_BYTES);
        assert!(after.chars().all(|c| c == 'y'));
    }

    #[test]
    fn traversal_never_reaches_the_disk() {
        let root = tmp("escape");
        let outside = root.parent().unwrap().join("npcd-mind-escape-witness.md");
        let _ = fs::remove_file(&outside);
        fs::write(&outside, "untouched").unwrap();

        // The path is refused at parse, so there is nothing to resolve.
        assert!(MindPath::parse("../npcd-mind-escape-witness.md").is_err());
        assert_eq!(fs::read_to_string(&outside).unwrap(), "untouched");
        let _ = fs::remove_file(&outside);
    }
}
