//! A store's lower layer as one operation sees it.
//!
//! A store operation takes one [`View`] at its start and asks it everything:
//! a branch is resolved to its tree once, so a listing, a search or a read
//! that looks at many paths sees them all from the same commit even while the
//! branch moves.

use std::io::Cursor;
use std::path::Path;
use std::sync::Arc;

use super::git_source::GitSource;
use super::tree::Tree;
use super::{folder, PageResult, VfsError, VfsStore, MAX_LOWER_FILE_BYTES};
use crate::GitError;

/// The lower layer at one moment.
pub(super) enum View<'s> {
    /// No lower layer: the session's changes are all there is.
    Empty,
    /// A folder on disk, read as it stands.
    Folder(&'s Path),
    /// A branch, as the tree it held when the view was taken.
    Branch {
        source: &'s GitSource,
        tree: Arc<Tree>,
    },
}

impl View<'_> {
    /// Whether `norm` is a file.
    pub(super) fn is_file(&self, norm: &str) -> bool {
        match self {
            View::Empty => false,
            View::Folder(root) => folder::is_file(root, norm),
            View::Branch { tree, .. } => !norm.is_empty() && tree.file(norm).is_some(),
        }
    }

    /// Whether `norm` is the same file in this view and `other` — the same
    /// blob, or absent from both. Only two branch views can say so; any
    /// other pair never vouches for it.
    pub(super) fn same_file(&self, other: &View<'_>, norm: &str) -> bool {
        match (self, other) {
            (View::Branch { tree: a, .. }, View::Branch { tree: b, .. }) => {
                a.file(norm).map(|(blob, _)| blob) == b.file(norm).map(|(blob, _)| blob)
            }
            _ => false,
        }
    }

    /// Whether `norm` is a folder; the root is one wherever there is a lower
    /// layer.
    pub(super) fn is_dir(&self, norm: &str) -> bool {
        match self {
            View::Empty => false,
            View::Folder(root) => folder::is_dir(root, norm),
            View::Branch { tree, .. } => tree.is_dir(norm),
        }
    }

    /// The file at `norm` as text; `None` when there is none. Refused above
    /// [`MAX_LOWER_FILE_BYTES`] and when it is not UTF-8.
    pub(super) fn read_text(&self, norm: &str) -> Result<Option<String>, VfsError> {
        match self {
            View::Empty => Ok(None),
            View::Folder(root) => folder::read_text(root, norm),
            View::Branch { source, tree } => {
                let Some((blob, size)) = tree.file(norm) else {
                    return Ok(None);
                };
                if size > MAX_LOWER_FILE_BYTES {
                    return Err(VfsStore::too_large(norm, size));
                }
                let bytes = source.blob(blob).map_err(|e| unreadable(norm, e))?;
                String::from_utf8(bytes)
                    .map(Some)
                    .map_err(|_| VfsError::Unreadable(format!("{norm} is not valid UTF-8 text")))
            }
        }
    }

    /// The file at `norm`, byte for byte and at any size; `None` when there
    /// is none.
    pub(super) fn read_bytes(&self, norm: &str) -> Result<Option<Vec<u8>>, VfsError> {
        match self {
            View::Empty => Ok(None),
            View::Folder(root) => folder::read_bytes(root, norm),
            View::Branch { source, tree } => match tree.file(norm) {
                Some((blob, _)) => source.blob(blob).map(Some).map_err(|e| unreadable(norm, e)),
                None => Ok(None),
            },
        }
    }

    /// One page of the file at `norm`; `None` when there is none.
    pub(super) fn read_page(&self, norm: &str, page: u32) -> Result<Option<PageResult>, VfsError> {
        match self {
            View::Empty => Ok(None),
            View::Folder(root) => folder::read_page(root, norm, page),
            View::Branch { .. } => {
                let Some(text) = self.read_text(norm)? else {
                    return Ok(None);
                };
                VfsStore::paginate(Cursor::new(text.as_bytes()), page)
                    .map(Some)
                    .map_err(|e| VfsError::Unreadable(format!("{norm} could not be read: {e}")))
            }
        }
    }

    /// The files and folders directly inside `norm`, as `(path, bytes,
    /// is_dir)`.
    pub(super) fn children(&self, norm: &str) -> Vec<(String, Option<usize>, bool)> {
        match self {
            View::Empty => Vec::new(),
            View::Folder(root) => folder::children(root, norm),
            View::Branch { tree, .. } => tree.children(norm),
        }
    }

    /// Every file under `prefix`.
    pub(super) fn files_under(&self, prefix: &str) -> Vec<String> {
        match self {
            View::Empty => Vec::new(),
            View::Folder(root) => folder::files_under(root, prefix),
            View::Branch { tree, .. } => tree.files_under(prefix),
        }
    }
}

fn unreadable(norm: &str, e: GitError) -> VfsError {
    VfsError::Unreadable(format!("{norm} could not be read from git: {e}"))
}
