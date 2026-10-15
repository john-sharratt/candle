//! Editing the mind's static content.
//!
//! The mind directory is the authored half of this product: canon under
//! `layers/`, the craft libraries in `responses/` and `moods/`, the characters
//! in `personalities/`, and the settings in `worlds/`. All of it is text a
//! person wrote, none of it is generated, and until now the only way to change
//! any of it was a text editor on the machine the daemon runs on.
//!
//! This module is the other way: browse the tree, open a document, save it,
//! add one, remove one — through the API, with the same role rules as every
//! other authored document (read signed-in, **write admin**), and with the
//! world's own filters applied on the way down.
//!
//! # Nothing outside this module knows there are files
//!
//! The API and the console speak [`Address`]es — `canon/ammo/bolt` — and
//! [`address`] is the only place that knows what one is on disk. A path, an
//! extension and a directory are implementation, and publishing them as a
//! contract would promise that canon lives under `layers/` and that prose is
//! markdown, forever.
//!
//! | File | Concern |
//! |---|---|
//! | [`address`] | what the corpus is made of, and where each part is stored |
//! | [`catalog`] | what is inside a place in the corpus |
//! | [`scope`] | what a world admits, from its `selects` / `excludes` / `personalities` |
//! | [`doc`] | reading, writing and removing text, atomically |
//! | [`section`] | a document as fields, so it is edited without knowing YAML |
//! | [`parts`] | one item inside a document, addressed on its own |
//! | [`path`] | a relative path that provably stays inside the mind — internal |
//!
//! # What this is not
//!
//! It is not the registry. [`crate::registry`] owns `worlds/` and
//! `personalities/` as *documents* — parsed, validated, patched key-by-key so
//! an author's comments survive a save. That is the right tool when the console
//! is editing a known field of a known shape.
//!
//! This is the tool for everything else: a markdown page of canon has no
//! schema to patch, and there are 1,818 of them. So a document here is text in
//! and text out, and the two coexist deliberately — a world's `name` is edited
//! through the registry and keeps its comments, while `layers/world/ammo.md` is
//! edited as the prose it is.

pub mod address;
pub mod catalog;
pub mod doc;
pub mod files;
pub mod parts;
pub mod path;
pub mod scope;
pub mod section;

use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::sim::shelf::Published;
use files::MindFiles;

pub use address::Address;
pub use path::MindPath;
pub use scope::Scope;

/// The mind directory, if this daemon has one.
///
/// `--mind` is optional: a daemon started without it serves the console and the
/// substrate and has no authored content at all. Every method here reports that
/// as a plain absence rather than an error, so the console can say "no mind
/// directory" once instead of the API failing five different ways.
///
/// It holds the mind's files as they stand ([`MindFiles`]): at the commit the
/// log names for the mind's branch, or in a plain folder. Every clone shares
/// one holder, so when the branch moves, [`Mind::replace`] moves every reader
/// to the new commit at once; a snapshot taken before keeps reading the old.
#[derive(Debug, Clone, Default)]
pub struct Mind {
    /// `None` for a daemon with no mind. Within, `None` while a mind read from
    /// the substrate has not arrived yet.
    current: Option<Arc<RwLock<Option<Arc<MindFiles>>>>>,
    /// The repository behind the files: the plain folder, or the mind's
    /// object store — where operations' stores and publishes work. Never read
    /// as the mind's files: a store checks nothing out.
    repo: Option<PathBuf>,
}

/// Two handles are the same mind when they share one holder.
impl PartialEq for Mind {
    fn eq(&self, other: &Self) -> bool {
        match (&self.current, &other.current) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        }
    }
}

impl Eq for Mind {}

impl Mind {
    /// The mind in the folder `root`, read as it stands. `None` when `--mind`
    /// was not given.
    pub fn new(root: Option<PathBuf>) -> Self {
        match root {
            Some(root) => Self::of(Arc::new(MindFiles::folder(&root)), root),
            None => Self::default(),
        }
    }

    /// The mind as `files` hold it, from the repository at `repo`.
    pub fn of(files: Arc<MindFiles>, repo: impl Into<PathBuf>) -> Self {
        Mind {
            current: Some(Arc::new(RwLock::new(Some(files)))),
            repo: Some(repo.into()),
        }
    }

    /// A mind read from the repository at `repo` that has not arrived yet:
    /// no files until [`Self::replace`] gives it some.
    pub fn pending(repo: impl Into<PathBuf>) -> Self {
        Mind {
            current: Some(Arc::new(RwLock::new(None))),
            repo: Some(repo.into()),
        }
    }

    /// The folder of the repository behind the files: what operations' stores
    /// read through and publishes land in. `None` with no mind.
    pub fn repo_dir(&self) -> Option<&std::path::Path> {
        self.repo.as_deref()
    }

    /// The mind's files as they stand now, or `None` when there is no mind —
    /// or it has not arrived.
    pub fn files(&self) -> Option<Arc<MindFiles>> {
        self.current.as_ref()?.read().unwrap().clone()
    }

    /// The published mind as it stands now, as a shelf.
    pub fn published(&self) -> Option<Published> {
        self.files().map(Published::of)
    }

    /// Whether this daemon has a mind it can read now.
    pub fn is_present(&self) -> bool {
        self.files().is_some()
    }

    /// Move every reader to `files`: the mind's branch moved, or the mind
    /// arrived. A daemon with no mind keeps none.
    pub fn replace(&self, files: Arc<MindFiles>) {
        if let Some(current) = &self.current {
            *current.write().unwrap() = Some(files);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_daemon_without_a_mind_says_so_rather_than_failing() {
        assert!(Mind::new(None).files().is_none());
    }

    #[test]
    fn a_daemon_with_a_mind_reports_its_root() {
        let dir = std::env::temp_dir();
        let files = Mind::new(Some(dir.clone())).files().unwrap();
        assert_eq!(files.folder_root(), Some(dir.as_path()));
    }
}
