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
pub mod parts;
pub mod path;
pub mod scope;
pub mod section;

use std::path::{Path, PathBuf};

pub use address::Address;
pub use path::MindPath;
pub use scope::Scope;

/// The mind directory, if this daemon has one.
///
/// `--mind` is optional: a daemon started without it serves the console and the
/// substrate and has no authored content at all. Every method here reports that
/// as a plain absence rather than an error, so the console can say "no mind
/// directory" once instead of the API failing five different ways.
#[derive(Debug, Clone)]
pub struct Mind {
    root: Option<PathBuf>,
}

impl Mind {
    /// Wrap the resolved mind directory. `None` when `--mind` was not given.
    pub fn new(root: Option<PathBuf>) -> Self {
        Mind { root }
    }

    /// The root, or `None` when there is no mind.
    pub fn root(&self) -> Option<&Path> {
        self.root.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_daemon_without_a_mind_says_so_rather_than_failing() {
        assert!(Mind::new(None).root().is_none());
    }

    #[test]
    fn a_daemon_with_a_mind_reports_its_root() {
        let dir = std::env::temp_dir();
        assert_eq!(Mind::new(Some(dir.clone())).root(), Some(dir.as_path()));
    }
}
