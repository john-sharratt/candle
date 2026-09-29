//! Which of a tree's files an ingest layer reads — decided from the listing
//! alone, before any file is opened (`docs/zend_branch_ingest.md` §5).

use crate::repo_scan::types::Language;

/// The largest file an ingest layer reads. Big enough for generated parsers,
/// vendored single-file libraries and long design documents; an accidentally
/// committed binary is typically far larger.
pub const MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// Where one ingest layer reads: a folder of the workspace (`--ingest-dir`)
/// and a depth bound (`--max-depth`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct IngestScope {
    /// Workspace-relative, no leading or trailing `/`: empty for every
    /// repository, a repository's name for all of it, or a folder inside
    /// one (`candle/zend/src`).
    folder: String,
    /// Path components below the walk's start — the repository's root, or
    /// `folder` — so `1` is the start's own files. `None` is unbounded.
    max_depth: Option<usize>,
}

impl IngestScope {
    pub fn new(folder: &str, max_depth: Option<usize>) -> Self {
        let folder = folder.trim_matches('/');
        Self {
            folder: if folder == "." { "" } else { folder }.to_string(),
            max_depth,
        }
    }

    /// The same scope at full depth — what a layer's held units are judged
    /// against. A unit past the depth bound is not read, but it is kept while
    /// a branch still holds it, so narrowing the bound drops nothing already
    /// ingested; and since a unit's key does not depend on the bound, the
    /// units found at full depth are the ones the layer holds.
    pub fn full_depth(&self) -> Self {
        Self {
            folder: self.folder.clone(),
            max_depth: None,
        }
    }

    /// Whether the scope reaches into `repo` at all.
    pub fn reaches(&self, repo: &str) -> bool {
        self.folder.is_empty() || self.folder.split('/').next() == Some(repo)
    }

    /// The language `repo`'s file at `path` (repository-relative), of `size`
    /// bytes, is read as — `None` when the layer does not read it: outside
    /// the scope or past the depth bound, hidden (any component starting
    /// with `.`), off the extension allowlist, or over [`MAX_FILE_BYTES`].
    pub fn admits(&self, repo: &str, path: &str, size: u64) -> Option<Language> {
        if size > MAX_FILE_BYTES || path.split('/').any(|c| c.starts_with('.')) {
            return None;
        }
        let key = format!("{repo}/{path}");
        let below = if self.folder.is_empty() {
            path
        } else {
            key.strip_prefix(&self.folder)?.strip_prefix('/')?
        };
        if self.max_depth.is_some_and(|d| below.split('/').count() > d) {
            return None;
        }
        language_of(path)
    }
}

/// The language a path's name says it is in, if it is one read at all.
pub fn language_of(path: &str) -> Option<Language> {
    let basename = path.rsplit('/').next().unwrap_or(path);
    if basename == "go.mod" || basename == "go.sum" {
        return Some(Language::Go);
    }
    let (stem, ext) = basename.rsplit_once('.')?;
    if stem.is_empty() {
        return None;
    }
    Language::from_extension(&ext.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unbounded_scope_reads_every_allowlisted_file() {
        let s = IngestScope::new("", None);
        assert_eq!(s.admits("r", "src/deep/x.rs", 10), Some(Language::Rust));
        assert_eq!(s.admits("r", "README.md", 10), Some(Language::Markdown));
        assert_eq!(s.admits("r", "go.mod", 10), Some(Language::Go));
        assert_eq!(s.admits("r", "k/decode.CU", 10), Some(Language::Cpp));
        assert!(s.reaches("r") && s.reaches("other"));
    }

    #[test]
    fn hidden_unlisted_and_oversize_files_are_not_read() {
        let s = IngestScope::new("", None);
        assert_eq!(s.admits("r", ".github/ci.yml", 10), None);
        assert_eq!(s.admits("r", "src/.cache/x.rs", 10), None);
        assert_eq!(s.admits("r", "LICENSE", 10), None);
        assert_eq!(s.admits("r", "shape.svg", 10), None);
        assert_eq!(
            s.admits("r", "src/.rs", 10),
            None,
            "a dotfile, not an extension"
        );
        assert_eq!(s.admits("r", "big.rs", MAX_FILE_BYTES + 1), None);
        assert_eq!(
            s.admits("r", "big.rs", MAX_FILE_BYTES),
            Some(Language::Rust)
        );
    }

    /// `1` is a repository's own files; `2` adds one folder down.
    #[test]
    fn depth_counts_from_the_repository_root() {
        let s = IngestScope::new("", Some(2));
        assert!(s.admits("r", "a.rs", 1).is_some());
        assert!(s.admits("r", "src/b.rs", 1).is_some());
        assert!(s.admits("r", "src/deep/c.rs", 1).is_none());
        assert!(IngestScope::new("", Some(1))
            .admits("r", "src/b.rs", 1)
            .is_none());
    }

    /// Full depth drops the bound and keeps the folder.
    #[test]
    fn full_depth_keeps_the_folder_and_drops_the_bound() {
        let s = IngestScope::new("alpha/src", Some(1));
        assert_eq!(s.full_depth(), IngestScope::new("alpha/src", None));
        assert!(s.full_depth().admits("alpha", "src/deep/x.rs", 1).is_some());
        assert!(s.full_depth().admits("alpha", "docs/a.md", 1).is_none());
    }

    /// A scope folder narrows the walk to one folder of one repository, and
    /// the depth bound counts from that folder.
    #[test]
    fn a_scope_folder_narrows_and_rebases_the_depth() {
        let s = IngestScope::new("/alpha/src/", Some(1));
        assert!(s.reaches("alpha") && !s.reaches("beta"));
        assert!(s.admits("alpha", "src/lib.rs", 1).is_some());
        assert!(
            s.admits("alpha", "src/deep/x.rs", 1).is_none(),
            "past the bound"
        );
        assert!(
            s.admits("alpha", "srcx/lib.rs", 1).is_none(),
            "a sibling, not inside"
        );
        assert!(s.admits("alpha", "docs/a.md", 1).is_none());
        assert!(s.admits("beta", "src/lib.rs", 1).is_none());
        let whole = IngestScope::new("alpha", None);
        assert!(whole.admits("alpha", "docs/a.md", 1).is_some());
        assert!(whole.admits("alphabet", "a.md", 1).is_none());
        assert_eq!(IngestScope::new(".", None), IngestScope::new("", None));
    }
}
