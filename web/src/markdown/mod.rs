//! Markdown, rendered on the way out.
//!
//! Papers and posts stay markdown files on disk: a paper is a document in
//! `docs/` that is still edited as a design document, and a post is one file
//! someone writes. Nothing is generated ahead of time, so publishing is saving
//! a file and there is no step between writing and reading — the same reason
//! the consoles have no bundler.
//!
//! Rendering a 170 KB paper is not free, so [`Cache`] keeps the result keyed by
//! the file's modification time: the first reader after an edit pays, everyone
//! else is served from memory, and touching the file is all it takes to
//! invalidate.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::SystemTime;

pub mod frontmatter;
pub mod math;
pub mod render;
pub mod slug;

pub use frontmatter::Post;
pub use render::{render, Document};

/// What was rendered, and the mtime it was rendered from.
type Rendered = (SystemTime, Arc<Document>);

/// Rendered documents, invalidated by mtime.
#[derive(Default, Clone)]
pub struct Cache {
    entries: Arc<RwLock<HashMap<PathBuf, Rendered>>>,
}

impl Cache {
    pub fn new() -> Self {
        Self::default()
    }

    /// Render `path`, or return the cached result if the file has not changed.
    ///
    /// `prepare` runs on the file's text before rendering — that is where a
    /// post's front matter is stripped. It runs only on a miss, which is
    /// correct because its output depends on nothing but the same file.
    pub async fn get(
        &self,
        path: &Path,
        prepare: impl Fn(&str) -> String,
    ) -> std::io::Result<Arc<Document>> {
        let mtime = tokio::fs::metadata(path).await?.modified()?;
        if let Some((seen, doc)) = self.entries.read().unwrap().get(path) {
            if *seen == mtime {
                return Ok(doc.clone());
            }
        }

        let text = tokio::fs::read_to_string(path).await?;
        let doc = Arc::new(render(&prepare(&text)));
        self.entries
            .write()
            .unwrap()
            .insert(path.to_path_buf(), (mtime, doc.clone()));
        Ok(doc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_second_read_is_served_from_the_cache_and_an_edit_invalidates_it() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("doc.md");
        tokio::fs::write(&path, "# One\n\nfirst\n").await.unwrap();

        let cache = Cache::new();
        let a = cache.get(&path, |s| s.to_string()).await.unwrap();
        let b = cache.get(&path, |s| s.to_string()).await.unwrap();
        assert!(Arc::ptr_eq(&a, &b), "the second read re-rendered");
        assert!(a.html.contains("first"));

        // Rewrite with a distinctly later mtime — filesystem timestamp
        // granularity is coarse enough that an immediate rewrite can share one.
        tokio::fs::write(&path, "# One\n\nsecond\n").await.unwrap();
        let later = SystemTime::now() + std::time::Duration::from_secs(2);
        filetime::set_file_mtime(&path, filetime::FileTime::from_system_time(later)).ok();

        let c = cache.get(&path, |s| s.to_string()).await.unwrap();
        assert!(
            c.html.contains("second"),
            "the edit was not picked up: {}",
            c.html
        );
    }

    #[tokio::test]
    async fn a_missing_file_is_an_error_rather_than_an_empty_document() {
        let cache = Cache::new();
        let err = cache
            .get(Path::new("definitely/not/here.md"), |s| s.to_string())
            .await;
        assert!(err.is_err());
    }
}
