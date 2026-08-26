//! Content roots — from disk, or embedded in the binary.
//!
//! A site's roots are searched in order, so one URL tree can be assembled from
//! several directories: `[content/npcd, content/common]` means `/lib/dom.js`
//! falls through to the shared copy while `/pages/roster.js` does not. Sites
//! share the framework without either of them owning it.
//!
//! Both sources satisfy the same lookup, so a daemon that `include_dir!`s its
//! content behaves identically to the proxy reading from disk.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

/// Where a site's files come from.
#[derive(Clone)]
pub enum Source {
    /// Read from the filesystem — edit and refresh, no rebuild.
    Disk(PathBuf),
    /// Compiled in. The daemon ships as one file with its console inside it.
    Embedded(&'static include_dir::Dir<'static>),
}

impl std::fmt::Debug for Source {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Source::Disk(p) => write!(f, "disk:{}", p.display()),
            Source::Embedded(_) => write!(f, "embedded"),
        }
    }
}

/// The ordered roots for one site.
#[derive(Clone, Debug, Default)]
pub struct Roots(pub Arc<Vec<Source>>);

impl Roots {
    /// Roots are canonicalised here rather than being trusted as given.
    ///
    /// [`read`](Self::read) canonicalises each candidate file and checks it is
    /// still under its root — the check that closes the symlink route out of
    /// the site. That comparison only holds if the root is canonical too: on
    /// Windows a canonical path carries a `\\?\` prefix, so an
    /// un-canonicalised root fails `starts_with` against every file beneath it
    /// and the site serves nothing at all. Doing it here means a `Roots` cannot
    /// be built in that state.
    pub fn disk(paths: &[PathBuf]) -> Self {
        Roots(Arc::new(
            paths
                .iter()
                .map(|p| Source::Disk(p.canonicalize().unwrap_or_else(|_| p.clone())))
                .collect(),
        ))
    }
    pub fn embedded(dirs: &[&'static include_dir::Dir<'static>]) -> Self {
        Roots(Arc::new(
            dirs.iter().copied().map(Source::Embedded).collect(),
        ))
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// File names directly inside `rel`, across every root, deduplicated and
    /// sorted.
    ///
    /// This is what lets a blog index itself: writing a post is adding one
    /// file, with no manifest to remember to update. Subdirectories are not
    /// descended — a listing that recursed would make an accidental copy in a
    /// subfolder appear as a published post.
    pub async fn list(&self, rel: &str) -> Vec<String> {
        let mut names: Vec<String> = Vec::new();
        for src in self.0.iter() {
            match src {
                Source::Disk(root) => {
                    let dir = root.join(rel);
                    let Ok(mut entries) = tokio::fs::read_dir(&dir).await else {
                        continue;
                    };
                    while let Ok(Some(e)) = entries.next_entry().await {
                        if e.file_type().await.map(|t| t.is_file()).unwrap_or(false) {
                            names.push(e.file_name().to_string_lossy().into_owned());
                        }
                    }
                }
                Source::Embedded(root) => {
                    if let Some(dir) = root.get_dir(rel) {
                        names.extend(dir.files().filter_map(|f| {
                            f.path()
                                .file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                        }));
                    }
                }
            }
        }
        names.sort();
        names.dedup();
        names
    }

    /// The on-disk path of `rel`, if it comes from a disk root.
    ///
    /// `None` for embedded content, which has no path and cannot change
    /// without a rebuild. Callers use it to key a cache by mtime; there is
    /// nothing to invalidate on the embedded side, so `None` is the answer
    /// rather than a synthesised path.
    pub fn disk_path(&self, rel: &str) -> Option<std::path::PathBuf> {
        self.0.iter().find_map(|src| match src {
            Source::Disk(root) => {
                let p = root.join(rel);
                p.is_file().then_some(p)
            }
            Source::Embedded(_) => None,
        })
    }

    /// First root that has `rel`, as bytes. `rel` must already be validated by
    /// [`safe_rel`].
    pub async fn read(&self, rel: &str) -> Option<Vec<u8>> {
        for src in self.0.iter() {
            match src {
                Source::Disk(root) => {
                    let joined = root.join(rel);
                    // Re-check after canonicalisation: this is what closes the
                    // symlink route out of the site, which the component check
                    // in `safe_rel` cannot see.
                    let Ok(real) = joined.canonicalize() else {
                        continue;
                    };
                    if !real.starts_with(root) {
                        continue;
                    }
                    let Ok(meta) = tokio::fs::metadata(&real).await else {
                        continue;
                    };
                    if !meta.is_file() {
                        continue;
                    }
                    if let Ok(b) = tokio::fs::read(&real).await {
                        return Some(b);
                    }
                }
                Source::Embedded(dir) => {
                    if let Some(f) = dir.get_file(rel) {
                        return Some(f.contents().to_vec());
                    }
                }
            }
        }
        None
    }
}

/// Normalise a request path to a root-relative path, or reject it.
///
/// `..` is refused outright rather than resolved: a request carrying a parent
/// segment is an attack or a bug either way, and resolving it politely is how a
/// static server ends up serving `/etc/passwd`. Percent-decoding refuses an
/// encoded separator or NUL so `%2e%2e%2f` cannot smuggle one past this check.
pub fn safe_rel(path: &str) -> Option<String> {
    let rel = path.trim_start_matches('/');
    if rel.is_empty() {
        return Some(String::new());
    }
    let decoded = percent_decode(rel)?;
    for c in Path::new(&decoded).components() {
        match c {
            Component::Normal(_) => {}
            _ => return None,
        }
    }
    Some(decoded)
}

fn percent_decode(s: &str) -> Option<String> {
    if !s.contains('%') {
        return (!s.contains('\0')).then(|| s.to_string());
    }
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' {
            if i + 2 >= b.len() {
                return None;
            }
            let hi = (b[i + 1] as char).to_digit(16)?;
            let lo = (b[i + 2] as char).to_digit(16)?;
            let byte = (hi * 16 + lo) as u8;
            if byte == 0 || byte == b'/' || byte == b'\\' {
                return None;
            }
            out.push(byte);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

pub fn has_extension(path: &str) -> bool {
    path.rsplit('/')
        .next()
        .map(|seg| seg.contains('.'))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_relative_root_still_serves_its_files() {
        // The guard in `read` compares a canonicalised file against its root,
        // so a root given in non-canonical form used to match nothing and the
        // whole site 404'd. `disk` canonicalises to make that unrepresentable.
        let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let roots = Roots::disk(&[manifest.join("content").join("common")]);
        assert!(
            roots.read("base.css").await.is_some(),
            "a plain root did not read"
        );

        let dotted = Roots::disk(&[manifest.join("content").join(".").join("common")]);
        assert!(
            dotted.read("base.css").await.is_some(),
            "a dotted root did not read"
        );
    }

    #[tokio::test]
    async fn listing_returns_files_and_not_directories() {
        let roots = Roots::disk(&[PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("content")]);
        let names = roots.list("tokera").await;
        assert!(names.contains(&"papers.yaml".to_string()), "{names:?}");
        assert!(
            !names.contains(&"blog".to_string()),
            "a directory was listed: {names:?}"
        );
    }

    #[test]
    fn parent_segments_refused() {
        assert!(safe_rel("/../secrets").is_none());
        assert!(safe_rel("/a/../../b").is_none());
    }

    #[test]
    fn encoded_separators_refused() {
        assert!(safe_rel("/..%2fsecrets").is_none());
        assert!(safe_rel("/%2e%2e%2fx").is_none());
    }

    #[test]
    fn malformed_escape_refused() {
        assert!(safe_rel("/%zz").is_none());
        assert!(safe_rel("/%4").is_none());
    }

    #[test]
    fn ordinary_paths_normalise() {
        assert_eq!(safe_rel("/lib/dom.js").as_deref(), Some("lib/dom.js"));
        assert_eq!(safe_rel("/").as_deref(), Some(""));
    }

    #[test]
    fn extension_detection_ignores_dotted_dirs() {
        assert!(has_extension("lib/dom.js"));
        assert!(!has_extension("npc/123"));
        assert!(!has_extension("v1.2/thing"));
    }
}
