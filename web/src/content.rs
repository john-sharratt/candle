//! Content roots — directories on disk, read per request.
//!
//! A site's roots are searched in order, so one URL tree can be assembled from
//! several directories: `[content/npcd, content/common]` means `/lib/dom.js`
//! falls through to the shared copy while `/pages/roster.js` does not. Sites
//! share the framework without either of them owning it.
//!
//! Every file is read when it is asked for, so saving one is the deployment:
//! the next request serves it, with no rebuild and no restart.

use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use tokio::fs;

/// The ordered roots for one site: canonical directories, searched in order.
#[derive(Clone, Debug, Default)]
pub struct Roots(pub Arc<Vec<PathBuf>>);

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
                .map(|p| p.canonicalize().unwrap_or_else(|_| p.clone()))
                .collect(),
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
        for root in self.0.iter() {
            let Ok(mut entries) = fs::read_dir(root.join(rel)).await else {
                continue;
            };
            while let Ok(Some(e)) = entries.next_entry().await {
                if e.file_type().await.map(|t| t.is_file()).unwrap_or(false) {
                    names.push(e.file_name().to_string_lossy().into_owned());
                }
            }
        }
        names.sort();
        names.dedup();
        names
    }

    /// The on-disk path of `rel` in the first root that has it — what a
    /// caller keys a cache by. `None` when no root holds it.
    pub fn disk_path(&self, rel: &str) -> Option<PathBuf> {
        self.0.iter().find_map(|root| {
            let p = root.join(rel);
            p.is_file().then_some(p)
        })
    }

    /// First root that has `rel`, as bytes. `rel` must already be validated by
    /// [`safe_rel`].
    pub async fn read(&self, rel: &str) -> Option<Vec<u8>> {
        for root in self.0.iter() {
            // Re-check after canonicalisation: this is what closes the symlink
            // route out of the site, which the component check in `safe_rel`
            // cannot see.
            let Ok(real) = root.join(rel).canonicalize() else {
                continue;
            };
            if !real.starts_with(root) {
                continue;
            }
            let Ok(meta) = fs::metadata(&real).await else {
                continue;
            };
            if !meta.is_file() {
                continue;
            }
            if let Ok(b) = fs::read(&real).await {
                return Some(b);
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
