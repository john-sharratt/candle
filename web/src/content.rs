//! Content roots — directories on disk, or compiled into the binary.
//!
//! A site's roots are searched in order, so one URL tree can be assembled from
//! several directories: `[content/npcd, content/common]` means `/lib/dom.js`
//! falls through to the shared copy while `/pages/roster.js` does not. Sites
//! share the framework without either of them owning it.
//!
//! The two sources answer the same lookups. The gateway reads its sites from
//! disk, so saving a file is the deployment: the next request serves it, with
//! no rebuild and no restart. A daemon serves the UI it was built with,
//! compiled in, so it ships as one file that needs no checkout beside it.
//!
//! What is served from a file is prepared once per version of it and kept
//! ([`Stamped`]): every request still resolves its path through the roots in
//! order — so a file added to an earlier root shadows at once — and a disk
//! file whose stamp has moved is prepared again.

mod serve;
pub mod stamped;

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use include_dir::{Dir, File};
use tokio::fs;

pub use serve::Kept;
pub use stamped::{Key, Origin, Stamp, Stamped};

/// Where one root's files come from.
#[derive(Clone)]
pub enum Source {
    /// A canonical directory, read per request.
    Disk(PathBuf),
    /// Compiled in.
    Embedded(&'static Dir<'static>),
}

impl Debug for Source {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            Source::Disk(p) => write!(f, "disk:{}", p.display()),
            Source::Embedded(d) => write!(f, "embedded:{}", d.path().display()),
        }
    }
}

/// The ordered roots for one site, searched in order, and what has been
/// prepared from their files — shared by every clone.
#[derive(Clone, Debug, Default)]
pub struct Roots {
    sources: Arc<Vec<Source>>,
    stamped: Arc<Stamped<Kept>>,
}

/// Where a path resolved to: a file on disk at its real path, as it stands
/// now, or a compiled-in file.
#[derive(Clone, Debug)]
pub enum Resolved {
    Disk { real: PathBuf, stamp: Stamp },
    Embedded(&'static File<'static>),
}

impl Resolved {
    /// The key of what is derived from this file as `kind`.
    pub fn key(&self, kind: &str) -> Key {
        let origin = match self {
            Resolved::Disk { real, .. } => Origin::Disk(real.clone()),
            Resolved::Embedded(f) => Origin::Embedded(*f as *const File<'static> as usize),
        };
        Key {
            origin,
            kind: kind.to_string(),
        }
    }

    /// The file's version: its stamp on disk; none for a compiled-in file,
    /// which cannot change while the process runs.
    pub fn stamp(&self) -> Option<Stamp> {
        match self {
            Resolved::Disk { stamp, .. } => Some(*stamp),
            Resolved::Embedded(_) => None,
        }
    }

    /// Whether the file is still the version it resolved to — what is checked
    /// after reading it, before keeping what was derived from the read.
    pub async fn unchanged(&self) -> bool {
        match self {
            Resolved::Disk { real, stamp } => stamp_of(real).await == Some(*stamp),
            Resolved::Embedded(_) => true,
        }
    }

    /// The file's bytes.
    pub async fn bytes(&self) -> Option<Vec<u8>> {
        match self {
            Resolved::Disk { real, .. } => fs::read(real).await.ok(),
            Resolved::Embedded(f) => Some(f.contents().to_vec()),
        }
    }
}

/// A disk file's stamp, or `None` when it is not a file.
pub async fn stamp_of(path: &Path) -> Option<Stamp> {
    let meta = fs::metadata(path).await.ok()?;
    if !meta.is_file() {
        return None;
    }
    Some((meta.modified().ok()?, meta.len()))
}

impl Roots {
    fn of(sources: Vec<Source>) -> Self {
        Roots {
            sources: Arc::new(sources),
            stamped: Arc::default(),
        }
    }

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
        Roots::of(
            paths
                .iter()
                .map(|p| Source::Disk(p.canonicalize().unwrap_or_else(|_| p.clone())))
                .collect(),
        )
    }

    /// Directories compiled in with `include_dir!`, searched in order.
    pub fn embedded(dirs: &[&'static Dir<'static>]) -> Self {
        Roots::of(dirs.iter().copied().map(Source::Embedded).collect())
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// The roots, in the order they are searched.
    pub fn sources(&self) -> &[Source] {
        &self.sources
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
        for src in self.sources.iter() {
            match src {
                Source::Disk(root) => {
                    let Ok(mut entries) = fs::read_dir(root.join(rel)).await else {
                        continue;
                    };
                    while let Ok(Some(e)) = entries.next_entry().await {
                        if e.file_type().await.map(|t| t.is_file()).unwrap_or(false) {
                            names.push(e.file_name().to_string_lossy().into_owned());
                        }
                    }
                }
                Source::Embedded(root) => {
                    let dir = if rel.is_empty() {
                        Some(*root)
                    } else {
                        root.get_dir(rel)
                    };
                    if let Some(dir) = dir {
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

    /// The on-disk path of `rel` in the first disk root that has it — what a
    /// caller keys a cache by. `None` when no disk root holds it: compiled-in
    /// content has no path, and cannot change while the process runs.
    pub fn disk_path(&self, rel: &str) -> Option<PathBuf> {
        self.sources.iter().find_map(|src| match src {
            Source::Disk(root) => {
                let p = root.join(rel);
                p.is_file().then_some(p)
            }
            Source::Embedded(_) => None,
        })
    }

    /// Where `rel` is: the first root that has it, looked for afresh. `rel`
    /// must already be validated by [`safe_rel`].
    pub async fn resolve(&self, rel: &str) -> Option<Resolved> {
        for src in self.sources.iter() {
            match src {
                Source::Disk(root) => {
                    // Re-check after canonicalisation: this is what closes the
                    // symlink route out of the site, which the component check
                    // in `safe_rel` cannot see.
                    let Ok(real) = fs::canonicalize(root.join(rel)).await else {
                        continue;
                    };
                    if !real.starts_with(root) {
                        continue;
                    }
                    if let Some(stamp) = stamp_of(&real).await {
                        return Some(Resolved::Disk { real, stamp });
                    }
                }
                Source::Embedded(dir) => {
                    if let Some(f) = dir.get_file(rel) {
                        return Some(Resolved::Embedded(f));
                    }
                }
            }
        }
        None
    }

    /// First root that has `rel`, as bytes. `rel` must already be validated by
    /// [`safe_rel`].
    pub async fn read(&self, rel: &str) -> Option<Vec<u8>> {
        self.resolve(rel).await?.bytes().await
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

    static COMMON: Dir<'static> = include_dir::include_dir!("$CARGO_MANIFEST_DIR/content/common");

    /// A compiled-in directory answers as the same directory on disk does:
    /// the same bytes, the same listing — and no disk path, since it has none.
    #[tokio::test]
    async fn an_embedded_root_answers_as_its_directory_on_disk() {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("content")
            .join("common");
        let disk = Roots::disk(&[dir]);
        let embedded = Roots::embedded(&[&COMMON]);
        for rel in ["base.css", "lib/aggregate.js"] {
            let want = disk.read(rel).await.expect("on disk");
            assert_eq!(embedded.read(rel).await, Some(want), "{rel}");
        }
        assert_eq!(embedded.list("lib").await, disk.list("lib").await);
        assert_eq!(embedded.read("lib/none.js").await, None);
        assert_eq!(embedded.disk_path("base.css"), None);
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
