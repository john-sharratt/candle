//! The mind's files at one commit (`docs/substrate_git.md` §8.1).
//!
//! Everything npcd reads of the mind — the schema, the libraries, the
//! registries, the layers, the corpus — reads it here, through one of two
//! backings:
//!
//! - **A commit** of the mind's object store, which the substrate fills
//!   ([`substrate_git::member::Repos`]): the tree indexed once when the
//!   snapshot is taken, each file's bytes read in process on demand. There is
//!   no folder: what a member reads is exactly what the log names.
//! - **A folder**, for a daemon run on a `--mind` that is no git repository.
//!
//! A snapshot never changes. When the mind's branch moves, the holder takes a
//! new one ([`super::Mind::replace`]).
//!
//! Paths are mind-relative and `/`-separated, checked by [`MindPath`]: nothing
//! read here can leave the mind, and a link is never followed.

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use zend_vfs::{FileMode, ObjectKind, Oid, Repo, Rev};

use crate::mind::path::MindPath;
use crate::sim::bench::refuse_link;
use crate::sim::shelf::{Shelf, NO_OPERATION};

/// One entry of a commit's tree.
#[derive(Debug, Clone)]
enum Node {
    File { oid: Oid, size: u64 },
    /// A symbolic link: never followed, never read as a document.
    Link,
    Dir,
}

enum Backing {
    Folder(PathBuf),
    Commit {
        commit: Oid,
        nodes: BTreeMap<String, Node>,
        blobs: zend_vfs::BlobReader,
    },
}

/// The mind's files, as they stand at one commit or in one folder.
pub struct MindFiles {
    backing: Backing,
}

impl std::fmt::Debug for MindFiles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.backing {
            Backing::Folder(root) => write!(f, "MindFiles(folder {})", root.display()),
            Backing::Commit { commit, nodes, .. } => {
                write!(f, "MindFiles(commit {commit}, {} entries)", nodes.len())
            }
        }
    }
}

impl MindFiles {
    /// The files of the folder `root`, read as they stand.
    pub fn folder(root: impl Into<PathBuf>) -> Self {
        Self {
            backing: Backing::Folder(root.into()),
        }
    }

    /// The files of `commit` in `repo`: its whole tree indexed now.
    pub fn at_commit(repo: &Repo, commit: &Oid) -> anyhow::Result<Self> {
        let blobs = repo.blobs();
        let (commit, tree) = blobs
            .commit_of(&Rev::Oid(commit.clone()))?
            .ok_or_else(|| anyhow::anyhow!("the mind's store holds no commit {commit}"))?;
        let mut nodes = BTreeMap::new();
        for sized in repo.ls_tree_all(&tree)? {
            let path = sized.entry.path.to_string();
            let node = match sized.entry.kind {
                ObjectKind::Tree => Node::Dir,
                ObjectKind::Blob if sized.entry.mode == FileMode::Symlink => Node::Link,
                ObjectKind::Blob => Node::File {
                    oid: sized.entry.oid,
                    size: sized.size.unwrap_or(0),
                },
                // A submodule's commit: no file of the mind's.
                _ => continue,
            };
            nodes.insert(path, node);
        }
        Ok(Self {
            backing: Backing::Commit {
                commit,
                nodes,
                blobs,
            },
        })
    }

    /// The commit these files are, or `None` for a folder.
    pub fn commit(&self) -> Option<&Oid> {
        match &self.backing {
            Backing::Commit { commit, .. } => Some(commit),
            Backing::Folder(_) => None,
        }
    }

    /// The folder these files are, or `None` for a commit — which has none.
    pub fn folder_root(&self) -> Option<&Path> {
        match &self.backing {
            Backing::Folder(root) => Some(root),
            Backing::Commit { .. } => None,
        }
    }

    /// What `rel` holds as text, or `None` when no document is there or it is
    /// not UTF-8.
    pub fn read(&self, rel: &str) -> Option<String> {
        String::from_utf8(self.bytes(rel)?).ok()
    }

    /// What `rel` holds, or `None` when no document is there.
    pub fn bytes(&self, rel: &str) -> Option<Vec<u8>> {
        match &self.backing {
            Backing::Folder(root) => std::fs::read(folder_file(root, rel)?).ok(),
            Backing::Commit { nodes, blobs, .. } => match nodes.get(&key(rel)?)? {
                Node::File { oid, .. } => blobs.read_blob(oid).ok().flatten(),
                Node::Dir | Node::Link => None,
            },
        }
    }

    /// Whether a document stands at `rel`.
    pub fn is_file(&self, rel: &str) -> bool {
        self.size(rel).is_some()
    }

    /// Whether something that is not a document stands at `rel`: a folder, or
    /// a link, which is never followed.
    pub fn not_a_document(&self, rel: &str) -> bool {
        match &self.backing {
            Backing::Folder(root) => {
                let Some(full) = MindPath::parse(rel).ok().and_then(|p| p.resolve(root).ok())
                else {
                    return false;
                };
                match full.symlink_metadata() {
                    Ok(meta) => !meta.is_file() || refuse_link(&full, rel).is_err(),
                    Err(_) => false,
                }
            }
            Backing::Commit { nodes, .. } => key(rel)
                .and_then(|k| nodes.get(&k))
                .is_some_and(|n| matches!(n, Node::Dir | Node::Link)),
        }
    }

    /// Whether a folder stands at `rel`; the root always does.
    pub fn is_dir(&self, rel: &str) -> bool {
        match &self.backing {
            Backing::Folder(root) => MindPath::parse(rel)
                .ok()
                .and_then(|p| p.resolve(root).ok())
                .filter(|full| refuse_link(full, rel).is_ok())
                .is_some_and(|full| full.is_dir()),
            Backing::Commit { nodes, .. } => match key(rel) {
                Some(k) if k.is_empty() => true,
                Some(k) => matches!(nodes.get(&k), Some(Node::Dir)),
                None => false,
            },
        }
    }

    /// How many bytes the document at `rel` holds, or `None` when none is
    /// there.
    pub fn size(&self, rel: &str) -> Option<u64> {
        match &self.backing {
            Backing::Folder(root) => {
                let meta = folder_file(root, rel)?.metadata().ok()?;
                meta.is_file().then_some(meta.len())
            }
            Backing::Commit { nodes, .. } => match nodes.get(&key(rel)?)? {
                Node::File { size, .. } => Some(*size),
                Node::Dir | Node::Link => None,
            },
        }
    }

    /// What is directly in the folder `dir`: documents by name, folders with a
    /// trailing `/`. Empty for a folder that is not there.
    pub fn list(&self, dir: &str) -> BTreeSet<String> {
        match &self.backing {
            Backing::Folder(root) => {
                let Some(full) = MindPath::parse(dir)
                    .ok()
                    .and_then(|p| p.resolve(root).ok())
                    .filter(|full| refuse_link(full, dir).is_ok())
                else {
                    return BTreeSet::new();
                };
                let Ok(entries) = std::fs::read_dir(full) else {
                    return BTreeSet::new();
                };
                entries
                    .flatten()
                    .map(|e| {
                        let name = e.file_name().to_string_lossy().into_owned();
                        match e.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                            true => format!("{name}/"),
                            false => name,
                        }
                    })
                    .collect()
            }
            Backing::Commit { nodes, .. } => {
                let Some(k) = key(dir) else {
                    return BTreeSet::new();
                };
                let prefix = if k.is_empty() { k } else { format!("{k}/") };
                nodes
                    .range(prefix.clone()..)
                    .take_while(|(p, _)| p.starts_with(&prefix))
                    .filter_map(|(p, node)| {
                        let rest = &p[prefix.len()..];
                        (!rest.contains('/')).then(|| match node {
                            Node::Dir => format!("{rest}/"),
                            Node::Link => rest.to_string(),
                            Node::File { .. } => rest.to_string(),
                        })
                    })
                    .collect()
            }
        }
    }

    /// Every document under the folder `dir`, at any depth, as mind-relative
    /// paths in order. Empty for a folder that is not there.
    pub fn walk(&self, dir: &str) -> Vec<String> {
        match &self.backing {
            Backing::Folder(_) => {
                let mut out = Vec::new();
                let mut pending = vec![key(dir).unwrap_or_default()];
                while let Some(at) = pending.pop() {
                    for name in self.list(&at) {
                        let path = match at.is_empty() {
                            true => name.clone(),
                            false => format!("{at}/{name}"),
                        };
                        match path.strip_suffix('/') {
                            Some(sub) => pending.push(sub.to_string()),
                            None => out.push(path),
                        }
                    }
                }
                out.sort();
                out
            }
            Backing::Commit { nodes, .. } => {
                let Some(k) = key(dir) else {
                    return Vec::new();
                };
                let prefix = if k.is_empty() { k } else { format!("{k}/") };
                nodes
                    .range(prefix.clone()..)
                    .take_while(|(p, _)| p.starts_with(&prefix))
                    .filter(|(_, n)| matches!(n, Node::File { .. }))
                    .map(|(p, _)| p.clone())
                    .collect()
            }
        }
    }
}

/// How two snapshots of the mind differ, document by document.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Changes {
    pub added: Vec<String>,
    pub changed: Vec<String>,
    pub removed: Vec<String>,
}

impl MindFiles {
    /// The documents `newer` adds, changes and removes against `self`. Two
    /// commits compare by blob id, without reading a document; anything else
    /// by its bytes.
    pub fn diff(&self, newer: &MindFiles) -> Changes {
        let mut out = Changes::default();
        let before: BTreeSet<String> = self.walk("").into_iter().collect();
        let after: BTreeSet<String> = newer.walk("").into_iter().collect();
        for path in after.difference(&before) {
            out.added.push(path.clone());
        }
        for path in before.difference(&after) {
            out.removed.push(path.clone());
        }
        for path in before.intersection(&after) {
            let same = match (self.blob(path), newer.blob(path)) {
                (Some(a), Some(b)) => a == b,
                _ => self.bytes(path) == newer.bytes(path),
            };
            if !same {
                out.changed.push(path.clone());
            }
        }
        out
    }

    /// The blob id of the document at `rel`, for a commit.
    fn blob(&self, rel: &str) -> Option<Oid> {
        match &self.backing {
            Backing::Commit { nodes, .. } => match nodes.get(&key(rel)?)? {
                Node::File { oid, .. } => Some(oid.clone()),
                _ => None,
            },
            Backing::Folder(_) => None,
        }
    }
}

/// A snapshot held by something that compares itself: two are equal when they
/// are the same snapshot. Empty when there is no mind.
#[derive(Clone, Default)]
pub struct MindRef(pub Option<std::sync::Arc<MindFiles>>);

impl PartialEq for MindRef {
    fn eq(&self, other: &Self) -> bool {
        match (&self.0, &other.0) {
            (Some(a), Some(b)) => std::sync::Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        }
    }
}

impl Eq for MindRef {}

impl std::fmt::Debug for MindRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Some(files) => files.fmt(f),
            None => f.write_str("no mind"),
        }
    }
}

/// The published mind takes no commit: nothing reaches it but through an
/// operation's publish step.
impl Shelf for MindFiles {
    fn read(&self, key: &str) -> Option<String> {
        MindFiles::read(self, key)
    }

    fn list(&self, dir: &str) -> BTreeSet<String> {
        MindFiles::list(self, dir)
    }

    fn commit(&self, _changes: &[(String, Option<String>)]) -> Result<(), String> {
        Err(NO_OPERATION.to_string())
    }

    fn size(&self, key: &str) -> Option<u64> {
        MindFiles::size(self, key)
    }
}

/// `rel` checked and spelled as a tree path: no leading or trailing `/`, the
/// root empty. `None` for a path [`MindPath`] refuses.
fn key(rel: &str) -> Option<String> {
    MindPath::parse(rel).ok().map(|p| p.segments().join("/"))
}

/// Where `rel` is in the folder `root`, refusing a path that leaves it or a
/// link.
fn folder_file(root: &Path, rel: &str) -> Option<PathBuf> {
    let full = MindPath::parse(rel).ok()?.resolve(root).ok()?;
    refuse_link(&full, rel).ok()?;
    Some(full)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::process::Command;

    fn git(dir: &Path, args: &[&str]) -> String {
        let out = Command::new("git")
            .current_dir(dir)
            .args(["-c", "user.name=t", "-c", "user.email=t@t"])
            .args(args)
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap().trim().to_string()
    }

    fn mind(dir: &Path) {
        std::fs::create_dir_all(dir.join("layers/eras")).unwrap();
        std::fs::write(dir.join("projection.yaml"), "layers: []\n").unwrap();
        std::fs::write(dir.join("layers/eras/first.md"), "the first era\n").unwrap();
        std::fs::write(dir.join("layers/eras/second.md"), "the second\n").unwrap();
    }

    fn check(files: &MindFiles) {
        assert_eq!(files.read("projection.yaml").as_deref(), Some("layers: []\n"));
        assert_eq!(files.read("layers/eras/none.md"), None);
        assert!(files.is_file("layers/eras/first.md"));
        assert!(files.is_dir("layers/eras"));
        assert!(files.is_dir(""));
        assert!(!files.is_dir("layers/eras/first.md"));
        assert_eq!(files.size("layers/eras/second.md"), Some(11));
        assert_eq!(
            files.list("layers").into_iter().collect::<Vec<_>>(),
            ["eras/"]
        );
        assert_eq!(
            files.list("").into_iter().collect::<Vec<_>>(),
            ["layers/", "projection.yaml"]
        );
        assert_eq!(
            files.walk("layers"),
            ["layers/eras/first.md", "layers/eras/second.md"]
        );
        assert_eq!(files.read("../escape"), None);
    }

    /// **A folder and a commit read the same**: every question answers alike.
    #[test]
    fn a_folder_and_a_commit_read_the_same() {
        let dir = tempfile::tempdir().unwrap();
        mind(dir.path());
        check(&MindFiles::folder(dir.path()));

        git(dir.path(), &["init", "-q", "-b", "main"]);
        git(dir.path(), &["add", "-A"]);
        git(dir.path(), &["commit", "-q", "-m", "mind"]);
        let head = Oid::parse(&git(dir.path(), &["rev-parse", "HEAD"])).unwrap();
        // Change the folder after the commit: the commit's files do not move.
        std::fs::write(dir.path().join("projection.yaml"), "changed\n").unwrap();
        let repo = Repo::open(dir.path()).unwrap();
        let files = MindFiles::at_commit(&repo, &head).unwrap();
        assert_eq!(files.commit(), Some(&head));
        check(&files);
    }
}
