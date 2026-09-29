//! A branch's files as git holds them: one committed tree, listed whole.
//!
//! A [`Tree`] is read once per tree id ([`super::git_source`] keeps the recent
//! ones) and answers every question a store asks of its lower layer without
//! another process: whether a path is a file or a folder, a folder's children,
//! every file under a prefix, and which blob holds a file's bytes.
//!
//! It holds what a reader of files would find in a checkout of the tree,
//! narrowed by the store's own rules: regular and executable files and the
//! folders holding them. A symbolic link and a submodule are not files a tool
//! reads, and a path under a protected folder, or one a Windows open would
//! resolve to another name, is left out entirely — the paths the folder layer
//! refuses too ([`VfsStore::is_protected`], [`VfsStore::addressable`]).
//!
//! Hidden entries — a segment starting with `.` — are left out of listings
//! and searches and still read by exact path, as the folder layer's walk
//! treats them: hidden below where the walk starts, visible when the listing
//! or search names their folder itself.

use std::collections::BTreeMap;
use std::ops::Bound;

use super::VfsStore;
use crate::{FileMode, GitError, ObjectKind, Oid, Repo, SizedEntry};

/// One path in a tree.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Node {
    File { blob: Oid, size: u64 },
    Dir,
}

/// Every file and folder of one tree, by repository-relative path.
#[derive(Debug, Default)]
pub struct Tree {
    /// `None` for the tree of a branch that does not exist: no files.
    id: Option<Oid>,
    nodes: BTreeMap<String, Node>,
}

impl Tree {
    /// No files — what a branch that does not exist holds, or a repository
    /// with no commit yet.
    pub(crate) fn empty() -> Self {
        Self::default()
    }

    /// The tree `id` of `repo`, listed whole in one process.
    pub fn read(repo: &Repo, id: &Oid) -> Result<Self, GitError> {
        Ok(Self::from_listing(id.clone(), repo.ls_tree_all(id)?))
    }

    /// The tree `id`, from its whole listing ([`crate::Repo::ls_tree_all`]).
    pub fn from_listing(id: Oid, listing: Vec<SizedEntry>) -> Self {
        let mut nodes = BTreeMap::new();
        for SizedEntry { entry, size } in listing {
            let path = entry.path.as_str();
            if VfsStore::is_protected(path) || !VfsStore::addressable(path) {
                continue;
            }
            let node = match (entry.kind, entry.mode) {
                (ObjectKind::Tree, _) => Node::Dir,
                (ObjectKind::Blob, FileMode::Regular | FileMode::Executable) => Node::File {
                    blob: entry.oid,
                    size: size.unwrap_or(0),
                },
                _ => continue,
            };
            nodes.insert(path.to_string(), node);
        }
        Self {
            id: Some(id),
            nodes,
        }
    }

    /// The tree's id; `None` for [`Self::empty`].
    pub fn id(&self) -> Option<&Oid> {
        self.id.as_ref()
    }

    /// Every file, hidden ones included, as `(path, blob, size)` in path
    /// order.
    pub fn files(&self) -> impl Iterator<Item = (&str, &Oid, u64)> {
        self.nodes.iter().filter_map(|(path, node)| match node {
            Node::File { blob, size } => Some((path.as_str(), blob, *size)),
            Node::Dir => None,
        })
    }

    /// The blob holding the file at `norm`, and its size — `None` when
    /// `norm` is no file here.
    pub fn file(&self, norm: &str) -> Option<(&Oid, u64)> {
        match self.nodes.get(norm)? {
            Node::File { blob, size } => Some((blob, *size)),
            Node::Dir => None,
        }
    }

    /// Whether `norm` is a folder. The root, `""`, always is.
    pub(crate) fn is_dir(&self, norm: &str) -> bool {
        norm.is_empty() || matches!(self.nodes.get(norm), Some(Node::Dir))
    }

    /// The files and folders directly inside the folder `dir`, as `(path,
    /// bytes, is_dir)` — `bytes` is `None` for a folder — sorted by path,
    /// hidden ones left out: what `file_list` shows of it.
    pub fn children(&self, dir: &str) -> Vec<(String, Option<usize>, bool)> {
        let inside = if dir.is_empty() {
            String::new()
        } else {
            format!("{dir}/")
        };
        self.under(&inside)
            .filter(|(path, _)| {
                let name = &path[inside.len()..];
                !name.contains('/') && !name.starts_with('.')
            })
            .map(|(path, node)| match node {
                Node::File { size, .. } => (path.clone(), Some(*size as usize), false),
                Node::Dir => (path.clone(), None, true),
            })
            .collect()
    }

    /// Every file under `prefix`, sorted. A prefix naming a folder selects
    /// what is inside it, hidden entries below it left out; any other prefix
    /// is a plain string prefix of the path — `src/ma` selects `src/main.rs`
    /// — with hidden entries anywhere left out.
    pub(crate) fn files_under(&self, prefix: &str) -> Vec<String> {
        let (from, shown) = if !prefix.is_empty() && self.is_dir(prefix) {
            (format!("{prefix}/"), prefix.len() + 1)
        } else {
            (prefix.to_string(), 0)
        };
        self.under(&from)
            .filter(|(_, node)| matches!(node, Node::File { .. }))
            .filter(|(path, _)| !is_hidden(&path[shown..]))
            .map(|(path, _)| path.clone())
            .collect()
    }

    /// Every entry whose path starts with `prefix`, in path order.
    fn under<'t>(&'t self, prefix: &'t str) -> impl Iterator<Item = (&'t String, &'t Node)> + 't {
        self.nodes
            .range::<str, _>((Bound::Included(prefix), Bound::Unbounded))
            .take_while(move |(path, _)| path.starts_with(prefix))
    }
}

/// Whether any segment of `path` starts with a `.`.
fn is_hidden(path: &str) -> bool {
    path.split('/').any(|s| s.starts_with('.'))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{RepoPath, TreeEntry};

    const BLOB: &str = "ce013625030ba8dba906f756967f9e9ca394464a";

    fn entry(path: &str, mode: FileMode, size: Option<u64>) -> SizedEntry {
        let kind = match mode {
            FileMode::Tree => ObjectKind::Tree,
            FileMode::Submodule => ObjectKind::Commit,
            _ => ObjectKind::Blob,
        };
        SizedEntry {
            entry: TreeEntry {
                mode,
                kind,
                oid: Oid::parse(BLOB).unwrap(),
                path: RepoPath::parse(path).unwrap(),
            },
            size,
        }
    }

    fn file(path: &str, size: u64) -> SizedEntry {
        entry(path, FileMode::Regular, Some(size))
    }

    fn dir(path: &str) -> SizedEntry {
        entry(path, FileMode::Tree, None)
    }

    fn tree(listing: Vec<SizedEntry>) -> Tree {
        Tree::from_listing(Oid::parse(BLOB).unwrap(), listing)
    }

    fn sample() -> Tree {
        tree(vec![
            file(".gitignore", 3),
            dir(".github"),
            file(".github/ci.yml", 4),
            file("README.md", 10),
            dir("src"),
            file("src/main.rs", 12),
            dir("src/util"),
            file("src/util/helper.rs", 14),
            dir("srcx"),
            file("srcx/other.rs", 5),
        ])
    }

    /// **Files and folders answer as themselves**, the root is a folder, and
    /// a file carries its blob and size.
    #[test]
    fn files_and_folders_are_told_apart() {
        let t = sample();
        assert_eq!(t.file("src/main.rs").map(|(_, s)| s), Some(12));
        assert_eq!(t.file("src"), None);
        assert_eq!(t.file("nope.rs"), None);
        assert!(t.is_dir("") && t.is_dir("src") && t.is_dir("src/util"));
        assert!(!t.is_dir("src/main.rs") && !t.is_dir("nope"));
        assert_eq!(t.file(".gitignore").map(|(_, s)| s), Some(3));
    }

    /// **A listing is one level, hidden entries out** — and a hidden folder
    /// named directly lists what is in it.
    #[test]
    fn a_listing_is_one_level_without_hidden_entries() {
        let t = sample();
        assert_eq!(
            t.children(""),
            vec![
                ("README.md".to_string(), Some(10), false),
                ("src".to_string(), None, true),
                ("srcx".to_string(), None, true),
            ]
        );
        assert_eq!(
            t.children("src"),
            vec![
                ("src/main.rs".to_string(), Some(12), false),
                ("src/util".to_string(), None, true),
            ]
        );
        assert_eq!(
            t.children(".github"),
            vec![(".github/ci.yml".to_string(), Some(4), false)]
        );
        assert!(t.children("nope").is_empty());
    }

    /// **A folder prefix selects that folder's files, a partial one is a
    /// string prefix** — `src` never takes in `srcx`, `src/ma` takes in
    /// `src/main.rs` — and hidden files show only below a folder named.
    #[test]
    fn a_search_prefix_is_a_folder_or_a_string_prefix() {
        let t = sample();
        assert_eq!(
            t.files_under(""),
            vec![
                "README.md",
                "src/main.rs",
                "src/util/helper.rs",
                "srcx/other.rs"
            ]
        );
        assert_eq!(
            t.files_under("src"),
            vec!["src/main.rs", "src/util/helper.rs"]
        );
        assert_eq!(t.files_under("src/ma"), vec!["src/main.rs"]);
        assert_eq!(t.files_under(".github"), vec![".github/ci.yml"]);
        assert!(t.files_under("nope").is_empty());
    }

    /// **What no tool may read is not in the tree**: links, submodules,
    /// protected folders, and names Windows would open as something else.
    #[test]
    fn links_submodules_and_protected_paths_are_left_out() {
        let t = tree(vec![
            file("kept.txt", 1),
            entry("link", FileMode::Symlink, Some(8)),
            entry("vendor/lib", FileMode::Submodule, None),
            dir("secrets"),
            file("secrets/key.pem", 9),
            file("web/Secrets/auth.yaml", 9),
            file("notes~2.md", 1),
            file("x./y", 1),
            entry("run.sh", FileMode::Executable, Some(2)),
        ]);
        assert_eq!(t.files_under(""), vec!["kept.txt", "run.sh"]);
        assert_eq!(t.file("link"), None);
        assert!(!t.is_dir("secrets"));
    }

    /// **Every file is enumerated, hidden ones too, in path order** — folders
    /// are not files, and what no tool may read was never kept.
    #[test]
    fn every_file_is_enumerated_in_path_order() {
        let t = sample();
        let files: Vec<(&str, u64)> = t.files().map(|(p, _, s)| (p, s)).collect();
        assert_eq!(
            files,
            vec![
                (".github/ci.yml", 4),
                (".gitignore", 3),
                ("README.md", 10),
                ("src/main.rs", 12),
                ("src/util/helper.rs", 14),
                ("srcx/other.rs", 5),
            ]
        );
    }

    #[test]
    fn the_empty_tree_holds_nothing() {
        let t = Tree::empty();
        assert_eq!(t.id(), None);
        assert!(t.is_dir(""));
        assert!(t.children("").is_empty() && t.files_under("").is_empty());
    }
}
