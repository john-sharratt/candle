//! The `repo_map` layer's units as one tree lists them: one per folder
//! holding files the layer reads, plus the workspace's own. Derived from the
//! tree alone — what the folder's listing shows, which file describes it,
//! which manifests it holds — so a unit's key is known before anything is
//! read. The ingest and a conversation's retrieval scope both derive units
//! here, so the two can never disagree on a key.

use std::collections::BTreeMap;

use zend_tools::tools::file::list::LIST_PAGE_ENTRIES;
use zend_vfs::vfs::Tree;
use zend_vfs::Oid;

use super::keys::{dir_key, Listing, Shown};
use super::manifest::is_manifest;
use crate::repo_scan::anchor::ANCHOR_NAMES;
use crate::repo_scan::types::Language;

/// One file of a tree that a layer reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeFile {
    /// Workspace-relative: `candle/zend/src/main.rs`.
    pub path: String,
    pub blob: Oid,
    pub size: u64,
    pub language: Language,
}

/// A file whose content a folder's turns show.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShownFile {
    /// Workspace-relative.
    pub path: String,
    pub blob: Oid,
    pub language: Language,
}

/// One folder as a tree lists it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FolderUnit {
    /// The repository it is in; empty for the workspace's own unit.
    pub repo: String,
    /// Workspace-relative with a trailing `/` (`candle/zend/src/`), or `.`
    /// for the workspace.
    pub dir: String,
    /// How many entries its listing holds.
    pub total: usize,
    /// The entries its listing's first page shows, workspace-relative, a
    /// folder ending in `/`, in the listing's order; for the workspace, each
    /// repository as `name/`.
    pub listed: Vec<String>,
    /// The file describing it — a README, else a crate or module root.
    pub anchor: Option<ShownFile>,
    /// Every manifest directly in it, in path order.
    pub manifests: Vec<ShownFile>,
    /// [`dir_key`] over the above.
    pub key: String,
}

/// The folder units of `repo`'s `tree`, given the files of it the layer
/// reads (`read`, workspace-relative, in path order). A folder holding no
/// file the layer reads has no unit; one that does is keyed by all its
/// listing shows, files the layer does not read and subfolders included.
pub fn folder_units(repo: &str, tree: &Tree, read: &[TreeFile]) -> Vec<FolderUnit> {
    let mut by_dir: BTreeMap<String, Vec<&TreeFile>> = BTreeMap::new();
    for file in read {
        by_dir.entry(dir_of(&file.path)).or_default().push(file);
    }
    let prefix = format!("{repo}/");
    by_dir
        .into_iter()
        .map(|(dir, direct)| {
            let inner = dir
                .strip_prefix(&prefix)
                .unwrap_or("")
                .trim_end_matches('/');
            let children = tree.children(inner);
            let listed: Vec<String> = children
                .iter()
                .take(LIST_PAGE_ENTRIES)
                .map(|(path, _, is_dir)| {
                    let slash = if *is_dir { "/" } else { "" };
                    format!("{repo}/{path}{slash}")
                })
                .collect();
            let anchor = choose_anchor(&direct).map(shown);
            let manifests: Vec<ShownFile> = direct
                .iter()
                .filter(|f| is_manifest(basename(&f.path)))
                .map(|f| shown(f))
                .collect();
            let key = dir_key(
                &dir,
                &Listing {
                    total: children.len(),
                    page: &listed,
                },
                anchor.as_ref().map(as_shown),
                &manifests.iter().map(as_shown).collect::<Vec<_>>(),
            );
            FolderUnit {
                repo: repo.to_string(),
                dir,
                total: children.len(),
                listed,
                anchor,
                manifests,
                key,
            }
        })
        .collect()
}

/// The workspace's own unit: its listing is every repository the workspace
/// holds (`names`, in the workspace's order), as `file_list` over them shows
/// it.
pub fn workspace_unit(names: &[String]) -> FolderUnit {
    let listed: Vec<String> = names
        .iter()
        .take(LIST_PAGE_ENTRIES)
        .map(|r| format!("{r}/"))
        .collect();
    let key = dir_key(
        ".",
        &Listing {
            total: names.len(),
            page: &listed,
        },
        None,
        &[],
    );
    FolderUnit {
        repo: String::new(),
        dir: ".".to_string(),
        total: names.len(),
        listed,
        anchor: None,
        manifests: Vec::new(),
        key,
    }
}

/// The file that describes a folder, by [`ANCHOR_NAMES`]' preference: a
/// README over a crate or module root, and only one directly inside it —
/// never a subfolder's.
fn choose_anchor<'a>(direct: &[&'a TreeFile]) -> Option<&'a TreeFile> {
    ANCHOR_NAMES.iter().find_map(|want| {
        direct
            .iter()
            .find(|f| basename(&f.path).eq_ignore_ascii_case(want))
            .copied()
    })
}

fn shown(file: &TreeFile) -> ShownFile {
    ShownFile {
        path: file.path.clone(),
        blob: file.blob.clone(),
        language: file.language,
    }
}

fn as_shown(file: &ShownFile) -> Shown<'_> {
    Shown {
        path: &file.path,
        blob: &file.blob,
    }
}

/// A workspace-relative file's folder, with a trailing `/`.
pub fn dir_of(path: &str) -> String {
    match path.rfind('/') {
        Some(i) => path[..=i].to_string(),
        None => ".".to_string(),
    }
}

fn basename(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// A tree holding `files` — `(repository-relative path, blob id)` — with
/// every folder above them, and each file's size its blob id's length: what
/// a test that needs a listing without a repository builds.
#[cfg(test)]
pub(crate) fn test_tree(files: &[(&str, &str)]) -> Tree {
    use std::collections::BTreeSet;
    use zend_vfs::{FileMode, ObjectKind, RepoPath, SizedEntry, TreeEntry};

    let entry = |path: &str, mode, kind, oid: &str, size| SizedEntry {
        entry: TreeEntry {
            mode,
            kind,
            oid: Oid::parse(oid).unwrap(),
            path: RepoPath::parse(path).unwrap(),
        },
        size,
    };
    let empty = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";
    let dirs: BTreeSet<&str> = files
        .iter()
        .flat_map(|(path, _)| path.match_indices('/').map(move |(i, _)| &path[..i]))
        .collect();
    let mut listing: Vec<SizedEntry> = dirs
        .into_iter()
        .map(|d| entry(d, FileMode::Tree, ObjectKind::Tree, empty, None))
        .collect();
    for (path, blob) in files {
        listing.push(entry(
            path,
            FileMode::Regular,
            ObjectKind::Blob,
            blob,
            Some(blob.len() as u64),
        ));
    }
    Tree::from_listing(Oid::parse(empty).unwrap(), listing)
}

/// The folder units of one repository holding `files` — workspace-relative,
/// the repository their first segment — the layer reading every one of
/// them: what a test that needs units without a repository builds.
#[cfg(test)]
pub(crate) fn test_units(files: &[(&str, Language)]) -> Vec<FolderUnit> {
    const BLOB: &str = "ce013625030ba8dba906f756967f9e9ca394464a";
    let repo = files[0].0.split('/').next().unwrap();
    let inner: Vec<(&str, &str)> = files
        .iter()
        .map(|(path, _)| (&path[repo.len() + 1..], BLOB))
        .collect();
    let read: Vec<TreeFile> = files
        .iter()
        .map(|(path, language)| TreeFile {
            path: path.to_string(),
            blob: Oid::parse(BLOB).unwrap(),
            size: BLOB.len() as u64,
            language: *language,
        })
        .collect();
    folder_units(repo, &test_tree(&inner), &read)
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "ce013625030ba8dba906f756967f9e9ca394464a";
    const B: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";

    /// The units of repository `r` holding `files` (repository-relative
    /// path, blob), every file read as the language its name says.
    fn units(files: &[(&str, &str)]) -> Vec<FolderUnit> {
        units_reading(files, |_| true)
    }

    /// As [`units`], the layer reading only the files `reads` admits.
    fn units_reading(files: &[(&str, &str)], reads: impl Fn(&str) -> bool) -> Vec<FolderUnit> {
        let tree = test_tree(files);
        let read: Vec<TreeFile> = files
            .iter()
            .filter(|(path, _)| reads(path))
            .map(|(path, blob)| TreeFile {
                path: format!("r/{path}"),
                blob: Oid::parse(blob).unwrap(),
                size: 1,
                language: Language::Rust,
            })
            .collect();
        folder_units("r", &tree, &read)
    }

    fn dirs(units: &[FolderUnit]) -> Vec<&str> {
        units.iter().map(|u| u.dir.as_str()).collect()
    }

    /// **One unit per folder holding a file**, in path order — a folder that
    /// only holds folders has none, and a repository's root is `repo/`. A
    /// listing shows the folder's subfolders as well as its files.
    #[test]
    fn one_unit_per_folder_that_holds_a_file() {
        let units = units(&[("a/b/c/y.rs", A), ("a/x.rs", A), ("top.rs", A)]);
        assert_eq!(dirs(&units), ["r/", "r/a/", "r/a/b/c/"]);
        assert_eq!(units[0].listed, ["r/a/", "r/top.rs"]);
        assert_eq!(units[1].listed, ["r/a/b/", "r/a/x.rs"]);
        assert!(units.iter().all(|u| u.repo == "r"));
    }

    /// **A README describes its folder over a module root**, and a folder is
    /// described only by a file directly inside it.
    #[test]
    fn a_readme_anchors_its_folder_over_a_module_root() {
        let units = units(&[
            ("a/README.md", A),
            ("a/mod.rs", A),
            ("b/lib.rs", A),
            ("b/sub/README.md", A),
            ("c/thing.rs", A),
        ]);
        let anchor = |dir: &str| {
            units
                .iter()
                .find(|u| u.dir == dir)
                .unwrap()
                .anchor
                .as_ref()
                .map(|a| a.path.as_str())
        };
        assert_eq!(anchor("r/a/"), Some("r/a/README.md"));
        assert_eq!(anchor("r/b/"), Some("r/b/lib.rs"));
        assert_eq!(anchor("r/c/"), None);
    }

    /// **The listing is what the turn shows, not what the layer reads**: a
    /// file it does not read and a new subfolder both move the key.
    #[test]
    fn what_the_layer_does_not_read_is_still_in_the_listing() {
        let reads = |p: &str| p.ends_with(".rs");
        let base = units_reading(&[("a/x.rs", A)], reads);
        let with_license = units_reading(&[("a/x.rs", A), ("a/LICENSE", A)], reads);
        let with_subfolder = units_reading(&[("a/x.rs", A), ("a/kernels/k.cu", A)], reads);
        let key = |u: &[FolderUnit]| u.iter().find(|u| u.dir == "r/a/").unwrap().key.clone();
        assert_eq!(with_license[0].listed, ["r/a/LICENSE", "r/a/x.rs"]);
        assert_ne!(key(&base), key(&with_license));
        assert_ne!(key(&base), key(&with_subfolder));
    }

    /// **Only the listing's first page is shown**: an entry past it moves
    /// the count alone, one on it moves the page.
    #[test]
    fn only_the_first_page_is_listed() {
        let names: Vec<String> = (0..LIST_PAGE_ENTRIES)
            .map(|i| format!("a/f{i:03}.rs"))
            .collect();
        let page: Vec<(&str, &str)> = names.iter().map(|n| (n.as_str(), A)).collect();
        let before = units(&page);
        assert_eq!(before[0].listed.len(), LIST_PAGE_ENTRIES);
        assert_eq!(before[0].total, LIST_PAGE_ENTRIES);

        let mut past = page.clone();
        past.push(("a/zz.rs", A));
        let after = units(&past);
        assert_eq!(after[0].listed, before[0].listed, "the page is the same");
        assert_eq!(after[0].total, LIST_PAGE_ENTRIES + 1);
        assert_ne!(after[0].key, before[0].key, "the count is shown too");
    }

    /// **An edited anchor or manifest moves the key; an edited file it only
    /// names does not.**
    #[test]
    fn what_the_folder_shows_moves_its_key() {
        let key = |files: &[(&str, &str)]| units(files)[0].key.clone();
        let base = key(&[("a/Cargo.toml", A), ("a/mod.rs", A), ("a/x.rs", A)]);
        let named_edit = key(&[("a/Cargo.toml", A), ("a/mod.rs", A), ("a/x.rs", B)]);
        let anchor_edit = key(&[("a/Cargo.toml", A), ("a/mod.rs", B), ("a/x.rs", A)]);
        let manifest_edit = key(&[("a/Cargo.toml", B), ("a/mod.rs", A), ("a/x.rs", A)]);
        assert_eq!(base, named_edit);
        assert_ne!(base, anchor_edit);
        assert_ne!(base, manifest_edit);
        assert_eq!(
            units(&[("a/Cargo.toml", A), ("a/x.rs", A)])[0]
                .manifests
                .len(),
            1
        );
    }

    /// **The workspace's unit lists every repository, in the workspace's
    /// order**, and a repository joining moves its key.
    #[test]
    fn the_workspace_unit_lists_the_repositories() {
        let two = workspace_unit(&["mind".into(), "candle".into()]);
        assert_eq!(two.dir, ".");
        assert_eq!(two.listed, ["mind/", "candle/"]);
        let one = workspace_unit(&["mind".into()]);
        assert_ne!(one.key, two.key);
    }
}
