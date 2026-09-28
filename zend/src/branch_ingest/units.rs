//! The `repo_map` layer's units as one tree lists them: one per folder
//! holding files the layer reads, plus the workspace's own. Derived from what
//! the folder's listing shows and the hint its manifest gives — the one file
//! read, once per manifest version (`manifest::Hints`) — so a unit's key is
//! known before any conversation runs. The ingest and a conversation's
//! retrieval scope both derive units here, so the two can never disagree on a
//! key.

use std::collections::BTreeMap;

use zend_tools::tools::file::list::LIST_PAGE_ENTRIES;
use zend_vfs::vfs::Tree;
use zend_vfs::Oid;

use super::keys::{dir_key, Listing};
use super::manifest::is_manifest;
use crate::repo_scan::types::{Language, ModuleHint};

/// One file of a tree that a layer reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TreeFile {
    /// Workspace-relative: `candle/zend/src/main.rs`.
    pub path: String,
    pub blob: Oid,
    pub size: u64,
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
    /// The hint its request carries: the first of its manifests, in path
    /// order, that gives one.
    pub module_hint: Option<ModuleHint>,
    /// [`dir_key`] over the above.
    pub key: String,
}

/// The folder units of `repo`'s `tree`, given the files of it the layer
/// reads (`read`, workspace-relative, in path order). A folder holding no
/// file the layer reads has no unit; one that does is keyed by all its
/// listing shows, files the layer does not read and subfolders included, and
/// by the hint `hint_of` finds in its manifests.
pub fn folder_units(
    repo: &str,
    tree: &Tree,
    read: &[TreeFile],
    hint_of: &mut dyn FnMut(&TreeFile) -> Option<ModuleHint>,
) -> Vec<FolderUnit> {
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
            let module_hint = direct
                .iter()
                .filter(|f| is_manifest(basename(&f.path)))
                .find_map(|f| hint_of(f));
            let key = dir_key(
                &dir,
                &Listing {
                    total: children.len(),
                    page: &listed,
                },
                module_hint.as_ref(),
            );
            FolderUnit {
                repo: repo.to_string(),
                dir,
                total: children.len(),
                listed,
                module_hint,
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
    );
    FolderUnit {
        repo: String::new(),
        dir: ".".to_string(),
        total: names.len(),
        listed,
        module_hint: None,
        key,
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
/// them, and no manifest giving a hint: what a test that needs units without
/// a repository builds.
#[cfg(test)]
pub(crate) fn test_units(files: &[(&str, Language)]) -> Vec<FolderUnit> {
    test_units_reading(files, |_| None)
}

/// As [`test_units`], a manifest's bytes read by its workspace-relative path
/// through `bytes`.
#[cfg(test)]
pub(crate) fn test_units_reading(
    files: &[(&str, Language)],
    bytes: impl Fn(&str) -> Option<Vec<u8>>,
) -> Vec<FolderUnit> {
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
    folder_units(repo, &test_tree(&inner), &read, &mut |f: &TreeFile| {
        super::manifest::hint(basename(&f.path), &bytes(&f.path)?)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "ce013625030ba8dba906f756967f9e9ca394464a";
    const B: &str = "4b825dc642cb6eb9a060e54bf8d69288fbee4904";
    /// A third blob: a `Cargo.toml` giving a different hint from `A`'s.
    const C: &str = "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391";

    /// What each blob holds when it is a `Cargo.toml`: `A` and `B` two
    /// versions of one crate's manifest — a version raised, the hint the
    /// same — and `C` a workspace root.
    fn manifest_bytes(blob: &Oid) -> Option<Vec<u8>> {
        let text: &[u8] = match blob.as_str() {
            A => b"[package]\nname = \"demo\"\nversion = \"0.1.0\"\n",
            B => b"[package]\nname = \"demo\"\nversion = \"0.2.0\"\n",
            C => b"[workspace]\nmembers = [\"a\"]\n",
            _ => return None,
        };
        Some(text.to_vec())
    }

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
        let mut hints = super::super::manifest::Hints::default();
        folder_units("r", &tree, &read, &mut |f: &TreeFile| {
            hints.of(basename(&f.path), &f.blob, manifest_bytes)
        })
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

    /// **Only a change to what the turns show moves the key.** An edited file
    /// the folder only names does not — a module root and a README included,
    /// since the turns never read a file of it — and nor does a manifest edit
    /// that leaves its hint as it was: the request reads the same. A manifest
    /// edit that changes the hint does.
    ///
    /// The second case is the one that matters for density: keyed on the
    /// manifest's bytes, every `Cargo.toml` version bump split the folder into
    /// another conversation saying the same thing — `candle/` stood ten times
    /// over seven distinct listings across its branches.
    #[test]
    fn what_the_folder_shows_moves_its_key() {
        let key = |files: &[(&str, &str)]| units(files)[0].key.clone();
        let base = key(&[("a/Cargo.toml", A), ("a/mod.rs", A), ("a/README.md", A)]);
        let module_root_edit = key(&[("a/Cargo.toml", A), ("a/mod.rs", B), ("a/README.md", A)]);
        let readme_edit = key(&[("a/Cargo.toml", A), ("a/mod.rs", A), ("a/README.md", B)]);
        let same_hint = key(&[("a/Cargo.toml", B), ("a/mod.rs", A), ("a/README.md", A)]);
        let new_hint = key(&[("a/Cargo.toml", C), ("a/mod.rs", A), ("a/README.md", A)]);
        assert_eq!(base, module_root_edit);
        assert_eq!(base, readme_edit);
        assert_eq!(base, same_hint, "a version bump gives the same request");
        assert_ne!(base, new_hint);
        assert_eq!(
            units(&[("a/Cargo.toml", A), ("a/x.rs", A)])[0].module_hint,
            Some(ModuleHint::CargoPackage {
                name: "demo".into()
            })
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
