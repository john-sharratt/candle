//! One tree's ingested units: every file of it, and every folder unit the
//! folder layer derives from it, looked up by key.

use std::collections::HashMap;

use candle_conversation::projection::{GroupId, TimelineId};
use zend_vfs::vfs::Tree;

use super::index::IngestIndex;
use crate::branch_ingest::filter::IngestScope;
use crate::branch_ingest::keys::{
    dir_key_at, file_key, unversioned, Listing, FIRST_VERSIONED, KEY_VERSION,
};
use crate::branch_ingest::units::{folder_units, FolderUnit, TreeFile};
use crate::repo_scan::types::ModuleHint;

/// One tree's units that have been ingested: files by path (repository
/// -relative), folders by folder (workspace-relative).
#[derive(Debug, Default)]
pub struct TreeScope {
    pub files: HashMap<String, TimelineId>,
    pub folders: HashMap<String, TimelineId>,
    /// Each unit of a layer of files of a file version this tree holds, built
    /// against the versions of its dependencies this tree holds.
    pub derived: Vec<DerivedScope>,
}

/// A unit of a layer of files in a tree's scope.
#[derive(Debug, Clone)]
pub struct DerivedScope {
    /// Its file, repository-relative.
    pub path: String,
    pub group: GroupId,
    pub timeline: TimelineId,
    /// The files of this repository it depends on, repository-relative: a
    /// conversation that changes one has the analysis out of scope too.
    pub deps: Vec<String>,
}

impl TreeScope {
    /// The ingested units of `repo`'s `tree`: every file looked up by its
    /// key, and — under `folder_scope`, the scope the folder layer walked —
    /// every folder unit derived exactly as the walk derives it, its
    /// manifest's hint found through `hint_of` as the walk finds it.
    pub fn of(
        index: &IngestIndex,
        repo: &str,
        tree: &Tree,
        folder_scope: Option<&IngestScope>,
        hint_of: &mut dyn FnMut(&TreeFile) -> Option<ModuleHint>,
    ) -> Self {
        let mut out = Self::default();
        let keys: HashMap<&str, String> = tree
            .files()
            .map(|(path, blob, _)| (path, file_key(&format!("{repo}/{path}"), blob)))
            .collect();
        let prefix = format!("{repo}/");
        for (path, key) in &keys {
            if let Some(tl) = index.get(key) {
                out.files.insert(path.to_string(), tl);
            }
            for unit in index.derived_of(key) {
                if let Some(deps) = built_against(&unit.deps, &prefix, &keys) {
                    out.derived.push(DerivedScope {
                        path: path.to_string(),
                        group: unit.group,
                        timeline: unit.timeline,
                        deps,
                    });
                }
            }
        }
        let Some(scope) = folder_scope else {
            return out;
        };
        let read: Vec<TreeFile> = tree
            .files()
            .filter_map(|(path, blob, size)| {
                Some(TreeFile {
                    path: format!("{repo}/{path}"),
                    blob: blob.clone(),
                    size,
                    language: scope.admits(repo, path, size)?,
                })
            })
            .collect();
        for unit in folder_units(repo, tree, &read, hint_of) {
            if let Some(tl) = folder_reading(index, &unit) {
                out.folders.insert(unit.dir, tl);
            }
        }
        out
    }
}

/// The reading of `unit`'s folder: under the current key version, or — while
/// a key version change is re-read — under the newest earlier one that read
/// it. A folder's digest covers its version, so an earlier reading is found by
/// its own version's key, recomputed, never by the current one.
fn folder_reading(index: &IngestIndex, unit: &FolderUnit) -> Option<TimelineId> {
    let listing = Listing {
        total: unit.total,
        page: &unit.listed,
    };
    (FIRST_VERSIONED..=KEY_VERSION).rev().find_map(|version| {
        index.get(&dir_key_at(
            version,
            &unit.dir,
            &listing,
            unit.module_hint.as_ref(),
        ))
    })
}

/// Whether a unit depending on `deps` (workspace-relative, each at its
/// `code_reading` key) was built against the versions a tree of the
/// repository `prefix` names holds — `keys`, repository-relative path → key.
/// An analysis split by what its file depends on is in scope only where its
/// dependencies are as it saw them; one in another repository is not this
/// tree's to judge. Answers the dependencies in this repository,
/// repository-relative, or `None` when one differs.
fn built_against(
    deps: &[(String, String)],
    prefix: &str,
    keys: &HashMap<&str, String>,
) -> Option<Vec<String>> {
    let mut here = Vec::new();
    for (dep, key) in deps {
        if let Some(rel) = dep.strip_prefix(prefix) {
            // The file version, whatever key version read it.
            if keys.get(rel).map(|k| unversioned(k)) != Some(unversioned(key)) {
                return None;
            }
            here.push(rel.to_string());
        }
    }
    Some(here)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_unit_is_in_a_tree_built_against_its_dependencies() {
        let keys = HashMap::from([("src/d.rs", "d@1".to_string())]);
        let dep = |p: &str, k: &str| (p.to_string(), k.to_string());
        assert_eq!(
            built_against(&[dep("r/src/d.rs", "d@1")], "r/", &keys),
            Some(vec!["src/d.rs".to_string()])
        );
        assert_eq!(
            built_against(&[dep("r/src/d.rs", "d@2")], "r/", &keys),
            None
        );
        assert_eq!(
            built_against(&[dep("r/src/gone.rs", "g@1")], "r/", &keys),
            None
        );
        // Another repository's file is not this tree's to judge.
        assert_eq!(
            built_against(&[dep("other/x.rs", "x@9")], "r/", &keys),
            Some(vec![])
        );
        // A dependency read under an earlier key version is the same file
        // version.
        let keys = HashMap::from([("src/d.rs", "v3:r/src/d.rs@1".to_string())]);
        assert_eq!(
            built_against(&[dep("r/src/d.rs", "v2:r/src/d.rs@1")], "r/", &keys),
            Some(vec!["src/d.rs".to_string()])
        );
        assert_eq!(
            built_against(&[dep("r/src/d.rs", "v2:r/src/d.rs@2")], "r/", &keys),
            None
        );
    }

    /// **A folder not yet re-read is found by its own version's key.**
    #[test]
    fn a_folder_reading_from_an_earlier_version_is_found() {
        let unit = FolderUnit {
            repo: "r".into(),
            dir: "r/a/".into(),
            total: 1,
            listed: vec!["r/a/x.rs".into()],
            module_hint: None,
            key: String::new(),
        };
        let listing = Listing {
            total: 1,
            page: &unit.listed,
        };
        let tl = |n| TimelineId::from_raw(n).unwrap();
        let old = dir_key_at(KEY_VERSION - 1, "r/a/", &listing, None);
        let index = IngestIndex::of(HashMap::from([(old, tl(1))]), Vec::new(), 0);
        assert_eq!(folder_reading(&index, &unit), Some(tl(1)));
        let new = dir_key_at(KEY_VERSION, "r/a/", &listing, None);
        let old = dir_key_at(KEY_VERSION - 1, "r/a/", &listing, None);
        let index = IngestIndex::of(HashMap::from([(old, tl(1)), (new, tl(2))]), Vec::new(), 0);
        assert_eq!(folder_reading(&index, &unit), Some(tl(2)), "the newest");
        let none = IngestIndex::of(HashMap::new(), Vec::new(), 0);
        assert_eq!(folder_reading(&none, &unit), None);
    }
}
