//! One tree's ingested units: every file of it, and every folder unit the
//! folder layer derives from it, looked up by key.

use std::collections::HashMap;

use candle_conversation::projection::TimelineId;
use zend_vfs::vfs::Tree;

use super::index::IngestIndex;
use crate::branch_ingest::filter::IngestScope;
use crate::branch_ingest::keys::file_key;
use crate::branch_ingest::units::{folder_units, TreeFile};

/// One tree's units that have been ingested: files by path (repository
/// -relative), folders by folder (workspace-relative).
#[derive(Debug, Default)]
pub struct TreeScope {
    pub files: HashMap<String, TimelineId>,
    pub folders: HashMap<String, TimelineId>,
}

impl TreeScope {
    /// The ingested units of `repo`'s `tree`: every file looked up by its
    /// key, and — under `folder_scope`, the scope the folder layer walked —
    /// every folder unit derived exactly as the walk derives it.
    pub fn of(
        index: &IngestIndex,
        repo: &str,
        tree: &Tree,
        folder_scope: Option<&IngestScope>,
    ) -> Self {
        let mut out = Self::default();
        for (path, blob, _) in tree.files() {
            if let Some(tl) = index.get(&file_key(&format!("{repo}/{path}"), blob)) {
                out.files.insert(path.to_string(), tl);
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
        for unit in folder_units(repo, tree, &read) {
            if let Some(tl) = index.get(&unit.key) {
                out.folders.insert(unit.dir, tl);
            }
        }
        out
    }
}
