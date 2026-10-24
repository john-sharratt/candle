//! Every committed ingest conversation, by content key — what a scope looks
//! a unit's key up in.

use std::collections::{HashMap, HashSet};

use candle_conversation::projection::{GroupId, TimelineId};
use candle_conversation::ConversationEngine;

use crate::branch_ingest::keys::{file_key_path, CONTENT_KEY};
use crate::code_read::{is_upload_path, PATH_KEY};
use crate::layers::keys::{parse_deps, DEPS_KEY, FILE_KEY};

/// A committed unit of a layer of files.
#[derive(Debug, Clone)]
pub struct DerivedUnit {
    pub group: GroupId,
    pub timeline: TimelineId,
    /// Each file it depends on (workspace-relative), at the `code_reading`
    /// key of the version it was built against.
    pub deps: Vec<(String, String)>,
}

/// Every committed ingest conversation, by content key.
#[derive(Debug, Default)]
pub struct IngestIndex {
    by_key: HashMap<String, TimelineId>,
    /// Every committed upload's conversation: uploads are in every scope.
    uploads: Vec<TimelineId>,
    /// Every committed unit of a layer of files, by the `code_reading` key of
    /// the file version it is of, with the group it is in.
    derived: HashMap<String, Vec<DerivedUnit>>,
    /// Bumped on every rebuild, so what was worked out against an older
    /// index is never served against this one.
    generation: u64,
}

impl IngestIndex {
    /// The index as the substrate holds it now, at `generation`.
    pub fn read(engine: &ConversationEngine, generation: u64) -> Self {
        let mut by_key = HashMap::new();
        for (timeline, key) in engine.conversations_with_metadata_key(CONTENT_KEY) {
            by_key.entry(key).or_insert(timeline);
        }
        let committed: HashSet<TimelineId> = by_key.values().copied().collect();
        let uploads = engine
            .conversations_with_metadata_key(PATH_KEY)
            .into_iter()
            .filter(|(tl, path)| is_upload_path(path) && committed.contains(tl))
            .map(|(tl, _)| tl)
            .collect();
        let mut derived: HashMap<String, Vec<DerivedUnit>> = HashMap::new();
        for (timeline, file_key) in engine.conversations_with_metadata_key(FILE_KEY) {
            if !committed.contains(&timeline) {
                continue;
            }
            if let Some((_, group)) = engine.timeline_target(timeline) {
                let deps = engine
                    .conversation_metadata(timeline)
                    .and_then(|m| m.get(DEPS_KEY).map(|v| parse_deps(v)))
                    .unwrap_or_default();
                derived.entry(file_key).or_default().push(DerivedUnit {
                    group,
                    timeline,
                    deps,
                });
            }
        }
        Self {
            by_key,
            uploads,
            derived,
            generation,
        }
    }

    /// An index holding exactly `by_key` and `uploads`.
    #[cfg(test)]
    pub(crate) fn of(
        by_key: HashMap<String, TimelineId>,
        uploads: Vec<TimelineId>,
        generation: u64,
    ) -> Self {
        Self {
            by_key,
            uploads,
            derived: HashMap::new(),
            generation,
        }
    }

    /// The conversation committed under `key`.
    pub fn get(&self, key: &str) -> Option<TimelineId> {
        self.by_key.get(key).copied()
    }

    /// Every committed unit of a layer of files of the file version `file_key`.
    pub fn derived_of(&self, file_key: &str) -> &[DerivedUnit] {
        self.derived.get(file_key).map(Vec::as_slice).unwrap_or(&[])
    }

    /// Every committed upload's conversation.
    pub fn uploads(&self) -> &[TimelineId] {
        &self.uploads
    }

    /// Every committed file reading, with the workspace-relative path it read.
    pub fn files(&self) -> impl Iterator<Item = (&str, TimelineId)> + '_ {
        self.by_key
            .iter()
            .filter_map(|(key, &tl)| file_key_path(key).map(|path| (path, tl)))
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}
