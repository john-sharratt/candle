//! Every committed ingest conversation, by content key — what a scope looks
//! a unit's key up in.

use std::collections::{HashMap, HashSet};

use candle_conversation::projection::TimelineId;
use candle_conversation::ConversationEngine;

use crate::branch_ingest::keys::CONTENT_KEY;
use crate::code_read::{is_upload_path, PATH_KEY};

/// Every committed ingest conversation, by content key.
#[derive(Debug, Default)]
pub struct IngestIndex {
    by_key: HashMap<String, TimelineId>,
    /// Every committed upload's conversation: uploads are in every scope.
    uploads: Vec<TimelineId>,
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
        Self {
            by_key,
            uploads,
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
            generation,
        }
    }

    /// The conversation committed under `key`.
    pub fn get(&self, key: &str) -> Option<TimelineId> {
        self.by_key.get(key).copied()
    }

    /// Every committed upload's conversation.
    pub fn uploads(&self) -> &[TimelineId] {
        &self.uploads
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}
