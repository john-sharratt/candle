//! Every committed ingest conversation, by content key — what a scope looks
//! a unit's key up in.

use std::collections::{HashMap, HashSet};

use candle_conversation::projection::{GroupId, TimelineId};
use candle_conversation::ConversationEngine;

use crate::branch_ingest::keys::{key_version, unversioned, CONTENT_KEY};
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
///
/// **Looked up without the key's version.** A
/// [`KEY_VERSION`](crate::branch_ingest::keys::KEY_VERSION) change re-keys
/// every unit, and its readings are replaced one by one over a whole pass: a
/// unit read under the old version stands until its replacement commits, as
/// the plan keeps it (`branch_ingest::plan`). Looked up by the current
/// version's key alone, every unit not yet re-read fell out of every scope for
/// the length of the pass — a dialogue's projection, and a calibration draft
/// forked from the reading, came up with no file in it. So a key is filed
/// under its [`unversioned`] form, and the newest version of a unit present is
/// the one found.
#[derive(Debug, Default)]
pub struct IngestIndex {
    /// Unversioned key → the newest version committed under it.
    by_key: HashMap<String, (u32, TimelineId)>,
    /// Every committed upload's conversation: uploads are in every scope.
    uploads: Vec<TimelineId>,
    /// Every committed unit of a layer of files, by the unversioned
    /// `code_reading` key of the file version it is of, with the group it is
    /// in — only those built on the newest reading of it present.
    derived: HashMap<String, (u32, Vec<DerivedUnit>)>,
    /// Bumped on every rebuild, so what was worked out against an older
    /// index is never served against this one.
    generation: u64,
}

impl IngestIndex {
    /// The index as the substrate holds it now, at `generation`.
    pub fn read(engine: &ConversationEngine, generation: u64) -> Self {
        let mut by_key: HashMap<String, (u32, TimelineId)> = HashMap::new();
        let mut committed: HashSet<TimelineId> = HashSet::new();
        for (timeline, key) in engine.conversations_with_metadata_key(CONTENT_KEY) {
            committed.insert(timeline);
            file_newest(&mut by_key, &key, timeline);
        }
        let uploads = engine
            .conversations_with_metadata_key(PATH_KEY)
            .into_iter()
            .filter(|(tl, path)| is_upload_path(path) && committed.contains(tl))
            .map(|(tl, _)| tl)
            .collect();
        let mut derived: HashMap<String, (u32, Vec<DerivedUnit>)> = HashMap::new();
        for (timeline, file_key) in engine.conversations_with_metadata_key(FILE_KEY) {
            if !committed.contains(&timeline) {
                continue;
            }
            if let Some((_, group)) = engine.timeline_target(timeline) {
                let deps = engine
                    .conversation_metadata(timeline)
                    .and_then(|m| m.get(DEPS_KEY).map(|v| parse_deps(v)))
                    .unwrap_or_default();
                let unit = DerivedUnit {
                    group,
                    timeline,
                    deps,
                };
                derived_newest(&mut derived, &file_key, unit);
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
        let mut index = HashMap::new();
        for (key, timeline) in by_key {
            file_newest(&mut index, &key, timeline);
        }
        Self {
            by_key: index,
            uploads,
            derived: HashMap::new(),
            generation,
        }
    }

    /// The conversation committed under `key` — under any key version, the
    /// newest one present.
    pub fn get(&self, key: &str) -> Option<TimelineId> {
        self.by_key.get(unversioned(key)).map(|&(_, tl)| tl)
    }

    /// Every committed unit of a layer of files of the file version `file_key`
    /// — those built on its newest reading present.
    pub fn derived_of(&self, file_key: &str) -> &[DerivedUnit] {
        self.derived
            .get(unversioned(file_key))
            .map(|(_, units)| units.as_slice())
            .unwrap_or(&[])
    }

    /// Every committed upload's conversation.
    pub fn uploads(&self) -> &[TimelineId] {
        &self.uploads
    }

    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// File `timeline`, committed under `key`, unless a newer version of its
/// unversioned key is already filed.
fn file_newest(index: &mut HashMap<String, (u32, TimelineId)>, key: &str, timeline: TimelineId) {
    let version = key_version(key);
    let slot = index
        .entry(unversioned(key).to_string())
        .or_insert((version, timeline));
    if version > slot.0 {
        *slot = (version, timeline);
    }
}

/// File `unit`, of the reading `file_key`, beside the units of the newest
/// reading of its file version: a newer reading's units replace an older's,
/// and an older's are not added beside a newer's.
fn derived_newest(
    index: &mut HashMap<String, (u32, Vec<DerivedUnit>)>,
    file_key: &str,
    unit: DerivedUnit,
) {
    let version = key_version(file_key);
    let slot = index
        .entry(unversioned(file_key).to_string())
        .or_insert((version, Vec::new()));
    if version > slot.0 {
        *slot = (version, Vec::new());
    }
    if version == slot.0 {
        slot.1.push(unit);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tl(n: u64) -> TimelineId {
        TimelineId::from_raw(n).unwrap()
    }

    /// **A unit not yet re-read is found under the current version's key**,
    /// and once its replacement commits the replacement is.
    #[test]
    fn a_unit_is_found_under_any_version_and_the_newest_wins() {
        let old = HashMap::from([("v2:r/a.rs@1".to_string(), tl(1))]);
        let index = IngestIndex::of(old, Vec::new(), 0);
        assert_eq!(index.get("v3:r/a.rs@1"), Some(tl(1)));
        assert_eq!(
            index.get("v3:r/a.rs@2"),
            None,
            "another blob is another file"
        );

        let both = HashMap::from([
            ("v2:r/a.rs@1".to_string(), tl(1)),
            ("v3:r/a.rs@1".to_string(), tl(2)),
        ]);
        let index = IngestIndex::of(both, Vec::new(), 0);
        assert_eq!(index.get("v3:r/a.rs@1"), Some(tl(2)));
        assert_eq!(index.get("v2:r/a.rs@1"), Some(tl(2)));
    }

    /// The units of a layer built on a file's newest reading are the ones in
    /// scope; those built on an older reading stand only until then.
    #[test]
    fn derived_units_follow_the_newest_reading() {
        let unit = |n| DerivedUnit {
            group: GroupId::from_raw(1).unwrap(),
            timeline: tl(n),
            deps: Vec::new(),
        };
        let mut index = HashMap::new();
        derived_newest(&mut index, "v2:r/a.rs@1", unit(1));
        derived_newest(&mut index, "v2:r/a.rs@1", unit(2));
        assert_eq!(index["r/a.rs@1"].1.len(), 2);
        derived_newest(&mut index, "v3:r/a.rs@1", unit(3));
        derived_newest(&mut index, "v2:r/a.rs@1", unit(4));
        let units: Vec<TimelineId> = index["r/a.rs@1"].1.iter().map(|u| u.timeline).collect();
        assert_eq!(units, vec![tl(3)]);
    }
}
