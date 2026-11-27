//! The knowledge-base sections a slot's target holds in its range, by the
//! section each is held as here — what the assembly declares as hit rounds
//! (`docs/placed_section_layout.md` §3.4).

use std::collections::{HashMap, HashSet};

use crate::files::names::chunk_name;
use crate::files::{files_owner, FileRecord, SectionKey, FILES_PREFIX};
use crate::projection::{ProjectionTarget, SectionId};
use crate::substrate::Substrate;

/// Every section `target`'s files set holds in its range, by the section id
/// this process holds it as; empty for a slot with no target, no range, or
/// nothing placed.
pub(in super::super) fn kb_sections(
    conv: &Substrate,
    target: Option<ProjectionTarget>,
) -> HashMap<SectionId, SectionKey> {
    let Some(target) = target else {
        return HashMap::new();
    };
    let Some(scope) = conv
        .files_set_of(target.timeline)
        .and_then(|s| s.scope.as_ref())
    else {
        return HashMap::new();
    };
    let wanted: HashSet<SectionKey> = scope.sections().map(|e| e.key).collect();
    if wanted.is_empty() {
        return HashMap::new();
    }
    let mut out = HashMap::with_capacity(wanted.len());
    for (obj, _) in conv.custom_objects_under(FILES_PREFIX) {
        let Some(record) = FileRecord::from_object(&obj) else {
            continue;
        };
        for (i, c) in record.chunks.iter().enumerate() {
            let key = record.section_key(i);
            if !wanted.contains(&key) {
                continue;
            }
            let name = chunk_name(&record.path, &record.blob, c.lines);
            if let Some(id) = conv.owned_section_named(files_owner(), &name) {
                out.insert(id, key);
            }
        }
    }
    out
}
