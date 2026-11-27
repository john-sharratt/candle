//! The knowledge-base sections a slot's target holds, by the section each is
//! held as here: the placed ones the assembly declares as hit rounds, and the
//! ones that may enter, for residency (`docs/placed_section_layout.md` §3.4,
//! §3.5).

use std::collections::{HashMap, HashSet};

use crate::files::names::chunk_name;
use crate::files::{files_owner, FileKey, FileRecord, LineSpan, SectionKey, FILES_PREFIX};
use crate::projection::{ProjectionTarget, SectionId};
use crate::substrate::Substrate;

/// One section as this process holds it: its member, its tokens, and the
/// file and lines it is of.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in super::super) struct HeldSection {
    pub(in super::super) id: SectionId,
    pub(in super::super) tokens: usize,
    pub(in super::super) file: FileKey,
    pub(in super::super) lines: LineSpan,
}

/// Each of `wanted` this process holds a member for, by its key.
pub(in super::super) fn held_sections(
    conv: &Substrate,
    wanted: &HashSet<SectionKey>,
) -> HashMap<SectionKey, HeldSection> {
    let mut out = HashMap::with_capacity(wanted.len());
    if wanted.is_empty() {
        return out;
    }
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
                out.insert(
                    key,
                    HeldSection {
                        id,
                        tokens: c.member.tokens as usize,
                        file: record.key(),
                        lines: c.lines,
                    },
                );
            }
        }
    }
    out
}

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
    held_sections(conv, &wanted)
        .into_iter()
        .map(|(key, held)| (held.id, key))
        .collect()
}
