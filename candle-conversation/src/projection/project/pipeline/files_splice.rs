//! The knowledge base's sections placed into the system prompt
//! (`docs/provenance_system_collection.md` §3, §4;
//! `docs/placed_section_layout.md` §3.5c).
//!
//! Each kind's sections are chosen under the kind's share of the room
//! ([`choose`]), strongest first. The sections of every kind then sit in the
//! order the conversation's files set keeps: at the opening strongest first,
//! afterwards every kept section where it was and every newcomer after them.
//! Every section carries its own heading, closing line and, short of its
//! file's end, the cut marker, so nothing is placed between them.
//!
//! A free knowledge base the seed dial leaves on opens on the mandatory files
//! (§3.5b), whole and ahead of the chosen sections, outside every kind's cap;
//! a pinned one holds its pin alone.

use std::collections::HashSet;

use super::*;
use crate::files::fill::choose;
use crate::files::names::chunk_name;
use crate::files::{
    mandatory_records, FileKey, FileRecord, FilesPin, LineSpan, PlacedFile, SectionKey,
};

/// The sections a projection places and the segments they are placed as.
pub(crate) struct FilesSplice {
    /// The placed sections as runs of one file's, in the order they sit,
    /// each run with its sections' lines.
    pub placed: Vec<PlacedFile>,
    /// The placed sections, in the order they sit.
    pub sections: Vec<SectionKey>,
    pub segments: Vec<ProjectionSegment>,
}

#[cfg(test)]
impl FilesSplice {
    /// The placed runs' files, in the order they sit.
    pub(crate) fn order(&self) -> Vec<FileKey> {
        self.placed.iter().map(|p| p.file).collect()
    }
}

/// `sections`, in the order they sit, as runs of one file's: a file's
/// sections placed one after another are one run.
fn runs(sections: &[(FileKey, LineSpan)]) -> Vec<PlacedFile> {
    let mut out: Vec<PlacedFile> = Vec::new();
    for &(file, lines) in sections {
        match out.last_mut() {
            Some(run) if run.file == file => run.chunks.push(lines),
            _ => out.push(PlacedFile {
                file,
                chunks: vec![lines],
            }),
        }
    }
    out
}

/// The sections to place under `kind_cap` — each working-set group of
/// `schema` a kind, its cap the tokens its share allows — and the segments
/// they are placed as; with `mandatory`, a free knowledge base opens on the
/// mandatory files whole (`files::mandatory`), outside every cap. Empty when
/// the target has no files set; a section whose member this process does not
/// hold is left out.
pub(crate) fn files_splice<R: ContentResolver>(
    schema: &Schema,
    resolver: &R,
    kind_cap: impl Fn(WorkingSetShare) -> usize,
    mandatory: bool,
) -> FilesSplice {
    let Some(set) = resolver.files_set() else {
        return FilesSplice {
            placed: Vec::new(),
            sections: Vec::new(),
            segments: Vec::new(),
        };
    };
    let all = resolver.files_records();
    // Every section of every record: its record and its place in it.
    let index: HashMap<SectionKey, (&FileRecord, usize)> = all
        .iter()
        .flat_map(|r| (0..r.chunks.len()).map(move |i| (r.section_key(i), (r, i))))
        .collect();
    // The mandatory files' sections, in line order, then each kind's choice
    // ranked by momentum — or, pinned, the pinned sections in the pin's
    // order, or nothing.
    let mut chosen: Vec<SectionKey> = Vec::new();
    match &set.pin {
        FilesPin::Empty => {}
        FilesPin::Only(sections) => chosen.extend(sections.iter().copied()),
        FilesPin::Free => {
            let paths = match mandatory {
                true => resolver.files_mandatory(),
                false => Vec::new(),
            };
            for record in mandatory_records(&all, &paths) {
                chosen.extend(record.section_keys());
            }
            // A path the mandatory files hold is no candidate of any kind.
            let held: HashSet<&str> = paths.iter().map(String::as_str).collect();
            let mut picked: Vec<(SectionKey, f32)> = Vec::new();
            for group in schema
                .layers
                .iter()
                .flat_map(|l| l.groups.iter())
                .filter(|g| g.is_working_set())
            {
                let SelectionRule::WorkingSet { share } = group.selection else {
                    continue;
                };
                let ranked: Vec<(SectionKey, f32)> = set
                    .ranked(group.id)
                    .into_iter()
                    .filter(|(k, _)| {
                        index
                            .get(k)
                            .is_some_and(|(r, _)| !held.contains(r.path.as_str()))
                    })
                    .collect();
                let momentum: HashMap<SectionKey, f32> = ranked.iter().copied().collect();
                let tokens_of = |k: SectionKey| {
                    index
                        .get(&k)
                        .map(|(r, i)| r.chunks[*i].member.tokens as usize)
                };
                for s in choose(&ranked, tokens_of, kind_cap(share)) {
                    picked.push((s, momentum.get(&s).copied().unwrap_or(0.0)));
                }
            }
            picked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            chosen.extend(picked.into_iter().map(|(s, _)| s));
        }
    }
    // Each section in the order it sits, its member resolved; one this
    // process does not hold is not placed.
    let resolved: Vec<(SectionKey, FileKey, LineSpan, SectionId)> = set
        .order(&chosen)
        .into_iter()
        .filter_map(|key| {
            let &(record, i) = index.get(&key)?;
            let lines = record.chunks[i].lines;
            let id = resolver.member_section(&chunk_name(&record.path, &record.blob, lines))?;
            Some((key, record.key(), lines, id))
        })
        .collect();
    let spans: Vec<(FileKey, LineSpan)> = resolved.iter().map(|&(_, f, l, _)| (f, l)).collect();
    FilesSplice {
        placed: runs(&spans),
        sections: resolved.iter().map(|&(k, ..)| k).collect(),
        segments: resolved
            .iter()
            .map(|&(.., id)| {
                ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection::new(id)))
            })
            .collect(),
    }
}
