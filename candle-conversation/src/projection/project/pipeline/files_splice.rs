//! The knowledge base's files placed into the system prompt
//! (`docs/provenance_system_collection.md` §3, §4).
//!
//! Each kind's files are chosen under the kind's share of the room
//! ([`choose`]), strongest first. The files of every kind then sit in the
//! order the conversation's files set keeps: at the opening strongest first,
//! afterwards every kept file where it was and every newcomer after them. Each
//! file is its placed sections in line order; every section carries its own
//! heading, closing line and, short of the file's end, the cut marker
//! (`docs/placed_section_layout.md` §3.5c), so nothing is placed between them.

use super::*;
use crate::files::fill::{choose, FilePlacement};
use crate::files::names::chunk_name;
use crate::files::{FileKey, FileRecord, FilesPin, LineSpan, PlacedFile};

/// The files a projection places and the segments they are placed as.
pub(crate) struct FilesSplice {
    /// The placed files, in the order they sit, each with its placed chunks'
    /// lines.
    pub placed: Vec<PlacedFile>,
    pub segments: Vec<ProjectionSegment>,
}

#[cfg(test)]
impl FilesSplice {
    /// The placed files' keys, in the order they sit.
    pub(crate) fn order(&self) -> Vec<FileKey> {
        self.placed.iter().map(|p| p.file).collect()
    }
}

/// The files to place under `kind_cap` — each working-set group of `schema`
/// a kind, its cap the tokens its share allows — and the segments they are
/// placed as. Empty when the target has no files set, or no file's members
/// are held in this process.
pub(crate) fn files_splice<R: ContentResolver>(
    schema: &Schema,
    resolver: &R,
    kind_cap: impl Fn(WorkingSetShare) -> usize,
) -> FilesSplice {
    let Some(set) = resolver.files_set() else {
        return FilesSplice {
            placed: Vec::new(),
            segments: Vec::new(),
        };
    };
    let records: HashMap<FileKey, FileRecord> = resolver
        .files_records()
        .into_iter()
        .map(|r| (r.key(), r))
        .collect();
    // Each kind's choice, then every chosen file ranked by its momentum — or,
    // pinned, the pinned files whole, in the pin's order, or nothing.
    let mut chosen: Vec<(FilePlacement, f32)> = Vec::new();
    match &set.pin {
        FilesPin::Empty => {}
        FilesPin::Only(files) => {
            for &file in files {
                if let Some(record) = records.get(&file) {
                    let chunks = (0..record.chunks.len()).collect();
                    chosen.push((FilePlacement { file, chunks }, 0.0));
                }
            }
        }
        FilesPin::Free => {
            for group in schema
                .layers
                .iter()
                .flat_map(|l| l.groups.iter())
                .filter(|g| g.is_working_set())
            {
                let SelectionRule::WorkingSet { share } = group.selection else {
                    continue;
                };
                let ranked = set.ranked(group.id);
                let momentum: HashMap<FileKey, f32> = ranked.iter().copied().collect();
                for p in choose(&ranked, |k| records.get(&k), kind_cap(share)) {
                    let m = momentum.get(&p.file).copied().unwrap_or(0.0);
                    chosen.push((p, m));
                }
            }
            chosen.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.file.cmp(&b.0.file)));
        }
    }
    let keys: Vec<FileKey> = chosen.iter().map(|(p, _)| p.file).collect();
    let order = set.order(&keys);
    let placement: HashMap<FileKey, &FilePlacement> =
        chosen.iter().map(|(p, _)| (p.file, p)).collect();
    // Every file whose members this process holds, its placed sections' lines
    // and ids resolved; a file with any member missing is not placed.
    let resolved: Vec<(FileKey, Vec<(LineSpan, SectionId)>)> = order
        .iter()
        .filter_map(|file| {
            let record = records.get(file)?;
            let p = placement.get(file)?;
            let chunks = p
                .chunks
                .iter()
                .map(|&c| {
                    let lines = record.chunks[c].lines;
                    resolver
                        .member_section(&chunk_name(&record.path, &record.blob, lines))
                        .map(|id| (lines, id))
                })
                .collect::<Option<Vec<_>>>()?;
            Some((*file, chunks))
        })
        .collect();
    let segments = resolved
        .iter()
        .flat_map(|(_, chunks)| chunks.iter())
        .map(|&(_, id)| ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection::new(id))))
        .collect();
    FilesSplice {
        placed: resolved
            .into_iter()
            .map(|(file, chunks)| PlacedFile {
                file,
                chunks: chunks.into_iter().map(|(lines, _)| lines).collect(),
            })
            .collect(),
        segments,
    }
}
