//! The knowledge base's files placed into the system prompt
//! (`docs/provenance_system_collection.md` §3, §4).
//!
//! Each kind's files are chosen under the kind's share of the room
//! ([`choose`]), strongest first. The files of every kind then sit in the
//! order the conversation's files set keeps: at the opening strongest first,
//! afterwards every kept file where it was and every newcomer after them. Each
//! file is its heading, its placed chunks — a gap piece between two runs that
//! are not contiguous — and its closing piece: `closing_join` when anything of
//! the knowledge base follows it, `closing_last` when it ends the knowledge
//! base, the close coming next.

use super::*;
use crate::files::fill::{choose, FilePlacement};
use crate::files::names::{
    chunk_name, heading_name, CLOSING_JOIN_NAME, CLOSING_LAST_NAME, GAP_NAME,
};
use crate::files::{FileKey, FileRecord, LineSpan};

/// The files a projection places and the segments they are placed as.
pub(crate) struct FilesSplice {
    /// The placed files, in the order they sit.
    pub order: Vec<FileKey>,
    pub segments: Vec<ProjectionSegment>,
}

/// The files to place under `kind_cap` — each working-set group of `schema`
/// a kind, its cap the tokens its share allows — and the segments they are
/// placed as; `seeded` when the seed follows them in the knowledge base.
/// Empty when the target has no files set, or no file's members are held in
/// this process.
pub(crate) fn files_splice<R: ContentResolver>(
    schema: &Schema,
    resolver: &R,
    kind_cap: impl Fn(WorkingSetShare) -> usize,
    seeded: bool,
) -> FilesSplice {
    let empty = FilesSplice {
        order: Vec::new(),
        segments: Vec::new(),
    };
    let Some(set) = resolver.files_set() else {
        return empty;
    };
    let pieces = (
        resolver.member_section(GAP_NAME),
        resolver.member_section(CLOSING_JOIN_NAME),
        resolver.member_section(CLOSING_LAST_NAME),
    );
    let (Some(gap), Some(join), Some(last)) = pieces else {
        return empty;
    };
    let closing_tokens = resolver.section_token_count(join) as u32;
    let records: HashMap<FileKey, FileRecord> = resolver
        .files_records()
        .into_iter()
        .map(|r| (r.key(), r))
        .collect();
    // Each kind's choice, then every chosen file ranked by its momentum.
    let mut chosen: Vec<(FilePlacement, f32)> = Vec::new();
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
        for p in choose(
            &ranked,
            |k| records.get(&k),
            kind_cap(share),
            closing_tokens,
        ) {
            let m = momentum.get(&p.file).copied().unwrap_or(0.0);
            chosen.push((p, m));
        }
    }
    chosen.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.file.cmp(&b.0.file)));
    let keys: Vec<FileKey> = chosen.iter().map(|(p, _)| p.file).collect();
    let order = set.order(&keys);
    let placement: HashMap<FileKey, &FilePlacement> =
        chosen.iter().map(|(p, _)| (p.file, p)).collect();
    // Every file whose members this process holds, its heading and its placed
    // chunks' lines and sections resolved; a file with any member missing is
    // not placed.
    let resolved: Vec<(FileKey, SectionId, Vec<(LineSpan, SectionId)>)> = order
        .iter()
        .filter_map(|file| {
            let record = records.get(file)?;
            let p = placement.get(file)?;
            let heading = resolver.member_section(&heading_name(&record.path, &record.blob))?;
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
            Some((*file, heading, chunks))
        })
        .collect();
    let section =
        |id: SectionId| ProjectionSegment::Sealed(SealedKind::Section(ResolvedSection::new(id)));
    let mut segments = Vec::new();
    for (i, (_, heading, chunks)) in resolved.iter().enumerate() {
        segments.push(section(*heading));
        for (j, &(lines, id)) in chunks.iter().enumerate() {
            if j > 0 && !lines.continues(chunks[j - 1].0) {
                segments.push(section(gap));
            }
            segments.push(section(id));
        }
        let ends = i + 1 == resolved.len() && !seeded;
        segments.push(section(if ends { last } else { join }));
    }
    FilesSplice {
        order: resolved.into_iter().map(|(f, _, _)| f).collect(),
        segments,
    }
}
