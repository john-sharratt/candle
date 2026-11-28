//! The knowledge base's sections placed into the system prompt
//! (`docs/provenance_system_collection.md` §3, §4;
//! `docs/placed_section_layout.md` §3.5c).
//!
//! Each kind's sections are chosen under the kind's share of the room
//! ([`choose`]), strongest first. The sections of every kind, the mandatory
//! files among them, then sit in relevance order by the set's order score —
//! the index's hit momentum this question, the scan's momentum for a section
//! the index has not read — the strongest at the top beside the question,
//! each weaker one further away.
//! Every section carries its own heading, closing line and, short of its
//! file's end, the cut marker, so nothing is placed between them.
//!
//! A free knowledge base the seed dial leaves on opens on the mandatory files
//! (§3.5b), whole and ahead of the chosen sections, outside every kind's cap;
//! a pinned one holds its pin alone.

use std::collections::HashSet;

use super::files_anchor::FILES_COLLECTION;
use super::*;
use crate::files::fill::choose;
use crate::files::names::chunk_name;
use crate::files::scope::{Candidate, Place, RangeScope};
use crate::files::{
    mandatory_records, FileKey, FileRecord, FilesPin, LineSpan, PlacedFile, SectionKey,
};

/// The sections a projection places and the segments they are placed as.
pub(crate) struct FilesSplice {
    /// The placed sections as runs of one file's, in the order they sit,
    /// each run with its sections' lines.
    pub placed: Vec<PlacedFile>,
    /// The placed sections, in the order they sit, from the front of the
    /// range to its top.
    pub sections: Vec<SectionKey>,
    pub segments: Vec<ProjectionSegment>,
    /// The range the knowledge base now holds, free or pinned, for the files
    /// set to keep and the decode to reorder; `None` when it places nothing.
    pub scope: Option<RangeScope>,
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
            scope: None,
        };
    };
    let all = resolver.files_records();
    // Every section of every record: its record and its place in it.
    let index: HashMap<SectionKey, (&FileRecord, usize)> = all
        .iter()
        .flat_map(|r| (0..r.chunks.len()).map(move |i| (r.section_key(i), (r, i))))
        .collect();
    let member = |key: SectionKey| -> Option<SectionId> {
        let &(record, i) = index.get(&key)?;
        resolver.member_section(&chunk_name(
            &record.path,
            &record.blob,
            record.chunks[i].lines,
        ))
    };
    let candidate = |key: SectionKey| -> Option<Candidate> {
        let &(record, i) = index.get(&key)?;
        member(key)?;
        Some(Candidate {
            key,
            tokens: record.chunks[i].member.tokens as usize,
        })
    };
    // Pinned, the pinned sections held in relevance order like a free range's
    // — the pin says what is placed, never the order — or nothing; free, the
    // range as it stood with what enters it now.
    let places: Vec<Place> = match &set.pin {
        FilesPin::Empty => Vec::new(),
        FilesPin::Only(sections) => {
            let pinned: Vec<Candidate> = sections.iter().filter_map(|&k| candidate(k)).collect();
            let positions = files_positions(schema, resolver.trained_window());
            let same = |s: &RangeScope| {
                s.positions == positions
                    && s.sections().count() == pinned.len()
                    && pinned.iter().all(|c| s.holds(c.key))
            };
            // The range the decode has been ordering, while it still holds
            // exactly the pin; otherwise the pin opened afresh, each section
            // held as a mandatory one, so none leaves, and ordered by the set's
            // order score. The pin lists the range from its front to its top, so
            // it opens last first: where nothing ranks them, the pin's own
            // order stands, its last section beside the question.
            let scope = match &set.scope {
                Some(s) if same(s) => s.clone(),
                _ => {
                    let top_first: Vec<Candidate> = pinned.iter().rev().copied().collect();
                    let mut s = RangeScope::new(positions);
                    s.open(&top_first, &[]);
                    s.reorder(|k| set.order_score(k), 0.0);
                    s
                }
            };
            let places = scope.places().iter().rev().copied().collect();
            return emit(places, &index, &member, Some(scope));
        }
        FilesPin::Free => {
            let paths = match mandatory {
                true => resolver.files_mandatory(),
                false => Vec::new(),
            };
            let mandatory_sections: Vec<Candidate> = mandatory_records(&all, &paths)
                .into_iter()
                .flat_map(FileRecord::section_keys)
                .filter_map(candidate)
                .collect();
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
            // A section its hit rate evicted this request stays out of it.
            let picked: Vec<Candidate> = picked
                .into_iter()
                .filter(|(k, _)| !set.evicted.contains(k))
                .filter_map(|(k, _)| candidate(k))
                .collect();
            let positions = files_positions(schema, resolver.trained_window());
            let mut scope = match &set.scope {
                Some(s) if s.positions == positions => s.clone(),
                _ => RangeScope::new(positions),
            };
            // A placed section this process no longer holds leaves the range.
            let gone: Vec<SectionKey> = scope
                .sections()
                .map(|e| e.key)
                .filter(|&k| member(k).is_none())
                .collect();
            for k in gone {
                scope.forget(k);
            }
            if scope.sections().next().is_none() {
                scope.open(&mandatory_sections, &picked);
            } else {
                for c in &mandatory_sections {
                    scope.enter(c.key, c.tokens, true);
                }
                for c in &picked {
                    scope.enter(c.key, c.tokens, false);
                }
            }
            // The first token reads the range in relevance order: strongest
            // beside the question, the mandatory files among the rest, by the
            // index's momentum — the question's first reading's rates, then
            // the decode's — and the scan's for a section the index has not
            // read (`FilesSet::order_score`, `docs/placed_section_layout.md`
            // §3.2). Every decode step reorders it from there.
            scope.reorder(|k| set.order_score(k), 0.0);
            let places = scope.places().iter().rev().copied().collect();
            return emit(places, &index, &member, Some(scope));
        }
    };
    emit(places, &index, &member, None)
}

/// The positions the knowledge base's range reserves on an engine of
/// `native` context: the `files` collection's range, or no bound when it
/// declares none.
fn files_positions(schema: &Schema, native: Option<usize>) -> usize {
    schema
        .system_prompt
        .items
        .iter()
        .find_map(|item| match item {
            SystemPromptItem::Collection(c) if c.name == FILES_COLLECTION => c.range,
            _ => None,
        })
        .map_or(usize::MAX / 2, |range| range.positions(native))
}

/// `places`, from the front of the range to its top, as the splice places
/// them: each section its member, each hole its positions.
fn emit(
    places: Vec<Place>,
    index: &HashMap<SectionKey, (&FileRecord, usize)>,
    member: &impl Fn(SectionKey) -> Option<SectionId>,
    scope: Option<RangeScope>,
) -> FilesSplice {
    let mut segments = Vec::with_capacity(places.len());
    let mut spans: Vec<(FileKey, LineSpan)> = Vec::new();
    let mut sections = Vec::new();
    for place in places {
        match place {
            Place::Hole(positions) => segments.push(ProjectionSegment::Hole { positions }),
            Place::Section(e) => {
                let (Some(&(record, i)), Some(id)) = (index.get(&e.key), member(e.key)) else {
                    continue;
                };
                spans.push((record.key(), record.chunks[i].lines));
                sections.push(e.key);
                segments.push(ProjectionSegment::Sealed(SealedKind::Section(
                    ResolvedSection::new(id),
                )));
            }
        }
    }
    FilesSplice {
        placed: runs(&spans),
        sections,
        segments,
        scope,
    }
}
