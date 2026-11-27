//! The knowledge base's files spliced into the system prompt
//! (`docs/provenance_system_collection.md` §3, §4).

use super::*;
use crate::files::names::chunk_name;
use crate::files::scope::RangeScope;
use crate::files::{
    ChunkRecord, FileKey, FileUnit, FilesPin, LineSpan, MemberRef, PlacedFile, SectionKey,
    SectionStanding,
};
use crate::persistence::content_hash::ContentHash;
use crate::persistence::streams::ContentAddress;
use crate::projection::project::{files_splice, ProjectionSegment, SealedKind};
use crate::projection::SystemPromptItem;

const YAML: &str = r#"
system_prompt:
  sections:
    - id: alpha
      content: "alpha"
layers:
  - name: code_reading
    window: 4000
    score_formula: max
    budget:
      priority: 20
    summary: &summary
      turns:
        max_tokens: 256
        user: { system_prompt: c, user_prompt: c }
        assistant: { system_prompt: c, user_prompt: c }
    groups:
      - id: scopes
        selection: { kind: working_set, share: readings }
  - name: dialogue
    window: 4000
    score_formula: max
    budget:
      priority: 100
    summary: *summary
    groups:
      - id: convo
        selection:
          kind: conversation
          recent: 4
          historical_top_k: 4
"#;

fn member(tokens: u32) -> MemberRef {
    MemberRef {
        address: ContentAddress {
            prefix_hash: ContentHash::ZERO,
            section_hash: ContentHash::ZERO,
        },
        tokens,
    }
}

fn record(path: &str, spans: &[(u32, u32)]) -> FileRecord {
    FileRecord {
        content_key: format!("v1:{path}@b"),
        path: path.into(),
        blob: "b".into(),
        unit: FileUnit::File,
        lines: spans.last().map_or(0, |s| s.1),
        lang: "rs".into(),
        branches: vec![],
        chunks: spans
            .iter()
            .map(|&(first, last)| ChunkRecord {
                lines: LineSpan { first, last },
                member: member(100),
            })
            .collect(),
    }
}

/// Every section of `file`, of `records`, in line order.
fn sections_of(records: &[FileRecord], file: FileKey) -> Vec<SectionKey> {
    records
        .iter()
        .find(|r| r.key() == file)
        .map_or(Vec::new(), FileRecord::section_keys)
}

/// A resolver holding `records`, every member registered, and a files set
/// of the readings kind in which every section of each file of `momentum`
/// stands at its file's momentum — less a thousandth per section after the
/// first, so a file's sections rank in line order — and `placed`'s files'
/// sections the range as last placed, the first at the top.
fn resolver(
    b: &Builder,
    records: &[FileRecord],
    momentum: &[(FileKey, f32)],
    placed: Vec<FileKey>,
) -> (MockResolver, HashMap<String, SectionId>) {
    let scopes = b.id_for_group("scopes").unwrap();
    let mut members: HashMap<String, SectionId> = HashMap::new();
    let mut next = 500u32;
    let mut id = |name: String, members: &mut HashMap<String, SectionId>| {
        next += 1;
        members.insert(name, SectionId::new(next));
    };
    for r in records {
        for c in &r.chunks {
            id(chunk_name(&r.path, &r.blob, c.lines), &mut members);
        }
    }
    let mut set = FilesSet::default();
    for &(f, m) in momentum {
        for (i, s) in sections_of(records, f).into_iter().enumerate() {
            let standing = SectionStanding {
                momentum: m - i as f32 / 1_000.0,
                kind: scopes,
                file: f,
            };
            set.members.insert(s, standing);
        }
    }
    if !placed.is_empty() {
        let mut scope = RangeScope::new(usize::MAX / 2);
        for s in placed.into_iter().flat_map(|f| sections_of(records, f)) {
            scope.enter(s, 100, false);
        }
        set.set_scope(scope);
    }
    let mut mock = MockResolver::new();
    mock.files = Some(set);
    mock.file_records = records.to_vec();
    mock.members = members.clone();
    (mock, members)
}

fn ids(segments: &[ProjectionSegment]) -> Vec<SectionId> {
    segments
        .iter()
        .map(|s| match s {
            ProjectionSegment::Sealed(SealedKind::Section(r)) => r.id,
            other => panic!("not a section: {other:?}"),
        })
        .collect()
}

/// **The opening places the strongest section at the top of the range,
/// nearest the question, and each after it in front** (`docs/placed_section_
/// layout.md` §3.4): b (900) at the top, a's first section (800) in front of
/// it, a's second in front of that — so from the front of the range to its
/// top the prompt reads a 401-600, a 1-200, b, nothing between them.
#[test]
fn the_opening_places_the_strongest_at_the_top() {
    let b = Builder::from_yaml(YAML).unwrap();
    let fa = record("candle/a.rs", &[(1, 200), (401, 600)]);
    let fb = record("candle/b.rs", &[(1, 50)]);
    let records = [fa.clone(), fb.clone()];
    let (mock, m) = resolver(
        &b,
        &records,
        &[(fa.key(), 800.0), (fb.key(), 900.0)],
        vec![],
    );
    let splice = files_splice(b.schema(), &mock, |_| 10_000, true);
    assert_eq!(splice.order(), vec![fa.key(), fb.key()]);
    // Each run names the lines of the sections it placed, in the order they
    // sit.
    assert_eq!(
        splice.placed[0],
        PlacedFile {
            file: fa.key(),
            chunks: vec![
                LineSpan {
                    first: 401,
                    last: 600
                },
                LineSpan {
                    first: 1,
                    last: 200
                }
            ],
        }
    );
    assert_eq!(
        ids(&splice.segments),
        vec![
            m[&chunk_name(
                "candle/a.rs",
                "b",
                LineSpan {
                    first: 401,
                    last: 600
                }
            )],
            m[&chunk_name(
                "candle/a.rs",
                "b",
                LineSpan {
                    first: 1,
                    last: 200
                }
            )],
            m[&chunk_name("candle/b.rs", "b", LineSpan { first: 1, last: 50 })],
        ]
    );
    let scope = splice.scope.expect("a free knowledge base keeps its range");
    assert_eq!(
        scope.sections().map(|e| e.key).collect::<Vec<_>>(),
        vec![fb.section_key(0), fa.section_key(0), fa.section_key(1)]
    );
}

/// **A pinned knowledge base holds its one file whole, or nothing**: pinned
/// to a, a's every section is placed under a cap that holds none of it, in
/// relevance order — its first section (100) beside the question, its second
/// (99.999) in front — b is left out though stronger; pinned empty, nothing is
/// placed.
#[test]
fn a_pinned_knowledge_base_holds_its_file_whole_or_nothing() {
    let b = Builder::from_yaml(YAML).unwrap();
    let fa = record("candle/a.rs", &[(1, 200), (201, 260)]);
    let fb = record("candle/b.rs", &[(1, 10)]);
    let records = [fa.clone(), fb.clone()];
    let (mut mock, m) = resolver(
        &b,
        &records,
        &[(fa.key(), 100.0), (fb.key(), 900.0)],
        vec![],
    );
    mock.files.as_mut().unwrap().pin = FilesPin::Only(fa.section_keys());
    let splice = files_splice(b.schema(), &mock, |_| 0, true);
    assert_eq!(splice.order(), vec![fa.key()]);
    assert_eq!(
        ids(&splice.segments),
        vec![
            m[&chunk_name(
                "candle/a.rs",
                "b",
                LineSpan {
                    first: 201,
                    last: 260
                }
            )],
            m[&chunk_name(
                "candle/a.rs",
                "b",
                LineSpan {
                    first: 1,
                    last: 200
                }
            )],
        ]
    );
    let scope = splice
        .scope
        .expect("a pinned knowledge base keeps its range");
    assert!(
        scope.sections().all(|e| e.mandatory),
        "no pinned section leaves"
    );
    mock.files.as_mut().unwrap().pin = FilesPin::Empty;
    let empty = files_splice(b.schema(), &mock, |_| 10_000, true);
    assert!(empty.placed.is_empty());
    assert!(empty.segments.is_empty());
}

/// **Pinned files sit in relevance order, not the pin's**: a pinned in
/// front of b, but a (900) stronger than b (100), so a takes the top beside
/// the question and b the front; with no momentum to rank them the pin's
/// order stands, its last file at the top.
#[test]
fn pinned_files_sit_in_relevance_order() {
    let b = Builder::from_yaml(YAML).unwrap();
    let fa = record("candle/a.rs", &[(1, 10)]);
    let fb = record("candle/b.rs", &[(1, 10)]);
    let records = [fa.clone(), fb.clone()];
    let (mut mock, m) = resolver(
        &b,
        &records,
        &[(fa.key(), 900.0), (fb.key(), 100.0)],
        vec![],
    );
    let pin = FilesPin::Only([fa.section_keys(), fb.section_keys()].concat());
    mock.files.as_mut().unwrap().pin = pin.clone();
    let splice = files_splice(b.schema(), &mock, |_| 0, true);
    assert_eq!(splice.order(), vec![fb.key(), fa.key()]);
    let chunk = |path: &str| m[&chunk_name(path, "b", LineSpan { first: 1, last: 10 })];
    assert_eq!(
        ids(&splice.segments),
        vec![chunk("candle/b.rs"), chunk("candle/a.rs")]
    );
    let (mut unranked, _) = resolver(&b, &records, &[], vec![]);
    unranked.files.as_mut().unwrap().pin = pin;
    assert_eq!(
        files_splice(b.schema(), &unranked, |_| 0, true).order(),
        vec![fa.key(), fb.key()]
    );
}

/// **A pinned knowledge base is placed for a target whose layer declares no
/// working set** — a calibration's draft and frames fork the files layer's
/// base, which has none: the pinned file sits where the `files` collection is
/// declared, after `files_open`, and an unpinned set on the same target places
/// nothing.
#[test]
fn a_pinned_knowledge_base_is_placed_for_a_layer_with_no_working_set() {
    const PINNED_YAML: &str = r#"
system_prompt:
  items:
    - kind: section
      id: alpha
      content: "alpha"
    - kind: section
      id: files_open
      content: "kb"
    - kind: collection
      name: files
      selection: { kind: always_visible }
      sections: []
layers:
  - name: code_reading
    window: 4000
    score_formula: max
    budget:
      priority: 20
    summary:
      turns:
        max_tokens: 256
        user: { system_prompt: c, user_prompt: c }
        assistant: { system_prompt: c, user_prompt: c }
    groups:
      - id: scopes
        selection: { kind: working_set, share: readings }
"#;
    let b = Builder::from_yaml(PINNED_YAML).unwrap();
    let fa = record("candle/a.rs", &[(1, 10)]);
    let (mut mock, m) = resolver(&b, std::slice::from_ref(&fa), &[(fa.key(), 900.0)], vec![]);
    let target = ProjectionTarget {
        layer: b.id_for_layer("code_reading").unwrap(),
        group: b.id_for_group("scopes").unwrap(),
        timeline: TimelineId::for_test(1),
    };
    let free = b.project(target, &mock);
    assert!(
        free.files_placed.is_empty(),
        "no working set: provenance places nothing"
    );
    mock.files.as_mut().unwrap().pin = FilesPin::Only(fa.section_keys());
    let proj = b.project(target, &mock);
    assert_eq!(proj.files_order(), fa.section_keys());
    let first = m[&chunk_name("candle/a.rs", "b", LineSpan { first: 1, last: 10 })];
    let sections: Vec<SectionId> = proj
        .segments
        .iter()
        .filter_map(|s| match s {
            ProjectionSegment::Sealed(SealedKind::Section(r)) => Some(r.id),
            _ => None,
        })
        .collect();
    let kb = b.schema().system_prompt.items.iter().find_map(|i| match i {
        SystemPromptItem::Section(s) if s.name == "files_open" => Some(s.id),
        _ => None,
    });
    let at = sections.iter().position(|&s| Some(s) == kb).unwrap();
    assert_eq!(sections[at + 1], first, "the file follows files_open");
}

/// **Kept sections and newcomers alike sit by relevance**: a at the top and
/// b in front of it, b now stronger, and c, new and strongest — c takes the
/// top, then b, then a at the front.
#[test]
fn kept_sections_and_newcomers_sit_by_relevance() {
    let b = Builder::from_yaml(YAML).unwrap();
    let fa = record("candle/a.rs", &[(1, 10)]);
    let fb = record("candle/b.rs", &[(1, 10)]);
    let fc = record("candle/c.rs", &[(1, 10)]);
    let records = [fa.clone(), fb.clone(), fc.clone()];
    let (mock, _) = resolver(
        &b,
        &records,
        &[(fa.key(), 100.0), (fb.key(), 900.0), (fc.key(), 950.0)],
        vec![fa.key(), fb.key()],
    );
    assert_eq!(
        files_splice(b.schema(), &mock, |_| 10_000, true).order(),
        vec![fa.key(), fb.key(), fc.key()]
    );
}

/// **The mandatory files open a free knowledge base, whole and outside the
/// caps, ordered like the rest, and the seed dial leaves them out**: m, bigger
/// than the cap, is placed whole but, weaker than a, at the front, a's one
/// section (under a 150-token cap) at the top; with the dial off, a alone. A
/// pinned knowledge base holds its pin alone.
#[test]
fn the_mandatory_files_open_a_free_knowledge_base() {
    let b = Builder::from_yaml(YAML).unwrap();
    let fm = record("candle/README.md", &[(1, 200), (201, 300)]);
    let fa = record("candle/a.rs", &[(1, 10)]);
    let records = [fm.clone(), fa.clone()];
    let (mut mock, _) = resolver(&b, &records, &[(fm.key(), 10.0), (fa.key(), 900.0)], vec![]);
    mock.mandatory_files = vec!["candle/README.md".into(), "candle/AGENTS.md".into()];
    let on = files_splice(b.schema(), &mock, |_| 150, true);
    assert_eq!(on.order(), vec![fm.key(), fa.key()]);
    assert_eq!(on.placed[0].chunks.len(), 2, "whole, whatever the cap");
    assert!(
        on.scope
            .as_ref()
            .unwrap()
            .entry(fm.section_key(0))
            .unwrap()
            .mandatory
    );
    assert_eq!(on.segments.len(), 3);
    let off = files_splice(b.schema(), &mock, |_| 150, false);
    assert_eq!(off.order(), vec![fa.key()]);
    mock.files.as_mut().unwrap().pin = FilesPin::Only(fa.section_keys());
    let pinned = files_splice(b.schema(), &mock, |_| 150, true);
    assert_eq!(pinned.order(), vec![fa.key()]);
}

/// **A file's sections compete apart** (`docs/placed_section_layout.md`
/// §3.5c): a's first section (900) takes the top, b (500) enters in front of
/// it, and a's second (100) no longer fits the 200-token cap — the placed
/// runs are b, a, and each names only what it placed.
#[test]
fn a_files_sections_compete_apart() {
    let b = Builder::from_yaml(YAML).unwrap();
    let fa = record("candle/a.rs", &[(1, 200), (201, 400)]);
    let fb = record("candle/b.rs", &[(1, 10)]);
    let records = [fa.clone(), fb.clone()];
    let (mut mock, _) = resolver(
        &b,
        &records,
        &[(fa.key(), 900.0), (fb.key(), 500.0)],
        vec![],
    );
    let second = fa.section_key(1);
    let set = mock.files.as_mut().unwrap();
    set.members.get_mut(&second).unwrap().momentum = 100.0;
    let splice = files_splice(b.schema(), &mock, |_| 200, true);
    assert_eq!(splice.order(), vec![fb.key(), fa.key()]);
    assert_eq!(splice.sections, vec![fb.section_key(0), fa.section_key(0)]);
    assert_eq!(
        splice.placed[1].chunks,
        vec![LineSpan {
            first: 1,
            last: 200
        }]
    );
}
