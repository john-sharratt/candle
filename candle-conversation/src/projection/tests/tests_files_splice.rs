//! The knowledge base's files spliced into the system prompt
//! (`docs/provenance_system_collection.md` §3, §4).

use super::*;
use crate::files::names::{
    chunk_name, heading_name, CLOSING_JOIN_NAME, CLOSING_LAST_NAME, GAP_NAME,
};
use crate::files::{ChunkRecord, FileKey, FileUnit, FilesPin, LineSpan, MemberRef};
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
        heading: member(10),
        chunks: spans
            .iter()
            .map(|&(first, last)| ChunkRecord {
                lines: LineSpan { first, last },
                member: member(100),
            })
            .collect(),
    }
}

/// A resolver holding `records`, every member registered, and a files set
/// of the readings kind with `momentum`, `placed` the order last placed.
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
    for name in [GAP_NAME, CLOSING_JOIN_NAME, CLOSING_LAST_NAME] {
        id(name.to_string(), &mut members);
    }
    for r in records {
        id(heading_name(&r.path, &r.blob), &mut members);
        for c in &r.chunks {
            id(chunk_name(&r.path, &r.blob, c.lines), &mut members);
        }
    }
    let mut set = FilesSet::default();
    for &(f, m) in momentum {
        set.members.insert(f, (m, scopes));
    }
    set.set_placed(placed);
    let mut mock = MockResolver::new();
    mock.files = Some(set);
    mock.file_records = records.to_vec();
    mock.members = members.clone();
    mock.section_tokens
        .insert(members[CLOSING_JOIN_NAME].raw(), 2);
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

/// **The files sit strongest first at the opening, each its heading, its
/// chunks with a gap between runs and its closing piece, the last closing the
/// knowledge base**: b (900) whole, then a (800), whose two chunks are not
/// contiguous, ending on `closing_last`.
#[test]
fn files_are_spliced_strongest_first_each_framed_and_the_last_ends_the_knowledge_base() {
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
    let splice = files_splice(b.schema(), &mock, |_| 10_000, false);
    assert_eq!(splice.order, vec![fb.key(), fa.key()]);
    assert_eq!(
        ids(&splice.segments),
        vec![
            m[&heading_name("candle/b.rs", "b")],
            m[&chunk_name("candle/b.rs", "b", LineSpan { first: 1, last: 50 })],
            m[CLOSING_JOIN_NAME],
            m[&heading_name("candle/a.rs", "b")],
            m[&chunk_name(
                "candle/a.rs",
                "b",
                LineSpan {
                    first: 1,
                    last: 200
                }
            )],
            m[GAP_NAME],
            m[&chunk_name(
                "candle/a.rs",
                "b",
                LineSpan {
                    first: 401,
                    last: 600
                }
            )],
            m[CLOSING_LAST_NAME],
        ]
    );
    // With the seed after it, the last file joins rather than ends.
    let seeded = files_splice(b.schema(), &mock, |_| 10_000, true);
    assert_eq!(ids(&seeded.segments).last(), Some(&m[CLOSING_JOIN_NAME]));
}

/// **A pinned knowledge base holds its one file whole, or nothing**: pinned
/// to a, a's every chunk is placed under a cap that holds none of it, b is
/// left out though stronger, and a ends the knowledge base; pinned empty,
/// nothing is placed.
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
    mock.files.as_mut().unwrap().pin = FilesPin::Only(fa.key());
    let splice = files_splice(b.schema(), &mock, |_| 0, false);
    assert_eq!(splice.order, vec![fa.key()]);
    assert_eq!(
        ids(&splice.segments),
        vec![
            m[&heading_name("candle/a.rs", "b")],
            m[&chunk_name(
                "candle/a.rs",
                "b",
                LineSpan {
                    first: 1,
                    last: 200
                }
            )],
            m[&chunk_name(
                "candle/a.rs",
                "b",
                LineSpan {
                    first: 201,
                    last: 260
                }
            )],
            m[CLOSING_LAST_NAME],
        ]
    );
    mock.files.as_mut().unwrap().pin = FilesPin::Empty;
    let empty = files_splice(b.schema(), &mock, |_| 10_000, false);
    assert!(empty.order.is_empty());
    assert!(empty.segments.is_empty());
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
    mock.files.as_mut().unwrap().pin = FilesPin::Only(fa.key());
    let proj = b.project(target, &mock);
    assert_eq!(proj.files_placed, vec![fa.key()]);
    let heading = m[&heading_name("candle/a.rs", "b")];
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
    assert_eq!(sections[at + 1], heading, "the file follows files_open");
}

/// **A kept file keeps its place**: a placed before b, b now stronger — a
/// stays first.
#[test]
fn a_kept_file_keeps_its_place_in_the_knowledge_base() {
    let b = Builder::from_yaml(YAML).unwrap();
    let fa = record("candle/a.rs", &[(1, 10)]);
    let fb = record("candle/b.rs", &[(1, 10)]);
    let records = [fa.clone(), fb.clone()];
    let (mock, _) = resolver(
        &b,
        &records,
        &[(fa.key(), 100.0), (fb.key(), 900.0)],
        vec![fa.key(), fb.key()],
    );
    assert_eq!(
        files_splice(b.schema(), &mock, |_| 10_000, false).order,
        vec![fa.key(), fb.key()]
    );
}
