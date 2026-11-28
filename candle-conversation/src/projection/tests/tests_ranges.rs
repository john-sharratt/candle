//! A collection's reserved range (`docs/placed_section_layout.md` §3.1): its
//! members and a hole of what they leave, so the range always takes its full
//! size — the hole ahead of the members in a range filled from the top, after
//! them in one filled from the bottom.

use super::*;
use crate::projection::{ProjectionMode, ProjectionSegment, SealedKind, SelectionState};

fn yaml(fill: &str) -> String {
    format!(
        r#"
system_prompt:
  items:
    - kind: section
      id: framing
      content: "framing"
    - kind: collection
      name: tools
      selection: {{ kind: top_k, k: 2 }}
      range: {{ tokens: 100, small_context: 40, fill: {fill} }}
      sections:
        - id: tool_a
          content: "A"
        - id: tool_b
          content: "B"
        - id: tool_c
          content: "C"
    - kind: section
      id: outro
      content: "outro"
layers:
  - name: dialogue
    window: 4000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: max
    budget: {{ priority: 100 }}
    groups:
      - id: convo
        selection: {{ kind: always_visible }}
"#
    )
}

/// The projection's segments as `(section, hole)` places, glue left out.
fn places(segments: &[ProjectionSegment]) -> Vec<(Option<SectionId>, Option<usize>)> {
    segments
        .iter()
        .filter_map(|s| match s {
            ProjectionSegment::Sealed(SealedKind::Section(r)) => Some((Some(r.id), None)),
            ProjectionSegment::Hole { positions } => Some((None, Some(*positions))),
            _ => None,
        })
        .collect()
}

fn project(b: &Builder, resolver: &MockResolver) -> Vec<ProjectionSegment> {
    let target = ProjectionTarget {
        layer: b.id_for_layer("dialogue").unwrap(),
        group: b.id_for_group("convo").unwrap(),
        timeline: TimelineId::for_test(1),
    };
    b.project_with_selection(
        target,
        resolver,
        ProjectionMode::Decode,
        &SelectionState::default(),
    )
    .segments
}

/// **A range filled from the top holds its members at the top**: a hole of
/// what the members that fit leave, then the members — c (50) does not fit
/// the 45 a and b leave.
#[test]
fn a_range_filled_from_the_top_holds_its_members_at_the_top() {
    let b = Builder::from_yaml(&yaml("top")).unwrap();
    let id = |n: &str| b.id_for_system_section(n).unwrap();
    let resolver = MockResolver::new()
        .with_section_score(id("tool_a"), 0.9)
        .with_section_score(id("tool_b"), 0.8)
        .with_section_score(id("tool_c"), 0.1)
        .with_section_tokens(id("framing"), 5)
        .with_section_tokens(id("tool_a"), 30)
        .with_section_tokens(id("tool_b"), 25)
        .with_section_tokens(id("tool_c"), 50)
        .with_section_tokens(id("outro"), 4);
    assert_eq!(
        places(&project(&b, &resolver)),
        vec![
            (Some(id("framing")), None),
            (None, Some(100 - 30 - 25)),
            (Some(id("tool_a")), None),
            (Some(id("tool_b")), None),
            (Some(id("outro")), None),
        ]
    );
}

/// **A collection's lead is a section of the schema but never one of its
/// members, and never placed** (`docs/placed_section_layout.md` §3.5f): it is
/// named, counted among the prompt's ids, and sealed by the creation walk, but
/// the projection lays down the same range as without it, scored or not.
#[test]
fn a_collections_lead_is_never_placed() {
    let yaml = yaml("top").replace(
        "      sections:\n        - id: tool_a",
        "      lead:\n        - id: tool_lead\n          content: \"L\"\n      sections:\n        - id: tool_a",
    );
    let b = Builder::from_yaml(&yaml).unwrap();
    let id = |n: &str| b.id_for_system_section(n).unwrap();
    let system = &b.schema().system_prompt;
    let tools = system.collection_named("tools").unwrap();
    assert_eq!(tools.lead_ids(), vec![id("tool_lead")]);
    assert_eq!(tools.sections.len(), 3);
    assert!(!system.is_collection_member(id("tool_lead")));
    assert!(system.all_section_ids().any(|s| s == id("tool_lead")));
    let resolver = MockResolver::new()
        .with_section_score(id("tool_lead"), 0.95)
        .with_section_score(id("tool_a"), 0.9)
        .with_section_score(id("tool_b"), 0.8)
        .with_section_score(id("tool_c"), 0.1)
        .with_section_tokens(id("framing"), 5)
        .with_section_tokens(id("tool_lead"), 10)
        .with_section_tokens(id("tool_a"), 30)
        .with_section_tokens(id("tool_b"), 25)
        .with_section_tokens(id("tool_c"), 50)
        .with_section_tokens(id("outro"), 4);
    assert_eq!(
        places(&project(&b, &resolver)),
        vec![
            (Some(id("framing")), None),
            (None, Some(100 - 30 - 25)),
            (Some(id("tool_a")), None),
            (Some(id("tool_b")), None),
            (Some(id("outro")), None),
        ]
    );
}

/// **A ranged collection is filled by tokens, not by its count**
/// (`docs/placed_section_layout.md` §3.5): `top_k` 2 notwithstanding, the
/// 60-position range takes a (30) and b (25) by score and would take a third
/// that fit; c (20) does not fit the 5 left. The hole is what they leave.
#[test]
fn a_ranged_collection_is_filled_by_tokens() {
    let yaml = yaml("top").replace(
        "range: { tokens: 100, small_context: 40, fill: top }",
        "range: { tokens: 60 }",
    );
    let b = Builder::from_yaml(&yaml).unwrap();
    let id = |n: &str| b.id_for_system_section(n).unwrap();
    let resolver = MockResolver::new()
        .with_section_score(id("tool_a"), 950.0)
        .with_section_score(id("tool_b"), 900.0)
        .with_section_score(id("tool_c"), 850.0)
        .with_section_tokens(id("tool_a"), 30)
        .with_section_tokens(id("tool_b"), 25)
        .with_section_tokens(id("tool_c"), 20);
    let got = places(&project(&b, &resolver));
    assert!(
        got.contains(&(None, Some(5))),
        "the range's hole is what a and b leave: {got:?}"
    );
    assert!(got.contains(&(Some(id("tool_a")), None)));
    assert!(got.contains(&(Some(id("tool_b")), None)));
    assert!(!got.contains(&(Some(id("tool_c")), None)), "{got:?}");
}

/// **A range with a member bound holds no more than it**, though there is
/// room: in 100 positions a (30), b (25) and c (20) would all fit, and `most:
/// 1` keeps the strongest alone. The hole is what it leaves.
#[test]
fn a_bounded_range_holds_no_more_members_than_its_bound() {
    let yaml = yaml("top").replace(
        "range: { tokens: 100, small_context: 40, fill: top }",
        "range: { tokens: 100, most: 1 }",
    );
    let b = Builder::from_yaml(&yaml).unwrap();
    let id = |n: &str| b.id_for_system_section(n).unwrap();
    let resolver = MockResolver::new()
        .with_section_score(id("tool_a"), 950.0)
        .with_section_score(id("tool_b"), 900.0)
        .with_section_score(id("tool_c"), 850.0)
        .with_section_tokens(id("tool_a"), 30)
        .with_section_tokens(id("tool_b"), 25)
        .with_section_tokens(id("tool_c"), 20);
    assert_eq!(
        places(&project(&b, &resolver)),
        vec![
            (Some(id("framing")), None),
            (None, Some(70)),
            (Some(id("tool_a")), None),
            (Some(id("outro")), None),
        ]
    );
}

/// **An offered range with no opinion holds no more than its bound either**:
/// with every offered member unscored the collection shows what was offered,
/// b (25) and c (20), and `most: 1` keeps the first of them, b, alone.
#[test]
fn an_offered_range_with_no_opinion_holds_its_bound() {
    let yaml = yaml("top")
        .replace(
            "selection: { kind: top_k, k: 2 }",
            "selection: { kind: offered, selector: tools, k: 1 }",
        )
        .replace(
            "range: { tokens: 100, small_context: 40, fill: top }",
            "range: { tokens: 100, most: 1 }",
        );
    let b = Builder::from_yaml(&yaml).unwrap();
    let id = |n: &str| b.id_for_system_section(n).unwrap();
    let resolver = MockResolver::new()
        .with_section_tokens(id("tool_a"), 30)
        .with_section_tokens(id("tool_b"), 25)
        .with_section_tokens(id("tool_c"), 20);
    let target = ProjectionTarget {
        layer: b.id_for_layer("dialogue").unwrap(),
        group: b.id_for_group("convo").unwrap(),
        timeline: TimelineId::for_test(1),
    };
    let mut offered = SelectionState::default();
    offered.select_all("tools", ["tool_b", "tool_c"]);
    let segments = b
        .project_with_selection(target, &resolver, ProjectionMode::Decode, &offered)
        .segments;
    assert_eq!(
        places(&segments),
        vec![
            (Some(id("framing")), None),
            (None, Some(75)),
            (Some(id("tool_b")), None),
            (Some(id("outro")), None),
        ]
    );
}

/// **A full range filled from the bottom is a window**: the oldest member
/// goes, the newest stay — in a 40-position range, a (30) is dropped and b
/// (25) and c (10) are placed, the hole the 5 they leave.
#[test]
fn a_full_bottom_range_drops_its_oldest_member() {
    let yaml = yaml("bottom")
        .replace(
            "selection: { kind: top_k, k: 2 }",
            "selection: { kind: always_visible }",
        )
        .replace(
            "range: { tokens: 100, small_context: 40, fill: bottom }",
            "range: { tokens: 40, fill: bottom }",
        );
    let b = Builder::from_yaml(&yaml).unwrap();
    let id = |n: &str| b.id_for_system_section(n).unwrap();
    let resolver = MockResolver::new()
        .with_section_tokens(id("tool_a"), 30)
        .with_section_tokens(id("tool_b"), 25)
        .with_section_tokens(id("tool_c"), 10);
    let got = places(&project(&b, &resolver));
    assert!(!got.contains(&(Some(id("tool_a")), None)), "{got:?}");
    assert!(got.contains(&(Some(id("tool_b")), None)), "{got:?}");
    assert!(got.contains(&(Some(id("tool_c")), None)), "{got:?}");
    assert!(got.contains(&(None, Some(5))), "{got:?}");
}

/// **A range filled from the bottom holds its members at the bottom**, the
/// hole after them.
#[test]
fn a_range_filled_from_the_bottom_holds_its_members_at_the_bottom() {
    let b = Builder::from_yaml(&yaml("bottom")).unwrap();
    let id = |n: &str| b.id_for_system_section(n).unwrap();
    let resolver = MockResolver::new()
        .with_section_score(id("tool_a"), 0.9)
        .with_section_score(id("tool_b"), 0.8)
        .with_section_tokens(id("tool_a"), 30)
        .with_section_tokens(id("tool_b"), 25);
    let got = places(&project(&b, &resolver));
    let hole = got.iter().position(|p| p.1.is_some()).expect("a hole");
    let members: Vec<usize> = got
        .iter()
        .enumerate()
        .filter(|(_, p)| p.0 == Some(id("tool_a")) || p.0 == Some(id("tool_b")))
        .map(|(i, _)| i)
        .collect();
    assert!(
        members.iter().all(|&m| m < hole),
        "the members come first: {got:?}"
    );
    assert_eq!(got[hole].1, Some(45));
}
