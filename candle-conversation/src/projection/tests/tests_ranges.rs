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
