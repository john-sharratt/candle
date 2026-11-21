//! Tests for choosing a collection's members by score, by threshold and by named
//! set.
//!
//! The top-k keeps the highest-scored members in declaration order, with ties
//! going to declaration order, and a score threshold removes those below its
//! floor. A single selection keeps only the maximum. A named selector emits every
//! member of its set in declaration order, with the collection's glue between
//! consecutive members and never before the first.

use super::*;

#[test]
pub(super) fn collection_top_k_keeps_highest_scored_in_declaration_order() {
    let b = Builder::from_yaml(COLLECTION_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let framing = b.id_for_system_section("framing").unwrap();
    let tools_intro = b.id_for_system_section("tools_intro").unwrap();
    let tool_a = b.id_for_system_section("tool_a").unwrap();
    let tool_b = b.id_for_system_section("tool_b").unwrap();
    let tool_c = b.id_for_system_section("tool_c").unwrap();
    let tool_d = b.id_for_system_section("tool_d").unwrap();
    let tools_outro = b.id_for_system_section("tools_outro").unwrap();

    // tool_b and tool_d score highest → top-k=2 keeps them.  Emission is
    // in declaration order: framing → tools_intro → (tool_b → tool_d) →
    // tools_outro.  Tools in lower scores (a, c) are filtered.
    let resolver = MockResolver::new()
        .with_section_score(tool_a, 0.1)
        .with_section_score(tool_b, 0.9)
        .with_section_score(tool_c, 0.2)
        .with_section_score(tool_d, 0.8);

    let p = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let ids: Vec<SectionId> = p.sealed_sections().map(|s| s.id).collect();
    assert_eq!(
        ids,
        vec![framing, tools_intro, tool_b, tool_d, tools_outro],
        "expected static framing intact + top-2 tools in declaration order"
    );
}

#[test]
pub(super) fn collection_top_k_with_priority_tiebreak() {
    let b = Builder::from_yaml(COLLECTION_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let tool_a = b.id_for_system_section("tool_a").unwrap();
    let tool_b = b.id_for_system_section("tool_b").unwrap();
    let tool_c = b.id_for_system_section("tool_c").unwrap();
    let tool_d = b.id_for_system_section("tool_d").unwrap();

    // All four tools tie on score → fall back to declaration order
    // (a, b, c, d) and pick the first two.
    let resolver = MockResolver::new()
        .with_section_score(tool_a, 0.5)
        .with_section_score(tool_b, 0.5)
        .with_section_score(tool_c, 0.5)
        .with_section_score(tool_d, 0.5);

    let p = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let tool_ids: Vec<SectionId> = p
        .sealed_sections()
        .map(|s| s.id)
        .filter(|&id| id == tool_a || id == tool_b || id == tool_c || id == tool_d)
        .collect();
    assert_eq!(tool_ids, vec![tool_a, tool_b]);
}

pub(super) const COLLECTION_YAML_THRESHOLD: &str = r#"
system_prompt:
  items:
    - kind: collection
      summary:
        chunk: 4
        categorize:
          max_tokens: 256
          system_prompt: Propose a few functional categories for the sections.
          user_prompt: Propose categories for the content above.
        assign:
          max_tokens: 128
          system_prompt: Assign each section to a category by number.
          user_prompt: Assign each section above to a category number.
      name: tools
      selection: { kind: top_k, k: 5 }
      score_threshold: 0.4
      sections:
        - id: low
          content: "low"
        - id: mid
          content: "mid"
        - id: high
          content: "high"
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
    budget: { priority: 100 }
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;

#[test]
pub(super) fn collection_score_threshold_filters_below_floor() {
    let b = Builder::from_yaml(COLLECTION_YAML_THRESHOLD).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let mid = b.id_for_system_section("mid").unwrap();
    let high = b.id_for_system_section("high").unwrap();

    let resolver = MockResolver::new()
        .with_section_score(b.id_for_system_section("low").unwrap(), 0.2)
        .with_section_score(mid, 0.5)
        .with_section_score(high, 0.9);

    let p = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let ids: Vec<SectionId> = p.sealed_sections().map(|s| s.id).collect();
    assert_eq!(ids, vec![mid, high], "low filtered by 0.4 threshold");
}

#[test]
pub(super) fn collection_single_picks_max_only() {
    const YAML: &str = r#"
system_prompt:
  items:
    - kind: collection
      summary:
        chunk: 4
        categorize:
          max_tokens: 256
          system_prompt: Propose a few functional categories for the sections.
          user_prompt: Propose categories for the content above.
        assign:
          max_tokens: 128
          system_prompt: Assign each section to a category by number.
          user_prompt: Assign each section above to a category number.
      name: choices
      selection: { kind: single }
      sections:
        - id: a
          content: "a"
        - id: b
          content: "b"
        - id: c
          content: "c"
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
    budget: { priority: 100 }
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let b_id = b.id_for_system_section("b").unwrap();

    let resolver = MockResolver::new()
        .with_section_score(b.id_for_system_section("a").unwrap(), 0.4)
        .with_section_score(b_id, 0.9)
        .with_section_score(b.id_for_system_section("c").unwrap(), 0.6);

    let p = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    assert_eq!(p.sealed_sections().count(), 1);
    assert_eq!(p.sealed_sections().next().unwrap().id, b_id);
}

#[test]
pub(super) fn collection_named_pins_member_by_runtime_selector() {
    use crate::projection::{ProjectionMode, SelectionState};
    const YAML: &str = r#"
system_prompt:
  items:
    - kind: collection
      summary:
        chunk: 4
        categorize:
          max_tokens: 256
          system_prompt: Propose a few functional categories for the sections.
          user_prompt: Propose categories for the content above.
        assign:
          max_tokens: 128
          system_prompt: Assign each section to a category by number.
          user_prompt: Assign each section above to a category number.
      name: tools
      selection: { kind: named, selector: tool }
      sections:
        - id: datetime
          content: "datetime def"
        - id: web_search
          content: "web_search def"
        - id: calc
          content: "calc def"
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
    budget: { priority: 100 }
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let web_search = b.id_for_system_section("web_search").unwrap();
    let target = ProjectionTarget {
        layer: dialogue,
        group: convo,
        timeline: TimelineId::for_test(1),
    };
    // Scores are deliberately hostile (the two *unwanted* members rank highest):
    // Named must ignore them entirely and pin the named member.
    let resolver = MockResolver::new()
        .with_section_score(b.id_for_system_section("datetime").unwrap(), 0.99)
        .with_section_score(web_search, 0.01)
        .with_section_score(b.id_for_system_section("calc").unwrap(), 0.99);

    // Pin web_search by name → exactly that one member, score notwithstanding.
    let mut sel = SelectionState::new();
    sel.select("tool", "web_search");
    let p = b.project_with_selection(target, &resolver, ProjectionMode::Decode, &sel);
    let ids: Vec<SectionId> = p.sealed_sections().map(|s| s.id).collect();
    assert_eq!(
        ids,
        vec![web_search],
        "named must pin exactly the selected member, ignoring score"
    );

    // Unset selector → no member emitted.
    let p_unset = b.project_with_selection(
        target,
        &resolver,
        ProjectionMode::Decode,
        &SelectionState::default(),
    );
    assert_eq!(
        p_unset.sealed_sections().count(),
        0,
        "unset selector selects no member"
    );

    // Selector names a non-member → no member emitted.
    let mut sel_bad = SelectionState::new();
    sel_bad.select("tool", "nonexistent");
    let p_bad = b.project_with_selection(target, &resolver, ProjectionMode::Decode, &sel_bad);
    assert_eq!(
        p_bad.sealed_sections().count(),
        0,
        "unknown member name selects nothing"
    );
}

/// A selector holds a set: `select_all` names several, `select` replaces the
/// set with one, and `get` reads the first.
#[test]
pub(super) fn a_selector_holds_a_set_and_select_replaces_it() {
    use crate::projection::SelectionState;
    let mut s = SelectionState::new();
    assert!(s.members("tool").is_empty());
    assert_eq!(s.get("tool"), None);
    s.select_all("tool", ["b", "a"]);
    assert_eq!(s.members("tool"), &["b".to_string(), "a".to_string()]);
    assert_eq!(s.get("tool"), Some("b"));
    s.select("tool", "c");
    assert_eq!(s.members("tool"), &["c".to_string()]);
    s.select_all("tool", Vec::<String>::new());
    assert!(s.members("tool").is_empty());
    assert_eq!(s.get("tool"), None);
}

/// **A `Named` selector naming a set emits every member of it** — in the
/// collection's declaration order, not the order named, with the collection's
/// `member_glue` as a real token between consecutive members and never before
/// the first. A name that is not a member is skipped, and one named twice
/// emits once. This is what lets a caller show a collection's members by a
/// fact it has and provenance does not: which acts a character can take here.
#[test]
pub(super) fn collection_named_emits_every_member_of_the_set_in_declaration_order() {
    use crate::projection::{ProjectionMode, ProjectionSegment, SealedKind, SelectionState};
    const YAML: &str = r#"
system_prompt:
  items:
    - kind: collection
      name: tools
      selection: { kind: named, selector: tool }
      member_glue: "\n"
      sections:
        - id: datetime
          content: "datetime def"
        - id: web_search
          content: "web_search def"
        - id: calc
          content: "calc def"
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
    budget: { priority: 100 }
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
    let mut b = Builder::from_yaml(YAML).unwrap();
    // Mock tokenizer: one byte → one token, so the glue "\n" is [10].
    b.tokenize_templates(|s: &str| Ok::<_, ()>(s.bytes().map(u32::from).collect()))
        .unwrap();
    let target = ProjectionTarget {
        layer: b.id_for_layer("dialogue").unwrap(),
        group: b.id_for_group("convo").unwrap(),
        timeline: TimelineId::for_test(1),
    };
    let datetime = b.id_for_system_section("datetime").unwrap();
    let calc = b.id_for_system_section("calc").unwrap();
    let resolver = MockResolver::new();

    let mut sel = SelectionState::new();
    sel.select_all("tool", ["calc", "nonexistent", "datetime", "calc"]);
    let p = b.project_with_selection(target, &resolver, ProjectionMode::Decode, &sel);
    let ids: Vec<SectionId> = p.sealed_sections().map(|s| s.id).collect();
    assert_eq!(ids, vec![datetime, calc], "declaration order, once each");

    let pos = |id: SectionId| {
        p.segments
            .iter()
            .position(
                |s| matches!(s, ProjectionSegment::Sealed(SealedKind::Section(r)) if r.id == id),
            )
            .unwrap()
    };
    let (first, second) = (pos(datetime), pos(calc));
    assert_eq!(second, first + 2, "exactly one segment between the members");
    assert!(
        matches!(&p.segments[first + 1], ProjectionSegment::Generated { tokens, .. } if tokens.as_ref() == &vec![10u32]),
        "the glue is the newline token: {:?}",
        p.segments[first + 1]
    );
    assert!(
        first == 0 || !matches!(&p.segments[first - 1], ProjectionSegment::Generated { .. }),
        "no glue leads the first member"
    );

    // An empty set names nothing, so nothing emits — glue included.
    let mut none = SelectionState::new();
    none.select_all("tool", Vec::<String>::new());
    let p_none = b.project_with_selection(target, &resolver, ProjectionMode::Decode, &none);
    assert_eq!(p_none.sealed_sections().count(), 0);
    assert!(
        !p_none
            .segments
            .iter()
            .any(|s| matches!(s, ProjectionSegment::Generated { tokens, .. } if tokens.as_ref() == &vec![10u32])),
        "glue with no members around it"
    );
}

#[test]
pub(super) fn collection_with_no_qualifying_sections_yields_empty_subset() {
    const YAML: &str = r#"
system_prompt:
  items:
    - kind: section
      id: framing
      content: "framing"
    - kind: collection
      summary:
        chunk: 4
        categorize:
          max_tokens: 256
          system_prompt: Propose a few functional categories for the sections.
          user_prompt: Propose categories for the content above.
        assign:
          max_tokens: 128
          system_prompt: Assign each section to a category by number.
          user_prompt: Assign each section above to a category number.
      name: tools
      selection: { kind: top_k, k: 2 }
      score_threshold: 0.5
      sections:
        - id: a
          content: "a"
        - id: b
          content: "b"
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
    budget: { priority: 100 }
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let framing = b.id_for_system_section("framing").unwrap();

    let resolver = MockResolver::new()
        .with_section_score(b.id_for_system_section("a").unwrap(), 0.1)
        .with_section_score(b.id_for_system_section("b").unwrap(), 0.2);

    let p = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    // Static framing still emits — only the collection drops everything.
    let ids: Vec<SectionId> = p.sealed_sections().map(|s| s.id).collect();
    assert_eq!(ids, vec![framing]);
}
