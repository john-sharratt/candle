//! Tests for when a collection's catalog summary and its notes are shown.
//!
//! A summary is emitted before the selected members when only some were selected,
//! and omitted when all were. It sits outside any structural markers, and an
//! unsealed summary is left out rather than projected empty. The generic framing
//! survives a projection that selected nothing, since the model must still be told
//! the facility exists.

use super::*;

pub(super) const COLLECTION_YAML: &str = r#"
system_prompt:
  items:
    - kind: section
      id: framing
      content: "framing"
    - kind: section
      id: tools_intro
      content: "tools_intro"
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
      sections:
        - id: tool_a
          content: "A"
        - id: tool_b
          content: "B"
        - id: tool_c
          content: "C"
        - id: tool_d
          content: "D"
    - kind: section
      id: tools_outro
      content: "tools_outro"
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

/// Like `COLLECTION_YAML` but the collection's `top_k` (5) exceeds its member
/// count (2), so every projection selects *all* tools — the all-selected case.
pub(super) const ALL_TOOLS_FIT_YAML: &str = r#"
system_prompt:
  items:
    - kind: section
      id: tools_intro
      content: "tools_intro"
    - kind: collection
      summary:
        chunk: 4
        categorize:
          max_tokens: 256
          system_prompt: Propose categories.
          user_prompt: Propose categories.
        assign:
          max_tokens: 128
          system_prompt: Assign by number.
          user_prompt: Assign by number.
      name: tools
      selection: { kind: top_k, k: 5 }
      sections:
        - id: tool_a
          content: "A"
        - id: tool_b
          content: "B"
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

/// Partial selection (top-2 of 4) + a sealed summary → the summary section is
/// emitted, just before the selected tool members.
#[test]
pub(super) fn collection_partial_selection_emits_summary_before_members() {
    let mut b = Builder::from_yaml(COLLECTION_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let tools = b.id_for_system_collection("tools").unwrap();
    let summary = SectionId::reserved(Reserved::ToolSummary);
    b.set_collection_summary_section(tools, summary).unwrap();
    let tools_intro = b.id_for_system_section("tools_intro").unwrap();
    let tool_b = b.id_for_system_section("tool_b").unwrap();
    let tool_d = b.id_for_system_section("tool_d").unwrap();

    let resolver = MockResolver::new()
        .with_section_score(tool_b, 0.9)
        .with_section_score(tool_d, 0.8)
        .with_section_tokens(summary, 100); // sealed

    let p = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let ids: Vec<SectionId> = p.sealed_sections().map(|s| s.id).collect();
    let sp = ids
        .iter()
        .position(|&i| i == summary)
        .unwrap_or_else(|| panic!("summary not emitted on partial selection: {ids:?}"));
    let intro_pos = ids.iter().position(|&i| i == tools_intro).unwrap();
    let tp = ids.iter().position(|&i| i == tool_b).unwrap();
    // After the static intro, before the tools.
    assert!(
        intro_pos < sp && sp < tp,
        "want intro < summary < tools: {ids:?}"
    );
    assert!(ids.contains(&tool_d));
}

/// **A turn's notes follow the summary, in order, before the members**, and a
/// note that was never sealed is left out rather than projected empty.
#[test]
pub(super) fn notes_on_a_summary_project_right_after_it() {
    let mut b = Builder::from_yaml(COLLECTION_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let tools = b.id_for_system_collection("tools").unwrap();
    let summary = SectionId::reserved(Reserved::ToolSummary);
    b.set_collection_summary_section(tools, summary).unwrap();
    let tool_b = b.id_for_system_section("tool_b").unwrap();
    let (first, second, unsealed) = (
        SectionId::reserved(Reserved::ToolUnavailable(0)),
        SectionId::reserved(Reserved::ToolUnavailable(3)),
        SectionId::reserved(Reserved::ToolUnavailable(4)),
    );

    let resolver = MockResolver::new()
        .with_section_score(tool_b, 0.9)
        .with_section_tokens(summary, 100)
        .with_section_tokens(first, 12)
        .with_section_tokens(second, 12);
    let target = ProjectionTarget {
        layer: dialogue,
        group: convo,
        timeline: TimelineId::for_test(1),
    };
    let mut selection = SelectionState::new();
    selection.note_summary(summary, vec![first, second, unsealed]);
    let p = b.project_with_selection(target, &resolver, ProjectionMode::Decode, &selection);
    let ids: Vec<SectionId> = p.sealed_sections().map(|s| s.id).collect();
    let at = ids.iter().position(|&i| i == summary).unwrap();
    assert_eq!(&ids[at..at + 3], &[summary, first, second], "{ids:?}");
    assert!(!ids.contains(&unsealed), "{ids:?}");
    let tool = ids.iter().position(|&i| i == tool_b).unwrap();
    assert!(at + 2 < tool, "the notes come before the members: {ids:?}");

    let none = b.project_with_selection(
        target,
        &resolver,
        ProjectionMode::Decode,
        &SelectionState::new(),
    );
    assert!(!none.sealed_sections().any(|s| s.id == first));
}

/// When the collection selects all its members (top-k ≥ member count), nothing
/// was dropped, so the summary is omitted even though it is sealed.
#[test]
pub(super) fn collection_all_selected_omits_summary() {
    let mut b = Builder::from_yaml(ALL_TOOLS_FIT_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let tools = b.id_for_system_collection("tools").unwrap();
    let summary = SectionId::reserved(Reserved::ToolSummary);
    b.set_collection_summary_section(tools, summary).unwrap();
    let tool_a = b.id_for_system_section("tool_a").unwrap();
    let tool_b = b.id_for_system_section("tool_b").unwrap();

    let resolver = MockResolver::new()
        .with_section_score(tool_a, 0.9)
        .with_section_score(tool_b, 0.8)
        .with_section_tokens(summary, 100); // sealed, but all selected

    let p = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let ids: Vec<SectionId> = p.sealed_sections().map(|s| s.id).collect();
    assert!(
        !ids.contains(&summary),
        "summary must be omitted when all tools are selected: {ids:?}"
    );
    assert!(ids.contains(&tool_a) && ids.contains(&tool_b));
}

/// Partial selection but the summary section was never sealed
/// (`section_token_count == 0`) → it is not emitted (no phantom segment).
#[test]
pub(super) fn collection_partial_but_unsealed_summary_omitted() {
    let mut b = Builder::from_yaml(COLLECTION_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let tools = b.id_for_system_collection("tools").unwrap();
    let summary = SectionId::reserved(Reserved::ToolSummary);
    b.set_collection_summary_section(tools, summary).unwrap();
    let tool_b = b.id_for_system_section("tool_b").unwrap();
    let tool_d = b.id_for_system_section("tool_d").unwrap();

    // Partial, but no `with_section_tokens` → summary is unsealed.
    let resolver = MockResolver::new()
        .with_section_score(tool_b, 0.9)
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
    assert!(
        !ids.contains(&summary),
        "an unsealed summary must not be emitted: {ids:?}"
    );
}

/// With a structural open marker (`<tools>`), the catalog summary emits OUTSIDE
/// it — after the preamble section, before the open template — not inside the
/// markers next to the members.
#[test]
pub(super) fn collection_summary_emits_outside_structural_markers() {
    use super::super::project::{ProjectionSegment, SealedKind};
    use candle_transformers::models::dialect::Dialect;

    let yaml = r#"
system_prompt:
  items:
    - kind: section
      id: tools_overview
      depends_on: tools
      content: "overview"
    - kind: template
      id: tools_open
      dialect: tool_block_open
      depends_on: tools
    - kind: collection
      name: tools
      selection: { kind: top_k, k: 1 }
      summary:
        chunk: 4
        categorize:
          max_tokens: 256
          system_prompt: x
          user_prompt: x
        assign:
          max_tokens: 128
          system_prompt: x
          user_prompt: x
      sections:
        - id: tool_a
          content: "A"
        - id: tool_b
          content: "B"
        - id: tool_c
          content: "C"
    - kind: template
      id: tools_close
      dialect: tool_block_close
      depends_on: tools
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
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
    let dlct = Dialect::chat_ml();
    let mut b = Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&dlct)).unwrap();
    b.tokenize_templates::<std::convert::Infallible, _>(|s| Ok(s.bytes().map(u32::from).collect()))
        .unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let tools = b.id_for_system_collection("tools").unwrap();
    let summary = SectionId::reserved(Reserved::ToolSummary);
    b.set_collection_summary_section(tools, summary).unwrap();
    let tool_a = b.id_for_system_section("tool_a").unwrap();

    // top_k=1 over 3 members → partial; summary sealed.
    let resolver = MockResolver::new()
        .with_section_score(tool_a, 0.9)
        .with_section_tokens(summary, 100);

    let proj = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    // Template sections seal like any other now, so they are identified by
    // section id rather than by a `Generated` run's name.
    let tools_open = b.id_for_system_section("tools_open").unwrap();
    let order: Vec<String> = proj
        .segments
        .iter()
        .map(|seg| match seg {
            ProjectionSegment::Generated { identity, .. } => identity.name.clone(),
            ProjectionSegment::Sealed(SealedKind::Section(s)) if s.id == summary => {
                "SUMMARY".to_string()
            }
            ProjectionSegment::Sealed(SealedKind::Section(s)) if s.id == tools_open => {
                "tools_open".to_string()
            }
            _ => "_".to_string(),
        })
        .collect();
    let sp = order
        .iter()
        .position(|s| s == "SUMMARY")
        .unwrap_or_else(|| panic!("summary not emitted: {order:?}"));
    let op = order
        .iter()
        .position(|s| s == "tools_open")
        .unwrap_or_else(|| panic!("tools_open not emitted: {order:?}"));
    // Summary sits OUTSIDE the markers: before `<tools>` opens.
    assert!(
        sp < op,
        "summary must emit before the <tools> open marker: {order:?}"
    );
}

/// The GENERIC half of a collection's framing — the prose that says the facility
/// exists, and the catalog summary that names its members — must survive a
/// projection in which provenance selected NOTHING. Only the detailed per-member
/// definitions, and the structural markers wrapping them, follow selection.
///
/// This is the regression that made zend's first turn of every conversation
/// unusable: scores are all zero at a turn's opening projection (it runs before
/// the turn is prefilled, so there is no decode-Q), the `min` budget declines to
/// force-fill from an all-zero belief, and every tool-related item hung off
/// `depends_on` — "materialised ≥1 this pass". The model was told it had no
/// tools and answered from memory.
#[test]
pub(super) fn configured_gate_keeps_overview_and_summary_when_nothing_is_selected() {
    use super::super::project::{ProjectionSegment, SealedKind};
    use candle_transformers::models::dialect::Dialect;

    let yaml = r#"
system_prompt:
  items:
    - kind: section
      id: tools_overview
      depends_on_configured: tools
      content: "overview"
    - kind: section
      id: no_tools
      depends_on_unconfigured: tools
      content: "no tools"
    - kind: template
      id: tools_open
      dialect: tool_block_open
      depends_on: tools
    - kind: collection
      name: tools
      selection: { kind: top_k, k: 1 }
      score_threshold: 10.0
      sections:
        - id: tool_a
          content: "A"
        - id: tool_b
          content: "B"
    - kind: template
      id: tools_close
      dialect: tool_block_close
      depends_on: tools
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
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
    let dlct = Dialect::chat_ml();
    let mut b = Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&dlct)).unwrap();
    b.tokenize_templates::<std::convert::Infallible, _>(|s| Ok(s.bytes().map(u32::from).collect()))
        .unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let tools = b.id_for_system_collection("tools").unwrap();
    let summary = SectionId::reserved(Reserved::ToolSummary);
    b.set_collection_summary_section(tools, summary).unwrap();
    let overview = b.id_for_system_section("tools_overview").unwrap();
    let no_tools = b.id_for_system_section("no_tools").unwrap();
    let tools_open = b.id_for_system_section("tools_open").unwrap();

    // Every member scores 0 — the opening-projection condition. The threshold
    // keeps them all out, so the collection materialises nothing.
    let resolver = MockResolver::new().with_section_tokens(summary, 100);
    let proj = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let ids: Vec<SectionId> = proj
        .segments
        .iter()
        .filter_map(|seg| match seg {
            ProjectionSegment::Sealed(SealedKind::Section(s)) => Some(s.id),
            _ => None,
        })
        .collect();

    // The generic half survives: the facility is configured, so say so.
    assert!(
        ids.contains(&overview),
        "overview must emit with tools configured but none selected: {ids:?}"
    );
    assert!(
        ids.contains(&summary),
        "catalog summary must emit with tools configured but none selected: {ids:?}"
    );
    // The detailed half does not: no members, so no markers to wrap them.
    assert!(
        !ids.contains(&tools_open),
        "the <tools> marker must stay gated on selection: {ids:?}"
    );
    // And the "tools are off" variant must NOT fire — tools are on, just unpicked.
    assert!(
        !ids.contains(&no_tools),
        "the no-tools variant must key on configuration, not selection: {ids:?}"
    );
}

/// `depends_on_absent` is the inverse of `depends_on`: its section emits only
/// when the collection materialised zero members (the no-tools variant), and is
/// suppressed when the collection has members.
#[test]
pub(super) fn depends_on_absent_emits_only_when_collection_empty() {
    const YAML: &str = r#"
system_prompt:
  items:
    - kind: section
      id: with_tools
      depends_on: tools
      content: "WITH"
    - kind: section
      id: no_tools
      depends_on_absent: tools
      content: "NO"
    - kind: collection
      name: tools
      selection: { kind: top_k, k: 2 }
      score_threshold: 0.5
      summary:
        chunk: 4
        categorize:
          max_tokens: 256
          system_prompt: x
          user_prompt: x
        assign:
          max_tokens: 128
          system_prompt: x
          user_prompt: x
      sections:
        - id: t1
          content: "tool one"
layers:
  - name: dialogue
    window: 1000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: c
          user_prompt: c
        assistant:
          system_prompt: c
          user_prompt: c
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let with = b.id_for_system_section("with_tools").unwrap();
    let no = b.id_for_system_section("no_tools").unwrap();
    let t1 = b.id_for_system_section("t1").unwrap();
    let target = ProjectionTarget {
        layer: dialogue,
        group: convo,
        timeline: TimelineId::for_test(1),
    };

    // Tools present (t1 above threshold) → `with_tools` emits, `no_tools` not.
    let present = MockResolver::new().with_section_score(t1, 0.9);
    let ids: Vec<SectionId> = b
        .project(target, &present)
        .sealed_sections()
        .map(|s| s.id)
        .collect();
    assert!(
        ids.contains(&with) && !ids.contains(&no),
        "tools present → only the tool-aware grounding: {ids:?}"
    );

    // No tools (t1 below threshold → collection empty) → `no_tools` emits.
    let empty = MockResolver::new();
    let ids: Vec<SectionId> = b
        .project(target, &empty)
        .sealed_sections()
        .map(|s| s.id)
        .collect();
    assert!(
        ids.contains(&no) && !ids.contains(&with),
        "no tools → only the no-tools grounding: {ids:?}"
    );
}
