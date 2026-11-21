//! Tests for the gates that decide which members of a collection a projection
//! shows.
//!
//! A force pin names one member and overrides belief, and a withheld member is
//! never shown however it scores. A mandatory member rides on top of the top-k
//! rather than taking one of its slots. An offered collection ranks only what the
//! room offers, and a provenance with no opinion shows everything offered rather
//! than nothing.

use super::*;

#[test]
pub(super) fn top_k_force_tool_pin_overrides_belief_then_falls_back() {
    use crate::projection::{ProjectionMode, SelectionState, FORCE_TOOL_SELECTOR};
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
    let target = ProjectionTarget {
        layer: dialogue,
        group: convo,
        timeline: TimelineId::for_test(1),
    };
    // Hostile scores: tool_b and tool_d rank highest, so ordinary top-2 belief
    // would keep exactly those. tool_c is deliberately the WORST-scored member.
    let resolver = MockResolver::new()
        .with_section_score(tool_a, 0.3)
        .with_section_score(tool_b, 0.9)
        .with_section_score(tool_c, 0.05)
        .with_section_score(tool_d, 0.8);

    // Pin tool_c by name via FORCE_TOOL_SELECTOR → exactly that one member is
    // emitted, score notwithstanding, and the belief top-2 is skipped entirely.
    let mut sel = SelectionState::new();
    sel.select(FORCE_TOOL_SELECTOR, "tool_c");
    let p = b.project_with_selection(target, &resolver, ProjectionMode::Decode, &sel);
    let ids: Vec<SectionId> = p.sealed_sections().map(|s| s.id).collect();
    assert_eq!(
        ids,
        vec![framing, tools_intro, tool_c, tools_outro],
        "force_tool pin must emit exactly the pinned member, overriding belief"
    );

    // Unset pin → ordinary belief top-2 (tool_b, tool_d by score).
    let p_unset = b.project_with_selection(
        target,
        &resolver,
        ProjectionMode::Decode,
        &SelectionState::default(),
    );
    let unset_ids: Vec<SectionId> = p_unset.sealed_sections().map(|s| s.id).collect();
    assert_eq!(
        unset_ids,
        vec![framing, tools_intro, tool_b, tool_d, tools_outro],
        "unset force_tool pin must fall through to belief top-2"
    );

    // Pin names a non-member → no override; belief top-2 still governs.
    let mut sel_bad = SelectionState::new();
    sel_bad.select(FORCE_TOOL_SELECTOR, "nonexistent");
    let p_bad = b.project_with_selection(target, &resolver, ProjectionMode::Decode, &sel_bad);
    let bad_ids: Vec<SectionId> = p_bad.sealed_sections().map(|s| s.id).collect();
    assert_eq!(
        bad_ids,
        vec![framing, tools_intro, tool_b, tool_d, tools_outro],
        "unknown force_tool member must fall through to belief, not blank the collection"
    );
}

/// **A withheld member is never shown**, however it scores: the top-k is
/// filled from the rest, so the prompt holds only tools the round can call.
#[test]
pub(super) fn a_withheld_member_is_never_projected() {
    use crate::projection::{ProjectionMode, SelectionState};
    let b = Builder::from_yaml(COLLECTION_YAML).unwrap();
    let target = ProjectionTarget {
        layer: b.id_for_layer("dialogue").unwrap(),
        group: b.id_for_group("convo").unwrap(),
        timeline: TimelineId::for_test(1),
    };
    let id = |name: &str| b.id_for_system_section(name).unwrap();
    let resolver = MockResolver::new()
        .with_section_score(id("tool_a"), 0.3)
        .with_section_score(id("tool_b"), 0.9)
        .with_section_score(id("tool_c"), 0.05)
        .with_section_score(id("tool_d"), 0.8);
    let mut sel = SelectionState::new();
    sel.withhold_members(["tool_b"]);
    let p = b.project_with_selection(target, &resolver, ProjectionMode::Decode, &sel);
    let ids: Vec<SectionId> = p.sealed_sections().map(|s| s.id).collect();
    assert_eq!(
        ids,
        vec![
            id("framing"),
            id("tools_intro"),
            id("tool_a"),
            id("tool_d"),
            id("tools_outro")
        ],
        "the best-scored member is held out and the next fills its place"
    );
}

/// A mandatory member is emitted on every belief-driven projection and does NOT
/// take one of the budget's slots: the top-k is filled from the other members.
#[test]
pub(super) fn a_mandatory_member_adds_to_the_top_k_rather_than_taking_a_slot() {
    use crate::projection::{ProjectionMode, SelectionState, FORCE_TOOL_SELECTOR};
    let mut b = Builder::from_yaml(COLLECTION_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let framing = b.id_for_system_section("framing").unwrap();
    let tools_intro = b.id_for_system_section("tools_intro").unwrap();
    let tool_a = b.id_for_system_section("tool_a").unwrap();
    let tool_b = b.id_for_system_section("tool_b").unwrap();
    let tool_c = b.id_for_system_section("tool_c").unwrap();
    let tool_d = b.id_for_system_section("tool_d").unwrap();
    let tools_outro = b.id_for_system_section("tools_outro").unwrap();
    let tools = b.id_for_system_collection("tools").unwrap();
    let target = ProjectionTarget {
        layer: dialogue,
        group: convo,
        timeline: TimelineId::for_test(1),
    };
    // top_k 2 over these scores alone keeps tool_b and tool_d; tool_c is the
    // worst-scored member and would never be selected on belief.
    let resolver = MockResolver::new()
        .with_section_score(tool_a, 0.3)
        .with_section_score(tool_b, 0.9)
        .with_section_score(tool_c, 0.05)
        .with_section_score(tool_d, 0.8);
    let project = |b: &Builder, sel: &SelectionState| -> Vec<SectionId> {
        b.project_with_selection(target, &resolver, ProjectionMode::Decode, sel)
            .sealed_sections()
            .map(|s| s.id)
            .collect()
    };

    b.set_collection_member_mandatory(tools, tool_c).unwrap();
    assert_eq!(
        project(&b, &SelectionState::default()),
        vec![framing, tools_intro, tool_b, tool_c, tool_d, tools_outro],
        "the mandatory member is added to the belief top-2, in catalog order"
    );

    // Making the top scorer mandatory frees its slot: the top-2 is now drawn
    // from the remaining members, so the next best (tool_a) comes in.
    b.set_collection_member_mandatory(tools, tool_b).unwrap();
    assert_eq!(
        project(&b, &SelectionState::default()),
        vec![
            framing,
            tools_intro,
            tool_a,
            tool_b,
            tool_c,
            tool_d,
            tools_outro
        ],
        "mandatory members never count against the budget"
    );

    // A force pin still emits exactly the members it names.
    let mut pin = SelectionState::new();
    pin.select(FORCE_TOOL_SELECTOR, "tool_a");
    assert_eq!(
        project(&b, &pin),
        vec![framing, tools_intro, tool_a, tools_outro],
        "a force pin overrides mandatory members like it overrides belief"
    );
}

/// An `Offered` collection ranks only what the runtime selector names: the
/// belief top-k is drawn from the offered members, an unoffered member never
/// shows however well it scores, a mandatory offered member rides on top, and a
/// provenance with no opinion shows everything offered rather than nothing.
#[test]
pub(super) fn an_offered_collection_ranks_only_what_the_room_offers() {
    use crate::projection::{ProjectionMode, SelectionRule, SelectionState};
    let mut b = Builder::from_yaml(COLLECTION_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let framing = b.id_for_system_section("framing").unwrap();
    let tools_intro = b.id_for_system_section("tools_intro").unwrap();
    let tool_a = b.id_for_system_section("tool_a").unwrap();
    let tool_b = b.id_for_system_section("tool_b").unwrap();
    let tool_c = b.id_for_system_section("tool_c").unwrap();
    let tool_d = b.id_for_system_section("tool_d").unwrap();
    let tools_outro = b.id_for_system_section("tools_outro").unwrap();
    let tools = b.id_for_system_collection("tools").unwrap();
    b.set_collection_selection(
        "tools",
        SelectionRule::Offered {
            selector: "room".to_string(),
            k: 2,
        },
    )
    .unwrap();
    let target = ProjectionTarget {
        layer: dialogue,
        group: convo,
        timeline: TimelineId::for_test(1),
    };
    let scored = MockResolver::new()
        .with_section_score(tool_a, 0.3)
        .with_section_score(tool_b, 0.9)
        .with_section_score(tool_c, 0.05)
        .with_section_score(tool_d, 0.8);
    let project = |b: &Builder, resolver: &MockResolver, offered: &[&str]| -> Vec<SectionId> {
        let mut sel = SelectionState::new();
        sel.select_all("room", offered.iter().copied());
        b.project_with_selection(target, resolver, ProjectionMode::Decode, &sel)
            .sealed_sections()
            .map(|s| s.id)
            .collect()
    };

    // tool_b scores best but the room does not offer it: the top-2 of what is
    // offered is tool_d and tool_a.
    assert_eq!(
        project(&b, &scored, &["tool_a", "tool_c", "tool_d"]),
        vec![framing, tools_intro, tool_a, tool_d, tools_outro],
        "an unoffered member is never shown, and the top-2 is drawn from the offered"
    );

    // A mandatory offered member adds to the selection; unoffered, it is absent.
    b.set_collection_member_mandatory(tools, tool_c).unwrap();
    assert_eq!(
        project(&b, &scored, &["tool_a", "tool_c", "tool_d"]),
        vec![framing, tools_intro, tool_a, tool_c, tool_d, tools_outro]
    );
    assert_eq!(
        project(&b, &scored, &["tool_b", "tool_d"]),
        vec![framing, tools_intro, tool_b, tool_d, tools_outro],
        "a mandatory member the room does not offer is not shown"
    );

    // No selector set → nothing offered → nothing shown.
    assert_eq!(
        project(&b, &scored, &[]),
        vec![framing, tools_intro, tools_outro]
    );

    // Cold scores: provenance has no opinion, so everything offered shows.
    assert_eq!(
        project(&b, &MockResolver::new(), &["tool_a", "tool_b", "tool_d"]),
        vec![framing, tools_intro, tool_a, tool_b, tool_d, tools_outro],
        "no provenance signal shows the whole offer rather than an empty section"
    );
}

/// A filter that drops a mandatory member drops it outright — the mandatory
/// flag never resurrects a tool the tools mode excluded.
#[test]
pub(super) fn a_retained_filter_drops_a_mandatory_member() {
    use crate::projection::{ProjectionMode, SelectionState};
    let mut b = Builder::from_yaml(COLLECTION_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let tool_a = b.id_for_system_section("tool_a").unwrap();
    let tool_b = b.id_for_system_section("tool_b").unwrap();
    let tool_c = b.id_for_system_section("tool_c").unwrap();
    let tool_d = b.id_for_system_section("tool_d").unwrap();
    let tools = b.id_for_system_collection("tools").unwrap();
    let target = ProjectionTarget {
        layer: dialogue,
        group: convo,
        timeline: TimelineId::for_test(1),
    };
    let resolver = MockResolver::new()
        .with_section_score(tool_a, 0.3)
        .with_section_score(tool_b, 0.9)
        .with_section_score(tool_c, 0.05)
        .with_section_score(tool_d, 0.8);
    b.set_collection_member_mandatory(tools, tool_c).unwrap();
    let keep: std::collections::HashSet<String> = ["tool_a", "tool_b", "tool_d"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    b.retain_collection_sections("tools", &keep).unwrap();
    let ids: Vec<SectionId> = b
        .project_with_selection(
            target,
            &resolver,
            ProjectionMode::Decode,
            &SelectionState::default(),
        )
        .sealed_sections()
        .map(|s| s.id)
        .collect();
    assert!(
        !ids.contains(&tool_c),
        "a filtered-out mandatory member stays out"
    );
    assert!(
        ids.contains(&tool_b) && ids.contains(&tool_d),
        "belief top-2 unchanged"
    );
}

#[test]
pub(super) fn mandatory_rejects_a_section_outside_the_collection() {
    use crate::projection::ConstructionError;
    let mut b = Builder::from_yaml(COLLECTION_YAML).unwrap();
    let tools = b.id_for_system_collection("tools").unwrap();
    let framing = b.id_for_system_section("framing").unwrap();
    assert!(matches!(
        b.set_collection_member_mandatory(tools, framing),
        Err(ConstructionError::MandatoryMember(_))
    ));
}

#[test]
pub(super) fn named_selection_rejects_empty_selector() {
    use crate::projection::ConstructionError;
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
      selection: { kind: named }
      sections:
        - id: datetime
          content: "datetime def"
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
    let err = Builder::from_yaml(YAML).unwrap_err();
    assert!(
        matches!(err, ConstructionError::EmptyNamedSelector { .. }),
        "named without a selector must fail construction, got {err:?}"
    );
}

#[test]
pub(super) fn collection_always_visible_emits_every_section() {
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
      name: all
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

    let resolver = MockResolver::new();
    let p = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    assert_eq!(
        p.sealed_sections().count(),
        2,
        "AlwaysVisible collection emits all"
    );
}
