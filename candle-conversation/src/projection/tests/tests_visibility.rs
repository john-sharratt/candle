//! Tests for what a projection shows, including the masking that hides layers and
//! groups.
//!
//! A section with no content is not projected, and a conversation with an empty
//! system prompt projects its turns alone. A disabled layer contributes nothing to
//! assembly, not only to scoring, and a layer out of tool rounds is left out of
//! those turns alone. The target layer is exempt from its own disablement, and a
//! higher layer is not visible to a lower target.

use super::*;

// —— Append + turn_count (now on resolver) —————————————————————————————————————

#[test]
pub(super) fn append_increments_correctly() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let conv = b.id_for_group("conversation").unwrap();

    let mut resolver = MockResolver::new();
    let i0 = resolver.append(conv);
    let i1 = resolver.append(conv);
    let i2 = resolver.append(conv);

    assert_eq!(i0.0, 0);
    assert_eq!(i1.0, 1);
    assert_eq!(i2.0, 2);
    assert_eq!(resolver.group_turns(conv).len(), 3);
}

#[test]
pub(super) fn turn_count_zero_for_empty_group() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let facts = b.id_for_group("facts").unwrap();
    let resolver = MockResolver::new();
    assert_eq!(resolver.group_turns(facts).len(), 0);
}

// —— Projection: basic visibility ——————————————————————————————————————————————

#[test]
pub(super) fn empty_builder_projection_has_no_turns() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let conv = b.id_for_group("conversation").unwrap();
    let resolver = MockResolver::new();

    let proj = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: conv,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    assert!(proj.sealed_turns().next().is_none());
}

#[test]
pub(super) fn system_prompt_sections_always_emitted() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let conv = b.id_for_group("conversation").unwrap();
    let resolver = MockResolver::new();

    let proj = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: conv,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    assert_eq!(proj.sealed_sections().count(), 2);
}

/// **A section with no content is not projected.** Ingest never seals one —
/// it tokenises to nothing — so a segment naming it is a section no walk can
/// place: the assembler counts it skipped, the projection is never recorded as
/// complete, and every reprojection rebuilds the slot, re-sealing the whole
/// active turn. A conversation opened with an empty system prompt (an API
/// request that carries none) projects its turns and nothing else.
#[test]
pub(super) fn an_empty_section_is_not_projected() {
    let project = |prompt: &str| {
        let b = Builder::for_plain_prompt(prompt, SectionId::new(7));
        let target = ProjectionTarget {
            layer: b.id_for_layer("dialogue").unwrap(),
            group: b.id_for_group("primary_conversation").unwrap(),
            timeline: TimelineId::for_test(1),
        };
        let ids: Vec<SectionId> = b
            .project(target, &MockResolver::new())
            .sealed_sections()
            .map(|s| s.id)
            .collect();
        ids
    };
    assert_eq!(project(""), Vec::<SectionId>::new());
    assert_eq!(
        project("You are a helpful assistant."),
        vec![SectionId::new(7)]
    );
}

#[test]
pub(super) fn turns_appear_after_append() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let conv = b.id_for_group("conversation").unwrap();

    let mut resolver = MockResolver::new();
    resolver.append(conv);
    resolver.append(conv);
    resolver.append(conv);

    let proj = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: conv,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    assert_eq!(proj.sealed_turns().count(), 3);
}

/// By design, a projected turn carries its resolved conversation: the timeline is
/// stamped ONCE at projection (from the target-aware resolver) onto every emitted
/// turn, so no downstream consumer re-derives `group → timeline` — which is what
/// once let the reproject pick the wrong conversation and drop a slot's history.
#[test]
pub(super) fn projected_turn_carries_its_resolved_timeline() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let conv = b.id_for_group("conversation").unwrap();

    // The mock gives each group its own conversation, mirroring the real model.
    let tl = MockResolver::timeline_of(conv);
    let mut resolver = MockResolver::new();
    resolver.append(conv);
    resolver.append(conv);

    let proj = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: conv,
            timeline: tl,
        },
        &resolver,
    );

    let turns: Vec<_> = proj.sealed_turns().collect();
    assert_eq!(turns.len(), 2);
    for t in turns {
        // The conversation is carried on the turn — not re-derived downstream.
        assert_eq!(t.timeline, Some(tl));
        assert_eq!(
            t.key(),
            Some(TurnKey {
                timeline: tl,
                index: t.index()
            })
        );
    }
}

// —— Masking ———————————————————————————————————————————————————————————————————

#[test]
pub(super) fn lower_layers_visible_for_dialogue_target() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let ground = b.id_for_layer("ground").unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let facts = b.id_for_group("facts").unwrap();
    let conv = b.id_for_group("conversation").unwrap();

    let mut resolver = MockResolver::new();
    resolver.append(facts);
    resolver.append(facts);
    resolver.append(conv);

    let proj = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: conv,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let groups: Vec<GroupId> = groups_in_order(proj.sealed_turns());
    assert!(groups.contains(&facts), "ground/facts should be visible");
    assert!(groups.contains(&conv), "target conv should be visible");
    let _ = ground;
}

/// **`--disable-layer` takes a layer out of the ASSEMBLY, not just out of
/// scoring.**
///
/// The scoring-time guards keep a disabled layer's turns from competing in the
/// belief gather, but a `Sequence` (recency) group is never scored at all — it
/// emits its window by position. So the exclusion is restated where the
/// assembly walks layers, and this is the test that holds it there: with
/// `ground` disabled its turns must not reach the projection, in exactly the
/// configuration `lower_layers_visible_for_dialogue_target` asserts the opposite
/// of. Without the assembly-side check that test and this one can both pass
/// while a recency group quietly keeps emitting.
#[test]
pub(super) fn a_disabled_layer_contributes_nothing_to_the_projection() {
    let mut b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let facts = b.id_for_group("facts").unwrap();
    let conv = b.id_for_group("conversation").unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();

    assert!(
        b.set_layer_gathered("ground", false),
        "the schema declares a `ground` layer, so the mutator must find it",
    );

    let mut resolver = MockResolver::new();
    resolver.append(facts);
    resolver.append(facts);
    resolver.append(conv);

    let proj = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: conv,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let groups: Vec<GroupId> = groups_in_order(proj.sealed_turns());
    assert!(
        !groups.contains(&facts),
        "a disabled layer's group reached the projection: {groups:?}",
    );
    assert!(
        groups.contains(&conv),
        "disabling one layer must not disturb the target layer",
    );
}

/// **A layer that sits out tool rounds is projected into every other turn.**
/// `in_tool_rounds: false` on `ground`: an ordinary turn sees its facts, a turn
/// marked with [`TOOL_ROUND_SELECTOR`] does not, and the target layer is
/// untouched either way. The repo_map layer declares this so its
/// summarize-this-folder requests cannot become the question a tool round
/// answers.
#[test]
pub(super) fn a_layer_out_of_tool_rounds_is_left_out_only_there() {
    let yaml = SIMPLE_YAML.replace(
        "  - name: ground\n    window: 8000\n",
        "  - name: ground\n    window: 8000\n    in_tool_rounds: false\n",
    );
    assert_ne!(yaml, SIMPLE_YAML, "the fixture must carry the flag");
    let b = Builder::from_yaml(&yaml).unwrap();
    let facts = b.id_for_group("facts").unwrap();
    let conv = b.id_for_group("conversation").unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();

    let mut resolver = MockResolver::new();
    resolver.append(facts);
    resolver.append(conv);
    let target = ProjectionTarget {
        layer: dialogue,
        group: conv,
        timeline: TimelineId::for_test(1),
    };
    let groups_for = |selection: &SelectionState| -> Vec<GroupId> {
        let proj = b.project_with_selection(target, &resolver, ProjectionMode::Decode, selection);
        groups_in_order(proj.sealed_turns())
    };

    let ordinary = groups_for(&SelectionState::new());
    assert!(
        ordinary.contains(&facts),
        "an ordinary turn sees the layer: {ordinary:?}"
    );

    let mut round = SelectionState::new();
    round.set_optional(TOOL_ROUND_SELECTOR, OptionalState::Present);
    let in_round = groups_for(&round);
    assert!(
        !in_round.contains(&facts),
        "a tool round must leave it out: {in_round:?}"
    );
    assert!(in_round.contains(&conv), "the target layer is untouched");

    // A layer that does not declare the flag stays in tool rounds.
    let plain = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let proj = plain.project_with_selection(target, &resolver, ProjectionMode::Decode, &round);
    assert!(groups_in_order(proj.sealed_turns()).contains(&facts));
}

/// The target layer is exempt from its own disablement. Disabling the layer you
/// are projecting FOR is not a meaningful request, and answering it with an
/// empty context is a worse outcome than ignoring the flag — the same exemption,
/// for the same reason, that the diagnostic `layer_toggle` kill switch carries.
#[test]
pub(super) fn the_target_layer_is_still_projected_when_disabled() {
    let mut b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let conv = b.id_for_group("conversation").unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    assert!(b.set_layer_gathered("dialogue", false));

    let mut resolver = MockResolver::new();
    resolver.append(conv);

    let proj = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: conv,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let groups: Vec<GroupId> = groups_in_order(proj.sealed_turns());
    assert!(
        groups.contains(&conv),
        "the target layer must project even when disabled — an empty context is \
         a worse answer than ignoring the flag",
    );
}

#[test]
pub(super) fn higher_layer_not_visible_for_lower_target() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let ground = b.id_for_layer("ground").unwrap();
    let facts = b.id_for_group("facts").unwrap();
    let conv = b.id_for_group("conversation").unwrap();

    let mut resolver = MockResolver::new();
    resolver.append(facts);
    resolver.append(conv); // conv is in dialogue (higher layer)

    let proj = b.project(
        ProjectionTarget {
            layer: ground,
            group: facts,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let groups: Vec<GroupId> = groups_in_order(proj.sealed_turns());
    assert!(
        !groups.contains(&conv),
        "dialogue group must NOT be visible from ground target"
    );
}

#[test]
pub(super) fn sibling_group_in_target_layer_not_visible() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer_a
    window: 9000
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
    groups:
      - id: group_a1
        selection: { kind: top_k, k: 5 }
      - id: group_a2
        selection: { kind: top_k, k: 5 }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer_a = b.id_for_layer("layer_a").unwrap();
    let g1 = b.id_for_group("group_a1").unwrap();
    let g2 = b.id_for_group("group_a2").unwrap();

    let mut resolver = MockResolver::new();
    resolver.append(g1);
    resolver.append(g2);

    let proj = b.project(
        ProjectionTarget {
            layer: layer_a,
            group: g1,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let groups = groups_in_order(proj.sealed_turns());
    assert!(groups.contains(&g1));
    assert!(
        !groups.contains(&g2),
        "sibling group_a2 must not be visible"
    );
}
