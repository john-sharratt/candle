//! Tests for belief-driven turn selection, and for the default a collection falls
//! back to.
//!
//! A turn group selects its turns by score against its threshold and top-k, and a
//! group whose turns all miss the threshold contributes nothing. A collection's
//! default is injected only when its selection is otherwise empty, so the
//! collection still contributes one member.

use super::*;

#[test]
pub(super) fn default_selection_is_always_visible() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: 49000
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
      - id: grp
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    for _ in 0..5 {
        resolver.append(grp);
    }

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    assert_eq!(
        proj.sealed_turns().count(),
        5,
        "default selection includes all turns"
    );
}

// ── Tag-based selection default ──────────────────────────────────────────────
//
// A group/collection may declare `default: { tag: … }`. When belief, scores, and
// the selection rule all pick nothing, the tagged member is injected so the
// group — and its layer — never drops out of the projection. The default fires
// only on an otherwise-empty selection, so a group with real picks ignores it.

/// A group whose every turn scores below its threshold selects nothing on the
/// rule path; with a `default` declared and its tag bound to a turn, that turn
/// is injected so the group survives instead of vanishing at the empty-group
/// retain.
/// One `top_k(2)` turn group behind a `score_threshold` of 0.5 — the shape the
/// threshold tests need. A turn group declares no fallback member: passing the
/// threshold is the only way in.
pub(super) const TURN_GROUP_TOPK_YAML: &str = r#"
system_prompt:
  sections:
    - id: frame
      content: "frame"
layers:
  - name: ground
    window: 8000
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
    budget:
      priority: 40
    groups:
      - id: structure
        selection: { kind: top_k, k: 2 }
        score_threshold: 0.5
"#;

/// **A turn group that qualifies nothing contributes nothing.**
///
/// The threshold is the whole gate. This used to inject the group's declared
/// `default` member with a score synthesised from the layer/group thresholds —
/// a sentinel built to clear the very bar the member had just failed — so a
/// group could never be empty and a cold probe always paid for its content.
/// Live, that put ~4,000 tokens of repo_map (a folder listing, its tool
/// response, and its summary) in front of a bare "Hi, how are you?".
#[test]
pub(super) fn a_group_whose_turns_all_miss_the_threshold_emits_nothing() {
    let b = Builder::from_yaml(TURN_GROUP_TOPK_YAML).unwrap();
    let ground = b.id_for_layer("ground").unwrap();
    let structure = b.id_for_group("structure").unwrap();

    // Three turns, all scoring 0.1 — below the group's 0.5 threshold, so the
    // top_k rule selects nothing.
    let mut resolver = MockResolver::new().with_default_score(0.1);
    resolver.append(structure);
    resolver.append(structure);
    resolver.append(structure);

    let proj = b.project(
        ProjectionTarget {
            layer: ground,
            group: structure,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    assert_eq!(
        group_turn_count(proj.sealed_turns(), structure),
        0,
        "an all-below-threshold group must emit nothing, not a floor member",
    );
    // And the layer goes with it: Step 7 drops the empty group, then the layer
    // that has lost every group. Nothing of `ground` survives.
    assert_eq!(
        proj.sealed_turns().count(),
        0,
        "the layer must drop out once its only group is empty",
    );
}

#[test]
pub(super) fn turns_clearing_the_threshold_are_the_only_ones_selected() {
    let b = Builder::from_yaml(TURN_GROUP_TOPK_YAML).unwrap();
    let ground = b.id_for_layer("ground").unwrap();
    let structure = b.id_for_group("structure").unwrap();

    // Turns 1 and 2 clear the 0.5 threshold and fill the two top_k slots; turn 0
    // is below it and stays out.
    let mut resolver = MockResolver::new().with_default_score(0.1);
    let _t0 = resolver.append(structure);
    let t1 = resolver.append(structure);
    let t2 = resolver.append(structure);
    resolver = resolver
        .with_score(structure, t1, 0.9)
        .with_score(structure, t2, 0.8);

    let proj = b.project(
        ProjectionTarget {
            layer: ground,
            group: structure,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let mut idxs: Vec<TurnIndex> = proj
        .sealed_turns()
        .filter(|t| t.group() == structure)
        .map(|t| t.index())
        .collect();
    idxs.sort();
    assert_eq!(
        idxs,
        vec![t1, t2],
        "only the turns clearing the threshold are selected",
    );
}

/// Collection analogue: a `top_k` collection whose members all score below the
/// threshold selects nothing, and the `default` (a section named by tag) is the
/// floor so the collection still contributes one section.
pub(super) const COLLECTION_DEFAULT_YAML: &str = r#"
system_prompt:
  items:
    - kind: collection
      name: tools
      selection: { kind: top_k, k: 2 }
      score_threshold: 0.5
      default: { tag: "tool_c" }
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
      sections:
        - id: tool_a
          content: "A"
        - id: tool_b
          content: "B"
        - id: tool_c
          content: "C"
        - id: tool_d
          content: "D"
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
pub(super) fn collection_default_parses() {
    let b = Builder::from_yaml(COLLECTION_DEFAULT_YAML).unwrap();
    let coll = b
        .schema()
        .system_prompt
        .items
        .iter()
        .find_map(|item| match item {
            super::super::schema::SystemPromptItem::Collection(c) if c.name == "tools" => Some(c),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        coll.default.as_ref().map(|d| d.tag.as_str()),
        Some("tool_c")
    );
}

#[test]
pub(super) fn collection_default_injected_when_selection_empty() {
    let b = Builder::from_yaml(COLLECTION_DEFAULT_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let tool_c = b.id_for_system_section("tool_c").unwrap();

    // Every section scores 0.0 (MockResolver default) — below the collection's
    // 0.5 min_score — so the belief pass selects no tool. The default injects
    // tool_c so the collection still contributes exactly one member.
    let resolver = MockResolver::new();

    let proj = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let tool_names: HashSet<&str> = ["tool_a", "tool_b", "tool_c", "tool_d"]
        .into_iter()
        .collect();
    let tool_ids: HashSet<SectionId> = tool_names
        .iter()
        .filter_map(|n| b.id_for_system_section(n))
        .collect();
    let emitted_tools: Vec<SectionId> = proj
        .sealed_sections()
        .map(|s| s.id)
        .filter(|id| tool_ids.contains(id))
        .collect();
    assert_eq!(
        emitted_tools,
        vec![tool_c],
        "only the default tool survives"
    );
}

// ── Belief-driven turn selection ─────────────────────────────────────────────
//
// A non-`Sequence` turn group is belief-driven: its turns are selected by
// RelLeak over their fresh wide-Q scores (budget + gates from the selection
// rule), the turn-axis analogue of the tool catalog. These tests drive the
// belief path via the MockResolver's per-turn scores.

#[test]
pub(super) fn belief_config_takes_budget_and_gates_from_the_rule() {
    // structure = top_k(2), score_threshold 0.5.
    let b = Builder::from_yaml(TURN_GROUP_TOPK_YAML).unwrap();
    let structure = b.id_for_group("structure").unwrap();
    let group = b.group(structure).unwrap();
    assert!(group.is_belief_driven());
    let cfg = group.belief_config(10);
    assert_eq!(cfg.budget_max, 2, "budget cap comes from top_k k");
    assert_eq!(cfg.budget_min, 0);
    assert_eq!(
        cfg.min_score, 0.5,
        "min/evict gate comes from score_threshold"
    );
    assert_eq!(cfg.evict_score, 0.5);
}

#[test]
pub(super) fn belief_driven_group_selects_top_k_by_score() {
    let b = Builder::from_yaml(TURN_GROUP_TOPK_YAML).unwrap();
    let ground = b.id_for_layer("ground").unwrap();
    let structure = b.id_for_group("structure").unwrap();

    // Four candidate turns; the two above the 0.5 threshold with the highest
    // scores must win the two slots (top_k k=2).
    let mut resolver = MockResolver::new().with_default_score(0.0);
    for _ in 0..4 {
        resolver.append(structure);
    }
    resolver = resolver
        .with_score(structure, TurnIndex(0), 0.9)
        .with_score(structure, TurnIndex(1), 0.6)
        .with_score(structure, TurnIndex(2), 0.95)
        .with_score(structure, TurnIndex(3), 0.55);

    let proj = b.project(
        ProjectionTarget {
            layer: ground,
            group: structure,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let mut idxs: Vec<u32> = proj
        .sealed_turns()
        .filter(|t| t.group() == structure)
        .map(|t| t.index().0)
        .collect();
    idxs.sort();
    // Highest two above threshold: turn 2 (0.95) and turn 0 (0.9).
    assert_eq!(
        idxs,
        vec![0, 2],
        "belief selects the two highest-scored turns"
    );
}
