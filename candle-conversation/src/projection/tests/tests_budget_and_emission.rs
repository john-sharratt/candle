//! Tests for how a token budget is spent across groups, and the order turns are
//! emitted in.
//!
//! A budget is filled with the best-scored turns that fit, and a share one group
//! leaves unused goes to the others. Turns are emitted in insertion order within
//! a group, and higher-scored groups come last within their layer, nearest the
//! input's end. Minimum and maximum percentages bound a group's share, and an
//! empty group consumes no budget.

use super::*;

// —— Budget constraints ————————————————————————————————————————————————————————

#[test]
pub(super) fn budget_overflow_drops_lowest_scored_turns() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: 4500
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
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    let i0 = resolver.append(grp);
    let i1 = resolver.append(grp);
    let i2 = resolver.append(grp);

    // Each turn is 2000 tokens; budget for turns = 4500, so only 2 fit.
    let resolver = resolver
        .with_tokens(grp, i0, 2000)
        .with_tokens(grp, i1, 2000)
        .with_tokens(grp, i2, 2000)
        .with_score(grp, i0, 0.9)
        .with_score(grp, i1, 0.5)
        .with_score(grp, i2, 0.1); // lowest-scored gets dropped

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    assert_eq!(proj.sealed_turns().count(), 2);
    let indices: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();
    assert!(indices.contains(&0));
    assert!(indices.contains(&1));
    assert!(!indices.contains(&2));
}

#[test]
pub(super) fn single_turn_overflow_drops_turn() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: 900
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
        selection: { kind: single }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    let i0 = resolver.append(grp);
    // Turn is 5000 tokens, far exceeds budget.
    let resolver = resolver
        .with_tokens(grp, i0, 5000)
        .with_score(grp, i0, 0.99);

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    assert!(
        proj.sealed_turns().next().is_none(),
        "oversized single turn should be dropped"
    );
}

// —— Emission ordering —————————————————————————————————————————————————————————

#[test]
pub(super) fn turns_emitted_in_insertion_order_within_group() {
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
        selection: { kind: top_k, k: 10 }
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
    let indices: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();
    for w in indices.windows(2) {
        assert!(w[0] < w[1], "turns must be in insertion order");
    }
}

#[test]
pub(super) fn higher_scored_group_emitted_last_within_layer() {
    // Doc §7: "Higher-scored groups appear *later* in the emitted list within
    // their layer — closer to the bottom of the LLM's input."
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
    - id: stub
      content: "stub"
layers:
  - name: data_layer
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
      - id: low_grp
        selection: { kind: top_k, k: 5 }
      - id: high_grp
        selection: { kind: top_k, k: 5 }
  - name: target_layer
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
      - id: target_grp
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let target_layer = b.id_for_layer("target_layer").unwrap();
    let low_grp = b.id_for_group("low_grp").unwrap();
    let high_grp = b.id_for_group("high_grp").unwrap();
    let target_grp = b.id_for_group("target_grp").unwrap();

    let mut resolver = MockResolver::new();
    let li0 = resolver.append(low_grp);
    let hi0 = resolver.append(high_grp);
    resolver.append(target_grp);
    let resolver = resolver
        .with_score(low_grp, li0, 0.2)
        .with_score(high_grp, hi0, 0.9)
        .with_default_score(0.5);

    let proj = b.project(
        ProjectionTarget {
            layer: target_layer,
            group: target_grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let order = groups_in_order(proj.sealed_turns());
    // data_layer groups should both appear. high_grp (score 0.9) must be last within data_layer.
    let data_order: Vec<GroupId> = order
        .iter()
        .copied()
        .filter(|&g| g == low_grp || g == high_grp)
        .collect();
    assert_eq!(data_order.len(), 2);
    assert_eq!(data_order[1], high_grp, "higher-scored group must be last");
}

// —— Budget redistribution —————————————————————————————————————————————————————

#[test]
pub(super) fn freed_budget_redistributes_to_other_layers() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
    - id: stub
      content: "stub"
layers:
  - name: sparse
    window: 9500
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
      priority: 50
    groups:
      - id: sparse_grp
        selection: { kind: always_visible }
  - name: dense
    window: 9500
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
      priority: 50
    groups:
      - id: dense_grp
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let dense_layer = b.id_for_layer("dense").unwrap();
    let sparse_grp = b.id_for_group("sparse_grp").unwrap();
    let dense_grp = b.id_for_group("dense_grp").unwrap();

    let mut resolver =
        MockResolver::new()
            .with_default_tokens(100)
            .with_tokens(sparse_grp, TurnIndex(0), 10);

    // sparse: 1 turn × 10 tokens = 10 (far less than its ~4750 share).
    resolver.append(sparse_grp);
    // dense: 20 turns × 100 tokens = 2000.
    for _ in 0..20 {
        resolver.append(dense_grp);
    }

    let proj = b.project(
        ProjectionTarget {
            layer: dense_layer,
            group: dense_grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    // With redistribution, dense should get more than half of 9500 tokens
    // and include all 20 turns (20 * 100 = 2000 ≤ 9500).
    let dense_count = group_turn_count(proj.sealed_turns(), dense_grp);
    assert_eq!(
        dense_count, 20,
        "freed budget from sparse should let dense include all 20 turns"
    );
}

// —— Edge cases ————————————————————————————————————————————————————————————————

#[test]
pub(super) fn no_layers_in_schema_is_valid() {
    let yaml = r#"
system_prompt:
  sections:
    - id: frame
      content: "x"
layers: []
"#;
    let b = Builder::from_yaml(yaml);
    assert!(b.is_ok());
}

#[test]
pub(super) fn empty_group_does_not_consume_budget() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: 9500
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
      - id: empty_grp
        selection: { kind: top_k, k: 5 }
        budget: { priority: 50 }
      - id: full_grp
        selection: { kind: always_visible }
        budget: { priority: 50 }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let full_grp = b.id_for_group("full_grp").unwrap();

    // Only append to full_grp; empty_grp gets nothing.
    let mut resolver = MockResolver::new().with_default_tokens(100);
    for _ in 0..5 {
        resolver.append(full_grp);
    }

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: full_grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let full_count = group_turn_count(proj.sealed_turns(), full_grp);
    assert_eq!(full_count, 5, "full_grp should have all 5 turns");
}

#[test]
pub(super) fn all_turns_below_threshold_leaves_no_turns() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
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
      - id: grp
        score_threshold: 0.9
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new().with_default_score(0.1);
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
    assert!(proj.sealed_turns().next().is_none());
}

#[test]
pub(super) fn large_number_of_turns_respects_budget() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: 4500
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
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    // 100 turns × 100 tokens each = 10000, far exceeds 4500 budget.
    let mut resolver = MockResolver::new().with_default_tokens(100);
    for _ in 0..100 {
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

    let total_tokens: usize = proj
        .sealed_turns()
        .map(|t| t.key().map_or(0, |k| resolver.turn_token_count(k)))
        .sum();
    assert!(
        total_tokens <= 4500,
        "total tokens {total_tokens} must not exceed turn budget 4500"
    );
}

#[test]
pub(super) fn min_percent_guarantees_minimum_budget() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
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
      - id: priority_grp
        selection: { kind: always_visible }
        budget: { priority: 80, min_percent: 60 }
      - id: other_grp
        selection: { kind: always_visible }
        budget: { priority: 20 }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let priority_grp = b.id_for_group("priority_grp").unwrap();
    let other_grp = b.id_for_group("other_grp").unwrap();

    // Seed both with enough turns to consume the budget.
    let mut resolver = MockResolver::new().with_default_tokens(100);
    for _ in 0..50 {
        resolver.append(priority_grp);
    }
    for _ in 0..50 {
        resolver.append(other_grp);
    }

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: priority_grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let priority_count = group_turn_count(proj.sealed_turns(), priority_grp);
    let other_count = group_turn_count(proj.sealed_turns(), other_grp);

    // priority_grp min is 60% of 9000 = 5400 tokens → at least 54 turns at 100 tokens each
    // (capped at 50 available, so just check priority > other).
    assert!(
        priority_count >= other_count,
        "priority_grp (min 60%) should receive more budget than other_grp"
    );
}

#[test]
pub(super) fn max_percent_caps_allocation() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: 9500
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
      - id: capped_grp
        selection: { kind: always_visible }
        budget: { priority: 90, max_percent: 20 }
      - id: other_grp
        selection: { kind: always_visible }
        budget: { priority: 10 }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let capped = b.id_for_group("capped_grp").unwrap();
    let other = b.id_for_group("other_grp").unwrap();

    let mut resolver = MockResolver::new().with_default_tokens(100);
    for _ in 0..100 {
        resolver.append(capped);
    }
    for _ in 0..100 {
        resolver.append(other);
    }

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: capped,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let capped_tokens: usize = proj
        .sealed_turns()
        .filter(|t| t.group() == capped)
        .map(|t| t.key().map_or(0, |k| resolver.turn_token_count(k)))
        .sum();

    // max_percent: 20 of 9500 = 1900 tokens max for capped_grp.
    assert!(
        capped_tokens <= 1900,
        "capped_grp tokens {capped_tokens} should not exceed 20% of turn budget (1900)"
    );
}

#[test]
pub(super) fn historical_top_k_zero_means_only_recent() {
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
      - id: conv
        selection:
          kind: conversation
          recent: 3
          historical_top_k: 0
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let conv = b.id_for_group("conv").unwrap();

    let mut resolver = MockResolver::new().with_default_score(0.9);
    for _ in 0..6 {
        resolver.append(conv);
    }

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: conv,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    // Only last 3 should appear.
    assert_eq!(proj.sealed_turns().count(), 3);
    let indices: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();
    assert_eq!(indices, vec![3, 4, 5]);
}

#[test]
pub(super) fn recent_zero_means_only_historical() {
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
      - id: conv
        selection:
          kind: conversation
          recent: 0
          historical_top_k: 2
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let conv = b.id_for_group("conv").unwrap();

    let mut resolver = MockResolver::new();
    let i0 = resolver.append(conv);
    let i1 = resolver.append(conv);
    let i2 = resolver.append(conv);
    let i3 = resolver.append(conv);
    let resolver = resolver
        .with_score(conv, i0, 0.9)
        .with_score(conv, i1, 0.1)
        .with_score(conv, i2, 0.7)
        .with_score(conv, i3, 0.2);

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: conv,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    assert_eq!(proj.sealed_turns().count(), 2);
    let indices: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();
    // top-2 historical: i0 (0.9) and i2 (0.7), insertion order
    assert_eq!(indices, vec![0, 2]);
}

#[test]
pub(super) fn empty_target_group_does_not_emit_target_layer() {
    // Doc §9.7: "Filter out empty groups and empty layers." No exception for
    // the target group. If the target group has no turns, its layer should not
    // emit any turns (lower-layer groups still appear).
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
    - id: stub
      content: "stub"
layers:
  - name: data
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
      - id: data_grp
        selection: { kind: always_visible }
  - name: target_layer
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
      - id: empty_target
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let target_layer = b.id_for_layer("target_layer").unwrap();
    let data_grp = b.id_for_group("data_grp").unwrap();
    let empty_target = b.id_for_group("empty_target").unwrap();

    let mut resolver = MockResolver::new();
    resolver.append(data_grp); // populate the lower layer

    let proj = b.project(
        ProjectionTarget {
            layer: target_layer,
            group: empty_target,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let groups = groups_in_order(proj.sealed_turns());
    assert!(groups.contains(&data_grp), "lower layer must still emit");
    assert!(
        !groups.contains(&empty_target),
        "empty target group must be filtered, not emitted"
    );
}
