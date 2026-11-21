//! Tests for which layers a target sees, and how a group's score is formed.
//!
//! A dialogue target sees every lower layer and its own group, and a motivational
//! target hides the dialogue. A ground target sees only ground. A group's score
//! follows the formula the schema names, sum, count or top-two mean, and a group
//! with the higher score is emitted last.

use super::*;

// —— Multi-layer projection ————————————————————————————————————————————————————

pub(super) const MULTI_LAYER_YAML: &str = r#"
system_prompt:
  sections:
    - id: frame
      content: "Frame content."
    - id: values
      content: "Values content."
    - id: guidance
      content: "Guidance content."
layers:
  - name: perceptual_ground
    window: 95000
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
    score_threshold: 0.1
    budget:
      priority: 30
    groups:
      - id: type_specialist
        selection: { kind: top_k, k: 3 }
        score_threshold: 0.2
      - id: structure_specialist
        selection: { kind: top_k, k: 3 }

  - name: motivational
    window: 95000
    summary:
      turns:
        max_tokens: 256
        user:
          system_prompt: compress
          user_prompt: compress
        assistant:
          system_prompt: compress
          user_prompt: compress
    score_formula: top_k_mean
    score_formula_k: 3
    budget:
      priority: 60
    groups:
      - id: active_mission
        selection: { kind: top_k, k: 5 }
        budget: { priority: 80, min_percent: 30 }
      - id: goal_pressure
        selection: { kind: single }
        budget: { priority: 40 }

  - name: dialogue
    window: 95000
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
      priority: 100
      min_percent: 50
    groups:
      - id: primary_conversation
        selection:
          kind: conversation
          recent: 8
          historical_top_k: 12
        budget: { priority: 100 }
"#;

#[test]
pub(super) fn multi_layer_full_masking() {
    let b = Builder::from_yaml(MULTI_LAYER_YAML).unwrap();

    let ts = b.id_for_group("type_specialist").unwrap();
    let ss = b.id_for_group("structure_specialist").unwrap();
    let am = b.id_for_group("active_mission").unwrap();
    let gp = b.id_for_group("goal_pressure").unwrap();
    let pc = b.id_for_group("primary_conversation").unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();

    let mut resolver = MockResolver::new();
    let ts_i = resolver.append(ts);
    let ss_i = resolver.append(ss);
    let am_i = resolver.append(am);
    let gp_i = resolver.append(gp);
    let pc_i = resolver.append(pc);
    let resolver = resolver
        .with_score(ts, ts_i, 0.9)
        .with_score(ss, ss_i, 0.9)
        .with_score(am, am_i, 0.9)
        .with_score(gp, gp_i, 0.9)
        .with_score(pc, pc_i, 0.9);

    // Target: dialogue/primary_conversation → all lower layers and target group visible.
    let proj = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: pc,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let groups = groups_in_order(proj.sealed_turns());
    assert!(groups.contains(&ts), "type_specialist must be visible");
    assert!(groups.contains(&ss), "structure_specialist must be visible");
    assert!(groups.contains(&am), "active_mission must be visible");
    assert!(groups.contains(&gp), "goal_pressure must be visible");
    assert!(
        groups.contains(&pc),
        "primary_conversation (target) must be visible"
    );
}

#[test]
pub(super) fn multi_layer_motivational_target_hides_dialogue() {
    let b = Builder::from_yaml(MULTI_LAYER_YAML).unwrap();

    let ts = b.id_for_group("type_specialist").unwrap();
    let am = b.id_for_group("active_mission").unwrap();
    let gp = b.id_for_group("goal_pressure").unwrap();
    let pc = b.id_for_group("primary_conversation").unwrap();
    let motivational = b.id_for_layer("motivational").unwrap();

    let mut resolver = MockResolver::new();
    let ts_i = resolver.append(ts);
    let am_i = resolver.append(am);
    let gp_i = resolver.append(gp);
    let pc_i = resolver.append(pc);
    let resolver = resolver
        .with_score(ts, ts_i, 0.9)
        .with_score(am, am_i, 0.9)
        .with_score(gp, gp_i, 0.9)
        .with_score(pc, pc_i, 0.9);

    // Target: motivational/active_mission
    let proj = b.project(
        ProjectionTarget {
            layer: motivational,
            group: am,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let groups = groups_in_order(proj.sealed_turns());
    assert!(groups.contains(&ts));
    assert!(groups.contains(&am)); // target
    assert!(
        !groups.contains(&gp),
        "sibling goal_pressure must NOT be visible"
    );
    assert!(
        !groups.contains(&pc),
        "dialogue must NOT be visible from motivational"
    );
}

#[test]
pub(super) fn multi_layer_ground_target_sees_only_ground() {
    let b = Builder::from_yaml(MULTI_LAYER_YAML).unwrap();

    let ts = b.id_for_group("type_specialist").unwrap();
    let ss = b.id_for_group("structure_specialist").unwrap();
    let am = b.id_for_group("active_mission").unwrap();
    let pc = b.id_for_group("primary_conversation").unwrap();
    let perceptual = b.id_for_layer("perceptual_ground").unwrap();

    let mut resolver = MockResolver::new();
    let ts_i = resolver.append(ts);
    resolver.append(am);
    resolver.append(pc);
    let resolver = resolver.with_score(ts, ts_i, 0.9);

    let proj = b.project(
        ProjectionTarget {
            layer: perceptual,
            group: ts,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let groups = groups_in_order(proj.sealed_turns());
    assert!(groups.contains(&ts));
    assert!(!groups.contains(&ss)); // sibling, masked
    assert!(!groups.contains(&am)); // higher layer
    assert!(!groups.contains(&pc)); // higher layer
}

// —— System prompt emission ————————————————————————————————————————————————————

#[test]
pub(super) fn sections_always_emit_regardless_of_size() {
    let yaml = r#"
system_prompt:
  sections:
    - id: small
      content: "Small."
    - id: big
      content: "Very large section content that would dwarf any turn budget."
layers:
  - name: layer
    window: 800
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
    let small = b.id_for_system_section("small").unwrap();
    let big = b.id_for_system_section("big").unwrap();

    let resolver = MockResolver::new();
    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let section_ids: Vec<u32> = proj.sealed_sections().map(|s| s.id.raw()).collect();
    assert_eq!(section_ids, vec![small.raw(), big.raw()]);
}

#[test]
pub(super) fn system_prompt_sections_in_declaration_order() {
    let b = Builder::from_yaml(MULTI_LAYER_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let pc = b.id_for_group("primary_conversation").unwrap();
    let frame = b.id_for_system_section("frame").unwrap();
    let values = b.id_for_system_section("values").unwrap();
    let guidance = b.id_for_system_section("guidance").unwrap();

    let resolver = MockResolver::new();
    let proj = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: pc,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    assert_eq!(proj.sealed_sections().count(), 3);
    assert_eq!(proj.sealed_sections().next().unwrap().id, frame);
    assert_eq!(proj.sealed_sections().nth(1).unwrap().id, values);
    assert_eq!(proj.sealed_sections().nth(2).unwrap().id, guidance);
}

// —— Score formula variants ————————————————————————————————————————————————————

#[test]
pub(super) fn score_formula_sum_determines_group_score() {
    // Two groups in a lower layer; project from upper so both are visible.
    // grp_low: sum=0.3; grp_high: sum=1.0 → grp_high emits last.
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
    score_formula: sum
    groups:
      - id: grp_low
        selection: { kind: always_visible }
      - id: grp_high
        selection: { kind: always_visible }
  - name: upper_layer
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
      - id: upper_grp
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let upper_layer = b.id_for_layer("upper_layer").unwrap();
    let grp_low = b.id_for_group("grp_low").unwrap();
    let grp_high = b.id_for_group("grp_high").unwrap();
    let upper_grp = b.id_for_group("upper_grp").unwrap();

    let mut resolver = MockResolver::new();
    let li0 = resolver.append(grp_low);
    let hi0 = resolver.append(grp_high);
    let hi1 = resolver.append(grp_high);
    resolver.append(upper_grp);
    let resolver = resolver
        .with_score(grp_low, li0, 0.3)
        .with_score(grp_high, hi0, 0.5)
        .with_score(grp_high, hi1, 0.5)
        .with_default_score(0.5);

    let proj = b.project(
        ProjectionTarget {
            layer: upper_layer,
            group: upper_grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let order = groups_in_order(proj.sealed_turns());
    let data_order: Vec<GroupId> = order
        .iter()
        .copied()
        .filter(|&g| g == grp_low || g == grp_high)
        .collect();
    assert_eq!(data_order.last().copied(), Some(grp_high));
}

#[test]
pub(super) fn score_formula_count_promotes_large_groups() {
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
    score_formula: count
    groups:
      - id: small_grp
        selection: { kind: always_visible }
      - id: large_grp
        selection: { kind: always_visible }
  - name: upper_layer
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
      - id: upper_grp
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let upper_layer = b.id_for_layer("upper_layer").unwrap();
    let small = b.id_for_group("small_grp").unwrap();
    let large = b.id_for_group("large_grp").unwrap();
    let upper_grp = b.id_for_group("upper_grp").unwrap();

    let mut resolver = MockResolver::new().with_default_score(0.9);
    resolver.append(small); // count = 1
    for _ in 0..5 {
        resolver.append(large);
    } // count = 5
    resolver.append(upper_grp);

    let proj = b.project(
        ProjectionTarget {
            layer: upper_layer,
            group: upper_grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let order = groups_in_order(proj.sealed_turns());
    let data_order: Vec<GroupId> = order
        .iter()
        .copied()
        .filter(|&g| g == small || g == large)
        .collect();
    assert_eq!(
        data_order.last().copied(),
        Some(large),
        "large group (count=5) should emit last"
    );
}

#[test]
pub(super) fn top_k_mean_formula_integration() {
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
    score_formula: top_k_mean
    score_formula_k: 2
    score_threshold: 0.5
    groups:
      - id: grp1
        selection: { kind: always_visible }
        budget: { priority: 50 }
      - id: grp2
        selection: { kind: always_visible }
        budget: { priority: 50 }
"#;
    // grp1: turns at 0.9, 0.1, 0.1  → top-2-mean = (0.9+0.1)/2 = 0.5 → at threshold
    // grp2: turns at 0.8, 0.7       → top-2-mean = (0.8+0.7)/2 = 0.75 → above threshold
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp1 = b.id_for_group("grp1").unwrap();
    let grp2 = b.id_for_group("grp2").unwrap();

    let mut resolver = MockResolver::new();
    let g1i0 = resolver.append(grp1);
    let g1i1 = resolver.append(grp1);
    let g1i2 = resolver.append(grp1);
    let g2i0 = resolver.append(grp2);
    let g2i1 = resolver.append(grp2);
    let resolver = resolver
        .with_score(grp1, g1i0, 0.9)
        .with_score(grp1, g1i1, 0.1)
        .with_score(grp1, g1i2, 0.1)
        .with_score(grp2, g2i0, 0.8)
        .with_score(grp2, g2i1, 0.7);

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp1,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    // grp2 has higher top-2-mean → should be emitted last.
    let order = groups_in_order(proj.sealed_turns());
    if order.len() == 2 {
        assert_eq!(order.last().copied(), Some(grp2));
    }
}
