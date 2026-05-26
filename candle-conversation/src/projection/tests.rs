//! Extensive integration tests for the projection engine.
//!
//! All tests use YAML-declared schemas and a `MockResolver` that returns
//! fully-controlled token counts and scores. Turn state is owned by the
//! resolver (not the builder), so tests use `resolver.append(group)` rather
//! than `builder.append(group)`.

use std::collections::HashMap;

use super::builder::Builder;
use super::ids::{GroupId, SectionId, TimelineId, TurnIndex};
use super::project::ProjectionTarget;
use crate::substrate::ContentResolver;

// ── Mock resolver ─────────────────────────────────────────────────────────────

/// Simple mock: explicit per-turn scores/tokens plus uniform defaults.
/// Owns turn counts so the projection engine can enumerate turns.
#[derive(Default)]
struct MockResolver {
    /// (group_raw, turn_raw) → score
    scores: HashMap<(u32, u32), f32>,
    /// (group_raw, turn_raw) → tokens
    tokens: HashMap<(u32, u32), usize>,
    /// turn_count per group
    turn_counts: HashMap<u32, u32>,
    /// section_id → score
    section_scores: HashMap<u32, f32>,
    default_score: f32,
    default_tokens: usize,
}

impl MockResolver {
    fn new() -> Self {
        Self {
            default_score: 0.5,
            default_tokens: 10,
            ..Default::default()
        }
    }

    /// Register a new turn in `group`; returns its TurnIndex.
    fn append(&mut self, group: GroupId) -> TurnIndex {
        let count = self.turn_counts.entry(group.raw()).or_insert(0);
        let idx = TurnIndex(*count);
        *count += 1;
        idx
    }

    fn with_score(mut self, group: GroupId, idx: TurnIndex, score: f32) -> Self {
        self.scores.insert((group.raw(), idx.0), score);
        self
    }

    fn with_tokens(mut self, group: GroupId, idx: TurnIndex, tokens: usize) -> Self {
        self.tokens.insert((group.raw(), idx.0), tokens);
        self
    }

    fn with_default_tokens(mut self, tokens: usize) -> Self {
        self.default_tokens = tokens;
        self
    }

    fn with_default_score(mut self, score: f32) -> Self {
        self.default_score = score;
        self
    }

    fn with_section_score(mut self, section: SectionId, score: f32) -> Self {
        self.section_scores.insert(section.raw(), score);
        self
    }
}

impl ContentResolver for MockResolver {
    fn turn_count(&self, group: GroupId) -> u32 {
        self.turn_counts.get(&group.raw()).copied().unwrap_or(0)
    }

    fn turn_token_count(&self, group: GroupId, index: TurnIndex) -> usize {
        *self
            .tokens
            .get(&(group.raw(), index.0))
            .unwrap_or(&self.default_tokens)
    }

    fn turn_score(
        &self,
        group: GroupId,
        index: TurnIndex,
        _formula: super::schema::ScoreFormula,
        _weights: &super::schema::DepthWeights,
    ) -> f32 {
        // Mock returns a single explicit score regardless of formula/weights.
        // The projection engine still drives the depth-aware path; tests just
        // bypass the multi-stat machinery.
        *self
            .scores
            .get(&(group.raw(), index.0))
            .unwrap_or(&self.default_score)
    }

    fn section_score(
        &self,
        section: SectionId,
        _formula: super::schema::ScoreFormula,
        _weights: &super::schema::DepthWeights,
    ) -> f32 {
        // Default 0.0 for any section the test didn't assign a score to.
        // Sections without explicit scores will lose to those with scores.
        *self.section_scores.get(&section.raw()).unwrap_or(&0.0)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn group_turn_count(groups: &[super::project::ResolvedTurn], gid: GroupId) -> usize {
    groups.iter().filter(|t| t.group() == gid).count()
}

fn groups_in_order(turns: &[super::project::ResolvedTurn]) -> Vec<GroupId> {
    let mut seen = Vec::new();
    for t in turns {
        if !seen.contains(&t.group()) {
            seen.push(t.group());
        }
    }
    seen
}

// ── YAML round-trip and id assignment ─────────────────────────────────────────

const SIMPLE_YAML: &str = r#"
layers:
  - name: ground
    system_prompt:
      sections:
        - id: stub
          content: "stub"
    window: 8000
    score_formula: max
    budget:
      priority: 40
    groups:
      - id: facts
        selection:
          kind: top_k
          k: 3

  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
      min_percent: 50
    system_prompt:
      sections:
        - id: frame
          content: "You are a helpful assistant."
        - id: values
          content: "Be honest."
    groups:
      - id: conversation
        selection:
          kind: conversation
          recent: 4
          historical_top_k: 8
"#;

#[test]
fn yaml_parses_and_assigns_ids() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();

    // Layers assigned in declaration order.
    let ground = b.id_for_layer("ground").unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    assert!(ground.raw() < dialogue.raw());

    // Groups globally unique.
    let facts = b.id_for_group("facts").unwrap();
    let conv = b.id_for_group("conversation").unwrap();
    assert_ne!(facts, conv);
    assert!(facts.raw() < conv.raw());

    // Sections (per-layer scoped — both live under `dialogue` here).
    let frame = b.id_for_section_in(dialogue, "frame").unwrap();
    let values = b.id_for_section_in(dialogue, "values").unwrap();
    assert!(frame.raw() < values.raw());
}

#[test]
fn unknown_layer_name_returns_none() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    assert!(b.id_for_layer("nonexistent").is_none());
}

#[test]
fn schema_accessors_work() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let ground = b.id_for_layer("ground").unwrap();
    let layer = b.layer(ground).unwrap();
    assert_eq!(layer.name, "ground");

    let facts = b.id_for_group("facts").unwrap();
    let group = b.group(facts).unwrap();
    assert_eq!(group.name, "facts");

    let dialogue = b.id_for_layer("dialogue").unwrap();
    let frame = b.id_for_section_in(dialogue, "frame").unwrap();
    let section = b.section(frame).unwrap();
    assert_eq!(section.name, "frame");
}

// ── Append + turn_count (now on resolver) ─────────────────────────────────────

#[test]
fn append_increments_correctly() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let conv = b.id_for_group("conversation").unwrap();

    let mut resolver = MockResolver::new();
    let i0 = resolver.append(conv);
    let i1 = resolver.append(conv);
    let i2 = resolver.append(conv);

    assert_eq!(i0.0, 0);
    assert_eq!(i1.0, 1);
    assert_eq!(i2.0, 2);
    assert_eq!(resolver.turn_count(conv), 3);
}

#[test]
fn turn_count_zero_for_empty_group() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let facts = b.id_for_group("facts").unwrap();
    let resolver = MockResolver::new();
    assert_eq!(resolver.turn_count(facts), 0);
}

// ── Projection: basic visibility ──────────────────────────────────────────────

#[test]
fn empty_builder_projection_has_no_turns() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let conv = b.id_for_group("conversation").unwrap();
    let resolver = MockResolver::new();

    let proj = b.project(ProjectionTarget { layer: dialogue, group: conv, timeline: TimelineId::for_test(1) }, &resolver);
    assert!(proj.turns.is_empty());
}

#[test]
fn system_prompt_sections_always_emitted() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let conv = b.id_for_group("conversation").unwrap();
    let resolver = MockResolver::new();

    let proj = b.project(ProjectionTarget { layer: dialogue, group: conv, timeline: TimelineId::for_test(1) }, &resolver);
    assert_eq!(proj.system_prompt.len(), 2);
}

#[test]
fn turns_appear_after_append() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let conv = b.id_for_group("conversation").unwrap();

    let mut resolver = MockResolver::new();
    resolver.append(conv);
    resolver.append(conv);
    resolver.append(conv);

    let proj = b.project(ProjectionTarget { layer: dialogue, group: conv, timeline: TimelineId::for_test(1) }, &resolver);
    assert_eq!(proj.turns.len(), 3);
}

// ── Masking ───────────────────────────────────────────────────────────────────

#[test]
fn lower_layers_visible_for_dialogue_target() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let ground = b.id_for_layer("ground").unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let facts = b.id_for_group("facts").unwrap();
    let conv = b.id_for_group("conversation").unwrap();

    let mut resolver = MockResolver::new();
    resolver.append(facts);
    resolver.append(facts);
    resolver.append(conv);

    let proj = b.project(ProjectionTarget { layer: dialogue, group: conv, timeline: TimelineId::for_test(1) }, &resolver);

    let groups: Vec<GroupId> = groups_in_order(&proj.turns);
    assert!(groups.contains(&facts), "ground/facts should be visible");
    assert!(groups.contains(&conv), "target conv should be visible");
    let _ = ground;
}

#[test]
fn higher_layer_not_visible_for_lower_target() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let ground = b.id_for_layer("ground").unwrap();
    let facts = b.id_for_group("facts").unwrap();
    let conv = b.id_for_group("conversation").unwrap();

    let mut resolver = MockResolver::new();
    resolver.append(facts);
    resolver.append(conv); // conv is in dialogue (higher layer)

    let proj = b.project(ProjectionTarget { layer: ground, group: facts, timeline: TimelineId::for_test(1) }, &resolver);

    let groups: Vec<GroupId> = groups_in_order(&proj.turns);
    assert!(!groups.contains(&conv), "dialogue group must NOT be visible from ground target");
}

#[test]
fn sibling_group_in_target_layer_not_visible() {
    let yaml = r#"
layers:
  - name: layer_a
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
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

    let proj = b.project(ProjectionTarget { layer: layer_a, group: g1, timeline: TimelineId::for_test(1) }, &resolver);

    let groups = groups_in_order(&proj.turns);
    assert!(groups.contains(&g1));
    assert!(!groups.contains(&g2), "sibling group_a2 must not be visible");
}

// ── Score threshold ───────────────────────────────────────────────────────────

#[test]
fn turns_below_score_threshold_filtered() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
    score_formula: max
    groups:
      - id: grp
        score_threshold: 0.6
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    let i0 = resolver.append(grp);
    let i1 = resolver.append(grp);
    let i2 = resolver.append(grp);
    let resolver = resolver
        .with_score(grp, i0, 0.9)
        .with_score(grp, i1, 0.3) // below threshold
        .with_score(grp, i2, 0.7);

    let proj = b.project(ProjectionTarget { layer, group: grp, timeline: TimelineId::for_test(1) }, &resolver);
    let indices: Vec<u32> = proj.turns.iter().map(|t| t.index().0).collect();
    assert!(indices.contains(&0));
    assert!(!indices.contains(&1), "turn with score 0.3 should be filtered");
    assert!(indices.contains(&2));
}

#[test]
fn layer_threshold_filters_whole_group() {
    let yaml = r#"
layers:
  - name: low
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
    score_formula: max
    score_threshold: 0.8
    budget:
      priority: 50
    groups:
      - id: low_grp
        selection: { kind: top_k, k: 5 }
  - name: high
    system_prompt:
      sections:
        - id: stub
          content: "stub"
    window: 9000
    score_formula: max
    budget:
      priority: 50
    groups:
      - id: high_grp
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let low_layer = b.id_for_layer("low").unwrap();
    let high_layer = b.id_for_layer("high").unwrap();
    let low_grp = b.id_for_group("low_grp").unwrap();
    let high_grp = b.id_for_group("high_grp").unwrap();

    let mut resolver = MockResolver::new();
    let i0 = resolver.append(low_grp); // low score → group score < 0.8
    resolver.append(high_grp);
    let resolver = resolver
        .with_score(low_grp, i0, 0.3); // group score = 0.3 < 0.8 threshold → entire group dropped

    let proj = b.project(ProjectionTarget { layer: high_layer, group: high_grp, timeline: TimelineId::for_test(1) }, &resolver);
    let groups = groups_in_order(&proj.turns);
    assert!(!groups.contains(&low_grp), "low_grp should be dropped by layer threshold");
    assert!(groups.contains(&high_grp));
    let _ = low_layer;
}

// ── Selection rules ───────────────────────────────────────────────────────────

#[test]
fn top_k_limits_turns() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
    score_formula: max
    groups:
      - id: grp
        selection: { kind: top_k, k: 2 }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    let i0 = resolver.append(grp);
    let i1 = resolver.append(grp);
    let i2 = resolver.append(grp);
    let i3 = resolver.append(grp);
    let resolver = resolver
        .with_score(grp, i0, 0.9)
        .with_score(grp, i1, 0.2)
        .with_score(grp, i2, 0.7)
        .with_score(grp, i3, 0.1);

    let proj = b.project(ProjectionTarget { layer, group: grp, timeline: TimelineId::for_test(1) }, &resolver);
    // top-2: i0 (0.9) and i2 (0.7), emitted in insertion order
    assert_eq!(proj.turns.len(), 2);
    assert_eq!(proj.turns[0].index().0, 0);
    assert_eq!(proj.turns[1].index().0, 2);
}

#[test]
fn single_selection_one_turn() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
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
    let i1 = resolver.append(grp);
    let i2 = resolver.append(grp);
    let resolver = resolver
        .with_score(grp, i0, 0.4)
        .with_score(grp, i1, 0.9) // winner
        .with_score(grp, i2, 0.6);

    let proj = b.project(ProjectionTarget { layer, group: grp, timeline: TimelineId::for_test(1) }, &resolver);
    assert_eq!(proj.turns.len(), 1);
    assert_eq!(proj.turns[0].index().0, 1);
}

#[test]
fn always_visible_selects_all_above_threshold() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 49000
    score_formula: max
    groups:
      - id: grp
        score_threshold: 0.4
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    let i0 = resolver.append(grp);
    let i1 = resolver.append(grp);
    let i2 = resolver.append(grp);
    let resolver = resolver
        .with_score(grp, i0, 0.9)
        .with_score(grp, i1, 0.1) // below threshold
        .with_score(grp, i2, 0.5);

    let proj = b.project(ProjectionTarget { layer, group: grp, timeline: TimelineId::for_test(1) }, &resolver);
    assert_eq!(proj.turns.len(), 2);
    let indices: Vec<u32> = proj.turns.iter().map(|t| t.index().0).collect();
    assert_eq!(indices, vec![0, 2]);
}

#[test]
fn conversation_recent_turns_always_included() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 49000
    score_formula: max
    groups:
      - id: conv
        selection:
          kind: conversation
          recent: 3
          historical_top_k: 2
        score_threshold: 0.5
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let conv = b.id_for_group("conv").unwrap();

    // Append 6 turns: first 3 old, last 3 recent.
    let mut resolver = MockResolver::new();
    let idxs: Vec<TurnIndex> = (0..6).map(|_| resolver.append(conv)).collect();

    // Old turns: one above threshold, one below, one below.
    // Recent turns: all below threshold but inviolate.
    let resolver = resolver
        .with_score(conv, idxs[0], 0.8) // old, high → historical
        .with_score(conv, idxs[1], 0.1) // old, below threshold
        .with_score(conv, idxs[2], 0.1) // old, below threshold
        .with_score(conv, idxs[3], 0.0) // recent, inviolate
        .with_score(conv, idxs[4], 0.0) // recent, inviolate
        .with_score(conv, idxs[5], 0.0); // recent, inviolate

    let proj = b.project(ProjectionTarget { layer, group: conv, timeline: TimelineId::for_test(1) }, &resolver);
    let indices: Vec<u32> = proj.turns.iter().map(|t| t.index().0).collect();

    // Expected: i0 (historical top-1), i3, i4, i5 (inviolate), in insertion order.
    assert!(indices.contains(&0), "high-score old turn expected");
    assert!(!indices.contains(&1));
    assert!(!indices.contains(&2));
    assert!(indices.contains(&3));
    assert!(indices.contains(&4));
    assert!(indices.contains(&5));
    // Emission insertion order.
    assert!(indices[0] < indices[1]);
}

// ── Budget constraints ────────────────────────────────────────────────────────

#[test]
fn budget_overflow_drops_lowest_scored_turns() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 4500
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

    let proj = b.project(ProjectionTarget { layer, group: grp, timeline: TimelineId::for_test(1) }, &resolver);
    assert_eq!(proj.turns.len(), 2);
    let indices: Vec<u32> = proj.turns.iter().map(|t| t.index().0).collect();
    assert!(indices.contains(&0));
    assert!(indices.contains(&1));
    assert!(!indices.contains(&2));
}

#[test]
fn single_turn_overflow_drops_turn() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 900
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

    let proj = b.project(ProjectionTarget { layer, group: grp, timeline: TimelineId::for_test(1) }, &resolver);
    assert!(
        proj.turns.is_empty(),
        "oversized single turn should be dropped"
    );
}

// ── Emission ordering ─────────────────────────────────────────────────────────

#[test]
fn turns_emitted_in_insertion_order_within_group() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 49000
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

    let proj = b.project(ProjectionTarget { layer, group: grp, timeline: TimelineId::for_test(1) }, &resolver);
    let indices: Vec<u32> = proj.turns.iter().map(|t| t.index().0).collect();
    for w in indices.windows(2) {
        assert!(w[0] < w[1], "turns must be in insertion order");
    }
}

#[test]
fn higher_scored_group_emitted_last_within_layer() {
    // Doc §7: "Higher-scored groups appear *later* in the emitted list within
    // their layer — closer to the bottom of the LLM's input."
    let yaml = r#"
layers:
  - name: data_layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 49000
    score_formula: max
    groups:
      - id: low_grp
        selection: { kind: top_k, k: 5 }
      - id: high_grp
        selection: { kind: top_k, k: 5 }
  - name: target_layer
    system_prompt:
      sections:
        - id: stub
          content: "stub"
    window: 49000
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

    let proj = b.project(ProjectionTarget { layer: target_layer, group: target_grp, timeline: TimelineId::for_test(1) }, &resolver);
    let order = groups_in_order(&proj.turns);
    // data_layer groups should both appear. high_grp (score 0.9) must be last within data_layer.
    let data_order: Vec<GroupId> = order.iter().copied().filter(|&g| g == low_grp || g == high_grp).collect();
    assert_eq!(data_order.len(), 2);
    assert_eq!(data_order[1], high_grp, "higher-scored group must be last");
}

// ── Multi-layer projection ────────────────────────────────────────────────────

const MULTI_LAYER_YAML: &str = r#"
layers:
  - name: perceptual_ground
    system_prompt:
      sections:
        - id: stub
          content: "stub"
    window: 95000
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
    system_prompt:
      sections:
        - id: stub
          content: "stub"
    window: 95000
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
    score_formula: max
    budget:
      priority: 100
      min_percent: 50
    system_prompt:
      sections:
        - id: frame
          content: "Frame content."
        - id: values
          content: "Values content."
        - id: guidance
          content: "Guidance content."
    groups:
      - id: primary_conversation
        selection:
          kind: conversation
          recent: 8
          historical_top_k: 12
        budget: { priority: 100 }
"#;

#[test]
fn multi_layer_full_masking() {
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
        ProjectionTarget { layer: dialogue, group: pc, timeline: TimelineId::for_test(1) },
        &resolver,
    );

    let groups = groups_in_order(&proj.turns);
    assert!(groups.contains(&ts), "type_specialist must be visible");
    assert!(groups.contains(&ss), "structure_specialist must be visible");
    assert!(groups.contains(&am), "active_mission must be visible");
    assert!(groups.contains(&gp), "goal_pressure must be visible");
    assert!(groups.contains(&pc), "primary_conversation (target) must be visible");
}

#[test]
fn multi_layer_motivational_target_hides_dialogue() {
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
        ProjectionTarget { layer: motivational, group: am, timeline: TimelineId::for_test(1) },
        &resolver,
    );

    let groups = groups_in_order(&proj.turns);
    assert!(groups.contains(&ts));
    assert!(groups.contains(&am)); // target
    assert!(!groups.contains(&gp), "sibling goal_pressure must NOT be visible");
    assert!(!groups.contains(&pc), "dialogue must NOT be visible from motivational");
}

#[test]
fn multi_layer_ground_target_sees_only_ground() {
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
        ProjectionTarget { layer: perceptual, group: ts, timeline: TimelineId::for_test(1) },
        &resolver,
    );

    let groups = groups_in_order(&proj.turns);
    assert!(groups.contains(&ts));
    assert!(!groups.contains(&ss));  // sibling, masked
    assert!(!groups.contains(&am));  // higher layer
    assert!(!groups.contains(&pc));  // higher layer
}

// ── System prompt emission ────────────────────────────────────────────────────

#[test]
fn sections_always_emit_regardless_of_size() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: small
          content: "Small."
        - id: big
          content: "Very large section content that would dwarf any turn budget."
    window: 800
    score_formula: max
    groups:
      - id: grp
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();
    let small = b.id_for_section_in(layer, "small").unwrap();
    let big = b.id_for_section_in(layer, "big").unwrap();

    let resolver = MockResolver::new();
    let proj = b.project(ProjectionTarget { layer, group: grp, timeline: TimelineId::for_test(1) }, &resolver);
    let section_ids: Vec<u32> = proj.system_prompt.iter().map(|s| s.id.raw()).collect();
    assert_eq!(section_ids, vec![small.raw(), big.raw()]);
}

#[test]
fn system_prompt_sections_in_declaration_order() {
    let b = Builder::from_yaml(MULTI_LAYER_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let pc = b.id_for_group("primary_conversation").unwrap();
    let frame = b.id_for_section_in(dialogue, "frame").unwrap();
    let values = b.id_for_section_in(dialogue, "values").unwrap();
    let guidance = b.id_for_section_in(dialogue, "guidance").unwrap();

    let resolver = MockResolver::new();
    let proj = b.project(ProjectionTarget { layer: dialogue, group: pc, timeline: TimelineId::for_test(1) }, &resolver);
    assert_eq!(proj.system_prompt.len(), 3);
    assert_eq!(proj.system_prompt[0].id, frame);
    assert_eq!(proj.system_prompt[1].id, values);
    assert_eq!(proj.system_prompt[2].id, guidance);
}

// ── YAML validation errors ────────────────────────────────────────────────────

#[test]
fn yaml_duplicate_group_name_is_error() {
    let yaml = r#"
layers:
  - name: layer1
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
    score_formula: max
    groups:
      - id: dup
        selection: { kind: always_visible }
  - name: layer2
    system_prompt:
      sections:
        - id: stub
          content: "stub"
    window: 9000
    score_formula: max
    groups:
      - id: dup
        selection: { kind: always_visible }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(
        matches!(err, super::error::ConstructionError::DuplicateGroupName(_)),
        "expected DuplicateGroupName, got {:?}",
        err
    );
}

#[test]
fn yaml_duplicate_section_name_is_error() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "A"
        - id: s1
          content: "B"
    window: 9000
    score_formula: max
    groups:
      - id: grp
        selection: { kind: always_visible }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::DuplicateSectionName(_)
    ));
}

#[test]
fn yaml_unknown_selection_kind_is_error() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
    score_formula: max
    groups:
      - id: grp
        selection: { kind: fuzzy_pick }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::UnknownSelectionKind(_)
    ));
}

#[test]
fn yaml_invalid_priority_zero_is_error() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
    score_formula: max
    groups:
      - id: grp
        selection: { kind: always_visible }
        budget: { priority: 0 }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::InvalidPriority { .. }
    ));
}

#[test]
fn yaml_invalid_percentage_over_100_is_error() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
    score_formula: max
    groups:
      - id: grp
        selection: { kind: always_visible }
        budget: { priority: 50, min_percent: 150 }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::InvalidPercentage { .. }
    ));
}

// ── Construction validation errors ───────────────────────────────────────────

#[test]
fn construction_sibling_min_percent_exceeds_100_is_error() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
    score_formula: max
    groups:
      - id: grp1
        selection: { kind: always_visible }
        budget: { priority: 50, min_percent: 60 }
      - id: grp2
        selection: { kind: always_visible }
        budget: { priority: 50, min_percent: 60 }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::MinPercentExceedsTotal { .. }
    ));
}

#[test]
fn construction_max_less_than_min_is_error() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
    score_formula: max
    groups:
      - id: grp
        selection: { kind: always_visible }
        budget: { priority: 50, min_percent: 60, max_percent: 40 }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::MaxLessThanMin { .. }
    ));
}

#[test]
fn construction_top_k_zero_is_error() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
    score_formula: max
    groups:
      - id: grp
        selection: { kind: top_k, k: 0 }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::InvalidTopK { .. }
    ));
}

#[test]
fn construction_conversation_both_zero_is_error() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
    score_formula: max
    groups:
      - id: grp
        selection:
          kind: conversation
          recent: 0
          historical_top_k: 0
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::InvalidConversationK { .. }
    ));
}

#[test]
fn construction_negative_score_threshold_is_error() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
    score_formula: max
    groups:
      - id: grp
        selection: { kind: always_visible }
        score_threshold: -0.5
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::NegativeScoreThreshold { .. }
    ));
}

// ── Score formula variants ────────────────────────────────────────────────────

#[test]
fn score_formula_sum_determines_group_score() {
    // Two groups in a lower layer; project from upper so both are visible.
    // grp_low: sum=0.3; grp_high: sum=1.0 → grp_high emits last.
    let yaml = r#"
layers:
  - name: data_layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 49000
    score_formula: sum
    groups:
      - id: grp_low
        selection: { kind: always_visible }
      - id: grp_high
        selection: { kind: always_visible }
  - name: upper_layer
    system_prompt:
      sections:
        - id: stub
          content: "stub"
    window: 49000
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

    let proj = b.project(ProjectionTarget { layer: upper_layer, group: upper_grp, timeline: TimelineId::for_test(1) }, &resolver);
    let order = groups_in_order(&proj.turns);
    let data_order: Vec<GroupId> = order.iter().copied().filter(|&g| g == grp_low || g == grp_high).collect();
    assert_eq!(data_order.last().copied(), Some(grp_high));
}

#[test]
fn score_formula_count_promotes_large_groups() {
    let yaml = r#"
layers:
  - name: data_layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 49000
    score_formula: count
    groups:
      - id: small_grp
        selection: { kind: always_visible }
      - id: large_grp
        selection: { kind: always_visible }
  - name: upper_layer
    system_prompt:
      sections:
        - id: stub
          content: "stub"
    window: 49000
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
    for _ in 0..5 { resolver.append(large); } // count = 5
    resolver.append(upper_grp);

    let proj = b.project(ProjectionTarget { layer: upper_layer, group: upper_grp, timeline: TimelineId::for_test(1) }, &resolver);
    let order = groups_in_order(&proj.turns);
    let data_order: Vec<GroupId> = order.iter().copied().filter(|&g| g == small || g == large).collect();
    assert_eq!(data_order.last().copied(), Some(large), "large group (count=5) should emit last");
}

// ── Budget redistribution ─────────────────────────────────────────────────────

#[test]
fn freed_budget_redistributes_to_other_layers() {
    let yaml = r#"
layers:
  - name: sparse
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9500
    score_formula: max
    budget:
      priority: 50
    groups:
      - id: sparse_grp
        selection: { kind: always_visible }
  - name: dense
    system_prompt:
      sections:
        - id: stub
          content: "stub"
    window: 9500
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

    let mut resolver = MockResolver::new()
        .with_default_tokens(100)
        .with_tokens(sparse_grp, TurnIndex(0), 10);

    // sparse: 1 turn × 10 tokens = 10 (far less than its ~4750 share).
    resolver.append(sparse_grp);
    // dense: 20 turns × 100 tokens = 2000.
    for _ in 0..20 { resolver.append(dense_grp); }

    let proj = b.project(
        ProjectionTarget { layer: dense_layer, group: dense_grp, timeline: TimelineId::for_test(1) },
        &resolver,
    );

    // With redistribution, dense should get more than half of 9500 tokens
    // and include all 20 turns (20 * 100 = 2000 ≤ 9500).
    let dense_count = group_turn_count(&proj.turns, dense_grp);
    assert_eq!(dense_count, 20, "freed budget from sparse should let dense include all 20 turns");
}

// ── Edge cases ────────────────────────────────────────────────────────────────

#[test]
fn no_layers_in_schema_is_valid() {
    let yaml = r#"
layers: []
"#;
    let b = Builder::from_yaml(yaml);
    assert!(b.is_ok());
}

#[test]
fn empty_group_does_not_consume_budget() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9500
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
    for _ in 0..5 { resolver.append(full_grp); }

    let proj = b.project(ProjectionTarget { layer, group: full_grp, timeline: TimelineId::for_test(1) }, &resolver);

    let full_count = group_turn_count(&proj.turns, full_grp);
    assert_eq!(full_count, 5, "full_grp should have all 5 turns");
}

#[test]
fn all_turns_below_threshold_leaves_no_turns() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
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
    for _ in 0..5 { resolver.append(grp); }

    let proj = b.project(ProjectionTarget { layer, group: grp, timeline: TimelineId::for_test(1) }, &resolver);
    assert!(proj.turns.is_empty());
}

#[test]
fn large_number_of_turns_respects_budget() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 4500
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
    for _ in 0..100 { resolver.append(grp); }

    let proj = b.project(ProjectionTarget { layer, group: grp, timeline: TimelineId::for_test(1) }, &resolver);

    let total_tokens: usize = proj
        .turns
        .iter()
        .map(|t| resolver.turn_token_count(t.group(), t.index()))
        .sum();
    assert!(
        total_tokens <= 4500,
        "total tokens {total_tokens} must not exceed turn budget 4500"
    );
}

#[test]
fn min_percent_guarantees_minimum_budget() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
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
    for _ in 0..50 { resolver.append(priority_grp); }
    for _ in 0..50 { resolver.append(other_grp); }

    let proj = b.project(ProjectionTarget { layer, group: priority_grp, timeline: TimelineId::for_test(1) }, &resolver);

    let priority_count = group_turn_count(&proj.turns, priority_grp);
    let other_count = group_turn_count(&proj.turns, other_grp);

    // priority_grp min is 60% of 9000 = 5400 tokens → at least 54 turns at 100 tokens each
    // (capped at 50 available, so just check priority > other).
    assert!(
        priority_count >= other_count,
        "priority_grp (min 60%) should receive more budget than other_grp"
    );
}

#[test]
fn top_k_mean_formula_integration() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 49000
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

    let proj = b.project(ProjectionTarget { layer, group: grp1, timeline: TimelineId::for_test(1) }, &resolver);
    // grp2 has higher top-2-mean → should be emitted last.
    let order = groups_in_order(&proj.turns);
    if order.len() == 2 {
        assert_eq!(order.last().copied(), Some(grp2));
    }
}

#[test]
fn max_percent_caps_allocation() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9500
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
    for _ in 0..100 { resolver.append(capped); }
    for _ in 0..100 { resolver.append(other); }

    let proj = b.project(ProjectionTarget { layer, group: capped, timeline: TimelineId::for_test(1) }, &resolver);

    let capped_tokens: usize = proj
        .turns
        .iter()
        .filter(|t| t.group() == capped)
        .map(|t| resolver.turn_token_count(t.group(), t.index()))
        .sum();

    // max_percent: 20 of 9500 = 1900 tokens max for capped_grp.
    assert!(
        capped_tokens <= 1900,
        "capped_grp tokens {capped_tokens} should not exceed 20% of turn budget (1900)"
    );
}

#[test]
fn historical_top_k_zero_means_only_recent() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 49000
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
    for _ in 0..6 { resolver.append(conv); }

    let proj = b.project(ProjectionTarget { layer, group: conv, timeline: TimelineId::for_test(1) }, &resolver);

    // Only last 3 should appear.
    assert_eq!(proj.turns.len(), 3);
    let indices: Vec<u32> = proj.turns.iter().map(|t| t.index().0).collect();
    assert_eq!(indices, vec![3, 4, 5]);
}

#[test]
fn recent_zero_means_only_historical() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 49000
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

    let proj = b.project(ProjectionTarget { layer, group: conv, timeline: TimelineId::for_test(1) }, &resolver);
    assert_eq!(proj.turns.len(), 2);
    let indices: Vec<u32> = proj.turns.iter().map(|t| t.index().0).collect();
    // top-2 historical: i0 (0.9) and i2 (0.7), insertion order
    assert_eq!(indices, vec![0, 2]);
}

#[test]
fn yaml_top_k_missing_k_is_error() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 9000
    score_formula: max
    groups:
      - id: grp
        selection: { kind: top_k }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::InvalidTopK { .. }
    ));
}

#[test]
fn missing_layer_system_prompt_is_parse_error() {
    let yaml = r#"
layers:
  - name: layer
    window: 9000
    score_formula: max
    groups:
      - id: grp
        selection: { kind: always_visible }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::YamlParse(_)
    ));
}

#[test]
fn empty_layer_system_prompt_is_construction_error() {
    let yaml = r#"
layers:
  - name: layer
    window: 9000
    score_formula: max
    system_prompt:
      sections: []
    groups:
      - id: grp
        selection: { kind: always_visible }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::EmptyLayerSystemPrompt { .. }
    ));
}

#[test]
fn empty_target_group_does_not_emit_target_layer() {
    // Doc §9.7: "Filter out empty groups and empty layers." No exception for
    // the target group. If the target group has no turns, its layer should not
    // emit any turns (lower-layer groups still appear).
    let yaml = r#"
layers:
  - name: data
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 49000
    score_formula: max
    groups:
      - id: data_grp
        selection: { kind: always_visible }
  - name: target_layer
    system_prompt:
      sections:
        - id: stub
          content: "stub"
    window: 49000
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

    let groups = groups_in_order(&proj.turns);
    assert!(groups.contains(&data_grp), "lower layer must still emit");
    assert!(
        !groups.contains(&empty_target),
        "empty target group must be filtered, not emitted"
    );
}

// ── Variable substitution ────────────────────────────────────────────────────

const SUBST_YAML: &str = r#"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget: { priority: 100 }
    system_prompt:
      sections:
        - id: frame
          content: "Hello {name}, welcome to {project}."
    groups:
      - id: chat
        selection: { kind: always_visible }
"#;

#[test]
fn substitution_replaces_placeholders() {
    let b = Builder::from_yaml_with_vars(
        SUBST_YAML,
        &[("name", "Alice"), ("project", "Candle")],
    )
    .unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let frame = b.id_for_section_in(dialogue, "frame").unwrap();
    let section = b.section(frame).unwrap();
    assert_eq!(section.content, "Hello Alice, welcome to Candle.");
}

#[test]
fn substitution_missing_var_is_error() {
    let err = Builder::from_yaml_with_vars(SUBST_YAML, &[("name", "Alice")]).unwrap_err();
    match err {
        super::error::ConstructionError::UnresolvedVariable { name } => {
            assert_eq!(name, "project");
        }
        other => panic!("expected UnresolvedVariable, got {other:?}"),
    }
}

#[test]
fn substitution_catches_template_typo() {
    let yaml = r#"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget: { priority: 100 }
    system_prompt:
      sections:
        - id: frame
          content: "Welcome to {wokrspace}."
    groups:
      - id: chat
        selection: { kind: always_visible }
"#;
    let err = Builder::from_yaml_with_vars(yaml, &[("workspace", "candle")]).unwrap_err();
    match err {
        super::error::ConstructionError::UnresolvedVariable { name } => {
            assert_eq!(name, "wokrspace");
        }
        other => panic!("expected UnresolvedVariable, got {other:?}"),
    }
}

#[test]
fn substitution_leaves_non_identifier_braces_alone() {
    let yaml = r#"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget: { priority: 100 }
    system_prompt:
      sections:
        - id: frame
          content: 'Example JSON: {"tool": "search", "args": {}}'
    groups:
      - id: chat
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml_with_vars(yaml, &[]).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let frame = b.id_for_section_in(dialogue, "frame").unwrap();
    let section = b.section(frame).unwrap();
    assert_eq!(section.content, r#"Example JSON: {"tool": "search", "args": {}}"#);
}

#[test]
fn from_yaml_rejects_any_placeholder() {
    let yaml = r#"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget: { priority: 100 }
    system_prompt:
      sections:
        - id: frame
          content: "Welcome to {workspace}."
    groups:
      - id: chat
        selection: { kind: always_visible }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::UnresolvedVariable { .. }
    ));
}

#[test]
fn substitution_baked_into_immutable_schema() {
    let b = Builder::from_yaml_with_vars(SUBST_YAML, &[("name", "A"), ("project", "B")]).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let frame = b.id_for_section_in(dialogue, "frame").unwrap();
    let content_first = b.section(frame).unwrap().content.clone();
    let content_second = b.section(frame).unwrap().content.clone();
    assert_eq!(content_first, content_second);
    assert_eq!(content_first, "Hello A, welcome to B.");
}

/// Sanity check: the live zend projection schema parses + validates.
#[test]
fn zend_projection_yaml_parses() {
    let yaml = include_str!("../../../zend/src/prompts/projection.yaml");
    let b = Builder::from_yaml_with_vars(yaml, &[("workspace", "candle")])
        .expect("zend projection.yaml must parse");

    let expected_layers = [
        "repo_map",
        "code_reading",
        "static_analysis",
        "dependency_analysis",
        "architectural_analysis",
        "critical_analysis",
        "bug_analysis",
        "daily_history",
        "dream_log",
        "dialogue",
    ];
    for name in &expected_layers {
        assert!(
            b.id_for_layer(name).is_some(),
            "layer {name:?} missing from zend schema"
        );
    }

    let last = b.schema().layers.last().expect("at least one layer");
    assert_eq!(last.name, "dialogue", "dialogue must be the topmost layer");

    assert!(b.id_for_group("primary_conversation").is_some());
}

#[test]
fn default_selection_is_always_visible() {
    let yaml = r#"
layers:
  - name: layer
    system_prompt:
      sections:
        - id: s1
          content: "X"
    window: 49000
    score_formula: max
    groups:
      - id: grp
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    for _ in 0..5 { resolver.append(grp); }

    let proj = b.project(ProjectionTarget { layer, group: grp, timeline: TimelineId::for_test(1) }, &resolver);
    assert_eq!(proj.turns.len(), 5, "default selection includes all turns");
}

// ── DepthWeights ──────────────────────────────────────────────────────────────

#[test]
fn depth_weights_default_is_universal_optimum() {
    use super::schema::DepthWeights;
    let w = DepthWeights::default();
    // Universal calibration optimum: syn:1 / sem:1 / prag:4
    // (1*3 + 1*6 + 4*9) / 6 = 45/6 = 7.5
    assert!((w.combine(3.0, 6.0, 9.0) - 7.5).abs() < 1e-6);
}

#[test]
fn depth_weights_unequal_weights_normalise_correctly() {
    use super::schema::DepthWeights;
    let w = DepthWeights {
        syntactic: 1.0,
        semantic: 2.0,
        pragmatic: 1.0,
    };
    // (1*4 + 2*8 + 1*12) / 4 = 32/4 = 8.0
    assert_eq!(w.combine(4.0, 8.0, 12.0), 8.0);
}

#[test]
fn depth_weights_all_zero_returns_zero() {
    use super::schema::DepthWeights;
    let w = DepthWeights {
        syntactic: 0.0,
        semantic: 0.0,
        pragmatic: 0.0,
    };
    // Defensive: division by zero would NaN; we return 0.0 instead.
    assert_eq!(w.combine(10.0, 20.0, 30.0), 0.0);
}

#[test]
fn depth_weights_single_depth_dominant() {
    use super::schema::DepthWeights;
    let w = DepthWeights {
        syntactic: 0.0,
        semantic: 1.0,
        pragmatic: 0.0,
    };
    // Only semantic contributes.
    assert_eq!(w.combine(99.0, 5.0, 99.0), 5.0);
}

#[test]
fn yaml_depth_weights_default_when_omitted() {
    use super::schema::DepthWeights;
    let yaml = r#"
layers:
  - name: layer
    window: 1000
    system_prompt:
      sections:
        - id: s
          content: "x"
    groups:
      - id: g
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer_id = b.id_for_layer("layer").unwrap();
    let layer = b.layer(layer_id).unwrap();
    assert_eq!(layer.depth_weights, DepthWeights::default());
}

#[test]
fn yaml_depth_weights_override() {
    let yaml = r#"
layers:
  - name: layer
    window: 1000
    depth_weights:
      syntactic: 0.2
      semantic: 0.6
      pragmatic: 0.2
    system_prompt:
      sections:
        - id: s
          content: "x"
    groups:
      - id: g
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer_id = b.id_for_layer("layer").unwrap();
    let layer = b.layer(layer_id).unwrap();
    assert!((layer.depth_weights.syntactic - 0.2).abs() < 1e-6);
    assert!((layer.depth_weights.semantic - 0.6).abs() < 1e-6);
    assert!((layer.depth_weights.pragmatic - 0.2).abs() < 1e-6);
}

#[test]
fn yaml_depth_weights_partial_override_uses_defaults() {
    use super::schema::DepthWeights;
    let yaml = r#"
layers:
  - name: layer
    window: 1000
    depth_weights:
      semantic: 5.0
    system_prompt:
      sections:
        - id: s
          content: "x"
    groups:
      - id: g
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer_id = b.id_for_layer("layer").unwrap();
    let layer = b.layer(layer_id).unwrap();
    let default = DepthWeights::default();
    assert_eq!(layer.depth_weights.syntactic, default.syntactic);
    assert_eq!(layer.depth_weights.semantic, 5.0);
    assert_eq!(layer.depth_weights.pragmatic, default.pragmatic);
}

#[test]
fn yaml_negative_depth_weight_is_error() {
    use super::error::ConstructionError;
    let yaml = r#"
layers:
  - name: layer
    window: 1000
    depth_weights:
      semantic: -0.5
    system_prompt:
      sections:
        - id: s
          content: "x"
    groups:
      - id: g
"#;
    match Builder::from_yaml(yaml) {
        Err(ConstructionError::NegativeDepthWeight { layer, depth, .. }) => {
            assert_eq!(layer, "layer");
            assert_eq!(depth, "semantic");
        }
        other => panic!("expected NegativeDepthWeight, got {other:?}"),
    }
}

// ── End-to-end: Substrate + BDP scores + projection ─────────────────────

#[test]
fn session_resolver_picks_correct_metric_per_score_formula() {
    use crate::substrate::{PerDepthScores, ProjectionScores, ScoredSubstrate, Substrate, TurnScores};
    use super::schema::{DepthWeights, ScoreFormula};
    use super::ContentResolver;

    let mut r = Substrate::new();
    let yaml = r#"
layers:
  - name: l
    window: 1000
    system_prompt:
      sections:
        - id: s
          content: "x"
    groups:
      - id: g
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let l = b.id_for_layer("l").unwrap();
    let g = b.id_for_group("g").unwrap();

    let idx = r.append_with_blocks_for_test(l, g, 10, 0, 1);
    let mut scores = ProjectionScores::new();
    scores.set_for_group_test(
        &r,
        g,
        idx,
        PerDepthScores {
            syn: TurnScores { max: 1.0, sum: 2.0, mean: 3.0, top_k_mean: 4.0, count: 5.0, span: 0.0, pertok_excess: 0.0 },
            sem: TurnScores { max: 1.0, sum: 2.0, mean: 3.0, top_k_mean: 4.0, count: 5.0, span: 0.0, pertok_excess: 0.0 },
            prag: TurnScores { max: 1.0, sum: 2.0, mean: 3.0, top_k_mean: 4.0, count: 5.0, span: 0.0, pertok_excess: 0.0 },
        },
    );
    let r = ScoredSubstrate::new(&r, &scores);

    let w = DepthWeights::default();
    assert_eq!(r.turn_score(g, idx, ScoreFormula::Max, &w), 1.0);
    assert_eq!(r.turn_score(g, idx, ScoreFormula::Sum, &w), 2.0);
    assert_eq!(r.turn_score(g, idx, ScoreFormula::Mean, &w), 3.0);
    assert_eq!(r.turn_score(g, idx, ScoreFormula::TopKMean { k: 8 }, &w), 4.0);
    assert_eq!(r.turn_score(g, idx, ScoreFormula::Count, &w), 5.0);
}

#[test]
fn session_resolver_combines_depths_with_weights() {
    use crate::substrate::{PerDepthScores, ProjectionScores, ScoredSubstrate, Substrate, TurnScores};
    use super::schema::{DepthWeights, ScoreFormula};
    use super::ContentResolver;

    let yaml = r#"
layers:
  - name: l
    window: 1000
    system_prompt:
      sections:
        - id: s
          content: "x"
    groups:
      - id: g
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let l = b.id_for_layer("l").unwrap();
    let g = b.id_for_group("g").unwrap();

    let mut r = Substrate::new();
    let idx = r.append_with_blocks_for_test(l, g, 10, 0, 1);

    // Distinct max values per depth so the combine weighting is observable.
    let mut scores = ProjectionScores::new();
    scores.set_for_group_test(
        &r,
        g,
        idx,
        PerDepthScores {
            syn: TurnScores { max: 10.0, ..Default::default() },
            sem: TurnScores { max: 50.0, ..Default::default() },
            prag: TurnScores { max: 100.0, ..Default::default() },
        },
    );
    let r = ScoredSubstrate::new(&r, &scores);

    // Equal weights: (10 + 50 + 100) / 3 = 53.333...
    let equal = DepthWeights { syntactic: 1.0, semantic: 1.0, pragmatic: 1.0 };
    let s = r.turn_score(g, idx, ScoreFormula::Max, &equal);
    assert!((s - 53.333_33).abs() < 1e-3);

    // Pragmatic-only: 100
    let prag_only = DepthWeights {
        syntactic: 0.0,
        semantic: 0.0,
        pragmatic: 1.0,
    };
    assert_eq!(r.turn_score(g, idx, ScoreFormula::Max, &prag_only), 100.0);

    // Weighted: w=(1, 2, 1) → (1*10 + 2*50 + 1*100) / 4 = 210 / 4 = 52.5
    let weighted = DepthWeights {
        syntactic: 1.0,
        semantic: 2.0,
        pragmatic: 1.0,
    };
    assert_eq!(r.turn_score(g, idx, ScoreFormula::Max, &weighted), 52.5);
}

#[test]
fn projection_uses_bdp_scores_to_pick_top_k() {
    use crate::substrate::{PerDepthScores, ProjectionScores, ScoredSubstrate, Substrate, TurnScores};

    // Five turns, top_k=2 by score.  Without BDP scores all are tied; we
    // set distinct max values to force a stable ordering.
    let yaml = r#"
layers:
  - name: l
    window: 10000
    score_formula: max
    system_prompt:
      sections:
        - id: s
          content: "x"
    groups:
      - id: g
        selection: { kind: top_k, k: 2 }
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("l").unwrap();
    let g = b.id_for_group("g").unwrap();

    let mut r = Substrate::new();
    let mut bdp_scores = ProjectionScores::new();
    // Distinct scores; turn 2 (idx=2) and turn 4 (idx=4) should win.
    let scores = [10.0_f32, 20.0, 50.0, 30.0, 100.0];
    for (i, &s) in scores.iter().enumerate() {
        let idx = r.append_with_blocks_for_test(layer, g, 10, i as u64, (i + 1) as u64);
        bdp_scores.set_for_group_test(
            &r,
            g,
            idx,
            PerDepthScores {
                syn: TurnScores { span: s, ..Default::default() },
                sem: TurnScores { span: s, ..Default::default() },
                prag: TurnScores { span: s, ..Default::default() },
            },
        );
    }
    let r = ScoredSubstrate::new(&r, &bdp_scores);

    let proj = b.project(ProjectionTarget { layer, group: g, timeline: TimelineId::for_test(1) }, &r);
    let picked: Vec<u32> = proj.turns.iter().map(|t| t.index().0).collect();
    // The top-2 by max score are idx 4 (100.0) and idx 2 (50.0); they emit
    // in insertion order regardless of selection order.
    assert_eq!(picked, vec![2, 4]);
}

#[test]
fn projection_per_layer_depth_weights_alter_ranking() {
    use crate::substrate::{PerDepthScores, ProjectionScores, ScoredSubstrate, Substrate, TurnScores};

    // Two turns; turn A has high syn but low prag, turn B has low syn but
    // high prag.  By tilting depth_weights toward one or the other, we
    // expect different winners under a top_k=1 rule.
    let yaml_syn_heavy = r#"
layers:
  - name: l
    window: 10000
    score_formula: max
    depth_weights:
      syntactic: 10.0
      semantic: 0.0
      pragmatic: 1.0
    system_prompt:
      sections:
        - id: s
          content: "x"
    groups:
      - id: g
        selection: { kind: top_k, k: 1 }
"#;
    let yaml_prag_heavy = yaml_syn_heavy.replace(
        "syntactic: 10.0\n      semantic: 0.0\n      pragmatic: 1.0",
        "syntactic: 1.0\n      semantic: 0.0\n      pragmatic: 10.0",
    );

    let b_syn = Builder::from_yaml(yaml_syn_heavy).unwrap();
    let g = b_syn.id_for_group("g").unwrap();
    let layer = b_syn.id_for_layer("l").unwrap();

    let mut r = Substrate::new();
    let a = r.append_with_blocks_for_test(layer, g, 10, 0, 1);
    let bturn = r.append_with_blocks_for_test(layer, g, 10, 1, 2);
    let mut bdp = ProjectionScores::new();
    bdp.set_for_group_test(
        &r,
        g,
        a,
        PerDepthScores {
            syn: TurnScores { span: 100.0, ..Default::default() },
            sem: TurnScores::default(),
            prag: TurnScores { span: 1.0, ..Default::default() },
        },
    );
    bdp.set_for_group_test(
        &r,
        g,
        bturn,
        PerDepthScores {
            syn: TurnScores { span: 1.0, ..Default::default() },
            sem: TurnScores::default(),
            prag: TurnScores { span: 100.0, ..Default::default() },
        },
    );
    let r_view = ScoredSubstrate::new(&r, &bdp);

    // Syn-heavy weights: turn A wins.
    let proj = b_syn.project(ProjectionTarget { layer, group: g, timeline: TimelineId::for_test(1) }, &r_view);
    assert_eq!(proj.turns.len(), 1);
    assert_eq!(proj.turns[0].index(), a);

    // Prag-heavy weights: turn B wins.
    let b_prag = Builder::from_yaml(&yaml_prag_heavy).unwrap();
    let g2 = b_prag.id_for_group("g").unwrap();
    let layer2 = b_prag.id_for_layer("l").unwrap();
    // Re-build resolver against the new schema's GroupId (which equals g2).
    // GroupIds are deterministic per yaml shape but for safety we re-append.
    let mut r2 = Substrate::new();
    let a2 = r2.append_with_blocks_for_test(layer2, g2, 10, 0, 1);
    let b2 = r2.append_with_blocks_for_test(layer2, g2, 10, 1, 2);
    let mut bdp2 = ProjectionScores::new();
    bdp2.set_for_group_test(
        &r2,
        g2,
        a2,
        PerDepthScores {
            syn: TurnScores { span: 100.0, ..Default::default() },
            sem: TurnScores::default(),
            prag: TurnScores { span: 1.0, ..Default::default() },
        },
    );
    bdp2.set_for_group_test(
        &r2,
        g2,
        b2,
        PerDepthScores {
            syn: TurnScores { span: 1.0, ..Default::default() },
            sem: TurnScores::default(),
            prag: TurnScores { span: 100.0, ..Default::default() },
        },
    );
    let r2_view = ScoredSubstrate::new(&r2, &bdp2);
    let proj = b_prag.project(ProjectionTarget { layer: layer2, group: g2, timeline: TimelineId::for_test(1) }, &r2_view);
    assert_eq!(proj.turns.len(), 1);
    assert_eq!(proj.turns[0].index(), b2);
}

#[test]
fn yaml_all_zero_depth_weights_is_error() {
    use super::error::ConstructionError;
    let yaml = r#"
layers:
  - name: layer
    window: 1000
    depth_weights:
      syntactic: 0.0
      semantic: 0.0
      pragmatic: 0.0
    system_prompt:
      sections:
        - id: s
          content: "x"
    groups:
      - id: g
"#;
    match Builder::from_yaml(yaml) {
        Err(ConstructionError::AllDepthWeightsZero { layer }) => {
            assert_eq!(layer, "layer");
        }
        other => panic!("expected AllDepthWeightsZero, got {other:?}"),
    }
}

// ── Section + collection emission tests ──────────────────────────────────────
//
// These tests cover the `SystemPromptItem` model: the layer's
// `system_prompt` is an ordered list of items, each either a single
// always-emit section or a `SectionCollection` with its own selection
// rule (TopK / Single / AlwaysVisible).  Emission preserves declaration
// order; selection picks by salience (BDP-derived score).

const SECTIONS_YAML_FLAT: &str = r#"
layers:
  - name: dialogue
    window: 4000
    score_formula: max
    budget:
      priority: 100
    system_prompt:
      sections:
        - id: alpha
          content: "alpha"
        - id: beta
          content: "beta"
        - id: gamma
          content: "gamma"
    groups:
      - id: convo
        selection:
          kind: conversation
          recent: 4
          historical_top_k: 4
"#;

#[test]
fn flat_sections_yaml_emits_all_in_declaration_order() {
    // Legacy `sections:` shortcut — every section is always-emit.
    let b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let alpha = b.id_for_section_in(dialogue, "alpha").unwrap();
    let beta = b.id_for_section_in(dialogue, "beta").unwrap();
    let gamma = b.id_for_section_in(dialogue, "gamma").unwrap();

    let resolver = MockResolver::new();
    let p = b.project(
        ProjectionTarget { layer: dialogue, group: convo, timeline: TimelineId::for_test(1) },
        &resolver,
    );

    assert_eq!(p.system_prompt.len(), 3);
    assert_eq!(p.system_prompt[0].id, alpha);
    assert_eq!(p.system_prompt[1].id, beta);
    assert_eq!(p.system_prompt[2].id, gamma);
}

const COLLECTION_YAML: &str = r#"
layers:
  - name: dialogue
    window: 4000
    score_formula: max
    budget: { priority: 100 }
    system_prompt:
      items:
        - kind: section
          id: framing
          content: "framing"
        - kind: section
          id: tools_intro
          content: "tools_intro"
        - kind: collection
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
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;

#[test]
fn collection_top_k_keeps_highest_scored_in_declaration_order() {
    let b = Builder::from_yaml(COLLECTION_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let framing = b.id_for_section_in(dialogue, "framing").unwrap();
    let tools_intro = b.id_for_section_in(dialogue, "tools_intro").unwrap();
    let tool_a = b.id_for_section_in(dialogue, "tool_a").unwrap();
    let tool_b = b.id_for_section_in(dialogue, "tool_b").unwrap();
    let tool_c = b.id_for_section_in(dialogue, "tool_c").unwrap();
    let tool_d = b.id_for_section_in(dialogue, "tool_d").unwrap();
    let tools_outro = b.id_for_section_in(dialogue, "tools_outro").unwrap();

    // tool_b and tool_d score highest → top-k=2 keeps them.  Emission is
    // in declaration order: framing → tools_intro → (tool_b → tool_d) →
    // tools_outro.  Tools in lower scores (a, c) are filtered.
    let resolver = MockResolver::new()
        .with_section_score(tool_a, 0.1)
        .with_section_score(tool_b, 0.9)
        .with_section_score(tool_c, 0.2)
        .with_section_score(tool_d, 0.8);

    let p = b.project(
        ProjectionTarget { layer: dialogue, group: convo, timeline: TimelineId::for_test(1) },
        &resolver,
    );

    let ids: Vec<SectionId> = p.system_prompt.iter().map(|s| s.id).collect();
    assert_eq!(
        ids,
        vec![framing, tools_intro, tool_b, tool_d, tools_outro],
        "expected static framing intact + top-2 tools in declaration order"
    );
}

#[test]
fn collection_top_k_with_priority_tiebreak() {
    let b = Builder::from_yaml(COLLECTION_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let tool_a = b.id_for_section_in(dialogue, "tool_a").unwrap();
    let tool_b = b.id_for_section_in(dialogue, "tool_b").unwrap();
    let tool_c = b.id_for_section_in(dialogue, "tool_c").unwrap();
    let tool_d = b.id_for_section_in(dialogue, "tool_d").unwrap();

    // All four tools tie on score → fall back to declaration order
    // (a, b, c, d) and pick the first two.
    let resolver = MockResolver::new()
        .with_section_score(tool_a, 0.5)
        .with_section_score(tool_b, 0.5)
        .with_section_score(tool_c, 0.5)
        .with_section_score(tool_d, 0.5);

    let p = b.project(
        ProjectionTarget { layer: dialogue, group: convo, timeline: TimelineId::for_test(1) },
        &resolver,
    );
    let tool_ids: Vec<SectionId> = p
        .system_prompt
        .iter()
        .map(|s| s.id)
        .filter(|&id| id == tool_a || id == tool_b || id == tool_c || id == tool_d)
        .collect();
    assert_eq!(tool_ids, vec![tool_a, tool_b]);
}

const COLLECTION_YAML_THRESHOLD: &str = r#"
layers:
  - name: dialogue
    window: 4000
    score_formula: max
    budget: { priority: 100 }
    system_prompt:
      items:
        - kind: collection
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
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;

#[test]
fn collection_score_threshold_filters_below_floor() {
    let b = Builder::from_yaml(COLLECTION_YAML_THRESHOLD).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let mid = b.id_for_section_in(dialogue, "mid").unwrap();
    let high = b.id_for_section_in(dialogue, "high").unwrap();

    let resolver = MockResolver::new()
        .with_section_score(b.id_for_section_in(dialogue, "low").unwrap(), 0.2)
        .with_section_score(mid, 0.5)
        .with_section_score(high, 0.9);

    let p = b.project(
        ProjectionTarget { layer: dialogue, group: convo, timeline: TimelineId::for_test(1) },
        &resolver,
    );
    let ids: Vec<SectionId> = p.system_prompt.iter().map(|s| s.id).collect();
    assert_eq!(ids, vec![mid, high], "low filtered by 0.4 threshold");
}

#[test]
fn collection_single_picks_max_only() {
    const YAML: &str = r#"
layers:
  - name: dialogue
    window: 4000
    score_formula: max
    budget: { priority: 100 }
    system_prompt:
      items:
        - kind: collection
          name: choices
          selection: { kind: single }
          sections:
            - id: a
              content: "a"
            - id: b
              content: "b"
            - id: c
              content: "c"
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let b_id = b.id_for_section_in(dialogue, "b").unwrap();

    let resolver = MockResolver::new()
        .with_section_score(b.id_for_section_in(dialogue, "a").unwrap(), 0.4)
        .with_section_score(b_id, 0.9)
        .with_section_score(b.id_for_section_in(dialogue, "c").unwrap(), 0.6);

    let p = b.project(
        ProjectionTarget { layer: dialogue, group: convo, timeline: TimelineId::for_test(1) },
        &resolver,
    );
    assert_eq!(p.system_prompt.len(), 1);
    assert_eq!(p.system_prompt[0].id, b_id);
}

#[test]
fn collection_always_visible_emits_every_section() {
    const YAML: &str = r#"
layers:
  - name: dialogue
    window: 4000
    score_formula: max
    budget: { priority: 100 }
    system_prompt:
      items:
        - kind: collection
          name: all
          sections:
            - id: a
              content: "a"
            - id: b
              content: "b"
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();

    let resolver = MockResolver::new();
    let p = b.project(
        ProjectionTarget { layer: dialogue, group: convo, timeline: TimelineId::for_test(1) },
        &resolver,
    );
    assert_eq!(p.system_prompt.len(), 2, "AlwaysVisible collection emits all");
}

#[test]
fn yaml_section_priority_field_parses() {
    const YAML: &str = r#"
layers:
  - name: dialogue
    window: 4000
    score_formula: max
    budget: { priority: 100 }
    system_prompt:
      items:
        - kind: collection
          name: pick
          selection: { kind: top_k, k: 1 }
          sections:
            - id: low_priority
              content: "L"
              priority: 10
            - id: high_priority
              content: "H"
              priority: 1000
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let high = b.id_for_section_in(dialogue, "high_priority").unwrap();

    let resolver = MockResolver::new(); // both score 0 by default → tie → priority wins
    let p = b.project(
        ProjectionTarget { layer: dialogue, group: convo, timeline: TimelineId::for_test(1) },
        &resolver,
    );
    assert_eq!(p.system_prompt.len(), 1);
    assert_eq!(p.system_prompt[0].id, high);
}

#[test]
fn collection_with_no_qualifying_sections_yields_empty_subset() {
    const YAML: &str = r#"
layers:
  - name: dialogue
    window: 4000
    score_formula: max
    budget: { priority: 100 }
    system_prompt:
      items:
        - kind: section
          id: framing
          content: "framing"
        - kind: collection
          name: tools
          selection: { kind: top_k, k: 2 }
          score_threshold: 0.5
          sections:
            - id: a
              content: "a"
            - id: b
              content: "b"
    groups:
      - id: convo
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml(YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let framing = b.id_for_section_in(dialogue, "framing").unwrap();

    let resolver = MockResolver::new()
        .with_section_score(b.id_for_section_in(dialogue, "a").unwrap(), 0.1)
        .with_section_score(b.id_for_section_in(dialogue, "b").unwrap(), 0.2);

    let p = b.project(
        ProjectionTarget { layer: dialogue, group: convo, timeline: TimelineId::for_test(1) },
        &resolver,
    );
    // Static framing still emits — only the collection drops everything.
    let ids: Vec<SectionId> = p.system_prompt.iter().map(|s| s.id).collect();
    assert_eq!(ids, vec![framing]);
}

// ── Builder runtime mutation ─────────────────────────────────────────────────

#[test]
fn add_section_appends_at_end_with_unique_id() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let n_before = b.schema().layers[0].system_prompt.items.len();
    let new_id = b
        .add_section(dialogue, "newsec", "new content", 75.0)
        .unwrap();
    assert_eq!(b.schema().layers[0].system_prompt.items.len(), n_before + 1);
    assert_eq!(b.id_for_section_in(dialogue, "newsec"), Some(new_id));
}

#[test]
fn add_section_duplicate_name_fails() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let result = b.add_section(dialogue, "alpha", "dup", 50.0);
    assert!(matches!(
        result,
        Err(super::error::ConstructionError::DuplicateSectionName(ref n)) if n == "alpha"
    ));
}

#[test]
fn add_collection_appends_and_returns_id() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let cid = b
        .add_collection(
            dialogue,
            "tools",
            super::schema::SelectionRule::TopK { k: 3 },
            0.0,
        )
        .unwrap();
    assert_eq!(b.id_for_collection_in(dialogue, "tools"), Some(cid));
}

#[test]
fn add_section_to_collection_appends_in_collection() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let cid = b
        .add_collection(
            dialogue,
            "tools",
            super::schema::SelectionRule::TopK { k: 2 },
            0.0,
        )
        .unwrap();
    let t1 = b
        .add_section_to_collection(dialogue, cid, "t1", "tool1", 50.0)
        .unwrap();
    let t2 = b
        .add_section_to_collection(dialogue, cid, "t2", "tool2", 50.0)
        .unwrap();
    let t3 = b
        .add_section_to_collection(dialogue, cid, "t3", "tool3", 50.0)
        .unwrap();
    // Section names are layer-scoped: t1/t2/t3 are unique even though
    // they're nested in a collection.
    assert_eq!(b.id_for_section_in(dialogue, "t1"), Some(t1));
    assert_eq!(b.id_for_section_in(dialogue, "t2"), Some(t2));
    assert_eq!(b.id_for_section_in(dialogue, "t3"), Some(t3));

    // Score them and verify projection top-k=2 picks the right two.
    let resolver = MockResolver::new()
        .with_section_score(t1, 0.1)
        .with_section_score(t2, 0.9)
        .with_section_score(t3, 0.5);

    let p = b.project(
        ProjectionTarget { layer: dialogue, group: convo, timeline: TimelineId::for_test(1) },
        &resolver,
    );
    // Static `alpha`/`beta`/`gamma` from SECTIONS_YAML_FLAT plus the
    // top-2 tools (t2, t3) from the collection.
    let ids: Vec<SectionId> = p.system_prompt.iter().map(|s| s.id).collect();
    assert!(ids.contains(&t2), "t2 (score 0.9) must survive top-2");
    assert!(ids.contains(&t3), "t3 (score 0.5) must survive top-2");
    assert!(!ids.contains(&t1), "t1 (score 0.1) must be filtered");
}

#[test]
fn add_section_to_unknown_collection_fails() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    // Construct a CollectionId that no add_collection call has issued.
    let bogus = super::ids::CollectionId::new(9999);
    let r = b.add_section_to_collection(dialogue, bogus, "x", "y", 50.0);
    assert!(matches!(
        r,
        Err(super::error::ConstructionError::UnknownCollection(_))
    ));
}

#[test]
fn duplicate_collection_name_fails() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    b.add_collection(
        dialogue,
        "tools",
        super::schema::SelectionRule::AlwaysVisible,
        0.0,
    )
    .unwrap();
    let r = b.add_collection(
        dialogue,
        "tools",
        super::schema::SelectionRule::AlwaysVisible,
        0.0,
    );
    assert!(matches!(
        r,
        Err(super::error::ConstructionError::DuplicateCollectionName(_))
    ));
}

#[test]
fn add_section_invalid_priority_fails() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let result = b.add_section(dialogue, "newsec", "content", 0.0);
    assert!(matches!(
        result,
        Err(super::error::ConstructionError::InvalidPriority { .. })
    ));
}
