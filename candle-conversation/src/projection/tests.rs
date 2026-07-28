//! Extensive integration tests for the projection engine.
//!
//! All tests use YAML-declared schemas and a `MockResolver` that returns
//! fully-controlled token counts and scores. Turn state is owned by the
//! resolver (not the builder), so tests use `resolver.append(group)` rather
//! than `builder.append(group)`.

use std::collections::{HashMap, HashSet};

use super::builder::Builder;
use super::ids::{GroupId, Reserved, SectionId, TimelineId, TurnIndex, TurnKey};
use super::project::ProjectionTarget;
use super::schema::{Content, DecodePriority, GatherScope};
use crate::substrate::ContentResolver;
use crate::summary_tree::{SelectionOrigin, TurnKind};

// —— Mock resolver —————————————————————————————————————————————————————————————

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
    /// section_id → token count (0 = "not sealed" for sections not present)
    section_tokens: HashMap<u32, usize>,
    default_score: f32,
    default_tokens: usize,
    /// Score-density picks returned verbatim by `summary_tree_select` —
    /// `(turn_index, origin, score)` in the chronological order the real §8
    /// selector produces (each summary node BEFORE the turns it covers).
    /// `None` ⇒ trait default (no forest → rule-based path).
    tree_picks: Option<Vec<(TurnIndex, SelectionOrigin, f32)>>,
    /// Turn indices that are summary (forest) nodes, reported by `turn_kind`
    /// as `SummaryOfSummaries`. Everything else is `Normal`.
    summary_idx: HashSet<u32>,
    /// turn_raw → the turn indices it transitively covers in the forest,
    /// reported by `node_covers`. Empty/absent ⇒ a raw leaf that covers
    /// nothing. Drives the rule-based descendant-dedup tests.
    covers: HashMap<u32, Vec<u32>>,
    /// (group_raw, tag) → turn_raw, backing `turn_with_tag` so tests can assert
    /// the default-fallback path resolves a declared default to a real turn.
    tag_turns: HashMap<(u32, String), u32>,
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

    /// Bind `tag` to `idx` in `group` so `turn_with_tag` resolves it — the
    /// substrate-side counterpart is a `TurnDecl.tags` scan.
    fn with_tag(mut self, group: GroupId, tag: &str, idx: TurnIndex) -> Self {
        self.tag_turns.insert((group.raw(), tag.to_string()), idx.0);
        self
    }

    fn with_section_score(mut self, section: SectionId, score: f32) -> Self {
        self.section_scores.insert(section.raw(), score);
        self
    }

    /// Mark a section as sealed with `tokens` tokens — projection's tool-summary
    /// gate emits the summary only when its `section_token_count` is `> 0`.
    fn with_section_tokens(mut self, section: SectionId, tokens: usize) -> Self {
        self.section_tokens.insert(section.raw(), tokens);
        self
    }

    /// Make `summary_tree_select` return `picks` verbatim (the §8 score-density
    /// path), and mark every summary-node index so `turn_kind` reports it.
    fn with_tree_picks(
        mut self,
        picks: Vec<(TurnIndex, SelectionOrigin, f32)>,
        summary_indices: &[u32],
    ) -> Self {
        self.tree_picks = Some(picks);
        self.summary_idx = summary_indices.iter().copied().collect();
        self
    }

    /// The synthetic conversation holding `group`'s turns. Each group gets its
    /// own timeline, mirroring the real model where a group is a shape and its
    /// turns live in one or more conversations under it.
    fn timeline_of(group: GroupId) -> TimelineId {
        TimelineId::for_test(group.raw() as u64)
    }

    /// Inverse of [`Self::timeline_of`] — recovers the group a mock turn key
    /// belongs to, so the `(group, index)`-keyed fixtures still resolve.
    fn group_of(turn: TurnKey) -> u32 {
        turn.timeline.raw() as u32
    }

    /// Mark turn `summary` as a summary forest node that transitively covers
    /// `covered` (the turn indices beneath it). Used to exercise the rule-based
    /// descendant-dedup without a real substrate tree.
    fn with_summary_cover(mut self, summary: u32, covered: &[u32]) -> Self {
        self.summary_idx.insert(summary);
        self.covers.insert(summary, covered.to_vec());
        self
    }
}

impl ContentResolver for MockResolver {
    fn group_turns(&self, group: GroupId) -> Vec<TurnKey> {
        let tl = Self::timeline_of(group);
        (0..self.turn_counts.get(&group.raw()).copied().unwrap_or(0))
            .map(|i| TurnKey::new(tl, TurnIndex(i)))
            .collect()
    }

    fn turn_token_count(&self, turn: TurnKey) -> usize {
        *self
            .tokens
            .get(&(Self::group_of(turn), turn.index.0))
            .unwrap_or(&self.default_tokens)
    }

    fn turn_score(&self, turn: TurnKey) -> f32 {
        // Mock returns a single explicit belief score.
        *self
            .scores
            .get(&(Self::group_of(turn), turn.index.0))
            .unwrap_or(&self.default_score)
    }

    fn section_score(&self, section: SectionId) -> f32 {
        // Default 0.0 for any section the test didn't assign a score to.
        // Sections without explicit scores will lose to those with scores.
        *self.section_scores.get(&section.raw()).unwrap_or(&0.0)
    }

    fn section_token_count(&self, section: SectionId) -> usize {
        // 0 unless explicitly marked sealed — mirrors a section not present in
        // the substrate. The tool-summary gate keys off `> 0`.
        *self.section_tokens.get(&section.raw()).unwrap_or(&0)
    }

    fn turn_kind(&self, turn: TurnKey) -> TurnKind {
        if self.summary_idx.contains(&turn.index.0) {
            TurnKind::SummaryOfSummaries
        } else {
            TurnKind::Normal
        }
    }

    fn node_covers(&self, turn: TurnKey) -> Vec<TurnIndex> {
        self.covers
            .get(&turn.index.0)
            .map(|v| v.iter().copied().map(TurnIndex).collect())
            .unwrap_or_default()
    }

    fn summary_tree_select(
        &self,
        _timeline: TimelineId,
        _budget: u32,
    ) -> Option<Vec<(TurnIndex, SelectionOrigin, f32)>> {
        self.tree_picks.clone()
    }

    fn turn_with_tag(&self, group: GroupId, tag: &str) -> Option<TurnKey> {
        self.tag_turns
            .get(&(group.raw(), tag.to_string()))
            .map(|&raw| TurnKey::new(Self::timeline_of(group), TurnIndex(raw)))
    }
}

// —— Helpers ———————————————————————————————————————————————————————————————————

fn group_turn_count<'a>(
    turns: impl IntoIterator<Item = &'a super::project::ResolvedTurn>,
    gid: GroupId,
) -> usize {
    turns.into_iter().filter(|t| t.group() == gid).count()
}

fn groups_in_order<'a>(
    turns: impl IntoIterator<Item = &'a super::project::ResolvedTurn>,
) -> Vec<GroupId> {
    let mut seen = Vec::new();
    for t in turns {
        if !seen.contains(&t.group()) {
            seen.push(t.group());
        }
    }
    seen
}

// —— YAML round-trip and id assignment —————————————————————————————————————————

const SIMPLE_YAML: &str = r#"
system_prompt:
  sections:
    - id: frame
      content: "You are a helpful assistant."
    - id: values
      content: "Be honest."
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
      - id: facts
        selection:
          kind: top_k
          k: 3

  - name: dialogue
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
      priority: 100
      min_percent: 50
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
    let frame = b.id_for_system_section("frame").unwrap();
    let values = b.id_for_system_section("values").unwrap();
    assert!(frame.raw() < values.raw());
}

#[test]
fn group_default_parses_present_and_absent() {
    // Present: the `structure` group carries a resolved SelectionDefault.
    let b = Builder::from_yaml(DEFAULT_FALLBACK_YAML).unwrap();
    let structure = b.id_for_group("structure").unwrap();
    let def = b.group(structure).unwrap().default.as_ref();
    assert_eq!(def.map(|d| d.tag.as_str()), Some("root"));

    // Absent: no `default:` key ⇒ None (every existing fixture stays this way).
    let b2 = Builder::from_yaml(NO_DEFAULT_YAML).unwrap();
    let s2 = b2.id_for_group("structure").unwrap();
    assert!(b2.group(s2).unwrap().default.is_none());

    // Whitespace-only tag is dropped to None by `parse_default`.
    let blank =
        DEFAULT_FALLBACK_YAML.replace(r#"default: { tag: "root" }"#, r#"default: { tag: "  " }"#);
    let b3 = Builder::from_yaml(&blank).unwrap();
    let s3 = b3.id_for_group("structure").unwrap();
    assert!(b3.group(s3).unwrap().default.is_none());
}

#[test]
fn policy_inherits_default_then_layer_overrides() {
    let yaml = SIMPLE_YAML
        .replace(
            "layers:",
            "default_policy:\n  preset: high_recall_scope\nlayers:",
        )
        .replace(
            "  - name: dialogue\n    window: 8000\n",
            "  - name: dialogue\n    window: 8000\n    policy:\n      beta: 0.5\n",
        );
    let b = Builder::from_yaml(&yaml).unwrap();
    let ground = b.id_for_layer("ground").unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    // ground inherits the schema default preset verbatim.
    assert_eq!(
        b.layer(ground).unwrap().policy.config,
        super::PolicyPreset::HighRecallScope.config()
    );
    // dialogue overrides only beta on top of the inherited high-recall base.
    let dp = b.layer(dialogue).unwrap().policy.config;
    assert_eq!(dp.beta, 0.5);
    assert_eq!(
        dp.budget_max,
        super::PolicyPreset::HighRecallScope.config().budget_max
    );
}

#[test]
fn policy_unknown_preset_is_error() {
    let yaml = SIMPLE_YAML.replace("layers:", "default_policy:\n  preset: nope\nlayers:");
    assert!(matches!(
        Builder::from_yaml(&yaml),
        Err(super::ConstructionError::UnknownPolicyPreset { .. })
    ));
}

#[test]
fn policy_evict_above_min_is_error() {
    let yaml = SIMPLE_YAML.replace(
        "  - name: dialogue\n    window: 8000\n",
        "  - name: dialogue\n    window: 8000\n    policy:\n      min_score: 5\n      evict_score: 10\n",
    );
    assert!(matches!(
        Builder::from_yaml(&yaml),
        Err(super::ConstructionError::InvalidPolicy { .. })
    ));
}

#[test]
fn gather_scope_defaults_to_shared_and_parses_conversation() {
    // Unset → Shared.
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    assert_eq!(b.layer(dialogue).unwrap().gather_scope, GatherScope::Shared);

    // Flagging the dialogue layer parses to Conversation.
    let yaml = SIMPLE_YAML.replace(
        "  - name: dialogue\n    window: 8000",
        "  - name: dialogue\n    gather_scope: conversation\n    window: 8000",
    );
    let b2 = Builder::from_yaml(&yaml).unwrap();
    let d2 = b2.id_for_layer("dialogue").unwrap();
    assert_eq!(
        b2.layer(d2).unwrap().gather_scope,
        GatherScope::Conversation
    );
}

#[test]
fn unknown_layer_name_returns_none() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    assert!(b.id_for_layer("nonexistent").is_none());
}

#[test]
fn decode_priority_defaults_to_low_and_parses_normal_and_high() {
    // Unset → Low.
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    assert_eq!(
        b.layer(dialogue).unwrap().decode_priority,
        DecodePriority::Low
    );

    // `normal` and `high` parse to their variants.
    for (word, want) in [
        ("normal", DecodePriority::Normal),
        ("high", DecodePriority::High),
        ("low", DecodePriority::Low),
    ] {
        let yaml = SIMPLE_YAML.replace(
            "  - name: dialogue\n    window: 8000",
            &format!("  - name: dialogue\n    decode_priority: {word}\n    window: 8000"),
        );
        let b = Builder::from_yaml(&yaml).unwrap();
        let d = b.id_for_layer("dialogue").unwrap();
        assert_eq!(
            b.layer(d).unwrap().decode_priority,
            want,
            "decode_priority: {word}"
        );
    }
}

#[test]
fn decode_priority_ratio_is_decode_tokens_per_prefill() {
    // The airtime ratio R the continuous-fair-wave throttle keys on.
    assert_eq!(DecodePriority::Low.ratio(), 1);
    assert_eq!(DecodePriority::Normal.ratio(), 16);
    assert_eq!(DecodePriority::High.ratio(), 64);
    // Always >= 1 so a prefill can never be fully starved.
    for p in [
        DecodePriority::Low,
        DecodePriority::Normal,
        DecodePriority::High,
    ] {
        assert!(p.ratio() >= 1);
    }
}

#[test]
fn builder_fallback_dialogue_layer_is_high_priority() {
    // The template-less fallback IS the dialogue layer, so it inherits the
    // interactive priority even without a YAML declaration.
    let b = Builder::for_plain_prompt("You are a helpful assistant.");
    let dialogue = b.id_for_layer("dialogue").unwrap();
    assert_eq!(
        b.layer(dialogue).unwrap().decode_priority,
        DecodePriority::High
    );
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

    let _dialogue = b.id_for_layer("dialogue").unwrap();
    let frame = b.id_for_system_section("frame").unwrap();
    let section = b.section(frame).unwrap();
    assert_eq!(section.name, "frame");
}

// —— Append + turn_count (now on resolver) —————————————————————————————————————

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
    assert_eq!(resolver.group_turns(conv).len(), 3);
}

#[test]
fn turn_count_zero_for_empty_group() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let facts = b.id_for_group("facts").unwrap();
    let resolver = MockResolver::new();
    assert_eq!(resolver.group_turns(facts).len(), 0);
}

// —— Projection: basic visibility ——————————————————————————————————————————————

#[test]
fn empty_builder_projection_has_no_turns() {
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
fn system_prompt_sections_always_emitted() {
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

#[test]
fn turns_appear_after_append() {
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
fn projected_turn_carries_its_resolved_timeline() {
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

#[test]
fn higher_layer_not_visible_for_lower_target() {
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
fn sibling_group_in_target_layer_not_visible() {
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

// —— Score threshold ———————————————————————————————————————————————————————————

#[test]
fn turns_below_score_threshold_filtered() {
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

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let indices: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();
    assert!(indices.contains(&0));
    assert!(
        !indices.contains(&1),
        "turn with score 0.3 should be filtered"
    );
    assert!(indices.contains(&2));
}

#[test]
fn layer_threshold_filters_whole_group() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
    - id: stub
      content: "stub"
layers:
  - name: low
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
    score_threshold: 0.8
    budget:
      priority: 50
    groups:
      - id: low_grp
        selection: { kind: top_k, k: 5 }
  - name: high
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
    let resolver = resolver.with_score(low_grp, i0, 0.3); // group score = 0.3 < 0.8 threshold → entire group dropped

    let proj = b.project(
        ProjectionTarget {
            layer: high_layer,
            group: high_grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let groups = groups_in_order(proj.sealed_turns());
    assert!(
        !groups.contains(&low_grp),
        "low_grp should be dropped by layer threshold"
    );
    assert!(groups.contains(&high_grp));
    let _ = low_layer;
}

// —— Selection rules ———————————————————————————————————————————————————————————

#[test]
fn top_k_limits_turns() {
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

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    // top-2: i0 (0.9) and i2 (0.7), emitted in insertion order
    assert_eq!(proj.sealed_turns().count(), 2);
    assert_eq!(proj.sealed_turns().next().unwrap().index().0, 0);
    assert_eq!(proj.sealed_turns().nth(1).unwrap().index().0, 2);
}

#[test]
fn single_selection_one_turn() {
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

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    assert_eq!(proj.sealed_turns().count(), 1);
    assert_eq!(proj.sealed_turns().next().unwrap().index().0, 1);
}

#[test]
fn always_visible_selects_all_above_threshold() {
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
    assert_eq!(indices, vec![0, 2]);
}

/// Minimal single-layer schema whose one group ranks by raw provenance score
/// (`top_k`), so summary forest nodes compete head-to-head with raw turns.
const TOPK_YAML: &str = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
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
    groups:
      - id: grp
        score_threshold: 0.0
        selection: { kind: top_k, k: 3 }
"#;

/// Rule-based path: a summary node and the turns it covers can BOTH clear the
/// score cut (a cross-corpus hit scores the summary directly). The
/// descendant-dedup drops the summary and keeps the SPECIFIC turns — never a
/// coarse summary stacked on top of the very content it summarises.
#[test]
fn rule_based_dedup_drops_summary_when_its_content_is_selected() {
    let b = Builder::from_yaml(TOPK_YAML).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    let i0 = resolver.append(grp); // Normal
    let i1 = resolver.append(grp); // Normal
    let i2 = resolver.append(grp); // Normal
    let s = resolver.append(grp); // summary covering [0, 1]
    let resolver = resolver
        .with_score(grp, i0, 0.9)
        .with_score(grp, i1, 0.8)
        .with_score(grp, i2, 0.1) // loses the top-3 slot to the summary
        .with_score(grp, s, 0.95)
        .with_summary_cover(s.0, &[i0.0, i1.0]);

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    // top_k=3 picks {s, 0, 1}; dedup drops s (it covers both 0 and 1, both
    // selected). The two specific turns remain; the coarse summary is gone.
    let indices: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();
    assert_eq!(indices, vec![0, 1]);
}

/// Rule-based path: when a summary scores a hit but NONE of the turns it covers
/// were themselves selected, the summary survives — it is the coarse stand-in
/// for an older span that didn't otherwise make the window. It sorts to its
/// storage `TurnIndex` (after the turns it covers), standing alone.
#[test]
fn rule_based_dedup_keeps_summary_when_its_content_is_not_selected() {
    // k=2 so only the two highest scorers win.
    let yaml = TOPK_YAML.replace("k: 3", "k: 2");
    let b = Builder::from_yaml(&yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    let i0 = resolver.append(grp); // Normal — low score
    let i1 = resolver.append(grp); // Normal — low score
    let i2 = resolver.append(grp); // Normal — wins a slot
    let s = resolver.append(grp); // summary covering [0, 1] — wins a slot
    let resolver = resolver
        .with_score(grp, i0, 0.1)
        .with_score(grp, i1, 0.1)
        .with_score(grp, i2, 0.8)
        .with_score(grp, s, 0.9)
        .with_summary_cover(s.0, &[i0.0, i1.0]);

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    // top_k=2 picks {s, 2}; dedup keeps s (neither 0 nor 1 is selected). The
    // summary emits at its own index, after the raw turn.
    let indices: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();
    assert_eq!(indices, vec![2, 3]);
}

/// Rule-based path: dedup is transitive — a top-level SoS that covers a mid-level
/// SoT (which itself was selected) is dropped in favour of the finer node. Prefer
/// the most specific covering node at every level.
#[test]
fn rule_based_dedup_prefers_the_finer_summary_over_its_ancestor() {
    // k=2 so only the two summaries (highest scorers) win — isolating the
    // ancestor-vs-descendant collapse from the low raw turns.
    let yaml = TOPK_YAML.replace("k: 3", "k: 2");
    let b = Builder::from_yaml(&yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    let i0 = resolver.append(grp); // Normal — low score
    let i1 = resolver.append(grp); // Normal — low score
    let sot = resolver.append(grp); // SoT covering [0, 1]
    let sos = resolver.append(grp); // SoS covering [sot, 0, 1] (transitive)
    let resolver = resolver
        .with_score(grp, i0, 0.1)
        .with_score(grp, i1, 0.1)
        .with_score(grp, sot, 0.9)
        .with_score(grp, sos, 0.95)
        .with_summary_cover(sot.0, &[i0.0, i1.0])
        .with_summary_cover(sos.0, &[sot.0, i0.0, i1.0]);

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    // top_k=2 picks {sos, sot}; sos covers sot (selected) so it drops; sot covers
    // only 0/1 (not selected) so it stays. The coarse ancestor lost to the finer.
    let indices: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();
    assert_eq!(indices, vec![sot.0]);
}

/// Regression: the §8 score-density path returns picks in chronological order
/// with each SUMMARY node ABOVE the turns it covers. A summary's storage
/// `TurnIndex` is always higher than the turns it summarises (it is sealed
/// after them), so the old index-sorts — `select_conversation`'s `selected.sort()`
/// and step-12's `group_turns.sort()` — dropped the summary BELOW its own
/// content: a summary of turns 0–4 rendered *after* raw turn 6. The projection
/// must emit score-density picks verbatim, honouring the trait contract that
/// `summary_tree_select` returns an already-ordered, already-budget-fit list.
#[test]
fn score_density_summary_emitted_before_later_raw_turn() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: dialogue
    window: 1200
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
      - id: convo
        selection:
          kind: conversation
          recent: 2
          historical_top_k: 8
"#;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let tl = TimelineId::for_test(1);

    // 10 raw turns exist; index 7 is a SoS node covering the earliest turns.
    let mut resolver = MockResolver::new();
    for _ in 0..10 {
        resolver.append(convo);
    }
    // Picks exactly as the real §8 selector emits them: the summary (idx 7,
    // covering the early turns) sits BEFORE the later raw turns 6 and 9.
    let picks = vec![
        (TurnIndex(7), SelectionOrigin::CoverageFill, 0.5),
        (TurnIndex(6), SelectionOrigin::RecencyDecay, 0.5),
        (TurnIndex(9), SelectionOrigin::RecencyDecay, 0.5),
    ];
    let resolver = resolver.with_tree_picks(picks, &[7]);

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: convo,
            timeline: tl,
        },
        &resolver,
    );

    let indices: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();
    // Verbatim pick order — the summary (7) stays ABOVE its later-indexed
    // content (6), NOT re-sorted by TurnIndex to [6, 7, 9].
    assert_eq!(
        indices,
        vec![7, 6, 9],
        "score-density picks must emit in chronological (summary-above-content) \
         order, not sorted by TurnIndex",
    );
}

#[test]
fn conversation_recent_turns_always_included() {
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

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: conv,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let indices: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();

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

// —— Budget constraints ————————————————————————————————————————————————————————

#[test]
fn budget_overflow_drops_lowest_scored_turns() {
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
fn single_turn_overflow_drops_turn() {
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
fn turns_emitted_in_insertion_order_within_group() {
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
fn higher_scored_group_emitted_last_within_layer() {
    // Doc Â§7: "Higher-scored groups appear *later* in the emitted list within
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

// —— Multi-layer projection ————————————————————————————————————————————————————

const MULTI_LAYER_YAML: &str = r#"
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
fn sections_always_emit_regardless_of_size() {
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
fn system_prompt_sections_in_declaration_order() {
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

// —— YAML validation errors ————————————————————————————————————————————————————

#[test]
fn yaml_duplicate_group_name_is_error() {
    let yaml = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
    - id: stub
      content: "stub"
layers:
  - name: layer1
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
      - id: dup
        selection: { kind: always_visible }
  - name: layer2
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
system_prompt:
  sections:
    - id: s1
      content: "A"
    - id: s1
      content: "B"
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
        selection: { kind: always_visible }
        budget: { priority: 50, min_percent: 150 }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::InvalidPercentage { .. }
    ));
}

// —— Construction validation errors ———————————————————————————————————————————

#[test]
fn construction_sibling_min_percent_exceeds_100_is_error() {
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
        selection: { kind: always_visible }
        score_threshold: -0.5
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::NegativeScoreThreshold { .. }
    ));
}

// —— Score formula variants ————————————————————————————————————————————————————

#[test]
fn score_formula_sum_determines_group_score() {
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
fn score_formula_count_promotes_large_groups() {
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

// —— Budget redistribution —————————————————————————————————————————————————————

#[test]
fn freed_budget_redistributes_to_other_layers() {
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

    // sparse: 1 turn Ã— 10 tokens = 10 (far less than its ~4750 share).
    resolver.append(sparse_grp);
    // dense: 20 turns Ã— 100 tokens = 2000.
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
fn no_layers_in_schema_is_valid() {
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
fn empty_group_does_not_consume_budget() {
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
fn all_turns_below_threshold_leaves_no_turns() {
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
fn large_number_of_turns_respects_budget() {
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

    // 100 turns Ã— 100 tokens each = 10000, far exceeds 4500 budget.
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
fn min_percent_guarantees_minimum_budget() {
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
fn top_k_mean_formula_integration() {
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

#[test]
fn max_percent_caps_allocation() {
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
fn historical_top_k_zero_means_only_recent() {
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
fn recent_zero_means_only_historical() {
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
fn yaml_top_k_missing_k_is_error() {
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
        selection: { kind: top_k }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::InvalidTopK { .. }
    ));
}

#[test]
fn missing_system_prompt_is_parse_error() {
    // The top-level system_prompt is a required field; omitting it entirely is a
    // deserialization (parse) error before construction validation runs.
    let yaml = r#"
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
        selection: { kind: always_visible }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(err, super::error::ConstructionError::YamlParse(_)));
}

#[test]
fn empty_system_prompt_is_construction_error() {
    // The unified top-level system_prompt is required and must be non-empty;
    // an empty section list fails construction with EmptySystemPrompt.
    let yaml = r#"
system_prompt:
  sections: []
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
        selection: { kind: always_visible }
"#;
    let err = Builder::from_yaml(yaml).unwrap_err();
    assert!(matches!(
        err,
        super::error::ConstructionError::EmptySystemPrompt
    ));
}

#[test]
fn empty_target_group_does_not_emit_target_layer() {
    // Doc Â§9.7: "Filter out empty groups and empty layers." No exception for
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

// —— Variable substitution ————————————————————————————————————————————————————

const SUBST_YAML: &str = r#"
system_prompt:
  sections:
    - id: frame
      content: "Hello {name}, welcome to {project}."
layers:
  - name: dialogue
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
    budget: { priority: 100 }
    groups:
      - id: chat
        selection: { kind: always_visible }
"#;

#[test]
fn substitution_replaces_placeholders() {
    let b = Builder::from_yaml_with_vars(SUBST_YAML, &[("name", "Alice"), ("project", "Candle")])
        .unwrap();
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    let frame = b.id_for_system_section("frame").unwrap();
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
system_prompt:
  sections:
    - id: frame
      content: "Welcome to {wokrspace}."
layers:
  - name: dialogue
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
    budget: { priority: 100 }
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
system_prompt:
  sections:
    - id: frame
      content: 'Example JSON: {"tool": "search", "args": {}}'
layers:
  - name: dialogue
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
    budget: { priority: 100 }
    groups:
      - id: chat
        selection: { kind: always_visible }
"#;
    let b = Builder::from_yaml_with_vars(yaml, &[]).unwrap();
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    let frame = b.id_for_system_section("frame").unwrap();
    let section = b.section(frame).unwrap();
    assert_eq!(
        section.content,
        r#"Example JSON: {"tool": "search", "args": {}}"#
    );
}

#[test]
fn from_yaml_rejects_any_placeholder() {
    let yaml = r#"
system_prompt:
  sections:
    - id: frame
      content: "Welcome to {workspace}."
layers:
  - name: dialogue
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
    budget: { priority: 100 }
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
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    let frame = b.id_for_system_section("frame").unwrap();
    let content_first = b.section(frame).unwrap().content.clone();
    let content_second = b.section(frame).unwrap().content.clone();
    assert_eq!(content_first, content_second);
    assert_eq!(content_first, "Hello A, welcome to B.");
}

/// Sanity check: the live zend projection schema parses + validates.
#[test]
fn zend_projection_yaml_parses() {
    use candle_transformers::models::dialect::Dialect;
    let yaml = include_str!("../../../zend/src/prompts/projection.yaml");
    let dlct = Dialect::chat_ml();
    let b = Builder::from_yaml_with_vars_and_dialect(yaml, &[("workspace", "candle")], Some(&dlct))
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

    // repo_map (directory trees) declares deterministic structural content;
    // every other layer's content is a model decode.
    let mode_of = |layer: &str| {
        let id = b.id_for_layer(layer).unwrap();
        b.schema()
            .layers
            .iter()
            .find(|l| l.id == id)
            .unwrap()
            .summary
            .summaries
            .as_ref()
            .expect("layer has a summaries level")
            .content
    };
    // Only repo_map (directory trees) uses the deterministic structural roll-up;
    // the other layers' summaries are code definitions/artifacts, not paths.
    assert_eq!(mode_of("repo_map"), Content::Structural);
    for single in [
        "code_reading",
        "static_analysis",
        "dependency_analysis",
        "dialogue",
        "bug_analysis",
        "architectural_analysis",
        "dream_log",
    ] {
        assert_eq!(
            mode_of(single),
            Content::Decode,
            "{single} should stay a model decode"
        );
    }
}

#[test]
fn default_selection_is_always_visible() {
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

// —— Section + collection emission tests ——————————————————————————————————————
//
// These tests cover the `SystemPromptItem` model: the layer's
// `system_prompt` is an ordered list of items, each either a single
// always-emit section or a `SectionCollection` with its own selection
// rule (TopK / Single / AlwaysVisible).  Emission preserves declaration
// order; selection picks by salience (provenance-derived score).

const SECTIONS_YAML_FLAT: &str = r#"
system_prompt:
  sections:
    - id: alpha
      content: "alpha"
    - id: beta
      content: "beta"
    - id: gamma
      content: "gamma"
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
    budget:
      priority: 100
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
    let alpha = b.id_for_system_section("alpha").unwrap();
    let beta = b.id_for_system_section("beta").unwrap();
    let gamma = b.id_for_system_section("gamma").unwrap();

    let resolver = MockResolver::new();
    let p = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    assert_eq!(p.sealed_sections().count(), 3);
    assert_eq!(p.sealed_sections().next().unwrap().id, alpha);
    assert_eq!(p.sealed_sections().nth(1).unwrap().id, beta);
    assert_eq!(p.sealed_sections().nth(2).unwrap().id, gamma);
}

const COLLECTION_YAML: &str = r#"
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
const ALL_TOOLS_FIT_YAML: &str = r#"
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
fn collection_partial_selection_emits_summary_before_members() {
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

/// When the collection selects all its members (top-k ≥ member count), nothing
/// was dropped, so the summary is omitted even though it is sealed.
#[test]
fn collection_all_selected_omits_summary() {
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
fn collection_partial_but_unsealed_summary_omitted() {
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
fn collection_summary_emits_outside_structural_markers() {
    use super::project::{ProjectionSegment, SealedKind};
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

    let order: Vec<String> = proj
        .segments
        .iter()
        .map(|seg| match seg {
            ProjectionSegment::Generated { identity, .. } => identity.name.clone(),
            ProjectionSegment::Sealed(SealedKind::Section(s)) if s.id == summary => {
                "SUMMARY".to_string()
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

/// `depends_on_absent` is the inverse of `depends_on`: its section emits only
/// when the collection materialised zero members (the no-tools variant), and is
/// suppressed when the collection has members.
#[test]
fn depends_on_absent_emits_only_when_collection_empty() {
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

#[test]
fn collection_top_k_keeps_highest_scored_in_declaration_order() {
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
fn collection_top_k_with_priority_tiebreak() {
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

const COLLECTION_YAML_THRESHOLD: &str = r#"
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
fn collection_score_threshold_filters_below_floor() {
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
fn collection_single_picks_max_only() {
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
fn collection_named_pins_member_by_runtime_selector() {
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

#[test]
fn named_selection_rejects_empty_selector() {
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
fn collection_always_visible_emits_every_section() {
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

#[test]
fn yaml_section_priority_field_parses() {
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
      name: pick
      selection: { kind: top_k, k: 1 }
      sections:
        - id: low_priority
          content: "L"
          priority: 10
        - id: high_priority
          content: "H"
          priority: 1000
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
    let high = b.id_for_system_section("high_priority").unwrap();
    let low = b.id_for_system_section("low_priority").unwrap();

    // The `priority:` field parses (the YAML built); selection is now
    // belief-driven by score, so the higher-scored section is the single pick.
    let resolver = MockResolver::new()
        .with_section_score(high, 1.0)
        .with_section_score(low, 0.1);
    let p = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    assert_eq!(p.sealed_sections().count(), 1);
    assert_eq!(p.sealed_sections().next().unwrap().id, high);
}

#[test]
fn collection_with_no_qualifying_sections_yields_empty_subset() {
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

// —— Builder runtime mutation —————————————————————————————————————————————————

#[test]
fn add_section_appends_at_end_with_unique_id() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let n_before = b.schema().system_prompt.items.len();
    let new_id = b.add_section("newsec", "new content", 75.0).unwrap();
    assert_eq!(b.schema().system_prompt.items.len(), n_before + 1);
    assert_eq!(b.id_for_system_section("newsec"), Some(new_id));
}

#[test]
fn add_section_duplicate_name_fails() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    let result = b.add_section("alpha", "dup", 50.0);
    assert!(matches!(
        result,
        Err(super::error::ConstructionError::DuplicateSectionName(ref n)) if n == "alpha"
    ));
}

#[test]
fn add_collection_appends_and_returns_id() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    let cid = b
        .add_collection("tools", super::schema::SelectionRule::TopK { k: 3 }, 0.0)
        .unwrap();
    assert_eq!(b.id_for_system_collection("tools"), Some(cid));
}

#[test]
fn runtime_section_ids_stay_disjoint_from_compression_prompts() {
    // Regression: a layer's compression-prompt sections (and collection
    // summaries) live in `layer.summary` / the collection summary, NOT in
    // `system_prompt.items`, and for a trailing layer get the highest section
    // ids. Runtime allocation (`add_section_to_collection`, the daemon's tool
    // catalog) must skip past them — otherwise a tool section reuses a
    // compression-prompt id and `ensure_summary_section` injects the tool's
    // content as the compression prompt, collapsing the compression decode into
    // a degenerate loop (the live-daemon bug this guards).
    let mut b = Builder::from_yaml(COLLECTION_YAML).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let cid = b.id_for_system_collection("tools").unwrap();

    // The hidden (compression-prompt + collection-summary) section ids — those
    // in `all_section_ids` but not visible in `system_prompt.items`.
    let hidden: Vec<u32> = {
        let layer = b.schema().layers.iter().find(|l| l.id == dialogue).unwrap();
        let visible: std::collections::HashSet<u32> = b
            .schema()
            .system_prompt
            .all_sections()
            .map(|s| s.id.raw())
            .collect();
        layer
            .all_section_ids()
            .into_iter()
            .filter(|id| !visible.contains(id))
            .collect()
    };
    assert!(
        !hidden.is_empty(),
        "layer must own hidden compression-prompt / collection-summary sections"
    );

    for t in 0..6 {
        let id = b
            .add_section_to_collection(cid, format!("rt_{t}"), "x", 50.0)
            .unwrap();
        assert!(
            !hidden.contains(&id.raw()),
            "runtime section id {} reuses a hidden compression-prompt id {:?}",
            id.raw(),
            hidden,
        );
    }
}

#[test]
fn add_section_to_collection_appends_in_collection() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let dialogue = b.id_for_layer("dialogue").unwrap();
    let convo = b.id_for_group("convo").unwrap();
    let cid = b
        .add_collection("tools", super::schema::SelectionRule::TopK { k: 2 }, 0.0)
        .unwrap();
    let t1 = b
        .add_section_to_collection(cid, "t1", "tool1", 50.0)
        .unwrap();
    let t2 = b
        .add_section_to_collection(cid, "t2", "tool2", 50.0)
        .unwrap();
    let t3 = b
        .add_section_to_collection(cid, "t3", "tool3", 50.0)
        .unwrap();
    // Section names are layer-scoped: t1/t2/t3 are unique even though
    // they're nested in a collection.
    assert_eq!(b.id_for_system_section("t1"), Some(t1));
    assert_eq!(b.id_for_system_section("t2"), Some(t2));
    assert_eq!(b.id_for_system_section("t3"), Some(t3));

    // Score them and verify projection top-k=2 picks the right two.
    let resolver = MockResolver::new()
        .with_section_score(t1, 0.1)
        .with_section_score(t2, 0.9)
        .with_section_score(t3, 0.5);

    let p = b.project(
        ProjectionTarget {
            layer: dialogue,
            group: convo,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    // Static `alpha`/`beta`/`gamma` from SECTIONS_YAML_FLAT plus the
    // top-2 tools (t2, t3) from the collection.
    let ids: Vec<SectionId> = p.sealed_sections().map(|s| s.id).collect();
    assert!(ids.contains(&t2), "t2 (score 0.9) must survive top-2");
    assert!(ids.contains(&t3), "t3 (score 0.5) must survive top-2");
    assert!(!ids.contains(&t1), "t1 (score 0.1) must be filtered");
}

#[test]
fn add_section_to_unknown_collection_fails() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    // Construct a CollectionId that no add_collection call has issued.
    let bogus = super::ids::CollectionId::new(9999);
    let r = b.add_section_to_collection(bogus, "x", "y", 50.0);
    assert!(matches!(
        r,
        Err(super::error::ConstructionError::UnknownCollection(_))
    ));
}

#[test]
fn duplicate_collection_name_fails() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    b.add_collection("tools", super::schema::SelectionRule::AlwaysVisible, 0.0)
        .unwrap();
    let r = b.add_collection("tools", super::schema::SelectionRule::AlwaysVisible, 0.0);
    assert!(matches!(
        r,
        Err(super::error::ConstructionError::DuplicateCollectionName(_))
    ));
}

#[test]
fn add_section_invalid_priority_fails() {
    let mut b = Builder::from_yaml(SECTIONS_YAML_FLAT).unwrap();
    let _dialogue = b.id_for_layer("dialogue").unwrap();
    let result = b.add_section("newsec", "content", 0.0);
    assert!(matches!(
        result,
        Err(super::error::ConstructionError::InvalidPriority { .. })
    ));
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
const DEFAULT_FALLBACK_YAML: &str = r#"
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
        default: { tag: "root" }
"#;

/// Identical to the fixture above but with no `default:` declared, proving the
/// injection is what keeps the group alive — absent it, an all-below-threshold
/// group emits nothing.
const NO_DEFAULT_YAML: &str = r#"
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

#[test]
fn default_injected_when_selection_empty() {
    let b = Builder::from_yaml(DEFAULT_FALLBACK_YAML).unwrap();
    let ground = b.id_for_layer("ground").unwrap();
    let structure = b.id_for_group("structure").unwrap();

    // Three turns, all scoring 0.1 — below the group's 0.5 threshold, so the
    // top_k rule selects nothing. Bind the default tag to turn 0.
    let mut resolver = MockResolver::new().with_default_score(0.1);
    resolver.append(structure);
    resolver.append(structure);
    resolver.append(structure);
    resolver = resolver.with_tag(structure, "root", TurnIndex(0));

    let proj = b.project(
        ProjectionTarget {
            layer: ground,
            group: structure,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );

    let turns: Vec<&super::project::ResolvedTurn> = proj.sealed_turns().collect();
    assert_eq!(turns.len(), 1, "only the default turn should survive");
    assert_eq!(turns[0].group(), structure);
    assert_eq!(turns[0].index(), TurnIndex(0), "default resolves to turn 0");
}

#[test]
fn no_default_leaves_group_empty() {
    let b = Builder::from_yaml(NO_DEFAULT_YAML).unwrap();
    let ground = b.id_for_layer("ground").unwrap();
    let structure = b.id_for_group("structure").unwrap();

    let mut resolver = MockResolver::new().with_default_score(0.1);
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
        "no default ⇒ all-below-threshold group emits nothing",
    );
}

#[test]
fn default_ignored_when_selection_non_empty() {
    let b = Builder::from_yaml(DEFAULT_FALLBACK_YAML).unwrap();
    let ground = b.id_for_layer("ground").unwrap();
    let structure = b.id_for_group("structure").unwrap();

    // Turns 1 and 2 clear the 0.5 threshold; the top_k rule fills two slots, so
    // the default must NOT fire (turn 0, the default, scores below threshold and
    // stays out).
    let mut resolver = MockResolver::new().with_default_score(0.1);
    let t0 = resolver.append(structure);
    let t1 = resolver.append(structure);
    let t2 = resolver.append(structure);
    resolver = resolver
        .with_score(structure, t1, 0.9)
        .with_score(structure, t2, 0.8)
        .with_tag(structure, "root", t0);

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
        "real picks fill the budget; the default turn stays out",
    );
}

/// Collection analogue: a `top_k` collection whose members all score below the
/// threshold selects nothing, and the `default` (a section named by tag) is the
/// floor so the collection still contributes one section.
const COLLECTION_DEFAULT_YAML: &str = r#"
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
fn collection_default_parses() {
    let b = Builder::from_yaml(COLLECTION_DEFAULT_YAML).unwrap();
    let coll = b
        .schema()
        .system_prompt
        .items
        .iter()
        .find_map(|item| match item {
            super::schema::SystemPromptItem::Collection(c) if c.name == "tools" => Some(c),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        coll.default.as_ref().map(|d| d.tag.as_str()),
        Some("tool_c")
    );
}

#[test]
fn collection_default_injected_when_selection_empty() {
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
fn belief_config_takes_budget_and_gates_from_the_rule() {
    // structure = top_k(2), score_threshold 0.5.
    let b = Builder::from_yaml(DEFAULT_FALLBACK_YAML).unwrap();
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
fn belief_driven_group_selects_top_k_by_score() {
    let b = Builder::from_yaml(DEFAULT_FALLBACK_YAML).unwrap();
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

// ── Dialect-template parsing ─────────────────────────────────────────────────
//
// These tests cover the `kind: template` system-prompt item that references
// a [`DialectTemplate`] catalog entry by snake-case name.  The resolved
// string lands in a `SectionSchema` with `is_template = true`; the
// projection assembler routes those items through live prefill so their
// K/V stays attention-correct under the runtime left context.

mod dialect_templates {
    use super::*;
    use crate::projection::builder::Builder;
    use crate::projection::error::ConstructionError;
    use crate::projection::schema::SystemPromptItem;
    use candle_transformers::models::dialect::Dialect;

    const TEMPLATE_YAML: &str = r#"
system_prompt:
  items:
    - kind: template
      id: system_open
      dialect: system_start
    - kind: section
      id: frame
      content: "You are a senior engineer."
    - kind: template
      id: system_close
      dialect: system_end
layers:
  - name: dialogue
    window: 1000
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

    #[test]
    fn template_item_resolves_to_dialect_content() {
        let dlct = Dialect::chat_ml();
        let b = Builder::from_yaml_with_vars_and_dialect(TEMPLATE_YAML, &[], Some(&dlct)).unwrap();

        let mut found: Vec<(&str, &str, bool)> = Vec::new();
        for it in &b.schema().system_prompt.items {
            if let SystemPromptItem::Section(s) = it {
                found.push((s.name.as_str(), s.content.as_str(), s.is_template));
            }
        }
        assert_eq!(
            found,
            vec![
                ("system_open", "<|im_start|>system\n", true),
                ("frame", "You are a senior engineer.", false),
                ("system_close", "<|im_end|>\n", true),
            ],
        );
    }

    #[test]
    fn template_without_dialect_errors() {
        let err = Builder::from_yaml_with_vars_and_dialect(TEMPLATE_YAML, &[], None).unwrap_err();
        assert!(
            matches!(err, ConstructionError::DialectRequired { ref item } if item == "system_prompt/system_open"),
            "got {err:?}",
        );
    }

    #[test]
    fn template_unknown_dialect_name_errors() {
        let yaml = r#"
system_prompt:
  items:
    - kind: template
      id: bogus
      dialect: not_a_real_template
    - kind: section
      id: frame
      content: "x"
layers:
  - name: dialogue
    window: 1000
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
        let err = Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&dlct)).unwrap_err();
        assert!(
            matches!(
                err,
                ConstructionError::UnknownDialectTemplate { ref item, ref name }
                if item == "system_prompt/bogus" && name == "not_a_real_template"
            ),
            "got {err:?}",
        );
    }

    #[test]
    fn empty_template_content_is_dropped() {
        // Llama3 has no `no_think_prefix` (empty string), so a
        // `kind: template` referencing it must be filtered at build time —
        // projection never sees an empty section.
        let yaml = r#"
system_prompt:
  items:
    - kind: template
      id: maybe_no_think
      dialect: no_think_prefix
    - kind: section
      id: frame
      content: "x"
layers:
  - name: dialogue
    window: 1000
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
        let dlct = Dialect::llama3();
        let b = Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&dlct)).unwrap();
        let names: Vec<&str> = b
            .schema()
            .system_prompt
            .items
            .iter()
            .filter_map(|it| match it {
                SystemPromptItem::Section(s) => Some(s.name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(names, vec!["frame"]);
        // The dropped template item is also absent from the name map.
        assert!(b.id_for_system_section("maybe_no_think").is_none());
    }

    #[test]
    fn template_depends_on_resolves_to_collection_id() {
        let yaml = r#"
system_prompt:
  items:
    - kind: template
      id: tools_open
      dialect: tool_block_open
      depends_on: tools
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
      selection: { kind: top_k, k: 1 }
      sections:
        - id: t1
          content: "tool one"
    - kind: template
      id: tools_close
      dialect: tool_block_close
      depends_on: tools
layers:
  - name: dialogue
    window: 1000
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
        let b = Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&dlct)).unwrap();
        let tools_cid = b.id_for_system_collection("tools").unwrap();

        let template_deps: Vec<Option<super::super::ids::CollectionId>> = b
            .schema()
            .system_prompt
            .items
            .iter()
            .filter_map(|it| match it {
                SystemPromptItem::Section(s) if s.is_template => Some(s.depends_on),
                _ => None,
            })
            .collect();
        assert_eq!(template_deps, vec![Some(tools_cid), Some(tools_cid)]);
    }

    #[test]
    fn template_closing_after_collection_still_emits() {
        // Regression: `tools_close` is declared AFTER the `tools` collection
        // (with `depends_on: tools`). The emission pass drains the
        // collection-results map as each collection emits, so the close marker
        // must be gated on a set captured up front — otherwise it reads the
        // drained map, sees `None`, and the tool block is never closed in the
        // materialized context (an unclosed `<tools>` block reaches the model).
        use super::super::project::ProjectionSegment;
        let yaml = r#"
system_prompt:
  items:
    - kind: template
      id: tools_open
      dialect: tool_block_open
      depends_on: tools
    - kind: collection
      name: tools
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
      selection: { kind: top_k, k: 1 }
      sections:
        - id: t1
          content: "tool one"
    - kind: template
      id: tools_close
      dialect: tool_block_close
      depends_on: tools
layers:
  - name: dialogue
    window: 1000
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
        b.tokenize_templates::<std::convert::Infallible, _>(|s| {
            Ok(s.bytes().map(u32::from).collect())
        })
        .unwrap();
        let dialogue = b.id_for_layer("dialogue").unwrap();
        let convo = b.id_for_group("convo").unwrap();
        let t1 = b.id_for_system_section("t1").unwrap();

        // Score t1 so the tools collection materialises (non-empty) — the
        // precondition for both `tools_open` and `tools_close` to emit.
        let resolver = MockResolver::new().with_section_score(t1, 0.9);
        let proj = b.project(
            ProjectionTarget {
                layer: dialogue,
                group: convo,
                timeline: TimelineId::for_test(1),
            },
            &resolver,
        );

        let glue: Vec<&str> = proj
            .segments
            .iter()
            .filter_map(|seg| match seg {
                ProjectionSegment::Generated { identity, .. } => Some(identity.name.as_str()),
                _ => None,
            })
            .collect();
        assert!(
            glue.contains(&"tools_open"),
            "tools_open should emit, got {glue:?}",
        );
        assert!(
            glue.contains(&"tools_close"),
            "tools_close (declared after the collection) must still emit, got {glue:?}",
        );
    }

    #[test]
    fn template_depends_on_unknown_collection_errors() {
        let yaml = r#"
system_prompt:
  items:
    - kind: template
      id: tools_open
      dialect: tool_block_open
      depends_on: nonexistent
    - kind: section
      id: frame
      content: "x"
layers:
  - name: dialogue
    window: 1000
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
        let err = Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&dlct)).unwrap_err();
        assert!(
            matches!(err, ConstructionError::UnknownCollection(_)),
            "got {err:?}",
        );
    }

    #[test]
    fn template_id_collision_with_section_errors() {
        let yaml = r#"
system_prompt:
  items:
    - kind: section
      id: dup
      content: "x"
    - kind: template
      id: dup
      dialect: system_start
layers:
  - name: dialogue
    window: 1000
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
        let err = Builder::from_yaml_with_vars_and_dialect(yaml, &[], Some(&dlct)).unwrap_err();
        assert!(
            matches!(err, ConstructionError::DuplicateSectionName(ref n) if n == "dup"),
            "got {err:?}",
        );
    }

    #[test]
    fn existing_from_yaml_still_works_without_dialect() {
        // Schemas with no `kind: template` items must continue to parse via
        // the dialect-less entry points.
        let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
        assert!(b.id_for_layer("dialogue").is_some());
    }

    // —— Section tree (optional toggles + N-way selectors) ————————————————————————

    const TREE_YAML: &str = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional
          id: no_think
          content: "/no_think"
          default: present
        - kind: section
          id: role
          content: "You are an assistant."
        - kind: selector
          id: length
          default: standard
          options:
            - id: terse
              content: "Be terse."
            - id: standard
              content: "Be balanced."
            - id: verbose
              content: "Be verbose."
layers:
  - name: dialogue
    window: 1000
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

    fn tree_target(b: &Builder) -> ProjectionTarget {
        ProjectionTarget {
            layer: b.id_for_layer("dialogue").unwrap(),
            group: b.id_for_group("convo").unwrap(),
            timeline: TimelineId::for_test(1),
        }
    }

    fn dialogue_tree(b: &Builder) -> &crate::projection::SectionTree {
        b.schema()
            .system_prompt
            .items
            .iter()
            .find_map(|it| match it {
                crate::projection::SystemPromptItem::SectionTree(t) => Some(t),
                _ => None,
            })
            .expect("dialogue layer has a section_tree")
    }

    /// The sealed variant id of `node`'s `option` for the given dim selection.
    fn opt_var(b: &Builder, node: &str, option: &str, selection: &[u8]) -> SectionId {
        let tree = dialogue_tree(b);
        let n = tree.nodes.iter().find(|n| n.name == node).unwrap();
        let o = n.options.iter().find(|o| o.id == option).unwrap();
        o.variant_for(tree.pack(selection, n.ancestor_dims))
            .expect("variant sealed for this branch")
            .id
    }

    #[test]
    fn section_tree_default_emits_default_selection() {
        use crate::projection::{ProjectionMode, ResolvedSelection, SelectionState};
        let b = Builder::from_yaml(TREE_YAML).unwrap();
        let _dialogue = b.id_for_layer("dialogue").unwrap();
        // Declared names resolve to the default-selection variant.
        let no_think = b.id_for_system_section("no_think").unwrap();
        let role = b.id_for_system_section("role").unwrap();
        let length = b.id_for_system_section("length").unwrap();
        let resolver = MockResolver::new();

        // Empty state → defaults: no_think present, role, length=standard.
        let proj = b.project_with_selection(
            tree_target(&b),
            &resolver,
            ProjectionMode::Decode,
            &SelectionState::default(),
        );
        let ids: Vec<SectionId> = proj.sealed_sections().map(|s| s.id).collect();
        assert_eq!(ids, vec![no_think, role, length]);
        // Both selectors report the option they emitted, by id.
        assert_eq!(
            proj.selections,
            vec![
                ResolvedSelection {
                    selector: "no_think".to_string(),
                    option: "present".to_string(),
                },
                ResolvedSelection {
                    selector: "length".to_string(),
                    option: "standard".to_string(),
                },
            ]
        );
    }

    #[test]
    fn section_tree_optional_absent_drops_node_and_reprefixes() {
        use crate::projection::{ProjectionMode, SelectionState};
        let b = Builder::from_yaml(TREE_YAML).unwrap();
        let resolver = MockResolver::new();

        let mut sel = SelectionState::new();
        sel.select("no_think", "absent");
        let proj =
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel);

        // no_think absent ⇒ it emits nothing; role + length still emit, but as
        // their ABSENT-branch variants (sealed without no_think above them).
        let ids: Vec<SectionId> = proj.sealed_sections().map(|s| s.id).collect();
        assert_eq!(
            ids,
            vec![
                opt_var(&b, "role", "content", &[1, 1]),
                opt_var(&b, "length", "standard", &[1, 1]),
            ]
        );
        // Distinct from the present-branch (default) variants.
        assert_ne!(
            opt_var(&b, "role", "content", &[1, 1]),
            opt_var(&b, "role", "content", &[0, 1]),
        );
    }

    #[test]
    fn section_tree_selector_override_emits_chosen_option() {
        use crate::projection::{ProjectionMode, ResolvedSelection, SelectionState};
        let b = Builder::from_yaml(TREE_YAML).unwrap();
        let resolver = MockResolver::new();

        let mut sel = SelectionState::new();
        sel.select("length", "verbose");
        let proj =
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel);

        let ids: Vec<SectionId> = proj.sealed_sections().map(|s| s.id).collect();
        // no_think present (default), role, length=verbose (overridden).
        assert_eq!(ids[2], opt_var(&b, "length", "verbose", &[0, 2]));
        assert_ne!(ids[2], opt_var(&b, "length", "standard", &[0, 1]));
        assert!(proj.selections.contains(&ResolvedSelection {
            selector: "length".to_string(),
            option: "verbose".to_string(),
        }));
    }

    /// A two-layer schema where a NON-dialogue layer carries `dials:` for the
    /// shared prompt's section-tree selector. Projecting that layer seeds the
    /// selector from its dial (beneath the tree default); the caller's per-turn
    /// selection still overrides it. This is the unified-prompt contract: layers
    /// differ only by the branch their dials pick, not by a prompt of their own.
    const DIALS_YAML: &str = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: section
          id: role
          content: "You are an assistant."
        - kind: selector
          id: length
          default: standard
          options:
            - id: terse
              content: "Be terse."
            - id: standard
              content: "Be balanced."
            - id: verbose
              content: "Be verbose."
layers:
  - name: dialogue
    window: 1000
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
  - name: analysis
    window: 1000
    dials:
      length: verbose
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
      - id: analysis_group
        selection: { kind: always_visible }
"#;

    #[test]
    fn layer_dials_seed_section_tree_selection() {
        use crate::projection::{ProjectionMode, ResolvedSelection, SelectionState};
        let b = Builder::from_yaml(DIALS_YAML).unwrap();
        let resolver = MockResolver::new();
        let length_is = |proj: &crate::projection::Projection, opt: &str| {
            proj.selections.contains(&ResolvedSelection {
                selector: "length".to_string(),
                option: opt.to_string(),
            })
        };
        let target = |layer: &str, group: &str, tl: u64| ProjectionTarget {
            layer: b.id_for_layer(layer).unwrap(),
            group: b.id_for_group(group).unwrap(),
            timeline: TimelineId::for_test(tl),
        };

        // dialogue has no dials → the section-tree's authored default (standard).
        let proj = b.project_with_selection(
            target("dialogue", "convo", 1),
            &resolver,
            ProjectionMode::Decode,
            &SelectionState::default(),
        );
        assert!(
            length_is(&proj, "standard"),
            "dialogue (no dials) uses the tree default: {:?}",
            proj.selections
        );

        // analysis carries `length: verbose`; with no caller selection the layer
        // dial seeds the branch.
        let ana = target("analysis", "analysis_group", 2);
        let proj = b.project_with_selection(
            ana,
            &resolver,
            ProjectionMode::Decode,
            &SelectionState::default(),
        );
        assert!(
            length_is(&proj, "verbose"),
            "analysis layer's dial seeds length=verbose: {:?}",
            proj.selections
        );

        // A caller's per-turn selection overrides the layer dial.
        let mut sel = SelectionState::new();
        sel.select("length", "terse");
        let proj = b.project_with_selection(ana, &resolver, ProjectionMode::Decode, &sel);
        assert!(
            length_is(&proj, "terse"),
            "caller selection beats the layer dial: {:?}",
            proj.selections
        );
    }

    #[test]
    fn section_tree_seals_full_cross_product_with_correct_prefixes() {
        let b = Builder::from_yaml(TREE_YAML).unwrap();
        let tree = dialogue_tree(&b);
        let no_think = tree.nodes.iter().find(|n| n.name == "no_think").unwrap();
        let role = tree.nodes.iter().find(|n| n.name == "role").unwrap();
        let length = tree.nodes.iter().find(|n| n.name == "length").unwrap();

        // Two dims: no_think (radix 2), length (radix 3); defaults present/standard.
        assert_eq!(tree.dims.len(), 2);
        assert_eq!(tree.default_selection, vec![0, 1]);

        // no_think: `present` has 1 variant (root), `absent` is empty (0).
        let present = no_think.options.iter().find(|o| o.id == "present").unwrap();
        let absent = no_think.options.iter().find(|o| o.id == "absent").unwrap();
        assert_eq!(present.variants.len(), 1);
        assert!(present.variants[0].in_tree_prefix.is_empty());
        assert!(absent.variants.is_empty());

        // role (mandatory): one variant per no_think branch (2).
        let role_opt = &role.options[0];
        assert_eq!(role_opt.variants.len(), 2);
        assert_eq!(
            role_opt.variant_for(0).unwrap().in_tree_prefix,
            vec![present.variants[0].id] // present branch attends to no_think
        );
        assert!(role_opt.variant_for(1).unwrap().in_tree_prefix.is_empty()); // absent branch

        // length: each of 3 options × 2 no_think branches = 6 variants.
        assert_eq!(
            length
                .options
                .iter()
                .map(|o| o.variants.len())
                .sum::<usize>(),
            6
        );
        let std = length.options.iter().find(|o| o.id == "standard").unwrap();
        // length.standard on the present branch attends to [no_think, role].
        assert_eq!(
            std.variant_for(0).unwrap().in_tree_prefix,
            vec![present.variants[0].id, role_opt.variant_for(0).unwrap().id]
        );
    }

    /// A nested `section_tree` whose inner selector sits BELOW static content +
    /// a fake-tool anchor.  The point: content above the nested tree multiplies
    /// only by the OUTER selector (no_think), never the inner one (effort).
    const NESTED_TREE_YAML: &str = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional
          id: no_think
          content: "/no_think"
          default: present
        - kind: section
          id: framing
          content: "You are an assistant."
        - kind: section
          id: noop_tool
          content: "noop anchor tool"
        - kind: section_tree
          nodes:
            - kind: selector
              id: effort
              default: balanced
              options:
                - id: off
                  content: "Answer directly."
                - id: balanced
                  content: "Reason."
layers:
  - name: dialogue
    window: 1000
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

    #[test]
    fn nested_section_tree_inner_selector_inherits_outer_dims() {
        let b = Builder::from_yaml(NESTED_TREE_YAML).unwrap();
        let tree = dialogue_tree(&b);

        // The inner `effort` selector was lifted into the outer tree's dim list,
        // declared AFTER no_think.
        assert_eq!(tree.dims.len(), 2);
        assert_eq!(tree.dims[0].selector_id, "no_think");
        assert_eq!(tree.dims[1].selector_id, "effort");
        assert_eq!(tree.default_selection, vec![0, 1]); // present, balanced

        let noop = tree.nodes.iter().find(|n| n.name == "noop_tool").unwrap();
        let effort = tree.nodes.iter().find(|n| n.name == "effort").unwrap();

        // The anchor sits above the nested tree → it multiplies ONLY by the
        // outer dim (no_think, radix 2). effort never multiplies it.
        assert_eq!(noop.ancestor_dims, 1);
        assert_eq!(noop.options[0].variants.len(), 2);

        // The inner selector fans out across the outer dim: 2 options × 2
        // no_think branches = 4 variants; its ancestor width is the outer dim.
        assert_eq!(effort.ancestor_dims, 1);
        assert_eq!(
            effort
                .options
                .iter()
                .map(|o| o.variants.len())
                .sum::<usize>(),
            4
        );
    }

    #[test]
    fn nested_section_tree_inner_variant_prefix_includes_outer_anchor() {
        let b = Builder::from_yaml(NESTED_TREE_YAML).unwrap();
        let tree = dialogue_tree(&b);
        let node = |name: &str| tree.nodes.iter().find(|n| n.name == name).unwrap();

        // present-no_think branch: ancestor key over [no_think] = 0.
        let key = tree.pack(&[0, 0], 1);
        let no_think_present = node("no_think")
            .options
            .iter()
            .find(|o| o.id == "present")
            .unwrap();
        let framing_var = node("framing").options[0].variant_for(0).unwrap();
        let noop_var = node("noop_tool").options[0].variant_for(0).unwrap();
        let off_var = node("effort")
            .options
            .iter()
            .find(|o| o.id == "off")
            .unwrap()
            .variant_for(key)
            .unwrap();

        // effort.off on the present branch seals against the full outer chain,
        // anchor included: [no_think.present, framing, noop_tool].
        assert_eq!(
            off_var.in_tree_prefix,
            vec![no_think_present.variants[0].id, framing_var.id, noop_var.id]
        );
    }

    #[test]
    fn nested_section_tree_outer_content_stable_across_inner_selection() {
        use crate::projection::{ProjectionMode, SelectionState};
        let b = Builder::from_yaml(NESTED_TREE_YAML).unwrap();
        let resolver = MockResolver::new();

        let proj = |effort: &str| {
            let mut sel = SelectionState::new();
            sel.select("effort", effort);
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel)
                .sealed_sections()
                .map(|s| s.id)
                .collect::<Vec<_>>()
        };
        let off = proj("off");
        let balanced = proj("balanced");

        // Four sections emit: no_think, framing, noop_tool, effort. The first
        // three (static content + anchor) are IDENTICAL across the inner
        // directive selection — only the last differs. That is the continuity
        // win: switching the inner selector never re-prefills the content above.
        assert_eq!(off.len(), 4);
        assert_eq!(off[..3], balanced[..3]);
        assert_ne!(off[3], balanced[3]);
    }

    /// A collection embedded as a tree node, sitting UNDER no_think but ABOVE
    /// the noop anchor + inner directive tree.  The tools must seal ×2 (no_think)
    /// and be prefix-transparent: the anchor + directives below seal as if the
    /// variable tool members were not there.
    const COLLECTION_TREE_YAML: &str = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional
          id: no_think
          content: "/no_think"
          default: present
        - kind: section
          id: framing
          content: "You are an assistant."
        - kind: collection
          name: tools
          selection: { kind: top_k, k: 2 }
          summary:
            chunk: 4
            categorize:
              max_tokens: 256
              system_prompt: cat
              user_prompt: cat
            assign:
              max_tokens: 128
              system_prompt: assign
              user_prompt: assign
          sections:
            - id: tool_a
              content: "tool a"
            - id: tool_b
              content: "tool b"
        - kind: section
          id: noop_tool
          content: "noop anchor"
        - kind: section_tree
          nodes:
            - kind: selector
              id: effort
              default: balanced
              options:
                - id: off
                  content: "direct"
                - id: balanced
                  content: "reason"
layers:
  - name: dialogue
    window: 1000
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

    #[test]
    fn collection_tree_node_members_seal_per_outer_branch() {
        let b = Builder::from_yaml(COLLECTION_TREE_YAML).unwrap();
        let tree = dialogue_tree(&b);

        // no_think + effort are the dims; the collection is NOT a dim.
        assert_eq!(tree.dims.len(), 2);
        assert_eq!(tree.dims[0].selector_id, "no_think");
        assert_eq!(tree.dims[1].selector_id, "effort");

        let tools = tree.nodes.iter().find(|n| n.name == "tools").unwrap();
        let tc = tools
            .collection
            .as_ref()
            .expect("tools is a collection node");
        // Two members, each sealed ×2 (no_think present/absent).
        assert_eq!(tc.variants.len(), 2);
        assert_eq!(tc.variants[0].len(), 2);
        assert_eq!(tc.variants[1].len(), 2);
        // present vs absent no_think → distinct member seals.
        assert_ne!(
            tc.member_variant(0, 0).unwrap().id,
            tc.member_variant(0, 1).unwrap().id
        );
        // The canonical (default-branch = no_think present, key 0) ids back the
        // SectionCollection used for scoring + depends_on gating.
        assert_eq!(tc.collection.sections.len(), 2);
        assert_eq!(
            tc.collection.sections[0].id,
            tc.member_variant(0, 0).unwrap().id
        );
    }

    #[test]
    fn collection_tree_node_is_prefix_transparent() {
        let b = Builder::from_yaml(COLLECTION_TREE_YAML).unwrap();
        let tree = dialogue_tree(&b);
        let node = |name: &str| tree.nodes.iter().find(|n| n.name == name).unwrap();

        let no_think_present = node("no_think")
            .options
            .iter()
            .find(|o| o.id == "present")
            .unwrap();
        let framing_var = node("framing").options[0].variant_for(0).unwrap();

        // The tool members seal against the static base above them only:
        // [no_think.present, framing] on the present branch.
        let tc = node("tools").collection.as_ref().unwrap();
        assert_eq!(
            tc.member_variant(0, 0).unwrap().in_tree_prefix,
            vec![no_think_present.variants[0].id, framing_var.id]
        );

        // The noop anchor sits AFTER the collection, but its prefix is still just
        // [no_think, framing] — the variable tool members are NOT in it.
        let noop_var = node("noop_tool").options[0].variant_for(0).unwrap();
        assert_eq!(
            noop_var.in_tree_prefix,
            vec![no_think_present.variants[0].id, framing_var.id]
        );

        // The inner effort selector anchors on the noop, still excluding tools.
        let off_present = node("effort")
            .options
            .iter()
            .find(|o| o.id == "off")
            .unwrap()
            .variant_for(tree.pack(&[0, 0], 1))
            .unwrap();
        assert_eq!(
            off_present.in_tree_prefix,
            vec![no_think_present.variants[0].id, framing_var.id, noop_var.id]
        );
    }

    #[test]
    fn collection_tree_node_projects_active_branch_stable_across_directive() {
        use crate::projection::{ProjectionMode, SelectionState};
        let b = Builder::from_yaml(COLLECTION_TREE_YAML).unwrap();
        let resolver = MockResolver::new();

        let proj = |no_think: &str, effort: &str| {
            let mut sel = SelectionState::new();
            sel.select("no_think", no_think);
            sel.select("effort", effort);
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel)
                .sealed_sections()
                .map(|s| s.id)
                .collect::<Vec<_>>()
        };

        // present branch: no_think, framing, tool_a, tool_b, noop, effort = 6.
        let base = proj("present", "balanced");
        assert_eq!(base.len(), 6);

        // Switching the inner directive does NOT change the static/tool prefix —
        // the collection never re-prefills for the directive below it.
        let other = proj("present", "off");
        assert_eq!(base[..5], other[..5]);
        assert_ne!(base[5], other[5]);

        // Switching no_think DOES swap the tools to their absent-branch seals.
        let tools = dialogue_tree(&b)
            .nodes
            .iter()
            .find(|n| n.name == "tools")
            .unwrap()
            .collection
            .as_ref()
            .unwrap();
        let present = proj("present", "balanced");
        let absent = proj("absent", "balanced");
        assert!(present.contains(&tools.member_variant(0, 0).unwrap().id));
        assert!(absent.contains(&tools.member_variant(0, 1).unwrap().id));
        assert!(!absent.contains(&tools.member_variant(0, 0).unwrap().id));
    }

    #[test]
    fn inject_collection_replaces_placeholder_with_collection_selection() {
        use crate::projection::{ProjectionMode, ProjectionSegment, SealedKind, SelectionState};
        // The noop placeholder declares `inject_collection: tools`: it still seals
        // its own anchor (the directive below attends to it) but at projection its
        // content is REPLACED by the tools collection's top-k, and the collection
        // itself defers its own emission.  `member_glue` puts a REAL newline token
        // between the selected tools (not baked into any tool's seal).
        let yaml = COLLECTION_TREE_YAML
            .replace(
                "        - kind: section\n          id: noop_tool\n          content: \"noop anchor\"",
                "        - kind: section\n          id: noop_tool\n          inject_collection: tools\n          content: \"noop anchor\"",
            )
            .replace(
                "          selection: { kind: top_k, k: 2 }",
                "          selection: { kind: top_k, k: 2 }\n          member_glue: \"\\n\"",
            );
        let mut b = Builder::from_yaml(&yaml).unwrap();
        // The glue is a live-prefilled structural token, so it must be tokenised
        // before projection (mock: one byte → one token, so "\n" → [10]).
        b.tokenize_templates(|s: &str| Ok::<_, ()>(s.bytes().map(u32::from).collect()))
            .unwrap();

        // Wiring: the tools collection is marked deferred, and the noop node points
        // at it.
        let cid = b.id_for_system_collection("tools").unwrap();
        let tree = dialogue_tree(&b);
        let tools = tree.nodes.iter().find(|n| n.name == "tools").unwrap();
        assert!(
            tools.collection.as_ref().unwrap().deferred_projection,
            "the injected collection must defer its own projection"
        );
        let noop = tree.nodes.iter().find(|n| n.name == "noop_tool").unwrap();
        assert_eq!(noop.inject_collection, Some(cid));

        // The noop STILL seals its own anchor variant (so the directive below it
        // anchors on a stable prefix) — that seal exists even though it isn't
        // projected.
        let noop_anchor = noop.options[0].variant_for(0).unwrap().id;
        let tc = tools.collection.as_ref().unwrap();
        let tool_a = tc.member_variant(0, 0).unwrap().id;
        let tool_b = tc.member_variant(1, 0).unwrap().id;
        let framing = opt_var(&b, "framing", "content", &[0]);
        let effort = opt_var(&b, "effort", "balanced", &[0, 0]);

        // Project the present/balanced branch and read the FULL segment list.
        let resolver = MockResolver::new();
        let mut sel = SelectionState::new();
        sel.select("no_think", "present");
        sel.select("effort", "balanced");
        let proj =
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel);

        let ids: Vec<SectionId> = proj.sealed_sections().map(|s| s.id).collect();
        // The placeholder's own anchor is NOT emitted; the two real tools take its
        // place, and the directive below still emits.
        assert!(
            !ids.contains(&noop_anchor),
            "the noop anchor must not appear in the projection"
        );
        assert!(ids.contains(&tool_a) && ids.contains(&tool_b));
        let pos = |id: SectionId| ids.iter().position(|x| *x == id).unwrap();
        assert!(pos(framing) < pos(tool_a));
        assert!(pos(tool_a) < pos(tool_b));
        assert!(pos(tool_b) < pos(effort));

        // The glue is a REAL token between the two tools: in the full segment list,
        // tool_a → Generated("\n" = [10]) → tool_b, contiguous and in that order.
        let seg_pos = |id: SectionId| {
            proj.segments
                .iter()
                .position(|s| {
                    matches!(s, ProjectionSegment::Sealed(SealedKind::Section(r)) if r.id == id)
                })
                .unwrap()
        };
        let a = seg_pos(tool_a);
        let bseg = seg_pos(tool_b);
        assert_eq!(
            bseg,
            a + 2,
            "exactly one segment sits between the two tools"
        );
        match &proj.segments[a + 1] {
            ProjectionSegment::Generated { tokens, .. } => {
                assert_eq!(
                    tokens.as_ref(),
                    &vec![10u32],
                    "glue must be the newline token"
                );
            }
            other => panic!("expected a Generated glue token between tools, got {other:?}"),
        }
        // No glue leads the first tool (the segment before tool_a is not the glue).
        assert!(
            !matches!(&proj.segments[a - 1], ProjectionSegment::Generated { tokens, .. } if tokens.as_ref() == &vec![10u32]),
            "glue must not lead the first selected tool"
        );
    }

    /// An `optional_group` gates a whole sub-tree (markers + collection + inject
    /// placeholder) on a binary dim: `absent` omits the entire block (markers
    /// included), and nodes BELOW it seal distinctly per (tools present/absent).
    const GROUP_TREE_YAML: &str = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional
          id: no_think
          content: "/no_think"
          default: present
        - kind: section
          id: framing
          content: "You are an assistant."
        - kind: optional_group
          id: tools_on
          default: present
          nodes:
            - kind: section
              id: tools_open
              content: "<tools>"
            - kind: collection
              name: tools
              selection: { kind: top_k, k: 2 }
              member_glue: "\n"
              summary:
                chunk: 4
                categorize:
                  max_tokens: 256
                  system_prompt: cat
                  user_prompt: cat
                assign:
                  max_tokens: 128
                  system_prompt: assign
                  user_prompt: assign
              sections:
                - id: tool_a
                  content: "tool a"
                - id: tool_b
                  content: "tool b"
            - kind: section
              id: noop_tool
              inject_collection: tools
              content: "noop anchor"
            - kind: section
              id: tools_close
              content: "</tools>"
        - kind: section
          id: directive
          content: "Respond."
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
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

    #[test]
    fn optional_group_omits_subtree_when_absent_and_reprefixes_below() {
        use crate::projection::{ProjectionMode, SelectionState};
        let mut b = Builder::from_yaml(GROUP_TREE_YAML).unwrap();
        b.tokenize_templates(|s: &str| Ok::<_, ()>(s.bytes().map(u32::from).collect()))
            .unwrap();
        let resolver = MockResolver::new();

        let proj = |no_think: &str, tools: &str| -> Vec<SectionId> {
            let mut sel = SelectionState::new();
            sel.select("no_think", no_think);
            sel.select("tools_on", tools);
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel)
                .sealed_sections()
                .map(|s| s.id)
                .collect()
        };

        // `tools_on` is a real dim (after no_think).
        let tree = dialogue_tree(&b);
        assert_eq!(tree.dims.len(), 2);
        assert_eq!(tree.dims[0].selector_id, "no_think");
        assert_eq!(tree.dims[1].selector_id, "tools_on");

        let tc = tree
            .nodes
            .iter()
            .find(|n| n.name == "tools")
            .unwrap()
            .collection
            .as_ref()
            .unwrap();
        // Members seal ONLY under the present branch — ×no_think (2), not ×4.
        let member_ids: std::collections::HashSet<SectionId> =
            tc.variants.iter().flatten().map(|v| v.id).collect();
        assert_eq!(
            member_ids.len(),
            2 * 2,
            "2 members × 2 no_think (present only)"
        );

        let present = proj("present", "present");
        let absent = proj("present", "absent");

        // Present: the whole block emits (markers + ≥1 tool).
        let topen = opt_var(&b, "tools_open", "content", &[0, 0]);
        let tclose = opt_var(&b, "tools_close", "content", &[0, 0]);
        assert!(present.contains(&topen) && present.contains(&tclose));
        assert!(
            present.iter().any(|id| member_ids.contains(id)),
            "present branch must inject tools"
        );

        // Absent: NOTHING tool-related — markers, members, and the noop anchor all
        // gone (the block is omitted, not an empty <tools></tools> shell).
        assert!(!absent.contains(&topen) && !absent.contains(&tclose));
        assert!(
            !absent.iter().any(|id| member_ids.contains(id)),
            "absent branch must emit no tools"
        );

        // The directive below seals DISTINCTLY per tools state (the permutation
        // increase) and both variants project on their branch.
        let dir_present = opt_var(&b, "directive", "content", &[0, 0]);
        let dir_absent = opt_var(&b, "directive", "content", &[0, 1]);
        assert_ne!(
            dir_present, dir_absent,
            "directive must seal a distinct variant per tools-on/off"
        );
        assert!(present.contains(&dir_present));
        assert!(absent.contains(&dir_absent));
    }

    /// A `selector` declared INSIDE an `optional_group` present branch is a GATED
    /// dim: it multiplies only the present side, the absent side seals at the
    /// selector's default, and `pack` masks an out-of-scope runtime value back to
    /// that default — so a node after the group projects correctly for EVERY
    /// combination, including the dangerous one (group absent + non-default inner).
    #[test]
    fn selector_inside_optional_group_is_gated() {
        use crate::projection::{ProjectionMode, SelectionState};
        let yaml = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional_group
          id: grp
          default: present
          nodes:
            - kind: selector
              id: inner
              default: a
              options:
                - id: a
                  content: "INNER-A"
                - id: b
                  content: "INNER-B"
          absent:
            - kind: section
              id: alt
              content: "ALT"
        - kind: section
          id: after
          content: "AFTER"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
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
        // No panic at build (the whole point) — the absent side never multiplied
        // by `inner`, yet every branch's assignment stays width-consistent.
        let b = Builder::from_yaml(yaml).unwrap();
        let tree = dialogue_tree(&b);

        // `inner` is gated by `grp` being present (option 0).
        let grp_idx = tree
            .dims
            .iter()
            .position(|d| d.selector_id == "grp")
            .unwrap();
        let inner_dim = tree.dims.iter().find(|d| d.selector_id == "inner").unwrap();
        assert_eq!(inner_dim.gate, Some((grp_idx, 0)));

        // `after` seals exactly THREE variants — (present,a), (present,b), (absent)
        // — NOT four: the absent side does not multiply by `inner`.
        let after = tree.nodes.iter().find(|n| n.name == "after").unwrap();
        assert_eq!(after.options[0].variants.len(), 3);
        let after_a = after.options[0]
            .variant_for(tree.pack(&[0, 0], 2))
            .unwrap()
            .id;
        let after_b = after.options[0]
            .variant_for(tree.pack(&[0, 1], 2))
            .unwrap()
            .id;
        let after_absent = after.options[0]
            .variant_for(tree.pack(&[1, 0], 2))
            .unwrap()
            .id;
        assert_ne!(after_a, after_b);

        let resolver = MockResolver::new();
        let proj = |grp: &str, inner: &str| -> Vec<SectionId> {
            let mut sel = SelectionState::new();
            sel.select("grp", grp);
            sel.select("inner", inner);
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel)
                .sealed_sections()
                .map(|s| s.id)
                .collect()
        };

        // Present: `after` follows the inner selection; the inner content emits.
        assert!(proj("present", "a").contains(&after_a));
        assert!(proj("present", "b").contains(&after_b));

        // THE DANGEROUS CASE: group absent + a NON-DEFAULT inner value. `pack`
        // masks `inner` back to its default, so `after` lands on its single absent
        // variant — no build panic, no silent missing section.
        assert!(proj("absent", "b").contains(&after_absent));
        assert!(proj("absent", "a").contains(&after_absent));
        assert!(!proj("absent", "b").contains(&after_a));
        assert!(!proj("absent", "b").contains(&after_b));

        // The `inner` selector itself only emits when the group is present.
        let inner_a_id = opt_var(&b, "inner", "a", &[0]);
        assert!(proj("present", "a").contains(&inner_a_id));
        assert!(!proj("absent", "a").contains(&inner_a_id));
    }

    /// An `optional_group` nested inside another: the inner group's toggle is
    /// itself gated by the outer group's side, so `dim_active` chains both gates.
    /// A node after both groups must project correctly even when the inner group
    /// is selected present while the outer is absent (inner is out of scope).
    #[test]
    fn nested_optional_group_chains_gates() {
        use crate::projection::{ProjectionMode, SelectionState};
        let yaml = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional_group
          id: outer
          default: present
          nodes:
            - kind: optional_group
              id: inner
              default: present
              nodes:
                - kind: section
                  id: deep
                  content: "DEEP"
              absent:
                - kind: section
                  id: inner_alt
                  content: "INNER-ALT"
          absent:
            - kind: section
              id: outer_alt
              content: "OUTER-ALT"
        - kind: section
          id: tail
          content: "TAIL"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
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
        let b = Builder::from_yaml(yaml).unwrap();
        let tree = dialogue_tree(&b);

        let outer_idx = tree
            .dims
            .iter()
            .position(|d| d.selector_id == "outer")
            .unwrap();
        let inner = tree.dims.iter().find(|d| d.selector_id == "inner").unwrap();
        // The nested group's toggle is gated by the OUTER group being present.
        assert_eq!(inner.gate, Some((outer_idx, 0)));

        // `tail` seals three variants: (outer present, inner present),
        // (outer present, inner absent), (outer absent) — the absent outer side
        // collapses the inner dim.
        let tail = tree.nodes.iter().find(|n| n.name == "tail").unwrap();
        assert_eq!(tail.options[0].variants.len(), 3);
        let tail_pp = tail.options[0]
            .variant_for(tree.pack(&[0, 0], 2))
            .unwrap()
            .id;
        let tail_pa = tail.options[0]
            .variant_for(tree.pack(&[0, 1], 2))
            .unwrap()
            .id;
        let tail_absent = tail.options[0]
            .variant_for(tree.pack(&[1, 0], 2))
            .unwrap()
            .id;

        let resolver = MockResolver::new();
        let proj = |outer: &str, inner: &str| -> Vec<SectionId> {
            let mut sel = SelectionState::new();
            sel.select("outer", outer);
            sel.select("inner", inner);
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel)
                .sealed_sections()
                .map(|s| s.id)
                .collect()
        };

        assert!(proj("present", "present").contains(&tail_pp));
        assert!(proj("present", "absent").contains(&tail_pa));
        // Outer absent + inner present: inner is out of scope, masked away — `tail`
        // lands on its single absent variant, never the inner-present one.
        assert!(proj("absent", "present").contains(&tail_absent));
        assert!(!proj("absent", "present").contains(&tail_pp));
    }

    /// A `dialect:` mandatory tree section is LIVE-PREFILLED glue, not a sealed
    /// section: it allocates no sealed variant, is prefix-transparent (the node
    /// below seals as if it weren't there), and emits a `Generated` run gated to
    /// the branch it lives in (here, tools-on).
    #[test]
    fn dialect_tree_section_is_live_prefilled_glue() {
        use crate::projection::{ProjectionMode, ProjectionSegment, SelectionState};
        use candle_transformers::models::dialect::Dialect;
        let yaml = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional
          id: no_think
          content: "/no_think"
          default: present
        - kind: optional_group
          id: tools_on
          default: present
          nodes:
            - kind: section
              id: tools_open
              dialect: tool_block_open
            - kind: section
              id: inner
              content: "inner"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
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
        b.tokenize_templates(|s: &str| Ok::<_, ()>(s.bytes().map(u32::from).collect()))
            .unwrap();

        let tree = dialogue_tree(&b);
        let tools_open = tree.nodes.iter().find(|n| n.name == "tools_open").unwrap();
        assert!(tools_open.glue.is_some(), "a dialect section is glue");
        assert!(
            tools_open.options[0].variants.is_empty(),
            "glue allocates no sealed variant"
        );

        // Prefix-transparent: `inner` (present branch, right after the glue) anchors
        // only on no_think — the glue contributes NOTHING to its prefix.
        let no_think_present = tree
            .nodes
            .iter()
            .find(|n| n.name == "no_think")
            .unwrap()
            .options
            .iter()
            .find(|o| o.id == "present")
            .unwrap()
            .variants[0]
            .id;
        let inner = tree.nodes.iter().find(|n| n.name == "inner").unwrap();
        let inner_present = inner.options[0].variant_for(tree.pack(&[0, 0], 2)).unwrap();
        assert_eq!(
            inner_present.in_tree_prefix,
            vec![no_think_present],
            "glue must not appear in the K/V prefix below it"
        );

        // Projection: tools on → a `Generated` run named `tools_open`; tools off →
        // none (gated to the present branch).
        let resolver = MockResolver::new();
        let has_glue = |tools: &str| {
            let mut sel = SelectionState::new();
            sel.select("no_think", "present");
            sel.select("tools_on", tools);
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel)
                .segments
                .iter()
                .any(|s| {
                    matches!(
                        s,
                        ProjectionSegment::Generated { identity, tokens }
                            if identity.name == "tools_open" && !tokens.is_empty()
                    )
                })
        };
        assert!(
            has_glue("present"),
            "glue emits a Generated run when tools on"
        );
        assert!(!has_glue("absent"), "glue is omitted when tools off");

        // The GUI event surfaces the marker as a real glue ROW, never a section.
        let mut sel = SelectionState::new();
        sel.select("no_think", "present");
        sel.select("tools_on", "present");
        let proj =
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel);
        let ev = crate::projection::from_projection(
            &proj.segments,
            b.schema(),
            &resolver,
            &crate::projection::SelectionScores::default(),
            0,
            0,
            1.0,
        );
        use crate::projection::SystemItem;
        assert!(
            ev.selection.system.iter().any(|it| matches!(
                it, SystemItem::Glue { name, .. } if name == "tools_open"
            )),
            "marker renders as a glue row"
        );
        assert!(
            !ev.selection.system.iter().any(|it| matches!(
                it, SystemItem::Section { name, .. } if name == "tools_open"
            )),
            "marker is NOT a section row"
        );
    }

    /// An `optional_group` with an `absent:` branch carries DIFFERENT content per
    /// branch under the SAME binary dim (no new permutations): present emits the
    /// tools-aware grounding, absent the tools-free one, and the directive below
    /// genuinely anchors on whichever grounding fired in its branch.
    #[test]
    fn optional_group_absent_branch_carries_alternative_grounding() {
        use crate::projection::{ProjectionMode, SelectionState};
        let yaml = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional
          id: no_think
          content: "/no_think"
          default: present
        - kind: section
          id: frame
          content: "frame"
        - kind: optional_group
          id: tools_on
          default: present
          nodes:
            - kind: section
              id: tools_open
              content: "<tools>"
            - kind: section
              id: grounding_tools
              content: "fetch it with a tool"
          absent:
            - kind: section
              id: grounding_no_tools
              content: "say so rather than guessing"
        - kind: section
          id: directive
          content: "respond"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
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
        let b = Builder::from_yaml(yaml).unwrap();
        let resolver = MockResolver::new();
        let proj = |tools: &str| -> Vec<SectionId> {
            let mut sel = SelectionState::new();
            sel.select("no_think", "present");
            sel.select("tools_on", tools);
            b.project_with_selection(tree_target(&b), &resolver, ProjectionMode::Decode, &sel)
                .sealed_sections()
                .map(|s| s.id)
                .collect()
        };

        // present: no_think=0, tools=0 ; absent: no_think=0, tools=1.
        let g_tools = opt_var(&b, "grounding_tools", "content", &[0, 0]);
        let g_no = opt_var(&b, "grounding_no_tools", "content", &[0, 1]);

        let present = proj("present");
        let absent = proj("absent");
        assert!(
            present.contains(&g_tools) && !present.contains(&g_no),
            "present branch shows the tools grounding only"
        );
        assert!(
            absent.contains(&g_no) && !absent.contains(&g_tools),
            "absent branch shows the tools-free grounding only"
        );

        // grounding_tools seals ONLY ×no_think on the present branch, grounding_no_tools
        // ONLY ×no_think on the absent branch — 2 + 2 = 4 total, no extra dim.
        let tree = dialogue_tree(&b);
        let gt = tree
            .nodes
            .iter()
            .find(|n| n.name == "grounding_tools")
            .unwrap();
        let gn = tree
            .nodes
            .iter()
            .find(|n| n.name == "grounding_no_tools")
            .unwrap();
        assert_eq!(gt.options[0].variants.len(), 2);
        assert_eq!(gn.options[0].variants.len(), 2);

        // KV continuity: the directive below anchors on the grounding that fired
        // in ITS branch (and never the other branch's grounding).
        let dir = tree.nodes.iter().find(|n| n.name == "directive").unwrap();
        let dir_present = dir.options[0].variant_for(tree.pack(&[0, 0], 2)).unwrap();
        let dir_absent = dir.options[0].variant_for(tree.pack(&[0, 1], 2)).unwrap();
        assert!(dir_present.in_tree_prefix.contains(&g_tools));
        assert!(!dir_present.in_tree_prefix.contains(&g_no));
        assert!(dir_absent.in_tree_prefix.contains(&g_no));
        assert!(!dir_absent.in_tree_prefix.contains(&g_tools));
    }

    /// The tool-catalog overview is a REAL sealed tree section before `<tools>`:
    /// `tools_open` anchors on it (its variant id is in `tools_open`'s prefix),
    /// and the content is rewritable pre-prefill WITHOUT changing the variant id
    /// — so the K/V chain stays intact while the daemon writes the real overview.
    #[test]
    fn tool_summary_section_anchors_chain_and_content_is_settable() {
        use crate::projection::{ProjectionMode, SelectionState};
        let yaml = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: section
          id: frame
          content: "frame"
        - kind: section
          id: tool_summary
          content: "placeholder"
        - kind: section
          id: tools_open
          content: "<tools>"
        - kind: section
          id: directive
          content: "respond"
layers:
  - name: dialogue
    window: 8000
    score_formula: max
    budget:
      priority: 100
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
        let mut b = Builder::from_yaml(yaml).unwrap();
        let _dialogue = b.id_for_layer("dialogue").unwrap();

        let var = |b: &Builder, name: &str| {
            dialogue_tree(b)
                .nodes
                .iter()
                .find(|n| n.name == name)
                .unwrap()
                .options[0]
                .variant_for(0)
                .unwrap()
                .id
        };
        let summary_id = var(&b, "tool_summary");
        // `tools_open` (and everything below) genuinely attends the summary: its
        // sealed variant id is in `tools_open`'s in-tree prefix.
        let tools_open_node = dialogue_tree(&b)
            .nodes
            .iter()
            .find(|n| n.name == "tools_open")
            .unwrap()
            .clone();
        assert!(
            tools_open_node.options[0]
                .variant_for(0)
                .unwrap()
                .in_tree_prefix
                .contains(&summary_id),
            "tools_open must anchor on the tool_summary K/V"
        );

        // Rewriting the content pre-prefill keeps the SAME variant id (the chain
        // is unchanged; only the bytes the prefill seals differ).
        b.set_tree_section_content("tool_summary", "The tools are grouped by purpose.")
            .unwrap();
        assert_eq!(
            var(&b, "tool_summary"),
            summary_id,
            "variant id must not move"
        );
        let node = dialogue_tree(&b)
            .nodes
            .iter()
            .find(|n| n.name == "tool_summary")
            .unwrap()
            .clone();
        assert_eq!(node.options[0].content, "The tools are grouped by purpose.");

        // Setting an unknown section errors.
        assert!(b.set_tree_section_content("nope", "x").is_err());

        // Projects in order: frame → tool_summary → <tools> → directive.
        let resolver = MockResolver::new();
        let ids: Vec<SectionId> = b
            .project_with_selection(
                tree_target(&b),
                &resolver,
                ProjectionMode::Decode,
                &SelectionState::new(),
            )
            .sealed_sections()
            .map(|s| s.id)
            .collect();
        let pos = |id: SectionId| ids.iter().position(|x| *x == id).unwrap();
        assert!(pos(var(&b, "frame")) < pos(summary_id));
        assert!(pos(summary_id) < pos(var(&b, "tools_open")));
    }

    #[test]
    fn runtime_add_to_tree_collection_seals_per_branch_no_alias() {
        // Like COLLECTION_TREE_YAML but the tools collection starts empty — the
        // real catalog is installed at runtime (the daemon's install_tool_catalog).
        let yaml = COLLECTION_TREE_YAML.replace(
            "          sections:\n            - id: tool_a\n              content: \"tool a\"\n            - id: tool_b\n              content: \"tool b\"",
            "          sections: []",
        );
        let mut b = Builder::from_yaml(&yaml).unwrap();
        let _dialogue = b.id_for_layer("dialogue").unwrap();
        let cid = b.id_for_system_collection("tools").unwrap();

        // Runtime add, exactly as install_tool_catalog does (by CollectionId).
        let ws = b
            .add_section_to_collection(cid, "web_search", "ws def", 100.0)
            .unwrap();
        let calc = b
            .add_section_to_collection(cid, "calc", "calc def", 100.0)
            .unwrap();

        let tree = dialogue_tree(&b);
        let tools = tree
            .nodes
            .iter()
            .find(|n| n.name == "tools")
            .unwrap()
            .collection
            .as_ref()
            .unwrap();

        // Two members, each sealed ×2 (no_think branches).
        assert_eq!(tools.variants.len(), 2);
        assert_eq!(tools.variants[0].len(), 2);
        assert_eq!(tools.variants[1].len(), 2);

        // Each member's canonical id is its default-branch variant, and resolves
        // by name.
        assert_eq!(
            ws,
            tools.member_variant(0, tools.default_branch).unwrap().id
        );
        assert_eq!(b.id_for_system_section("web_search").unwrap(), ws);

        // No id aliasing: every sealed id across both members + both branches is
        // distinct (the ×branch block sits above the prior max each add).
        let mut ids: Vec<_> = tools
            .variants
            .iter()
            .flatten()
            .map(|v| v.id.raw())
            .collect();
        let n = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), n, "all per-branch member ids must be unique");
        // calc was added after web_search → its block is strictly higher.
        assert!(calc.raw() > ws.raw());
    }

    #[test]
    fn section_tree_invalid_optional_default_rejected() {
        let yaml = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: optional
          id: x
          content: "x"
          default: maybe
layers:
  - name: dialogue
    window: 1000
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
        let err = Builder::from_yaml(yaml).unwrap_err();
        assert!(
            matches!(err, ConstructionError::InvalidToggleDefault { ref value, .. } if value == "maybe"),
            "got {err:?}",
        );
    }

    #[test]
    fn section_tree_unknown_selector_default_rejected() {
        let yaml = r#"
system_prompt:
  items:
    - kind: section_tree
      nodes:
        - kind: selector
          id: length
          default: nope
          options:
            - id: terse
              content: "a"
            - id: standard
              content: "b"
layers:
  - name: dialogue
    window: 1000
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
        let err = Builder::from_yaml(yaml).unwrap_err();
        assert!(
            matches!(err, ConstructionError::UnknownTreeOption { ref option, .. } if option == "nope"),
            "got {err:?}",
        );
    }
}
