//! Tests for dialogue faces, and for ingest conversations reading their own turns.
//!
//! A dialogue holds a unit's opening by its face, while an ingest projection holds
//! its own opening. An ingest conversation sees its own turns while it generates,
//! though they carry no belief yet, and the window does not trim them. Inherited
//! turns are named by why they are there, and an ingest target reads a layer below
//! it at that layer's own window. Retrieval still gates unscored turns.

use super::*;

// —— Faces ————————————————————————————————————————————————————————————————————

/// A dialogue over a ground layer whose `facts` turn 0 has a face, and the
/// face's key.
pub(super) fn faced_facts(ingest_self: bool) -> (Builder, ProjectionTarget, MockResolver, TurnKey) {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let facts = b.id_for_group("facts").unwrap();
    let target = ProjectionTarget {
        layer: b.id_for_layer("dialogue").unwrap(),
        group: b.id_for_group("conversation").unwrap(),
        timeline: TimelineId::for_test(1),
    };
    let face = TurnKey::new(TimelineId::for_test(99), TurnIndex(0));
    let mut resolver = MockResolver::new();
    let opening = resolver.append(facts);
    resolver.append(facts);
    let resolver = resolver.with_face(facts, opening, face);
    // An ingest target reads a layer below it only when it compounds on it.
    let resolver = match ingest_self {
        true => resolver.as_ingest_self().compounding_on(facts),
        false => resolver,
    };
    (b, target, resolver, face)
}

/// **A dialogue holds a unit's opening by its face**: the face's turn is
/// emitted where the opening was selected, and the unit's other turns as they
/// are.
#[test]
pub(super) fn a_dialogue_holds_a_units_opening_by_its_face() {
    let (b, target, resolver, face) = faced_facts(false);
    let facts = MockResolver::timeline_of(b.id_for_group("facts").unwrap());
    let proj = b.project(target, &resolver);
    let placed: Vec<Option<TurnKey>> = proj.sealed_turns().map(|t| t.key()).collect();
    assert!(placed.contains(&Some(face)), "{placed:?}");
    assert!(
        !placed.contains(&Some(TurnKey::new(facts, TurnIndex(0)))),
        "{placed:?}"
    );
    assert!(
        placed.contains(&Some(TurnKey::new(facts, TurnIndex(1)))),
        "{placed:?}"
    );
}

/// **A unit being read projects its own opening** — the request it is
/// answering — never its face.
#[test]
pub(super) fn an_ingest_projection_holds_its_own_opening() {
    let (b, target, resolver, face) = faced_facts(true);
    let facts = MockResolver::timeline_of(b.id_for_group("facts").unwrap());
    let proj = b.project(target, &resolver);
    let placed: Vec<Option<TurnKey>> = proj.sealed_turns().map(|t| t.key()).collect();
    assert!(!placed.contains(&Some(face)), "{placed:?}");
    assert!(
        placed.contains(&Some(TurnKey::new(facts, TurnIndex(0)))),
        "{placed:?}"
    );
}

// —— Score threshold ———————————————————————————————————————————————————————————

/// An ingest conversation must see its OWN turns while it generates, even
/// though they carry no wide-Q belief yet and so score zero.
///
/// This is the defect that made every `repo_map` folder summary useless: the
/// request turn was inserted, scored 0, was filtered out by the group's band,
/// and the decode answered "the user hasn't asked a specific question yet" with
/// the request sitting one turn back in its own conversation.
#[test]
pub(super) fn ingest_self_projection_sees_its_own_unscored_turns() {
    let yaml = INGEST_GATED_YAML;
    let b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    let i0 = resolver.append(grp);
    let i1 = resolver.append(grp);
    // Freshly inserted ingest turns: no belief yet, so zero — exactly what the
    // group's threshold would otherwise reject.
    let resolver = resolver
        .with_score(grp, i0, 0.0)
        .with_score(grp, i1, 0.0)
        .as_ingest_self();

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let indices: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();
    assert!(
        indices.contains(&0) && indices.contains(&1),
        "an ingest conversation must project its own turns regardless of score: {indices:?}",
    );
}

/// **The window does not trim an ingest conversation's own turns.** Four
/// zero-scored turns of 4000 tokens each are 16000 against a 9000-token window.
/// Trimmed by score they lost to the tie-break — newest key first — so a file
/// conversation several pages in answered its final turn with its own request
/// and its earlier pages gone, and summarised the priming chain's last document
/// instead. Every one is the conversation itself, and every one is projected.
#[test]
pub(super) fn ingest_self_projection_keeps_its_own_turns_past_the_window() {
    let b = Builder::from_yaml(INGEST_GATED_YAML).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    let turns: Vec<TurnIndex> = (0..4).map(|_| resolver.append(grp)).collect();
    let resolver = turns
        .iter()
        .fold(resolver, |r, &i| {
            r.with_score(grp, i, 0.0).with_tokens(grp, i, 4000)
        })
        .as_ingest_self();

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let indices: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();
    assert_eq!(indices, [0, 1, 2, 3], "every own turn is projected");
}

/// **A conversation's inherited turns are named for why they are there**, not
/// by where they fall against the recency window. A dialogue on this member's
/// seed (two links' turns) and on one other ancestor, with six turns of its own
/// against `recent: 4`: the seed's turns are `Seed`, the other ancestor's
/// `Inherited`, and the window splits the own turns alone — the last four
/// `Recent`, the first two `Historical`. Labelled by position in the combined
/// list, a new conversation's whole seed read `Recent`.
#[test]
pub(super) fn inherited_turns_are_named_seed_or_inherited_and_the_window_splits_own_turns() {
    let b = Builder::from_yaml(SIMPLE_YAML).unwrap();
    let layer = b.id_for_layer("dialogue").unwrap();
    let grp = b.id_for_group("conversation").unwrap();
    let seed_link = TimelineId::for_test(900);
    let ancestor = TimelineId::for_test(901);

    let mut resolver = MockResolver::new();
    for _ in 0..6 {
        resolver.append(grp);
    }
    let resolver = resolver
        .with_inherited(grp, TurnKey::new(seed_link, TurnIndex(0)))
        .with_inherited(grp, TurnKey::new(seed_link, TurnIndex(1)))
        .with_inherited(grp, TurnKey::new(ancestor, TurnIndex(0)))
        .with_seed(seed_link);

    let own = MockResolver::timeline_of(grp);
    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: own,
        },
        &resolver,
    );
    let origin = |timeline, index| {
        proj.selection_origins
            .get(&TurnKey::new(timeline, TurnIndex(index)))
            .copied()
    };
    assert_eq!(origin(seed_link, 0), Some(SelectionOrigin::Seed));
    assert_eq!(origin(seed_link, 1), Some(SelectionOrigin::Seed));
    assert_eq!(origin(ancestor, 0), Some(SelectionOrigin::Inherited));
    let own_origins: Vec<Option<SelectionOrigin>> = (0..6).map(|i| origin(own, i)).collect();
    assert_eq!(
        own_origins,
        [
            Some(SelectionOrigin::Historical),
            Some(SelectionOrigin::Historical),
            Some(SelectionOrigin::Recent),
            Some(SelectionOrigin::Recent),
            Some(SelectionOrigin::Recent),
            Some(SelectionOrigin::Recent),
        ]
    );
}

/// A layer below an ingest target, and the target layer with a window far
/// smaller than its own lineage — an analysis unit continuing a file's whole
/// reading, compounding on the analyses below it.
pub(super) const COMPOUNDING_YAML: &str = r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: lower
    window: 3000
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
      - id: below
        score_threshold: 0.6
        selection: { kind: top_k, k: 4 }
  - name: upper
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
    score_formula: max
    groups:
      - id: own
        score_threshold: 0.6
        selection: { kind: top_k, k: 4 }
"#;

/// **An ingest target compounding on a layer below sees it at that layer's
/// window**, not a slice of its own. The target's own turns — 16000 tokens of
/// lineage — are kept whole and ask the budget for nothing; its window is
/// 1000; the layer below holds two strong 1200-token units, which its own
/// 3000-token window fits. Priced from the target's window, with its own turns
/// taking their share, the units below were dropped for want of room.
#[test]
pub(super) fn an_ingest_target_draws_the_layer_below_at_that_layers_window() {
    let b = Builder::from_yaml(COMPOUNDING_YAML).unwrap();
    let upper = b.id_for_layer("upper").unwrap();
    let own = b.id_for_group("own").unwrap();
    let below = b.id_for_group("below").unwrap();

    let mut resolver = MockResolver::new();
    let own_turns: Vec<TurnIndex> = (0..4).map(|_| resolver.append(own)).collect();
    let b0 = resolver.append(below);
    let b1 = resolver.append(below);
    let resolver = own_turns
        .iter()
        .fold(resolver, |r, &i| {
            r.with_score(own, i, 0.0).with_tokens(own, i, 4000)
        })
        .with_score(below, b0, 0.9)
        .with_tokens(below, b0, 1200)
        .with_score(below, b1, 0.8)
        .with_tokens(below, b1, 1200)
        .as_ingest_self()
        .compounding_on(below);

    let proj = b.project(
        ProjectionTarget {
            layer: upper,
            group: own,
            timeline: MockResolver::timeline_of(own),
        },
        &resolver,
    );
    let drawn: Vec<u32> = proj
        .sealed_turns()
        .filter(|t| t.group() == below)
        .map(|t| t.index().0)
        .collect();
    assert_eq!(drawn, [0, 1], "both units below are drawn: {drawn:?}");
    let own_kept = proj.sealed_turns().filter(|t| t.group() == own).count();
    assert_eq!(own_kept, 4, "the target's own turns are kept whole");
}

/// The same schema and the same zero-scored turns, but the target is NOT an
/// ingest layer — retrieval, where a turn must earn its slot. The band applies
/// exactly as before, so dialogue pulling this content back is untouched.
#[test]
pub(super) fn retrieval_still_gates_unscored_turns() {
    let b = Builder::from_yaml(INGEST_GATED_YAML).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    let i0 = resolver.append(grp);
    let i1 = resolver.append(grp);
    let resolver = resolver.with_score(grp, i0, 0.0).with_score(grp, i1, 0.0);

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let indices: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();
    assert!(
        indices.is_empty(),
        "retrieval must still gate zero-scored turns: {indices:?}",
    );
}

/// Schema shared by the two tests above: a gated group, as both ingest layers
/// declare (`repo_map` min_score 250, `code_reading` score_threshold 100).
pub(super) const INGEST_GATED_YAML: &str = r#"
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
        selection: { kind: top_k, k: 4 }
"#;
