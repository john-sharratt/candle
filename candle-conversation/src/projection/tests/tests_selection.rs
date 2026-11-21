//! Tests for choosing turns by threshold, top-k and recency, and for tagged
//! groups.
//!
//! Turns below a score threshold are filtered, and a layer threshold removes a
//! whole group. Top-k keeps the highest scorers in insertion order, and recent
//! turns are included whatever their score. A tagged group admits only its owner's
//! turns at the depth a reader asks for, and the reader's own group is never
//! scoped.

use super::*;

#[test]
pub(super) fn turns_below_score_threshold_filtered() {
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
pub(super) fn layer_threshold_filters_whole_group() {
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
pub(super) fn top_k_limits_turns() {
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

/// **A tagged group admits only its owner's turns, at the depth the reader
/// asks for.** One group holds every owner's conversations; a reader scoped to
/// one owner must never be shown another's, however well the other's scores —
/// here the two best-scoring turns belong to somebody else and one is untagged.
#[test]
pub(super) fn a_tagged_group_admits_only_its_owners_turns_at_the_readers_depth() {
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
      - id: live
        selection: { kind: conversation, recent: 4 }
  # Declared after the reader's layer and ranked beneath it — how a layer is
  # added to a live schema without moving any id already persisted.
  - name: store
    window: 9000
    rank: -1
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
        # The mock's scores are raw, not on the belief band, so the gate is
        # declared off rather than left to the default band.
        score_threshold: 0.0
"#;
    let mut b = Builder::from_yaml(yaml).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let live = b.id_for_group("live").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    // The reader's own conversation: one untagged turn, as a live one is.
    let own = resolver.append(live);
    let theirs = resolver.append(grp);
    let mine_low = resolver.append(grp);
    let mine_high = resolver.append(grp);
    let untagged = resolver.append(grp);
    let resolver = resolver
        .with_score(grp, theirs, 0.9)
        .with_score(grp, mine_low, 0.2)
        .with_score(grp, mine_high, 0.7)
        .with_score(grp, untagged, 0.8)
        .with_turn_tags(grp, theirs, &["dreams", "dreams:9"])
        .with_turn_tags(grp, mine_low, &["dreams", "dreams:7"])
        .with_turn_tags(grp, mine_high, &["dreams", "dreams:7"]);
    let target = ProjectionTarget {
        layer,
        group: live,
        timeline: TimelineId::for_test(live.raw() as u64),
    };
    let in_group = |b: &Builder, g| -> Vec<u32> {
        b.project(target, &resolver)
            .sealed_turns()
            .filter(|t| t.group() == g)
            .map(|t| t.index().0)
            .collect()
    };
    let picked = |b: &Builder| in_group(b, grp);

    // Untagged, the group is what it always was: the best two of everything.
    assert_eq!(picked(&b), vec![theirs.0, untagged.0]);

    b.set_group_tags("grp", vec!["dreams:7".into()]).unwrap();
    assert_eq!(
        picked(&b),
        vec![mine_low.0, mine_high.0],
        "only owner 7's turns"
    );

    b.set_group_selection("grp", SelectionRule::TopK { k: 1 })
        .unwrap();
    assert_eq!(
        picked(&b),
        vec![mine_high.0],
        "and only as many as asked for"
    );

    // **The reader's own group is never scoped.** It is its own conversation
    // already, and its turns carry no tags — least of all the one being
    // written, which a prefill waits on forever if its own projection drops it.
    b.set_group_tags("live", vec!["dreams:7".into()]).unwrap();
    assert_eq!(
        in_group(&b, live),
        vec![own.0],
        "the target lost its own turn"
    );

    assert!(b.set_group_tags("nowhere", Vec::new()).is_err());
    assert!(b
        .set_group_selection("nowhere", SelectionRule::TopK { k: 1 })
        .is_err());
}

/// **A closed reserved conversation admits no other job's turns**: its one
/// group is scoped to a tag no turn carries, on the reserved kind's ids.
#[test]
pub(super) fn a_closed_reserved_schema_scopes_its_group_to_nothing() {
    let b = Builder::for_closed_reserved(
        "You keep summaries.",
        Reserved::Prose,
        SectionId::new(9_001),
        DecodePriority::Low,
    );
    let layer = &b.schema().layers[0];
    assert_eq!(layer.id, LayerId::reserved(Reserved::Prose));
    let group = &layer.groups[0];
    assert_eq!(
        (group.id, group.name.as_str()),
        (GroupId::reserved(Reserved::Prose), PLAIN_PROMPT_GROUP)
    );
    assert_eq!(group.policy.tags, vec!["group:closed".to_string()]);
    let open = Builder::for_plain_prompt_reserved(
        "You keep summaries.",
        Reserved::Prose,
        SectionId::new(9_001),
        DecodePriority::Low,
    );
    assert!(open.schema().layers[0].groups[0].policy.tags.is_empty());
}

#[test]
pub(super) fn single_selection_one_turn() {
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
pub(super) fn always_visible_selects_all_above_threshold() {
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

#[test]
pub(super) fn conversation_recent_turns_always_included() {
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
