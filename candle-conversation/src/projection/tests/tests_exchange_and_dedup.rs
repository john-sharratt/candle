//! Tests for keeping a tool round trip whole, and for dropping summaries that
//! repeat selected content.
//!
//! A tool call and the response that answers it form one exchange, so one
//! selected turn brings the whole chain, and a chain too large for the budget is
//! dropped whole rather than trimmed to its cheap half. Dedup drops a summary
//! whose covered turns are selected and keeps it otherwise. Score-density picks
//! are emitted in the order the selector gives, so a summary stays above the later
//! turns it covers.

use super::*;

/// The YAML for the exchange tests: one `top_k` group over a folder-chain
/// shaped timeline. `k` and the layer `window` are substituted per test.
pub(super) fn exchange_yaml(k: usize, window: usize) -> String {
    format!(
        r#"
system_prompt:
  sections:
    - id: s1
      content: "X"
layers:
  - name: layer
    window: {window}
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
        selection: {{ kind: top_k, k: {k} }}
"#
    )
}

/// A `repo_map` folder chain: `request → <tool_call>` (0), `<tool_response> →
/// <tool_call>` (1), `<tool_response> → summary` (2), coupled 0→1→2, plus one
/// uncoupled decoy turn.
///
/// One seat must buy the WHOLE chain. Seating turn 0 alone is what put a
/// dangling `<tool_call>` in front of the model — it read the projection as
/// dialogue and imitated the call, answering a question by emitting
/// `<tool_call>{"name":"file_list",…}` for a folder nobody had asked about.
#[test]
pub(super) fn one_top_k_seat_buys_the_whole_tool_round_trip_not_just_the_call() {
    let b = Builder::from_yaml(&exchange_yaml(1, 9000)).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    let call = resolver.append(grp);
    let mid = resolver.append(grp);
    let answer = resolver.append(grp);
    let decoy = resolver.append(grp);
    let resolver = resolver
        .with_coupling(grp, call)
        .with_coupling(grp, mid)
        .with_score(grp, call, 0.9)
        .with_score(grp, mid, 0.9)
        .with_score(grp, answer, 0.9)
        .with_score(grp, decoy, 0.5);

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let emitted: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();
    assert_eq!(emitted, vec![0, 1, 2], "one seat = one whole exchange");
}

/// The hit can land on any member. A scan that matches the ANSWER must still
/// bring the call that provoked it — the whole run travels together, in order.
#[test]
pub(super) fn a_hit_on_the_answer_half_brings_the_call_that_provoked_it() {
    let b = Builder::from_yaml(&exchange_yaml(1, 9000)).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    let call = resolver.append(grp);
    let mid = resolver.append(grp);
    let answer = resolver.append(grp);
    let decoy = resolver.append(grp);
    let resolver = resolver
        .with_coupling(grp, call)
        .with_coupling(grp, mid)
        // Only the closing summary scores; the generic call framing does not.
        .with_score(grp, call, 0.0)
        .with_score(grp, mid, 0.0)
        .with_score(grp, answer, 0.9)
        .with_score(grp, decoy, 0.5);

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let emitted: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();
    assert_eq!(emitted, vec![0, 1, 2]);
}

/// The budget prices and trims whole runs. A chain that does not fit is dropped
/// entirely — never shaved down to its cheap half, which is exactly what a
/// per-turn shave keeps: the `<tool_call>` turn is ~50 tokens while the
/// `<tool_response>` halves carry the file listing and the excerpt, so the
/// dangling call is the *last* thing a token trim would drop.
#[test]
pub(super) fn a_budget_too_small_for_a_round_trip_drops_it_whole() {
    // The chain costs 40+400+400; the higher-scored decoy costs 40. Only one of
    // them fits, and the chain is the one that has to go.
    let b = Builder::from_yaml(&exchange_yaml(2, 200)).unwrap();
    let layer = b.id_for_layer("layer").unwrap();
    let grp = b.id_for_group("grp").unwrap();

    let mut resolver = MockResolver::new();
    let call = resolver.append(grp);
    let mid = resolver.append(grp);
    let answer = resolver.append(grp);
    let decoy = resolver.append(grp);
    let resolver = resolver
        .with_coupling(grp, call)
        .with_coupling(grp, mid)
        .with_tokens(grp, call, 40)
        .with_tokens(grp, mid, 400)
        .with_tokens(grp, answer, 400)
        .with_tokens(grp, decoy, 40)
        .with_score(grp, call, 0.5)
        .with_score(grp, mid, 0.5)
        .with_score(grp, answer, 0.5)
        .with_score(grp, decoy, 0.9);

    let proj = b.project(
        ProjectionTarget {
            layer,
            group: grp,
            timeline: TimelineId::for_test(1),
        },
        &resolver,
    );
    let emitted: Vec<u32> = proj.sealed_turns().map(|t| t.index().0).collect();
    assert_eq!(
        emitted,
        vec![3],
        "the chain's cheap call half must not survive on its own",
    );
}

/// Minimal single-layer schema whose one group ranks by raw provenance score
/// (`top_k`), so summary forest nodes compete head-to-head with raw turns.
pub(super) const TOPK_YAML: &str = r#"
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
pub(super) fn rule_based_dedup_drops_summary_when_its_content_is_selected() {
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
pub(super) fn rule_based_dedup_keeps_summary_when_its_content_is_not_selected() {
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
pub(super) fn rule_based_dedup_prefers_the_finer_summary_over_its_ancestor() {
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
pub(super) fn score_density_summary_emitted_before_later_raw_turn() {
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
