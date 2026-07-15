//! Phase 2 turn-level wide-Q scan: `score_belief_groups` scores a belief-driven
//! turn group's own turns by self-match (identity slot map), so a probe that
//! matches one turn's stored `WideQSig` window scores that turn highest.
//!
//! This is the turn-axis analogue of the tool-catalog belief scan: where a
//! collection scans a tag gallery of past turns → section slots, a turn group's
//! retrieval target IS the turn, so each candidate turn is its own slot.

use candle_conversation::persistence::content_hash::turn_stream_id;
use candle_conversation::projection::{Builder, ProjectionTarget};
use candle_conversation::provenance::{encode_wide_sigs, WideQSig};
use candle_conversation::substrate::{ProjectionScores, TurnPartWrite};
use candle_conversation::turn::Role;

mod common;
use common::open_conversation;

const SCAN_YAML: &str = r#"
layers:
  - name: mem
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
    system_prompt:
      sections:
        - id: frame
          content: "frame"
    groups:
      - id: clusters
        selection: { kind: top_k, k: 2 }
"#;

/// A folded `WideQSig`: 12 heads (3 layer-groups × 4 kv-heads), head_dim 128 →
/// 24 u64 words, all set to `fill`. Mirrors the provenance scan's own fixture so
/// the `z × margin` late-fusion produces a real vote (a sub-group-width signature
/// is gated to zero).
fn sig(fill: u64) -> WideQSig {
    WideQSig {
        n_heads: 12,
        words: vec![fill; 24],
    }
}

#[test]
fn score_belief_groups_self_matches_the_probed_turn() {
    let dir = tempfile::tempdir().unwrap();
    let conv = open_conversation(dir.path());

    let builder = Builder::from_yaml(SCAN_YAML).unwrap();
    let layer = builder.id_for_layer("mem").unwrap();
    let group = builder.id_for_group("clusters").unwrap();

    let timeline = candle_conversation::projection::TimelineId::from_raw(7).expect("timeline id");
    conv.register_timeline(timeline, layer, group);

    // Three turns with distinct sign patterns: turn 1's `0x5555…` is the bitwise
    // complement of turn 0's `0xAAAA…`, so a probe matching turn 1 agrees fully
    // with it and not at all with turn 0.
    let fills = [
        0xAAAA_AAAA_AAAA_AAAAu64, // turn 0
        0x5555_5555_5555_5555u64, // turn 1
        0xFFFF_FFFF_FFFF_FFFFu64, // turn 2
    ];
    for fill in fills {
        let idx = conv
            .record_turn(
                timeline,
                Role::User,
                TurnPartWrite {
                    token_count: 4,
                    tags: vec!["repo_map".to_string()],
                    ..Default::default()
                },
                |seqs| Ok(seqs.to_vec()),
            )
            .expect("record_turn");
        let stream_id = turn_stream_id(timeline.raw(), idx.0);
        conv.persist_wide_q_sigs(stream_id, &encode_wide_sigs(&[sig(fill)]))
            .expect("persist sigs");
    }

    // Probe = turn 1's own window: the self-match must rank turn 1 highest.
    let probe = vec![sig(fills[1])];
    let mut scores = ProjectionScores::new();
    let target = ProjectionTarget {
        layer,
        group,
        timeline,
    };
    let candidates =
        conv.score_belief_groups(&builder.schema().layers[0], target, &probe, &mut scores);

    // The scan reported all three candidate turns for the group.
    assert_eq!(candidates.len(), 1, "one belief-driven group scanned");
    assert_eq!(candidates[0].0, group);
    assert_eq!(candidates[0].1.len(), 3, "all three turns are candidates");

    use candle_conversation::projection::TurnIndex;
    let s0 = scores.turn(timeline, TurnIndex(0));
    let s1 = scores.turn(timeline, TurnIndex(1));
    let s2 = scores.turn(timeline, TurnIndex(2));
    assert!(
        s1 > s0 && s1 > s2,
        "probed turn 1 must score highest (self-match): s0={s0}, s1={s1}, s2={s2}"
    );
}

const TWO_LAYER_YAML: &str = r#"
layers:
  - name: mem
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
    system_prompt:
      sections:
        - id: frame
          content: "frame"
    groups:
      - id: clusters
        selection: { kind: top_k, k: 2 }
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
    system_prompt:
      sections:
        - id: frame2
          content: "frame2"
    groups:
      - id: convo
        selection: { kind: conversation, recent: 4, historical_top_k: 8 }
"#;

/// Regression: `score_beliefs` must score belief-driven turn groups in EVERY
/// layer, not just the target layer. The `clusters` group lives in the `mem`
/// layer while the projection target is the `dialogue` layer; before the fix the
/// scan only covered the target layer, leaving `clusters` on all-zero scores (a
/// degenerate index tie-break instead of relevance).
#[test]
fn score_beliefs_scores_a_group_in_a_non_target_layer() {
    use candle_conversation::projection::TurnIndex;

    let dir = tempfile::tempdir().unwrap();
    let conv = open_conversation(dir.path());
    let builder = Builder::from_yaml(TWO_LAYER_YAML).unwrap();
    let mem_layer = builder.id_for_layer("mem").unwrap();
    let clusters = builder.id_for_group("clusters").unwrap();
    let dialogue_layer = builder.id_for_layer("dialogue").unwrap();
    let convo = builder.id_for_group("convo").unwrap();

    // The clusters group's turns live on their own (mem) timeline.
    let mem_tl = candle_conversation::projection::TimelineId::from_raw(11).unwrap();
    conv.register_timeline(mem_tl, mem_layer, clusters);
    let fills = [
        0xAAAA_AAAA_AAAA_AAAAu64,
        0x5555_5555_5555_5555u64,
        0xFFFF_FFFF_FFFF_FFFFu64,
    ];
    for fill in fills {
        let idx = conv
            .record_turn(
                mem_tl,
                Role::User,
                TurnPartWrite {
                    token_count: 4,
                    ..Default::default()
                },
                |seqs| Ok(seqs.to_vec()),
            )
            .expect("record_turn");
        let sid = turn_stream_id(mem_tl.raw(), idx.0);
        conv.persist_wide_q_sigs(sid, &encode_wide_sigs(&[sig(fill)]))
            .expect("persist sigs");
    }

    // Target is the DIALOGUE layer (a different layer than `clusters`).
    let dlg_tl = candle_conversation::projection::TimelineId::from_raw(22).unwrap();
    conv.register_timeline(dlg_tl, dialogue_layer, convo);
    let target = ProjectionTarget {
        layer: dialogue_layer,
        group: convo,
        timeline: dlg_tl,
    };
    let probe = vec![sig(fills[1])];
    let (scores, cands) = conv.score_beliefs(builder.schema(), target, &probe);

    // The non-target clusters group was scored, and the probed turn wins.
    let s0 = scores.turn(mem_tl, TurnIndex(0));
    let s1 = scores.turn(mem_tl, TurnIndex(1));
    let s2 = scores.turn(mem_tl, TurnIndex(2));
    assert!(
        s1 > s0 && s1 > s2 && s1 > 0.0,
        "non-target clusters group must be scored: s0={s0}, s1={s1}, s2={s2}"
    );
    assert!(
        cands.iter().any(|(gid, _)| *gid == clusters),
        "clusters candidates must be reported for the challenger",
    );
}

#[test]
fn score_belief_groups_ignores_recency_groups_and_empty_probe() {
    let dir = tempfile::tempdir().unwrap();
    let conv = open_conversation(dir.path());
    let builder = Builder::from_yaml(SCAN_YAML).unwrap();
    let layer = builder.id_for_layer("mem").unwrap();
    let group = builder.id_for_group("clusters").unwrap();
    let timeline = candle_conversation::projection::TimelineId::from_raw(9).expect("timeline id");
    conv.register_timeline(timeline, layer, group);

    // An empty probe scores nothing and reports no candidates.
    let mut scores = ProjectionScores::new();
    let target = ProjectionTarget {
        layer,
        group,
        timeline,
    };
    let candidates =
        conv.score_belief_groups(&builder.schema().layers[0], target, &[], &mut scores);
    assert!(candidates.is_empty(), "empty probe → no scan");
}
