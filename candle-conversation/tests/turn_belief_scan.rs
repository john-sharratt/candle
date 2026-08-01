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
system_prompt:
  sections:
    - id: frame
      content: "frame"
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
    let candidates = conv.score_belief_groups(
        &builder.schema().layers[0],
        target,
        &probe,
        &mut scores,
        false,
        None,
    );

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

/// A coupled tool round-trip (a code-read scope) must be scored as ONE exchange:
/// the call turn and its response turn share a single normalized score, so
/// provenance selecting either half brings in the whole pair. Turn 0 is coupled
/// to turn 1 (its response); a probe matching turn 0 must lift BOTH to the same
/// score, above the uncoupled turn 2.
#[test]
fn score_belief_groups_scores_a_coupled_pair_as_one_exchange() {
    use candle_conversation::projection::TurnIndex;

    let dir = tempfile::tempdir().unwrap();
    let conv = open_conversation(dir.path());

    let builder = Builder::from_yaml(SCAN_YAML).unwrap();
    let layer = builder.id_for_layer("mem").unwrap();
    let group = builder.id_for_group("clusters").unwrap();
    let timeline = candle_conversation::projection::TimelineId::from_raw(13).expect("timeline id");
    conv.register_timeline(timeline, layer, group);

    // Turn 0 = the call (0xAAAA…), turn 1 = its response (0x5555…, the bitwise
    // complement — so it does NOT self-match the probe on its own), turn 2 = an
    // uncoupled turn (0xFFFF…).
    let fills = [
        0xAAAA_AAAA_AAAA_AAAAu64,
        0x5555_5555_5555_5555u64,
        0xFFFF_FFFF_FFFF_FFFFu64,
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
    // Couple turn 0 → turn 1: they are the two halves of one round-trip.
    conv.write().couple_turn(timeline, 0);

    // Probe = the CALL turn's window. Its response (turn 1) is the complement, so
    // pre-exchange it would score near zero; grouping must lift it to turn 0's.
    let probe = vec![sig(fills[0])];
    let mut scores = ProjectionScores::new();
    let target = ProjectionTarget {
        layer,
        group,
        timeline,
    };
    let candidates = conv.score_belief_groups(
        &builder.schema().layers[0],
        target,
        &probe,
        &mut scores,
        false,
        None,
    );

    // Every member turn is still reported (both halves + the singleton).
    assert_eq!(candidates.len(), 1);
    assert_eq!(candidates[0].1.len(), 3, "both halves + the uncoupled turn");

    let s0 = scores.turn(timeline, TurnIndex(0));
    let s1 = scores.turn(timeline, TurnIndex(1));
    let s2 = scores.turn(timeline, TurnIndex(2));
    assert_eq!(
        s0, s1,
        "the coupled call and response share one exchange score: s0={s0}, s1={s1}"
    );
    assert!(
        s0 > s2 && s0 > 0.0,
        "the matched exchange must outscore the uncoupled turn: s0={s0}, s2={s2}"
    );
}

const TWO_LAYER_YAML: &str = r#"
system_prompt:
  sections:
    - id: frame
      content: "frame"
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
    let (scores, cands) = conv.score_beliefs(builder.schema(), target, &probe, false, None);

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

/// Self-local ingest gate: when the projection TARGET is on an append-only ingest
/// layer, every belief group — including non-target cross-layer ones — is masked
/// to the target's own timeline, so a scope-summary is grounded in its own scope
/// and never pulls cross-file/cross-group turns. Same setup as
/// `score_beliefs_scores_a_group_in_a_non_target_layer`, but the target layer is
/// marked append-only, so the clusters group (which HAS matching turns on `mem_tl`)
/// scores NOTHING — its turns are masked away because the target timeline has none.
#[test]
fn append_only_target_masks_belief_groups_self_local() {
    use candle_conversation::projection::TurnIndex;

    let dir = tempfile::tempdir().unwrap();
    let conv = open_conversation(dir.path());
    let builder = Builder::from_yaml(TWO_LAYER_YAML).unwrap();
    let mem_layer = builder.id_for_layer("mem").unwrap();
    let clusters = builder.id_for_group("clusters").unwrap();
    let dialogue_layer = builder.id_for_layer("dialogue").unwrap();
    let convo = builder.id_for_group("convo").unwrap();

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
        conv.persist_wide_q_sigs(
            turn_stream_id(mem_tl.raw(), idx.0),
            &encode_wide_sigs(&[sig(fill)]),
        )
        .expect("persist sigs");
    }

    let dlg_tl = candle_conversation::projection::TimelineId::from_raw(22).unwrap();
    conv.register_timeline(dlg_tl, dialogue_layer, convo);
    // Mark the TARGET layer append-only — this is the whole difference from the
    // base test (where clusters scores across its timelines).
    conv.mark_layer_append_only(dialogue_layer);
    let target = ProjectionTarget {
        layer: dialogue_layer,
        group: convo,
        timeline: dlg_tl,
    };
    let probe = vec![sig(fills[1])];
    let (scores, cands) = conv.score_beliefs(builder.schema(), target, &probe, false, None);

    // The clusters group's turns (on mem_tl) are masked to the target timeline
    // (dlg_tl), which has none — so nothing is scored and no candidates surface.
    assert_eq!(scores.turn(mem_tl, TurnIndex(0)), 0.0);
    assert_eq!(scores.turn(mem_tl, TurnIndex(1)), 0.0);
    assert_eq!(scores.turn(mem_tl, TurnIndex(2)), 0.0);
    assert!(
        !cands.iter().any(|(gid, _)| *gid == clusters),
        "append-only target: cross-layer clusters group must NOT be scored",
    );
}

/// Regression: a belief-driven group holding MANY conversations
/// (`code_reading` declares one timeline per file) must have EVERY conversation
/// scored, not just the first-registered one. Before the fix, the scan resolved
/// the group to a single timeline, so every file but the first sat at score 0 —
/// structurally unreachable no matter how well it matched the query.
#[test]
fn score_belief_groups_scores_every_conversation_in_a_multi_file_group() {
    use candle_conversation::projection::{TimelineId, TurnIndex};

    let dir = tempfile::tempdir().unwrap();
    let conv = open_conversation(dir.path());
    let builder = Builder::from_yaml(TWO_LAYER_YAML).unwrap();
    let mem_layer = builder.id_for_layer("mem").unwrap();
    let clusters = builder.id_for_group("clusters").unwrap();
    let dialogue_layer = builder.id_for_layer("dialogue").unwrap();
    let convo = builder.id_for_group("convo").unwrap();

    // Two "files", each its own conversation under the `clusters` group. Both
    // carry the same three signatures — crucially their turn indices COLLIDE
    // (each file has turns 0/1/2), the ambiguity the (timeline, index) key
    // resolves. Turn 1 (`0x5555…`) is the probe target in BOTH files.
    let file_a = TimelineId::from_raw(201).unwrap();
    let file_b = TimelineId::from_raw(202).unwrap();
    conv.register_timeline(file_a, mem_layer, clusters);
    conv.register_timeline(file_b, mem_layer, clusters);
    let fills = [
        0xAAAA_AAAA_AAAA_AAAAu64,
        0x5555_5555_5555_5555u64,
        0xFFFF_FFFF_FFFF_FFFFu64,
    ];
    for tl in [file_a, file_b] {
        for fill in fills {
            let idx = conv
                .record_turn(
                    tl,
                    Role::User,
                    TurnPartWrite {
                        token_count: 4,
                        tags: vec!["repo_map".to_string()],
                        ..Default::default()
                    },
                    |seqs| Ok(seqs.to_vec()),
                )
                .expect("record_turn");
            conv.persist_wide_q_sigs(
                turn_stream_id(tl.raw(), idx.0),
                &encode_wide_sigs(&[sig(fill)]),
            )
            .expect("persist sigs");
        }
    }

    // Target the DIALOGUE group, so `clusters` is a non-target group and its
    // conversations aren't masked to a single timeline.
    let dlg_tl = TimelineId::from_raw(299).unwrap();
    conv.register_timeline(dlg_tl, dialogue_layer, convo);
    let target = ProjectionTarget {
        layer: dialogue_layer,
        group: convo,
        timeline: dlg_tl,
    };
    let probe = vec![sig(fills[1])];
    let (scores, cands) = conv.score_beliefs(builder.schema(), target, &probe, false, None);

    // The probed turn (index 1) wins in EVERY file — including `file_b`, the
    // second-registered one, which the old collapse never scored.
    for (label, tl) in [("file_a", file_a), ("file_b", file_b)] {
        let s0 = scores.turn(tl, TurnIndex(0));
        let s1 = scores.turn(tl, TurnIndex(1));
        let s2 = scores.turn(tl, TurnIndex(2));
        assert!(
            s1 > 0.0 && s1 > s0 && s1 > s2,
            "{label}: probed turn 1 must be scored highest, not left at 0: \
             s0={s0}, s1={s1}, s2={s2}",
        );
    }

    // The challenger candidates carry turns from BOTH conversations, keyed by
    // (timeline, index) so the colliding indices stay distinct.
    let cluster_cands = cands
        .iter()
        .find(|(g, _)| *g == clusters)
        .map(|(_, c)| c)
        .expect("clusters group reported candidates");
    assert_eq!(
        cluster_cands.len(),
        6,
        "three turns from each of the two files",
    );
    assert!(cluster_cands.iter().any(|(k, _)| k.timeline == file_a));
    assert!(cluster_cands.iter().any(|(k, _)| k.timeline == file_b));
}

/// The wired reproject path (`device = Some`) must produce the SAME per-turn
/// scores as the CPU per-file scan (`device = None`) — the GPU segmented scan is
/// a proven-equivalent accelerator, not a behaviour change. This drives the
/// resolver's actual GPU branch (build → cache → segmented `scan_weighted` →
/// per-file split → normalize) rather than the raw primitive, and calls it TWICE
/// so the second call exercises the resident-gallery cache-hit branch. Skips
/// cleanly when no CUDA device is present.
#[test]
fn score_belief_groups_gpu_matches_cpu_and_caches() {
    use candle_conversation::projection::{TimelineId, TurnIndex};

    let device = match candle::Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => return, // no GPU here — skip
    };

    let dir = tempfile::tempdir().unwrap();
    let conv = open_conversation(dir.path());
    let builder = Builder::from_yaml(TWO_LAYER_YAML).unwrap();
    let mem_layer = builder.id_for_layer("mem").unwrap();
    let clusters = builder.id_for_group("clusters").unwrap();
    let dialogue_layer = builder.id_for_layer("dialogue").unwrap();
    let convo = builder.id_for_group("convo").unwrap();

    // Two files under `clusters`, three turns each with distinct sign patterns.
    let file_a = TimelineId::from_raw(401).unwrap();
    let file_b = TimelineId::from_raw(402).unwrap();
    conv.register_timeline(file_a, mem_layer, clusters);
    conv.register_timeline(file_b, mem_layer, clusters);
    let fills = [
        0xAAAA_AAAA_AAAA_AAAAu64,
        0x5555_5555_5555_5555u64,
        0xF0F0_F0F0_0F0F_0F0Fu64,
    ];
    for tl in [file_a, file_b] {
        for fill in fills {
            let idx = conv
                .record_turn(
                    tl,
                    Role::User,
                    TurnPartWrite {
                        token_count: 4,
                        tags: vec!["repo_map".to_string()],
                        ..Default::default()
                    },
                    |seqs| Ok(seqs.to_vec()),
                )
                .expect("record_turn");
            conv.persist_wide_q_sigs(
                turn_stream_id(tl.raw(), idx.0),
                &encode_wide_sigs(&[sig(fill)]),
            )
            .expect("persist sigs");
        }
    }

    // Non-target group so both files are scored (not masked to one timeline).
    let dlg_tl = TimelineId::from_raw(499).unwrap();
    conv.register_timeline(dlg_tl, dialogue_layer, convo);
    let target = ProjectionTarget {
        layer: dialogue_layer,
        group: convo,
        timeline: dlg_tl,
    };
    let probe = vec![sig(fills[1]), sig(fills[2])];

    // The paged gallery arena (folded geometry: wpt 24, 3 groups).
    let arena = candle_conversation::provenance::GalleryArena::new(&device, 24, 3).unwrap();

    let all_scores = |arena: Option<&candle_conversation::provenance::GalleryArena>| {
        let mut scores = ProjectionScores::new();
        conv.score_belief_groups(
            &builder.schema().layers[0],
            target,
            &probe,
            &mut scores,
            false,
            arena,
        );
        let mut v = Vec::new();
        for tl in [file_a, file_b] {
            for i in 0..3 {
                v.push(scores.turn(tl, TurnIndex(i)));
            }
        }
        v
    };

    let cpu = all_scores(None);
    let gpu1 = all_scores(Some(&arena)); // first: makes the turns resident + scans
    let gpu2 = all_scores(Some(&arena)); // second: all-resident hit (no upload)

    // Same normalized scores on the CPU and paged-GPU paths (fast-math ⇒ ~ULP).
    for (i, (c, g)) in cpu.iter().zip(&gpu1).enumerate() {
        assert!(
            (c - g).abs() <= 1e-3 * (1.0 + c.abs().max(g.abs())),
            "turn {i}: CPU {c} vs paged-GPU {g} exceeds tolerance"
        );
    }
    // The resident-hit scan is bit-identical to the first (same resident pages).
    assert_eq!(gpu1, gpu2, "resident-hit scan must equal the first scan");
    // The second scan must not have grown residency (all turns already resident).
    assert_eq!(
        arena.resident_turns(),
        6,
        "six turns resident, no re-upload"
    );
    // Sanity: the scored turns aren't all zero (the scan actually ran on GPU).
    assert!(
        gpu1.iter().any(|&s| s > 0.0),
        "paged-GPU scan produced non-zero scores"
    );
}

const COLLECTION_YAML: &str = r#"
system_prompt:
  items:
    - kind: section
      id: frame
      content: "frame"
    - kind: collection
      name: tools
      selection: { kind: top_k, k: 2 }
      policy:
        tags: ["alpha", "beta"]
      sections:
        - id: alpha
          content: "alpha tool"
        - id: beta
          content: "beta tool"
        - id: gamma
          content: "gamma tool"
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
    groups:
      - id: clusters
        selection: { kind: top_k, k: 2 }
"#;

/// The COLLECTION belief scan (tools/moods/responses — the reproject's other
/// scan besides the turn groups) must produce the same per-section scores on
/// the GPU arena path as on the CPU `score_slots_weighted` path. The gallery is
/// tag-scoped turns whose tag names a section; the arena scans it as one
/// global-z segment. Skips without CUDA.
#[test]
fn score_belief_collections_gpu_matches_cpu() {
    let device = match candle::Device::new_cuda(0) {
        Ok(d) => d,
        Err(_) => return, // no GPU here — skip
    };

    let dir = tempfile::tempdir().unwrap();
    let conv = open_conversation(dir.path());
    let builder = Builder::from_yaml(COLLECTION_YAML).unwrap();
    let layer = builder.id_for_layer("mem").unwrap();
    let group = builder.id_for_group("clusters").unwrap();
    let timeline = candle_conversation::projection::TimelineId::from_raw(21).expect("timeline id");
    conv.register_timeline(timeline, layer, group);

    // Gallery: turns tagged with a section name become that section's exemplars.
    // Multi-token windows (40 tokens = two arena pages; 17 = partial page).
    let fills: [(&str, u64, usize); 4] = [
        ("alpha", 0xAAAA_AAAA_AAAA_AAAA, 40),
        ("alpha", 0xABAB_ABAB_ABAB_ABAB, 17),
        ("beta", 0x5555_5555_5555_5555, 33),
        ("beta", 0x1234_5678_9ABC_DEF0, 8),
    ];
    for (tag, fill, len) in fills {
        let idx = conv
            .record_turn(
                timeline,
                Role::User,
                TurnPartWrite {
                    token_count: 4,
                    tags: vec![tag.to_string()],
                    ..Default::default()
                },
                |seqs| Ok(seqs.to_vec()),
            )
            .expect("record_turn");
        let window: Vec<WideQSig> = (0..len).map(|_| sig(fill)).collect();
        conv.persist_wide_q_sigs(
            turn_stream_id(timeline.raw(), idx.0),
            &encode_wide_sigs(&window),
        )
        .expect("persist sigs");
    }

    // Probe matching the alpha pattern → alpha must win on both paths.
    let probe: Vec<WideQSig> = (0..6).map(|_| sig(0xAAAA_AAAA_AAAA_AAAA)).collect();
    let sp = &builder.schema().system_prompt;
    let coll = sp.collection_named("tools").expect("tools collection");

    let arena = candle_conversation::provenance::GalleryArena::new(&device, 24, 3).unwrap();
    let cpu = conv.score_belief_collections(sp, &probe, None);
    let gpu = conv.score_belief_collections(sp, &probe, Some(&arena));

    let mut cpu_scores = Vec::new();
    let mut gpu_scores = Vec::new();
    for s in &coll.sections {
        let c = cpu.section(s.id);
        let g = gpu.section(s.id);
        assert!(
            (c - g).abs() <= 1e-3 * (1.0 + c.abs().max(g.abs())),
            "section {}: CPU {c} vs GPU {g} exceeds tolerance",
            s.name
        );
        cpu_scores.push((s.name.clone(), c));
        gpu_scores.push((s.name.clone(), g));
    }
    // Alpha dominates on both paths; gamma (no gallery) reads zero.
    for scores in [&cpu_scores, &gpu_scores] {
        let alpha = scores.iter().find(|(n, _)| n == "alpha").unwrap().1;
        let beta = scores.iter().find(|(n, _)| n == "beta").unwrap().1;
        let gamma = scores.iter().find(|(n, _)| n == "gamma").unwrap().1;
        assert!(
            alpha > beta && alpha > 0.0,
            "alpha must dominate: alpha={alpha}, beta={beta}"
        );
        assert_eq!(gamma, 0.0, "gamma has no gallery exemplars");
    }
    // A second GPU scan hits the arena's index cache and must be identical.
    let gpu2 = conv.score_belief_collections(sp, &probe, Some(&arena));
    for s in &coll.sections {
        assert_eq!(
            gpu.section(s.id).to_bits(),
            gpu2.section(s.id).to_bits(),
            "cached collection scan must be bit-identical"
        );
    }
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
    let candidates = conv.score_belief_groups(
        &builder.schema().layers[0],
        target,
        &[],
        &mut scores,
        false,
        None,
    );
    assert!(candidates.is_empty(), "empty probe → no scan");
}
