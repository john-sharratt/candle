//! Integration tests for the binary directional provenance retrieval system.
//!
//! Tests prove end-to-end accuracy of the sign-bit pipeline without requiring
//! a real model or GPU:
//!
//!   synthetic Q data → TokenSignature → ProvenanceFile::append
//!                      scan_entries   → TurnChunkRank (correct turn at index 0)

use candle_conversation::provenance::signature::{
    extract_signatures_from_r16_dump, r16_block_to_turn_signatures,
};
use candle_conversation::provenance::{
    ProbeSignatures, ProvenanceFile, SigEntry, TokenSignature, TurnSignatures,
};
use candle_nn::kv_cache::CHUNK_SIZE;

// ── Helpers ───────────────────────────────────────────────────────────────────

fn q_hot_n(n_hot: usize) -> Vec<f32> {
    (0..128)
        .map(|i| if i < n_hot { 1.0_f32 } else { -1.0_f32 })
        .collect()
}

/// Build a `ProbeSignatures` with the same signature at all three depths.
fn uniform_probe(sig: TokenSignature) -> ProbeSignatures {
    ProbeSignatures {
        syntactic: sig,
        semantic: sig,
        pragmatic: sig,
    }
}

/// Append one chunk group (same sig repeated `n_tokens` times at all three depths)
/// and return the resulting `SigEntry`.
fn append_uniform(pf: &ProvenanceFile, sig: TokenSignature, n_tokens: usize) -> SigEntry {
    let sigs: Vec<TokenSignature> = vec![sig; n_tokens];
    pf.append(&sigs, &sigs, &sigs).unwrap()
}

fn make_r16_q_flat(n_kv_head: usize, head_dim: usize, value: f32) -> Vec<f32> {
    let sub_head_dim = head_dim / 4;
    let floats_per_head = 4 * CHUNK_SIZE * sub_head_dim;
    vec![value; n_kv_head * floats_per_head]
}

// ── TokenSignature math ───────────────────────────────────────────────────────

#[test]
fn agreement_of_identical_signatures_is_128() {
    let q: Vec<f32> = (0..128)
        .map(|i| if i % 3 == 0 { 1.0 } else { -1.0 })
        .collect();
    let s = TokenSignature::from_q_flat(&q);
    assert_eq!(s.agreement(&s), 128);
}

#[test]
fn agreement_of_opposite_signatures_is_zero() {
    let pos = TokenSignature::from_q_flat(&[1.0_f32; 128]);
    let neg = TokenSignature::from_q_flat(&[-1.0_f32; 128]);
    assert_eq!(pos.agreement(&neg), 0);
}

#[test]
fn agreement_plus_hamming_distance_is_always_128() {
    for n_hot in [0, 32, 64, 96, 128] {
        let s1 = TokenSignature::from_q_flat(&q_hot_n(n_hot));
        let s2 = TokenSignature::from_q_flat(&q_hot_n(128 - n_hot));
        assert_eq!(
            s1.agreement(&s2) + s1.hamming_distance(&s2),
            128,
            "n_hot={n_hot}"
        );
    }
}

// ── Density score formula ─────────────────────────────────────────────────────

#[test]
fn density_score_formula_is_hits_squared_over_length() {
    // 6 hits out of 10 tokens → combined score = 3 × (36/10) = 10.8
    let pf = ProvenanceFile::new().unwrap();
    let hit = TokenSignature::from_q_flat(&[1.0_f32; 128]);
    let miss = TokenSignature::from_q_flat(&[-1.0_f32; 128]);
    let sigs: Vec<TokenSignature> = (0..10).map(|i| if i < 6 { hit } else { miss }).collect();
    let entry = pf.append(&sigs, &sigs, &sigs).unwrap();

    let probe = uniform_probe(hit);
    let entries = vec![(0u64, entry)];
    let ranks = pf.scan_entries(&entries, &probe, 80, 10).unwrap();

    assert_eq!(ranks.len(), 1);
    let per_depth = 36.0_f64 / 10.0;
    let expected = per_depth * 3.0; // three depths, all identical
    assert!(
        (ranks[0].score - expected).abs() < 1e-9,
        "score={}",
        ranks[0].score
    );
}

// ── Single-probe retrieval accuracy ──────────────────────────────────────────

#[test]
fn scan_entries_returns_correct_turn_first() {
    let q_a: Vec<f32> = (0..128).map(|i| if i < 64 { 1.0 } else { -1.0 }).collect();
    let q_b: Vec<f32> = (0..128).map(|i| if i >= 64 { 1.0 } else { -1.0 }).collect();

    let sig_a = TokenSignature::from_q_flat(&q_a);
    let sig_b = TokenSignature::from_q_flat(&q_b);

    let pf = ProvenanceFile::new().unwrap();
    let e0 = append_uniform(&pf, sig_a, 8);
    let e1 = append_uniform(&pf, sig_b, 8);

    let entries = vec![(0u64, e0), (1u64, e1)];
    let probe = uniform_probe(sig_a);
    let ranks = pf.scan_entries(&entries, &probe, 80, 10).unwrap();

    assert!(!ranks.is_empty());
    assert_eq!(ranks[0].turn_id, 0, "turn 0 (topic A) must rank first");
    // topic_b has agreement 0 — excluded
    assert_eq!(ranks.len(), 1, "only topic A should pass threshold 80");
}

#[test]
fn scan_entries_topic_b_probe_selects_topic_b_turn() {
    let q_a: Vec<f32> = (0..128).map(|i| if i < 64 { 1.0 } else { -1.0 }).collect();
    let q_b: Vec<f32> = (0..128).map(|i| if i >= 64 { 1.0 } else { -1.0 }).collect();

    let pf = ProvenanceFile::new().unwrap();
    let e0 = append_uniform(&pf, TokenSignature::from_q_flat(&q_a), 10);
    let e1 = append_uniform(&pf, TokenSignature::from_q_flat(&q_b), 10);

    let probe = uniform_probe(TokenSignature::from_q_flat(&q_b));
    let ranks = pf
        .scan_entries(&[(0u64, e0), (1u64, e1)], &probe, 80, 10)
        .unwrap();

    assert_eq!(ranks.len(), 1);
    assert_eq!(ranks[0].turn_id, 1);
}

#[test]
fn scan_entries_density_favours_dense_turn_over_longer_sparse_turn() {
    let hit = TokenSignature::from_q_flat(&[1.0_f32; 128]);
    let miss = TokenSignature::from_q_flat(&[-1.0_f32; 128]);

    let pf = ProvenanceFile::new().unwrap();
    // Turn 0: 5 hot tokens in 5 → score per depth = 25/5 = 5.0
    let sigs0 = vec![hit; 5];
    let e0 = pf.append(&sigs0, &sigs0, &sigs0).unwrap();
    // Turn 1: 3 hot tokens in 20 → score per depth = 9/20 = 0.45
    let sigs1: Vec<_> = (0..20).map(|i| if i < 3 { hit } else { miss }).collect();
    let e1 = pf.append(&sigs1, &sigs1, &sigs1).unwrap();

    let probe = uniform_probe(hit);
    let ranks = pf
        .scan_entries(&[(0u64, e0), (1u64, e1)], &probe, 80, 10)
        .unwrap();

    assert_eq!(ranks.len(), 2);
    assert_eq!(ranks[0].turn_id, 0, "dense short turn should rank first");
    assert!(ranks[0].score > ranks[1].score);
    assert!((ranks[0].score - 15.0).abs() < 1e-9); // 5.0 × 3 depths
    assert!((ranks[1].score - 9.0 / 20.0 * 3.0).abs() < 1e-9);
}

// ── top_k limit ───────────────────────────────────────────────────────────────

#[test]
fn scan_entries_top_k_limits_results() {
    let hit = TokenSignature::from_q_flat(&[1.0_f32; 128]);
    let pf = ProvenanceFile::new().unwrap();
    let entries: Vec<(u64, SigEntry)> = (0..5)
        .map(|i| {
            let sigs = vec![hit; i + 1];
            (i as u64, pf.append(&sigs, &sigs, &sigs).unwrap())
        })
        .collect();

    let probe = uniform_probe(hit);
    let ranks = pf.scan_entries(&entries, &probe, 80, 3).unwrap();
    assert_eq!(ranks.len(), 3);
}

// ── Edge cases ────────────────────────────────────────────────────────────────

#[test]
fn scan_entries_empty_entries_returns_empty() {
    let pf = ProvenanceFile::new().unwrap();
    let probe = uniform_probe(TokenSignature::from_q_flat(&[1.0_f32; 128]));
    let ranks = pf.scan_entries(&[], &probe, 64, 10).unwrap();
    assert!(ranks.is_empty());
}

#[test]
fn scan_entries_zero_token_entry_is_skipped() {
    let pf = ProvenanceFile::new().unwrap();
    let zero_entry = pf.append(&[], &[], &[]).unwrap();
    let hit = TokenSignature::from_q_flat(&[1.0_f32; 128]);
    let real_entry = append_uniform(&pf, hit, 4);

    let probe = uniform_probe(hit);
    let ranks = pf
        .scan_entries(&[(0u64, zero_entry), (1u64, real_entry)], &probe, 80, 10)
        .unwrap();
    assert_eq!(ranks.len(), 1);
    assert_eq!(ranks[0].turn_id, 1);
}

// ── R16 bridge ────────────────────────────────────────────────────────────────

#[test]
fn r16_block_positive_q_produces_all_ones_signatures() {
    let q = make_r16_q_flat(4, 128, 1.0);
    let sigs = r16_block_to_turn_signatures(&q, 4, 128, CHUNK_SIZE);
    assert_eq!(sigs.sigs.len(), CHUNK_SIZE);
    for s in &sigs.sigs {
        assert_eq!(s.as_u128(), u128::MAX);
    }
}

#[test]
fn r16_block_negative_q_produces_all_zeros_signatures() {
    let q = make_r16_q_flat(4, 128, -1.0);
    let sigs = r16_block_to_turn_signatures(&q, 4, 128, CHUNK_SIZE);
    for s in &sigs.sigs {
        assert_eq!(s.as_u128(), 0u128);
    }
}

#[test]
fn r16_block_partial_tokens_count_is_correct() {
    let q = make_r16_q_flat(2, 128, 1.0);
    let sigs = r16_block_to_turn_signatures(&q, 2, 128, 7);
    assert_eq!(sigs.sigs.len(), 7);
}

#[test]
fn r16_extract_dump_produces_one_turn_signatures_per_block() {
    let q = make_r16_q_flat(4, 128, 1.0);
    let blocks = vec![
        (0, vec![], vec![], q.clone()),
        (1, vec![], vec![], q.clone()),
        (2, vec![], vec![], q.clone()),
    ];
    let result = extract_signatures_from_r16_dump(&blocks, 4, 128, CHUNK_SIZE);
    assert_eq!(result.len(), 3);
    for ts in &result {
        assert_eq!(ts.sigs.len(), CHUNK_SIZE);
    }
}

// ── End-to-end: R16 dump → ProvenanceFile → scan_entries ─────────────────────

#[test]
fn end_to_end_r16_to_ranked_retrieval() {
    let q_a = make_r16_q_flat(4, 128, 1.0); // topic A — positive
    let q_b = make_r16_q_flat(4, 128, -1.0); // topic B — negative

    let sigs_a = extract_signatures_from_r16_dump(&[(0, vec![], vec![], q_a)], 4, 128, CHUNK_SIZE);
    let sigs_b = extract_signatures_from_r16_dump(&[(1, vec![], vec![], q_b)], 4, 128, CHUNK_SIZE);

    let pf = ProvenanceFile::new().unwrap();
    let e0 = pf
        .append(&sigs_a[0].sigs, &sigs_a[0].sigs, &sigs_a[0].sigs)
        .unwrap();
    let e1 = pf
        .append(&sigs_b[0].sigs, &sigs_b[0].sigs, &sigs_b[0].sigs)
        .unwrap();

    // Probe: positive Q → should retrieve turn 0 (topic A).
    let probe_sig = TokenSignature::from_q_flat(&[1.0_f32; 128]);
    let probe = uniform_probe(probe_sig);

    let ranks = pf
        .scan_entries(&[(0u64, e0), (1u64, e1)], &probe, 80, 10)
        .unwrap();

    assert!(!ranks.is_empty());
    assert_eq!(
        ranks[0].turn_id, 0,
        "turn 0 (positive/topic A) must rank first"
    );
    assert_eq!(ranks.len(), 1, "topic B (agreement=0) should not appear");
}

// ── TurnSignatures helpers ────────────────────────────────────────────────────

#[test]
fn from_q_flat_token_major_produces_correct_signatures() {
    let mut q_flat = Vec::with_capacity(3 * 128);
    q_flat.extend(vec![1.0_f32; 128]);
    q_flat.extend(vec![-1.0_f32; 128]);
    q_flat.extend((0..128).map(|i| if i % 2 == 0 { 1.0_f32 } else { -1.0_f32 }));

    let ts = TurnSignatures::from_q_flat_token_major(&q_flat, 3, 128);
    assert_eq!(ts.sigs.len(), 3);
    assert_eq!(ts.sigs[0].as_u128(), u128::MAX);
    assert_eq!(ts.sigs[1].as_u128(), 0);
    let expected: u128 = u128::from_le_bytes([0x55u8; 16]);
    assert_eq!(ts.sigs[2].as_u128(), expected);
}
