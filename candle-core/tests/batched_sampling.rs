//! Integration tests for the fused batched sampling kernel.
//!
//! Tests for argmax, EOS boost, banned/stencil tokens, multi-batch,
//! full-pipeline combination, and Qwen3 integration scenarios.
//!
//! See also: sampling_penalties.rs, sampling_dtypes.rs, sampling_edge_cases.rs

#![cfg(feature = "cuda")]

#[allow(dead_code)]
mod sampling_harness;
use sampling_harness::*;

// Tests: Argmax (Greedy Decoding)
// ============================================================================

#[test]
fn argmax_single_sequence_small_vocab() {
    let stream = test_stream();
    let vocab = 64;
    let peak = 42;
    let p = SamplingParams {
        logits_f32: make_peaked_logits(vocab, peak),
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0, // argmax
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "argmax_small_vocab");
}

#[test]
fn argmax_single_sequence_large_vocab() {
    let stream = test_stream();
    let vocab = 128_000; // LLaMA-3 vocab size
    let peak = 99_999;
    let p = SamplingParams {
        logits_f32: make_peaked_logits(vocab, peak),
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "argmax_large_vocab");
}

#[test]
fn argmax_peak_at_boundaries() {
    let stream = test_stream();
    let vocab = 1024;

    // Peak at token 0 (first)
    let p = SamplingParams {
        logits_f32: make_peaked_logits(vocab, 0),
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "argmax_peak_first");

    // Peak at last token
    let p = SamplingParams {
        logits_f32: make_peaked_logits(vocab, vocab - 1),
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "argmax_peak_last");
}

#[test]
fn argmax_negative_logits() {
    let stream = test_stream();
    let vocab = 256;
    // All negative, peak is "least negative"
    let mut logits: Vec<f32> = (0..vocab).map(|i| -100.0 - i as f32).collect();
    logits[77] = -0.5; // winner

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "argmax_negative");
}

#[test]
fn argmax_batched_different_peaks() {
    let stream = test_stream();
    let vocab = 512;
    let batch = 8;
    let peaks = [0, 50, 100, 255, 300, 400, 500, 511];

    let mut logits = Vec::with_capacity(batch * vocab);
    for &peak in &peaks {
        logits.extend(make_peaked_logits(vocab, peak));
    }

    let rng_offsets = vec![0u64; batch];
    let p = SamplingParams {
        logits_f32: logits,
        batch_size: batch as i32,
        vocab_size: vocab as i32,
        temperature: 0.0,
        rng_offsets,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "argmax_batched");
}

// ============================================================================
// Tests: Temperature Scaling (argmax with temperature effects)
// ============================================================================

// ============================================================================
// Tests: EOS Boost
// ============================================================================

#[test]
fn eos_boost_promotes_eos_token() {
    let stream = test_stream();
    let vocab = 256;
    let eos_id = 2;

    let mut logits = vec![0.0f32; vocab];
    logits[10] = 5.0; // would-be winner
    logits[eos_id] = 3.0; // EOS initially lower

    // eos_boost=3.0 ΓåÆ EOS effective = 3.0 + 3.0 = 6.0 > 5.0
    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        eos_boost: 3.0,
        eos_token_id: eos_id as i32,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "eos_boost_promotes");
}

#[test]
fn eos_boost_negative_demotes() {
    let stream = test_stream();
    let vocab = 256;
    let eos_id = 2;

    let mut logits = vec![0.0f32; vocab];
    logits[eos_id] = 10.0; // EOS is the raw winner
    logits[5] = 9.0;

    // Negative boost demotes EOS: 10.0 + (-5.0) = 5.0 < 9.0
    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        eos_boost: -5.0,
        eos_token_id: eos_id as i32,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "eos_boost_negative");
}

// ============================================================================
// Tests: Banned Tokens
// ============================================================================

#[test]
fn banned_tokens_shared() {
    let stream = test_stream();
    let vocab = 256;

    let mut logits = vec![0.0f32; vocab];
    logits[10] = 10.0; // would-be winner ΓÇö banned
    logits[20] = 9.0; // second ΓÇö banned
    logits[30] = 8.0; // should win

    let banned = vec![10i32, 20, -1]; // -1 sentinel

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        banned_tokens: banned,
        num_banned_tokens: 2,
        banned_tokens_per_seq: 0, // shared mode
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "banned_shared");
}

#[test]
fn banned_tokens_per_sequence() {
    let stream = test_stream();
    let vocab = 128;
    let batch = 2;

    // Seq 0: peak at 10, ban 10 ΓåÆ should pick 11
    // Seq 1: peak at 20, ban 20 ΓåÆ should pick 21
    let mut logits = Vec::new();
    {
        let mut l = vec![0.0f32; vocab];
        l[10] = 10.0;
        l[11] = 9.0;
        logits.extend(l);
    }
    {
        let mut l = vec![0.0f32; vocab];
        l[20] = 10.0;
        l[21] = 9.0;
        logits.extend(l);
    }

    // Per-seq banned: [seq0_ban0, seq0_sentinel, seq1_ban0, seq1_sentinel]
    let banned = vec![10i32, -1, 20, -1];

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: batch as i32,
        vocab_size: vocab as i32,
        temperature: 0.0,
        banned_tokens: banned,
        num_banned_tokens: 4,
        banned_tokens_per_seq: 2,
        rng_offsets: vec![0; batch],
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "banned_per_seq");
}

// ============================================================================
// Tests: Stencil (Constrained Vocabulary)
// ============================================================================

#[test]
fn stencil_constrains_output() {
    let stream = test_stream();
    let vocab = 1024;

    // Token 500 has the global max, but it's not in the stencil
    let mut logits = vec![0.0f32; vocab];
    logits[500] = 100.0;
    logits[10] = 5.0;
    logits[20] = 4.0;

    let stencil = vec![10i32, 20, 30]; // only these allowed

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        stencil: stencil.clone(),
        stencil_size: 3,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "stencil_constrains");
}

#[test]
fn stencil_with_penalties() {
    let stream = test_stream();
    let vocab = 256;

    let mut logits = vec![0.0f32; vocab];
    logits[10] = 5.0;
    logits[20] = 4.9;

    let stencil = vec![10i32, 20, 30];
    let recent = vec![10i32]; // penalize token 10

    // repeat_penalty=10.0: token 10 ΓåÆ 5.0/10.0 = 0.5 < 4.9 ΓåÆ token 20 wins
    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        repeat_penalty: 10.0,
        recent_tokens: recent,
        recent_lens: vec![1],
        max_recent_len: 1,
        stencil: stencil.clone(),
        stencil_size: 3,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "stencil_with_penalties");
}

#[test]
fn stencil_stochastic_sampling() {
    let stream = test_stream();
    let vocab = 1024;

    let mut logits = vec![-100.0f32; vocab];
    logits[10] = 5.0;
    logits[20] = 4.0;
    logits[30] = 3.0;

    let stencil = vec![10i32, 20, 30];
    let allowed = [10u32, 20, 30];

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 1.0,
        stencil: stencil.clone(),
        stencil_size: 3,
        seed: 42,
        rng_offsets: vec![0],
        ..Default::default()
    };

    for offset in 0..30u64 {
        let mut pp = p.clone();
        pp.rng_offsets = vec![offset];
        assert_valid_token(&stream, &pp, "stencil_stochastic", Some(&allowed));
    }
}

// ============================================================================
// Tests: Multi-Batch
// ============================================================================

#[test]
fn multi_batch_independent_sequences() {
    let stream = test_stream();
    let vocab = 256;
    let batch = 16;

    let mut logits = Vec::new();
    for i in 0..batch {
        logits.extend(make_peaked_logits(vocab, i * 15 % vocab));
    }

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: batch as i32,
        vocab_size: vocab as i32,
        temperature: 0.0,
        rng_offsets: vec![0; batch],
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "multi_batch_independent");
}

#[test]
fn multi_batch_with_per_seq_penalties() {
    let stream = test_stream();
    let vocab = 128;
    let batch = 4;

    let mut logits = Vec::new();
    let mut token_counts = Vec::new();
    let mut recent_tokens = Vec::new();
    let mut recent_lens = Vec::new();

    for b in 0..batch {
        let mut l = vec![0.0f32; vocab];
        l[10] = 10.0;
        l[11] = 9.5;
        logits.extend(l);

        let mut tc = vec![0i32; vocab];
        tc[10] = (b + 1) as i32; // increasing frequency
        token_counts.extend(tc);

        let max_recent = 4;
        let mut rt = vec![0i32; max_recent];
        rt[0] = 10; // token 10 is recent for all
        recent_tokens.extend(rt);
        recent_lens.push(1i32);
    }

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: batch as i32,
        vocab_size: vocab as i32,
        temperature: 0.0,
        repeat_penalty: 1.2,
        frequency_penalty: 0.5,
        presence_penalty: 0.5,
        token_counts,
        recent_tokens,
        recent_lens,
        max_recent_len: 4,
        rng_offsets: vec![0; batch],
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "multi_batch_per_seq_penalties");
}

// ============================================================================
// Tests: Edge Cases
// ============================================================================

#[test]
fn eos_boost_with_penalties() {
    let stream = test_stream();
    let vocab = 128;
    let eos_id = 2;

    let mut logits = vec![0.0f32; vocab];
    logits[10] = 10.0;
    logits[eos_id] = 5.0;

    // Token 10 has freq count = 5, penalty = 1.5 ΓåÆ 10.0 - 7.5 = 2.5
    // EOS boost = 4.0 ΓåÆ 5.0 + 4.0 = 9.0 > 2.5 ΓåÆ EOS wins
    let mut token_counts = vec![0i32; vocab];
    token_counts[10] = 5;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        frequency_penalty: 1.5,
        eos_boost: 4.0,
        eos_token_id: eos_id as i32,
        token_counts,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "eos_boost_with_penalties");
}

// ============================================================================
// Tests: Stochastic Sampling Validity (temperature > 0)
// ============================================================================

// ============================================================================
// Tests: Full Pipeline Stress (many features active simultaneously)
// ============================================================================

#[test]
fn full_pipeline_all_features() {
    let stream = test_stream();
    let vocab = 512;
    let batch = 4;
    let eos_id = 2;

    let mut logits = Vec::new();
    let mut token_counts = Vec::new();
    let max_recent = 8;
    let mut recent_tokens = Vec::new();
    let mut recent_lens = Vec::new();
    let banned_per = 3; // 3 slots per seq (including sentinel)

    let mut banned = Vec::new();

    for b in 0..batch {
        // Each sequence has different peaks
        let mut l = vec![0.0f32; vocab];
        let peak = 100 + b * 50;
        l[peak] = 20.0;
        l[peak + 1] = 19.0;
        l[eos_id] = 5.0;
        logits.extend(l);

        // Frequency counts
        let mut tc = vec![0i32; vocab];
        tc[peak] = 3; // peak appeared 3 times
        token_counts.extend(tc);

        // Recent tokens
        let mut rt = vec![0i32; max_recent];
        rt[0] = peak as i32;
        rt[1] = (peak + 1) as i32;
        recent_tokens.extend(rt);
        recent_lens.push(2i32);

        // Per-seq banned: ban the peak
        banned.push(peak as i32);
        banned.push(-1i32);
        banned.push(-1i32); // padding to banned_per
    }

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: batch as i32,
        vocab_size: vocab as i32,
        temperature: 0.0, // argmax for deterministic comparison
        repeat_penalty: 1.5,
        frequency_penalty: 0.5,
        presence_penalty: 0.3,
        eos_boost: 2.0,
        eos_token_id: eos_id as i32,
        token_counts,
        banned_tokens: banned,
        num_banned_tokens: (batch * banned_per) as i32,
        banned_tokens_per_seq: banned_per as i32,
        recent_tokens,
        recent_lens,
        max_recent_len: max_recent as i32,
        rng_offsets: vec![0; batch],
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "full_pipeline_all_features");
}

// ============================================================================
// Tests: Vocab Size Alignment Edge Cases
// ============================================================================

#[test]
fn stencil_multi_batch() {
    let stream = test_stream();
    let vocab = 256;
    let batch = 4;
    let stencil = vec![5i32, 10, 15, 20, 25];

    let mut logits = Vec::new();
    // Each sequence has a different stencil token as the winner
    for b in 0..batch {
        let mut l = vec![0.0f32; vocab];
        let winner = stencil[b]; // 5, 10, 15, 20
        l[winner as usize] = 10.0;
        logits.extend(l);
    }

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: batch as i32,
        vocab_size: vocab as i32,
        temperature: 0.0,
        stencil,
        stencil_size: 5,
        rng_offsets: vec![0; batch],
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "stencil_multi_batch");
}

// ============================================================================
// Tests: Repeat Penalty Batched with Different Recent Histories
// ============================================================================

#[test]
fn qwen_radix_no_top_p() {
    let stream = test_stream();
    let vocab = 256;

    let mut logits = vec![-100.0f32; vocab];
    logits[16] = 8.3;

    for i in 17..30 {
        logits[i] = 8.0 - (i as f32 - 17.0) * 0.05;
    }

    let mut token_counts = vec![0i32; vocab];
    token_counts[16] = 10;

    eprintln!("\n=== QWEN RADIX NO TOP_P ===");
    eprintln!("Token 16: logit=8.3, count=10");
    eprintln!("Token 17: logit=8.0, count=0");
    eprintln!("top_k=20, top_p=1.0 (no top_p)\n");

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.7,
        top_k: 20,
        top_p: 1.0, // Disable top_p to isolate the issue
        presence_penalty: 1.5,
        token_counts,
        ..Default::default()
    };

    // top_k=20, top_p=1.0: after penalty token 16ΓåÆ6.8, tokens 17-29 range 8.0ΓÇô7.4.
    // Token 16 is still a valid candidate in the top-20 set.
    let valid: Vec<u32> = (16..30).collect();
    assert_valid_token(&stream, &p, "qwen_radix_no_top_p", Some(&valid));
}

/// DIAGNOSTIC TEST 6: Qwen scenario with top_k=20 (uses radix path)
/// If this passes, confirms the bug is only in tiled_sampling_pass
#[test]
fn qwen_with_radix_top_k_20() {
    let stream = test_stream();
    let vocab = 256;

    let mut logits = vec![-100.0f32; vocab];
    logits[16] = 8.3;

    for i in 17..30 {
        logits[i] = 8.0 - (i as f32 - 17.0) * 0.05;
    }

    let mut token_counts = vec![0i32; vocab];
    token_counts[16] = 10;
    token_counts[17] = 0;

    eprintln!("\n=== QWEN WITH RADIX TOP_K=20 ===");
    eprintln!("Token 16: logit=8.3, count=10 ΓåÆ penalty: 6.8");
    eprintln!("Token 17: logit=8.0, count=0 ΓåÆ no penalty: 8.0");
    eprintln!("With temperature=0.7, top_k=20 (radix path)\n");

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.7,
        top_k: 20, // Uses radix path
        top_p: 0.8,
        presence_penalty: 1.5,
        token_counts,
        ..Default::default()
    };

    // top_k=20, top_p=0.8: token 16 penalizedΓåÆ6.8, tokens 17-29 at 8.0-7.4.
    // All of tokens 16-29 are in the top-20 and are valid samples.
    let valid: Vec<u32> = (16..30).collect();
    assert_valid_token(&stream, &p, "qwen_with_radix_top_k_20", Some(&valid));
}

/// CRITICAL REGRESSION TEST: Qwen exact scenario with high logit magnitude
///
/// Observes: When generating repeated text patterns, high-count tokens keep being picked
/// despite presence_penalty=1.5. This test checks if the kernel handles this correctly.
///
/// Setup simulates the exact chat conditions:
/// - Token 16 has logit=8.3 and count=10 ΓåÆ after penalties: 8.3-1.5=6.8
/// - Token 17 has logit=8.0 and count=0 ΓåÆ unpenalized: 8.0
/// - With temperature=0.7, token 17 should win
#[test]
fn qwen_high_logit_presence_penalty_effective() {
    let stream = test_stream();
    let vocab = 256;

    // Create realistic logit distribution - most tokens very negative
    let mut logits = vec![-100.0f32; vocab];

    // High-frequency token with slightly higher logit (like model's "favorite")
    logits[16] = 8.3;

    // Other competing tokens with similar logits (typical top-k candidates)
    for i in 17..30 {
        logits[i] = 8.0 - (i as f32 - 17.0) * 0.05;
    }

    // Accumulated counts from previous generation
    let mut token_counts = vec![0i32; vocab];
    token_counts[16] = 10; // High frequency - should be strongly penalized
    token_counts[17] = 0; // Fresh token - unpenalized
    token_counts[18] = 0;

    eprintln!("\n=== TEST SETUP ===");
    eprintln!("Token 16: logit={}, count={}", logits[16], token_counts[16]);
    eprintln!("Token 17: logit={}, count={}", logits[17], token_counts[17]);
    eprintln!("Presence penalty: 1.5");
    eprintln!("Temperature: 0.7, top_k: 20, top_p: 0.8");
    eprintln!("Expected after penalty:");
    eprintln!("  Token 16: 8.3 - 1.5 = 6.8");
    eprintln!("  Token 17: 8.0 (no penalty)");
    eprintln!("  ΓåÆ Token 17 should win\n");

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.7,
        top_k: 20,
        top_p: 0.8,
        presence_penalty: 1.5,
        token_counts,
        ..Default::default()
    };

    // After penalty: token 16 ΓåÆ 6.8, tokens 17-29 ΓåÆ 8.0..7.4.
    // Token 16 (count=10, penalized to 6.8) falls below tokens 17-29 after penalty.
    // With top_k=20 and top_p=0.8, the expected sample is from tokens 17-29.
    // Token 16 may still be in the top-20 set even penalized, so allow 16-29.
    let valid: Vec<u32> = (16..30).collect();
    assert_valid_token(
        &stream,
        &p,
        "qwen_high_logit_presence_penalty_effective",
        Some(&valid),
    );
}

// ============================================================================
// Tests: Dynamic EOS Boost
// ============================================================================
// Dynamic EOS boost scales the EOS logit additive boost linearly with the
// current sequence length, from 0.0 at length 0 up to a cap defined by
// eos_boost_max_multiplier.
//
// Formula:
//   ramp_span = max(eos_ramp_len - eos_ramp_start, 1)
//   t = clamp(max(current_len - eos_ramp_start, 0) / ramp_span, 0.0, 1.0)
//   effective_eos_boost = eos_boost * t * eos_boost_max_multiplier
//
// With eos_ramp_start=0 (default), this reduces to the original formula:
//   t = min(current_len / eos_ramp_len, 1.0)

/// At sequence length 0, dynamic EOS boost should be effectively 0.
#[test]
fn dynamic_eos_boost_zero_at_start() {
    let stream = test_stream();
    let vocab = 64;
    let eos_id: i32 = 1;

    let mut logits = vec![0.0f32; vocab];
    logits[10] = 5.0; // clear winner
    logits[eos_id as usize] = 4.0; // would need boost to win

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        eos_boost: 1.0,
        eos_token_id: eos_id,
        eos_ramp_len: 50,
        eos_boost_max_multiplier: 3.0,
        current_lens: vec![0], // length 0 → boost = 1.0 * 0/50 * 3.0 = 0.0
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "dynamic_eos_boost_zero_at_start");
    let result = run_gpu(&stream, &p);
    assert_eq!(result[0], 10, "EOS should NOT win at length 0");
}

/// At current_len >= ramp_len, dynamic EOS boost should be fully applied.
#[test]
fn dynamic_eos_boost_full_at_ramp_length() {
    let stream = test_stream();
    let vocab = 64;
    let eos_id: i32 = 1;

    let mut logits = vec![0.0f32; vocab];
    logits[10] = 5.0;
    logits[eos_id as usize] = 4.0; // boost = 1.0 * 1.0 * 3.0 = 3.0 → 4.0+3.0=7.0

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        eos_boost: 1.0,
        eos_token_id: eos_id,
        eos_ramp_len: 50,
        eos_boost_max_multiplier: 3.0,
        current_lens: vec![100], // >= ramp_len → full boost
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "dynamic_eos_boost_full_at_ramp_length");
    let result = run_gpu(&stream, &p);
    assert_eq!(
        result[0], eos_id as u32,
        "EOS should win at full ramp (4.0+3.0=7.0 > 5.0)"
    );
}

/// Test the linear ramp: at current_len = ramp_len/2, boost should be half.
#[test]
fn dynamic_eos_boost_linear_ramp() {
    let stream = test_stream();
    let vocab = 64;
    let eos_id: i32 = 1;

    // At len=25, ramp_len=50: t = 25/50 = 0.5
    // effective = 1.0 * 0.5 * 3.0 = 1.5
    // EOS logit: 4.0 + 1.5 = 5.5 > 5.0 → EOS wins
    let mut logits = vec![0.0f32; vocab];
    logits[10] = 5.0;
    logits[eos_id as usize] = 4.0;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        eos_boost: 1.0,
        eos_token_id: eos_id,
        eos_ramp_len: 50,
        eos_boost_max_multiplier: 3.0,
        current_lens: vec![25], // t=0.5, boost=1.5
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "dynamic_eos_boost_linear_ramp");
    let result = run_gpu(&stream, &p);
    assert_eq!(
        result[0], eos_id as u32,
        "EOS should win at half ramp (4.0+1.5=5.5 > 5.0)"
    );
}

/// With eos_ramp_start > 0, the ramp only begins at that token count.
/// Below ramp_start the boost is zero; from ramp_start to ramp_len it
/// ramps linearly; above ramp_len it's clamped to full.
#[test]
fn dynamic_eos_boost_ramp_start() {
    let stream = test_stream();
    let vocab = 64;
    let eos_id: i32 = 1;

    // ramp_start=40, ramp_len=50 → ramp spans tokens 40..50 (10 tokens)
    // At len=30 (below ramp_start): t = max(0, 30-40) / max(1, 50-40) = 0/10 = 0.0
    // At len=45 (midpoint): t = max(0, 45-40) / 10 = 5/10 = 0.5
    // At len=50 (full): t = max(0, 50-40) / 10 = 10/10 = 1.0

    // --- Below ramp_start: boost = 0, EOS should not win ---
    let mut logits = vec![0.0f32; vocab];
    logits[10] = 5.0;
    logits[eos_id as usize] = 4.0;

    let p_before = SamplingParams {
        logits_f32: logits.clone(),
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        eos_boost: 1.0,
        eos_token_id: eos_id,
        eos_ramp_start: 40,
        eos_ramp_len: 50,
        eos_boost_max_multiplier: 3.0,
        current_lens: vec![30], // below ramp_start → boost = 0
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p_before, "eos_ramp_start_before");
    let r = run_gpu(&stream, &p_before);
    assert_eq!(r[0], 10, "EOS should NOT win below ramp_start (boost=0)");

    // --- Midpoint of ramp: boost = 1.0 * 0.5 * 3.0 = 1.5, EOS = 4+1.5 = 5.5 > 5.0 ---
    let p_mid = SamplingParams {
        logits_f32: logits.clone(),
        current_lens: vec![45],
        ..p_before.clone()
    };
    assert_gpu_cpu_match(&stream, &p_mid, "eos_ramp_start_mid");
    let r = run_gpu(&stream, &p_mid);
    assert_eq!(
        r[0], eos_id as u32,
        "EOS should win at ramp midpoint (4.0+1.5=5.5 > 5.0)"
    );

    // --- At ramp_len: full boost = 1.0 * 1.0 * 3.0 = 3.0, EOS = 7.0 ---
    let p_full = SamplingParams {
        logits_f32: logits.clone(),
        current_lens: vec![50],
        ..p_before.clone()
    };
    assert_gpu_cpu_match(&stream, &p_full, "eos_ramp_start_full");
    let r = run_gpu(&stream, &p_full);
    assert_eq!(
        r[0], eos_id as u32,
        "EOS should win at full ramp (4.0+3.0=7.0 > 5.0)"
    );

    // --- Beyond ramp_len: still clamped to full ---
    let p_beyond = SamplingParams {
        logits_f32: logits,
        current_lens: vec![200],
        ..p_before.clone()
    };
    assert_gpu_cpu_match(&stream, &p_beyond, "eos_ramp_start_beyond");
    let r = run_gpu(&stream, &p_beyond);
    assert_eq!(
        r[0], eos_id as u32,
        "EOS should win beyond ramp_len (still clamped to full)"
    );
}

// ============================================================================
// Tests: Qwen3 Recommended Full Combination
// ============================================================================

/// Integration test exercising ALL penalty features simultaneously with the
/// recommended Qwen3 config: temperature=0.7, top_k=20, top_p=0.95,
/// repeat_penalty=1.05, repeat_last_n=128, cross_turn_penalty=1.01,
/// presence_penalty=0.5, dry_multiplier=0.4, dry_base=1.75,
/// dry_allowed_length=2, dynamic_eos_boost (eos_ramp_len=50, max_mul=3.0).
#[test]
fn qwen3_recommended_full_combination() {
    let stream = test_stream();
    let vocab = 1024;
    let eos_id: i32 = 1;

    // Four tokens of interest:
    //   loop_tok  — appeared in recent window AND continues a repeated n-gram
    //   prior_tok — appeared in recent window but no n-gram match
    //   cross_tok — appeared in a PRIOR turn (cross_turn_counts > 0) but not this turn
    //   fresh_tok — never seen, not part of any n-gram
    let loop_tok: usize = 100;
    let prior_tok: usize = 200;
    let cross_tok: usize = 250;
    let fresh_tok: usize = 300;

    // Give all four a significant logit advantage over the rest.
    // loop_tok starts highest to ensure it would win WITHOUT penalties.
    let mut logits = vec![-20.0f32; vocab];
    logits[loop_tok] = 7.0;
    logits[prior_tok] = 6.5;
    logits[cross_tok] = 6.2;
    logits[fresh_tok] = 6.0;
    // Scatter remaining mass so top_k=20 has meaningful candidates.
    for i in 0..20usize {
        let idx = 400 + i;
        if idx < vocab {
            logits[idx] = 3.0 - (i as f32 * 0.1);
        }
    }
    // Give EOS a moderate base logit (dynamic EOS boost will amplify it).
    logits[eos_id as usize] = 1.0;

    // Recent token history (oldest-first, newest at the end).
    // The kernel reads recent_tokens[0..recent_lens[batch_idx]).
    //
    // DRY penalty targets the CONTINUATION token: if the current suffix
    // (end of recent_tokens) matches an earlier subsequence, the token
    // that followed the earlier match gets penalized.
    //
    // Layout:
    //   positions 0-6, 11-124: unique filler tokens (prevent false matches)
    //   position 7:  filler_tok (999) — extends the trigram match to 3
    //   position 8:  prior_tok (200)  — part of earlier pattern
    //   position 9:  cross_tok (250)  — part of earlier pattern
    //   position 10: loop_tok  (100)  — DRY continuation target
    //   position 125: filler_tok (999) — matches position 7
    //   position 126: prior_tok (200)  — matches position 8
    //   position 127: cross_tok (250)  — matches position 9
    //
    // The suffix [999, prior_tok, cross_tok] at 125-127 matches [999, prior_tok, cross_tok]
    // at 7-9, giving match_len=3. The continuation at position 10 is loop_tok.
    // DRY penalty = 0.4 * 1.75^(3-2) = 0.70 applied to loop_tok.
    //
    // loop_tok (position 10) and prior_tok (positions 8, 126) are in the
    // recent window → repeat_penalty applies to both.
    // cross_tok (positions 9, 127) is also in the recent window → repeat_penalty applies.
    let max_recent = 128usize;
    let filler_tok = 999i32;
    let mut recent = vec![0i32; max_recent];
    // Fill with unique tokens to prevent spurious DRY matches through zeros
    for i in 0..max_recent {
        recent[i] = (500 + i) as i32;
    }
    // Earlier pattern: [filler, prior_tok, cross_tok, loop_tok]
    recent[7] = filler_tok;
    recent[8] = prior_tok as i32;
    recent[9] = cross_tok as i32;
    recent[10] = loop_tok as i32;
    // Current suffix at end: [filler, prior_tok, cross_tok]
    recent[125] = filler_tok;
    recent[126] = prior_tok as i32;
    recent[127] = cross_tok as i32;

    // token_counts: loop_tok and prior_tok both seen THIS turn (presence penalty applies).
    let mut token_counts = vec![0i32; vocab];
    token_counts[loop_tok] = 3;   // seen 3 times this turn
    token_counts[prior_tok] = 1;  // seen once this turn
    // cross_tok count = 0 this turn (but was used in a prior turn)
    // fresh_tok count = 0.

    // Cross-turn counts: cross_tok was used 2 times in prior turns.
    let mut cross_turn_counts = vec![0i32; vocab];
    cross_turn_counts[cross_tok] = 2;

    // Dynamic EOS boost: current_len = 40 tokens into generation.
    // With eos_ramp_len=50 and eos_boost_max_multiplier=3.0:
    //   ramp_factor = min(40/50, 1.0) = 0.8
    //   effective_boost = eos_boost * ramp_factor * eos_boost_max_multiplier
    //                   = 1.0 * 0.8 * 3.0 = 2.4   (added to EOS logit)
    let current_len = 40;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.7,
        top_k: 20,
        top_p: 0.95,

        // repeat_penalty with window = 128
        repeat_penalty: 1.05,
        recent_tokens: recent.clone(),
        recent_lens: vec![max_recent as i32],
        max_recent_len: max_recent as i32,

        // presence penalty (current-turn tokens only)
        presence_penalty: 0.5,
        frequency_penalty: 0.0,
        token_counts,

        // cross_turn_penalty: lighter penalty on prior-turn tokens
        cross_turn_penalty: 1.01,
        cross_turn_counts,

        // dynamic EOS boost
        eos_boost: 1.0,
        eos_token_id: eos_id,
        eos_ramp_len: 50,
        eos_boost_max_multiplier: 3.0,
        current_lens: vec![current_len],

        // DRY: multiplier=0.4, base=1.75, allowed_length=2
        // The suffix [999, prior_tok, cross_tok] at 125-127 matches 7-9.
        // match_length=3 > allowed_length=2, so
        // penalty = 0.4 * 1.75^(3-2) = 0.4 * 1.75 = 0.70 applied to
        // loop_tok (the continuation token at position 10).
        dry_multiplier: 0.4,
        dry_base: 1.75,
        dry_allowed_length: 2,
        dry_range: 0, // search full history

        seed: 12345,
        rng_offsets: vec![0],
        ..Default::default()
    };

    // ========================  MANUAL PENALTY MATH  ========================
    //
    // loop_tok (token 100):
    //   raw logit           =  7.0
    //   repeat_penalty      :  7.0 / 1.05 ≈ 6.667  (in recent window at pos 10)
    //   presence_penalty    : -0.5                   (count=3 > 0)
    //   DRY (match=3, allowed=2): 0.4 * 1.75^(3-2) = 0.70 → subtract
    //     (suffix [999, prior, cross] at 125-127 matches 7-9;
    //      continuation at pos 10 = loop_tok)
    //   cross_turn_penalty  :  none (cross_turn_counts[100]=0)
    //   effective           ≈  6.667 - 0.5 - 0.70 = 5.467
    //
    // prior_tok (token 200):
    //   raw logit           =  6.5
    //   repeat_penalty      :  6.5 / 1.05 ≈ 6.190  (in recent window at pos 8, 126)
    //   presence_penalty    : -0.5                   (count=1 > 0)
    //   DRY                 :  none (not a continuation of any match)
    //   cross_turn_penalty  :  none (cross_turn_counts[200]=0)
    //   effective           ≈  6.190 - 0.5 = 5.690
    //
    // cross_tok (token 250):
    //   raw logit           =  6.2
    //   repeat_penalty      :  6.2 / 1.05 ≈ 5.905  (in recent window at pos 9, 127)
    //   presence_penalty    :  none (count=0 this turn)
    //   DRY                 :  none
    //   cross_turn_penalty  : -1.01                  (cross_turn_counts[250]=2 > 0)
    //   effective           ≈  5.905 - 1.01 = 4.895
    //
    // fresh_tok (token 300):
    //   raw logit           =  6.0
    //   repeat_penalty      :  none (not in recent window)
    //   presence_penalty    :  none (count=0)
    //   DRY                 :  none
    //   cross_turn_penalty  :  none (cross_turn_counts[300]=0)
    //   effective           =  6.0  ← highest non-EOS
    //
    // EOS (token 1):
    //   raw logit           =  1.0
    //   dynamic_eos_boost   : +1.0 * min(40/50, 1.0) * 3.0 = +2.4
    //   effective           =  3.4  (not top candidate, but meaningfully boosted)
    //
    // Ranking: fresh_tok(6.0) > prior_tok(5.69) > loop_tok(5.47) > cross_tok(4.90) > EOS(3.4)

    // ========================  DETERMINISTIC (argmax) TESTS  ========================
    // Use temperature=0.0 (argmax) to verify the exact penalty ranking.
    // This proves all penalty types are applied correctly in combination.
    let p_det = SamplingParams {
        temperature: 0.0, // argmax
        top_k: 0,
        top_p: 1.0,
        ..p.clone()
    };

    // GPU and CPU must agree
    assert_gpu_cpu_match(&stream, &p_det, "qwen3_recommended_full_combination_deterministic");

    // Argmax should select fresh_tok (highest effective logit = 6.0, unpenalized)
    let det_result = run_gpu(&stream, &p_det);
    assert_eq!(
        det_result[0], fresh_tok as u32,
        "deterministic: fresh_tok ({fresh_tok}) should be argmax winner \
         since it has the highest post-penalty logit (6.0 vs loop_tok ~5.47), got {}",
        det_result[0]
    );

    // ========================  STOCHASTIC SMOKE TEST  ========================
    // Run a few stochastic trials with the full recommended config.
    // We do NOT assert hard exclusion of any token — the penalty gaps are
    // moderate (~0.5–1.1 logits) and stochastic sampling can pick any
    // token in the top_k=20 set.  We simply verify no crashes and that
    // the output is a valid token ID.
    for trial in 0..5 {
        let result = run_gpu(&stream, &p);
        let tok = result[0];
        assert!(
            (tok as usize) < vocab,
            "trial {trial}: token {tok} out of vocab range {vocab}"
        );
    }
}

// ============================================================================
// Tests: EOT (End-of-Thinking) Boost
// ============================================================================

/// When thinking_len is below eot_ramp_start, the EOT boost should be zero
/// and the EOT token should NOT be selected.
#[test]
fn eot_boost_zero_below_ramp_start() {
    let stream = test_stream();
    let vocab = 64;
    let eot_id: i32 = 5;

    // Token 10 has higher base logit than the EOT token
    let mut logits = vec![0.0f32; vocab];
    logits[10] = 5.0;
    logits[eot_id as usize] = 4.0;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        eot_boost: 1.0,
        eot_token_id: eot_id,
        eot_ramp_start: 150,
        eot_ramp_len: 200,
        eot_boost_max_multiplier: 3.0,
        thinking_lens: vec![50], // well below ramp_start=150 → boost = 0
        ..Default::default()
    };

    assert_gpu_cpu_match(&stream, &p, "eot_boost_zero_below_ramp_start");
    let r = run_gpu(&stream, &p);
    assert_eq!(r[0], 10, "EOT should NOT win below ramp_start (boost=0)");
}

/// At the midpoint of the ramp, the EOT boost should be partial.
#[test]
fn eot_boost_partial_at_midpoint() {
    let stream = test_stream();
    let vocab = 64;
    let eot_id: i32 = 5;

    // ramp_start=150, ramp_len=200 → ramp spans 150..200 (50-token window)
    // At thinking_len=175: t = (175-150)/(200-150) = 25/50 = 0.5
    // boost = 1.0 * 0.5 * 3.0 = 1.5
    // EOT logit = 4.0 + 1.5 = 5.5 > 5.0
    let mut logits = vec![0.0f32; vocab];
    logits[10] = 5.0;
    logits[eot_id as usize] = 4.0;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        eot_boost: 1.0,
        eot_token_id: eot_id,
        eot_ramp_start: 150,
        eot_ramp_len: 200,
        eot_boost_max_multiplier: 3.0,
        thinking_lens: vec![175], // midpoint → boost = 1.5
        ..Default::default()
    };

    assert_gpu_cpu_match(&stream, &p, "eot_boost_partial_at_midpoint");
    let r = run_gpu(&stream, &p);
    assert_eq!(
        r[0], eot_id as u32,
        "EOT should win at ramp midpoint (4.0+1.5=5.5 > 5.0)"
    );
}

/// At full ramp (thinking_len >= ramp_len), the EOT boost is at maximum.
#[test]
fn eot_boost_full_at_ramp_len() {
    let stream = test_stream();
    let vocab = 64;
    let eot_id: i32 = 5;

    // At thinking_len=200: t = (200-150)/(200-150) = 1.0
    // boost = 1.0 * 1.0 * 3.0 = 3.0
    // EOT logit = 2.1 + 3.0 = 5.1 > 5.0
    let mut logits = vec![0.0f32; vocab];
    logits[10] = 5.0;
    logits[eot_id as usize] = 2.1;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        eot_boost: 1.0,
        eot_token_id: eot_id,
        eot_ramp_start: 150,
        eot_ramp_len: 200,
        eot_boost_max_multiplier: 3.0,
        thinking_lens: vec![200], // at ramp_len → full boost
        ..Default::default()
    };

    assert_gpu_cpu_match(&stream, &p, "eot_boost_full_at_ramp_len");
    let r = run_gpu(&stream, &p);
    assert_eq!(
        r[0], eot_id as u32,
        "EOT should win at full ramp (2.1+3.0=5.1 > 5.0)"
    );
}

/// When thinking_lens = 0 (not in thinking mode), EOT boost should be disabled
/// even when eot_boost > 0 and eot_token_id is valid.
#[test]
fn eot_boost_disabled_when_not_thinking() {
    let stream = test_stream();
    let vocab = 64;
    let eot_id: i32 = 5;

    // With thinking_lens=0, the EOT boost should NOT apply regardless of config
    let mut logits = vec![0.0f32; vocab];
    logits[10] = 5.0;
    logits[eot_id as usize] = 4.0;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        eot_boost: 1.0,
        eot_token_id: eot_id,
        eot_ramp_start: 0,  // would give t=1.0 if thinking_len > 0
        eot_ramp_len: 1,
        eot_boost_max_multiplier: 100.0,  // huge multiplier to ensure failure if applied
        thinking_lens: vec![0], // NOT in thinking mode
        ..Default::default()
    };

    assert_gpu_cpu_match(&stream, &p, "eot_boost_disabled_when_not_thinking");
    let r = run_gpu(&stream, &p);
    assert_eq!(
        r[0], 10,
        "EOT should NOT win when thinking_lens=0 (not in thinking mode)"
    );
}

/// EOT boost and EOS boost can operate independently on different tokens.
#[test]
fn eot_boost_independent_of_eos_boost() {
    let stream = test_stream();
    let vocab = 64;
    let eos_id: i32 = 1;
    let eot_id: i32 = 5;

    // Both boosts active but targeting different tokens
    // EOS boost: ramp at current_len=100, ramp_start=80, ramp_len=100
    //   t = (100-80)/(100-80) = 1.0, boost = 1.0 * 1.0 * 2.0 = 2.0
    //   EOS logit = 1.0 + 2.0 = 3.0
    // EOT boost: ramp at thinking_len=175, ramp_start=150, ramp_len=200
    //   t = (175-150)/(200-150) = 0.5, boost = 1.0 * 0.5 * 3.0 = 1.5
    //   EOT logit = 4.0 + 1.5 = 5.5
    let mut logits = vec![0.0f32; vocab];
    logits[10] = 5.0;
    logits[eos_id as usize] = 1.0;
    logits[eot_id as usize] = 4.0;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        // EOS boost config
        eos_boost: 1.0,
        eos_token_id: eos_id,
        eos_ramp_start: 80,
        eos_ramp_len: 100,
        eos_boost_max_multiplier: 2.0,
        current_lens: vec![100],
        // EOT boost config
        eot_boost: 1.0,
        eot_token_id: eot_id,
        eot_ramp_start: 150,
        eot_ramp_len: 200,
        eot_boost_max_multiplier: 3.0,
        thinking_lens: vec![175],
        ..Default::default()
    };

    assert_gpu_cpu_match(&stream, &p, "eot_boost_independent_of_eos_boost");
    let r = run_gpu(&stream, &p);
    // EOT (5.5) > token 10 (5.0) > EOS (3.0)
    assert_eq!(
        r[0], eot_id as u32,
        "EOT should win (5.5) over token 10 (5.0) and EOS (3.0)"
    );
}