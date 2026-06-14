//! Dtype-specific tests for the batched sampling kernel.
//!
//! Tests for f16, bf16, fp8 (e4m3) logit types, diagnostic tests,
//! and cross-dtype agreement tests.

#![cfg(feature = "cuda")]

#[allow(dead_code)]
mod sampling_harness;
use sampling_harness::*;

#[test]
fn f16_argmax_basic() {
    let stream = test_stream();
    let vocab = 1024;
    let peak = 500;
    let logits = make_peaked_logits(vocab, peak);
    let (typed, quantised) = f32_to_f16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::F16 as i32,
        "f16_argmax_basic",
    );
}

#[test]
fn f16_argmax_large_vocab() {
    let stream = test_stream();
    let vocab = 128_000;
    let peak = 99_999;
    let logits = make_peaked_logits(vocab, peak);
    let (typed, quantised) = f32_to_f16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::F16 as i32,
        "f16_argmax_large_vocab",
    );
}

#[test]
fn f16_argmax_batched() {
    let stream = test_stream();
    let vocab = 512;
    let batch = 8;
    let peaks = [0, 50, 100, 255, 300, 400, 500, 511];
    let mut logits = Vec::with_capacity(batch * vocab);
    for &peak in &peaks {
        logits.extend(make_peaked_logits(vocab, peak));
    }
    let (typed, quantised) = f32_to_f16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: batch as i32,
        vocab_size: vocab as i32,
        temperature: 0.0,
        rng_offsets: vec![0u64; batch],
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::F16 as i32,
        "f16_argmax_batched",
    );
}

#[test]
fn f16_with_penalties() {
    let stream = test_stream();
    let vocab = 256;
    let mut logits = vec![0.0f32; vocab];
    logits[10] = 5.0;
    logits[11] = 4.9;
    let recent = vec![10i32];
    let (typed, quantised) = f32_to_f16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        repeat_penalty: 10.0,
        recent_tokens: recent,
        recent_lens: vec![1],
        max_recent_len: 1,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::F16 as i32,
        "f16_with_penalties",
    );
}

#[test]
fn diag_f32_stochastic_topk_extreme_logits() {
    // Diagnose: does f32 stochastic + top_k fail with extreme logit gap?
    let stream = test_stream();
    let vocab = 256;
    let mut logits = vec![-100.0f32; vocab];
    logits[10] = 50.0;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 1.0,
        top_k: 5,
        seed: 42,
        rng_offsets: vec![0],
        ..Default::default()
    };
    let r = run_gpu(&stream, &p);
    eprintln!("diag f32 extreme gap top_k=5: token {}", r[0]);
    assert_eq!(r[0], 10);
}

#[test]
fn diag_f32_stochastic_topk_moderate_logits() {
    // Same as above but with moderate logit gap (like the passing test)
    let stream = test_stream();
    let vocab = 256;
    let mut logits = vec![-10.0f32; vocab];
    logits[10] = 5.0;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 1.0,
        top_k: 5,
        seed: 42,
        rng_offsets: vec![0],
        ..Default::default()
    };
    let r = run_gpu(&stream, &p);
    eprintln!("diag f32 moderate gap top_k=5: token {}", r[0]);
    // Token 10 has overwhelming probability, should almost always be picked
    assert_eq!(r[0], 10);
}

#[test]
fn diag_f16_stochastic_no_topk() {
    // f16 stochastic with NO top_k (bypasses radix select entirely)
    let stream = test_stream();
    let vocab = 256;
    let mut logits = vec![-100.0f32; vocab];
    logits[10] = 50.0;
    let (typed, _) = f32_to_f16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 1.0,
        top_k: 0,
        seed: 42,
        rng_offsets: vec![0],
        ..Default::default()
    };
    let r = run_gpu_typed(&stream, &typed, &p, DType::F16 as i32);
    eprintln!("diag f16 no top_k: token {}", r[0]);
    assert_eq!(r[0], 10);
}

#[test]
fn diag_f16_stochastic_topk1() {
    // f16 stochastic with top_k=1 (should behave like argmax)
    let stream = test_stream();
    let vocab = 256;
    let mut logits = vec![-100.0f32; vocab];
    logits[10] = 50.0;
    let (typed, _) = f32_to_f16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 1.0,
        top_k: 1,
        seed: 42,
        rng_offsets: vec![0],
        ..Default::default()
    };
    let r = run_gpu_typed(&stream, &typed, &p, DType::F16 as i32);
    eprintln!("diag f16 top_k=1: token {}", r[0]);
    assert_eq!(r[0], 10);
}

#[test]
fn diag_f16_stochastic_topk_moderate() {
    // f16 stochastic with top_k=5 and moderate logit gap
    let stream = test_stream();
    let vocab = 256;
    let mut logits = vec![-10.0f32; vocab];
    logits[10] = 5.0;
    let (typed, _) = f32_to_f16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 1.0,
        top_k: 5,
        seed: 42,
        rng_offsets: vec![0],
        ..Default::default()
    };
    let r = run_gpu_typed(&stream, &typed, &p, DType::F16 as i32);
    eprintln!("diag f16 moderate gap top_k=5: token {}", r[0]);
    assert_eq!(r[0], 10);
}

#[test]
fn diag_f32_topk_vocab_sweep() {
    // Test different vocab sizes to find the boundary
    let stream = test_stream();
    for vocab in [128, 256, 257, 512, 1024] {
        let mut logits = vec![-10.0f32; vocab];
        logits[10] = 5.0;

        let p = SamplingParams {
            logits_f32: logits,
            batch_size: 1,
            vocab_size: vocab as i32,
            temperature: 1.0,
            top_k: 5,
            seed: 42,
            rng_offsets: vec![0],
            ..Default::default()
        };
        let r = run_gpu(&stream, &p);
        eprintln!("diag f32 vocab={}: token {}", vocab, r[0]);
    }
}

#[test]
fn f16_with_temperature_and_topk() {
    let stream = test_stream();
    let vocab = 256;
    let mut logits = vec![-100.0f32; vocab];
    logits[10] = 50.0; // overwhelming winner even in f16
    let (typed, _quantised) = f32_to_f16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 1.0,
        top_k: 5,
        seed: 42,
        rng_offsets: vec![0],
        ..Default::default()
    };
    // With such a dominant logit (prob â‰ˆ 1.0), every seed must pick token 10.
    // This also tests that collect_above_threshold correctly prioritises the
    // high-probability token over zero-probability tokens at the threshold.
    for offset in 0..10u64 {
        let mut pp = p.clone();
        pp.rng_offsets = vec![offset];
        let gpu = run_gpu_typed(&stream, &typed, &pp, DType::F16 as i32);
        assert_eq!(
            gpu[0], 10,
            "f16_topk dominant logit should always win, offset={offset}"
        );
    }
}

#[test]
fn f16_max_representable_value() {
    // f16 max = 65504.0. Test logits at and near the f16 boundary.
    let stream = test_stream();
    let vocab = 64;
    let mut logits = vec![0.0f32; vocab];
    logits[0] = 65504.0; // f16 max
    logits[1] = 65500.0; // just below max
    logits[2] = 60000.0;

    let (typed, quantised) = f32_to_f16(&logits);
    // Verify the conversion didn't saturate to Inf
    assert!(
        quantised[0].is_finite(),
        "65504.0 should be representable in f16"
    );

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::F16 as i32,
        "f16_max_value",
    );
}

#[test]
fn f16_overflow_to_inf() {
    // Values > 65504 overflow to Inf in f16. The kernel's load_as_float converts
    // to f32, so Inf stays Inf. Argmax should pick the first Inf token.
    let stream = test_stream();
    let vocab = 64;
    let mut logits = vec![0.0f32; vocab];
    logits[5] = 70000.0; // overflows f16 â†’ Inf
    logits[10] = 65504.0; // f16 max, finite

    let (typed, quantised) = f32_to_f16(&logits);
    assert!(
        quantised[5].is_infinite(),
        "70000.0 should overflow to Inf in f16"
    );
    assert!(quantised[10].is_finite(), "65504.0 should be finite in f16");

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::F16 as i32,
        "f16_overflow_inf",
    );
}

#[test]
fn f16_precision_loss_changes_winner() {
    // f16 has ~3.3 decimal digits of precision. Two logits that differ by less
    // than the f16 epsilon at their magnitude will quantise to the same value.
    // The argmax should then prefer the lower-indexed token (tie-breaking).
    let stream = test_stream();
    let vocab = 64;
    let mut logits = vec![0.0f32; vocab];
    // At magnitude ~1024, f16 epsilon is 1.0. So 1024.0 and 1024.5 both
    // round to 1024.0 in f16.
    logits[10] = 1024.5;
    logits[20] = 1024.0;

    let (typed, quantised) = f32_to_f16(&logits);
    // Both should quantise to the same value in f16
    let same = quantised[10] == quantised[20];

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    let gpu = run_gpu_typed(&stream, &typed, &p, DType::F16 as i32);

    if same {
        // Tie: GPU parallel reduction may pick either tied token
        assert!(
            gpu[0] == 10 || gpu[0] == 20,
            "f16_precision_tie: should pick one of the tied tokens, got {}",
            gpu[0]
        );
    } else {
        // If they didn't quantise the same, higher one wins
        assert_typed_gpu_cpu_match(
            &stream,
            &typed,
            &quantised,
            &p,
            DType::F16 as i32,
            "f16_precision_distinct",
        );
    }
}

#[test]
fn f16_small_values_near_min_normal() {
    // f16 smallest normal â‰ˆ 6.1e-5. Subnormals go down to ~5.96e-8.
    // Test logits near these boundaries.
    let stream = test_stream();
    let vocab = 64;
    let mut logits = vec![-1.0f32; vocab]; // all slightly negative
    logits[0] = 6.1e-5; // near f16 min normal
    logits[1] = 1.0e-7; // subnormal territory for f16
    logits[2] = 0.001; // well above min normal

    let (typed, quantised) = f32_to_f16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::F16 as i32,
        "f16_small_values",
    );
}

#[test]
fn f16_negative_values() {
    let stream = test_stream();
    let vocab = 128;
    let mut logits: Vec<f32> = (0..vocab).map(|i| -100.0 - i as f32).collect();
    logits[77] = -0.5;
    let (typed, quantised) = f32_to_f16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::F16 as i32,
        "f16_negative_values",
    );
}

#[test]
fn f16_banned_tokens() {
    let stream = test_stream();
    let vocab = 256;
    let mut logits = vec![0.0f32; vocab];
    logits[10] = 10.0;
    logits[20] = 9.0;
    logits[30] = 8.0;
    let banned = vec![10i32, 20, -1];
    let (typed, quantised) = f32_to_f16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        banned_tokens: banned,
        num_banned_tokens: 2,
        banned_tokens_per_seq: 0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::F16 as i32,
        "f16_banned_tokens",
    );
}

#[test]
fn f16_stencil() {
    let stream = test_stream();
    let vocab = 1024;
    let mut logits = vec![0.0f32; vocab];
    logits[500] = 100.0;
    logits[10] = 5.0;
    logits[20] = 4.0;
    let stencil = vec![10i32, 20, 30];
    let (typed, quantised) = f32_to_f16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        stencil,
        stencil_size: 3,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::F16 as i32,
        "f16_stencil",
    );
}

// ============================================================================
// Tests: BF16 Dtype
// ============================================================================

#[test]
fn bf16_argmax_basic() {
    let stream = test_stream();
    let vocab = 1024;
    let peak = 700;
    let logits = make_peaked_logits(vocab, peak);
    let (typed, quantised) = f32_to_bf16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::BF16 as i32,
        "bf16_argmax_basic",
    );
}

#[test]
fn bf16_argmax_large_vocab() {
    let stream = test_stream();
    let vocab = 128_000;
    let peak = 63_999;
    let logits = make_peaked_logits(vocab, peak);
    let (typed, quantised) = f32_to_bf16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::BF16 as i32,
        "bf16_argmax_large_vocab",
    );
}

#[test]
fn bf16_argmax_batched() {
    let stream = test_stream();
    let vocab = 512;
    let batch = 8;
    let peaks = [0, 50, 100, 255, 300, 400, 500, 511];
    let mut logits = Vec::with_capacity(batch * vocab);
    for &peak in &peaks {
        logits.extend(make_peaked_logits(vocab, peak));
    }
    let (typed, quantised) = f32_to_bf16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: batch as i32,
        vocab_size: vocab as i32,
        temperature: 0.0,
        rng_offsets: vec![0u64; batch],
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::BF16 as i32,
        "bf16_argmax_batched",
    );
}

#[test]
fn bf16_with_penalties() {
    let stream = test_stream();
    let vocab = 256;
    let mut logits = vec![0.0f32; vocab];
    logits[10] = 5.0;
    logits[11] = 4.9;
    let recent = vec![10i32];
    let (typed, quantised) = f32_to_bf16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        repeat_penalty: 10.0,
        recent_tokens: recent,
        recent_lens: vec![1],
        max_recent_len: 1,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::BF16 as i32,
        "bf16_with_penalties",
    );
}

#[test]
fn bf16_precision_loss_changes_winner() {
    // bf16 has only ~2.4 decimal digits (7 mantissa bits + implicit).
    // At magnitude 256, bf16 epsilon is 2.0. So 256.0 and 257.0 map to 256.
    let stream = test_stream();
    let vocab = 64;
    let mut logits = vec![0.0f32; vocab];
    logits[10] = 257.0;
    logits[20] = 256.0;

    let (typed, quantised) = f32_to_bf16(&logits);
    let same = quantised[10] == quantised[20];

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    let gpu = run_gpu_typed(&stream, &typed, &p, DType::BF16 as i32);

    if same {
        // Tie: GPU parallel reduction may pick either tied token
        assert!(
            gpu[0] == 10 || gpu[0] == 20,
            "bf16_precision_tie: should pick one of the tied tokens, got {}",
            gpu[0]
        );
    } else {
        assert_typed_gpu_cpu_match(
            &stream,
            &typed,
            &quantised,
            &p,
            DType::BF16 as i32,
            "bf16_precision_distinct",
        );
    }
}

#[test]
fn bf16_large_magnitude() {
    // bf16 has the same exponent range as f32 (8 bits), so it can represent
    // values up to ~3.4e38. Test large logits.
    let stream = test_stream();
    let vocab = 64;
    let mut logits = vec![0.0f32; vocab];
    logits[0] = 1.0e30;
    logits[1] = 1.0e20;
    logits[2] = 1.0e10;
    let (typed, quantised) = f32_to_bf16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::BF16 as i32,
        "bf16_large_magnitude",
    );
}

#[test]
fn bf16_frequency_penalty() {
    let stream = test_stream();
    let vocab = 64;
    let mut logits = vec![0.0f32; vocab];
    logits[5] = 10.0;
    logits[6] = 9.5;
    let mut token_counts = vec![0i32; vocab];
    token_counts[5] = 3;
    let (typed, quantised) = f32_to_bf16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        frequency_penalty: 2.0,
        token_counts,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::BF16 as i32,
        "bf16_frequency_penalty",
    );
}

#[test]
fn bf16_banned_tokens() {
    let stream = test_stream();
    let vocab = 256;
    let mut logits = vec![0.0f32; vocab];
    logits[10] = 10.0;
    logits[20] = 9.0;
    logits[30] = 8.0;
    let banned = vec![10i32, 20, -1];
    let (typed, quantised) = f32_to_bf16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        banned_tokens: banned,
        num_banned_tokens: 2,
        banned_tokens_per_seq: 0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::BF16 as i32,
        "bf16_banned_tokens",
    );
}

#[test]
fn bf16_stencil() {
    let stream = test_stream();
    let vocab = 1024;
    let mut logits = vec![0.0f32; vocab];
    logits[500] = 100.0;
    logits[10] = 5.0;
    logits[20] = 4.0;
    let stencil = vec![10i32, 20, 30];
    let (typed, quantised) = f32_to_bf16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        stencil,
        stencil_size: 3,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::BF16 as i32,
        "bf16_stencil",
    );
}

#[test]
fn bf16_with_temperature_and_topk() {
    let stream = test_stream();
    let vocab = 256;
    let mut logits = vec![-100.0f32; vocab];
    logits[10] = 50.0;
    let (typed, _quantised) = f32_to_bf16(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 1.0,
        top_k: 5,
        seed: 42,
        rng_offsets: vec![0],
        ..Default::default()
    };
    // With such a dominant logit (prob â‰ˆ 1.0), every seed must pick token 10.
    // This also tests that collect_above_threshold correctly prioritises the
    // high-probability token over zero-probability tokens at the threshold.
    for offset in 0..10u64 {
        let mut pp = p.clone();
        pp.rng_offsets = vec![offset];
        let gpu = run_gpu_typed(&stream, &typed, &pp, DType::BF16 as i32);
        assert_eq!(
            gpu[0], 10,
            "bf16_topk dominant logit should always win, offset={offset}"
        );
    }
}

// ============================================================================
// Tests: FP8 E4M3 Dtype
// ============================================================================

#[test]
fn fp8_argmax_basic() {
    let stream = test_stream();
    let vocab = 256;
    let peak = 100;
    // fp8 e4m3 max = 448.0. Use logits well within range.
    let mut logits = vec![0.0f32; vocab];
    for (i, v) in logits.iter_mut().enumerate() {
        *v = -(((i as i64 - peak as i64).abs()) as f32);
    }
    logits[peak] = 10.0;
    let (typed, quantised) = f32_to_fp8(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::FP8E4M3 as i32,
        "fp8_argmax_basic",
    );
}

#[test]
fn fp8_argmax_batched() {
    let stream = test_stream();
    let vocab = 128;
    let batch = 4;
    let peaks = [0, 30, 64, 127];
    let mut logits = Vec::with_capacity(batch * vocab);
    for &peak in &peaks {
        let mut l = vec![0.0f32; vocab];
        l[peak] = 10.0;
        logits.extend(l);
    }
    let (typed, quantised) = f32_to_fp8(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: batch as i32,
        vocab_size: vocab as i32,
        temperature: 0.0,
        rng_offsets: vec![0u64; batch],
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::FP8E4M3 as i32,
        "fp8_argmax_batched",
    );
}

#[test]
fn fp8_max_representable_value() {
    // fp8 e4m3 max = 448.0. Test at the boundary.
    let stream = test_stream();
    let vocab = 32;
    let mut logits = vec![0.0f32; vocab];
    logits[0] = 448.0;
    logits[1] = 440.0;
    logits[2] = 400.0;
    let (typed, quantised) = f32_to_fp8(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::FP8E4M3 as i32,
        "fp8_max_value",
    );
}

#[test]
fn fp8_coarse_precision_quantisation() {
    // fp8 e4m3 has only 3 mantissa bits â†’ precision of 1 part in 8 at each exponent.
    // At magnitude 64, representable values are 60, 62, 64, 68, 72, ...
    // Test that two logits that round to the same fp8 value produce consistent results.
    let stream = test_stream();
    let vocab = 32;
    let mut logits = vec![0.0f32; vocab];
    logits[5] = 65.0; // might round to 64 in fp8
    logits[10] = 64.0;

    let (typed, quantised) = f32_to_fp8(&logits);
    let same = quantised[5] == quantised[10];

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    let gpu = run_gpu_typed(&stream, &typed, &p, DType::FP8E4M3 as i32);

    if same {
        // Tie: GPU parallel reduction may pick either tied token
        assert!(
            gpu[0] == 5 || gpu[0] == 10,
            "fp8_precision_tie: should pick one of the tied tokens, got {}",
            gpu[0]
        );
    } else {
        assert_typed_gpu_cpu_match(
            &stream,
            &typed,
            &quantised,
            &p,
            DType::FP8E4M3 as i32,
            "fp8_precision_distinct",
        );
    }
}

#[test]
fn fp8_small_positive_values() {
    // fp8 e4m3 smallest subnormal â‰ˆ 2^-9 = 0.001953125
    let stream = test_stream();
    let vocab = 32;
    let mut logits = vec![-10.0f32; vocab]; // all negative
    logits[0] = 0.01; // small but representable in fp8
    logits[1] = 0.001; // may flush to subnormal or zero in fp8

    let (typed, quantised) = f32_to_fp8(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::FP8E4M3 as i32,
        "fp8_small_positive",
    );
}

#[test]
fn fp8_banned_tokens() {
    let stream = test_stream();
    let vocab = 64;
    let mut logits = vec![0.0f32; vocab];
    logits[10] = 100.0;
    logits[20] = 50.0;
    logits[30] = 40.0;
    let banned = vec![10i32, 20, -1];
    let (typed, quantised) = f32_to_fp8(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        banned_tokens: banned,
        num_banned_tokens: 2,
        banned_tokens_per_seq: 0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::FP8E4M3 as i32,
        "fp8_banned_tokens",
    );
}

#[test]
fn fp8_with_penalties() {
    let stream = test_stream();
    let vocab = 64;
    let mut logits = vec![0.0f32; vocab];
    logits[10] = 10.0;
    logits[11] = 9.0;
    let recent = vec![10i32];
    let (typed, quantised) = f32_to_fp8(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        repeat_penalty: 5.0,
        recent_tokens: recent,
        recent_lens: vec![1],
        max_recent_len: 1,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::FP8E4M3 as i32,
        "fp8_with_penalties",
    );
}

#[test]
fn fp8_negative_logits() {
    // fp8 e4m3 supports negative values via sign bit
    let stream = test_stream();
    let vocab = 64;
    let mut logits: Vec<f32> = (0..vocab).map(|i| -(i as f32) - 1.0).collect();
    logits[30] = -0.25;
    let (typed, quantised) = f32_to_fp8(&logits);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_typed_gpu_cpu_match(
        &stream,
        &typed,
        &quantised,
        &p,
        DType::FP8E4M3 as i32,
        "fp8_negative_logits",
    );
}

// ============================================================================
// Tests: Precision Edge Cases (F32 â€” testing kernel robustness)
// ============================================================================

#[test]
fn all_dtypes_agree_on_clear_winner() {
    // When the peak logit is far enough from the second that no precision loss
    // can change the winner, all dtypes should agree.
    let stream = test_stream();
    let vocab = 256;
    let peak = 42;
    let mut logits = vec![0.0f32; vocab];
    logits[peak] = 100.0; // well within all dtype ranges, huge gap to second

    let p = SamplingParams {
        logits_f32: logits.clone(),
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };

    // F32
    let gpu_f32 = run_gpu(&stream, &p);
    assert_eq!(gpu_f32[0], peak as u32, "f32 should pick peak");

    // F16
    let (f16_typed, _) = f32_to_f16(&logits);
    let gpu_f16 = run_gpu_typed(&stream, &f16_typed, &p, DType::F16 as i32);
    assert_eq!(gpu_f16[0], peak as u32, "f16 should pick peak");

    // BF16
    let (bf16_typed, _) = f32_to_bf16(&logits);
    let gpu_bf16 = run_gpu_typed(&stream, &bf16_typed, &p, DType::BF16 as i32);
    assert_eq!(gpu_bf16[0], peak as u32, "bf16 should pick peak");

    // FP8
    let (fp8_typed, _) = f32_to_fp8(&logits);
    let gpu_fp8 = run_gpu_typed(&stream, &fp8_typed, &p, DType::FP8E4M3 as i32);
    assert_eq!(gpu_fp8[0], peak as u32, "fp8 should pick peak");
}

#[test]
fn all_dtypes_agree_with_banned_and_stencil() {
    let stream = test_stream();
    let vocab = 128;
    let mut logits = vec![0.0f32; vocab];
    logits[10] = 50.0; // banned
    logits[20] = 40.0; // in stencil, should win
    logits[30] = 35.0; // in stencil
    logits[50] = 45.0; // NOT in stencil

    let stencil = vec![20i32, 30, 40];
    let banned = vec![10i32, -1];

    let p = SamplingParams {
        logits_f32: logits.clone(),
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        stencil,
        stencil_size: 3,
        banned_tokens: banned,
        num_banned_tokens: 1,
        banned_tokens_per_seq: 0,
        ..Default::default()
    };

    let gpu_f32 = run_gpu(&stream, &p);
    assert_eq!(gpu_f32[0], 20, "f32: stencil winner");

    let (f16_typed, _) = f32_to_f16(&logits);
    let gpu_f16 = run_gpu_typed(&stream, &f16_typed, &p, DType::F16 as i32);
    assert_eq!(gpu_f16[0], 20, "f16: stencil winner");

    let (bf16_typed, _) = f32_to_bf16(&logits);
    let gpu_bf16 = run_gpu_typed(&stream, &bf16_typed, &p, DType::BF16 as i32);
    assert_eq!(gpu_bf16[0], 20, "bf16: stencil winner");

    let (fp8_typed, _) = f32_to_fp8(&logits);
    let gpu_fp8 = run_gpu_typed(&stream, &fp8_typed, &p, DType::FP8E4M3 as i32);
    assert_eq!(gpu_fp8[0], 20, "fp8: stencil winner");
}

#[test]
fn all_dtypes_batched_multi_peak() {
    let stream = test_stream();
    let vocab = 128;
    let batch = 4;
    let peaks = [10, 50, 90, 120];

    let mut logits = Vec::with_capacity(batch * vocab);
    for &peak in &peaks {
        let mut l = vec![0.0f32; vocab];
        l[peak] = 80.0; // well within fp8 range (max 448)
        logits.extend(l);
    }

    let p = SamplingParams {
        logits_f32: logits.clone(),
        batch_size: batch as i32,
        vocab_size: vocab as i32,
        temperature: 0.0,
        rng_offsets: vec![0u64; batch],
        ..Default::default()
    };

    // F32
    let gpu_f32 = run_gpu(&stream, &p);
    for (i, &peak) in peaks.iter().enumerate() {
        assert_eq!(gpu_f32[i], peak as u32, "f32 batch[{i}]");
    }

    // F16
    let (f16_typed, _) = f32_to_f16(&logits);
    let gpu_f16 = run_gpu_typed(&stream, &f16_typed, &p, DType::F16 as i32);
    for (i, &peak) in peaks.iter().enumerate() {
        assert_eq!(gpu_f16[i], peak as u32, "f16 batch[{i}]");
    }

    // BF16
    let (bf16_typed, _) = f32_to_bf16(&logits);
    let gpu_bf16 = run_gpu_typed(&stream, &bf16_typed, &p, DType::BF16 as i32);
    for (i, &peak) in peaks.iter().enumerate() {
        assert_eq!(gpu_bf16[i], peak as u32, "bf16 batch[{i}]");
    }

    // FP8
    let (fp8_typed, _) = f32_to_fp8(&logits);
    let gpu_fp8 = run_gpu_typed(&stream, &fp8_typed, &p, DType::FP8E4M3 as i32);
    for (i, &peak) in peaks.iter().enumerate() {
        assert_eq!(gpu_fp8[i], peak as u32, "fp8 batch[{i}]");
    }
}
