//! Sampling and edge-case tests for the batched sampling kernel.
//!
//! Tests for temperature, top_k, top_p, stochastic sampling,
//! vocab size edge cases, and f32 numerical edge cases.

#![cfg(feature = "cuda")]
// Test code: loop indices are vocab/batch coordinates; casts match the kernel ABI.
#![allow(clippy::needless_range_loop, clippy::unnecessary_cast)]

#[allow(dead_code)]
mod sampling_harness;
use sampling_harness::*;

#[test]
fn temperature_does_not_change_argmax() {
    // With temperature > 0 but a clear peak, the kernel should still sample
    // the peak token. We use a very dominant logit to make this deterministic.
    let stream = test_stream();
    let vocab = 256;
    let peak = 100;
    let mut logits = vec![-100.0f32; vocab]; // all very negative
    logits[peak] = 50.0; // overwhelming winner

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 1.0,
        seed: 12345,
        rng_offsets: vec![0],
        ..Default::default()
    };

    // Run multiple times with different RNG offsets â€” should always pick peak
    for offset in 0..20u64 {
        let mut pp = p.clone();
        pp.rng_offsets = vec![offset];
        let gpu = run_gpu(&stream, &pp);
        assert_eq!(
            gpu[0], peak as u32,
            "temp=1.0 with dominant logit should always pick peak, offset={offset}"
        );
    }
}

// ============================================================================
// Tests: Top-K Filtering
// ============================================================================

#[test]
fn topk_selects_from_top_candidates() {
    let stream = test_stream();
    let vocab = 1024;
    // Create logits where tokens 10,11,12 are the top-3
    let mut logits = vec![-10.0f32; vocab];
    logits[10] = 5.0;
    logits[11] = 4.0;
    logits[12] = 3.0;

    // With argmax + top_k=3, should pick token 10
    let p = SamplingParams {
        logits_f32: logits.clone(),
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0, // argmax
        top_k: 3,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "topk_argmax");

    // With temperature > 0 and top_k=3, should only pick from {10, 11, 12}
    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 1.0,
        top_k: 3,
        seed: 42,
        rng_offsets: vec![0],
        ..Default::default()
    };
    let allowed = [10u32, 11, 12];
    for offset in 0..50u64 {
        let mut pp = p.clone();
        pp.rng_offsets = vec![offset];
        assert_valid_token(
            &stream,
            &pp,
            &format!("topk_stochastic_offset{offset}"),
            Some(&allowed),
        );
    }
}

#[test]
fn topk_1_equals_argmax() {
    let stream = test_stream();
    let vocab = 512;
    let peak = 333;
    let logits = make_peaked_logits(vocab, peak);

    // top_k=1 with any temperature should always pick argmax
    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 1.0,
        top_k: 1,
        seed: 99,
        rng_offsets: vec![0],
        ..Default::default()
    };

    for offset in 0..10u64 {
        let mut pp = p.clone();
        pp.rng_offsets = vec![offset];
        let gpu = run_gpu(&stream, &pp);
        assert_eq!(gpu[0], peak as u32, "top_k=1 should be argmax");
    }
}

// ============================================================================
// Tests: Top-P (Nucleus) Filtering
// ============================================================================

#[test]
fn topp_filters_low_probability_tokens() {
    let stream = test_stream();
    let vocab = 256;
    // Token 0: 90% of probability mass, Token 1: 9%, rest: ~0.004% each
    let mut logits = vec![-10.0f32; vocab];
    logits[0] = 10.0; // dominant
    logits[1] = 7.6; // second

    // top_p=0.5 should only keep token 0 (since it has >50% probability)
    let p = SamplingParams {
        logits_f32: logits.clone(),
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 1.0,
        top_p: 0.5,
        seed: 42,
        rng_offsets: vec![0],
        ..Default::default()
    };

    for offset in 0..20u64 {
        let mut pp = p.clone();
        pp.rng_offsets = vec![offset];
        let gpu = run_gpu(&stream, &pp);
        assert_eq!(
            gpu[0], 0,
            "top_p=0.5 with dominant token should always pick it"
        );
    }
}

// ============================================================================
// Tests: Combined Top-K + Top-P
// ============================================================================

#[test]
fn topk_topp_combined() {
    let stream = test_stream();
    let vocab = 1024;
    // 5 tokens with descending logits, rest very negative
    let mut logits = vec![-50.0f32; vocab];
    logits[10] = 10.0;
    logits[20] = 9.0;
    logits[30] = 8.0;
    logits[40] = 7.0;
    logits[50] = 6.0;

    // top_k=5, top_p=0.95 â€” should select from {10, 20, 30, 40, 50}
    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 1.0,
        top_k: 5,
        top_p: 0.95,
        seed: 42,
        rng_offsets: vec![0],
        ..Default::default()
    };
    let allowed = [10u32, 20, 30, 40, 50];
    for offset in 0..30u64 {
        let mut pp = p.clone();
        pp.rng_offsets = vec![offset];
        assert_valid_token(&stream, &pp, "topk_topp_combined", Some(&allowed));
    }
}

// ============================================================================
// Tests: Repeat Penalty
// ============================================================================

#[test]
fn single_vocab_token() {
    let stream = test_stream();
    // vocab_size = 1 â†’ only token 0 is possible
    let p = SamplingParams {
        logits_f32: vec![42.0f32],
        batch_size: 1,
        vocab_size: 1,
        temperature: 0.0,
        ..Default::default()
    };
    let gpu = run_gpu(&stream, &p);
    assert_eq!(gpu[0], 0, "single vocab must return token 0");
}

#[test]
fn all_equal_logits_picks_token_zero() {
    // When all logits are equal, argmax should deterministically pick the first
    // (or some consistent choice). Both GPU and CPU should agree.
    let stream = test_stream();
    let vocab = 256;
    let logits = make_uniform_logits(vocab, 5.0);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "all_equal_logits");
}

#[test]
fn all_banned_except_one() {
    let stream = test_stream();
    let vocab = 32;

    let logits = vec![1.0f32; vocab];
    // Ban all tokens except token 15
    let mut banned: Vec<i32> = (0..vocab as i32).filter(|&i| i != 15).collect();
    banned.push(-1); // sentinel
    let num_banned = banned.len() as i32 - 1; // exclude sentinel

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        banned_tokens: banned,
        num_banned_tokens: num_banned,
        banned_tokens_per_seq: 0,
        ..Default::default()
    };
    let gpu = run_gpu(&stream, &p);
    assert_eq!(gpu[0], 15, "only unbanned token should be selected");
}

#[test]
fn very_large_batch() {
    let stream = test_stream();
    let vocab = 256;
    let batch = 128;

    let mut logits = Vec::with_capacity(batch * vocab);
    for b in 0..batch {
        logits.extend(make_peaked_logits(vocab, b % vocab));
    }

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: batch as i32,
        vocab_size: vocab as i32,
        temperature: 0.0,
        rng_offsets: vec![0; batch],
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "very_large_batch");
}

#[test]
fn vocab_not_power_of_two() {
    // Test odd vocab sizes that don't align with float4 vectorization
    let stream = test_stream();

    for vocab in [33, 65, 127, 255, 513, 1000, 1023, 4097] {
        let peak = vocab / 2;
        let p = SamplingParams {
            logits_f32: make_peaked_logits(vocab, peak),
            batch_size: 1,
            vocab_size: vocab as i32,
            temperature: 0.0,
            ..Default::default()
        };
        assert_gpu_cpu_match(&stream, &p, &format!("vocab_{vocab}"));
    }
}

// ============================================================================
// Tests: Repeat Penalty with Multiple Recent Tokens
// ============================================================================

#[test]
fn stochastic_sampling_produces_valid_tokens() {
    let stream = test_stream();
    let vocab = 1024;
    let batch = 4;

    // Random-ish logits
    let mut logits = Vec::with_capacity(batch * vocab);
    for b in 0..batch {
        for i in 0..vocab {
            logits.push(((b * 1000 + i) as f32 * 0.01).sin());
        }
    }

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: batch as i32,
        vocab_size: vocab as i32,
        temperature: 0.8,
        top_k: 50,
        top_p: 0.95,
        seed: 12345,
        rng_offsets: vec![0; batch],
        ..Default::default()
    };

    for offset in 0..10u64 {
        let mut pp = p.clone();
        pp.rng_offsets = (0..batch).map(|b| offset + b as u64 * 100).collect();
        assert_valid_token(&stream, &pp, "stochastic_valid_tokens", None);
    }
}

#[test]
fn different_seeds_produce_different_results() {
    // Statistical test: with entropy, different seeds should produce variation
    let stream = test_stream();
    let vocab = 256;

    // Flat-ish logits so there's real randomness
    let logits: Vec<f32> = (0..vocab).map(|i| (i as f32 * 0.1).sin()).collect();

    let mut results = std::collections::HashSet::new();
    for seed in 0..50u64 {
        let p = SamplingParams {
            logits_f32: logits.clone(),
            batch_size: 1,
            vocab_size: vocab as i32,
            temperature: 1.0,
            seed,
            rng_offsets: vec![0],
            ..Default::default()
        };
        let gpu = run_gpu(&stream, &p);
        results.insert(gpu[0]);
    }
    // With 50 different seeds and reasonable entropy, we should get multiple distinct tokens
    assert!(
        results.len() > 3,
        "Expected variety with different seeds, got only {} distinct tokens: {:?}",
        results.len(),
        results
    );
}

// ============================================================================
// Tests: Full Pipeline Stress (many features active simultaneously)
// ============================================================================

#[test]
fn vocab_size_exact_tile_multiple() {
    // TILE_SIZE=1024 in the kernel; test exact multiples
    let stream = test_stream();

    for vocab in [1024, 2048, 4096] {
        let peak = vocab - 1;
        let p = SamplingParams {
            logits_f32: make_peaked_logits(vocab, peak),
            batch_size: 1,
            vocab_size: vocab as i32,
            temperature: 0.0,
            ..Default::default()
        };
        assert_gpu_cpu_match(&stream, &p, &format!("tile_multiple_{vocab}"));
    }
}

#[test]
fn vocab_size_tile_plus_one() {
    // Test TILE_SIZE + 1, 2*TILE_SIZE + 1 for off-by-one in tile processing
    let stream = test_stream();

    for vocab in [1025, 2049] {
        let peak = vocab - 1;
        let p = SamplingParams {
            logits_f32: make_peaked_logits(vocab, peak),
            batch_size: 1,
            vocab_size: vocab as i32,
            temperature: 0.0,
            ..Default::default()
        };
        assert_gpu_cpu_match(&stream, &p, &format!("tile_plus_one_{vocab}"));
    }
}

// ============================================================================
// Tests: Penalties Enabled but with Neutral Values (no actual effect)
// ============================================================================

#[test]
fn topk_larger_than_vocab() {
    // top_k > vocab_size should behave as if no top-k
    let stream = test_stream();
    let vocab = 64;
    let peak = 32;

    let logits = make_peaked_logits(vocab, peak);

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        top_k: 256, // larger than vocab
        ..Default::default()
    };

    let gpu = run_gpu(&stream, &p);
    assert_eq!(
        gpu[0], peak as u32,
        "top_k > vocab should still pick argmax"
    );
}

// ============================================================================
// Tests: Multi-batch stencil
// ============================================================================

#[test]
fn f32_inf_logit_is_argmax_winner() {
    // A +Inf logit should always be picked by argmax.
    let stream = test_stream();
    let vocab = 256;
    let mut logits = make_peaked_logits(vocab, 100); // token 100 has 10.0
    logits[42] = f32::INFINITY;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "f32_inf_logit_winner");
}

#[test]
fn f32_neg_inf_logit_is_never_picked() {
    // A -Inf logit should never win argmax (unless all are -Inf).
    let stream = test_stream();
    let vocab = 64;
    let mut logits = vec![1.0f32; vocab];
    logits[0] = f32::NEG_INFINITY;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    let gpu = run_gpu(&stream, &p);
    assert_ne!(gpu[0], 0, "NEG_INFINITY logit should not be argmax");
}

#[test]
fn f32_all_neg_inf_except_one() {
    // Only one finite logit â€” it must win.
    let stream = test_stream();
    let vocab = 128;
    let mut logits = vec![f32::NEG_INFINITY; vocab];
    logits[55] = 0.0;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    let gpu = run_gpu(&stream, &p);
    assert_eq!(gpu[0], 55, "only finite logit must win");
}

#[test]
fn f32_mixed_inf_and_finite() {
    // Multiple +Inf logits â€” tie-break: first one wins.
    let stream = test_stream();
    let vocab = 64;
    let mut logits = vec![0.0f32; vocab];
    logits[10] = f32::INFINITY;
    logits[20] = f32::INFINITY;
    logits[5] = 1000.0; // finite, should lose to Inf

    let p = SamplingParams {
        logits_f32: logits.clone(),
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    let gpu = run_gpu(&stream, &p);
    // Both 10 and 20 are +Inf â€” GPU parallel reduction may pick either
    assert!(
        gpu[0] == 10 || gpu[0] == 20,
        "should pick one of the +Inf tokens, got {}",
        gpu[0]
    );
}

#[test]
fn f32_very_large_logits_no_overflow() {
    // Logits near f32 max should not cause overflow in kernel's internal math.
    let stream = test_stream();
    let vocab = 64;
    let mut logits = vec![0.0f32; vocab];
    logits[0] = 1.0e30;
    logits[1] = 1.0e29;
    logits[2] = 1.0e28;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "f32_very_large_logits");
}

#[test]
fn f32_very_small_differences() {
    // Logits that differ by exactly 1 ULP at a given magnitude.
    let stream = test_stream();
    let vocab = 64;
    let base = 1.0f32;
    let next = f32::from_bits(base.to_bits() + 1); // 1 ULP above 1.0

    let mut logits = vec![0.0f32; vocab];
    logits[10] = next; // barely larger
    logits[20] = base;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "f32_ulp_difference");
}

#[test]
fn f32_subnormal_logits() {
    // Subnormal f32 values (< ~1.18e-38). Kernel should handle without crashing.
    let stream = test_stream();
    let vocab = 32;
    let mut logits = vec![-1.0f32; vocab];
    logits[0] = 1.0e-40; // subnormal
    logits[1] = 1.0e-45; // near smallest subnormal

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "f32_subnormal_logits");
}

#[test]
fn f32_alternating_extreme_logits() {
    // Alternating +Inf / -Inf / normal values â€” stress test for reduction.
    let stream = test_stream();
    let vocab = 256;
    let mut logits = Vec::with_capacity(vocab);
    for i in 0..vocab {
        logits.push(match i % 3 {
            0 => f32::NEG_INFINITY,
            1 => (i as f32) - 128.0,
            _ => f32::NEG_INFINITY,
        });
    }
    // Make sure there's one clear winner
    logits[199] = 500.0;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "f32_alternating_extreme");
}

#[test]
fn f32_zero_logits_with_penalties() {
    // All logits = 0.0, penalties distinguish tokens. Tests that penalty
    // application works correctly even when all base logits are identical.
    let stream = test_stream();
    let vocab = 64;
    let logits = vec![0.0f32; vocab];

    let mut token_counts = vec![0i32; vocab];
    // Every token has count except token 30
    for i in 0..vocab {
        if i != 30 {
            token_counts[i] = (i + 1) as i32;
        }
    }

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        frequency_penalty: 1.0,
        token_counts,
        ..Default::default()
    };
    // Token 30 (count=0) should win because all others are penalized
    assert_gpu_cpu_match(&stream, &p, "f32_zero_logits_penalties");
}

// ============================================================================
// Tests: `top_k <= 0` ("no limit") over a production-sized vocabulary
// ============================================================================

/// Temperature sampling with `top_k <= 0` must be able to return a token from
/// anywhere in the vocabulary, not just the low-id window.
///
/// Both no-top-k collectors gather candidates with `prob >= 0.0` — which every
/// token satisfies — and keep whichever `atomicAdd` tickets land under `max_k`.
/// The vocabulary is walked in ascending token order, so the first tile claims
/// every slot and the candidate set is tokens `0..MAX_K`: the byte-level and
/// punctuation range of a BPE vocabulary. A model whose real answer lives above
/// that window cannot emit it. Live, this collapsed the decode onto token 0 —
/// `!` in the Qwen vocabulary — and, where a low-id token happened to carry the
/// mass, onto short punctuation fragments.
///
/// Every existing high-peak test runs at `temperature: 0.0`, which takes the
/// argmax path and never reaches the collectors, which is why this survived.
/// `top_k` is 0 here because that is what `SamplingConfig::for_gguf_architecture`
/// derives for the Qwen lineage, so it is the shipping configuration.
#[test]
fn no_top_k_temperature_sampling_reaches_high_token_ids() {
    let stream = test_stream();
    // The Qwen3.6-35B-A3B padded vocabulary, and a peak far above any
    // collection window.
    let vocab = 248_320;
    let peak = 100_000;
    let p = SamplingParams {
        logits_f32: make_peaked_logits(vocab, peak),
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.7,
        top_k: 0,
        top_p: 0.9,
        ..Default::default()
    };
    let got = run_gpu(&stream, &p);
    assert_eq!(
        got[0], peak as u32,
        "top_k=0 sampling returned token {}; a distribution peaked at {} must \
         yield its peak, not a token from the low-id collection window",
        got[0], peak,
    );
}

/// zend's shipping decode configuration, against a realistic logit row.
///
/// Measured live off Qwen3.6-35B-A3B: `finite_min -10.49`, `finite_max 21.47`,
/// and token 0 sitting mid-pack at `12.09`. At `temperature 0.7` that is a
/// probability ratio of `exp((12.09 - 21.47) / 0.7)` — about one in a million —
/// yet the sampler returned token 0 on eight consecutive steps, which is `!` in
/// this vocabulary. The row is finite and its argmax is NOT token 0, so this is
/// the sampler choosing against the distribution rather than a bad forward.
///
/// `banned_tokens` is what routes the row down the penalty path here; zend gets
/// there the same way, via the structural think-close ban, with
/// `repeat_penalty` left at its neutral 1.0.
#[test]
fn shipping_decode_config_does_not_collapse_onto_token_zero() {
    let stream = test_stream();
    let vocab = 248_320;
    let peak = 100_000;

    // A plausible decode row: a low floor, one clear winner, and token 0 well
    // above the floor but far below the winner.
    let mut logits = vec![-10.492_188f32; vocab];
    for (i, v) in logits.iter_mut().enumerate().take(4096) {
        // A little structure near the low ids so the row is not artificially flat.
        *v = -10.0 + (i % 97) as f32 * 0.02;
    }
    logits[0] = 12.085_938;
    logits[peak] = 21.468_75;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.7,
        top_k: 0,
        top_p: 0.9,
        // Neutral, exactly as shipped — the ban is what turns penalties on.
        repeat_penalty: 1.0,
        banned_tokens: vec![151_645],
        num_banned_tokens: 1,
        token_counts: vec![0; vocab],
        ..Default::default()
    };

    // Sampling is stochastic, so judge a run rather than a single draw: at this
    // ratio token 0 should not appear at all.
    let mut zeros = 0;
    for _ in 0..16 {
        if run_gpu(&stream, &p)[0] == 0 {
            zeros += 1;
        }
    }
    assert_eq!(
        zeros, 0,
        "token 0 was sampled {zeros}/16 times from a row where it sits at 12.09 \
         against a maximum of 21.47 — the sampler is not following the distribution"
    );
}

#[test]
#[ignore = "diagnostic sweep"]
fn probe_dry_vocab_crossover() {
    let stream = test_stream();
    for vocab in [
        100_000usize, 200_000, 230_000, 235_000, 237_000, 240_000, 248_320,
    ] {
        let peak = vocab / 2;
        let mut logits = vec![-10.0f32; vocab];
        logits[0] = 12.0;
        logits[peak] = 21.5;
        let p = SamplingParams {
            logits_f32: logits,
            batch_size: 1,
            vocab_size: vocab as i32,
            temperature: 0.7,
            top_k: 0,
            top_p: 0.9,
            banned_tokens: vec![5],
            num_banned_tokens: 1,
            token_counts: vec![0; vocab],
            dry_multiplier: 0.8,
            dry_range: 1024,
            recent_tokens: vec![5, 6, 7],
            recent_lens: vec![3],
            current_lens: vec![3],
            ..Default::default()
        };
        let got = run_gpu(&stream, &p)[0];
        let bitset_kb = (vocab.div_ceil(32) * 4) as f64 / 1024.0;
        println!(
            "vocab={vocab:7} bitset={bitset_kb:6.1}KiB  got={got:7}  {}",
            if got == peak as u32 { "ok" } else { "COLLAPSED" }
        );
    }
}

/// The same realistic row across every sampling path that can reach it.
///
/// The two no-top-k collectors sit behind different template arms — one for
/// `USE_PENALTIES`, one without — so fixing either alone leaves the other
/// returning low token ids. `num_banned_tokens` is what selects the arm, and it
/// is a field separate from `banned_tokens`: leaving it at its default routes a
/// row that looks penalised down the no-penalty path instead, which is exactly
/// how the second collector stayed hidden after the first was fixed.
#[test]
fn realistic_row_never_collapses_on_any_sampling_path() {
    let stream = test_stream();
    let vocab = 248_320;
    let peak = 100_000;
    let base = || {
        let mut logits = vec![-10.492_188f32; vocab];
        for (i, v) in logits.iter_mut().enumerate().take(4096) {
            *v = -10.0 + (i % 97) as f32 * 0.02;
        }
        logits[0] = 12.085_938;
        logits[peak] = 21.468_75;
        logits
    };
    let cases: Vec<(&str, SamplingParams)> = vec![
        (
            "baseline temp.7 k0 p.9 banned",
            SamplingParams {
                logits_f32: base(),
                batch_size: 1,
                vocab_size: vocab as i32,
                temperature: 0.7,
                top_k: 0,
                top_p: 0.9,
                banned_tokens: vec![151_645],
                num_banned_tokens: 1,
                token_counts: vec![0; vocab],
                ..Default::default()
            },
        ),
        (
            "top_p=1.0 (nucleus off)",
            SamplingParams {
                logits_f32: base(),
                batch_size: 1,
                vocab_size: vocab as i32,
                temperature: 0.7,
                top_k: 0,
                top_p: 1.0,
                banned_tokens: vec![151_645],
                num_banned_tokens: 1,
                token_counts: vec![0; vocab],
                ..Default::default()
            },
        ),
        (
            "top_k=256 explicit",
            SamplingParams {
                logits_f32: base(),
                batch_size: 1,
                vocab_size: vocab as i32,
                temperature: 0.7,
                top_k: 256,
                top_p: 0.9,
                banned_tokens: vec![151_645],
                num_banned_tokens: 1,
                token_counts: vec![0; vocab],
                ..Default::default()
            },
        ),
        (
            "no penalties (no banned)",
            SamplingParams {
                logits_f32: base(),
                batch_size: 1,
                vocab_size: vocab as i32,
                temperature: 0.7,
                top_k: 0,
                top_p: 0.9,
                ..Default::default()
            },
        ),
        (
            "DRY enabled (extra smem)",
            SamplingParams {
                logits_f32: base(),
                batch_size: 1,
                vocab_size: vocab as i32,
                temperature: 0.7,
                top_k: 0,
                top_p: 0.9,
                banned_tokens: vec![151_645],
                num_banned_tokens: 1,
                token_counts: vec![0; vocab],
                dry_multiplier: 0.8,
                dry_range: 1024,
                recent_tokens: vec![5, 6, 7],
                recent_lens: vec![3],
                current_lens: vec![3],
                ..Default::default()
            },
        ),
        (
            "argmax temp=0",
            SamplingParams {
                logits_f32: base(),
                batch_size: 1,
                vocab_size: vocab as i32,
                temperature: 0.0,
                top_k: 0,
                top_p: 0.9,
                banned_tokens: vec![151_645],
                num_banned_tokens: 1,
                token_counts: vec![0; vocab],
                ..Default::default()
            },
        ),
    ];
    for (name, p) in &cases {
        let mut zeros = 0;
        for _ in 0..16 {
            if run_gpu(&stream, p)[0] == 0 {
                zeros += 1;
            }
        }
        assert_eq!(
            zeros, 0,
            "[{name}] token 0 sampled {zeros}/16 from a row where it sits at 12.09 \
             against a maximum of 21.47"
        );
    }
}

/// The same shape with penalties enabled, which takes the other no-top-k
/// collector (`tiled_sampling_pass`). Both carry the identical defect, so both
/// need holding.
#[test]
fn no_top_k_with_penalties_reaches_high_token_ids() {
    let stream = test_stream();
    let vocab = 248_320;
    let peak = 100_000;
    let p = SamplingParams {
        logits_f32: make_peaked_logits(vocab, peak),
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.7,
        top_k: 0,
        top_p: 0.9,
        // Any non-neutral penalty routes the row down the USE_PENALTIES path.
        repeat_penalty: 1.1,
        token_counts: vec![0; vocab],
        ..Default::default()
    };
    let got = run_gpu(&stream, &p);
    assert_eq!(
        got[0], peak as u32,
        "top_k=0 sampling with penalties returned token {}; expected the peak {}",
        got[0], peak,
    );
}

// ============================================================================
// Tests: Cross-Dtype Consistency
// ============================================================================
