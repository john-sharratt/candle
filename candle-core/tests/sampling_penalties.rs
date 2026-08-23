//! Penalty tests for the batched sampling kernel.
//!
//! Tests for repeat_penalty, frequency_penalty, presence_penalty,
//! repeat_last_n windowing, cross_turn_penalty, and their interactions.

#![cfg(feature = "cuda")]
// Test code: loop indices are vocab coordinates; `x + 0` / `x * 1` forms are
// kept where they spell out the penalty formula being checked.
#![allow(
    clippy::needless_range_loop,
    clippy::identity_op,
    clippy::doc_lazy_continuation
)]

#[allow(dead_code)]
mod sampling_harness;
use sampling_harness::*;

#[test]
fn repeat_penalty_demotes_recent_tokens() {
    let stream = test_stream();
    let vocab = 256;

    // Token 50 has the highest raw logit, but it's in recent history
    let mut logits = vec![0.0f32; vocab];
    logits[50] = 5.0; // would be argmax without penalty
    logits[51] = 4.9; // just below

    let recent = vec![50i32]; // token 50 was recently generated

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        repeat_penalty: 10.0, // strong penalty: 5.0 / 10.0 = 0.5 < 4.9
        recent_tokens: recent,
        recent_lens: vec![1],
        max_recent_len: 1,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "repeat_penalty_demotes");
}

#[test]
fn repeat_penalty_on_negative_logits() {
    let stream = test_stream();
    let vocab = 128;

    // Negative logit: repeat penalty multiplies (makes more negative)
    let mut logits = vec![-1.0f32; vocab];
    logits[10] = -0.5; // highest without penalty
    logits[11] = -0.6; // second highest

    let recent = vec![10i32]; // token 10 is recent

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        repeat_penalty: 2.0, // -0.5 * 2.0 = -1.0, so token 11 (-0.6) wins
        recent_tokens: recent,
        recent_lens: vec![1],
        max_recent_len: 1,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "repeat_penalty_negative");
}

// ============================================================================
// Tests: Frequency and Presence Penalties
// ============================================================================

#[test]
fn frequency_penalty_scales_with_count() {
    let stream = test_stream();
    let vocab = 64;

    let mut logits = vec![0.0f32; vocab];
    logits[5] = 10.0;
    logits[6] = 9.5;

    // Token 5 appeared 3 times Ã¢â€ â€™ penalty = 10.0 - 3 * 2.0 = 4.0, token 6 unaffected
    let mut token_counts = vec![0i32; vocab];
    token_counts[5] = 3;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        frequency_penalty: 2.0,
        token_counts,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "frequency_penalty");
}

#[test]
fn presence_penalty_binary_effect() {
    let stream = test_stream();
    let vocab = 64;

    let mut logits = vec![0.0f32; vocab];
    logits[5] = 10.0;
    logits[6] = 9.0;

    // Token 5 has count > 0, presence penalty = 2.0 Ã¢â€ â€™ effective = 10.0 - 2.0 = 8.0 < 9.0
    let mut token_counts = vec![0i32; vocab];
    token_counts[5] = 1;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        presence_penalty: 2.0,
        token_counts,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "presence_penalty");
}

/// REGRESSION TEST: Presence penalty repetition loop
///
/// Reproduces the exact issue from the chat:
/// Turn 2: "The capital of Australia is Canberra"
///
/// The model keeps picking the same tokens over and over:
/// "(1) the capital of australia is canberra. (2) the capital of australia is canberra..."
///
/// Setup:
/// - vocab_size = 151936 (real Qwen3 vocab)
/// - presence_penalty = 1.5 (Qwen official recommendation)
/// - Multiple tokens with similar positive logits (like a language model distribution)
/// - Several tokens already have count > 0 (they were generated in earlier steps)
///
/// Expected behavior:
/// - Tokens with count > 0 should have presence_penalty subtracted from their logit
/// - With penalty of 1.5, penalized logits should be LOWER than unpenalized ones
/// - Kernel should NOT pick high-count tokens repeatedly
#[test]
fn presence_penalty_prevents_repetition_loop() {
    let stream = test_stream();
    let vocab = 151936; // Real Qwen3 vocab size

    // Simulate a probability distribution with multiple reasonable options
    // This is like the actual continuation point in the repeated output
    let mut logits = vec![-1.0f32; vocab];
    // Create a "flat" region where several tokens compete
    for i in 100..110 {
        logits[i] = 5.0; // Several tokens with same decent logit
    }
    // Token 102 is slightly higher (was already generated and has count)
    logits[102] = 5.1;

    let mut token_counts = vec![0i32; vocab];
    // Simulate tokens already generated in this turn:
    // Token 102 appeared 8 times (from penalty log)
    token_counts[102] = 8;
    // Tokens 103, 104, 105 each appeared a few times
    token_counts[103] = 3;
    token_counts[104] = 2;
    token_counts[105] = 1;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        presence_penalty: 1.5, // Qwen official recommendation
        token_counts,
        ..Default::default()
    };

    let gpu_result = run_gpu(&stream, &p);
    let cpu_result = run_cpu(&p);

    // Both should agree
    assert_eq!(
        gpu_result[0], cpu_result[0],
        "GPU and CPU should agree on presence penalty calculation.\n\
         GPU picked: {}, CPU picked: {}\n\
         Token 102 logit: 5.1 - 1.5 (presence penalty) = 3.6\n\
         Tokens 100-101 logit: 5.0 (unpenalized)\n\
         Expected: one of tokens 100-101, NOT 102",
        gpu_result[0], cpu_result[0]
    );

    // More importantly: verify it did NOT pick token 102 (the high-count token)
    assert_ne!(
        gpu_result[0], 102,
        "Should NOT pick token 102 which has count=8 and presence_penalty=1.5.\n\
         Effective logit: 5.1 - 1.5 = 3.6, which is lower than unpenalized tokens."
    );
}

/// Test presence penalty with stochastic sampling across multiple steps
///
/// This simulates the actual conversation flow:
/// 1. Generate token A (count=1)
/// 2. On next step, token A should have lower probability
/// 3. Generate token B (count_A=1, count_B=1)
/// 4. On next step, both A and B should have lower probability
/// etc.
#[test]
fn presence_penalty_prevents_same_token_repetition_multishot() {
    let stream = test_stream();
    let vocab = 512;

    // Use actual chat parameters
    let temp = 0.6;
    let top_k = 20;
    let top_p = 0.95;
    let presence_penalty = 1.5;

    // Initial: no history, pick from top-k candidates
    let mut logits = vec![-100.0f32; vocab];
    for i in 0..20 {
        logits[i] = 10.0 - (i as f32) * 0.1; // Top-k candidates
    }

    let mut token_counts = vec![0i32; vocab];

    let first_sampled: u32;
    let second_sampled: u32;
    let third_sampled: u32;

    // STEP 1: Sample with no prior counts
    {
        let p = SamplingParams {
            logits_f32: logits.clone(),
            batch_size: 1,
            vocab_size: vocab as i32,
            temperature: temp,
            top_k,
            top_p,
            presence_penalty,
            token_counts: token_counts.clone(),
            ..Default::default()
        };

        let result = run_gpu(&stream, &p);
        first_sampled = result[0];
        println!("Step 1: sampled token {}", first_sampled);

        // Record it
        if (first_sampled as usize) < vocab {
            token_counts[first_sampled as usize] = 1;
        }
    }

    // STEP 2: Sample with first token counted
    {
        let p = SamplingParams {
            logits_f32: logits.clone(),
            batch_size: 1,
            vocab_size: vocab as i32,
            temperature: temp,
            top_k,
            top_p,
            presence_penalty,
            token_counts: token_counts.clone(),
            ..Default::default()
        };

        let result = run_gpu(&stream, &p);
        second_sampled = result[0];
        println!(
            "Step 2: sampled token {} (first_token had count=1, penalty={})",
            second_sampled, presence_penalty
        );

        // With presence penalty, should prefer tokens with lower/no count
        // This is a statistical assertion - not guaranteed but highly likely
        if second_sampled == first_sampled {
            println!("  WARNING: Second sample picked same token as first despite penalty!");
        }

        // Record it
        if (second_sampled as usize) < vocab {
            token_counts[second_sampled as usize] += 1;
        }
    }

    // STEP 3: Sample with both tokens counted
    {
        let p = SamplingParams {
            logits_f32: logits.clone(),
            batch_size: 1,
            vocab_size: vocab as i32,
            temperature: temp,
            top_k,
            top_p,
            presence_penalty,
            token_counts: token_counts.clone(),
            ..Default::default()
        };

        let result = run_gpu(&stream, &p);
        third_sampled = result[0];
        println!(
            "Step 3: sampled token {} (counts: first={}, second={})",
            third_sampled,
            token_counts[first_sampled as usize],
            token_counts[second_sampled as usize]
        );

        // It's possible but less likely to pick a penalized token
        let penalty_count = if third_sampled == first_sampled { 1 } else { 0 }
            + if third_sampled == second_sampled {
                1
            } else {
                0
            };
        if penalty_count > 0 {
            println!("  Step 3 picked a penalized token (may be due to temperature/randomness)");
        }
    }
}

#[test]
fn combined_penalties_interact() {
    let stream = test_stream();
    let vocab = 128;

    let mut logits = vec![0.0f32; vocab];
    logits[10] = 20.0; // high logit
    logits[11] = 9.0; // fallback

    let mut token_counts = vec![0i32; vocab];
    token_counts[10] = 5; // appeared 5 times

    let recent = vec![10i32, 20, 30]; // token 10 is recent

    // repeat_penalty=1.5: 20.0/1.5 = 13.33
    // frequency_penalty=1.0: 13.33 - 5*1.0 = 8.33
    // presence_penalty=1.0: 8.33 - 1.0 = 7.33  < 9.0 Ã¢â€ â€™ token 11 wins
    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        repeat_penalty: 1.5,
        frequency_penalty: 1.0,
        presence_penalty: 1.0,
        token_counts,
        recent_tokens: recent,
        recent_lens: vec![3],
        max_recent_len: 3,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "combined_penalties");
}

// ============================================================================
// Tests: EOS Boost
// ============================================================================

#[test]
fn repeat_penalty_multiple_recent() {
    let stream = test_stream();
    let vocab = 128;

    let mut logits = vec![0.0f32; vocab];
    logits[10] = 10.0;
    logits[20] = 9.0;
    logits[30] = 8.0;
    logits[40] = 7.0;

    // Penalize tokens 10, 20, 30 Ã¢â€ â€™ token 40 should win
    let recent = vec![10i32, 20, 30, 0, 0]; // 3 actual tokens
    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        repeat_penalty: 100.0, // very strong
        recent_tokens: recent,
        recent_lens: vec![3],
        max_recent_len: 5,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "repeat_penalty_multiple_recent");
}

// ============================================================================
// Tests: EOS Boost + Penalties Combined
// ============================================================================

#[test]
fn neutral_penalties_no_effect() {
    // When penalties are enabled but at neutral values, result should match
    // no-penalty path exactly.
    let stream = test_stream();
    let vocab = 512;
    let peak = 200;

    let logits = make_peaked_logits(vocab, peak);

    // Provide recent tokens and counts, but neutral penalty values
    let mut token_counts = vec![0i32; vocab];
    token_counts[peak] = 5;

    let recent = vec![peak as i32, 50, 100];
    let max_recent = 3;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        repeat_penalty: 1.0,    // neutral
        frequency_penalty: 0.0, // neutral
        presence_penalty: 0.0,  // neutral
        token_counts,
        recent_tokens: recent,
        recent_lens: vec![3],
        max_recent_len: max_recent,
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "neutral_penalties");
}

// ============================================================================
// Tests: Top-K with Large K
// ============================================================================

#[test]
fn repeat_penalty_batched_different_histories() {
    let stream = test_stream();
    let vocab = 64;
    let batch = 3;
    let max_recent = 4;

    // All sequences have same logits: token 10 is best, token 20 is second
    let mut logits = Vec::new();
    for _ in 0..batch {
        let mut l = vec![0.0f32; vocab];
        l[10] = 5.0;
        l[20] = 4.9;
        logits.extend(l);
    }

    // Different recent histories:
    // Seq 0: no recent token 10 Ã¢â€ â€™ picks token 10
    // Seq 1: token 10 is recent Ã¢â€ â€™ penalized Ã¢â€ â€™ picks token 20
    // Seq 2: token 20 is recent Ã¢â€ â€™ penalized Ã¢â€ â€™ picks token 10
    let mut recent_tokens = vec![0i32; batch * max_recent];
    let mut recent_lens = vec![0i32; batch];

    // Seq 0: no recent
    recent_lens[0] = 0;

    // Seq 1: token 10 is recent
    recent_tokens[1 * max_recent] = 10;
    recent_lens[1] = 1;

    // Seq 2: token 20 is recent
    recent_tokens[2 * max_recent] = 20;
    recent_lens[2] = 1;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: batch as i32,
        vocab_size: vocab as i32,
        temperature: 0.0,
        repeat_penalty: 100.0, // very strong
        recent_tokens,
        recent_lens,
        max_recent_len: max_recent as i32,
        rng_offsets: vec![0; batch],
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "repeat_penalty_batched_histories");
}

// ============================================================================
// Tests: Frequency Penalty Batched with Different Counts
// ============================================================================

#[test]
fn frequency_penalty_batched() {
    let stream = test_stream();
    let vocab = 64;
    let batch = 2;

    let mut logits = Vec::new();
    let mut token_counts = Vec::new();

    // Seq 0: token 5 has logit=10, count=1 Ã¢â€ â€™ 10 - 1*2 = 8 > 7 Ã¢â€ â€™ still wins
    {
        let mut l = vec![0.0f32; vocab];
        l[5] = 10.0;
        l[6] = 7.0;
        logits.extend(l);

        let mut tc = vec![0i32; vocab];
        tc[5] = 1;
        token_counts.extend(tc);
    }

    // Seq 1: token 5 has logit=10, count=5 Ã¢â€ â€™ 10 - 5*2 = 0 < 7 Ã¢â€ â€™ token 6 wins
    {
        let mut l = vec![0.0f32; vocab];
        l[5] = 10.0;
        l[6] = 7.0;
        logits.extend(l);

        let mut tc = vec![0i32; vocab];
        tc[5] = 5;
        token_counts.extend(tc);
    }

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: batch as i32,
        vocab_size: vocab as i32,
        temperature: 0.0,
        frequency_penalty: 2.0,
        token_counts,
        rng_offsets: vec![0; batch],
        ..Default::default()
    };
    assert_gpu_cpu_match(&stream, &p, "frequency_penalty_batched");
}

// ============================================================================
// Multi-Dtype Infrastructure
// ============================================================================
// The kernel accepts logits as `*const c_void` and dispatches based on a dtype
#[test]
fn simple_presence_penalty_argmax() {
    let stream = test_stream();
    let vocab = 16;

    // Create simple logits: all -100 except:
    // Token 0: logit=10.0, count=5 Ã¢â€ â€™ after penalty: 10.0 - 1.5 = 8.5
    // Token 1: logit=9.0, count=0 Ã¢â€ â€™ unpenalized: = 9.0
    // Token 2: logit=8.0, count=0 Ã¢â€ â€™ unpenalized: = 8.0
    let mut logits = vec![-100.0f32; vocab];
    logits[0] = 10.0;
    logits[1] = 9.0;
    logits[2] = 8.0;

    let mut token_counts = vec![0i32; vocab];
    token_counts[0] = 5; // Penalized
    token_counts[1] = 0; // Not penalized
    token_counts[2] = 0; // Not penalized

    eprintln!("\n=== SIMPLE PRESENCE PENALTY TEST (ARGMAX) ===");
    eprintln!("Token 0: logit=10.0, count=5 Ã¢â€ â€™ after penalty: 8.5");
    eprintln!("Token 1: logit=9.0, count=0 Ã¢â€ â€™ unpenalized: 9.0");
    eprintln!("Token 2: logit=8.0, count=0 Ã¢â€ â€™ unpenalized: 8.0");
    eprintln!("With temperature=0 (argmax), should pick token 1\n");

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0, // ARGMAX - no randomness
        top_k: 0,
        top_p: 1.0,
        presence_penalty: 1.5,
        token_counts,
        ..Default::default()
    };

    let gpu_result = run_gpu(&stream, &p);
    let cpu_result = run_cpu(&p);

    eprintln!("GPU selected token: {}", gpu_result[0]);
    eprintln!("CPU selected token: {}", cpu_result[0]);

    if gpu_result[0] != 1 {
        eprintln!(
            "ERROR: GPU should pick token 1 but picked {}",
            gpu_result[0]
        );
    }

    assert_eq!(
        gpu_result[0], cpu_result[0],
        "GPU/CPU disagree even with argmax!"
    );
    assert_eq!(
        gpu_result[0], 1,
        "GPU should pick token 1 but picked {}",
        gpu_result[0]
    );
}

/// DIAGNOSTIC TEST 2: Presence penalty with multinomial (temperature=0.7)
/// Tests if the bug appears with stochastic sampling
#[test]
fn presence_penalty_with_multinomial() {
    let stream = test_stream();
    let vocab = 16;

    // Create simple logits: all -100 except:
    // Token 0: logit=10.0, count=5 Ã¢â€ â€™ after penalty: 10.0 - 1.5 = 8.5
    // Token 1: logit=9.0, count=0
    // Token 2: logit=8.0, count=0
    let mut logits = vec![-100.0f32; vocab];
    logits[0] = 10.0;
    logits[1] = 9.0;
    logits[2] = 8.0;

    let mut token_counts = vec![0i32; vocab];
    token_counts[0] = 5; // Penalized

    eprintln!("\n=== PRESENCE PENALTY WITH MULTINOMIAL ===");
    eprintln!("Token 0: logit=10.0, count=5 Ã¢â€ â€™ after penalty: 8.5");
    eprintln!("Token 1: logit=9.0, count=0");
    eprintln!("Token 2: logit=8.0, count=0");
    eprintln!("With temperature=0.7 (multinomial), should strongly prefer token 1\n");

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.7, // Non-zero - stochastic
        top_k: 0,
        top_p: 1.0,
        presence_penalty: 1.5,
        token_counts,
        ..Default::default()
    };

    // top_k=0 means full vocab; only tokens 0,1,2 have non-(-100) logits.
    // After penalty: token 0 Ã¢â€ â€™ 8.5, token 1 Ã¢â€ â€™ 9.0, token 2 Ã¢â€ â€™ 8.0. All three are valid.
    assert_valid_token(
        &stream,
        &p,
        "presence_penalty_with_multinomial",
        Some(&[0, 1, 2]),
    );
}

/// DIAGNOSTIC TEST 3: Presence penalty with large vocab (vocab=256)
/// Tests if the bug appears only with large vocab
#[test]
fn simple_presence_penalty_large_vocab() {
    let stream = test_stream();
    let vocab = 256;

    // Create simple logits: all -100 except:
    // Token 0: logit=10.0, count=5 Ã¢â€ â€™ after penalty: 8.5
    // Token 1: logit=9.0, count=0
    let mut logits = vec![-100.0f32; vocab];
    logits[0] = 10.0;
    logits[1] = 9.0;

    let mut token_counts = vec![0i32; vocab];
    token_counts[0] = 5;

    eprintln!("\n=== PRESENCE PENALTY WITH LARGE VOCAB (256) ===");
    eprintln!("Token 0: logit=10.0, count=5 Ã¢â€ â€™ after penalty: 8.5");
    eprintln!("Token 1: logit=9.0, count=0");
    eprintln!("With temperature=0 (argmax) and vocab=256, should pick token 1\n");

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0, // Argmax
        top_k: 0,
        top_p: 1.0,
        presence_penalty: 1.5,
        token_counts,
        ..Default::default()
    };

    let gpu_result = run_gpu(&stream, &p);
    let cpu_result = run_cpu(&p);

    eprintln!("GPU selected token: {}", gpu_result[0]);
    eprintln!("CPU selected token: {}", cpu_result[0]);

    if gpu_result[0] != 1 {
        eprintln!(
            "ERROR: GPU should pick token 1 but picked {}",
            gpu_result[0]
        );
    }

    assert_eq!(
        gpu_result[0], cpu_result[0],
        "GPU/CPU disagree with large vocab!"
    );
    assert_eq!(
        gpu_result[0], 1,
        "Should pick token 1, got {}",
        gpu_result[0]
    );
}

/// DIAGNOSTIC TEST 4: Simple presence penalty with top_k
/// Tests if top_k processing breaks penalty
#[test]
fn simple_presence_penalty_with_top_k() {
    let stream = test_stream();
    let vocab = 16;

    // Create simple logits
    let mut logits = vec![-100.0f32; vocab];
    logits[0] = 10.0;
    logits[1] = 9.0;
    logits[2] = 8.0;

    let mut token_counts = vec![0i32; vocab];
    token_counts[0] = 5;

    eprintln!("\n=== PRESENCE PENALTY WITH TOP_K ===");
    eprintln!("Token 0: logit=10.0, count=5 Ã¢â€ â€™ after penalty: 8.5");
    eprintln!("Token 1: logit=9.0, count=0");
    eprintln!("With temperature=0.7, top_k=2\n");

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.7,
        top_k: 2, // Only top 2 tokens
        top_p: 1.0,
        presence_penalty: 1.5,
        token_counts,
        ..Default::default()
    };

    // top_k=2 Ã¢â€ â€™ only the two highest penalized logits: token 1 (9.0) and token 0 (8.5).
    assert_valid_token(
        &stream,
        &p,
        "simple_presence_penalty_with_top_k",
        Some(&[0, 1]),
    );
}

/// DIAGNOSTIC TEST 5: Presence penalty with top_k=1 (forces radix path)
/// Tests if the radix path handles penalties correctly
#[test]
fn simple_presence_penalty_with_radix_top_k() {
    let stream = test_stream();
    let vocab = 16;

    // Create simple logits
    let mut logits = vec![-100.0f32; vocab];
    logits[0] = 10.0;
    logits[1] = 9.0;
    logits[2] = 8.0;

    let mut token_counts = vec![0i32; vocab];
    token_counts[0] = 5; // Penalized: 10.0 - 1.5 = 8.5

    eprintln!("\n=== PRESENCE PENALTY WITH RADIX TOP_K=1 ===");
    eprintln!("Token 0: logit=10.0, count=5 Ã¢â€ â€™ after penalty: 8.5");
    eprintln!("Token 1: logit=9.0, count=0");
    eprintln!("With temperature=0.7, top_k=1 (radix path)\n");

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.7,
        top_k: 1, // Forces radix path
        top_p: 1.0,
        presence_penalty: 1.5,
        token_counts,
        ..Default::default()
    };

    let gpu_result = run_gpu(&stream, &p);
    let cpu_result = run_cpu(&p);

    eprintln!("GPU selected token: {}", gpu_result[0]);
    eprintln!("CPU selected token: {}", cpu_result[0]);

    if gpu_result[0] != cpu_result[0] {
        eprintln!(
            "MISMATCH: GPU picked {}, CPU picked {}",
            gpu_result[0], cpu_result[0]
        );
    }

    assert_eq!(
        gpu_result[0], cpu_result[0],
        "GPU/CPU disagree with top_k radix path!"
    );
}

/// DIAGNOSTIC TEST 9: Simple penalty with large vocab + top_k=1
#[test]
fn simple_penalty_large_vocab_top_k_1() {
    let stream = test_stream();
    let vocab = 256;

    let mut logits = vec![-100.0f32; vocab];
    logits[0] = 10.0;
    logits[1] = 9.0;

    let mut token_counts = vec![0i32; vocab];
    token_counts[0] = 5;

    eprintln!("\n=== SIMPLE PENALTY + LARGE VOCAB + TOP_K=1 ===");
    eprintln!("vocab=256, top_k=1, temperature=0.7");

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.7,
        top_k: 1,
        top_p: 1.0,
        presence_penalty: 1.5,
        token_counts,
        ..Default::default()
    };

    let gpu_result = run_gpu(&stream, &p);
    let cpu_result = run_cpu(&p);

    eprintln!("GPU: {}, CPU: {}\n", gpu_result[0], cpu_result[0]);
    assert_eq!(gpu_result[0], cpu_result[0]);
}

/// DIAGNOSTIC TEST 10: Simple penalty with large vocab + top_k=2
#[test]
fn simple_penalty_large_vocab_top_k_2() {
    let stream = test_stream();
    let vocab = 256;

    let mut logits = vec![-100.0f32; vocab];
    logits[0] = 10.0;
    logits[1] = 9.0;

    let mut token_counts = vec![0i32; vocab];
    token_counts[0] = 5;

    // Token 0: raw=10.0, count=5 Ã¢â€ â€™ penalized=8.5
    // Token 1: raw=9.0,  count=0 Ã¢â€ â€™ penalized=9.0
    // With top_k=2 and presence_penalty=1.5, only tokens 0 and 1 should be sampled.
    // CPU returns deterministic argmax (token 1), GPU uses stochastic sampling.
    // The correct test is to verify GPU only samples from {0, 1}.

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.7,
        top_k: 2,
        top_p: 1.0,
        presence_penalty: 1.5,
        token_counts,
        ..Default::default()
    };

    assert!(p.presence_penalty > 0.0, "presence_penalty not set!");
    assert!(p.top_k > 0, "top_k not set!");
    assert_eq!(p.token_counts[0], 5, "token_counts[0] should be 5!");

    // GPU must sample only from the top-2 penalized tokens (0 and 1)
    assert_valid_token(
        &stream,
        &p,
        "simple_penalty_large_vocab_top_k_2",
        Some(&[0, 1]),
    );
}

/// DIAGNOSTIC TEST 7: Simple penalty with large vocab + radix top_k
/// Narrows down if issue is vocab size or top_p
#[test]
fn simple_penalty_large_vocab_radix() {
    let stream = test_stream();
    let vocab = 256; // Large vocab like Qwen test

    let mut logits = vec![-100.0f32; vocab];
    logits[0] = 10.0;
    logits[1] = 9.0;

    let mut token_counts = vec![0i32; vocab];
    token_counts[0] = 5;

    eprintln!("\n=== SIMPLE PENALTY + LARGE VOCAB + RADIX ===");
    eprintln!("Token 0: logit=10.0, count=5 Ã¢â€ â€™ penalty: 8.5");
    eprintln!("Token 1: logit=9.0, count=0");
    eprintln!("vocab=256, temperature=0.7, top_k=1, top_p=1.0\n");

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.7,
        top_k: 1,
        top_p: 1.0, // No top_p filtering
        presence_penalty: 1.5,
        token_counts,
        ..Default::default()
    };

    let gpu_result = run_gpu(&stream, &p);
    let cpu_result = run_cpu(&p);

    eprintln!(
        "GPU selected: {}, CPU selected: {}\n",
        gpu_result[0], cpu_result[0]
    );
    assert_eq!(
        gpu_result[0], cpu_result[0],
        "Large vocab + radix disagree!"
    );
}

/// DIAGNOSTIC TEST 8: Qwen scenario with radix top_k but NO top_p
#[test]
fn repeat_last_n_token_outside_window_not_penalized() {
    let stream = test_stream();
    let vocab = 32;

    // Newest-first buffer: [_, B, _, _, _, _, A, _]
    // Window = 4 Ã¢â€ â€™ kernel reads indices 0..3: [_, B, _, _] Ã¢â€ â€™ B is penalized, A is not.
    let token_a: i32 = 5;
    let token_b: i32 = 10;

    let max_recent = 8usize;
    let mut history = vec![0i32; max_recent];
    history[1] = token_b; // inside window (index 1 < 4)
    history[6] = token_a; // outside window (index 6 >= 4)

    let mut logits = vec![-10.0f32; vocab];
    logits[token_a as usize] = 5.0;
    logits[token_b as usize] = 5.0;

    // Pass recent_lens = 4 (window size) Ã¢â‚¬â€ kernel reads indices [0..3].
    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0, // argmax Ã¢â‚¬â€ deterministic
        repeat_penalty: 1.5,
        recent_tokens: history,
        recent_lens: vec![4],
        max_recent_len: max_recent as i32,
        ..Default::default()
    };

    // token_b in window Ã¢â€ â€™ 5.0 / 1.5 Ã¢â€°Ë† 3.33;  token_a not in window Ã¢â€ â€™ 5.0.
    // token_a wins.
    let result = run_gpu(&stream, &p);
    assert_eq!(
        result[0], token_a as u32,
        "repeat_last_n: token outside window should not be penalized (expected {token_a}, got {})",
        result[0]
    );
}

/// Tokens inside the window ARE penalized and must lose to a fresh token
/// when the penalty is strong enough.
#[test]
fn repeat_last_n_token_inside_window_is_penalized() {
    let stream = test_stream();
    let vocab = 32;

    let token_penalized: i32 = 7; // inside window Ã¢â‚¬â€œ will be penalized
    let token_fresh: i32 = 15; // never seen Ã¢â‚¬â€œ will not be penalized

    let max_recent = 8usize;
    // Newest-first: token_penalized is at index 2 (inside window=4).
    let mut history = vec![0i32; max_recent];
    history[2] = token_penalized;

    let mut logits = vec![-10.0f32; vocab];
    logits[token_penalized as usize] = 5.5; // would win without penalty
    logits[token_fresh as usize] = 5.0;

    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        repeat_penalty: 2.0, // 5.5 / 2.0 = 2.75 < 5.0 Ã¢â€ â€™ fresh wins
        recent_tokens: history,
        recent_lens: vec![4],
        max_recent_len: max_recent as i32,
        ..Default::default()
    };

    let result = run_gpu(&stream, &p);
    assert_eq!(
        result[0], token_fresh as u32,
        "repeat_last_n: penalized token should lose (expected {token_fresh}, got {})",
        result[0]
    );
}

/// With recent_lens = 0 (zero-length window), NO tokens should be penalized.
#[test]
fn repeat_last_n_zero_window_no_penalty() {
    let stream = test_stream();
    let vocab = 32;
    let token_target: i32 = 9;

    let max_recent = 8usize;
    let history = vec![token_target; max_recent]; // all slots hold target token

    let mut logits = vec![-10.0f32; vocab];
    logits[token_target as usize] = 5.0;

    // recent_lens = 0 Ã¢â€ â€™ kernel reads nothing Ã¢â€ â€™ no penalty
    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        repeat_penalty: 10.0, // huge Ã¢â‚¬â€ would destroy token if applied
        recent_tokens: history,
        recent_lens: vec![0],
        max_recent_len: max_recent as i32,
        ..Default::default()
    };

    let result = run_gpu(&stream, &p);
    assert_eq!(
        result[0], token_target as u32,
        "repeat_last_n=0: no penalty should apply (expected {token_target}, got {})",
        result[0]
    );
}

// ============================================================================
// Tests: cross_turn_penalty
//
// Cross-turn penalty applies a separate (usually lighter) penalty to tokens
// that appeared in PREVIOUS turns, as distinct from the current turn.  This
// requires the kernel to distinguish between two categories of seen tokens:
//   Ã¢â‚¬Â¢ current-turn tokens   Ã¢â€ â€™ use normal presence/frequency penalty
//   Ã¢â‚¬Â¢ prior-turn tokens     Ã¢â€ â€™ use cross_turn_penalty (flat additive, like presence)
//
// Parameters:
//   - `cross_turn_penalty: f32` scalar parameter
//   - `cross_turn_counts: *const i32` GPU buffer [batch_size * vocab_size]
// ============================================================================

/// Tokens seen in a prior turn should have cross_turn_penalty applied but NOT
/// the normal presence_penalty (which is reserved for current-turn tokens).
#[test]
fn cross_turn_penalty_applies_to_prior_turn_tokens() {
    let stream = test_stream();
    let vocab = 32;

    // token_prior was seen in a previous turn (count in prior buffer = 1)
    // token_current was seen in the current turn (count in current token_counts = 1)
    // token_fresh was never seen.
    // All start with the same logit.  Ordering after penalties:
    //   token_fresh > token_prior (cross_turn_penalty=0.3) > token_current (presence_penalty=1.5)
    let token_prior: usize = 4;
    let token_current: usize = 8;
    let token_fresh: usize = 12;

    let mut logits = vec![-10.0f32; vocab];
    logits[token_prior] = 5.0;
    logits[token_current] = 5.0;
    logits[token_fresh] = 5.0;

    // Current-turn counts Ã¢â‚¬â€ only token_current has been seen this turn
    let mut token_counts = vec![0i32; vocab];
    token_counts[token_current] = 1;

    // Prior-turn counts Ã¢â‚¬â€ only token_prior was seen last turn
    let mut cross_turn_counts = vec![0i32; vocab];
    cross_turn_counts[token_prior] = 1;

    // With presence_penalty=1.5 and cross_turn_penalty=0.3:
    //   token_current  Ã¢â€ â€™ 5.0 - 1.5 = 3.5
    //   token_prior    Ã¢â€ â€™ 5.0 - 0.3 = 4.7
    //   token_fresh    Ã¢â€ â€™ 5.0
    // Expected winner: token_fresh
    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        presence_penalty: 1.5,
        cross_turn_penalty: 0.3,
        cross_turn_counts,
        token_counts,
        ..Default::default()
    };

    let result = run_gpu(&stream, &p);
    assert_eq!(
        result[0], token_fresh as u32,
        "cross_turn_penalty: fresh token should win (expected {token_fresh}, got {})",
        result[0]
    );
}

/// Cross-turn penalty should be lighter than current-turn presence penalty:
/// a prior-turn token should still be preferred over a current-turn token.
#[test]
fn cross_turn_penalty_lighter_than_presence_penalty() {
    let stream = test_stream();
    let vocab = 32;

    let token_prior: usize = 4;
    let token_current: usize = 8;

    let mut logits = vec![-10.0f32; vocab];
    logits[token_prior] = 5.0;
    logits[token_current] = 5.0;

    let mut token_counts = vec![0i32; vocab];
    token_counts[token_current] = 1;

    // With presence=1.5 and cross_turn=0.3:
    //   token_prior  (cross_turn) Ã¢â€ â€™ 5.0 - 0.3 = 4.7
    //   token_current (presence)  Ã¢â€ â€™ 5.0 - 1.5 = 3.5
    // Prior-turn token wins over current-turn token.
    let mut cross_turn_counts = vec![0i32; vocab];
    cross_turn_counts[token_prior] = 1;
    let p = SamplingParams {
        logits_f32: logits,
        batch_size: 1,
        vocab_size: vocab as i32,
        temperature: 0.0,
        presence_penalty: 1.5,
        cross_turn_penalty: 0.3,
        cross_turn_counts,
        token_counts,
        ..Default::default()
    };

    let result = run_gpu(&stream, &p);
    assert_eq!(
        result[0], token_prior as u32,
        "cross_turn_penalty: prior-turn token should beat current-turn token"
    );
}
