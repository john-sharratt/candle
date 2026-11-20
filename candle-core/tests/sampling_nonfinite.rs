//! The batched sampling kernel's non-finite verdict (`nonfinite_rows`).
//!
//! A row whose logits hold a NaN or a +inf came from a broken forward, and its
//! token means nothing: an argmax over it returns some token, because NaN never
//! wins a `>`. Every path of the kernel — argmax, sampled, sampled with
//! penalties, the stencil kernel — reports such a row in its verdict word,
//! wherever in the row the value sits, and passes a row whose only
//! non-numbers are −inf, which a masked or banned logit is on purpose.

#![cfg(feature = "cuda")]

#[allow(dead_code)]
mod sampling_harness;
use sampling_harness::*;

const VOCAB: usize = 4096;

/// Four rows: clean; a NaN; a +inf; most of the row masked to −inf.
fn rows() -> Vec<f32> {
    let row = |i: usize| -(((i * 37) % 101) as f32) / 10.0;
    let mut logits: Vec<f32> = (0..4 * VOCAB).map(|i| row(i % VOCAB)).collect();
    logits[VOCAB + 1234] = f32::NAN;
    logits[2 * VOCAB + 77] = f32::INFINITY;
    for v in &mut logits[3 * VOCAB..3 * VOCAB + VOCAB - 10] {
        *v = f32::NEG_INFINITY;
    }
    logits
}

fn params(temperature: f32, repeat_penalty: f32) -> SamplingParams {
    SamplingParams {
        logits_f32: rows(),
        batch_size: 4,
        vocab_size: VOCAB as i32,
        temperature,
        repeat_penalty,
        recent_tokens: vec![5; 4],
        recent_lens: vec![1; 4],
        max_recent_len: 1,
        rng_offsets: vec![0; 4],
        ..Default::default()
    }
}

/// Each full-vocabulary path flags the NaN and +inf rows and only those.
#[test]
fn every_full_vocabulary_path_flags_a_nan_or_inf_row_and_no_other() {
    let stream = test_stream();
    for (name, p) in [
        ("argmax", params(0.0, 1.0)),
        ("argmax with penalties", params(0.0, 1.3)),
        ("sampled", params(0.8, 1.0)),
        ("sampled with penalties", params(0.8, 1.3)),
    ] {
        let (_, verdicts) = run_gpu_verdicts(&stream, &p);
        assert_eq!(verdicts, vec![0, 1, 1, 0], "{name}");
    }
}

/// The stencil kernel reads only its stencil's logits, and judges those.
#[test]
fn a_stencil_row_is_judged_on_its_stencil_tokens() {
    let stream = test_stream();
    let mut p = params(0.0, 1.0);
    // The NaN row's NaN is in the stencil; the +inf row's is not.
    p.stencil = vec![3, 1234, 4000];
    p.stencil_size = 3;
    for v in &mut p.logits_f32[3 * VOCAB..3 * VOCAB + VOCAB] {
        *v = -1.0;
    }
    p.logits_f32[3 * VOCAB + 4000] = f32::NEG_INFINITY;
    let (_, verdicts) = run_gpu_verdicts(&stream, &p);
    assert_eq!(verdicts, vec![0, 1, 0, 0]);
}

/// The last live logit is read too — a value in the tail of the stride.
#[test]
fn a_nan_in_the_last_logit_is_flagged() {
    let stream = test_stream();
    let vocab = 248_320;
    let mut logits = vec![0.5f32; vocab];
    logits[vocab - 1] = f32::NAN;
    for temperature in [0.0, 0.8] {
        let p = SamplingParams {
            logits_f32: logits.clone(),
            batch_size: 1,
            vocab_size: vocab as i32,
            temperature,
            ..Default::default()
        };
        let (_, verdicts) = run_gpu_verdicts(&stream, &p);
        assert_eq!(verdicts, vec![1], "temperature {temperature}");
    }
}

/// Half-precision rows are judged as they widen: a NaN or an infinity in an
/// F16 or BF16 row is flagged as in F32.
#[test]
fn half_precision_rows_are_judged_alike() {
    let stream = test_stream();
    let logits = rows();
    let p = params(0.0, 1.0);
    let (f16_rows, _) = f32_to_f16(&logits);
    let (_, verdicts) = run_gpu_typed_verdicts(&stream, &f16_rows, &p, DType::F16 as i32);
    assert_eq!(verdicts, vec![0, 1, 1, 0], "f16");
    let (bf16_rows, _) = f32_to_bf16(&logits);
    let (_, verdicts) = run_gpu_typed_verdicts(&stream, &bf16_rows, &p, DType::BF16 as i32);
    assert_eq!(verdicts, vec![0, 1, 1, 0], "bf16");
}
