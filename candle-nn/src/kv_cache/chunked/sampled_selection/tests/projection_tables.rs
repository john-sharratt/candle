//! Offline KV-cache format-selection analysis tests.
//!
//! These tests load binary dumps produced by the two dump tools and run a CPU
//! implementation of the per-block cosine-distance format-selection algorithm.
//!
//! Data files (relative to candle-nn crate root):
//!   `src/kv_cache/chunked/tests/data/qwen3-kv-data.bin`  — Qwen3-30B-A3B
//!   `src/kv_cache/chunked/tests/data/llama-kv-data.bin`   — Llama-3.2-3B
//!
//! If both files are absent the tests are silently skipped, so they never
//! block a clean-room CI build.  To produce the files, run:
//!
//!   cargo test --release --features cuda --lib --package candle-transformers \
//!     quantized_qwen3_moe::tests::test_dump_kv_cache_data -- --ignored --nocapture
//!   cargo test --release --features cuda --lib --package candle-transformers \
//!     quantized_llama::tests::test_dump_kv_cache_data -- --ignored --nocapture

use crate::kv_cache::chunked::sampled_selection::{
    cpu_parallel_kernel_map, cpu_parallel_kernel_range, qrel_threshold,
    DEFAULT_CALIBRATION_ARENA_CHUNKS, DEFAULT_REPORT_ARENA_CHUNKS, ERROR_MARGIN_ABS,
    PRODUCTION_K_QREL_GAMMAS as K_QREL_GAMMAS,
    PRODUCTION_K_QREL_HIGH_THRESHOLDS as K_QREL_HIGH_THRESHOLDS,
    PRODUCTION_K_QREL_LOW_THRESHOLDS as K_QREL_LOW_THRESHOLDS, PRODUCTION_LEVEL_TIER as LEVEL_TIER,
    PRODUCTION_V_QREL_GAMMAS as V_QREL_GAMMAS,
    PRODUCTION_V_QREL_HIGH_THRESHOLDS as V_QREL_HIGH_THRESHOLDS,
    PRODUCTION_V_QREL_LOW_THRESHOLDS as V_QREL_LOW_THRESHOLDS, QREL_HIGH_PERCENTILE, SELECT_BLOCK,
};
use crate::kv_cache::chunked::tests::dump_reader::{load_dump, ChunkData};

const K_MAGWEIGHT_THRESHOLDS: [f32; 11] = K_QREL_HIGH_THRESHOLDS;
const V_COSINE_THRESHOLDS_PROPOSED: [f32; 11] = V_QREL_HIGH_THRESHOLDS;
use crate::kv_cache::chunked::sampled_selection::{cpu_palette4_reduce, SampleFormat};
#[cfg(feature = "cuda")]
use crate::kv_cache::chunked::sampled_selection::PagedSelectionGpuInputs;
use crate::kv_cache::chunked::CompressionPolicy;
use crate::kv_cache::{KvFormat, QuantFormat};
use half::f16;
use std::time::{Duration, Instant};

#[cfg(feature = "cuda")]
type GpuSelector = PagedSelectionGpuInputs;
#[cfg(not(feature = "cuda"))]
struct GpuSelector;

/// Paths to the pre-generated binary dumps, relative to this crate's root.
const QWEN3_DUMP_REL_PATH: &str = "src/kv_cache/chunked/tests/data/qwen3-kv-data.bin";
const LLAMA_DUMP_REL_PATH: &str = "src/kv_cache/chunked/tests/data/llama-kv-data.bin";

/// The end-of-test calibration sweep is only used to propose updated thresholds,
/// so it runs on a representative evenly spaced subset rather than the full dump.
const CALIBRATION_SAMPLE_CHUNKS: usize = 8;

#[test]
fn cpu_smoke() {
    let n_batch = 2;
    let n_head = 2;
    let head_dim = 8;
    let values = make_synthetic_batch(n_batch, n_head, head_dim);
    let candidates = candidate_formats();

    let surface = sample_error_surface_cpu(
        &values,
        n_batch,
        n_head,
        CHUNK_SIZE,
        head_dim,
        7,
        &candidates,
        SampleSide::Value,
    )
    .expect("cpu sampling");

    assert_eq!(surface.n_batch, n_batch);
    assert_eq!(surface.n_head, n_head);
    assert_eq!(surface.n_dim, head_dim);
    assert_eq!(surface.n_quant, candidates.len());

    let q0_err = surface.get(0, 0, 0, 0);
    let q8_idx = candidates
        .iter()
        .position(|f| *f == SampleFormat::Q8_0)
        .expect("q8 index");
    let q8_err = surface.get(0, 0, q8_idx, 0);
    assert!(
        q8_err <= q0_err,
        "higher-fidelity quant should not be worse"
    );

    let winners = select_smallest_passing(&surface, 0.02);
    let summary = model_compression_from_surface(&surface, &winners, &candidates).expect("model");

    assert!(summary.ideal_cr >= summary.palette4_cr);
    assert!(summary.palette4_cr >= summary.head_cr);
    assert!(summary.palette4_cr > 1.0);
}

// ============================================================================
// CPU quantisation helpers
// ============================================================================

/// Quantize a 32-element block to Q8_0 (8-bit symmetric, one scale per block)
/// and dequantize back to f32. Returns the reconstructed values.
fn round_trip_q8_0(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    let amax = block.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if amax == 0.0 {
        return [0.0f32; SELECT_BLOCK];
    }
    let scale = amax / 127.0;
    let inv_scale = 1.0 / scale;
    let mut out = [0.0f32; SELECT_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        let q = (x * inv_scale).round().clamp(-128.0, 127.0) as i8;
        out[i] = q as f32 * scale;
    }
    out
}

/// Quantize a 32-element block to BF16 (bfloat16) and dequantize back to f32.
/// Each element is rounded to the nearest representable BF16 value by
/// truncating the f32 mantissa from 23 bits to 7 bits.  Unlike the Qn_0
/// formats, BF16 applies no block scaling — each value is independently
/// encoded with its own exponent, preserving the full f32 dynamic range.
fn round_trip_bf16(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    let mut out = [0.0f32; SELECT_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        let bits = x.to_bits();
        // Round-to-nearest-even: add rounding bias then mask to top 16 bits.
        let bias = 0x7FFFu32 + ((bits >> 16) & 1);
        out[i] = f32::from_bits(bits.wrapping_add(bias) & 0xFFFF_0000);
    }
    out
}

/// Quantize a 32-element block to Q4_0 (4-bit symmetric, GGML convention)
/// and dequantize back to f32.
///
/// Matches the CUDA `trial_rt_q4_0`: scale = amax/8, q ∈ [-8, 7].
fn round_trip_q4_0(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    let amax = block.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if amax == 0.0 {
        return [0.0f32; SELECT_BLOCK];
    }
    let scale = amax / 8.0;
    let inv_scale = 1.0 / scale;
    let mut out = [0.0f32; SELECT_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        let q = (x * inv_scale).round().clamp(-8.0, 7.0) as i8;
        out[i] = q as f32 * scale;
    }
    out
}

/// Quantize a 32-element block to Q3_0 (3-bit symmetric, unsigned-centered, one scale per block)
/// and dequantize back to f32.
///
/// Matches the CUDA `trial_rt_q3_0`: levels 0–7 centered at 3.5, scale = amax/3.5.
fn round_trip_q3_0(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    let amax = block.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if amax == 0.0 {
        return [0.0f32; SELECT_BLOCK];
    }
    let d = amax / 3.5;
    let id = 3.5 / amax;
    let mut out = [0.0f32; SELECT_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        let q = (x * id + 3.5).round().clamp(0.0, 7.0);
        out[i] = (q - 3.5) * d;
    }
    out
}

/// Quantize a 32-element block to Q2_0 (2-bit symmetric, unsigned-centered, one scale per block)
/// and dequantize back to f32.
///
/// Matches the CUDA `trial_rt_q2_0`: levels 0–3 centered at 1.5, scale = amax/1.5.
fn round_trip_q2_0(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    let amax = block.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if amax == 0.0 {
        return [0.0f32; SELECT_BLOCK];
    }
    let d = amax / 1.5;
    let id = 1.5 / amax;
    let mut out = [0.0f32; SELECT_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        let q = (x * id + 1.5).round().clamp(0.0, 3.0);
        out[i] = (q - 1.5) * d;
    }
    out
}

/// Q8 with scale + sum (Q8_1): symmetric 8-bit — same quantization as Q8_0.
/// The "1" variant only differs in block metadata (stores sum for dot products).
/// Matches CUDA `trial_rt_q8_1`: d = amax/127, q = round(x/d) clamped [-128,127].
fn round_trip_q8_1(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    let amax = block.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if amax == 0.0 {
        return [0.0f32; SELECT_BLOCK];
    }
    let d = amax / 127.0;
    let mut out = [0.0f32; SELECT_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        let q = (x / d).round().clamp(-128.0, 127.0);
        out[i] = q * d;
    }
    out
}

/// Q8 with attention-sink sub-block scaling (Q8_KS): A (0-3) and B (4-31)
/// sub-blocks with two-level scaling (coarse_d × uint8/255).
/// Matches CUDA `trial_rt_q8_ks` exactly.
fn round_trip_q8_ks(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    let amax_a = block[..4]
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max);
    let amax_b = block[4..]
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max);
    let amax = amax_a.max(amax_b);
    if amax == 0.0 {
        return [0.0f32; SELECT_BLOCK];
    }
    let coarse_d = amax / 127.0;
    let sa = (amax_a / amax * 255.0).round().clamp(1.0, 255.0);
    let sb = (amax_b / amax * 255.0).round().clamp(1.0, 255.0);

    let mut out = [0.0f32; SELECT_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        let s = if i < 4 { sa } else { sb };
        let actual_d = coarse_d * s / 255.0;
        if actual_d == 0.0 {
            out[i] = 0.0;
            continue;
        }
        let q = (x / actual_d).round().clamp(-127.0, 127.0);
        out[i] = q * actual_d;
    }
    out
}

/// Q4 with scale + min (Q4_1): asymmetric 4-bit — min-max scaling.
fn round_trip_q4_1(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    let vmin = block.iter().copied().fold(f32::INFINITY, f32::min);
    let vmax = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = vmax - vmin;
    if range == 0.0 {
        return [vmin; SELECT_BLOCK];
    }
    let scale = range / 15.0;
    let inv_scale = 1.0 / scale;
    let mut out = [0.0f32; SELECT_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        let q = ((x - vmin) * inv_scale).round().clamp(0.0, 15.0) as u8;
        out[i] = q as f32 * scale + vmin;
    }
    out
}

/// Q4 with attention-sink sub-block scaling (Q4_KS): A (0-3) and B (4-31)
/// sub-blocks with two-level scaling (coarse_d × uint8/255), range [-7,+7].
/// Matches CUDA `trial_rt_q4_ks` exactly.
fn round_trip_q4_ks(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    let amax_a = block[..4]
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max);
    let amax_b = block[4..]
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max);
    let amax = amax_a.max(amax_b);
    if amax == 0.0 {
        return [0.0f32; SELECT_BLOCK];
    }
    let coarse_d = amax / 7.0;
    let sa = (amax_a / amax * 255.0).round().clamp(1.0, 255.0);
    let sb = (amax_b / amax * 255.0).round().clamp(1.0, 255.0);

    let mut out = [0.0f32; SELECT_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        let s = if i < 4 { sa } else { sb };
        let actual_d = coarse_d * s / 255.0;
        if actual_d == 0.0 {
            out[i] = 0.0;
            continue;
        }
        let q = (x / actual_d).round().clamp(-7.0, 7.0);
        out[i] = q * actual_d;
    }
    out
}

// ============================================================================
// FP8 E4M3 encode/decode (matches candle-kernels fp8_e4m3_utils.cuh)
// ============================================================================

/// Portable frexpf: returns (frac, exp) such that val = frac * 2^exp, 0.5 <= |frac| < 1.0.
fn frexpf_rs(val: f32) -> (f32, i32) {
    if val == 0.0 || val.is_nan() || val.is_infinite() {
        return (val, 0);
    }
    let bits = val.to_bits();
    let exp_bits = ((bits >> 23) & 0xFF) as i32;
    if exp_bits == 0 {
        // Subnormal — normalize first
        let normalized = val * (1u64 << 23) as f32;
        let nbits = normalized.to_bits();
        let nexp = ((nbits >> 23) & 0xFF) as i32;
        let frac = f32::from_bits((nbits & 0x807F_FFFF) | 0x3F00_0000);
        (frac, nexp - 126 - 23)
    } else {
        let frac = f32::from_bits((bits & 0x807F_FFFF) | 0x3F00_0000);
        (frac, exp_bits - 126)
    }
}

/// Encode f32 → E4M3 (1 sign + 4 exponent + 3 mantissa, bias 7, max 448).
fn encode_e4m3(val: f32) -> u8 {
    if val == 0.0 {
        return 0;
    }
    let sign: u8 = if val < 0.0 { 1 } else { 0 };
    let val = val.abs();
    if val >= 448.0 {
        return (sign << 7) | (14 << 3) | 7;
    }
    let (frac, exp_raw) = frexpf_rs(val);
    let exp = exp_raw - 1;
    let mut biased_e = exp + 7;
    if biased_e <= 0 {
        let m = (val * 512.0).round().min(7.0) as u8;
        return (sign << 7) | m;
    }
    let mantissa = frac * 2.0 - 1.0;
    let mut m = (mantissa * 8.0).round() as i32;
    if m >= 8 {
        biased_e += 1;
        m = 0;
    }
    if biased_e >= 15 {
        return (sign << 7) | (14 << 3) | 7;
    }
    (sign << 7) | ((biased_e as u8) << 3) | ((m & 7) as u8)
}

/// Decode E4M3 → f32.
fn decode_e4m3(val: u8) -> f32 {
    if val == 0 || val == 0x80 {
        return 0.0;
    }
    let s = (val >> 7) & 1;
    let e = ((val >> 3) & 0xF) as i32;
    let m = (val & 0x7) as f32;
    let result = if e == 0 {
        // Subnormal: m * 2^(-9)
        m * (2.0f32).powi(-9)
    } else if e == 15 && m != 0.0 {
        return 0.0; // NaN → 0
    } else {
        // Normal: (1 + m/8) * 2^(e-7)
        (1.0 + m * 0.125) * (2.0f32).powi(e - 7)
    };
    if s == 1 {
        -result
    } else {
        result
    }
}

// ============================================================================
// New-format round-trip helpers
// ============================================================================

/// Q0: constant block — all 32 elements reconstructed as the block mean,
/// round-tripped through E4M3 encoding.
fn round_trip_q0(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    // Use XOR-tree reduction matching GPU __shfl_xor_sync pattern.
    let mut vals = *block;
    let mut offset = SELECT_BLOCK / 2; // 16
    while offset > 0 {
        for i in 0..SELECT_BLOCK {
            let partner = i ^ offset;
            if partner > i {
                let sum = vals[i] + vals[partner];
                let sum2 = vals[partner] + vals[i];
                vals[i] = sum;
                vals[partner] = sum2;
            }
        }
        offset >>= 1;
    }
    let mean = vals[0] * (1.0f32 / 32.0f32);
    let val = decode_e4m3(encode_e4m3(mean));
    [val; SELECT_BLOCK]
}

/// Q1_S: 1-bit sign + FP8 E4M3 scale.
/// Each element becomes +amax or -amax based on its sign.
fn round_trip_q1_s(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    let amax = block.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if amax == 0.0 {
        return [0.0f32; SELECT_BLOCK];
    }
    let mut out = [0.0f32; SELECT_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        out[i] = if x >= 0.0 { amax } else { -amax };
    }
    out
}

/// Q2_S: 2-bit symmetric + FP8 E4M3 scale.
/// Same quantization math as Q2_0 (scale stored as E4M3 in practice,
/// but the trial round-trip uses exact float scale).
fn round_trip_q2_s(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    let amax = block.iter().copied().map(f32::abs).fold(0.0f32, f32::max);
    if amax == 0.0 {
        return [0.0f32; SELECT_BLOCK];
    }
    let d = amax / 1.5;
    let id = 1.5 / amax;
    let mut out = [0.0f32; SELECT_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        let q = (x * id + 1.5).round().clamp(0.0, 3.0);
        out[i] = (q - 1.5) * d;
    }
    out
}

/// Q2_A: 2-bit asymmetric + FP8 E4M3 scale + FP8 bias.
/// Levels 0–3, min-max scaling with range/3 step.
fn round_trip_q2_a(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    let vmin = block.iter().copied().fold(f32::INFINITY, f32::min);
    let vmax = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = vmax - vmin;
    if range == 0.0 {
        return [vmin; SELECT_BLOCK];
    }
    let d = range / 3.0;
    let id = 3.0 / range;
    let mut out = [0.0f32; SELECT_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        let q = ((x - vmin) * id).round().clamp(0.0, 3.0);
        out[i] = q.mul_add(d, vmin); // matches GPU __fmaf_rn(q, d, vmin)
    }
    out
}

/// Q2_1: 2-bit asymmetric + F16 scale + F16 min.
/// Same quantization math as Q2_A (F16 vs E4M3 scale storage
/// differs only at the byte level, not in the trial round-trip).
fn round_trip_q2_1(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    let vmin = block.iter().copied().fold(f32::INFINITY, f32::min);
    let vmax = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = vmax - vmin;
    if range == 0.0 {
        return [vmin; SELECT_BLOCK];
    }
    let d = range / 3.0;
    let id = 3.0 / range;
    let mut out = [0.0f32; SELECT_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        let q = ((x - vmin) * id).round().clamp(0.0, 3.0);
        out[i] = q.mul_add(d, vmin); // matches GPU __fmaf_rn(q, d, vmin)
    }
    out
}

/// Q3_1: 3-bit asymmetric + F16 scale + F16 min.
/// Levels 0–7, min-max scaling with range/7 step.
fn round_trip_q3_1(block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
    let vmin = block.iter().copied().fold(f32::INFINITY, f32::min);
    let vmax = block.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let range = vmax - vmin;
    if range == 0.0 {
        return [vmin; SELECT_BLOCK];
    }
    let d = range / 7.0;
    let id = 7.0 / range;
    let mut out = [0.0f32; SELECT_BLOCK];
    for (i, &x) in block.iter().enumerate() {
        let q = ((x - vmin) * id).round().clamp(0.0, 7.0);
        out[i] = q.mul_add(d, vmin); // matches GPU __fmaf_rn(q, d, vmin)
    }
    out
}

/// Compute cosine distance between original and reconstructed values.
/// Returns a value in [0, 1]: 0 = identical, 1 = orthogonal / opposite.
/// Matches CUDA `cosine_loss`: uses sqrtf(a)*sqrtf(b), clamp cos_sim to [0,1].
///
/// Accumulates per-element products then reduces via a binary tree matching
/// the CUDA `__shfl_xor_sync` pattern (offsets 16, 8, 4, 2, 1).  This
/// ensures the floating-point addition order is identical, producing
/// bit-identical sums despite IEEE 754 non-associativity.
fn cosine_distance(orig: &[f32; SELECT_BLOCK], recon: &[f32; SELECT_BLOCK]) -> f32 {
    // Per-lane products (matches __fmul_rn in CUDA)
    let mut dot_arr = [0.0f32; SELECT_BLOCK];
    let mut nx2_arr = [0.0f32; SELECT_BLOCK];
    let mut nr2_arr = [0.0f32; SELECT_BLOCK];
    for i in 0..SELECT_BLOCK {
        dot_arr[i] = orig[i] * recon[i];
        nx2_arr[i] = orig[i] * orig[i];
        nr2_arr[i] = recon[i] * recon[i];
    }

    // XOR tree reduction (matches __shfl_xor_sync offsets 16, 8, 4, 2, 1)
    fn shfl_xor_tree_sum(vals: &mut [f32; SELECT_BLOCK]) {
        let mut offset = SELECT_BLOCK / 2; // 16
        while offset > 0 {
            for i in 0..SELECT_BLOCK {
                let partner = i ^ offset;
                if partner > i {
                    let sum = vals[i] + vals[partner];
                    let sum2 = vals[partner] + vals[i];
                    vals[i] = sum;
                    vals[partner] = sum2;
                }
            }
            offset >>= 1;
        }
    }
    shfl_xor_tree_sum(&mut dot_arr);
    shfl_xor_tree_sum(&mut nx2_arr);
    shfl_xor_tree_sum(&mut nr2_arr);

    let dot = dot_arr[0];
    let norm_x2 = nx2_arr[0];
    let norm_r2 = nr2_arr[0];

    if norm_x2 == 0.0 {
        return 0.0;
    }
    if norm_r2 == 0.0 {
        return 1.0;
    }
    let cos_sim = dot / (norm_x2.sqrt() * norm_r2.sqrt());
    let cos_sim = cos_sim.clamp(0.0, 1.0);
    1.0 - cos_sim
}

fn normalized_l2_distance(orig: &[f32; SELECT_BLOCK], recon: &[f32; SELECT_BLOCK]) -> f32 {
    const EPS: f32 = 1.0e-8;
    let mut err2 = 0.0f32;
    let mut norm_x2 = 0.0f32;
    for i in 0..SELECT_BLOCK {
        let diff = orig[i] - recon[i];
        err2 += diff * diff;
        norm_x2 += orig[i] * orig[i];
    }
    if norm_x2 <= EPS {
        return 0.0;
    }
    err2 / (norm_x2 + EPS)
}

/// Magnitude-weighted error metric for K blocks.
///
/// Σ(K - K̂)² · K² / Σ(K²)
///
/// Upweights errors in high-magnitude dimensions — the ones carrying name
/// identity, gender signal, and other features that dominate Q·K dot products.
/// A block where error is concentrated in the top-magnitude elements scores
/// much higher than under cosine distance; uniform error scores about the same.
fn magnitude_weighted_distance(orig: &[f32; SELECT_BLOCK], recon: &[f32; SELECT_BLOCK]) -> f32 {
    // Per-lane products in f32, matching GPU __fmul_rn calls.
    let mut werr_arr = [0.0f32; SELECT_BLOCK];
    let mut nx2_arr = [0.0f32; SELECT_BLOCK];
    for i in 0..SELECT_BLOCK {
        let e = orig[i] - recon[i];
        werr_arr[i] = (e * e) * (orig[i] * orig[i]);
        nx2_arr[i] = orig[i] * orig[i];
    }

    // XOR-tree reduction matching GPU __shfl_xor_sync pattern.
    fn shfl_xor_tree_sum(vals: &mut [f32; SELECT_BLOCK]) {
        let mut offset = SELECT_BLOCK / 2;
        while offset > 0 {
            for i in 0..SELECT_BLOCK {
                let partner = i ^ offset;
                if partner > i {
                    let sum = vals[i] + vals[partner];
                    let sum2 = vals[partner] + vals[i];
                    vals[i] = sum;
                    vals[partner] = sum2;
                }
            }
            offset >>= 1;
        }
    }
    shfl_xor_tree_sum(&mut werr_arr);
    shfl_xor_tree_sum(&mut nx2_arr);

    let weighted_err = werr_arr[0];
    let norm_x2 = nx2_arr[0];

    if norm_x2 == 0.0 {
        return 0.0;
    }
    weighted_err / norm_x2
}

fn select_best_passing_format(
    block: &[f32; SELECT_BLOCK],
    candidates: &[BlockFormat],
    threshold: f32,
    is_key: bool,
) -> BlockFormat {
    let mut ordered_cands: Vec<BlockFormat> = candidates.iter().copied().collect();
    if ordered_cands.is_empty() {
        return BlockFormat::F16;
    }
    ordered_cands.sort_by(|a, b| {
        a.bits_per_elem()
            .partial_cmp(&b.bits_per_elem())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.table_index().cmp(&b.table_index()))
    });

    let default_fmt = *ordered_cands.first().unwrap_or(&BlockFormat::F16);
    let mut best_passing: Option<(BlockFormat, f32)> = None;
    let mut least_error: Option<(BlockFormat, f32)> = None;

    for fmt in ordered_cands {
        let dist = if matches!(fmt, BlockFormat::F16 | BlockFormat::BF16) {
            0.0
        } else {
            let recon = fmt.apply_quant(block);
            if is_key {
                let recon_adj = apply_error_margin_block(block, &recon, ERROR_MARGIN_ABS);
                magnitude_weighted_distance(block, &recon_adj)
            } else {
                normalized_l2_distance(block, &recon)
            }
        };

        let better_least_error = match least_error {
            Some((best_fmt, best_dist)) => {
                dist < best_dist
                    || (dist == best_dist
                        && (fmt.bits_per_elem() < best_fmt.bits_per_elem()
                            || (fmt.bits_per_elem() == best_fmt.bits_per_elem()
                                && fmt.table_index() < best_fmt.table_index())))
            }
            None => true,
        };
        if better_least_error {
            least_error = Some((fmt, dist));
        }

        if dist <= threshold {
            let better = match best_passing {
                Some((best_fmt, best_dist)) => {
                    let fmt_bpe = fmt.bits_per_elem();
                    let best_bpe = best_fmt.bits_per_elem();
                    fmt_bpe < best_bpe
                        || (fmt_bpe == best_bpe
                            && (dist < best_dist
                                || (dist == best_dist
                                    && fmt.table_index() < best_fmt.table_index())))
                }
                None => true,
            };
            if better {
                best_passing = Some((fmt, dist));
            }
        }
    }

    best_passing
        .or(least_error)
        .map(|(fmt, _)| fmt)
        .unwrap_or(default_fmt)
}

/// K-specific format selection using magnitude-weighted distance.
fn select_format_from_candidates_k(
    block: &[f32; SELECT_BLOCK],
    candidates: &[BlockFormat],
    threshold: f32,
) -> BlockFormat {
    select_best_passing_format(block, candidates, threshold, true)
}

// ============================================================================
// Compression-level threshold table
//
// Mirrors the threshold table used by the CUDA `select_kv_format` kernel.
// Index = compression_level (0–9). Lower threshold = more selective (less
// aggressive quantisation). Higher threshold = more permissive (more blocks
// get quantised to the lowest-fidelity format).
// ============================================================================
const COSINE_THRESHOLDS: [f32; 11] = [
    0.0001, // C0: near-lossless
    0.0002, // C1
    0.0005, // C2
    0.0010, // C3
    0.0020, // C4
    0.0050, // C5
    0.0100, // C6
    0.0200, // C7
    0.0500, // C8
    0.1000, // C9: maximum compression
    0.2000, // C10: V-push (2× C9)
];

/// Format name for display.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(non_camel_case_types)]
enum BlockFormat {
    F16,
    BF16,
    Q8_KS,
    Q8_1,
    Q8_0,
    Q4_KS,
    Q4_1,
    Q4_0,
    Q3_1,
    Q3_0,
    Q2_1,
    Q2_A,
    Q2_S,
    Q2_0,
    Q1_S,
    Q0,
    Q0_V,
    Q1_A,
    Q0_X,
    Q0_M2,
    Q0_M4,
}

impl std::fmt::Display for BlockFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F16 => write!(f, "F16"),
            Self::BF16 => write!(f, "BF16"),
            Self::Q8_KS => write!(f, "Q8_KS"),
            Self::Q8_1 => write!(f, "Q8_1"),
            Self::Q8_0 => write!(f, "Q8_0"),
            Self::Q4_KS => write!(f, "Q4_KS"),
            Self::Q4_1 => write!(f, "Q4_1"),
            Self::Q4_0 => write!(f, "Q4_0"),
            Self::Q3_1 => write!(f, "Q3_1"),
            Self::Q3_0 => write!(f, "Q3_0"),
            Self::Q2_1 => write!(f, "Q2_1"),
            Self::Q2_A => write!(f, "Q2_A"),
            Self::Q2_S => write!(f, "Q2_S"),
            Self::Q2_0 => write!(f, "Q2_0"),
            Self::Q1_S => write!(f, "Q1_S"),
            Self::Q0 => write!(f, "Q0"),
            Self::Q0_V => write!(f, "Q0_V"),
            Self::Q1_A => write!(f, "Q1_A"),
            Self::Q0_X => write!(f, "Q0_X"),
            Self::Q0_M2 => write!(f, "Q0_M2"),
            Self::Q0_M4 => write!(f, "Q0_M4"),
        }
    }
}

impl BlockFormat {
    /// Effective bits per element.
    /// Formats with block scaling carry overhead per 32-element block.
    /// BF16 and F16 encode each element independently — no scale header.
    fn bits_per_elem(self) -> f32 {
        match self {
            Self::F16 => 16.0,
            Self::BF16 => 16.0,
            Self::Q8_KS => 8.0 + 32.0 / 32.0, // 9.0 (36 bytes / 32 elem)
            Self::Q8_1 => 8.0 + 32.0 / 32.0,  // 9.0 (36 bytes / 32 elem)
            Self::Q8_0 => 8.0 + 16.0 / 32.0,  // 8.5 (34 bytes / 32 elem)
            Self::Q4_KS => 4.0 + 32.0 / 32.0, // 5.0 (20 bytes / 32 elem)
            Self::Q4_1 => 4.0 + 32.0 / 32.0,  // 5.0 (20 bytes / 32 elem)
            Self::Q4_0 => 4.0 + 16.0 / 32.0,  // 4.5 (18 bytes / 32 elem)
            Self::Q3_1 => 3.0 + 32.0 / 32.0,  // 4.0 (16 bytes / 32 elem)
            Self::Q3_0 => 3.0 + 16.0 / 32.0,  // 3.5 (14 bytes / 32 elem)
            Self::Q2_1 => 2.0 + 32.0 / 32.0,  // 3.0 (12 bytes / 32 elem)
            Self::Q2_A => 2.0 + 16.0 / 32.0, // 2.5 (10 bytes / 32 elem)
            Self::Q2_S => 2.0 + 8.0 / 32.0, // 2.25 (9 bytes / 32 elem)
            Self::Q2_0 => 2.0 + 16.0 / 32.0,  // 2.5 (10 bytes / 32 elem)
            Self::Q1_S => 1.0 + 8.0 / 32.0, // 1.25 (5 bytes / 32 elem)
            Self::Q0 => 8.0 / 32.0,           // 0.25 (1 byte / 32 elem)
            Self::Q0_V => 16.0 / 32.0,        // 0.50 (2 byte / 32 elem)
            Self::Q1_A => 48.0 / 32.0,        // 1.50 (6 byte / 32 elem)
            Self::Q0_X => 16.0 / 32.0,        // 0.50 (2 byte / 32 elem)
            Self::Q0_M2 => 24.0 / 32.0,       // 0.75 (3 byte / 32 elem)
            Self::Q0_M4 => 64.0 / 32.0,       // 2.0 (8 byte / 32 elem)
        }
    }

    /// Apply quantisation round-trip for this format.
    fn apply_quant(self, block: &[f32; SELECT_BLOCK]) -> [f32; SELECT_BLOCK] {
        match self {
            Self::F16 => *block,
            Self::BF16 => round_trip_bf16(block),
            Self::Q8_KS => round_trip_q8_ks(block),
            Self::Q8_1 => round_trip_q8_1(block),
            Self::Q8_0 => round_trip_q8_0(block),
            Self::Q4_KS => round_trip_q4_ks(block),
            Self::Q4_1 => round_trip_q4_1(block),
            Self::Q4_0 => round_trip_q4_0(block),
            Self::Q3_1 => round_trip_q3_1(block),
            Self::Q3_0 => round_trip_q3_0(block),
            Self::Q2_1 => round_trip_q2_1(block),
            Self::Q2_A => round_trip_q2_a(block),
            Self::Q2_S => round_trip_q2_s(block),
            Self::Q2_0 => round_trip_q2_0(block),
            Self::Q1_S => round_trip_q1_s(block),
            Self::Q0 => round_trip_q0(block),
            Self::Q0_V => round_trip_q0_v(block),
            Self::Q1_A => round_trip_q1_a(block),
            Self::Q0_X => round_trip_q0_x(block),
            Self::Q0_M2 => round_trip_q0_m2(block),
            Self::Q0_M4 => round_trip_q0_m4(block),
        }
    }

    /// Column index in the table ordering (most aggressive first).
    fn table_index(self) -> usize {
        match self {
            Self::Q0 => 0,
            Self::Q1_S => 1,
            Self::Q2_S => 2,
            Self::Q2_0 => 3,
            Self::Q2_A => 4,
            Self::Q2_1 => 5,
            Self::Q3_0 => 6,
            Self::Q3_1 => 7,
            Self::Q4_0 => 8,
            Self::Q4_1 => 9,
            Self::Q4_KS => 10,
            Self::Q8_0 => 11,
            Self::Q8_1 => 12,
            Self::Q8_KS => 13,
            Self::BF16 => 14,
            Self::F16 => 15,
            Self::Q0_V => 16,
            Self::Q1_A => 17,
            Self::Q0_X => 18,
            Self::Q0_M2 => 19,
            Self::Q0_M4 => 20,
        }
    }

    #[allow(dead_code)]
    fn from_table_index(idx: usize) -> Self {
        match idx {
            0 => Self::Q0,
            1 => Self::Q1_S,
            2 => Self::Q2_S,
            3 => Self::Q2_0,
            4 => Self::Q2_A,
            5 => Self::Q2_1,
            6 => Self::Q3_0,
            7 => Self::Q3_1,
            8 => Self::Q4_0,
            9 => Self::Q4_1,
            10 => Self::Q4_KS,
            11 => Self::Q8_0,
            12 => Self::Q8_1,
            13 => Self::Q8_KS,
            14 => Self::BF16,
            15 => Self::F16,
            16 => Self::Q0_V,
            17 => Self::Q1_A,
            18 => Self::Q0_X,
            19 => Self::Q0_M2,
            20 => Self::Q0_M4,
            _ => Self::F16,
        }
    }

    #[allow(dead_code)]
    fn grid_label(self) -> &'static str {
        match self {
            Self::Q0 => "Q0",
            Self::Q1_S => "Q1S",
            Self::Q2_S => "Q2S",
            Self::Q2_0 => "Q20",
            Self::Q2_A => "Q2A",
            Self::Q2_1 => "Q21",
            Self::Q3_0 => "Q30",
            Self::Q3_1 => "Q31",
            Self::Q4_0 => "Q40",
            Self::Q4_1 => "Q41",
            Self::Q4_KS => "Q4K",
            Self::Q8_0 => "Q80",
            Self::Q8_1 => "Q81",
            Self::Q8_KS => "Q8K",
            Self::BF16 => "BFL",
            Self::F16 => "F16",
            Self::Q0_V => "Q0V",
            Self::Q1_A => "Q1A",
            Self::Q0_X => "Q0X",
            Self::Q0_M2 => "QM2",
            Self::Q0_M4 => "QM4",
        }
    }
}

#[allow(dead_code)]
fn summarize_top_formats(counts: &[usize; 16], total: usize, top_n: usize) -> String {
    if total == 0 {
        return "-".to_string();
    }
    let mut items: Vec<(usize, usize)> = counts
        .iter()
        .enumerate()
        .filter_map(|(idx, &count)| if count > 0 { Some((idx, count)) } else { None })
        .collect();
    items.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    items
        .into_iter()
        .take(top_n)
        .map(|(idx, count)| {
            format!(
                "{} {:>4.1}%",
                BlockFormat::from_table_index(idx).grid_label(),
                count as f64 / total as f64 * 100.0
            )
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Candidates ordered from most aggressive to least, matching the CUDA kernel scan order.
/// BF16 and F16 are the high-fidelity tail: BF16 is tried before falling back to full F16.
const CANDIDATES: &[(BlockFormat, fn(&[f32; 32]) -> [f32; 32])] = &[
    (BlockFormat::Q0, round_trip_q0),
    (BlockFormat::Q0_V, round_trip_q0_v),
    (BlockFormat::Q1_A, round_trip_q1_a),
    (BlockFormat::Q0_X, round_trip_q0_x),
    (BlockFormat::Q0_M2, round_trip_q0_m2),
    (BlockFormat::Q1_S, round_trip_q1_s),
    (BlockFormat::Q0_M4, round_trip_q0_m4),
    (BlockFormat::Q2_S, round_trip_q2_s),
    (BlockFormat::Q2_0, round_trip_q2_0),
    (BlockFormat::Q2_A, round_trip_q2_a),
    (BlockFormat::Q2_1, round_trip_q2_1),
    (BlockFormat::Q3_0, round_trip_q3_0),
    (BlockFormat::Q3_1, round_trip_q3_1),
    (BlockFormat::Q4_0, round_trip_q4_0),
    (BlockFormat::Q4_1, round_trip_q4_1),
    (BlockFormat::Q4_KS, round_trip_q4_ks),
    (BlockFormat::Q8_0, round_trip_q8_0),
    (BlockFormat::Q8_1, round_trip_q8_1),
    (BlockFormat::Q8_KS, round_trip_q8_ks),
    (BlockFormat::BF16, round_trip_bf16),
];

/// Select the most aggressive format whose normalized L2 error is within `threshold`.
/// Returns `BF16` if no candidate passes (V uses BF16 as full-precision format).
fn select_format(block: &[f32; SELECT_BLOCK], threshold: f32) -> BlockFormat {
    select_format_with_error(block, threshold).0
}

/// Like select_format but also returns the achieved normalized L2 error.
fn select_format_with_error(block: &[f32; SELECT_BLOCK], threshold: f32) -> (BlockFormat, f32) {
    for &(fmt, round_trip) in CANDIDATES {
        let recon = round_trip(block);
        let dist = normalized_l2_distance(block, &recon);
        if dist <= threshold {
            return (fmt, dist);
        }
    }
    (BlockFormat::BF16, 0.0)
}

// ============================================================================
// Tests
// ============================================================================

/// Return path to the Qwen3 dump file, or None if absent.
fn dump_path() -> Option<std::path::PathBuf> {
    dump_path_for(QWEN3_DUMP_REL_PATH)
}

/// Return the absolute path for a dump file given its crate-relative path, or None if absent.
fn dump_path_for(rel: &str) -> Option<std::path::PathBuf> {
    let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    if p.exists() { Some(p) } else { None }
}

#[test]
#[ignore]
fn test_kv_format_selection_statistics() {
    let path = match dump_path() {
        Some(p) => p,
        None => {
            println!(
                "kv_selection_tests: dump file absent, skipping. \
                 Run test_dump_kv_cache_data to generate it."
            );
            return;
        }
    };

    let (header, chunks) = load_dump(&path).expect("failed to load dump");
    println!(
        "Loaded dump: {} layers, {} kv-heads, chunk_size={}, head_dim={}",
        header.num_layers, header.n_kv_head, header.chunk_size, header.head_dim
    );
    println!("Total float chunks: {}", chunks.len());

    let elems = header.n_kv_head * header.chunk_size * header.head_dim;
    let num_blocks_per_chunk = elems / SELECT_BLOCK;

    // For each compression level, count how many K blocks and V blocks would
    // be assigned to each format.
    println!("\n=== Per-compression-level format distribution ===\n");
    println!(
        "{:>4}  {:>32}  {:>32}",
        "C", "K-cache format counts", "V-cache format counts"
    );

    for (level, &threshold) in COSINE_THRESHOLDS.iter().enumerate() {
        let mut k_counts = std::collections::HashMap::<BlockFormat, usize>::new();
        let mut v_counts = std::collections::HashMap::<BlockFormat, usize>::new();
        let mut total_k_blocks = 0usize;
        let mut total_v_blocks = 0usize;

        for chunk in &chunks {
            for b in 0..num_blocks_per_chunk {
                let start = b * SELECT_BLOCK;
                let end = start + SELECT_BLOCK;

                if end <= chunk.k.len() {
                    let blk_k: [f32; SELECT_BLOCK] = chunk.k[start..end].try_into().unwrap();
                    let fmt = select_format(&blk_k, threshold);
                    *k_counts.entry(fmt).or_insert(0) += 1;
                    total_k_blocks += 1;
                }

                if end <= chunk.v.len() {
                    let blk_v: [f32; SELECT_BLOCK] = chunk.v[start..end].try_into().unwrap();
                    let fmt = select_format(&blk_v, threshold);
                    *v_counts.entry(fmt).or_insert(0) += 1;
                    total_v_blocks += 1;
                }
            }
        }

        let fmt_summary =
            |counts: &std::collections::HashMap<BlockFormat, usize>, total: usize| -> String {
                let formats = [
                    BlockFormat::Q2_0,
                    BlockFormat::Q3_0,
                    BlockFormat::Q4_0,
                    BlockFormat::Q8_0,
                    BlockFormat::F16,
                ];
                formats
                    .iter()
                    .filter_map(|&f| {
                        counts
                            .get(&f)
                            .map(|&n| format!("{}:{:.0}%", f, n as f64 / total as f64 * 100.0))
                    })
                    .collect::<Vec<_>>()
                    .join(" ")
            };

        println!(
            "C{:1} (thr={:.4})  K[{}]  V[{}]",
            level,
            threshold,
            fmt_summary(&k_counts, total_k_blocks),
            fmt_summary(&v_counts, total_v_blocks),
        );
    }
}

#[test]
#[ignore]
fn test_kv_cosine_distance_by_layer() {
    let path = match dump_path() {
        Some(p) => p,
        None => {
            println!(
                "kv_selection_tests: dump file absent, skipping. \
                 Run test_dump_kv_cache_data to generate it."
            );
            return;
        }
    };

    let (header, chunks) = load_dump(&path).expect("failed to load dump");
    let elems = header.n_kv_head * header.chunk_size * header.head_dim;
    let num_blocks_per_chunk = elems / SELECT_BLOCK;

    // Report mean Q8_0 cosine distance per layer to help identify outlier layers.
    println!("\n=== Mean Q8_0 cosine distance per layer ===");
    println!(
        "{:>6}  {:>12}  {:>12}  {:>8}",
        "Layer", "K mean dist", "V mean dist", "Chunks"
    );

    for layer_idx in 0..header.num_layers {
        let layer_chunks: Vec<&ChunkData> =
            chunks.iter().filter(|c| c.layer_idx == layer_idx).collect();

        if layer_chunks.is_empty() {
            continue;
        }

        let mut k_dist_sum = 0.0f64;
        let mut v_dist_sum = 0.0f64;
        let mut count = 0usize;

        for chunk in &layer_chunks {
            for b in 0..num_blocks_per_chunk {
                let start = b * SELECT_BLOCK;
                let end = start + SELECT_BLOCK;
                if end > chunk.k.len() || end > chunk.v.len() {
                    break;
                }

                let blk_k: [f32; SELECT_BLOCK] = chunk.k[start..end].try_into().unwrap();
                let recon_k = round_trip_q8_0(&blk_k);
                k_dist_sum += cosine_distance(&blk_k, &recon_k) as f64;

                let blk_v: [f32; SELECT_BLOCK] = chunk.v[start..end].try_into().unwrap();
                let recon_v = round_trip_q8_0(&blk_v);
                v_dist_sum += cosine_distance(&blk_v, &recon_v) as f64;

                count += 1;
            }
        }

        if count > 0 {
            println!(
                "{:>6}  {:>12.6}  {:>12.6}  {:>8}",
                layer_idx,
                k_dist_sum / count as f64,
                v_dist_sum / count as f64,
                layer_chunks.len(),
            );
        }
    }
}

// ============================================================================
// Comprehensive compression-mode analysis table
// ============================================================================

/// Accumulated statistics for one component (K or V) at one compression level.
struct ComponentStats {
    /// Block counts by format, indexed by `BlockFormat::table_index()`.
    fmt_counts: [usize; 16],
    /// Sum of bits-per-element over all blocks.
    bpe_sum: f64,
    /// Sum of cosine distances.
    cos_sum: f64,
    /// All per-block cosine distances (used for percentile reporting).
    cos_dists: Vec<f32>,
    /// Aggregate signal energy: Σ x².
    sig: f64,
    /// Aggregate noise energy: Σ (x − x̂)².
    noise: f64,
    /// Maximum cosine distance seen across all blocks.
    max_cos: f32,
    /// Total number of blocks processed.
    n_blocks: usize,
}

impl ComponentStats {
    fn new() -> Self {
        Self {
            fmt_counts: [0; 16],
            bpe_sum: 0.0,
            cos_sum: 0.0,
            cos_dists: Vec::new(),
            sig: 0.0,
            noise: 0.0,
            max_cos: 0.0,
            n_blocks: 0,
        }
    }

    /// 95th-percentile cosine distance across all processed blocks.
    fn p95_cos(&self) -> f64 {
        if self.cos_dists.is_empty() {
            return 0.0;
        }
        let mut s = self.cos_dists.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((s.len() as f64 * 0.95) as usize).min(s.len() - 1);
        s[idx] as f64
    }
}

fn process_blocks(blocks: &[[f32; SELECT_BLOCK]], threshold: f32) -> ComponentStats {
    let mut s = ComponentStats::new();
    for blk in blocks {
        let fmt = select_format(blk, threshold);
        s.fmt_counts[fmt.table_index()] += 1;
        s.bpe_sum += fmt.bits_per_elem() as f64;
        let recon = fmt.apply_quant(blk);
        let cd = cosine_distance(blk, &recon);
        s.cos_sum += cd as f64;
        s.cos_dists.push(cd);
        s.max_cos = s.max_cos.max(cd);
        for (&x, &xh) in blk.iter().zip(recon.iter()) {
            s.sig += (x as f64) * (x as f64);
            s.noise += ((x - xh) as f64) * ((x - xh) as f64);
        }
        s.n_blocks += 1;
    }
    s
}

/// Comprehensive per-compression-level analysis table.
///
/// Produces two tables suitable for academic publication:
///   Table 1 — Format selection distribution and effective compression ratio
///   Table 2 — Quality metrics: mean cosine distance, aggregate SNR (dB), NRMSE
///
/// Run with:
///   cargo test --release --lib --package candle-nn \
///     kv_selection_tests::test_compression_mode_table -- --ignored --nocapture
#[test]
#[ignore]
fn test_compression_mode_table() {
    let path = match dump_path() {
        Some(p) => p,
        None => {
            println!(
                "kv_selection_tests: dump file absent, skipping. \
                 Run test_dump_kv_cache_data to generate it."
            );
            return;
        }
    };

    let (header, chunks) = load_dump(&path).expect("failed to load dump");
    let elems_per_chunk = header.n_kv_head * header.chunk_size * header.head_dim;
    let num_blocks_per_chunk = elems_per_chunk / SELECT_BLOCK;

    // Collect all K and V 32-element sub-blocks once.
    let mut k_blocks: Vec<[f32; SELECT_BLOCK]> = Vec::new();
    let mut v_blocks: Vec<[f32; SELECT_BLOCK]> = Vec::new();
    for chunk in &chunks {
        for b in 0..num_blocks_per_chunk {
            let s = b * SELECT_BLOCK;
            let e = s + SELECT_BLOCK;
            if e <= chunk.k.len() {
                k_blocks.push(chunk.k[s..e].try_into().unwrap());
            }
            if e <= chunk.v.len() {
                v_blocks.push(chunk.v[s..e].try_into().unwrap());
            }
        }
    }

    let n_k = k_blocks.len() as f64;
    let n_v = v_blocks.len() as f64;

    // Compute stats for every level.
    let mut level_stats: Vec<(ComponentStats, ComponentStats)> = Vec::new();
    for &threshold in &COSINE_THRESHOLDS {
        level_stats.push((
            process_blocks(&k_blocks, threshold),
            process_blocks(&v_blocks, threshold),
        ));
    }

    // ─── Header ───────────────────────────────────────────────────────────────
    let sep = "=".repeat(115);
    println!("\n{sep}");
    println!("Adaptive KV-Cache Compression: Selection Analysis");
    println!(
        "  {layers} layers × {chunks} chunks × {blocks} 32-elem sub-blocks/chunk  \
         |  {nk} K-blocks  {nv} V-blocks",
        layers = header.num_layers,
        chunks = chunks.len() / header.num_layers,
        blocks = num_blocks_per_chunk,
        nk = k_blocks.len(),
        nv = v_blocks.len(),
    );
    println!(
        "  Selection: cosine-distance threshold scan Q2→Q3→Q4→Q8→BF16→F16 (mirrors CUDA kernel)"
    );
    println!("{sep}");

    // ─── Table 1: Format Distribution ─────────────────────────────────────────
    println!("\nTable 1 — Format selection distribution and effective compression ratio");
    println!("  BPE = mean bits/element (K+V combined)  |  CR = compression ratio vs F16 (16 bpe)");
    println!("  Format columns indexed by table_index: Q0 Q1S Q2S Q2_0 Q2A Q2_1 Q3_0 Q3_1 Q4_0 Q4_1 Q4K Q8_0 Q8_1 Q8K BF16 F16");
    println!();
    // Column labels for the 16 formats
    let fmt_labels = [
        "Q0", "Q1S", "Q2S", "Q20", "Q2A", "Q21", "Q30", "Q31", "Q40", "Q41", "Q4K", "Q80", "Q81",
        "Q8K", "BF", "F16",
    ];
    let mut hdr1 = format!("  {:<5} {:<8} ", "Mode", "Thr");
    for l in &fmt_labels {
        hdr1 += &format!(" K{:<3}", l);
    }
    hdr1 += " ";
    for l in &fmt_labels {
        hdr1 += &format!(" V{:<3}", l);
    }
    hdr1 += &format!("  {:>6}  {:>5}", "BPE", "CR");
    println!("{hdr1}");
    println!("  {}", "-".repeat(hdr1.len() - 2));

    for (level, (k, v)) in level_stats.iter().enumerate() {
        let eff_bpe = (k.bpe_sum + v.bpe_sum) / (n_k + n_v);
        let cr = 16.0 / eff_bpe;
        let kp: Vec<f64> = k
            .fmt_counts
            .iter()
            .map(|&c| c as f64 / n_k * 100.0)
            .collect();
        let vp: Vec<f64> = v
            .fmt_counts
            .iter()
            .map(|&c| c as f64 / n_v * 100.0)
            .collect();
        let mut line = format!("  C{level:<4} {:<8.4} ", COSINE_THRESHOLDS[level]);
        for p in &kp {
            line += &format!(" {:>4.1}", p);
        }
        line += " ";
        for p in &vp {
            line += &format!(" {:>4.1}", p);
        }
        line += &format!("  {:>6.2}  {:>4.2}×", eff_bpe, cr);
        println!("{line}");
    }

    // ─── Table 2: Quality Metrics ─────────────────────────────────────────────
    println!();
    println!("Table 2 — Quality metrics (aggregate over all blocks at each compression level)");
    println!("  SNR  = 10·log₁₀(Σx²/Σ(x−x̂)²)  |  NRMSE = √(Σ(x−x̂)²/Σx²) = 1/√SNR_linear");
    println!("  cos_μ = mean cosine distance  |  p95 = 95th-percentile cosine distance per block");
    println!("  p95 is preferred over mean (zero-pad floor skews mean) and max (max = threshold).");
    println!();
    let hdr2 = format!(
        "  {:<5} {:<8}  {:>10}  {:>8}  {:>8}  {:>8}    {:>10}  {:>8}  {:>8}  {:>8}  {:>10}",
        "Mode",
        "Thr",
        "K_cos_μ",
        "K_p95",
        "K_SNR_dB",
        "K_NRMSE",
        "V_cos_μ",
        "V_p95",
        "V_SNR_dB",
        "V_NRMSE",
        "MaxCosDist",
    );
    println!("{hdr2}");
    println!("  {}", "-".repeat(hdr2.len() - 2));

    for (level, (k, v)) in level_stats.iter().enumerate() {
        let k_cos_mean = k.cos_sum / n_k;
        let v_cos_mean = v.cos_sum / n_v;

        let snr_db = |sig: f64, noise: f64| -> f64 {
            if noise > 1e-30 {
                10.0 * (sig / noise).log10()
            } else {
                f64::INFINITY
            }
        };
        let nrmse = |sig: f64, noise: f64| -> f64 {
            if sig > 1e-30 {
                (noise / sig).sqrt()
            } else {
                0.0
            }
        };

        let k_snr = snr_db(k.sig, k.noise);
        let v_snr = snr_db(v.sig, v.noise);
        let k_nr = nrmse(k.sig, k.noise);
        let v_nr = nrmse(v.sig, v.noise);
        let max_cd = k.max_cos.max(v.max_cos);

        let fmt_snr = |x: f64| -> String {
            if x.is_infinite() {
                "    ∞".to_string()
            } else {
                format!("{x:>8.2}")
            }
        };
        println!(
            "  C{level:<4} {thr:<8.4}  {kcos:>10.6}  {kp95:>8.6}  {ksnr}  {knr:>8.5}    {vcos:>10.6}  {vp95:>8.6}  {vsnr}  {vnr:>8.5}  {mc:>10.6}",
            level=level, thr=COSINE_THRESHOLDS[level],
            kcos=k_cos_mean, kp95=k.p95_cos(), ksnr=fmt_snr(k_snr), knr=k_nr,
            vcos=v_cos_mean, vp95=v.p95_cos(), vsnr=fmt_snr(v_snr), vnr=v_nr,
            mc=max_cd,
        );
    }

    // ─── F16 baseline row for reference ────────────────────────────────────────
    println!();
    println!("  F16 baseline:  BPE=16.0  CR=1.00×  SNR=∞ dB  NRMSE=0  cos_μ=0  (reference)");
    println!("{sep}");
}

// ============================================================================
// Redesigned threshold curve  — test-driven specification
// ============================================================================

/// Proposed replacement for COSINE_THRESHOLDS.
///
/// Key observations driving the redesign:
///
///   1. ZERO-PAD FLOOR (18.8% free)
///      The partial last chunk (token_start=96) covers positions 96–127 but only
///      8 tokens were generated; positions 104–127 are genuinely zero.  These
///      768 zero blocks per chunk (75% of the last chunk) always pass Q2_0 at
///      cosine_distance=0.  They account for the permanent 18.8% Q2_0 floor and
///      do NOT represent real activation data.
///
///   2. Q4_0 CLIFF IN THE CURRENT CURVE
///      With the old thresholds 0.001–0.005 the Q4_0 fraction jumps from ~1%
///      to ~37% in one step.  The non-zero block Q4_0 distribution has mode
///      near 0.006; thresholds must span 0.002–0.016 to cover it evenly.
///
///   3. ATTENTION-SINK PROTECTION
///      Chunk with token_start=0 holds positions 0–31.  Positions 0–3 are known
///      attention sinks (100x+ mean attention weight per StreamingLLM / SinkCache
///      research).  A quantisation error there propagates to every subsequent
///      query.  Sink protection (threshold divisor) has been removed; all chunks
///      use the base threshold uniformly.
///
///   4. COMPRESSION CEILING
///      Current C9=0.10 stops before Q2_0 enters non-zero blocks.  Raising to
///      0.15 pushes ~30–40% of non-zero blocks to Q2_0, yielding 5.2–5.5× CR
///      while remaining above the empirical coherence floor (~8 dB SNR).
///
/// Tier design — empirically calibrated against the Q8_0 cosine-distance distribution:
///   K_mean = 0.000018  K_p95 = 0.000054  K_max ≈ 0.000099
///   V_mean = 0.000014  V_p95 = 0.000031  V_max ≈ 0.000099
///
///   C0–C2  quality   — thresholds inside Q8_0 distribution → BF16 backstop actively used
///                      C0: below V_mean (~70% BF16, true reference)
///                      C1: between mean and p95 (~30% BF16, near-lossless blend)
///                      C2: between p95 and max (~5% BF16, practical Q8_0 mode)
///   C3–C6  sweet     — 4 evenly-separated CR steps; SNR 17–36 dB
///   C7–C9  compress  — maximise CR; C7 threshold set at p95+0.006 for distribution headroom
/// K thresholds — magnitude-weighted distance metric.
///
/// Calibrated via binary search to produce the same per-head K BPE as the
/// previous cosine-distance thresholds.  The magnitude-weighted metric
/// Σ(K-K̂)²·K² / Σ(K²) upweights high-magnitude dimensions that carry
/// name/gender/entity identity signals, causing blocks with concentrated
/// error in those dims to be bumped to more conservative formats.
///
/// Note: thresholds are NOT monotonic across levels because each level has
/// a different candidate list. The resulting BPE is monotonically decreasing.
///
/// The actual threshold table and per-level candidate ladder are imported from
/// the shared production adaptive profile so the analysis harness cannot drift.

/// Format selection (sink protection removed; all chunks use the base threshold).
fn select_format_sink_aware(
    block: &[f32; SELECT_BLOCK],
    base_threshold: f32,
    _token_start: usize,
) -> BlockFormat {
    select_format(block, base_threshold)
}

fn compute_q_relevance_split(
    chunks: &[ChunkData],
    blocks_per_chunk: usize,
    quantile: f64,
) -> Option<f32> {
    let mut rels = Vec::new();
    for chunk in chunks {
        if let Some(q) = chunk.q.as_ref() {
            for b in 0..blocks_per_chunk {
                let start = b * SELECT_BLOCK;
                let end = start + SELECT_BLOCK;
                if end <= chunk.k.len() && end <= q.len() {
                    let k_blk: [f32; SELECT_BLOCK] = chunk.k[start..end].try_into().unwrap();
                    let q_blk: [f32; SELECT_BLOCK] = q[start..end].try_into().unwrap();
                    rels.push(cpu_block_relevance(&k_blk, &q_blk));
                }
            }
        }
    }
    if rels.is_empty() {
        for chunk in chunks {
            for b in 0..blocks_per_chunk {
                let start = b * SELECT_BLOCK;
                let end = start + SELECT_BLOCK;
                if end <= chunk.k.len() {
                    let k_blk: [f32; SELECT_BLOCK] = chunk.k[start..end].try_into().unwrap();
                    let energy = k_blk.iter().map(|&x| x * x).sum::<f32>() / SELECT_BLOCK as f32;
                    rels.push(energy);
                }
            }
        }
    }
    if rels.is_empty() {
        return None;
    }
    rels.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let idx = ((rels.len() as f64 * quantile) as usize).min(rels.len() - 1);
    Some(rels[idx])
}

/// Accumulated statistics for the proposed-curve test.
#[derive(Default, Clone)]
struct CurveStat {
    counts: [usize; 16], // indexed by BlockFormat::table_index()
    bpe_sum: f64,
    cos_sum: f64,
    cos_dists: Vec<f32>, // all per-block cosine distances for percentile reporting
    sig: f64,
    noise: f64,
    max_cos: f32,
    n: usize,
}

impl CurveStat {
    fn push(&mut self, block: &[f32; SELECT_BLOCK], fmt: BlockFormat) {
        self.counts[fmt.table_index()] += 1;
        self.bpe_sum += fmt.bits_per_elem() as f64;
        let recon = fmt.apply_quant(block);
        let cd = cosine_distance(block, &recon);
        self.cos_sum += cd as f64;
        self.cos_dists.push(cd);
        self.max_cos = self.max_cos.max(cd);
        for (&x, &xh) in block.iter().zip(recon.iter()) {
            self.sig += (x as f64) * (x as f64);
            self.noise += ((x - xh) as f64) * ((x - xh) as f64);
        }
        self.n += 1;
    }
    fn bpe(&self) -> f64 {
        if self.n > 0 {
            self.bpe_sum / self.n as f64
        } else {
            0.0
        }
    }
    fn cos_mu(&self) -> f64 {
        if self.n > 0 {
            self.cos_sum / self.n as f64
        } else {
            0.0
        }
    }
    fn snr_db(&self) -> f64 {
        if self.noise > 1e-30 {
            10.0 * (self.sig / self.noise).log10()
        } else {
            f64::INFINITY
        }
    }
    fn nrmse(&self) -> f64 {
        if self.sig > 1e-30 {
            (self.noise / self.sig).sqrt()
        } else {
            0.0
        }
    }
    fn pct(&self, i: usize) -> f64 {
        if self.n > 0 {
            self.counts[i] as f64 / self.n as f64 * 100.0
        } else {
            0.0
        }
    }
    /// 95th-percentile cosine distance across all processed blocks.
    /// More informative than mean (zero-pad floor skews mean) and max
    /// (max is just the threshold; it carries no distributional information).
    fn p95(&self) -> f64 {
        if self.cos_dists.is_empty() {
            return 0.0;
        }
        let mut s = self.cos_dists.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let idx = ((s.len() as f64 * 0.95) as usize).min(s.len() - 1);
        s[idx] as f64
    }
}

/// Collect naive + sink-aware stats for one threshold level from KV chunk data.
/// Returns [k_naive, v_naive, k_sink, v_sink].
fn curve_level_stats(
    chunks: &[ChunkData],
    num_blocks_per_chunk: usize,
    k_threshold: f32,
    v_threshold: f32,
) -> [CurveStat; 4] {
    let mut s: [CurveStat; 4] = Default::default();
    for chunk in chunks {
        for b in 0..num_blocks_per_chunk {
            let start = b * SELECT_BLOCK;
            let end = start + SELECT_BLOCK;
            if end <= chunk.k.len() {
                let blk: [f32; SELECT_BLOCK] = chunk.k[start..end].try_into().unwrap();
                s[0].push(&blk, select_format(&blk, k_threshold));
                s[2].push(
                    &blk,
                    select_format_sink_aware(&blk, k_threshold, chunk.token_start),
                );
            }
            if end <= chunk.v.len() {
                let blk: [f32; SELECT_BLOCK] = chunk.v[start..end].try_into().unwrap();
                s[1].push(&blk, select_format(&blk, v_threshold));
                s[3].push(
                    &blk,
                    select_format_sink_aware(&blk, v_threshold, chunk.token_start),
                );
            }
        }
    }
    s
}

/// Proposed compression curve — test-driven specification for the revised CUDA kernel.
///
/// Shows three tables:
///   Table 1 — Naive: format distribution (%) + BPE + CR
///   Table 2 — Sink-aware: same columns + CR penalty for protection
///   Table 3 — Quality: K/V SNR, NRMSE, cos_μ, delta SNR from sink-aware protection
///
/// Prints proposed threshold constants ready to paste into the CUDA kernel.
///
/// Run with:
///   cargo test --release --lib --package candle-nn \
///     kv_selection_tests::test_proposed_compression_curve -- --ignored --nocapture
#[test]
#[ignore]
fn test_proposed_compression_curve() {
    let path = match dump_path() {
        Some(p) => p,
        None => {
            println!("kv_selection_tests: dump absent — run test_dump_kv_cache_data first.");
            return;
        }
    };

    let (header, chunks) = load_dump(&path).expect("failed to load dump");
    let elems = header.n_kv_head * header.chunk_size * header.head_dim;
    let num_blocks = elems / SELECT_BLOCK;

    let all: Vec<[CurveStat; 4]> = (0..11)
        .map(|i| {
            curve_level_stats(
                &chunks,
                num_blocks,
                K_MAGWEIGHT_THRESHOLDS[i],
                V_COSINE_THRESHOLDS_PROPOSED[i],
            )
        })
        .collect();

    let k_n = |i: usize| &all[i][0];
    let v_n = |i: usize| &all[i][1];
    let k_s = |i: usize| &all[i][2];
    let v_s = |i: usize| &all[i][3];

    let cr_naive = |i: usize| 16.0 / ((k_n(i).bpe() + v_n(i).bpe()) / 2.0);
    let cr_sink = |i: usize| 16.0 / ((k_s(i).bpe() + v_s(i).bpe()) / 2.0);

    let sep = "=".repeat(105);
    let sep80 = "-".repeat(105);

    println!("\n{sep}");
    println!("Proposed KV-Cache Compression Curve  (test-driven — CUDA kernel target)");
    println!("  {} layers  {} chunks total  {} K-blocks  {} V-blocks",
        header.num_layers,
        chunks.len(),
        all[0][0].n, all[0][1].n,
    );

    // ── Table 1: Naive selection ──────────────────────────────────────────────
    println!("\n{sep80}");
    println!("Table 1 -- Naive selection: format distribution and compression ratio");
    println!(
        "  NOTE: ~18.8% Q2_0 at every level = zero-pad blocks (tokens 104-127 are\n\
              \x20        genuinely zero; they contribute 0 error and are not attention sinks)."
    );
    println!();
    let h1 = "  Mode  Tier       Thr      KQ0 KQ1S KQ2S KQ20 KQ2A KQ21 KQ30 KQ31 KQ40 KQ41 KQ4K KQ80 KQ81 KQ8K KBFL KF16    VQ0 VQ1S VQ2S VQ20 VQ2A VQ21 VQ30 VQ31 VQ40 VQ41 VQ4K VQ80 VQ81 VQ8K VBFL VF16   BPE      CR";
    println!("{h1}");
    println!("  {}", "-".repeat(h1.len() - 2));
    for i in 0..10 {
        let bpe = (k_n(i).bpe() + v_n(i).bpe()) / 2.0;
        let mut line = format!(
            "  C{i}  {:<9} {:<7.5} ",
            LEVEL_TIER[i], K_MAGWEIGHT_THRESHOLDS[i]
        );
        for j in 0..16 {
            line += &format!(" {:>4.1}", k_n(i).pct(j));
        }
        line += " ";
        for j in 0..16 {
            line += &format!(" {:>4.1}", v_n(i).pct(j));
        }
        line += &format!("  {:>6.3}  {:>5.2}x", bpe, cr_naive(i));
        println!("{line}");
    }

    // ── Table 2: Sink-aware selection ─────────────────────────────────────────
    println!("\n{sep80}");
    println!("Table 2 -- Sink-aware");
    println!("  CR_cost = cr_sink - cr_naive  (negative = slightly lower CR due to protection).");
    println!();
    let h2 = "  Mode  Tier       Thr      KQ0 KQ1S KQ2S KQ20 KQ2A KQ21 KQ30 KQ31 KQ40 KQ41 KQ4K KQ80 KQ81 KQ8K KBFL KF16    VQ0 VQ1S VQ2S VQ20 VQ2A VQ21 VQ30 VQ31 VQ40 VQ41 VQ4K VQ80 VQ81 VQ8K VBFL VF16   BPE      CR   CR_cost";
    println!("{h2}");
    println!("  {}", "-".repeat(h2.len() - 2));
    for i in 0..10 {
        let bpe = (k_s(i).bpe() + v_s(i).bpe()) / 2.0;
        let cost = cr_sink(i) - cr_naive(i);
        let mut line = format!(
            "  C{i}  {:<9} {:<7.5} ",
            LEVEL_TIER[i], K_MAGWEIGHT_THRESHOLDS[i]
        );
        for j in 0..16 {
            line += &format!(" {:>4.1}", k_s(i).pct(j));
        }
        line += " ";
        for j in 0..16 {
            line += &format!(" {:>4.1}", v_s(i).pct(j));
        }
        line += &format!("  {:>6.3}  {:>5.2}x  {:>+6.3}x", bpe, cr_sink(i), cost);
        println!("{line}");
    }

    // ── Table 3: Quality metrics ───────────────────────────────────────────────
    println!("\n{sep80}");
    println!("Table 3 -- Quality metrics (naive selection, all blocks including zero-pad)");
    println!("  SNR     = 10*log10(sum x^2 / sum err^2)  [dB]");
    println!("  NRMSE   = sqrt(sum err^2 / sum x^2 )     [ ]");
    println!(
        "  cos_mu  = mean cosine distance  (reported for reference; skewed by 18.8% zero-pad)"
    );
    println!("  cos_p95 = 95th-percentile cosine distance  (skew-resistant; preferred metric)");
    println!("  dSNR    = SNR(sink-aware) - SNR(naive)  [dB]  gain from sink protection");
    println!();
    let h3 = "  Mode  Tier        K_SNR_dB   K_NRMSE  K_cos_mu   K_cos_p95    V_SNR_dB   V_NRMSE  V_cos_mu   V_cos_p95    dSNR_K";
    println!("{h3}");
    println!("  {}", "-".repeat(h3.len() - 2));
    for i in 0..11 {
        let dk = k_s(i).snr_db() - k_n(i).snr_db();
        let dks = if dk.is_finite() && dk.abs() > 0.001 {
            format!("{:>+8.2}", dk)
        } else {
            "      --".to_string()
        };
        let snr_fmt = |x: f64| {
            if x.is_infinite() {
                "      inf".to_string()
            } else {
                format!("{x:>9.2}")
            }
        };
        println!(
            "  C{i}  {:<9}  {} {:>9.5} {:>9.6} {:>11.6}   {} {:>9.5} {:>9.6} {:>11.6}  {}",
            LEVEL_TIER[i],
            snr_fmt(k_n(i).snr_db()),
            k_n(i).nrmse(),
            k_n(i).cos_mu(),
            k_n(i).p95(),
            snr_fmt(v_n(i).snr_db()),
            v_n(i).nrmse(),
            v_n(i).cos_mu(),
            v_n(i).p95(),
            dks,
        );
    }

    // ── Threshold migration table ──────────────────────────────────────────────
    println!("\n{sep80}");
    let old = [
        0.0001f32, 0.0002, 0.0005, 0.0010, 0.0020, 0.0050, 0.0100, 0.0200, 0.0500, 0.1000,
    ];
    let note = [
        "0.0001->0.000014: below V_mean; most real blocks → BF16 reference",
        "0.0002->0.000035: K_mean<thr<K_p95; ~30% BF16 near-lossless blend",
        "0.0005->0.000065: K_p95<thr<K_max; ~5% BF16 tuned Q8_0",
        "0.0010->0.0025: bridge dead zone, sweet entry",
        "0.0020->0.0050: align with validated sweet core",
        "0.0050->0.0090: even sweet-spot spacing",
        "0.0100->0.0160: sweet ceiling, Q3_0 entering",
        "0.0200->0.0360: raised from 0.033; K_p95=0.030 → p95+0.006 headroom",
        "0.0500->0.0750: Q3_0 heavy compression (unchanged)",
        "0.1000->0.1500: Q2_0 enters non-zero, max CR (unchanged)",
    ];
    println!("Threshold migration:  old COSINE_THRESHOLDS  ->  K_MAGWEIGHT_THRESHOLDS");
    println!("  {:<5}  {:<9}  {:<9}  Note", "Mode", "Old", "Proposed");
    println!("  {}", "-".repeat(70));
    for (i, (&o, &p)) in old.iter().zip(K_MAGWEIGHT_THRESHOLDS.iter()).enumerate() {
        println!("  C{i}     {o:<9.5}  {p:<9.5}  {}", note[i]);
    }

    // ── Copy-paste constants ───────────────────────────────────────────────────
    println!("\n{sep80}");
    println!("// Paste into CUDA kernel / mod kv_cache:");
    println!("const COSINE_THRESHOLDS: [f32; 10] = [");
    for (i, &t) in K_MAGWEIGHT_THRESHOLDS.iter().enumerate() {
        println!("    {:.5}, // C{i} {}", t, LEVEL_TIER[i].trim());
    }
    println!("];");
    println!("{sep}");
}

// ============================================================================
// CUDA kernel vs CPU model comparison test
// ============================================================================

/// Map a CUDA kernel format tag (integer) back to a [`BlockFormat`].
/// Codes match SELECT_FMT_* defines in select_kv_format.cuh (GgmlDType-aligned).
/// Returns `BlockFormat::F16` for any unknown tag.
#[cfg(feature = "cuda")]
fn tag_to_block_format(tag: i32) -> BlockFormat {
    match tag {
        1 => BlockFormat::F16,      // SELECT_FMT_F16
        2 => BlockFormat::BF16,     // SELECT_FMT_BF16
        7 => BlockFormat::Q8_0,     // SELECT_FMT_Q8_0
        8 => BlockFormat::Q8_1,     // SELECT_FMT_Q8_1
        10 => BlockFormat::Q8_KS,   // SELECT_FMT_Q8_KS
        15 => BlockFormat::Q4_0,    // SELECT_FMT_Q4_0
        16 => BlockFormat::Q4_1,    // SELECT_FMT_Q4_1
        18 => BlockFormat::Q4_KS,   // SELECT_FMT_Q4_KS
        19 => BlockFormat::Q3_0,    // SELECT_FMT_Q3_0
        20 => BlockFormat::Q3_1,    // SELECT_FMT_Q3_1
        22 => BlockFormat::Q2_0,    // SELECT_FMT_Q2_0
        23 => BlockFormat::Q2_1,    // SELECT_FMT_Q2_1
        25 => BlockFormat::Q2_S, // SELECT_FMT_Q2_S
        26 => BlockFormat::Q2_A, // SELECT_FMT_Q2_A
        27 => BlockFormat::Q1_S, // SELECT_FMT_Q1_S
        28 => BlockFormat::Q0_V,    // SELECT_FMT_Q0_V
        29 => BlockFormat::Q1_A,    // SELECT_FMT_Q1_A
        30 => BlockFormat::Q0_X,    // SELECT_FMT_Q0_X
        31 => BlockFormat::Q0_M2,   // SELECT_FMT_Q0_M2
        32 => BlockFormat::Q0_M4,   // SELECT_FMT_Q0_M4
        33 => BlockFormat::Q0,      // SELECT_FMT_Q0
        _ => BlockFormat::F16,      // unknown
    }
}

/// Format-ladder step distance: number of positions between two formats.
/// F16 and BF16 are both "float fallback" so distance between them is 0.
#[cfg(feature = "cuda")]
fn ladder_distance(a: BlockFormat, b: BlockFormat) -> usize {
    let is_float = |f: BlockFormat| matches!(f, BlockFormat::F16 | BlockFormat::BF16);
    if is_float(a) && is_float(b) {
        return 0;
    }
    // Ordered least-aggressive → most-aggressive
    let rank = |f: BlockFormat| match f {
        BlockFormat::F16 | BlockFormat::BF16 => 0usize,
        BlockFormat::Q8_KS => 1,
        BlockFormat::Q8_1 => 2,
        BlockFormat::Q8_0 => 3,
        BlockFormat::Q4_KS => 4,
        BlockFormat::Q4_1 => 5,
        BlockFormat::Q4_0 => 6,
        BlockFormat::Q3_1 => 7,
        BlockFormat::Q3_0 => 10,
        BlockFormat::Q2_1 => 11,
        BlockFormat::Q2_A => 12,
        BlockFormat::Q2_S => 13,
        BlockFormat::Q2_0 => 14,
        BlockFormat::Q0_M4 => 15,
        BlockFormat::Q1_S => 16,
        BlockFormat::Q0_M2 => 17,
        BlockFormat::Q0_V => 18,
        BlockFormat::Q1_A => 19,
        BlockFormat::Q0_X => 20,
        BlockFormat::Q0 => 21,
    };
    rank(a).abs_diff(rank(b))
}

/// Verify that the CUDA `select_kv_format` kernel selects the same format
/// per block as the CPU reference model, using the **per-level candidate lists**
/// from `level_candidates()`.
///
/// This tests the actual selection kernel on real KV cache data with the exact
/// same candidate lists used in production.  For each C-level:
///   - CPU: `select_format_from_candidates(block, k_candidates, threshold)`
///   - CUDA: `select_kv_format(k_gpu, v_gpu, k_ggml, v_ggml, threshold, ...)`
///
/// The test also prints a side-by-side format-distribution table (CPU vs CUDA)
/// so discrepancies in compression ratio are immediately visible.
///
/// Run with:
///   cargo test --release --features cuda --lib --package candle-nn \
///     test_cuda_selection_matches_cpu -- --ignored --nocapture
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_cuda_selection_matches_cpu() {
    use candle::quantized::{cuda::select_kv_format_paged_batched_raw, GgmlDType};

    // Convert BlockFormat → GgmlDType for passing to CUDA kernel.
    // Float entries (F16, BF16) are included — the CUDA kernel handles them.
    let bf_to_ggml = |bf: BlockFormat| -> GgmlDType {
        match bf {
            BlockFormat::F16 => GgmlDType::F16,
            BlockFormat::BF16 => GgmlDType::BF16,
            BlockFormat::Q8_KS => GgmlDType::Q8_KS,
            BlockFormat::Q8_1 => GgmlDType::Q8_1,
            BlockFormat::Q8_0 => GgmlDType::Q8_0,
            BlockFormat::Q4_KS => GgmlDType::Q4_KS,
            BlockFormat::Q4_1 => GgmlDType::Q4_1,
            BlockFormat::Q4_0 => GgmlDType::Q4_0,
            BlockFormat::Q3_1 => GgmlDType::Q3_1,
            BlockFormat::Q3_0 => GgmlDType::Q3_0,
            BlockFormat::Q2_1 => GgmlDType::Q2_1,
            BlockFormat::Q2_A => GgmlDType::Q2_A,
            BlockFormat::Q2_S => GgmlDType::Q2_S,
            BlockFormat::Q2_0 => GgmlDType::Q2_0,
            BlockFormat::Q1_S => GgmlDType::Q1_S,
            BlockFormat::Q0 => GgmlDType::Q0,
            BlockFormat::Q0_V => GgmlDType::Q0_V,
            BlockFormat::Q1_A => GgmlDType::Q1_A,
            BlockFormat::Q0_X => GgmlDType::Q0_X,
            BlockFormat::Q0_M2 => GgmlDType::Q0_M2,
            BlockFormat::Q0_M4 => GgmlDType::Q0_M4,
        }
    };

    // Per-level candidate lists (same as batched_inference.rs)
    let candidates = level_candidates();

    // ── Load dump ──────────────────────────────────────────────────────────────
    let path = match dump_path() {
        Some(p) => p,
        None => {
            println!("SKIP: dump file not found at {QWEN3_DUMP_REL_PATH}");
            return;
        }
    };
    let (header, chunks) = match load_dump(&path) {
        Some(v) => v,
        None => {
            println!("SKIP: failed to parse dump");
            return;
        }
    };
    println!(
        "Loaded dump: {} layers, n_kv_head={}, chunk_size={}, head_dim={}, {} chunks total",
        header.num_layers,
        header.n_kv_head,
        header.chunk_size,
        header.head_dim,
        chunks.len()
    );

    // ── Get CUDA device ────────────────────────────────────────────────────────
    let dev = candle::Device::cuda_if_available(0).expect("cuda_if_available");
    let cuda_dev = match &dev {
        candle::Device::Cuda(d) => d.clone(),
        _ => {
            println!("SKIP: no CUDA device available");
            return;
        }
    };

    let sep = "─".repeat(120);
    println!("\n{sep}");
    println!("  CUDA vs CPU format-selection comparison  (per-level candidates, sink-aware)");
    println!("{sep}");

    let mut grand_total = 0usize;
    let mut grand_mismatches = 0usize;
    let mut _grand_non_adjacent = 0usize;

    // Per-level format counts: [level][k=0/v=1][format_index]
    let mut cpu_counts = vec![[[0usize; 16]; 2]; 10];
    let mut cuda_counts = vec![[[0usize; 16]; 2]; 10];
    let mut cpu_bpe_sum = vec![[0.0f64; 2]; 10];
    let mut cuda_bpe_sum = vec![[0.0f64; 2]; 10];
    let mut block_count = vec![[0usize; 2]; 10];

    // Mismatch tracking per level
    let mut level_total = vec![0usize; 10];
    let mut level_mismatches = vec![0usize; 10];
    let mut level_non_adjacent = vec![0usize; 10];
    let mut level_first_mismatches: Vec<Vec<String>> = vec![Vec::new(); 10];

    // Boundary-margin tracking: for every mismatch, measure how close the
    // block's cosine distance was to the threshold.  If the margin is tiny
    // (< epsilon) we know it's genuine FP noise.  If any mismatch has a large
    // margin, that's a real bug — the kernel made a qualitatively wrong decision.
    //
    // We distinguish two mismatch types:
    //   Type A (BPE-tie): CPU and CUDA chose different formats with identical
    //     bits-per-element.  Both formats pass the threshold, so either choice
    //     is equally valid.  These are harmless tie-breaks, not bugs.
    //   Type B (BPE-diff): CPU and CUDA chose formats with different BPE.
    //     One side accepted a format the other rejected.  The margin measures
    //     how close the disputed format was to the threshold.
    //
    // Only Type B mismatches are checked against the margin envelope.
    let mut level_max_margin = vec![0.0f64; 10]; // worst margin per level
    let mut level_margin_sum = vec![0.0f64; 10]; // for computing mean
    let mut level_margin_count = vec![0usize; 10];
    let mut level_bpe_ties = vec![0usize; 10]; // Type A count
    let mut worst_margin_example: Vec<String> = vec![String::new(); 10];

    // ── Upload ALL chunk K/V data to GPU once ──────────────────────────────────
    // Each chunk becomes its own "arena" in the arena table.
    // Arena table layout: 3 × i64 per arena = (k_ptr, v_ptr, metadata).
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let stream = cuda_dev.cuda_stream();
    struct ChunkGpu {
        k_gpu: candle::cuda_backend::cudarc::driver::CudaSlice<f32>,
        v_gpu: candle::cuda_backend::cudarc::driver::CudaSlice<f32>,
        num_blocks: usize,
        is_sink: bool,
    }
    let chunk_gpus: Vec<ChunkGpu> = chunks
        .iter()
        .map(|chunk| {
            let k_gpu = cuda_dev.memcpy_stod(&chunk.k).expect("GPU upload k");
            let v_gpu = cuda_dev.memcpy_stod(&chunk.v).expect("GPU upload v");
            let num_blocks = chunk.k.len() / SELECT_BLOCK;
            let is_sink = chunk.token_start == 0;
            ChunkGpu {
                k_gpu,
                v_gpu,
                num_blocks,
                is_sink,
            }
        })
        .collect();

    // Build per-head table: one arena per chunk, n_kv_head=1, F32 format.
    // Each arena has: k_ptr, v_ptr, k_byte_offset=0, v_byte_offset=0,
    // k_chunk_byte_stride = num_blocks * 32 * 4 (F32), same for V,
    // metadata = 0 (F32 format, GPU location).
    let blocks_per_chunk = chunk_gpus[0].num_blocks;
    assert!(
        chunk_gpus
            .iter()
            .all(|cg| cg.num_blocks == blocks_per_chunk),
        "All chunks must have the same number of blocks for the paged kernel"
    );
    let chunk_byte_stride = (blocks_per_chunk * 32 * 4) as i64; // F32: 4 bytes per elem

    let per_head_table_host: Vec<i64> = chunk_gpus
        .iter()
        .map(|cg| {
            let (k_ptr, _) = cg.k_gpu.device_ptr(&stream);
            let (v_ptr, _) = cg.v_gpu.device_ptr(&stream);
            // PerHeadTableEntry: [k_ptr, v_ptr, k_byte_offset, v_byte_offset,
            //                     k_chunk_byte_stride, v_chunk_byte_stride, metadata]
            // metadata: (k_format_tag << 16) | (v_format_tag << 8) | location
            // ArenaFormat::F32 = 0, so metadata = 0
            [
                k_ptr as i64,
                v_ptr as i64,
                0i64,
                0i64,
                chunk_byte_stride,
                chunk_byte_stride,
                0i64,
            ]
        })
        .flatten()
        .collect();
    let per_head_table_gpu = cuda_dev
        .memcpy_stod(&per_head_table_host)
        .expect("per-head table upload");

    // Build chunk descriptors: each chunk is its own arena at chunk_idx=0.
    // GID = arena_idx * ARENA_CHUNKS + chunk_idx = arena_idx * ARENA_CHUNKS + 0
    // n_kv_head=1, so head_gids = [K_GID, V_GID] per chunk.
    const TEST_ARENA_CHUNKS: i64 = 8192; // matches ARENA_CHUNKS in types.rs
    let mut head_gids: Vec<i64> = Vec::with_capacity(chunk_gpus.len() * 2);
    for (i, _cg) in chunk_gpus.iter().enumerate() {
        head_gids.push(i as i64 * TEST_ARENA_CHUNKS); // K GID: arena_idx=i, chunk_idx=0
        head_gids.push(i as i64 * TEST_ARENA_CHUNKS); // V GID: same arena
    }

    let _total_blocks = chunks.len() * blocks_per_chunk;

    for level in 0..10 {
        let base_k_threshold = K_MAGWEIGHT_THRESHOLDS[level];
        let base_v_threshold = V_COSINE_THRESHOLDS_PROPOSED[level];
        let (ref k_cands, ref v_cands) = candidates[level];

        // Convert BlockFormat candidates to GgmlDType for CUDA kernel.
        // The CUDA kernel only evaluates quantized formats; float entries
        // (F16/BF16) produce the fallback sentinel (99/21).
        // Filter to quant-only for CUDA, matching what chunk_ops does.
        let k_ggml: Vec<GgmlDType> = k_cands
            .iter()
            .filter(|f| !matches!(f, BlockFormat::F16 | BlockFormat::BF16))
            .map(|f| bf_to_ggml(*f))
            .collect();
        let v_ggml: Vec<GgmlDType> = v_cands
            .iter()
            .filter(|f| !matches!(f, BlockFormat::F16 | BlockFormat::BF16))
            .map(|f| bf_to_ggml(*f))
            .collect();

        // ── CPU reference (all chunks at this level) ──
        // Collect per-chunk CPU format decisions for comparison.
        let mut all_cpu_k_fmts: Vec<Vec<BlockFormat>> = Vec::with_capacity(chunks.len());
        let mut all_cpu_v_fmts: Vec<Vec<BlockFormat>> = Vec::with_capacity(chunks.len());

        for chunk in &chunks {
            let num_blocks = chunk.k.len() / SELECT_BLOCK;
            let eff_k_threshold = base_k_threshold;
            let eff_v_threshold = base_v_threshold;

            let mut cpu_k_fmts: Vec<BlockFormat> = Vec::with_capacity(num_blocks);
            let mut cpu_v_fmts: Vec<BlockFormat> = Vec::with_capacity(num_blocks);
            for b in 0..num_blocks {
                let k_block: [f32; SELECT_BLOCK] = chunk.k
                    [b * SELECT_BLOCK..(b + 1) * SELECT_BLOCK]
                    .try_into()
                    .unwrap();
                let v_block: [f32; SELECT_BLOCK] = chunk.v
                    [b * SELECT_BLOCK..(b + 1) * SELECT_BLOCK]
                    .try_into()
                    .unwrap();
                let kf = select_format_from_candidates_k(&k_block, k_cands, eff_k_threshold);
                let vf = select_format_from_candidates(&v_block, v_cands, eff_v_threshold);
                cpu_k_fmts.push(kf);
                cpu_v_fmts.push(vf);

                cpu_counts[level][0][kf.table_index()] += 1;
                cpu_counts[level][1][vf.table_index()] += 1;
                cpu_bpe_sum[level][0] += kf.bits_per_elem() as f64;
                cpu_bpe_sum[level][1] += vf.bits_per_elem() as f64;
                block_count[level][0] += 1;
                block_count[level][1] += 1;
            }
            all_cpu_k_fmts.push(cpu_k_fmts);
            all_cpu_v_fmts.push(cpu_v_fmts);
        }

        // ── CUDA paged kernel (single launch for ALL chunks at this level) ──
        let (k_tags_gpu, v_tags_gpu) = select_kv_format_paged_batched_raw(
            &per_head_table_gpu,
            &head_gids,
            &k_ggml,
            &v_ggml,
            base_k_threshold,
            base_k_threshold,
            base_v_threshold,
            base_v_threshold,
            None, // no Q2_0 promotion in this test
            blocks_per_chunk,
            1, // n_kv_head=1 in this test
            TEST_ARENA_CHUNKS as usize,
            7,
            &cuda_dev,
        )
        .expect("select_kv_format_palette4_paged CUDA");
        let all_k_tags: Vec<i32> = cuda_dev.memcpy_dtov(&k_tags_gpu).expect("GPU download k");
        let all_v_tags: Vec<i32> = cuda_dev.memcpy_dtov(&v_tags_gpu).expect("GPU download v");

        // ── Compare (per chunk, per block) ──
        for (ci, chunk) in chunks.iter().enumerate() {
            let num_blocks = chunk.k.len() / SELECT_BLOCK;
            let eff_k_threshold = base_k_threshold;
            let eff_v_threshold = base_v_threshold;
            let cpu_k_fmts = &all_cpu_k_fmts[ci];
            let cpu_v_fmts = &all_cpu_v_fmts[ci];
            let tag_offset = ci * blocks_per_chunk;

            for b in 0..num_blocks {
                let cuda_k = tag_to_block_format(all_k_tags[tag_offset + b]);
                let cuda_v = tag_to_block_format(all_v_tags[tag_offset + b]);

                cuda_counts[level][0][cuda_k.table_index()] += 1;
                cuda_counts[level][1][cuda_v.table_index()] += 1;
                cuda_bpe_sum[level][0] += cuda_k.bits_per_elem() as f64;
                cuda_bpe_sum[level][1] += cuda_v.bits_per_elem() as f64;

                level_total[level] += 2;

                // F16 (CUDA sentinel) and BF16 (CPU float fallback) both mean
                // "stay float — don't quantize this block".  Treat as equivalent.
                let is_float = |f: BlockFormat| matches!(f, BlockFormat::F16 | BlockFormat::BF16);
                let k_match =
                    cuda_k == cpu_k_fmts[b] || (is_float(cuda_k) && is_float(cpu_k_fmts[b]));
                let v_match =
                    cuda_v == cpu_v_fmts[b] || (is_float(cuda_v) && is_float(cpu_v_fmts[b]));

                if !k_match {
                    level_mismatches[level] += 1;
                    let dist = ladder_distance(cuda_k, cpu_k_fmts[b]);
                    if dist > 1 {
                        level_non_adjacent[level] += 1;
                    }
                    // Classify: BPE-tie (Type A) vs BPE-diff (Type B)
                    let cpu_bpe = cpu_k_fmts[b].bits_per_elem();
                    let cuda_bpe = cuda_k.bits_per_elem();
                    if (cpu_bpe - cuda_bpe).abs() < 0.01 {
                        // Type A: same BPE, tie-break difference — harmless
                        level_bpe_ties[level] += 1;
                        if level_first_mismatches[level].len() < 5 {
                            level_first_mismatches[level].push(format!(
                                "    L{} tok{} blk{} K: CPU={}({:.1}bpe) CUDA={}({:.1}bpe) [BPE-TIE]",
                                chunk.layer_idx, chunk.token_start, b,
                                cpu_k_fmts[b], cpu_bpe, cuda_k, cuda_bpe
                            ));
                        }
                    } else {
                        // Type B: different BPE — compute boundary margin
                        let k_block: [f32; SELECT_BLOCK] = chunk.k
                            [b * SELECT_BLOCK..(b + 1) * SELECT_BLOCK]
                            .try_into()
                            .unwrap();
                        let err_of = |fmt: BlockFormat| -> f32 {
                            if matches!(fmt, BlockFormat::F16 | BlockFormat::BF16) {
                                return 0.0;
                            }
                            let recon = fmt.apply_quant(&k_block);
                            magnitude_weighted_distance(&k_block, &recon)
                        };
                        let cpu_err = err_of(cpu_k_fmts[b]);
                        let cuda_err = err_of(cuda_k);
                        let thr = eff_k_threshold as f64;
                        let margin_cpu = ((cpu_err as f64 - thr) / thr).abs();
                        let margin_cuda = ((cuda_err as f64 - thr) / thr).abs();
                        let margin = margin_cpu.min(margin_cuda);
                        level_margin_sum[level] += margin;
                        level_margin_count[level] += 1;
                        if margin > level_max_margin[level] {
                            level_max_margin[level] = margin;
                            worst_margin_example[level] = format!(
                                "    L{} tok{} blk{} K: CPU={}(mw={:.8},{:.1}bpe) CUDA={}(mw={:.8},{:.1}bpe) thr={:.8} margin={:.6}",
                                chunk.layer_idx, chunk.token_start, b,
                                cpu_k_fmts[b], cpu_err, cpu_bpe, cuda_k, cuda_err, cuda_bpe, eff_k_threshold, margin
                            );
                        }
                        if level_first_mismatches[level].len() < 5 {
                            level_first_mismatches[level].push(format!(
                                "    L{} tok{} blk{} K: CPU={}(mw={:.8},{:.1}bpe) CUDA={}(mw={:.8},{:.1}bpe) thr={:.8} margin={:.6}",
                                chunk.layer_idx, chunk.token_start, b,
                                cpu_k_fmts[b], cpu_err, cpu_bpe, cuda_k, cuda_err, cuda_bpe, eff_k_threshold, margin
                            ));
                        }
                    }
                }
                if !v_match {
                    level_mismatches[level] += 1;
                    let dist = ladder_distance(cuda_v, cpu_v_fmts[b]);
                    if dist > 1 {
                        level_non_adjacent[level] += 1;
                    }
                    let cpu_bpe = cpu_v_fmts[b].bits_per_elem();
                    let cuda_bpe = cuda_v.bits_per_elem();
                    if (cpu_bpe - cuda_bpe).abs() < 0.01 {
                        level_bpe_ties[level] += 1;
                        if level_first_mismatches[level].len() < 5 {
                            level_first_mismatches[level].push(format!(
                                "    L{} tok{} blk{} V: CPU={}({:.1}bpe) CUDA={}({:.1}bpe) [BPE-TIE]",
                                chunk.layer_idx, chunk.token_start, b,
                                cpu_v_fmts[b], cpu_bpe, cuda_v, cuda_bpe
                            ));
                        }
                    } else {
                        // Type B: different BPE — compute boundary margin
                        let v_block: [f32; SELECT_BLOCK] = chunk.v
                            [b * SELECT_BLOCK..(b + 1) * SELECT_BLOCK]
                            .try_into()
                            .unwrap();
                        let cos_of = |fmt: BlockFormat| -> f32 {
                            if matches!(fmt, BlockFormat::F16 | BlockFormat::BF16) {
                                return 0.0;
                            }
                            let recon = fmt.apply_quant(&v_block);
                            cosine_distance(&v_block, &recon)
                        };
                        let cpu_cos = cos_of(cpu_v_fmts[b]);
                        let cuda_cos = cos_of(cuda_v);
                        let thr = eff_v_threshold as f64;
                        let margin_cpu = ((cpu_cos as f64 - thr) / thr).abs();
                        let margin_cuda = ((cuda_cos as f64 - thr) / thr).abs();
                        let margin = margin_cpu.min(margin_cuda);
                        level_margin_sum[level] += margin;
                        level_margin_count[level] += 1;
                        if margin > level_max_margin[level] {
                            level_max_margin[level] = margin;
                            worst_margin_example[level] = format!(
                                "    L{} tok{} blk{} V: CPU={}(cos={:.8},{:.1}bpe) CUDA={}(cos={:.8},{:.1}bpe) thr={:.8} margin={:.6}",
                                chunk.layer_idx, chunk.token_start, b,
                                cpu_v_fmts[b], cpu_cos, cpu_bpe, cuda_v, cuda_cos, cuda_bpe, eff_v_threshold, margin
                            );
                        }
                        if level_first_mismatches[level].len() < 5 {
                            level_first_mismatches[level].push(format!(
                                "    L{} tok{} blk{} V: CPU={}(cos={:.8},{:.1}bpe) CUDA={}(cos={:.8},{:.1}bpe) thr={:.8} margin={:.6}",
                                chunk.layer_idx, chunk.token_start, b,
                                cpu_v_fmts[b], cpu_cos, cpu_bpe, cuda_v, cuda_cos, cuda_bpe, eff_v_threshold, margin
                            ));
                        }
                    }
                }
            }
        }

        grand_total += level_total[level];
        grand_mismatches += level_mismatches[level];
        _grand_non_adjacent += level_non_adjacent[level];
    }

    // ── Table 1: Mismatch summary ──
    println!("\nTable 1 — Per-block decision comparison");
    println!(
        "  {:4}  {:9}  {:>9}  {:>10}  {:>10}  {:>4}  {:>7}",
        "Mode", "K_Thr/V_Thr", "Decisions", "Mismatches", "Mismatch%", "NonAdj", "Status"
    );
    println!("  {}", "─".repeat(76));
    for level in 0..10 {
        let pct = 100.0 * level_mismatches[level] as f64 / level_total[level].max(1) as f64;
        let status = if pct > 1.0 {
            "HIGH"
        } else if pct < 0.5 {
            if level_mismatches[level] == 0 {
                "PASS"
            } else {
                "FP-OK"
            }
        } else {
            "WARN"
        };
        let thr_str = if K_MAGWEIGHT_THRESHOLDS[level] == V_COSINE_THRESHOLDS_PROPOSED[level] {
            format!("{:.6}", K_MAGWEIGHT_THRESHOLDS[level])
        } else {
            format!(
                "{:.5}/{:.5}",
                K_MAGWEIGHT_THRESHOLDS[level], V_COSINE_THRESHOLDS_PROPOSED[level]
            )
        };
        println!(
            "  C{level}   {:>13}  {:>8}  {:>10}  {:>9.4}%  {:>4}  {status:>7}",
            thr_str, level_total[level], level_mismatches[level], pct, level_non_adjacent[level],
        );
        if !level_first_mismatches[level].is_empty() {
            for m in &level_first_mismatches[level] {
                println!("{m}");
            }
        }
    }
    let grand_pct = 100.0 * grand_mismatches as f64 / grand_total.max(1) as f64;
    println!("  {}", "─".repeat(66));
    println!("  TOTAL: {grand_total} decisions, {grand_mismatches} mismatches ({grand_pct:.4}%)");

    // ── Table 2: Format distribution comparison (CPU vs CUDA) ──
    let fmt_labels = [
        "Q0", "Q1S", "Q2S", "Q20", "Q2A", "Q21", "Q30", "Q31", "Q40", "Q41", "Q4K", "Q80", "Q81",
        "Q8K", "BF", "F16",
    ];
    println!("\n{sep}");
    println!("Table 2 — Format distribution CPU vs CUDA  (% of blocks)");
    println!();
    let mut hdr = format!("  {:4} {:5} ", "Mode", "Side");
    for l in &fmt_labels {
        hdr += &format!(" K{:<3}", l);
    }
    hdr += " ";
    for l in &fmt_labels {
        hdr += &format!(" V{:<3}", l);
    }
    hdr += &format!("  {:>6}  {:>5}", "BPE", "CR");
    println!("{hdr}");
    println!("  {}", "─".repeat(hdr.len() - 2));
    for level in 0..10 {
        let nk = block_count[level][0].max(1) as f64;
        let nv = block_count[level][1].max(1) as f64;
        for (label, counts, bpe_s) in [
            ("CPU", &cpu_counts, &cpu_bpe_sum),
            ("CUDA", &cuda_counts, &cuda_bpe_sum),
        ] {
            let bpe = (bpe_s[level][0] + bpe_s[level][1]) / (nk + nv);
            let cr = 16.0 / bpe;
            let mut line = format!("  C{level}  {label:<5}");
            for j in 0..16 {
                line += &format!(" {:>4.1}", counts[level][0][j] as f64 / nk * 100.0);
            }
            line += " ";
            for j in 0..16 {
                line += &format!(" {:>4.1}", counts[level][1][j] as f64 / nv * 100.0);
            }
            line += &format!("  {:>6.2}  {:>4.2}×", bpe, cr);
            println!("{line}");
        }
    }
    println!("{sep}");
    println!();
    println!("  Legend: PASS=zero mismatch, FP-OK=boundary FP only (<0.5%), WARN=0.5-1%, HIGH=>1%");
    println!("{sep}");

    // ── Table 3: Boundary margin analysis ──
    println!("\nTable 3 — Boundary margin analysis (mismatched blocks only)");
    println!(
        "  {:4}  {:>10}  {:>8}  {:>8}  {:>12}  {:>12}  {}",
        "Mode", "Mismatches", "BPE-Tie", "BPE-Diff", "Mean_Margin", "Max_Margin", "Worst example"
    );
    println!("  {}", "─".repeat(110));
    let mut global_max_margin = 0.0f64;
    let mut global_worst_example = String::new();
    for level in 0..10 {
        let total_mm = level_mismatches[level];
        let ties = level_bpe_ties[level];
        let diffs = level_margin_count[level];
        if total_mm == 0 {
            println!("  C{level}          0        0        0           —            —");
            continue;
        }
        let mean_margin = if diffs > 0 {
            level_margin_sum[level] / diffs as f64
        } else {
            0.0
        };
        let max_margin = level_max_margin[level];
        if max_margin > global_max_margin {
            global_max_margin = max_margin;
            global_worst_example = worst_margin_example[level].clone();
        }
        println!(
            "  C{level}   {:>8}    {:>5}    {:>5}    {mean_margin:>10.6}    {max_margin:>10.6}",
            total_mm, ties, diffs,
        );
        if !worst_margin_example[level].is_empty() && diffs > 0 {
            println!("{}", worst_margin_example[level]);
        }
    }
    println!("  {}", "─".repeat(110));
    let total_ties: usize = level_bpe_ties.iter().sum();
    let total_diffs: usize = level_margin_count.iter().sum();
    println!("  BPE-Tie (harmless): {total_ties}   BPE-Diff (checked): {total_diffs}   Global max margin: {global_max_margin:.6}");
    if !global_worst_example.is_empty() {
        println!("{global_worst_example}");
    }
    println!("{sep}");

    // Hard assertions.
    //
    // The CUDA kernel uses IEEE-precise intrinsics (__fdiv_rn, __fmul_rn,
    // __fsqrt_rn) throughout to override --use_fast_math's approximate
    // division, sqrt, and FMA contraction.  Sequential lane-by-lane cosine
    // accumulation matches the CPU's left-to-right loop order.
    //
    // Residual divergence (< 7%) comes from flush-to-zero (--ftz=true) on
    // denormal intermediates, which is harmless for format selection.
    //
    // Guards:
    // (1) total mismatch rate < 0.5%
    // (2) boundary margin < 10% — catches algorithmic bugs while allowing
    //     residual FTZ-induced divergence
    let grand_pct_val = 100.0 * grand_mismatches as f64 / grand_total.max(1) as f64;
    assert!(
        grand_pct_val < 0.5,
        "Mismatch rate {grand_pct_val:.4}% exceeds 0.5% tolerance"
    );
    // Safety envelope: 10% catches real bugs (a truly wrong decision would
    // show margin >> 100%) while accommodating denormal flush-to-zero effects.
    assert!(
        global_max_margin < 0.10,
        "Mismatched block has cosine distance {:.6}× away from threshold — \
         this exceeds the 10% FP-noise envelope and may indicate a kernel bug.\n\
         Worst example:\n{global_worst_example}",
        global_max_margin
    );
}

// ============================================================================
// CUDA vs CPU comparison — per-head reduction kernel validation
// ============================================================================
//
// Validates the full `select_kv_format_paged_per_head` pipeline:
//   Step 1: per-block format selection  (same kernel as test_cuda_selection_matches_cpu)
//   Step 2: reduce_head_format — worst-case-reduces each head to one format tag
//
// CPU reference:
//   1. Per-block selection with the same candidate lists and thresholds
//   2. Worst-case per-head reduction: max(table_index) over all blocks in head
//      (F16 has the highest table_index, so one F16 block → whole head = F16,
//       matching the CUDA kernel's SELECT_FMT_F16-sentinel propagation)
//
// Comparison granularity: one format tag per (chunk × head).
// In this test n_kv_head=1, so there is one head-tag per chunk.
//
// Mismatches here have two possible causes:
//   A. Per-block FP noise (known <0.5% from test_cuda_selection_matches_cpu)
//      propagating after worst-case reduction — a single FP-noise block that
//      flips to F16 forces the whole head to F16.
//   B. A real bug in reduce_head_format (e.g. wrong traversal of candidates).
//
// The test prints:
//   Table 1: per-level head mismatch rate
//   Table 2: CPU vs GPU per-head format distribution + BPE/CR
//
// Run with:
//   cargo test --release --features cuda --lib --package candle-nn \
//     kv_selection_tests::test_cuda_per_head_matches_cpu -- --ignored --nocapture
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_cuda_per_head_matches_cpu() {
    use candle::quantized::{cuda::select_kv_format_paged_per_head, GgmlDType};

    let bf_to_ggml = |bf: BlockFormat| -> GgmlDType {
        match bf {
            BlockFormat::F16 => GgmlDType::F16,
            BlockFormat::BF16 => GgmlDType::BF16,
            BlockFormat::Q8_KS => GgmlDType::Q8_KS,
            BlockFormat::Q8_1 => GgmlDType::Q8_1,
            BlockFormat::Q8_0 => GgmlDType::Q8_0,
            BlockFormat::Q4_KS => GgmlDType::Q4_KS,
            BlockFormat::Q4_1 => GgmlDType::Q4_1,
            BlockFormat::Q4_0 => GgmlDType::Q4_0,
            BlockFormat::Q3_1 => GgmlDType::Q3_1,
            BlockFormat::Q3_0 => GgmlDType::Q3_0,
            BlockFormat::Q2_1 => GgmlDType::Q2_1,
            BlockFormat::Q2_A => GgmlDType::Q2_A,
            BlockFormat::Q2_S => GgmlDType::Q2_S,
            BlockFormat::Q2_0 => GgmlDType::Q2_0,
            BlockFormat::Q1_S => GgmlDType::Q1_S,
            BlockFormat::Q0 => GgmlDType::Q0,
            BlockFormat::Q0_V => GgmlDType::Q0_V,
            BlockFormat::Q1_A => GgmlDType::Q1_A,
            BlockFormat::Q0_X => GgmlDType::Q0_X,
            BlockFormat::Q0_M2 => GgmlDType::Q0_M2,
            BlockFormat::Q0_M4 => GgmlDType::Q0_M4,
        }
    };

    let candidates = level_candidates();

    let path = match dump_path() {
        Some(p) => p,
        None => {
            println!("SKIP: dump file not found at {QWEN3_DUMP_REL_PATH}");
            return;
        }
    };
    let (header, chunks) = match load_dump(&path) {
        Some(v) => v,
        None => {
            println!("SKIP: failed to parse dump");
            return;
        }
    };
    println!(
        "Loaded dump: {} layers, n_kv_head={}, chunk_size={}, head_dim={}, {} chunks total",
        header.num_layers,
        header.n_kv_head,
        header.chunk_size,
        header.head_dim,
        chunks.len()
    );

    let dev = candle::Device::cuda_if_available(0).expect("cuda_if_available");
    let cuda_dev = match &dev {
        candle::Device::Cuda(d) => d.clone(),
        _ => {
            println!("SKIP: no CUDA device available");
            return;
        }
    };

    let sep = "─".repeat(120);
    println!("\n{sep}");
    println!("  CUDA vs CPU per-head reduction comparison  (select_kv_format_paged_per_head)");
    println!("{sep}");

    // ── Upload all chunk K/V data to GPU (F32) ────────────────────────────────
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let stream = cuda_dev.cuda_stream();
    struct ChunkGpu {
        k_gpu: candle::cuda_backend::cudarc::driver::CudaSlice<f32>,
        v_gpu: candle::cuda_backend::cudarc::driver::CudaSlice<f32>,
        num_blocks: usize,
        is_sink: bool,
    }
    let chunk_gpus: Vec<ChunkGpu> = chunks
        .iter()
        .map(|chunk| {
            let k_gpu = cuda_dev.memcpy_stod(&chunk.k).expect("GPU upload k");
            let v_gpu = cuda_dev.memcpy_stod(&chunk.v).expect("GPU upload v");
            let num_blocks = chunk.k.len() / SELECT_BLOCK;
            let is_sink = chunk.token_start == 0;
            ChunkGpu {
                k_gpu,
                v_gpu,
                num_blocks,
                is_sink,
            }
        })
        .collect();

    let blocks_per_chunk = chunk_gpus[0].num_blocks;
    assert!(chunk_gpus
        .iter()
        .all(|cg| cg.num_blocks == blocks_per_chunk));
    let chunk_byte_stride = (blocks_per_chunk * 32 * 4) as i64; // F32: 4 bytes/elem

    let per_head_table_host: Vec<i64> = chunk_gpus
        .iter()
        .map(|cg| {
            let (k_ptr, _) = cg.k_gpu.device_ptr(&stream);
            let (v_ptr, _) = cg.v_gpu.device_ptr(&stream);
            // [k_ptr, v_ptr, k_byte_offset, v_byte_offset,
            //  k_chunk_byte_stride, v_chunk_byte_stride, metadata]
            // metadata=0 → F32 format, GPU location
            [
                k_ptr as i64,
                v_ptr as i64,
                0i64,
                0i64,
                chunk_byte_stride,
                chunk_byte_stride,
                0i64,
            ]
        })
        .flatten()
        .collect();
    let per_head_table_gpu = cuda_dev
        .memcpy_stod(&per_head_table_host)
        .expect("per-head table upload");

    const TEST_ARENA_CHUNKS: i64 = 8192;
    let mut head_gids: Vec<i64> = Vec::with_capacity(chunks.len() * 2);
    for (i, _cg) in chunk_gpus.iter().enumerate() {
        head_gids.push(i as i64 * TEST_ARENA_CHUNKS); // K GID
        head_gids.push(i as i64 * TEST_ARENA_CHUNKS); // V GID (same arena)
    }

    // With n_kv_head=1, blocks_per_head = blocks_per_chunk.
    // The GPU returns one head-tag per chunk (n_chunks * 1 = n_chunks tags).
    let blocks_per_head = blocks_per_chunk;

    let mut grand_total_heads = 0usize;
    let mut grand_mismatch_heads = 0usize;
    let mut grand_bpe_ties = 0usize;
    let mut level_mismatch_heads = vec![0usize; 10];
    let mut level_bpe_ties = vec![0usize; 10];
    let mut level_total_heads = vec![0usize; 10];
    let mut cpu_head_dist = vec![[0usize; 16]; 10];
    let mut gpu_head_dist = vec![[0usize; 16]; 10];
    let mut cpu_head_bpe = vec![0.0f64; 10];
    let mut gpu_head_bpe = vec![0.0f64; 10];
    let mut level_first_mismatches: Vec<Vec<String>> = vec![Vec::new(); 10];

    for level in 0..10 {
        let base_k_threshold = K_MAGWEIGHT_THRESHOLDS[level];
        let base_v_threshold = V_COSINE_THRESHOLDS_PROPOSED[level];
        let (ref k_cands, ref v_cands) = candidates[level];

        let k_ggml: Vec<GgmlDType> = k_cands
            .iter()
            .filter(|f| !matches!(f, BlockFormat::F16 | BlockFormat::BF16))
            .map(|f| bf_to_ggml(*f))
            .collect();
        let v_ggml: Vec<GgmlDType> = v_cands
            .iter()
            .filter(|f| !matches!(f, BlockFormat::F16 | BlockFormat::BF16))
            .map(|f| bf_to_ggml(*f))
            .collect();

        // ── CPU: per-block selection → worst-case per-head reduction ──────────
        // (n_kv_head=1, so one head per chunk, one tag per chunk)
        let mut cpu_k_head_fmts: Vec<BlockFormat> = Vec::with_capacity(chunks.len());
        let mut cpu_v_head_fmts: Vec<BlockFormat> = Vec::with_capacity(chunks.len());

        for chunk in &chunks {
            let eff_k = base_k_threshold;
            let eff_v = base_v_threshold;
            let num_blocks = chunk.k.len() / SELECT_BLOCK;

            // Start with the most aggressive candidate (lowest table_index) and walk
            // up to find the worst-case (highest table_index = least aggressive).
            // This mirrors the GPU reduce_head_format which takes the max candidate index.
            let k_most_aggressive = k_cands
                .iter()
                .copied()
                .filter(|f| !matches!(f, BlockFormat::F16 | BlockFormat::BF16))
                .min_by_key(|f| f.table_index())
                .unwrap_or(BlockFormat::Q0);
            let v_most_aggressive = v_cands
                .iter()
                .copied()
                .filter(|f| !matches!(f, BlockFormat::F16 | BlockFormat::BF16))
                .min_by_key(|f| f.table_index())
                .unwrap_or(BlockFormat::Q0);
            let mut k_worst = k_most_aggressive;
            let mut v_worst = v_most_aggressive;
            for b in 0..num_blocks {
                let k_blk: [f32; SELECT_BLOCK] = chunk.k[b * SELECT_BLOCK..(b + 1) * SELECT_BLOCK]
                    .try_into()
                    .unwrap();
                let v_blk: [f32; SELECT_BLOCK] = chunk.v[b * SELECT_BLOCK..(b + 1) * SELECT_BLOCK]
                    .try_into()
                    .unwrap();
                let kf = select_format_from_candidates_k(&k_blk, k_cands, eff_k);
                let vf = select_format_from_candidates(&v_blk, v_cands, eff_v);
                if kf.table_index() > k_worst.table_index() {
                    k_worst = kf;
                }
                if vf.table_index() > v_worst.table_index() {
                    v_worst = vf;
                }
            }
            cpu_k_head_fmts.push(k_worst);
            cpu_v_head_fmts.push(v_worst);
        }

        // ── CUDA: full per-head pipeline (selection + reduce_head_format) ──────
        let (k_head_tags_gpu, v_head_tags_gpu) = select_kv_format_paged_per_head(
            &per_head_table_gpu,
            &head_gids,
            &k_ggml,
            &v_ggml,
            base_k_threshold,
            base_k_threshold,
            base_v_threshold,
            base_v_threshold,
            None, // no Q2_0 promotion
            blocks_per_head,
            1, // n_kv_head=1
            TEST_ARENA_CHUNKS as usize,
            7,
            &cuda_dev,
        )
        .expect("select_kv_format_paged_per_head");

        let k_head_tags: Vec<i32> = cuda_dev
            .memcpy_dtov(&k_head_tags_gpu)
            .expect("download k head tags");
        let v_head_tags: Vec<i32> = cuda_dev
            .memcpy_dtov(&v_head_tags_gpu)
            .expect("download v head tags");

        // ── Compare per-head decisions ────────────────────────────────────────
        // Mismatches are classified:
        //   BPE-tie  (Type A): different format label, identical bpe → harmless tie-break
        //   BPE-diff (Type B): different bpe → real compression difference
        for (ci, chunk) in chunks.iter().enumerate() {
            let gpu_k = tag_to_block_format(k_head_tags[ci]);
            let gpu_v = tag_to_block_format(v_head_tags[ci]);
            let cpu_k = cpu_k_head_fmts[ci];
            let cpu_v = cpu_v_head_fmts[ci];

            gpu_head_dist[level][gpu_k.table_index()] += 1;
            gpu_head_dist[level][gpu_v.table_index()] += 1;
            gpu_head_bpe[level] += gpu_k.bits_per_elem() as f64;
            gpu_head_bpe[level] += gpu_v.bits_per_elem() as f64;

            cpu_head_dist[level][cpu_k.table_index()] += 1;
            cpu_head_dist[level][cpu_v.table_index()] += 1;
            cpu_head_bpe[level] += cpu_k.bits_per_elem() as f64;
            cpu_head_bpe[level] += cpu_v.bits_per_elem() as f64;

            level_total_heads[level] += 2;

            let is_float = |f: BlockFormat| matches!(f, BlockFormat::F16 | BlockFormat::BF16);
            let k_match = gpu_k == cpu_k || (is_float(gpu_k) && is_float(cpu_k));
            let v_match = gpu_v == cpu_v || (is_float(gpu_v) && is_float(cpu_v));

            if !k_match {
                let is_bpe_tie = (gpu_k.bits_per_elem() - cpu_k.bits_per_elem()).abs() < 0.01;
                if is_bpe_tie {
                    level_bpe_ties[level] += 1;
                } else {
                    level_mismatch_heads[level] += 1;
                }
                if level_first_mismatches[level].len() < 4 {
                    let tag = if is_bpe_tie {
                        "[BPE-TIE]"
                    } else {
                        "[BPE-DIFF]"
                    };
                    level_first_mismatches[level].push(format!(
                        "    L{} tok{} K-head: CPU={}({:.1}bpe) GPU={}({:.1}bpe) {tag}",
                        chunk.layer_idx,
                        chunk.token_start,
                        cpu_k,
                        cpu_k.bits_per_elem(),
                        gpu_k,
                        gpu_k.bits_per_elem()
                    ));
                }
            }
            if !v_match {
                let is_bpe_tie = (gpu_v.bits_per_elem() - cpu_v.bits_per_elem()).abs() < 0.01;
                if is_bpe_tie {
                    level_bpe_ties[level] += 1;
                } else {
                    level_mismatch_heads[level] += 1;
                }
                if level_first_mismatches[level].len() < 4 {
                    let tag = if is_bpe_tie {
                        "[BPE-TIE]"
                    } else {
                        "[BPE-DIFF]"
                    };
                    level_first_mismatches[level].push(format!(
                        "    L{} tok{} V-head: CPU={}({:.1}bpe) GPU={}({:.1}bpe) {tag}",
                        chunk.layer_idx,
                        chunk.token_start,
                        cpu_v,
                        cpu_v.bits_per_elem(),
                        gpu_v,
                        gpu_v.bits_per_elem()
                    ));
                }
            }
        }
        grand_total_heads += level_total_heads[level];
        grand_mismatch_heads += level_mismatch_heads[level];
        grand_bpe_ties += level_bpe_ties[level];
    }

    // ── Table 1: Mismatch summary ─────────────────────────────────────────────
    println!("\nTable 1 — Per-head decision comparison (worst-case reduction)");
    println!("  BPE-Tie  = different format label but identical bpe → harmless tie-break");
    println!("  BPE-Diff = different bpe → real compression discrepancy (should be 0)");
    println!(
        "  {:4}  {:>9}  {:>9}  {:>8}  {:>8}  {:>8}  {:>6}",
        "Mode", "K_Thr", "Decisions", "BPE-Diff", "BPE-Ties", "Mismatch%", "Status"
    );
    println!("  {}", "─".repeat(72));
    for level in 0..10 {
        let total_mm = level_mismatch_heads[level] + level_bpe_ties[level];
        let pct = 100.0 * total_mm as f64 / level_total_heads[level].max(1) as f64;
        let diff_pct =
            100.0 * level_mismatch_heads[level] as f64 / level_total_heads[level].max(1) as f64;
        let status = if level_mismatch_heads[level] == 0 {
            if level_bpe_ties[level] == 0 {
                "PASS"
            } else {
                "TIE-OK"
            }
        } else if diff_pct < 2.0 {
            "FP-OK"
        } else {
            "HIGH"
        };
        println!(
            "  C{level}   {:>9.6}  {:>9}  {:>8}  {:>8}  {:>7.3}%  {status}",
            K_MAGWEIGHT_THRESHOLDS[level],
            level_total_heads[level],
            level_mismatch_heads[level],
            level_bpe_ties[level],
            pct,
        );
        for m in &level_first_mismatches[level] {
            println!("{m}");
        }
    }
    let grand_mm_total = grand_mismatch_heads + grand_bpe_ties;
    let grand_pct = 100.0 * grand_mm_total as f64 / grand_total_heads.max(1) as f64;
    let grand_diff_pct = 100.0 * grand_mismatch_heads as f64 / grand_total_heads.max(1) as f64;
    println!("  {}", "─".repeat(72));
    println!(
        "  TOTAL: {grand_total_heads} decisions  BPE-Diff: {grand_mismatch_heads} ({grand_diff_pct:.3}%)  BPE-Tie: {grand_bpe_ties}  All-mm: {grand_mm_total} ({grand_pct:.3}%)"
    );

    // ── Table 2: CPU vs GPU per-head format distribution + BPE/CR ────────────
    let fmt_labels = [
        "Q0", "Q1S", "Q2S", "Q20", "Q2A", "Q21", "Q30", "Q31", "Q40", "Q41", "Q4K", "Q80", "Q81",
        "Q8K", "BFL", "F16",
    ];
    println!("\n{sep}");
    println!("Table 2 — Per-head format distribution (%) CPU vs GPU  +  BPE/CR");
    println!("  The CR column shows what compression the GPU will actually deliver in production.");
    println!();
    let mut hdr2 = format!("  {:4} {:4}", "Mode", "Side");
    for l in &fmt_labels {
        hdr2 += &format!(" {:>4}", l);
    }
    hdr2 += &format!("    {:>6}  {:>6}", "BPE", "CR");
    println!("{hdr2}");
    println!("  {}", "─".repeat(hdr2.len() - 2));
    for level in 0..10 {
        let n2 = level_total_heads[level].max(1) as f64;
        for (label, dist, bpe_sum) in [
            ("CPU", &cpu_head_dist[level], cpu_head_bpe[level]),
            ("GPU", &gpu_head_dist[level], gpu_head_bpe[level]),
        ] {
            let bpe = bpe_sum / n2;
            let cr = 16.0 / bpe;
            let mut line = format!("  C{level}  {label:<4}");
            for j in 0..16 {
                line += &format!(" {:>4.1}", dist[j] as f64 / n2 * 100.0);
            }
            line += &format!("    {:>6.3}  {:>5.2}×", bpe, cr);
            println!("{line}");
        }
    }
    println!("{sep}");
    println!();
    println!("  Legend: CR = 16 / BPE (combined K+V average).  CPU = expected, GPU = actual.");
    println!("  If GPU CR < CPU CR, the reduce_head_format kernel is being overly conservative.");
    println!("{sep}");

    // Hard assertion: only BPE-diff (real compression discrepancies) must be near zero.
    // BPE-tie mismatches (same compression, different format label) are harmless.
    // FP noise from per-block selection can still propagate to per-head if a
    // noisy boundary block flips to a format with a different BPE.
    assert!(
        grand_diff_pct < 2.0,
        "Per-head BPE-diff mismatch rate {grand_diff_pct:.3}% exceeds 2% tolerance — \
         this indicates a real compression discrepancy between CPU and GPU"
    );
}

// ============================================================================
// CUDA vs CPU comparison — R16 K arenas with Q-projected K selection
// ============================================================================
//
// This test exercises the Q-attention-weighted K selection path in the CUDA
// kernel by feeding R16 K arenas (K+Q packed as F16) and F16 V arenas.
//
// The CPU reference computes the same per-element Q²-weighted error:
//   q_attn_weighted_loss = Σ(q² · (k - k̂)²) / Σ(q² · k²)
// and the same block relevance:
//   block_relevance = Σ(q² · k²) / Σ(k²)
// with two-sided gating: rel < lo → Q2_0, rel > hi → most conservative.
//
// K and Q values are F16-quantized before comparison to match the GPU's
// R16 block format (which stores both as half).

/// CPU per-element Q²-weighted K error, matching CUDA `q_attn_weighted_loss`.
/// Inputs should already be F16-truncated to match GPU precision.
#[allow(dead_code)]
fn cpu_q_attn_weighted_loss(
    k_block: &[f32; SELECT_BLOCK],
    k_recon: &[f32; SELECT_BLOCK],
    q_block: &[f32; SELECT_BLOCK],
) -> f32 {
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for i in 0..SELECT_BLOCK {
        let q2 = (q_block[i] as f64) * (q_block[i] as f64);
        let e = (k_block[i] - k_recon[i]) as f64;
        num += q2 * e * e;
        den += q2 * (k_block[i] as f64) * (k_block[i] as f64);
    }
    if den == 0.0 {
        return 0.0;
    }
    (num / den) as f32
}

/// CPU block relevance, matching CUDA `block_relevance`.
#[allow(dead_code)]
fn cpu_block_relevance(k_block: &[f32; SELECT_BLOCK], q_block: &[f32; SELECT_BLOCK]) -> f32 {
    let mut qk2 = 0.0f64;
    let mut k2 = 0.0f64;
    for i in 0..SELECT_BLOCK {
        let qi = q_block[i] as f64;
        let ki = k_block[i] as f64;
        qk2 += qi * qi * ki * ki;
        k2 += ki * ki;
    }
    if k2 == 0.0 {
        return 0.0;
    }
    (qk2 / k2) as f32
}

/// CPU Q-projected K format selection with two-sided relevance gating.
/// Returns chosen format for one 32-element K block.
#[allow(dead_code)]
fn cpu_select_k_qproj(
    k_block: &[f32; SELECT_BLOCK],
    q_block: &[f32; SELECT_BLOCK],
    quant_candidates: &[BlockFormat],
    threshold_hi: f32,
    threshold_lo: f32,
    relevance_lo: f32,
    relevance_hi: f32,
    relevance_gamma: f32,
) -> BlockFormat {
    let fallback = quant_candidates
        .iter()
        .copied()
        .find(|f| !matches!(f, BlockFormat::F16 | BlockFormat::BF16))
        .unwrap_or(BlockFormat::F16);

    let amax = k_block.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let vmin = k_block.iter().cloned().fold(f32::INFINITY, f32::min);
    if amax == 0.0 && vmin >= 0.0 {
        return fallback;
    }

    let relevance = cpu_block_relevance(k_block, q_block);
    let threshold = qrel_threshold(
        threshold_lo,
        threshold_hi,
        relevance,
        relevance_lo,
        relevance_hi,
        relevance_gamma,
    );

    let mut best: Option<(BlockFormat, f32)> = None;
    for &fmt in quant_candidates {
        if matches!(fmt, BlockFormat::F16 | BlockFormat::BF16) {
            continue;
        }
        let recon = fmt.apply_quant(k_block);
        let recon_adj = apply_error_margin_block(k_block, &recon, ERROR_MARGIN_ABS);
        let err = cpu_q_attn_weighted_loss(k_block, &recon_adj, q_block);
        if err <= threshold {
            let better = match best {
                Some((best_fmt, best_err)) => {
                    let fmt_bpe = fmt.bits_per_elem();
                    let best_bpe = best_fmt.bits_per_elem();
                    fmt_bpe < best_bpe
                        || (fmt_bpe == best_bpe
                            && (err < best_err
                                || (err == best_err && fmt.table_index() < best_fmt.table_index())))
                }
                None => true,
            };
            if better {
                best = Some((fmt, err));
            }
        }
    }

    best.map(|(fmt, _)| fmt).unwrap_or(fallback)
}

/// Truncate f32 to F16 and back, matching the precision loss in R16 blocks.
#[allow(dead_code)]
fn f32_to_f16_to_f32(x: f32) -> f32 {
    f16::from_f32(x).to_f32()
}

/// Pack K and Q float data into R16 block format (byte buffer).
///
/// R16 block layout (128 bytes per 32-element block):
///   d[0..32]: K values as F16 (64 bytes)
///   q[0..32]: Q values as F16 (64 bytes)
///
/// Input data is [n_blocks * 32] f32 values.  Output is [n_blocks * 128] bytes.
#[allow(dead_code)]
fn pack_r16_blocks(k_data: &[f32], q_data: &[f32]) -> Vec<u8> {
    assert_eq!(k_data.len(), q_data.len());
    assert!(k_data.len() % SELECT_BLOCK == 0);
    let n_blocks = k_data.len() / SELECT_BLOCK;
    let mut buf = vec![0u8; n_blocks * 128];
    for b in 0..n_blocks {
        let block_start = b * 128;
        for i in 0..SELECT_BLOCK {
            let k_f16 = f16::from_f32(k_data[b * SELECT_BLOCK + i]);
            let q_f16 = f16::from_f32(q_data[b * SELECT_BLOCK + i]);
            // d[i] at offset 0 + i*2
            buf[block_start + i * 2..block_start + i * 2 + 2].copy_from_slice(&k_f16.to_le_bytes());
            // q[i] at offset 64 + i*2
            buf[block_start + 64 + i * 2..block_start + 64 + i * 2 + 2]
                .copy_from_slice(&q_f16.to_le_bytes());
        }
    }
    buf
}

/// Pack V float data as F16 values.
#[allow(dead_code)]
fn pack_f16(data: &[f32]) -> Vec<u8> {
    let mut buf = vec![0u8; data.len() * 2];
    for (i, &v) in data.iter().enumerate() {
        let h = f16::from_f32(v);
        buf[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
    }
    buf
}

/// Test: CUDA R16 Q-projected K selection matches CPU reference.
///
/// Uses v4 dump data (which includes Q values) to:
/// 1. Pack K+Q into R16 block format, V into F16
/// 2. Run CUDA kernel with k_is_r16=true (Q-attention-weighted K selection)
/// 3. Run CPU reference with same per-element Q²-weighted error metric
/// 4. Compare per-block format decisions
///
/// Run:
///     cargo test --release --lib --package candle-nn --features cuda \
///         kv_selection_tests::test_cuda_r16_qproj_matches_cpu -- --ignored --nocapture
#[test]
#[ignore]
#[cfg(feature = "cuda")]
fn test_cuda_r16_qproj_matches_cpu() {
    use candle::quantized::{cuda::select_kv_format_paged_batched_raw, GgmlDType};

    let bf_to_ggml = |bf: BlockFormat| -> GgmlDType {
        match bf {
            BlockFormat::F16 => GgmlDType::F16,
            BlockFormat::BF16 => GgmlDType::BF16,
            BlockFormat::Q8_KS => GgmlDType::Q8_KS,
            BlockFormat::Q8_1 => GgmlDType::Q8_1,
            BlockFormat::Q8_0 => GgmlDType::Q8_0,
            BlockFormat::Q4_KS => GgmlDType::Q4_KS,
            BlockFormat::Q4_1 => GgmlDType::Q4_1,
            BlockFormat::Q4_0 => GgmlDType::Q4_0,
            BlockFormat::Q3_1 => GgmlDType::Q3_1,
            BlockFormat::Q3_0 => GgmlDType::Q3_0,
            BlockFormat::Q2_1 => GgmlDType::Q2_1,
            BlockFormat::Q2_A => GgmlDType::Q2_A,
            BlockFormat::Q2_S => GgmlDType::Q2_S,
            BlockFormat::Q2_0 => GgmlDType::Q2_0,
            BlockFormat::Q1_S => GgmlDType::Q1_S,
            BlockFormat::Q0 => GgmlDType::Q0,
            BlockFormat::Q0_V => GgmlDType::Q0_V,
            BlockFormat::Q1_A => GgmlDType::Q1_A,
            BlockFormat::Q0_X => GgmlDType::Q0_X,
            BlockFormat::Q0_M2 => GgmlDType::Q0_M2,
            BlockFormat::Q0_M4 => GgmlDType::Q0_M4,
        }
    };

    let candidates = level_candidates();

    // ── Load dump (must be v4 with Q data) ─────────────────────────────────────
    // Try R16 dump first (v4 format with Q data), then fall back to main dump.
    let path = match r16_dump_path() {
        Some(p) => p,
        None => match dump_path() {
            Some(p) => p,
            None => {
                println!("SKIP: no dump files found");
                return;
            }
        },
    };
    let (header, chunks) = match load_dump(&path) {
        Some(v) => v,
        None => {
            println!("SKIP: failed to parse dump");
            return;
        }
    };

    // Check that Q data is available (v4 dump)
    let have_q = chunks.iter().all(|c| c.q.is_some());
    if !have_q {
        println!("SKIP: dump does not contain Q data (need v4 format)");
        return;
    }
    println!(
        "Loaded v4 dump: {} layers, n_kv_head={}, chunk_size={}, head_dim={}, {} chunks",
        header.num_layers,
        header.n_kv_head,
        header.chunk_size,
        header.head_dim,
        chunks.len()
    );

    // ── CUDA device ────────────────────────────────────────────────────────────
    let dev = candle::Device::cuda_if_available(0).expect("cuda_if_available");
    let cuda_dev = match &dev {
        candle::Device::Cuda(d) => d.clone(),
        _ => {
            println!("SKIP: no CUDA device available");
            return;
        }
    };

    let sep = "─".repeat(120);
    println!("\n{sep}");
    println!("  CUDA R16 Q-projected K selection vs CPU reference");
    println!("{sep}");

    // Relevance thresholds for two-sided gating
    let rel_lo = 0.20f32;
    let rel_hi = 0.95f32;

    // ── Upload R16 K arenas and F16 V arenas ───────────────────────────────────
    use candle::cuda_backend::cudarc::driver::DevicePtr;
    let stream = cuda_dev.cuda_stream();

    struct ChunkGpu {
        k_r16_gpu: candle::cuda_backend::cudarc::driver::CudaSlice<u8>,
        v_f16_gpu: candle::cuda_backend::cudarc::driver::CudaSlice<u8>,
        num_blocks: usize,
        is_sink: bool,
    }
    let chunk_gpus: Vec<ChunkGpu> = chunks
        .iter()
        .map(|chunk| {
            let q = chunk.q.as_ref().unwrap();
            let r16_bytes = pack_r16_blocks(&chunk.k, q);
            let v_f16_bytes = pack_f16(&chunk.v);
            let k_r16_gpu = cuda_dev.memcpy_stod(&r16_bytes).expect("GPU upload K R16");
            let v_f16_gpu = cuda_dev
                .memcpy_stod(&v_f16_bytes)
                .expect("GPU upload V F16");
            let num_blocks = chunk.k.len() / SELECT_BLOCK;
            let is_sink = chunk.token_start == 0;
            ChunkGpu {
                k_r16_gpu,
                v_f16_gpu,
                num_blocks,
                is_sink,
            }
        })
        .collect();

    // Per-head table: K ptr = R16 byte buffer, V ptr = F16 byte buffer
    // R16: 128 bytes per block (block_r16), chunk_byte_stride = blocks_per_chunk * 128
    // F16 V: chunk_byte_stride = blocks_per_chunk * 32 * 2 (F16: 2 bytes per elem)
    let blocks_per_chunk = chunk_gpus[0].num_blocks;
    assert!(
        chunk_gpus
            .iter()
            .all(|cg| cg.num_blocks == blocks_per_chunk),
        "All chunks must have the same number of blocks"
    );
    let k_chunk_byte_stride = (blocks_per_chunk * 128) as i64; // R16: 128 bytes per block
    let v_chunk_byte_stride = (blocks_per_chunk * 32 * 2) as i64; // F16: 2 bytes per elem

    let per_head_table_host: Vec<i64> = chunk_gpus
        .iter()
        .map(|cg| {
            let (k_ptr, _) = cg.k_r16_gpu.device_ptr(&stream);
            let (v_ptr, _) = cg.v_f16_gpu.device_ptr(&stream);
            // metadata: (k_format_tag << 16) | (v_format_tag << 8) | location
            // ArenaFormat::R16 = 39, ArenaFormat::F16 = 1
            let metadata = (39i64 << 16) | (1i64 << 8) | 0i64;
            [
                k_ptr as i64,
                v_ptr as i64,
                0i64,
                0i64,
                k_chunk_byte_stride,
                v_chunk_byte_stride,
                metadata,
            ]
        })
        .flatten()
        .collect();
    let per_head_table_gpu = cuda_dev
        .memcpy_stod(&per_head_table_host)
        .expect("per-head table upload");

    // Chunk descriptors with head_gids
    const TEST_ARENA_CHUNKS_R16: i64 = 8192;
    let mut head_gids: Vec<i64> = Vec::with_capacity(chunk_gpus.len() * 2);
    for (i, _cg) in chunk_gpus.iter().enumerate() {
        head_gids.push(i as i64 * TEST_ARENA_CHUNKS_R16); // K GID
        head_gids.push(i as i64 * TEST_ARENA_CHUNKS_R16); // V GID (same arena)
    }

    // Tracking
    let mut grand_total = 0usize;
    let mut grand_mismatches = 0usize;
    let mut level_total = vec![0usize; 10];
    let mut level_mismatches = vec![0usize; 10];
    let mut level_bpe_ties = vec![0usize; 10];
    let mut level_max_margin = vec![0.0f64; 10];
    let mut level_margin_sum = vec![0.0f64; 10];
    let mut level_margin_count = vec![0usize; 10];
    let mut level_first_mismatches: Vec<Vec<String>> = vec![Vec::new(); 10];
    let mut level_relevance_skips = vec![[0usize; 2]; 10]; // [lo, hi] per level

    for level in 0..10 {
        let base_k_threshold = K_MAGWEIGHT_THRESHOLDS[level];
        let base_v_threshold = V_COSINE_THRESHOLDS_PROPOSED[level];
        let (ref k_cands, ref v_cands) = candidates[level];

        let k_ggml: Vec<GgmlDType> = k_cands
            .iter()
            .filter(|f| !matches!(f, BlockFormat::F16 | BlockFormat::BF16))
            .map(|f| bf_to_ggml(*f))
            .collect();
        let v_ggml: Vec<GgmlDType> = v_cands
            .iter()
            .filter(|f| !matches!(f, BlockFormat::F16 | BlockFormat::BF16))
            .map(|f| bf_to_ggml(*f))
            .collect();

        // Sort K candidates by BPE (low→high) to match kernel scan order
        let mut k_quant_sorted: Vec<BlockFormat> = k_cands
            .iter()
            .copied()
            .filter(|f| !matches!(f, BlockFormat::F16 | BlockFormat::BF16))
            .collect();
        k_quant_sorted.sort_by(|a, b| {
            a.bits_per_elem()
                .partial_cmp(&b.bits_per_elem())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // ── CPU reference (Q-projected, F16-truncated K and Q) ──
        let mut all_cpu_k_fmts: Vec<Vec<BlockFormat>> = Vec::with_capacity(chunks.len());
        let mut all_cpu_v_fmts: Vec<Vec<BlockFormat>> = Vec::with_capacity(chunks.len());

        for chunk in &chunks {
            let q_data = chunk.q.as_ref().unwrap();
            let num_blocks = chunk.k.len() / SELECT_BLOCK;
            let eff_k_threshold = base_k_threshold;
            let eff_v_threshold = base_v_threshold;

            let mut cpu_k_fmts: Vec<BlockFormat> = Vec::with_capacity(num_blocks);
            let mut cpu_v_fmts: Vec<BlockFormat> = Vec::with_capacity(num_blocks);
            for b in 0..num_blocks {
                let start = b * SELECT_BLOCK;
                // F16-truncate K and Q to match R16 block precision
                let k_block: [f32; SELECT_BLOCK] =
                    std::array::from_fn(|i| f32_to_f16_to_f32(chunk.k[start + i]));
                let q_block: [f32; SELECT_BLOCK] =
                    std::array::from_fn(|i| f32_to_f16_to_f32(q_data[start + i]));
                // V is also F16-truncated
                let v_block: [f32; SELECT_BLOCK] =
                    std::array::from_fn(|i| f32_to_f16_to_f32(chunk.v[start + i]));

                let kf = cpu_select_k_qproj(
                    &k_block,
                    &q_block,
                    &k_quant_sorted,
                    eff_k_threshold,
                    eff_k_threshold,
                    rel_lo,
                    rel_hi,
                    K_QREL_GAMMAS[level.min(10)],
                );
                let vf = select_format_from_candidates(&v_block, v_cands, eff_v_threshold);

                // Track relevance skips
                let rel = cpu_block_relevance(&k_block, &q_block);
                if rel < rel_lo {
                    level_relevance_skips[level][0] += 1;
                } else if rel > rel_hi {
                    level_relevance_skips[level][1] += 1;
                }

                cpu_k_fmts.push(kf);
                cpu_v_fmts.push(vf);
            }
            all_cpu_k_fmts.push(cpu_k_fmts);
            all_cpu_v_fmts.push(cpu_v_fmts);
        }

        // ── CUDA kernel (R16 K, F16 V) ──
        let (k_tags_gpu, v_tags_gpu) = select_kv_format_paged_batched_raw(
            &per_head_table_gpu,
            &head_gids,
            &k_ggml,
            &v_ggml,
            base_k_threshold,
            base_k_threshold,
            base_v_threshold,
            base_v_threshold,
            None, // no V Q2_0 promotion in this test
            rel_lo,
            rel_hi,
            blocks_per_chunk,
            1, // n_kv_head=1 in this test
            TEST_ARENA_CHUNKS_R16 as usize,
            7,
            &cuda_dev,
        )
        .expect("select_kv_format_palette4_paged CUDA R16");
        let all_k_tags: Vec<i32> = cuda_dev.memcpy_dtov(&k_tags_gpu).expect("GPU download k");
        let all_v_tags: Vec<i32> = cuda_dev.memcpy_dtov(&v_tags_gpu).expect("GPU download v");

        // ── Compare ──
        for (ci, chunk) in chunks.iter().enumerate() {
            let q_data = chunk.q.as_ref().unwrap();
            let num_blocks = chunk.k.len() / SELECT_BLOCK;
            let eff_k_threshold = base_k_threshold;
            let cpu_k_fmts = &all_cpu_k_fmts[ci];
            let cpu_v_fmts = &all_cpu_v_fmts[ci];
            let tag_offset = ci * blocks_per_chunk;

            for b in 0..num_blocks {
                let cuda_k = tag_to_block_format(all_k_tags[tag_offset + b]);
                let cuda_v = tag_to_block_format(all_v_tags[tag_offset + b]);
                level_total[level] += 2;

                let is_float = |f: BlockFormat| matches!(f, BlockFormat::F16 | BlockFormat::BF16);
                let k_match =
                    cuda_k == cpu_k_fmts[b] || (is_float(cuda_k) && is_float(cpu_k_fmts[b]));
                let v_match =
                    cuda_v == cpu_v_fmts[b] || (is_float(cuda_v) && is_float(cpu_v_fmts[b]));

                if !k_match {
                    level_mismatches[level] += 1;
                    let cpu_bpe = cpu_k_fmts[b].bits_per_elem();
                    let cuda_bpe = cuda_k.bits_per_elem();
                    if (cpu_bpe - cuda_bpe).abs() < 0.01 {
                        level_bpe_ties[level] += 1;
                    } else {
                        // Compute margin: Q-weighted error distance from threshold
                        let start = b * SELECT_BLOCK;
                        let k_block: [f32; SELECT_BLOCK] =
                            std::array::from_fn(|i| f32_to_f16_to_f32(chunk.k[start + i]));
                        let q_block: [f32; SELECT_BLOCK] =
                            std::array::from_fn(|i| f32_to_f16_to_f32(q_data[start + i]));
                        let err_of = |fmt: BlockFormat| -> f32 {
                            if matches!(fmt, BlockFormat::F16 | BlockFormat::BF16) {
                                return 0.0;
                            }
                            let recon = fmt.apply_quant(&k_block);
                            cpu_q_attn_weighted_loss(&k_block, &recon, &q_block)
                        };
                        let cpu_err = err_of(cpu_k_fmts[b]);
                        let cuda_err = err_of(cuda_k);
                        let thr = eff_k_threshold as f64;
                        let margin = ((cpu_err as f64 - thr) / thr)
                            .abs()
                            .min(((cuda_err as f64 - thr) / thr).abs());
                        level_margin_sum[level] += margin;
                        level_margin_count[level] += 1;
                        if margin > level_max_margin[level] {
                            level_max_margin[level] = margin;
                        }
                    }
                    if level_first_mismatches[level].len() < 5 {
                        level_first_mismatches[level].push(format!(
                            "    L{} tok{} blk{} K: CPU={}({:.1}bpe) CUDA={}({:.1}bpe)",
                            chunk.layer_idx,
                            chunk.token_start,
                            b,
                            cpu_k_fmts[b],
                            cpu_k_fmts[b].bits_per_elem(),
                            cuda_k,
                            cuda_k.bits_per_elem()
                        ));
                    }
                }

                if !v_match {
                    level_mismatches[level] += 1;
                    let cpu_bpe = cpu_v_fmts[b].bits_per_elem();
                    let cuda_bpe = cuda_v.bits_per_elem();
                    if (cpu_bpe - cuda_bpe).abs() < 0.01 {
                        level_bpe_ties[level] += 1;
                    } else {
                        level_margin_count[level] += 1;
                    }
                    if level_first_mismatches[level].len() < 5 {
                        level_first_mismatches[level].push(format!(
                            "    L{} tok{} blk{} V: CPU={}({:.1}bpe) CUDA={}({:.1}bpe)",
                            chunk.layer_idx,
                            chunk.token_start,
                            b,
                            cpu_v_fmts[b],
                            cpu_bpe,
                            cuda_v,
                            cuda_bpe
                        ));
                    }
                }
            }
        }

        grand_total += level_total[level];
        grand_mismatches += level_mismatches[level];
    }

    // ── Results table ──
    println!(
        "\n  {:4}  {:>9}  {:>9}  {:>10}  {:>8}  {:>8}  {:>7}",
        "Mode", "Decisions", "Mismatch", "Mismatch%", "BPE-Tie", "Rel-Lo", "Rel-Hi"
    );
    println!("  {}", "─".repeat(76));
    for level in 0..10 {
        let pct = 100.0 * level_mismatches[level] as f64 / level_total[level].max(1) as f64;
        let status = if pct == 0.0 {
            "PASS"
        } else if pct < 0.5 {
            "FP-OK"
        } else if pct < 1.0 {
            "WARN"
        } else {
            "HIGH"
        };
        println!(
            "  C{level}   {:>8}  {:>8}  {:>9.4}%  {:>7}  {:>7}  {:>7}  {status}",
            level_total[level],
            level_mismatches[level],
            pct,
            level_bpe_ties[level],
            level_relevance_skips[level][0],
            level_relevance_skips[level][1],
        );
        for m in &level_first_mismatches[level] {
            println!("{m}");
        }
    }
    let grand_pct = 100.0 * grand_mismatches as f64 / grand_total.max(1) as f64;
    println!("  {}", "─".repeat(76));
    println!("  TOTAL: {grand_total} decisions, {grand_mismatches} mismatches ({grand_pct:.4}%)");

    // Margin analysis
    let global_max_margin: f64 = level_max_margin.iter().cloned().fold(0.0, f64::max);
    let total_diffs: usize = level_margin_count.iter().sum();
    let total_ties: usize = level_bpe_ties.iter().sum();
    println!(
        "  BPE-Tie: {total_ties}  BPE-Diff: {total_diffs}  Max margin: {global_max_margin:.6}"
    );
    println!("{sep}");

    // Hard assertions (same as the float test)
    assert!(
        grand_pct < 0.5,
        "R16 Q-projected mismatch rate {grand_pct:.4}% exceeds 0.5% tolerance"
    );
    assert!(
        global_max_margin < 0.10,
        "R16 Q-projected max margin {global_max_margin:.6} exceeds 10% envelope"
    );
}

// ============================================================================
// Asymmetric K/V attention-output error model
// ============================================================================

/// Simulate scaled dot-product attention for one head of one chunk and measure
/// output error when K and/or V are quantized.
///
/// Given float K [chunk_size, head_dim] and V [chunk_size, head_dim], a query
/// vector q [head_dim], and format selections for K and V:
///
///   scores = q · K^T / √head_dim        → [chunk_size]
///   attn   = softmax(scores)             → [chunk_size]
///   output = attn · V                    → [head_dim]
///
/// Returns (ref_output, quant_output) where ref is computed with float K/V
/// and quant is computed with the specified K/V formats applied per-block.
fn attention_output(
    k_flat: &[f32], // [chunk_size * head_dim]
    v_flat: &[f32], // [chunk_size * head_dim]
    query: &[f32],  // [head_dim]
    head_dim: usize,
    chunk_size: usize,
    k_fmt: BlockFormat,
    v_fmt: BlockFormat,
) -> (Vec<f32>, Vec<f32>) {
    let sqrt_d = (head_dim as f32).sqrt();

    // Quantize K and V per 32-element block
    let quantize_flat = |data: &[f32], fmt: BlockFormat| -> Vec<f32> {
        let mut out = vec![0.0f32; data.len()];
        for b in (0..data.len()).step_by(SELECT_BLOCK) {
            let end = (b + SELECT_BLOCK).min(data.len());
            if end - b == SELECT_BLOCK {
                let blk: [f32; SELECT_BLOCK] = data[b..end].try_into().unwrap();
                let recon = fmt.apply_quant(&blk);
                out[b..end].copy_from_slice(&recon);
            } else {
                out[b..end].copy_from_slice(&data[b..end]);
            }
        }
        out
    };

    let compute = |k_data: &[f32], v_data: &[f32]| -> Vec<f32> {
        // scores[t] = q · k[t] / √d
        let mut scores = vec![0.0f32; chunk_size];
        for t in 0..chunk_size {
            let mut dot = 0.0f32;
            for d in 0..head_dim {
                dot += query[d] * k_data[t * head_dim + d];
            }
            scores[t] = dot / sqrt_d;
        }
        // softmax
        let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut exp_sum = 0.0f32;
        for s in &mut scores {
            *s = (*s - max_s).exp();
            exp_sum += *s;
        }
        for s in &mut scores {
            *s /= exp_sum;
        }
        // output = attn · V
        let mut output = vec![0.0f32; head_dim];
        for t in 0..chunk_size {
            for d in 0..head_dim {
                output[d] += scores[t] * v_data[t * head_dim + d];
            }
        }
        output
    };

    let ref_output = compute(k_flat, v_flat);

    let k_q = quantize_flat(k_flat, k_fmt);
    let v_q = quantize_flat(v_flat, v_fmt);
    let quant_output = compute(&k_q, &v_q);

    (ref_output, quant_output)
}

/// Cosine distance between two f32 slices.
fn vec_cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-30 {
        0.0
    } else {
        (1.0 - dot / denom).max(0.0)
    }
}

/// NRMSE between two f32 slices.
fn vec_nrmse(reference: &[f32], test: &[f32]) -> f64 {
    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    for (&r, &t) in reference.iter().zip(test.iter()) {
        sig += r as f64 * r as f64;
        noise += (r - t) as f64 * (r - t) as f64;
    }
    if sig > 1e-30 {
        (noise / sig).sqrt()
    } else {
        0.0
    }
}

/// Model the asymmetric K/V precision amplification effect through attention.
///
/// For each compression level, simulates attention with different K/V format
/// combinations and measures the downstream output error:
///
///   - K=F16/V=F16      (reference)
///   - K=Q8_0/V=Q8_0    (symmetric high)
///   - K=Q4_0/V=Q4_0    (symmetric low — C4 style)
///   - K=Q8_0/V=Q4_0    (asymmetric — C3 style)
///
/// The key finding: K=Q8_0/V=Q4_0 produces WORSE output than K=Q4_0/V=Q4_0
/// because sharp K attention concentrates on fewer V tokens, amplifying their
/// Q4_0 errors instead of averaging them out.
///
/// Run with:
///   cargo test --release --lib --package candle-nn \
///     kv_selection_tests::test_asymmetric_kv_attention_error -- --ignored --nocapture
#[test]
#[ignore]
fn test_asymmetric_kv_attention_error() {
    let path = match dump_path() {
        Some(p) => p,
        None => {
            println!("kv_selection_tests: dump absent — run test_dump_kv_cache_data first.");
            return;
        }
    };

    let (header, chunks) = load_dump(&path).expect("failed to load dump");
    let chunk_size = header.chunk_size;
    let head_dim = header.head_dim;
    let n_kv_head = header.n_kv_head;

    // K/V format combinations to test
    let combos: &[(BlockFormat, BlockFormat, &str)] = &[
        (BlockFormat::Q8_0, BlockFormat::Q8_0, "K=Q8_0 V=Q8_0"),
        (BlockFormat::Q4_0, BlockFormat::Q4_0, "K=Q4_0 V=Q4_0"),
        (BlockFormat::Q8_0, BlockFormat::Q4_0, "K=Q8_0 V=Q4_0"),
        (BlockFormat::Q4_0, BlockFormat::Q8_0, "K=Q4_0 V=Q8_0"),
    ];

    let sep = "=".repeat(100);
    let thin = "-".repeat(100);
    println!("\n{sep}");
    println!("Asymmetric K/V Attention-Output Error Model");
    println!("  Simulates scaled dot-product attention with quantized K/V per head per chunk.");
    println!("  Query = last K token (realistic: model attending to recent context).");
    println!(
        "  {} chunks × {} heads = {} attention simulations per combo.",
        chunks.len(),
        n_kv_head,
        chunks.len() * n_kv_head
    );
    println!("{sep}");

    // Accumulate per-combo stats
    struct ComboStats {
        cos_dists: Vec<f64>,
        nrmses: Vec<f64>,
    }
    impl ComboStats {
        fn new() -> Self {
            Self {
                cos_dists: Vec::new(),
                nrmses: Vec::new(),
            }
        }
        fn mean_cos(&self) -> f64 {
            if self.cos_dists.is_empty() {
                0.0
            } else {
                self.cos_dists.iter().sum::<f64>() / self.cos_dists.len() as f64
            }
        }
        fn p95_cos(&self) -> f64 {
            if self.cos_dists.is_empty() {
                return 0.0;
            }
            let mut s = self.cos_dists.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[((s.len() as f64 * 0.95) as usize).min(s.len() - 1)]
        }
        fn max_cos(&self) -> f64 {
            self.cos_dists.iter().copied().fold(0.0f64, f64::max)
        }
        fn mean_nrmse(&self) -> f64 {
            if self.nrmses.is_empty() {
                0.0
            } else {
                self.nrmses.iter().sum::<f64>() / self.nrmses.len() as f64
            }
        }
        fn p95_nrmse(&self) -> f64 {
            if self.nrmses.is_empty() {
                return 0.0;
            }
            let mut s = self.nrmses.clone();
            s.sort_by(|a, b| a.partial_cmp(b).unwrap());
            s[((s.len() as f64 * 0.95) as usize).min(s.len() - 1)]
        }
    }

    let mut all_stats: Vec<ComboStats> = combos.iter().map(|_| ComboStats::new()).collect();

    for chunk in &chunks {
        for h in 0..n_kv_head {
            let head_offset = h * chunk_size * head_dim;
            let k_head = &chunk.k[head_offset..head_offset + chunk_size * head_dim];
            let v_head = &chunk.v[head_offset..head_offset + chunk_size * head_dim];

            // Use last token's K as query (realistic: attending to recent context)
            let q_start = (chunk_size - 1) * head_dim;
            let query = &k_head[q_start..q_start + head_dim];

            // Skip all-zero heads (padding)
            let k_energy: f64 = k_head.iter().map(|&x| x as f64 * x as f64).sum();
            if k_energy < 1e-10 {
                continue;
            }

            for (ci, &(k_fmt, v_fmt, _)) in combos.iter().enumerate() {
                let (ref_out, quant_out) =
                    attention_output(k_head, v_head, query, head_dim, chunk_size, k_fmt, v_fmt);
                let cd = vec_cosine_distance(&ref_out, &quant_out);
                let nr = vec_nrmse(&ref_out, &quant_out);
                all_stats[ci].cos_dists.push(cd);
                all_stats[ci].nrmses.push(nr);
            }
        }
    }

    // ── Table 1: Overall attention-output error by K/V combo ──
    println!("\nTable 1 — Attention-output error by K/V format combination (all chunks)");
    println!("  Measures error in the attention output vector, NOT per-block quantization error.");
    println!("  This captures the amplification effect of sharp K scores on V errors.");
    println!();
    println!(
        "  {:<20} {:>10} {:>10} {:>10} {:>12} {:>12} {:>8}",
        "Combo", "cos_μ", "cos_p95", "cos_max", "nrmse_μ", "nrmse_p95", "N"
    );
    println!("  {thin}");

    for (ci, &(_, _, label)) in combos.iter().enumerate() {
        let s = &all_stats[ci];
        println!(
            "  {:<20} {:>10.7} {:>10.7} {:>10.7} {:>12.7} {:>12.7} {:>8}",
            label,
            s.mean_cos(),
            s.p95_cos(),
            s.max_cos(),
            s.mean_nrmse(),
            s.p95_nrmse(),
            s.cos_dists.len()
        );
    }

    // ── Amplification ratio ──
    println!();
    println!("  Amplification ratio (asymmetric / symmetric-low):");
    let sym_low = &all_stats[1]; // K=Q4_0/V=Q4_0
    let asym = &all_stats[2]; // K=Q8_0/V=Q4_0
    if sym_low.mean_cos() > 1e-15 {
        println!(
            "    cos_μ  amplification: {:.2}x  (K=Q8_0/V=Q4_0 vs K=Q4_0/V=Q4_0)",
            asym.mean_cos() / sym_low.mean_cos()
        );
        println!(
            "    cos_p95 amplification: {:.2}x",
            asym.p95_cos() / sym_low.p95_cos().max(1e-15)
        );
        println!(
            "    nrmse_μ amplification: {:.2}x",
            asym.mean_nrmse() / sym_low.mean_nrmse().max(1e-15)
        );
    }

    // ── Table 2: What V threshold compensates for K=Q8_0? ──
    // Scan V thresholds to find where K=Q8_0/V=adaptive matches C4 quality.
    println!();
    println!("{thin}");
    println!("Table 2 — V threshold scan for K=Q8_0 (finding C3 fix)");
    println!("  Target: match K=Q4_0/V=Q4_0 output quality with K=Q8_0/V=adaptive.");
    println!(
        "  For each V threshold, select V format per-block adaptively while K=Q8_0 is locked."
    );
    println!();

    let c4_p95 = sym_low.p95_cos();
    println!("  C4 reference (K=Q4_0/V=Q4_0): cos_p95 = {:.7}", c4_p95);
    println!();

    // Test V thresholds from very tight to the current C3/C4 value
    let v_thresholds: &[f32] = &[
        0.0005, 0.0010, 0.0015, 0.0020, 0.0025, 0.0030, 0.0035, 0.0040, 0.0050,
    ];

    println!(
        "  {:<10} {:>10} {:>10} {:>10} {:>12} {:>6} {:>8}",
        "V_thr", "cos_μ", "cos_p95", "cos_max", "nrmse_μ", "V_BPE", "Status"
    );
    println!("  {}", "-".repeat(76));

    for &v_thr in v_thresholds {
        let mut stats = ComboStats::new();
        let mut v_bpe_sum = 0.0f64;
        let mut v_blocks = 0usize;

        for chunk in &chunks {
            for h in 0..n_kv_head {
                let head_offset = h * chunk_size * head_dim;
                let k_head = &chunk.k[head_offset..head_offset + chunk_size * head_dim];
                let v_head = &chunk.v[head_offset..head_offset + chunk_size * head_dim];

                let q_start = (chunk_size - 1) * head_dim;
                let query = &k_head[q_start..q_start + head_dim];

                let k_energy: f64 = k_head.iter().map(|&x| x as f64 * x as f64).sum();
                if k_energy < 1e-10 {
                    continue;
                }

                // K is locked at Q8_0, V is selected per-block by v_thr
                // Quantize K as Q8_0
                let sqrt_d = (head_dim as f32).sqrt();
                let mut k_q = vec![0.0f32; k_head.len()];
                for b in (0..k_head.len()).step_by(SELECT_BLOCK) {
                    let end = (b + SELECT_BLOCK).min(k_head.len());
                    if end - b == SELECT_BLOCK {
                        let blk: [f32; SELECT_BLOCK] = k_head[b..end].try_into().unwrap();
                        let recon = BlockFormat::Q8_0.apply_quant(&blk);
                        k_q[b..end].copy_from_slice(&recon);
                    } else {
                        k_q[b..end].copy_from_slice(&k_head[b..end]);
                    }
                }

                // Quantize V per-block using v_thr selection
                let eff_thr = v_thr;
                let mut v_q = vec![0.0f32; v_head.len()];
                for b in (0..v_head.len()).step_by(SELECT_BLOCK) {
                    let end = (b + SELECT_BLOCK).min(v_head.len());
                    if end - b == SELECT_BLOCK {
                        let blk: [f32; SELECT_BLOCK] = v_head[b..end].try_into().unwrap();
                        let v_fmt = select_format(&blk, eff_thr);
                        let recon = v_fmt.apply_quant(&blk);
                        v_q[b..end].copy_from_slice(&recon);
                        v_bpe_sum += v_fmt.bits_per_elem() as f64;
                        v_blocks += 1;
                    } else {
                        v_q[b..end].copy_from_slice(&v_head[b..end]);
                    }
                }

                // Compute attention with quantized K/V
                let mut scores = vec![0.0f32; chunk_size];
                for t in 0..chunk_size {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += query[d] * k_q[t * head_dim + d];
                    }
                    scores[t] = dot / sqrt_d;
                }
                let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut exp_sum = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    exp_sum += *s;
                }
                for s in &mut scores {
                    *s /= exp_sum;
                }
                let mut quant_out = vec![0.0f32; head_dim];
                for t in 0..chunk_size {
                    for d in 0..head_dim {
                        quant_out[d] += scores[t] * v_q[t * head_dim + d];
                    }
                }

                // Reference output (float K/V)
                let mut ref_scores = vec![0.0f32; chunk_size];
                for t in 0..chunk_size {
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += query[d] * k_head[t * head_dim + d];
                    }
                    ref_scores[t] = dot / sqrt_d;
                }
                let ref_max_s = ref_scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut ref_exp_sum = 0.0f32;
                for s in &mut ref_scores {
                    *s = (*s - ref_max_s).exp();
                    ref_exp_sum += *s;
                }
                for s in &mut ref_scores {
                    *s /= ref_exp_sum;
                }
                let mut ref_out = vec![0.0f32; head_dim];
                for t in 0..chunk_size {
                    for d in 0..head_dim {
                        ref_out[d] += ref_scores[t] * v_head[t * head_dim + d];
                    }
                }

                let cd = vec_cosine_distance(&ref_out, &quant_out);
                let nr = vec_nrmse(&ref_out, &quant_out);
                stats.cos_dists.push(cd);
                stats.nrmses.push(nr);
            }
        }

        let v_bpe = if v_blocks > 0 {
            v_bpe_sum / v_blocks as f64
        } else {
            0.0
        };
        let status = if stats.p95_cos() <= c4_p95 {
            "OK"
        } else {
            "WORSE"
        };
        println!(
            "  {:<10.4} {:>10.7} {:>10.7} {:>10.7} {:>12.7} {:>6.2} {:>8}",
            v_thr,
            stats.mean_cos(),
            stats.p95_cos(),
            stats.max_cos(),
            stats.mean_nrmse(),
            v_bpe,
            status
        );
    }

    println!();
    println!("  Status: OK = cos_p95 ≤ C4 reference, WORSE = exceeds C4 quality");
    println!("  V_BPE: mean bits/element for V (lower = more compression)");
    println!("  Fix: use the tightest V_thr that shows OK to get C3 quality with distinct CR.");
    println!("{sep}");
}

// ============================================================================
// New candidate-list compression curve model
// ============================================================================

/// Select the most aggressive format from a candidate list whose cosine distance
/// is within `threshold`.  Only evaluates quantized entries; float entries are
/// skipped.  Returns the selected `BlockFormat`, or the float fallback format
/// if no quantized candidate passes.
///
/// Candidates are sorted ascending by BPE (best compression first).  The first
/// format that passes the threshold is returned immediately — no need to
/// evaluate the rest since later candidates have equal or higher BPE.
/// Low-value error margin: absolute dead-zone (matches CUDA ERROR_MARGIN_ABS).
/// Subtracts a fixed dead-zone from per-element absolute error before computing
/// the loss metric, so near-zero values don't inflate relative error.

/// Apply dead-zone error margin to a reconstructed block.
/// For each element, shrinks |orig - recon| by `margin`, clamping at zero,
/// returning an adjusted recon that is closer to orig for small differences.
fn apply_error_margin_block(
    orig: &[f32; SELECT_BLOCK],
    recon: &[f32; SELECT_BLOCK],
    margin: f32,
) -> [f32; SELECT_BLOCK] {
    let mut adj = [0.0f32; SELECT_BLOCK];
    for i in 0..SELECT_BLOCK {
        let diff = orig[i] - recon[i];
        let abs_diff = diff.abs();
        let adj_diff = (abs_diff - margin).max(0.0);
        adj[i] = orig[i] - adj_diff.copysign(diff);
    }
    adj
}

fn select_format_from_candidates(
    block: &[f32; SELECT_BLOCK],
    candidates: &[BlockFormat],
    threshold: f32,
) -> BlockFormat {
    select_best_passing_format(block, candidates, threshold, false)
}

fn block_format_from_kv(fmt: KvFormat) -> BlockFormat {
    use BlockFormat::*;

    match fmt {
        KvFormat::Float(candle::DType::F16) => F16,
        KvFormat::Float(candle::DType::BF16) => BF16,
        KvFormat::Quantized(QuantFormat::Q8_KS) => Q8_KS,
        KvFormat::Quantized(QuantFormat::Q8_1) => Q8_1,
        KvFormat::Quantized(QuantFormat::Q8_0) => Q8_0,
        KvFormat::Quantized(QuantFormat::Q4_KS) => Q4_KS,
        KvFormat::Quantized(QuantFormat::Q4_1) => Q4_1,
        KvFormat::Quantized(QuantFormat::Q4_0) => Q4_0,
        KvFormat::Quantized(QuantFormat::Q3_1) => Q3_1,
        KvFormat::Quantized(QuantFormat::Q3_0) => Q3_0,
        KvFormat::Quantized(QuantFormat::Q2_1) => Q2_1,
        KvFormat::Quantized(QuantFormat::Q2_A) => Q2_A,
        KvFormat::Quantized(QuantFormat::Q2_S) => Q2_S,
        KvFormat::Quantized(QuantFormat::Q2_0) => Q2_0,
        KvFormat::Quantized(QuantFormat::Q1_S) => Q1_S,
        KvFormat::Quantized(QuantFormat::Q0_V) => Q0_V,
        KvFormat::Quantized(QuantFormat::Q1_A) => Q1_A,
        KvFormat::Quantized(QuantFormat::Q0_X) => Q0_X,
        KvFormat::Quantized(QuantFormat::Q0_M2) => Q0_M2,
        KvFormat::Quantized(QuantFormat::Q0_M4) => Q0_M4,
        KvFormat::Quantized(QuantFormat::Q0) => Q0,
        KvFormat::Quantized(QuantFormat::Q0_V) => Q0_V,
        KvFormat::Quantized(QuantFormat::Q1_A) => Q1_A,
        KvFormat::Quantized(QuantFormat::Q0_X) => Q0_X,
        KvFormat::Quantized(QuantFormat::Q0_M2) => Q0_M2,
        KvFormat::Quantized(QuantFormat::Q0_M4) => Q0_M4,
        other => panic!("unsupported production candidate in table harness: {other:?}"),
    }
}

#[cfg(feature = "cuda")]
fn sample_format_from_block(fmt: BlockFormat) -> SampleFormat {
    match fmt {
        BlockFormat::F16 => SampleFormat::F16,
        BlockFormat::BF16 => SampleFormat::BF16,
        BlockFormat::Q8_KS => SampleFormat::Q8KS,
        BlockFormat::Q8_1 => SampleFormat::Q8_1,
        BlockFormat::Q8_0 => SampleFormat::Q8_0,
        BlockFormat::Q4_KS => SampleFormat::Q4KS,
        BlockFormat::Q4_1 => SampleFormat::Q4_1,
        BlockFormat::Q4_0 => SampleFormat::Q4_0,
        BlockFormat::Q3_1 => SampleFormat::Q3_1,
        BlockFormat::Q3_0 => SampleFormat::Q3_0,
        BlockFormat::Q2_1 => SampleFormat::Q2_1,
        BlockFormat::Q2_A => SampleFormat::Q2A,
        BlockFormat::Q2_S => SampleFormat::Q2S,
        BlockFormat::Q2_0 => SampleFormat::Q2_0,
        BlockFormat::Q1_S => SampleFormat::Q1S,
        BlockFormat::Q0 => SampleFormat::Q0,
        BlockFormat::Q0_V => SampleFormat::Q0_V,
        BlockFormat::Q1_A => SampleFormat::Q1_A,
        BlockFormat::Q0_X => SampleFormat::Q0_X,
        BlockFormat::Q0_M2 => SampleFormat::Q0_M2,
        BlockFormat::Q0_M4 => SampleFormat::Q0_M4,
    }
}

#[cfg(feature = "cuda")]
fn block_format_from_sample(fmt: SampleFormat) -> BlockFormat {
    match fmt {
        SampleFormat::F16 => BlockFormat::F16,
        SampleFormat::BF16 => BlockFormat::BF16,
        SampleFormat::Q8KS => BlockFormat::Q8_KS,
        SampleFormat::Q8_1 => BlockFormat::Q8_1,
        SampleFormat::Q8_0 => BlockFormat::Q8_0,
        SampleFormat::Q4KS => BlockFormat::Q4_KS,
        SampleFormat::Q4_1 => BlockFormat::Q4_1,
        SampleFormat::Q4_0 => BlockFormat::Q4_0,
        SampleFormat::Q3_1 => BlockFormat::Q3_1,
        SampleFormat::Q3_0 => BlockFormat::Q3_0,
        SampleFormat::Q2_1 => BlockFormat::Q2_1,
        SampleFormat::Q2A => BlockFormat::Q2_A,
        SampleFormat::Q2S => BlockFormat::Q2_S,
        SampleFormat::Q2_0 => BlockFormat::Q2_0,
        SampleFormat::Q1S => BlockFormat::Q1_S,
        SampleFormat::Q0_V => BlockFormat::Q0_V,
        SampleFormat::Q0_M => BlockFormat::Q0_M,
        SampleFormat::Q0_X => BlockFormat::Q0_X,
        SampleFormat::Q0_M2 => BlockFormat::Q0_M2,
        SampleFormat::Q0_M4 => BlockFormat::Q0_M4,
        SampleFormat::Q0 => BlockFormat::Q0,
    }
}

/// Per-level candidate lists are sourced directly from the shared production profile.
fn level_candidates() -> Vec<(Vec<BlockFormat>, Vec<BlockFormat>)> {
    (0..11)
        .map(|level| {
            let (k_candidates, v_candidates) =
                CompressionPolicy::production_candidates(level as u8);
            (
                k_candidates.into_iter().map(block_format_from_kv).collect(),
                v_candidates.into_iter().map(block_format_from_kv).collect(),
            )
        })
        .collect()
}

/// Collect per-level statistics using the new per-level candidate lists.
/// Returns [k_stats, v_stats] for each level.
#[allow(dead_code)]
fn curve_level_stats_v2(
    chunks: &[ChunkData],
    num_blocks_per_chunk: usize,
    k_threshold: f32,
    v_threshold: f32,
    k_candidates: &[BlockFormat],
    v_candidates: &[BlockFormat],
) -> [CurveStat; 2] {
    let mut s: [CurveStat; 2] = Default::default();
    for chunk in chunks {
        for b in 0..num_blocks_per_chunk {
            let start = b * SELECT_BLOCK;
            let end = start + SELECT_BLOCK;

            // Sink-aware thresholds
            let kt = k_threshold;
            let vt = v_threshold;

            if end <= chunk.k.len() {
                let blk: [f32; SELECT_BLOCK] = chunk.k[start..end].try_into().unwrap();
                let fmt = select_format_from_candidates_k(&blk, k_candidates, kt);
                s[0].push(&blk, fmt);
            }
            if end <= chunk.v.len() {
                let blk: [f32; SELECT_BLOCK] = chunk.v[start..end].try_into().unwrap();
                let fmt = select_format_from_candidates(&blk, v_candidates, vt);
                s[1].push(&blk, fmt);
            }
        }
    }
    s
}

#[derive(Debug, Clone, Copy, Default)]
struct SelectionPassTiming {
    qrel_mask: Duration,
    block_select: Duration,
    stats_fold: Duration,
}

/// Combined: collects block-level format assignments AND quality stats in one pass.
/// Returns ([k_stats, v_stats], k_block_fmts, v_block_fmts).
/// This replaces calling both `curve_level_stats_v2` and `collect_block_formats` separately.
fn collect_formats_and_stats(
    chunks: &[ChunkData],
    blocks_per_chunk: usize,
    k_threshold: f32,
    v_threshold: f32,
    k_high_threshold: f32,
    k_low_threshold: f32,
    v_high_threshold: f32,
    v_low_threshold: f32,
    k_candidates: &[BlockFormat],
    v_candidates: &[BlockFormat],
    q_relevance_split: Option<f32>,
    gpu_selector: Option<&GpuSelector>,
) -> (
    [CurveStat; 2],
    [CurveStat; 2],
    [CurveStat; 2],
    Vec<BlockFormat>,
    Vec<BlockFormat>,
    Vec<bool>,
    SelectionPassTiming,
) {
    let qrel_mask_start = Instant::now();
    let mut qrel_hi_mask = Vec::with_capacity(chunks.len() * blocks_per_chunk);
    for chunk in chunks {
        for b in 0..blocks_per_chunk {
            let start = b * SELECT_BLOCK;
            let end = start + SELECT_BLOCK;
            if end <= chunk.k.len() {
                let blk: [f32; SELECT_BLOCK] = chunk.k[start..end].try_into().unwrap();
                let is_high = if let Some(split) = q_relevance_split {
                    let rel = if let Some(q) = chunk.q.as_ref() {
                        if end <= q.len() {
                            let q_blk: [f32; SELECT_BLOCK] = q[start..end].try_into().unwrap();
                            cpu_block_relevance(&blk, &q_blk)
                        } else {
                            blk.iter().map(|&x| x * x).sum::<f32>() / SELECT_BLOCK as f32
                        }
                    } else {
                        blk.iter().map(|&x| x * x).sum::<f32>() / SELECT_BLOCK as f32
                    };
                    rel >= split
                } else {
                    false
                };
                qrel_hi_mask.push(is_high);
            }
        }
    }
    let qrel_mask_time = qrel_mask_start.elapsed();

    #[cfg(feature = "cuda")]
    if let Some(gpu) = gpu_selector {
        let k_sample_candidates: Vec<SampleFormat> = k_candidates
            .iter()
            .copied()
            .map(sample_format_from_block)
            .collect();
        let v_sample_candidates: Vec<SampleFormat> = v_candidates
            .iter()
            .copied()
            .map(sample_format_from_block)
            .collect();
        let select_start = Instant::now();
        let (k_selected, v_selected) = gpu
            .select_block_formats(
                &k_sample_candidates,
                &v_sample_candidates,
                k_high_threshold,
                k_low_threshold,
                v_high_threshold,
                v_low_threshold,
            )
            .expect("CUDA paged selection failed");
        let block_select_time = select_start.elapsed();

        let k_fmts: Vec<BlockFormat> = k_selected
            .into_iter()
            .map(block_format_from_sample)
            .collect();
        let v_fmts: Vec<BlockFormat> = v_selected
            .into_iter()
            .map(block_format_from_sample)
            .collect();

        let stats_fold_start = Instant::now();
        let mut s: [CurveStat; 2] = Default::default();
        let mut s_hi: [CurveStat; 2] = Default::default();
        let mut s_lo: [CurveStat; 2] = Default::default();
        let mut k_idx = 0usize;
        let mut v_idx = 0usize;

        for chunk in chunks {
            for b in 0..blocks_per_chunk {
                let start = b * SELECT_BLOCK;
                let end = start + SELECT_BLOCK;
                if end <= chunk.k.len() {
                    let blk: [f32; SELECT_BLOCK] = chunk.k[start..end].try_into().unwrap();
                    let fmt = k_fmts[k_idx];
                    let is_high = qrel_hi_mask.get(k_idx).copied().unwrap_or(false);
                    s[0].push(&blk, fmt);
                    if is_high {
                        s_hi[0].push(&blk, fmt);
                    } else {
                        s_lo[0].push(&blk, fmt);
                    }
                    k_idx += 1;
                }
                if end <= chunk.v.len() {
                    let blk: [f32; SELECT_BLOCK] = chunk.v[start..end].try_into().unwrap();
                    let fmt = v_fmts[v_idx];
                    let is_high = qrel_hi_mask.get(v_idx).copied().unwrap_or(false);
                    s[1].push(&blk, fmt);
                    if is_high {
                        s_hi[1].push(&blk, fmt);
                    } else {
                        s_lo[1].push(&blk, fmt);
                    }
                    v_idx += 1;
                }
            }
        }
        let stats_fold_time = stats_fold_start.elapsed();

        return (
            s,
            s_hi,
            s_lo,
            k_fmts,
            v_fmts,
            qrel_hi_mask,
            SelectionPassTiming {
                qrel_mask: qrel_mask_time,
                block_select: block_select_time,
                stats_fold: stats_fold_time,
            },
        );
    }

    let loop_start = Instant::now();
    let mut s: [CurveStat; 2] = Default::default();
    let mut s_hi: [CurveStat; 2] = Default::default();
    let mut s_lo: [CurveStat; 2] = Default::default();
    let mut k_fmts = Vec::with_capacity(chunks.len() * blocks_per_chunk);
    let mut v_fmts = Vec::with_capacity(chunks.len() * blocks_per_chunk);
    let mut qrel_idx = 0usize;
    for chunk in chunks {
        let kt_base = k_threshold;
        let vt_base = v_threshold;
        let kt_hi = k_high_threshold;
        let kt_lo = k_low_threshold;
        let vt_hi = v_high_threshold;
        let vt_lo = v_low_threshold;
        for b in 0..blocks_per_chunk {
            let start = b * SELECT_BLOCK;
            let end = start + SELECT_BLOCK;
            let mut qrel_bucket: Option<bool> = None;
            let mut q_relevance = 0.0f32;
            if end <= chunk.k.len() {
                let blk: [f32; SELECT_BLOCK] = chunk.k[start..end].try_into().unwrap();
                let is_high = qrel_hi_mask.get(qrel_idx).copied().unwrap_or(false);
                qrel_bucket = q_relevance_split.map(|_| is_high);
                if q_relevance_split.is_some() {
                    q_relevance = if let Some(q) = chunk.q.as_ref() {
                        if end <= q.len() {
                            let q_blk: [f32; SELECT_BLOCK] = q[start..end].try_into().unwrap();
                            cpu_block_relevance(&blk, &q_blk)
                        } else {
                            0.0
                        }
                    } else {
                        0.0
                    };
                }
                let kt = if let Some(split) = q_relevance_split {
                    qrel_threshold(
                        kt_lo,
                        kt_hi,
                        q_relevance,
                        0.0,
                        split,
                        K_QREL_GAMMAS[level.min(10)],
                    )
                } else {
                    kt_base
                };
                let fmt = select_format_from_candidates_k(&blk, k_candidates, kt);
                s[0].push(&blk, fmt);
                if is_high {
                    s_hi[0].push(&blk, fmt);
                } else {
                    s_lo[0].push(&blk, fmt);
                }
                k_fmts.push(fmt);
                qrel_idx += 1;
            }
            if end <= chunk.v.len() {
                let blk: [f32; SELECT_BLOCK] = chunk.v[start..end].try_into().unwrap();
                let vt = if let Some(split) = q_relevance_split {
                    qrel_threshold(
                        vt_lo,
                        vt_hi,
                        q_relevance,
                        0.0,
                        split,
                        V_QREL_GAMMAS[level.min(10)],
                    )
                } else {
                    vt_base
                };
                let fmt = select_format_from_candidates(&blk, v_candidates, vt);
                s[1].push(&blk, fmt);
                if qrel_bucket.unwrap_or(false) {
                    s_hi[1].push(&blk, fmt);
                } else {
                    s_lo[1].push(&blk, fmt);
                }
                v_fmts.push(fmt);
            }
        }
    }
    let block_select_time = loop_start.elapsed();
    (
        s,
        s_hi,
        s_lo,
        k_fmts,
        v_fmts,
        qrel_hi_mask,
        SelectionPassTiming {
            qrel_mask: qrel_mask_time,
            block_select: block_select_time,
            stats_fold: Duration::default(),
        },
    )
}

// ============================================================================
// Threshold sweep for K/V error margin optimization
// ============================================================================

/// Sweep a single threshold across all blocks for one side (K or V),
/// using the given candidate list. Returns (bpe, snr_db, nrmse, cos_p95).
fn evenly_sampled_chunks(chunks: &[ChunkData], target_chunks: usize) -> Vec<ChunkData> {
    let target_chunks = target_chunks.max(1);
    if chunks.len() <= target_chunks {
        return chunks
            .iter()
            .map(|chunk| ChunkData {
                layer_idx: chunk.layer_idx,
                block_idx: chunk.block_idx,
                token_start: chunk.token_start,
                k: chunk.k.clone(),
                v: chunk.v.clone(),
                q: chunk.q.clone(),
            })
            .collect();
    }
    let last = chunks.len() - 1;
    let mut sampled = Vec::with_capacity(target_chunks);
    let mut prev_idx = None;
    for i in 0..target_chunks {
        let idx = if target_chunks <= 1 {
            0
        } else {
            i * last / (target_chunks - 1)
        };
        if prev_idx != Some(idx) {
            let chunk = &chunks[idx];
            sampled.push(ChunkData {
                layer_idx: chunk.layer_idx,
                block_idx: chunk.block_idx,
                token_start: chunk.token_start,
                k: chunk.k.clone(),
                v: chunk.v.clone(),
                q: chunk.q.clone(),
            });
            prev_idx = Some(idx);
        }
    }
    sampled
}

fn sweep_single_side(
    chunks: &[ChunkData],
    num_blocks_per_chunk: usize,
    threshold: f32,
    candidates: &[BlockFormat],
    is_key: bool,
) -> (f64, f64, f64, f64) {
    let mut stat = CurveStat::default();
    for chunk in chunks {
        let data = if is_key { &chunk.k } else { &chunk.v };
        let t = threshold;
        for b in 0..num_blocks_per_chunk {
            let start = b * SELECT_BLOCK;
            let end = start + SELECT_BLOCK;
            if end <= data.len() {
                let blk: [f32; SELECT_BLOCK] = data[start..end].try_into().unwrap();
                let fmt = if is_key {
                    select_format_from_candidates_k(&blk, candidates, t)
                } else {
                    select_format_from_candidates(&blk, candidates, t)
                };
                stat.push(&blk, fmt);
            }
        }
    }
    (stat.bpe(), stat.snr_db(), stat.nrmse(), stat.p95())
}

/// Threshold sweep test — produces CSV data for graphing the compression
/// ratio landscape as a function of K and V error margins.
///
/// Outputs CSV sections for K-sweep, V-sweep, combined CR grid, and
/// per-level candidate-aware sweeps.
///
/// The CSV is written to `tests/kv_threshold_sweep.csv`.
///
/// Run with:
///   cargo test --release --lib --package candle-nn \
///     kv_selection_tests::test_threshold_sweep -- --ignored --nocapture
#[test]
#[ignore]
fn test_threshold_sweep() {
    let path = match dump_path() {
        Some(p) => p,
        None => {
            println!("kv_selection_tests: dump absent — run test_dump_kv_cache_data first.");
            return;
        }
    };

    let (header, chunks) = load_dump(&path).expect("failed to load dump");
    let elems = header.n_kv_head * header.chunk_size * header.head_dim;
    let num_blocks = elems / SELECT_BLOCK;
    let candidates = level_candidates();

    // Use C9's candidate lists (most inclusive quantized formats)
    // to isolate the threshold effect from candidate-list differences.
    let (ref k_cands, ref v_cands) = candidates[9];

    // Threshold sweep points — log-spaced from 1e-6 to 0.3
    let sweep_thresholds: Vec<f32> = {
        let mut v = Vec::new();
        let log_min = -6.0f64;
        let log_max = -0.5f64; // 10^(-0.5) ≈ 0.316
        let n_steps = 200;
        for i in 0..=n_steps {
            let log_val = log_min + (log_max - log_min) * i as f64 / n_steps as f64;
            v.push(10.0f64.powf(log_val) as f32);
        }
        v
    };

    println!(
        "Sweeping {} threshold points across {} chunks ({} blocks/chunk)...",
        sweep_thresholds.len(),
        chunks.len(),
        num_blocks
    );

    // ── Section 1: K-sweep ──
    let k_results: Vec<(f32, f64, f64, f64, f64)> = sweep_thresholds
        .iter()
        .map(|&t| {
            let (bpe, snr, nrmse, p95) = sweep_single_side(&chunks, num_blocks, t, k_cands, true);
            (t, bpe, snr, nrmse, p95)
        })
        .collect();

    // ── Section 2: V-sweep ──
    let v_results: Vec<(f32, f64, f64, f64, f64)> = sweep_thresholds
        .iter()
        .map(|&t| {
            let (bpe, snr, nrmse, p95) = sweep_single_side(&chunks, num_blocks, t, v_cands, false);
            (t, bpe, snr, nrmse, p95)
        })
        .collect();

    // ── Section 3: Combined CR grid (coarse) ──
    let coarse_thresholds: Vec<f32> = {
        let mut v = Vec::new();
        let log_min = -6.0f64;
        let log_max = -0.5f64; // 10^(-0.5) ≈ 0.316
        let n_steps = 40;
        for i in 0..=n_steps {
            let log_val = log_min + (log_max - log_min) * i as f64 / n_steps as f64;
            v.push(10.0f64.powf(log_val) as f32);
        }
        v
    };

    println!(
        "Computing {}×{} combined CR grid...",
        coarse_thresholds.len(),
        coarse_thresholds.len()
    );

    let k_bpe_at: Vec<f64> = coarse_thresholds
        .iter()
        .map(|&t| sweep_single_side(&chunks, num_blocks, t, k_cands, true).0)
        .collect();
    let v_bpe_at: Vec<f64> = coarse_thresholds
        .iter()
        .map(|&t| sweep_single_side(&chunks, num_blocks, t, v_cands, false).0)
        .collect();

    // ── Write CSV ──
    let csv_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("kv_threshold_sweep.csv");
    let mut csv = String::new();

    csv += "# KV Threshold Sweep Data\n";
    csv += &format!(
        "# {} layers, {} chunks, {} blocks/chunk\n",
        header.num_layers,
        chunks.len(),
        num_blocks,
    );
    csv += &format!(
        "# K candidates (C9): {}\n",
        k_cands
            .iter()
            .map(|f| format!("{f}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    csv += &format!(
        "# V candidates (C9): {}\n",
        v_cands
            .iter()
            .map(|f| format!("{f}"))
            .collect::<Vec<_>>()
            .join(",")
    );

    csv += "\n# SECTION: K_SWEEP\n";
    csv += "k_threshold,k_bpe,k_snr_db,k_nrmse,k_cos_p95,k_cr\n";
    for &(t, bpe, snr, nrmse, p95) in &k_results {
        let cr = 16.0 / bpe;
        csv += &format!("{t:.8},{bpe:.4},{snr:.2},{nrmse:.6},{p95:.8},{cr:.4}\n");
    }

    csv += "\n# SECTION: V_SWEEP\n";
    csv += "v_threshold,v_bpe,v_snr_db,v_nrmse,v_cos_p95,v_cr\n";
    for &(t, bpe, snr, nrmse, p95) in &v_results {
        let cr = 16.0 / bpe;
        csv += &format!("{t:.8},{bpe:.4},{snr:.2},{nrmse:.6},{p95:.8},{cr:.4}\n");
    }

    csv += "\n# SECTION: COMBINED_CR_GRID\n";
    csv += "# Rows = K thresholds, Columns = V thresholds\n";
    csv += "# Values = combined CR = 16 / ((k_bpe + v_bpe) / 2)\n";
    csv += "k_thr\\v_thr";
    for &vt in &coarse_thresholds {
        csv += &format!(",{vt:.8}");
    }
    csv += "\n";
    for (ki, &kt) in coarse_thresholds.iter().enumerate() {
        csv += &format!("{kt:.8}");
        for vi in 0..coarse_thresholds.len() {
            let combined_bpe = (k_bpe_at[ki] + v_bpe_at[vi]) / 2.0;
            let cr = 16.0 / combined_bpe;
            csv += &format!(",{cr:.4}");
        }
        csv += "\n";
    }

    csv += "\n# SECTION: PER_LEVEL_K_SWEEP\n";
    csv += "level,k_threshold,k_bpe,k_snr_db,k_cos_p95\n";
    for level in 0..10 {
        let (ref kc, _) = candidates[level];
        for &t in &sweep_thresholds {
            let (bpe, snr, _, p95) = sweep_single_side(&chunks, num_blocks, t, kc, true);
            csv += &format!("{level},{t:.8},{bpe:.4},{snr:.2},{p95:.8}\n");
        }
    }

    csv += "\n# SECTION: PER_LEVEL_V_SWEEP\n";
    csv += "level,v_threshold,v_bpe,v_snr_db,v_cos_p95\n";
    for level in 0..10 {
        let (_, ref vc) = candidates[level];
        for &t in &sweep_thresholds {
            let (bpe, snr, _, p95) = sweep_single_side(&chunks, num_blocks, t, vc, false);
            csv += &format!("{level},{t:.8},{bpe:.4},{snr:.2},{p95:.8}\n");
        }
    }

    std::fs::write(&csv_path, &csv).expect("write CSV");
    println!("\nCSV written to: {}", csv_path.display());

    // ── Print summary table ──
    let sep = "═".repeat(100);
    println!("\n{sep}");
    println!("  K Threshold → BPE/CR step-change points  (>0.1 BPE drop from previous)");
    println!("{sep}");
    println!(
        "  {:>12}  {:>7}  {:>7}  {:>9}  {:>11}",
        "K_Threshold", "K_BPE", "K_CR", "K_SNR_dB", "K_cos_p95"
    );
    let mut prev_bpe = 16.0f64;
    for &(t, bpe, snr, _nrmse, p95) in &k_results {
        if prev_bpe - bpe > 0.1
            || (t - 1e-6).abs() < 1e-7
            || (t - sweep_thresholds.last().unwrap()).abs() < 1e-7
        {
            let cr = 16.0 / bpe;
            println!(
                "  {:>12.8}  {:>7.3}  {:>6.2}x  {:>9.2}  {:>11.8}",
                t, bpe, cr, snr, p95
            );
        }
        prev_bpe = bpe;
    }

    println!("\n{sep}");
    println!("  V Threshold → BPE/CR step-change points  (>0.1 BPE drop from previous)");
    println!("{sep}");
    println!(
        "  {:>12}  {:>7}  {:>7}  {:>9}  {:>11}",
        "V_Threshold", "V_BPE", "V_CR", "V_SNR_dB", "V_cos_p95"
    );
    prev_bpe = 16.0;
    for &(t, bpe, snr, _nrmse, p95) in &v_results {
        if prev_bpe - bpe > 0.1
            || (t - 1e-6).abs() < 1e-7
            || (t - sweep_thresholds.last().unwrap()).abs() < 1e-7
        {
            let cr = 16.0 / bpe;
            println!(
                "  {:>12.8}  {:>7.3}  {:>6.2}x  {:>9.2}  {:>11.8}",
                t, bpe, cr, snr, p95
            );
        }
        prev_bpe = bpe;
    }

    // ── Print combined CR at key (K_thr, V_thr) pairs ──
    println!("\n{sep}");
    println!("  Combined CR heatmap (selected slices)");
    println!("  CR = 16 / ((K_BPE + V_BPE) / 2)");
    println!("{sep}");
    let v_display_indices: Vec<usize> = (0..coarse_thresholds.len())
        .step_by(coarse_thresholds.len() / 8)
        .collect();
    print!("  {:>12}", "K_thr \\ V_thr");
    for &vi in &v_display_indices {
        print!("  {:>10.6}", coarse_thresholds[vi]);
    }
    println!();
    for (ki, &kt) in coarse_thresholds
        .iter()
        .enumerate()
        .step_by(coarse_thresholds.len() / 12)
    {
        print!("  {:>12.8}", kt);
        for &vi in &v_display_indices {
            let combined_bpe = (k_bpe_at[ki] + v_bpe_at[vi]) / 2.0;
            let cr = 16.0 / combined_bpe;
            print!("  {:>10.3}x", cr);
        }
        println!();
    }
    println!("{sep}");
}

/// Proposed compression curve v2 — models the new per-level candidate lists.
///
/// Each C-level has independent K and V candidate lists.  The selection kernel
/// only evaluates quantized entries; float entries (F16 for K, BF16 for V) act
/// as fallback when no quantized candidate is within threshold.
///
/// Two compression levers:
///   - Add aggressive tail: lower levels add Q4_0, Q3_0, Q2_0
///   - Raise floor: higher levels remove F16/BF16 and Q8_0
///
/// Run with:
///   cargo test --release --lib --package candle-nn \
///     kv_selection_tests::test_candidate_list_compression_curve -- --ignored --nocapture
#[test]
#[ignore]
fn test_candidate_list_compression_curve() {
    let total_start = Instant::now();
    let qwen3_path = dump_path_for(QWEN3_DUMP_REL_PATH);
    let llama_path = dump_path_for(LLAMA_DUMP_REL_PATH);
    if qwen3_path.is_none() && llama_path.is_none() {
        println!("kv_selection_tests: no dump files found — run test_dump_kv_cache_data first.");
        return;
    }

    let load_start = Instant::now();
    let (header, mut chunks) = if let Some(p) = qwen3_path.as_ref() {
        load_dump(p).expect("failed to load qwen3-kv-data.bin")
    } else {
        load_dump(llama_path.as_ref().unwrap()).expect("failed to load llama-kv-data.bin")
    };
    let primary_name = if qwen3_path.is_some() { "qwen3-kv-data.bin" } else { "llama-kv-data.bin" };
    let mut data_sources = vec![primary_name.to_string()];
    if qwen3_path.is_some() {
        if let Some(p) = llama_path.as_ref() {
            if let Some((llama_header, llama_chunks)) = load_dump(p) {
                if llama_header.n_kv_head == header.n_kv_head
                    && llama_header.head_dim == header.head_dim
                    && llama_header.chunk_size == header.chunk_size
                {
                    chunks.extend(llama_chunks);
                    data_sources.push("llama-kv-data.bin".to_string());
                } else {
                    println!("llama-kv-data.bin header mismatch — skipping");
                }
            } else {
                println!("llama-kv-data.bin failed to parse — skipping");
            }
        }
    }
    let load_time = load_start.elapsed();

    // Pre-round all data through F16 to match GPU arena precision.
    // Production stores KV in F16 arenas before the format selector runs,
    // so cosine distances are computed against F16-rounded data, not F32.
    let round_start = Instant::now();
    for chunk in chunks.iter_mut() {
        for x in chunk.k.iter_mut() {
            *x = f16::from_f32(*x).to_f32();
        }
        for x in chunk.v.iter_mut() {
            *x = f16::from_f32(*x).to_f32();
        }
    }
    let f16_round_time = round_start.elapsed();

    let candidates = level_candidates();

    let v_auto_thresholds = V_COSINE_THRESHOLDS_PROPOSED;

    let sep = "=".repeat(115);
    let sep80 = "-".repeat(115);

    // ── Combined per-level computation (format selection + stats + reductions + quality) ──
    // Uses the shared sampled_selection CPU parallel dispatch so the unit test
    // stays focused on scenario setup and report rendering.
    let blocks_per_head = (header.chunk_size * header.head_dim) / SELECT_BLOCK;
    let blocks_per_chunk = header.n_kv_head * blocks_per_head;

    let _dump_tokens = chunks.len() * header.chunk_size;
    let has_q_relevance_capture = chunks.iter().any(|chunk| chunk.q.is_some());
    let qrel_split_start = Instant::now();
    let q_relevance_split =
        compute_q_relevance_split(&chunks, blocks_per_chunk, QREL_HIGH_PERCENTILE);
    let qrel_split_time = qrel_split_start.elapsed();

    let gpu_init_start = Instant::now();
    #[cfg(feature = "cuda")]
    let (_gpu_selection_generation, gpu_selector): (
        Option<candle::quantized::pinned_staging::Generation>,
        Option<GpuSelector>,
    ) = match candle::Device::cuda_if_available(0) {
        Ok(candle::Device::Cuda(cuda_dev)) => {
            let stager = candle::quantized::pinned_staging::PinnedStager::new(&cuda_dev);
            let generation = stager.begin_generation();
            let k_chunks: Vec<&[f32]> = chunks.iter().map(|chunk| chunk.k.as_slice()).collect();
            let v_chunks: Vec<&[f32]> = chunks.iter().map(|chunk| chunk.v.as_slice()).collect();
            let gpu = PagedSelectionGpuInputs::from_f32_chunks(
                &k_chunks,
                &v_chunks,
                blocks_per_chunk,
                header.n_kv_head,
                DEFAULT_REPORT_ARENA_CHUNKS,
                Some(&generation),
                &cuda_dev,
            )
            .ok();
            (Some(generation), gpu)
        }
        _ => (None, None),
    };
    #[cfg(not(feature = "cuda"))]
    let gpu_selector: Option<GpuSelector> = None;
    let gpu_selector_init_time = gpu_init_start.elapsed();

    #[allow(dead_code)]
    struct LevelResult {
        stats: [CurveStat; 2],
        _k_bpe: f64,
        _v_bpe: f64,
        cr: f64,
        k_head_dist: [usize; 16],
        v_head_dist: [usize; 16],
        k_pal4_dist: [usize; 21],
        v_pal4_dist: [usize; 21],
        k_head_dist_hi: [usize; 16],
        k_head_dist_lo: [usize; 16],
        v_head_dist_hi: [usize; 16],
        v_head_dist_lo: [usize; 16],
        k_total_heads: usize,
        v_total_heads: usize,
        k_total_pal4_blocks: usize,
        v_total_pal4_blocks: usize,
        k_total_heads_hi: usize,
        k_total_heads_lo: usize,
        v_total_heads_hi: usize,
        v_total_heads_lo: usize,
        k_bpe_p4: f64,
        v_bpe_p4: f64,
        cr_p4: f64,
        k_q_blk: (f64, f64, f64),
        v_q_blk: (f64, f64, f64),
        k_q_w1: (f64, f64, f64),
        v_q_w1: (f64, f64, f64),
        k_q_p4: (f64, f64, f64),
        v_q_p4: (f64, f64, f64),
        k_snr_hi: f64,
        k_snr_lo: f64,
        v_snr_hi: f64,
        v_snr_lo: f64,
        selection_timing: SelectionPassTiming,
        timing_reduction: Duration,
        timing_palette4: Duration,
        timing_quality: Duration,
    }

    let levels_compute_start = Instant::now();
    let levels: Vec<LevelResult> = cpu_parallel_kernel_range(11, |i| {
        // Single pass: format selection + block-level quality stats + format vectors
        let (stats, stats_hi, stats_lo, k_fmts, v_fmts, qrel_hi_mask, selection_timing) =
            collect_formats_and_stats(
                &chunks,
                blocks_per_chunk,
                K_MAGWEIGHT_THRESHOLDS[i],
                v_auto_thresholds[i],
                K_QREL_HIGH_THRESHOLDS[i],
                K_QREL_LOW_THRESHOLDS[i],
                V_QREL_HIGH_THRESHOLDS[i],
                V_QREL_LOW_THRESHOLDS[i],
                &candidates[i].0,
                &candidates[i].1,
                q_relevance_split,
                gpu_selector.as_ref(),
            );

        let reduction_start = Instant::now();
        let k_snr_hi = if stats_hi[0].n > 0 {
            stats_hi[0].snr_db()
        } else {
            f64::NAN
        };
        let k_snr_lo = if stats_lo[0].n > 0 {
            stats_lo[0].snr_db()
        } else {
            f64::NAN
        };
        let v_snr_hi = if stats_hi[1].n > 0 {
            stats_hi[1].snr_db()
        } else {
            f64::NAN
        };
        let v_snr_lo = if stats_lo[1].n > 0 {
            stats_lo[1].snr_db()
        } else {
            f64::NAN
        };

        // Block-level quality from CurveStat (already computed, no re-quant)
        let k_q_blk = (stats[0].snr_db(), stats[0].nrmse(), stats[0].p95());
        let v_q_blk = (stats[1].snr_db(), stats[1].nrmse(), stats[1].p95());

        let qrel_lo_mask: Vec<bool> = qrel_hi_mask.iter().map(|&is_high| !is_high).collect();
        let k_head_dist_hi = reduce_per_head_bucket_with_dist(
            &k_fmts,
            &qrel_hi_mask,
            blocks_per_chunk,
            blocks_per_head,
        );
        let k_head_dist_lo = reduce_per_head_bucket_with_dist(
            &k_fmts,
            &qrel_lo_mask,
            blocks_per_chunk,
            blocks_per_head,
        );
        let v_head_dist_hi = reduce_per_head_bucket_with_dist(
            &v_fmts,
            &qrel_hi_mask,
            blocks_per_chunk,
            blocks_per_head,
        );
        let v_head_dist_lo = reduce_per_head_bucket_with_dist(
            &v_fmts,
            &qrel_lo_mask,
            blocks_per_chunk,
            blocks_per_head,
        );

        #[cfg(feature = "cuda")]
        let gpu_quality_agg = if let Some(gpu) = gpu_selector.as_ref() {
            let k_sample: Vec<SampleFormat> = k_fmts
                .iter()
                .copied()
                .map(sample_format_from_block)
                .collect();
            let v_sample: Vec<SampleFormat> = v_fmts
                .iter()
                .copied()
                .map(sample_format_from_block)
                .collect();
            Some(
                gpu.aggregate_quality_metric_formats_gpu(&k_sample, &v_sample)
                    .expect("GPU quality aggregation failed"),
            )
        } else {
            None
        };

        #[cfg(feature = "cuda")]
        let (k_quant_bpe, v_quant_bpe, k_head_dist, v_head_dist, k_w1_fmts, v_w1_fmts) =
            if let Some(gpu_agg) = gpu_quality_agg.as_ref() {
                let k_head_tags: Vec<BlockFormat> = gpu_agg
                    .k_head_formats
                    .iter()
                    .copied()
                    .map(block_format_from_sample)
                    .collect();
                let v_head_tags: Vec<BlockFormat> = gpu_agg
                    .v_head_formats
                    .iter()
                    .copied()
                    .map(block_format_from_sample)
                    .collect();
                let k_w1_fmts: Vec<BlockFormat> = gpu_agg
                    .k_worst_case_formats
                    .iter()
                    .copied()
                    .map(block_format_from_sample)
                    .collect();
                let v_w1_fmts: Vec<BlockFormat> = gpu_agg
                    .v_worst_case_formats
                    .iter()
                    .copied()
                    .map(block_format_from_sample)
                    .collect();
                let (k_quant_bpe, k_head_dist) =
                    reduce_head_tags_with_dist(&k_head_tags, blocks_per_head);
                let (v_quant_bpe, v_head_dist) =
                    reduce_head_tags_with_dist(&v_head_tags, blocks_per_head);

                let (cpu_k_bpe, cpu_k_dist) =
                    reduce_per_head_with_dist(&k_fmts, blocks_per_chunk, blocks_per_head);
                let (cpu_v_bpe, cpu_v_dist) =
                    reduce_per_head_with_dist(&v_fmts, blocks_per_chunk, blocks_per_head);
                assert!(
                    (k_quant_bpe - cpu_k_bpe).abs() < 1e-6 && k_head_dist == cpu_k_dist,
                    "GPU K head reduction mismatch at level {i}"
                );
                assert!(
                    (v_quant_bpe - cpu_v_bpe).abs() < 1e-6 && v_head_dist == cpu_v_dist,
                    "GPU V head reduction mismatch at level {i}"
                );

                (
                    k_quant_bpe,
                    v_quant_bpe,
                    k_head_dist,
                    v_head_dist,
                    k_w1_fmts,
                    v_w1_fmts,
                )
            } else {
                let (k_quant_bpe, k_head_dist) =
                    reduce_per_head_with_dist(&k_fmts, blocks_per_chunk, blocks_per_head);
                let (v_quant_bpe, v_head_dist) =
                    reduce_per_head_with_dist(&v_fmts, blocks_per_chunk, blocks_per_head);
                let k_w1_fmts =
                    apply_worst_case_reduction(&k_fmts, blocks_per_chunk, blocks_per_head);
                let v_w1_fmts =
                    apply_worst_case_reduction(&v_fmts, blocks_per_chunk, blocks_per_head);
                (
                    k_quant_bpe,
                    v_quant_bpe,
                    k_head_dist,
                    v_head_dist,
                    k_w1_fmts,
                    v_w1_fmts,
                )
            };

        #[cfg(not(feature = "cuda"))]
        let (k_quant_bpe, v_quant_bpe, k_head_dist, v_head_dist, k_w1_fmts, v_w1_fmts) = {
            let (k_quant_bpe, k_head_dist) =
                reduce_per_head_with_dist(&k_fmts, blocks_per_chunk, blocks_per_head);
            let (v_quant_bpe, v_head_dist) =
                reduce_per_head_with_dist(&v_fmts, blocks_per_chunk, blocks_per_head);
            let k_w1_fmts = apply_worst_case_reduction(&k_fmts, blocks_per_chunk, blocks_per_head);
            let v_w1_fmts = apply_worst_case_reduction(&v_fmts, blocks_per_chunk, blocks_per_head);
            (
                k_quant_bpe,
                v_quant_bpe,
                k_head_dist,
                v_head_dist,
                k_w1_fmts,
                v_w1_fmts,
            )
        };

        let k_total_heads: usize = k_head_dist.iter().sum();
        let v_total_heads: usize = v_head_dist.iter().sum();
        let k_total_heads_hi: usize = k_head_dist_hi.iter().sum();
        let k_total_heads_lo: usize = k_head_dist_lo.iter().sum();
        let v_total_heads_hi: usize = v_head_dist_hi.iter().sum();
        let v_total_heads_lo: usize = v_head_dist_lo.iter().sum();
        let timing_reduction = reduction_start.elapsed();

        let palette4_start = Instant::now();
        #[cfg(feature = "cuda")]
        let (k_pal4_bpe, v_pal4_bpe, k_p4_fmts, v_p4_fmts) = if let Some(gpu_agg) =
            gpu_quality_agg.as_ref()
        {
            let k_p4_fmts: Vec<BlockFormat> = gpu_agg
                .k_palette4_formats
                .iter()
                .copied()
                .map(block_format_from_sample)
                .collect();
            let v_p4_fmts: Vec<BlockFormat> = gpu_agg
                .v_palette4_formats
                .iter()
                .copied()
                .map(block_format_from_sample)
                .collect();
            let k_pal4_bpe = palette4_effective_bpe(&k_p4_fmts, blocks_per_head);
            let v_pal4_bpe = palette4_effective_bpe(&v_p4_fmts, blocks_per_head);

            // CPU mirror: same quota-slot algorithm on identical per-block inputs — must agree exactly (≤0.01 bpe).
            let k_pal4_sample: Vec<SampleFormat> =
                k_fmts.iter().copied().map(sample_format_from_block).collect();
            let v_pal4_sample: Vec<SampleFormat> =
                v_fmts.iter().copied().map(sample_format_from_block).collect();
            let (cpu_k_pal4_bpe, _) = cpu_palette4_reduce(&k_pal4_sample, blocks_per_head);
            let (cpu_v_pal4_bpe, _) = cpu_palette4_reduce(&v_pal4_sample, blocks_per_head);
            assert!(
                    (k_pal4_bpe - cpu_k_pal4_bpe).abs() < 0.01,
                    "GPU K palette4 reduction mismatch at level {i}: gpu={k_pal4_bpe:.4} cpu={cpu_k_pal4_bpe:.4}"
                );
            assert!(
                    (v_pal4_bpe - cpu_v_pal4_bpe).abs() < 0.01,
                    "GPU V palette4 reduction mismatch at level {i}: gpu={v_pal4_bpe:.4} cpu={cpu_v_pal4_bpe:.4}"
                );
            (k_pal4_bpe, v_pal4_bpe, k_p4_fmts, v_p4_fmts)
        } else {
            let k_sample: Vec<SampleFormat> =
                k_fmts.iter().copied().map(sample_format_from_block).collect();
            let (k_pal4_bpe, k_p4_sample) = cpu_palette4_reduce(&k_sample, blocks_per_head);
            let k_p4_fmts: Vec<BlockFormat> =
                k_p4_sample.iter().copied().map(block_format_from_sample).collect();
            let v_sample: Vec<SampleFormat> =
                v_fmts.iter().copied().map(sample_format_from_block).collect();
            let (v_pal4_bpe, v_p4_sample) = cpu_palette4_reduce(&v_sample, blocks_per_head);
            let v_p4_fmts: Vec<BlockFormat> =
                v_p4_sample.iter().copied().map(block_format_from_sample).collect();
            (k_pal4_bpe, v_pal4_bpe, k_p4_fmts, v_p4_fmts)
        };

        #[cfg(not(feature = "cuda"))]
        let (k_pal4_bpe, v_pal4_bpe, k_p4_fmts, v_p4_fmts) = {
            let k_sample: Vec<SampleFormat> =
                k_fmts.iter().copied().map(sample_format_from_block).collect();
            let (k_pal4_bpe, k_p4_sample) = cpu_palette4_reduce(&k_sample, blocks_per_head);
            let k_p4_fmts: Vec<BlockFormat> =
                k_p4_sample.iter().copied().map(block_format_from_sample).collect();
            let v_sample: Vec<SampleFormat> =
                v_fmts.iter().copied().map(sample_format_from_block).collect();
            let (v_pal4_bpe, v_p4_sample) = cpu_palette4_reduce(&v_sample, blocks_per_head);
            let v_p4_fmts: Vec<BlockFormat> =
                v_p4_sample.iter().copied().map(block_format_from_sample).collect();
            (k_pal4_bpe, v_pal4_bpe, k_p4_fmts, v_p4_fmts)
        };
        let timing_palette4 = palette4_start.elapsed();

        let mut k_pal4_dist = [0usize; 21];
        let mut v_pal4_dist = [0usize; 21];
        for fmt in &k_p4_fmts { k_pal4_dist[fmt.table_index()] += 1; }
        for fmt in &v_p4_fmts { v_pal4_dist[fmt.table_index()] += 1; }
        let k_total_pal4_blocks = k_p4_fmts.len();
        let v_total_pal4_blocks = v_p4_fmts.len();

        let quality_start = Instant::now();
        let (k_q_w1, k_q_p4) =
            compute_quality_w1_p4(&chunks, blocks_per_chunk, &k_w1_fmts, &k_p4_fmts, true);
        let (v_q_w1, v_q_p4) =
            compute_quality_w1_p4(&chunks, blocks_per_chunk, &v_w1_fmts, &v_p4_fmts, false);
        let timing_quality = quality_start.elapsed();

        let k_raw_bpe = if k_fmts.is_empty() { 16.0 } else { k_fmts.iter().map(|f| f.bits_per_elem() as f64).sum::<f64>() / k_fmts.len() as f64 };
        let v_raw_bpe = if v_fmts.is_empty() { 16.0 } else { v_fmts.iter().map(|f| f.bits_per_elem() as f64).sum::<f64>() / v_fmts.len() as f64 };
        let cr = 16.0 / ((k_raw_bpe + v_raw_bpe) / 2.0);
        let k_bpe_p4 = k_pal4_bpe;
        let v_bpe_p4 = v_pal4_bpe;
        let cr_p4 = 16.0 / ((k_bpe_p4 + v_bpe_p4) / 2.0);
        LevelResult {
            stats,
            _k_bpe: k_raw_bpe,
            _v_bpe: v_raw_bpe,
            cr,
            k_head_dist,
            v_head_dist,
            k_pal4_dist,
            v_pal4_dist,
            k_head_dist_hi,
            k_head_dist_lo,
            v_head_dist_hi,
            v_head_dist_lo,
            k_total_heads,
            v_total_heads,
            k_total_pal4_blocks,
            v_total_pal4_blocks,
            k_total_heads_hi,
            k_total_heads_lo,
            v_total_heads_hi,
            v_total_heads_lo,
            k_bpe_p4,
            v_bpe_p4,
            cr_p4,
            k_q_blk,
            v_q_blk,
            k_q_w1,
            v_q_w1,
            k_q_p4,
            v_q_p4,
            k_snr_hi,
            k_snr_lo,
            v_snr_hi,
            v_snr_lo,
            selection_timing,
            timing_reduction,
            timing_palette4,
            timing_quality,
        }
    });
    let levels_compute_time = levels_compute_start.elapsed();

    // Convenience accessors
    let k = |i: usize| &levels[i].stats[0];
    let v = |i: usize| &levels[i].stats[1];
    let cr = |i: usize| 16.0 / ((k(i).bpe() + v(i).bpe()) / 2.0);
    let ph_cr = |i: usize| levels[i].cr;
    let pal4_cr = |i: usize| levels[i].cr_p4;
    let fmt_ms = |d: Duration| format!("{:>8.1} ms", d.as_secs_f64() * 1_000.0);

    let render_start = Instant::now();
    println!("\n{sep}");
    println!("Candidate-List Compression Curve v2 (per-level K/V candidates, sink-aware)");
    println!("  Data sources: {}", data_sources.join(", "));
    println!(
        "  {} layers  {} chunks  {} K-blocks  {} V-blocks",
        header.num_layers,
        chunks.len(),
        levels[0].stats[0].n,
        levels[0].stats[1].n
    );
    match q_relevance_split {
        Some(split) if has_q_relevance_capture => println!(
            "  Q-relevance split: high >= {:.6} (top {:.0}%), using fixed per-mode H/L thresholds",
            split,
            (1.0 - QREL_HIGH_PERCENTILE) * 100.0,
        ),
        Some(split) => println!(
            "  Q-relevance proxy split (K-energy fallback): high >= {:.6} (top {:.0}%), using fixed per-mode H/L thresholds",
            split,
            (1.0 - QREL_HIGH_PERCENTILE) * 100.0,
        ),
        None => println!("  Q-relevance split: unavailable"),
    }
    println!(
        "  Selection engine for compression tables: {}",
        if gpu_selector.is_some() {
            "CUDA paged kernel via shared sampled_selection module"
        } else {
            "CPU reference fallback"
        }
    );

    let fmt_db = |x: f64| {
        if x.is_nan() {
            format!("{:>6}", "-")
        } else if x.is_infinite() {
            format!("{:>6}", "inf")
        } else {
            format!("{:>6.2}", x)
        }
    };
    let fmt_pct = |counts: &[usize], total: usize, fmt: BlockFormat| {
        if total == 0 {
            format!("{:>5}", "-")
        } else {
            format!(
                "{:>5.1}",
                counts[fmt.table_index()] as f64 * 100.0 / total as f64
            )
        }
    };

    // ── Table 0: Candidate list summary ──
    println!("\n{sep80}");
    println!("Table 0 -- Current configuration of candidates per compression mode");
    println!("  These candidate lists are the allowed K/V formats used to produce the compression tables below.");
    println!();
    for i in 0..11 {
        let k_cands = candidates[i]
            .0
            .iter()
            .map(|f| format!("{f}"))
            .collect::<Vec<_>>()
            .join(", ");
        let v_cands = candidates[i]
            .1
            .iter()
            .map(|f| format!("{f}"))
            .collect::<Vec<_>>()
            .join(", ");
        println!(
            "  C{i}: K=[{k_cands}]  V=[{v_cands}]  blk={:.2}x  head={:.2}x",
            cr(i),
            ph_cr(i),
        );
    }

    println!("\n{sep80}");
    println!("Table 1a -- K full distribution grid");
    println!("  Palette-4 block K format distribution (%) for each compression mode, with fixed H/L thresholds and SNR.");
    println!();
    let k_grid_formats = [
        BlockFormat::F16,
        BlockFormat::Q8_KS,
        BlockFormat::Q8_0,
        BlockFormat::Q4_KS,
        BlockFormat::Q4_1,
        BlockFormat::Q4_0,
        BlockFormat::Q3_1,
        BlockFormat::Q3_0,
        BlockFormat::Q2_1,
        BlockFormat::Q2_A,
        BlockFormat::Q2_S,
        BlockFormat::Q2_0,
        BlockFormat::Q0_M4,
        BlockFormat::Q1_S,
        BlockFormat::Q0_M2,
        BlockFormat::Q0_V,
        BlockFormat::Q1_A,
        BlockFormat::Q0_X,
        BlockFormat::Q0,
    ];
    let mut hk = "  Mode | H_SNR | L_SNR |   CR  |".to_string();
    for fmt in &k_grid_formats {
        hk.push_str(&format!(" {:>5}", fmt.grid_label()));
    }
    println!("{hk}");
    println!("  {}", "-".repeat(hk.len() - 2));
    for i in 0..11 {
        let mut line = format!(
            "  C{i:<2} | {} | {} | {:>5.2}x |",
            fmt_db(levels[i].k_snr_hi),
            fmt_db(levels[i].k_snr_lo),
            ph_cr(i),
        );
        for fmt in &k_grid_formats {
            line.push_str(&format!(
                " {}",
                fmt_pct(&levels[i].k_pal4_dist, levels[i].k_total_pal4_blocks, *fmt)
            ));
        }
        println!("{line}");
    }

    println!("\n{sep80}");
    println!("Table 1b -- V full distribution grid");
    println!("  Palette-4 block V format distribution (%) for each compression mode, with fixed H/L thresholds and SNR.");
    println!();
    let v_grid_formats = [
        BlockFormat::F16,
        BlockFormat::Q8_0,
        BlockFormat::Q4_1,
        BlockFormat::Q4_0,
        BlockFormat::Q3_1,
        BlockFormat::Q3_0,
        BlockFormat::Q2_1,
        BlockFormat::Q2_A,
        BlockFormat::Q2_S,
        BlockFormat::Q2_0,
        BlockFormat::Q0_M4,
        BlockFormat::Q1_S,
        BlockFormat::Q0_M2,
        BlockFormat::Q0_V,
        BlockFormat::Q1_A,
        BlockFormat::Q0_X,
        BlockFormat::Q0,
    ];
    let mut hv = "  Mode | H_SNR | L_SNR |   CR  |".to_string();
    for fmt in &v_grid_formats {
        hv.push_str(&format!(" {:>5}", fmt.grid_label()));
    }
    println!("{hv}");
    println!("  {}", "-".repeat(hv.len() - 2));
    for i in 0..11 {
        let mut line = format!(
            "  C{i:<2} | {} | {} | {:>5.2}x |",
            fmt_db(levels[i].v_snr_hi),
            fmt_db(levels[i].v_snr_lo),
            ph_cr(i),
        );
        for fmt in &v_grid_formats {
            line.push_str(&format!(
                " {}",
                fmt_pct(&levels[i].v_pal4_dist, levels[i].v_total_pal4_blocks, *fmt)
            ));
        }
        println!("{line}");
    }

    // ── Table 2b: Palette-4 vs worst-case reduction comparison ──
    println!("\n{sep80}");
    println!("Table 2b -- Palette-4 vs worst-case per-head reduction");
    println!(
        "  Palette-4: best 4 formats per head + 2-bit/block offset table (36 bytes overhead/head)"
    );
    println!();
    println!("  Mode  Tier      blk_CR  worst1_CR  pal4_CR   K_bpe_w1  K_bpe_p4  V_bpe_w1  V_bpe_p4  gain");
    println!("  {}", "-".repeat(100));
    for i in 0..11 {
        let gain = levels[i].cr_p4 / levels[i].cr;
        println!(
            "  C{i}  {:<9} {:>5.2}x   {:>5.2}x     {:>5.2}x   {:>8.3}  {:>8.3}  {:>8.3}  {:>8.3}  {:>5.1}%",
            LEVEL_TIER[i],
            cr(i),
            ph_cr(i),
            levels[i].cr_p4,
            levels[i]._k_bpe,
            levels[i].k_bpe_p4,
            levels[i]._v_bpe,
            levels[i].v_bpe_p4,
            (gain - 1.0) * 100.0,
        );
    }

    // ── Table 2c: Quality impact of reduction strategies ──
    println!("\n{sep80}");
    println!("Table 2c -- Quality impact: block-level vs worst-case vs palette-4");
    println!("  SNR (dB): higher = better quality.  cos_p95: 95th percentile cosine distance (lower = better)");
    println!();
    println!("  Mode  Tier      --- K SNR (dB) ---      --- V SNR (dB) ---      --- K cos_p95 ---        --- V cos_p95 ---");
    println!("                  blk    worst1  pal4     blk    worst1  pal4     blk      worst1   pal4    blk      worst1   pal4");
    println!("  {}", "-".repeat(130));
    for i in 0..11 {
        let lv = &levels[i];
        println!(
            "  C{i}  {:<9} {:>6.1}  {:>6.1}  {:>6.1}   {:>6.1}  {:>6.1}  {:>6.1}   {:>.2e}  {:>.2e}  {:>.2e}  {:>.2e}  {:>.2e}  {:>.2e}",
            LEVEL_TIER[i],
            lv.k_q_blk.0, lv.k_q_w1.0, lv.k_q_p4.0,
            lv.v_q_blk.0, lv.v_q_w1.0, lv.v_q_p4.0,
            lv.k_q_blk.2, lv.k_q_w1.2, lv.k_q_p4.2,
            lv.v_q_blk.2, lv.v_q_w1.2, lv.v_q_p4.2,
        );
    }
    println!();
    println!("  SNR delta (worst1 - pal4, dB): positive = quality LOST by switching to palette-4");
    println!("  Mode  Tier      K_delta  V_delta");
    println!("  {}", "-".repeat(40));
    for i in 0..11 {
        let lv = &levels[i];
        let k_delta = lv.k_q_w1.0 - lv.k_q_p4.0;
        let v_delta = lv.v_q_w1.0 - lv.v_q_p4.0;
        println!(
            "  C{i}  {:<9} {:>+6.2}   {:>+6.2}",
            LEVEL_TIER[i], k_delta, v_delta,
        );
    }

    // ── Monotonicity check (on palette-4 CR, which is what production uses) ──
    println!("\n{sep80}");
    println!("Monotonicity check (palette-4 CR):");
    let mut monotonic = true;
    for i in 1..11 {
        let c_prev = pal4_cr(i - 1);
        let c_curr = pal4_cr(i);
        let status = if c_curr > c_prev { "OK" } else { "BAND" };
        if c_curr <= c_prev {
            monotonic = false;
        }
        println!(
            "  C{} → C{}: {:.3}x → {:.3}x  step={:+.3}x  {status}",
            i - 1,
            i,
            c_prev,
            c_curr,
            c_curr - c_prev
        );
    }
    println!(
        "  Result: {}",
        if monotonic {
            "ALL DISTINCT ✓"
        } else {
            "BANDING DETECTED ✗"
        }
    );
    println!("{sep}");

    let report_render_time = render_start.elapsed();
    let qrel_mask_sum = levels.iter().fold(Duration::default(), |acc, lv| {
        acc + lv.selection_timing.qrel_mask
    });
    let block_select_sum = levels.iter().fold(Duration::default(), |acc, lv| {
        acc + lv.selection_timing.block_select
    });
    let stats_fold_sum = levels.iter().fold(Duration::default(), |acc, lv| {
        acc + lv.selection_timing.stats_fold
    });
    let head_reduce_sum = levels
        .iter()
        .fold(Duration::default(), |acc, lv| acc + lv.timing_reduction);
    let palette4_sum = levels
        .iter()
        .fold(Duration::default(), |acc, lv| acc + lv.timing_palette4);
    let quality_sum = levels
        .iter()
        .fold(Duration::default(), |acc, lv| acc + lv.timing_quality);

    println!("\n{sep80}");
    println!("Timing summary -- report build");
    println!("  load dump                       {}", fmt_ms(load_time));
    println!(
        "  F16 arena rounding              {}",
        fmt_ms(f16_round_time)
    );
    println!(
        "  q-relevance split               {}",
        fmt_ms(qrel_split_time)
    );
    println!(
        "  GPU selector init/upload        {}",
        fmt_ms(gpu_selector_init_time)
    );
    println!(
        "  per-level compute (wall)        {}",
        fmt_ms(levels_compute_time)
    );
    println!(
        "    qrel mask build (sum)         {}",
        fmt_ms(qrel_mask_sum)
    );
    println!(
        "    block selection (sum)         {}",
        fmt_ms(block_select_sum)
    );
    println!(
        "    stats fold (sum)              {}",
        fmt_ms(stats_fold_sum)
    );
    println!(
        "    per-head reductions (sum)     {}",
        fmt_ms(head_reduce_sum)
    );
    println!("    palette-4 reductions (sum)    {}", fmt_ms(palette4_sum));
    println!("    quality eval (sum)            {}", fmt_ms(quality_sum));
    println!(
        "  table rendering/output (wall)   {}",
        fmt_ms(report_render_time)
    );
    println!(
        "  report section total            {}",
        fmt_ms(total_start.elapsed())
    );

    // ── GPU-accelerated threshold calibration sweep ──
    // For each level, find the magnitude-weighted K threshold that produces the
    // same per-head K BPE as the old cosine-based K threshold.  This keeps overall
    // CR the same while shifting format selections to protect high-magnitude dims.
    //
    // Also find more aggressive V thresholds targeting 2× current V BPE reduction
    // (i.e. V contributes more compression).
    //
    // Uses the GPU selection kernel for the binary search trials instead of the
    // slow CPU collect_block_formats + reduce_per_head loop.
    println!("\n{sep}");
    #[cfg(feature = "cuda")]
    {
        let calibration_start = Instant::now();
        println!("Threshold calibration (GPU-accelerated)");
        println!("  K: find mag-weighted threshold → same per-head K BPE as cosine baseline");
        println!("  V: find cosine threshold → 1.25× current V BPE reduction (more aggressive)");
        let calibration_chunks = evenly_sampled_chunks(&chunks, CALIBRATION_SAMPLE_CHUNKS);
        println!(
            "  Calibration dataset: {} / {} chunks ({:.1}%) evenly sampled from the dump",
            calibration_chunks.len(),
            chunks.len(),
            calibration_chunks.len() as f64 * 100.0 / chunks.len() as f64,
        );
        println!();

        let (_calibration_generation, gpu) = if !calibration_chunks.is_empty() {
            let k_chunks: Vec<&[f32]> = calibration_chunks
                .iter()
                .map(|chunk| chunk.k.as_slice())
                .collect();
            let v_chunks: Vec<&[f32]> = calibration_chunks
                .iter()
                .map(|chunk| chunk.v.as_slice())
                .collect();
            match candle::Device::cuda_if_available(0) {
                Ok(candle::Device::Cuda(cuda_dev)) => {
                    let stager = candle::quantized::pinned_staging::PinnedStager::new(&cuda_dev);
                    let generation = stager.begin_generation();
                    let gpu = PagedSelectionGpuInputs::from_f32_chunks(
                        &k_chunks,
                        &v_chunks,
                        blocks_per_chunk,
                        header.n_kv_head,
                        DEFAULT_CALIBRATION_ARENA_CHUNKS,
                        Some(&generation),
                        &cuda_dev,
                    )
                    .ok();
                    (Some(generation), gpu)
                }
                _ => (None, None),
            }
        } else {
            (None, None)
        };

        let gpu = if let Some(gpu) = gpu.as_ref() {
            gpu
        } else {
            println!("  SKIP GPU calibration: no CUDA device — falling back to CPU");
            // CPU fallback for the calibration section
            struct Baseline {
                k_bpe: f64,
                v_bpe: f64,
            }
            let baselines: Vec<Baseline> = (0..11)
                .map(|i| {
                    let k_fmts_cosine: Vec<BlockFormat> = {
                        let mut fmts = Vec::new();
                        for chunk in &calibration_chunks {
                            let t = K_MAGWEIGHT_THRESHOLDS[i];
                            for b in 0..blocks_per_chunk {
                                let start = b * SELECT_BLOCK;
                                let end = start + SELECT_BLOCK;
                                if end <= chunk.k.len() {
                                    let blk: [f32; SELECT_BLOCK] =
                                        chunk.k[start..end].try_into().unwrap();
                                    fmts.push(select_format_from_candidates(
                                        &blk,
                                        &candidates[i].0,
                                        t,
                                    ));
                                }
                            }
                        }
                        fmts
                    };
                    let k_bpe = reduce_per_head(&k_fmts_cosine, blocks_per_chunk, blocks_per_head);
                    let v_fmts = collect_block_formats(
                        &calibration_chunks,
                        blocks_per_chunk,
                        v_auto_thresholds[i],
                        &candidates[i].1,
                        false,
                    );
                    let v_bpe = reduce_per_head(&v_fmts, blocks_per_chunk, blocks_per_head);
                    Baseline { k_bpe, v_bpe }
                })
                .collect();

            let find_k_threshold = |level: usize, target_k_bpe: f64| -> f32 {
                let mut lo = 1e-9_f32;
                let mut hi = 10.0_f32;
                for _ in 0..60 {
                    let mid = (lo + hi) / 2.0;
                    let k_fmts = collect_block_formats(
                        &calibration_chunks,
                        blocks_per_chunk,
                        mid,
                        &candidates[level].0,
                        true,
                    );
                    let k_bpe = reduce_per_head(&k_fmts, blocks_per_chunk, blocks_per_head);
                    if k_bpe > target_k_bpe {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                (lo + hi) / 2.0
            };
            let find_v_threshold = |level: usize, target_v_bpe: f64| -> f32 {
                let mut lo = 1e-7_f32;
                let mut hi = 10.0_f32;
                for _ in 0..60 {
                    let mid = (lo + hi) / 2.0;
                    let v_fmts = collect_block_formats(
                        &calibration_chunks,
                        blocks_per_chunk,
                        mid,
                        &candidates[level].1,
                        false,
                    );
                    let v_bpe = reduce_per_head(&v_fmts, blocks_per_chunk, blocks_per_head);
                    if v_bpe > target_v_bpe {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                }
                (lo + hi) / 2.0
            };

            println!(
                "  {:>4}  {:>9}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
                "Mode",
                "Tier",
                "K_cos_thr",
                "K_cos_bpe",
                "K_mag_thr",
                "V_cos_thr",
                "V_cos_bpe",
                "V_aggr_thr"
            );
            println!("  {}", "-".repeat(90));
            let mut new_k_thresholds = [0.0f32; 11];
            let mut new_v_thresholds = [0.0f32; 11];
            for i in 0..11 {
                let k_target_bpe = baselines[i].k_bpe;
                let v_target_bpe = baselines[i].v_bpe * 0.80;
                let k_mag_thr = find_k_threshold(i, k_target_bpe);
                let v_aggr_thr = find_v_threshold(i, v_target_bpe);
                new_k_thresholds[i] = k_mag_thr;
                new_v_thresholds[i] = v_aggr_thr;
                println!(
                    "  C{i}    {:<9} {:>10.6}  {:>10.3}  {:>10.6}  {:>10.6}  {:>10.3}  {:>10.6}",
                    LEVEL_TIER[i],
                    K_MAGWEIGHT_THRESHOLDS[i],
                    k_target_bpe,
                    k_mag_thr,
                    v_auto_thresholds[i],
                    baselines[i].v_bpe,
                    v_aggr_thr,
                );
            }
            println!("\n  Proposed K thresholds (magnitude-weighted):");
            for i in 0..11 {
                println!(
                    "    {:.6}, // C{i} {}",
                    new_k_thresholds[i],
                    LEVEL_TIER[i].trim()
                );
            }
            println!("\n  Proposed V thresholds (cosine, 20% more aggressive):");
            for i in 0..11 {
                println!(
                    "    {:.6}, // C{i} {}",
                    new_v_thresholds[i],
                    LEVEL_TIER[i].trim()
                );
            }
            println!("{sep}");
            return;
        };

        let run_gpu_bpe = |level: usize, k_thr: f32, v_thr: f32| -> (f64, f64) {
            let k_sample_candidates: Vec<SampleFormat> = candidates[level]
                .0
                .iter()
                .copied()
                .map(sample_format_from_block)
                .collect();
            let v_sample_candidates: Vec<SampleFormat> = candidates[level]
                .1
                .iter()
                .copied()
                .map(sample_format_from_block)
                .collect();
            let (k_selected, v_selected) = gpu
                .select_block_formats(
                    &k_sample_candidates,
                    &v_sample_candidates,
                    k_thr,
                    k_thr,
                    v_thr,
                    v_thr,
                )
                .expect("GPU calibration select");
            let k_block_fmts: Vec<BlockFormat> = k_selected
                .into_iter()
                .map(block_format_from_sample)
                .collect();
            let v_block_fmts: Vec<BlockFormat> = v_selected
                .into_iter()
                .map(block_format_from_sample)
                .collect();
            (
                reduce_per_head(&k_block_fmts, blocks_per_chunk, blocks_per_head),
                reduce_per_head(&v_block_fmts, blocks_per_chunk, blocks_per_head),
            )
        };

        // Build a tiny 2-chunk GPU object for the diagnostic comparison only.
        let diag_chunks: Vec<ChunkData> = calibration_chunks
            .iter()
            .take(2)
            .cloned()
            .collect();
        let (_diag_generation, diag_gpu) = if !diag_chunks.is_empty() {
            let k_slices: Vec<&[f32]> = diag_chunks.iter().map(|c| c.k.as_slice()).collect();
            let v_slices: Vec<&[f32]> = diag_chunks.iter().map(|c| c.v.as_slice()).collect();
            match candle::Device::cuda_if_available(0) {
                Ok(candle::Device::Cuda(cuda_dev)) => {
                    let stager = candle::quantized::pinned_staging::PinnedStager::new(&cuda_dev);
                    let generation = stager.begin_generation();
                    let g = PagedSelectionGpuInputs::from_f32_chunks(
                        &k_slices, &v_slices,
                        blocks_per_chunk, header.n_kv_head,
                        DEFAULT_CALIBRATION_ARENA_CHUNKS, Some(&generation), &cuda_dev,
                    ).ok();
                    (Some(generation), g)
                }
                _ => (None, None),
            }
        } else {
            (None, None)
        };

        let diag_start = Instant::now();
        println!("  GPU diagnostic check (3 levels, {} chunks)...", diag_chunks.len());
        let diag_levels = [0, 5, 9];
        for &level in &diag_levels {
            let (ref k_cands, ref v_cands) = candidates[level];
            let k_fmts = collect_block_formats(
                &diag_chunks,
                blocks_per_chunk,
                K_MAGWEIGHT_THRESHOLDS[level],
                k_cands,
                true,
            );
            let v_fmts = collect_block_formats(
                &diag_chunks,
                blocks_per_chunk,
                v_auto_thresholds[level],
                v_cands,
                false,
            );
            let cpu_k_bpe = reduce_per_head(&k_fmts, blocks_per_chunk, blocks_per_head);
            let cpu_v_bpe = reduce_per_head(&v_fmts, blocks_per_chunk, blocks_per_head);
            let (gpu_k_bpe, gpu_v_bpe) = if let Some(dg) = diag_gpu.as_ref() {
                let k_sc: Vec<SampleFormat> = k_cands.iter().copied().map(sample_format_from_block).collect();
                let v_sc: Vec<SampleFormat> = v_cands.iter().copied().map(sample_format_from_block).collect();
                let (ks, vs) = dg.select_block_formats(&k_sc, &v_sc,
                    K_MAGWEIGHT_THRESHOLDS[level], K_MAGWEIGHT_THRESHOLDS[level],
                    v_auto_thresholds[level], v_auto_thresholds[level])
                    .expect("diag GPU select");
                let kbf: Vec<BlockFormat> = ks.into_iter().map(block_format_from_sample).collect();
                let vbf: Vec<BlockFormat> = vs.into_iter().map(block_format_from_sample).collect();
                (
                    reduce_per_head(&kbf, blocks_per_chunk, blocks_per_head),
                    reduce_per_head(&vbf, blocks_per_chunk, blocks_per_head),
                )
            } else {
                (cpu_k_bpe, cpu_v_bpe)
            };

            let k_bpe_err = (gpu_k_bpe - cpu_k_bpe).abs();
            let v_bpe_err = (gpu_v_bpe - cpu_v_bpe).abs();
            let k_ok = k_bpe_err < 0.1;
            let v_ok = v_bpe_err < 0.1;
            println!(
                "    C{level}: K_bpe cpu={cpu_k_bpe:.3} gpu={gpu_k_bpe:.3} err={k_bpe_err:.4} {}  V_bpe cpu={cpu_v_bpe:.3} gpu={gpu_v_bpe:.3} err={v_bpe_err:.4} {}",
                if k_ok { "OK" } else { "FAIL" },
                if v_ok { "OK" } else { "FAIL" },
            );
            assert!(
                k_ok && v_ok,
                "Shared CUDA diagnostic FAILED at C{level}: K_err={k_bpe_err:.4} V_err={v_bpe_err:.4}. \
             CUDA report path does not match CPU reference."
            );
        }
        let diag_time = diag_start.elapsed();
        println!("  GPU diagnostic: PASS ✓");
        println!();

        // First, compute baseline K and V per-head BPE using OLD cosine thresholds
        struct Baseline {
            k_bpe: f64,
            v_bpe: f64,
        }
        let baselines_start = Instant::now();
        let baselines: Vec<Baseline> = (0..11)
            .map(|i| {
                // K baseline: use cosine distance (temporarily swap metric) — must use CPU
                // since GPU always uses magnitude-weighted for K
                let k_fmts_cosine: Vec<BlockFormat> = {
                    let mut fmts = Vec::new();
                    for chunk in &calibration_chunks {
                        let t = K_MAGWEIGHT_THRESHOLDS[i];
                        for b in 0..blocks_per_chunk {
                            let start = b * SELECT_BLOCK;
                            let end = start + SELECT_BLOCK;
                            if end <= chunk.k.len() {
                                let blk: [f32; SELECT_BLOCK] =
                                    chunk.k[start..end].try_into().unwrap();
                                fmts.push(select_format_from_candidates(&blk, &candidates[i].0, t));
                            }
                        }
                    }
                    fmts
                };
                let k_bpe = reduce_per_head(&k_fmts_cosine, blocks_per_chunk, blocks_per_head);

                // V baseline via the shared CUDA report path
                let v_bpe = run_gpu_bpe(i, K_MAGWEIGHT_THRESHOLDS[i], v_auto_thresholds[i]).1;
                Baseline { k_bpe, v_bpe }
            })
            .collect();
        let baselines_time = baselines_start.elapsed();

        // GPU-accelerated binary search for K threshold (magnitude-weighted)
        let find_k_threshold_gpu = |level: usize, target_k_bpe: f64| -> f32 {
            let mut lo = 1e-9_f32;
            let mut hi = 10.0_f32;
            for _ in 0..60 {
                let mid = (lo + hi) / 2.0;
                let k_bpe = run_gpu_bpe(level, mid, v_auto_thresholds[level]).0;
                if k_bpe > target_k_bpe {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            (lo + hi) / 2.0
        };

        // GPU-accelerated binary search for V threshold (cosine)
        let find_v_threshold_gpu = |level: usize, target_v_bpe: f64| -> f32 {
            let mut lo = 1e-7_f32;
            let mut hi = 10.0_f32;
            for _ in 0..60 {
                let mid = (lo + hi) / 2.0;
                let v_bpe = run_gpu_bpe(level, K_MAGWEIGHT_THRESHOLDS[level], mid).1;
                if v_bpe > target_v_bpe {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            (lo + hi) / 2.0
        };

        println!(
            "  {:>4}  {:>9}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
            "Mode",
            "Tier",
            "K_cos_thr",
            "K_cos_bpe",
            "K_mag_thr",
            "V_cos_thr",
            "V_cos_bpe",
            "V_aggr_thr"
        );
        println!("  {}", "-".repeat(90));
        let mut new_k_thresholds = [0.0f32; 11];
        let mut new_v_thresholds = [0.0f32; 11];
        let mut k_search_time = Duration::default();
        let mut v_search_time = Duration::default();
        for i in 0..11 {
            let k_target_bpe = baselines[i].k_bpe;
            let v_target_bpe = baselines[i].v_bpe * 0.80; // 20% lower BPE = more aggressive
            let k_search_start = Instant::now();
            let k_mag_thr = find_k_threshold_gpu(i, k_target_bpe);
            k_search_time += k_search_start.elapsed();
            let v_search_start = Instant::now();
            let v_aggr_thr = find_v_threshold_gpu(i, v_target_bpe);
            v_search_time += v_search_start.elapsed();
            new_k_thresholds[i] = k_mag_thr;
            new_v_thresholds[i] = v_aggr_thr;
            println!(
                "  C{i}    {:<9} {:>10.6}  {:>10.3}  {:>10.6}  {:>10.6}  {:>10.3}  {:>10.6}",
                LEVEL_TIER[i],
                K_MAGWEIGHT_THRESHOLDS[i],
                k_target_bpe,
                k_mag_thr,
                v_auto_thresholds[i],
                baselines[i].v_bpe,
                v_aggr_thr,
            );
        }

        // Print as Rust constants for copy-paste
        println!("\n  Proposed K thresholds (magnitude-weighted):");
        for i in 0..11 {
            println!(
                "    {:.6}, // C{i} {}",
                new_k_thresholds[i],
                LEVEL_TIER[i].trim()
            );
        }
        println!("\n  Proposed V thresholds (cosine, 20% more aggressive):");
        for i in 0..11 {
            println!(
                "    {:.6}, // C{i} {}",
                new_v_thresholds[i],
                LEVEL_TIER[i].trim()
            );
        }

        let calibration_time = calibration_start.elapsed();
        println!();
        println!("  Calibration timing:");
        println!("    diagnostic check             {}", fmt_ms(diag_time));
        println!(
            "    baseline BPE pass            {}",
            fmt_ms(baselines_time)
        );
        println!("    K threshold search (sum)     {}", fmt_ms(k_search_time));
        println!("    V threshold search (sum)     {}", fmt_ms(v_search_time));
        println!(
            "    calibration wall             {}",
            fmt_ms(calibration_time)
        );
        println!(
            "    total test runtime           {}",
            fmt_ms(total_start.elapsed())
        );
        println!("{sep}");
    } // end #[cfg(feature = "cuda")]

    #[cfg(not(feature = "cuda"))]
    {
        println!("Threshold calibration SKIPPED (requires --features cuda)");
        println!("{sep}");
    }
}

// ============================================================================
// Reduction granularity comparison: per-chunk vs per-chunk-head
// ============================================================================

/// Given per-block format selections, apply worst-case reduction at chunk
/// granularity (512 blocks → 1 format) and return the effective BPE.
fn reduce_per_chunk(block_fmts: &[BlockFormat], blocks_per_chunk: usize) -> f64 {
    let mut total_bpe = 0.0f64;
    let mut total_blocks = 0usize;
    for chunk_fmts in block_fmts.chunks(blocks_per_chunk) {
        // Worst-case: highest table_index = most conservative
        let worst = chunk_fmts
            .iter()
            .max_by_key(|f| f.table_index())
            .copied()
            .unwrap_or(BlockFormat::F16);
        total_bpe += worst.bits_per_elem() as f64 * chunk_fmts.len() as f64;
        total_blocks += chunk_fmts.len();
    }
    if total_blocks > 0 {
        total_bpe / total_blocks as f64
    } else {
        16.0
    }
}

/// Given per-block format selections, apply worst-case reduction at
/// per-head granularity (128 blocks → 1 format per head) and return
/// the effective BPE.
fn reduce_per_head(
    block_fmts: &[BlockFormat],
    blocks_per_chunk: usize,
    blocks_per_head: usize,
) -> f64 {
    reduce_per_head_with_dist(block_fmts, blocks_per_chunk, blocks_per_head).0
}



/// Apply worst-case-per-head reduction and return the effective format for each block.
fn apply_worst_case_reduction(
    block_fmts: &[BlockFormat],
    blocks_per_chunk: usize,
    blocks_per_head: usize,
) -> Vec<BlockFormat> {
    let mut out = Vec::with_capacity(block_fmts.len());
    for chunk_fmts in block_fmts.chunks(blocks_per_chunk) {
        for head_fmts in chunk_fmts.chunks(blocks_per_head) {
            let worst = head_fmts
                .iter()
                .max_by_key(|f| f.table_index())
                .copied()
                .unwrap_or(BlockFormat::F16);
            out.extend(std::iter::repeat(worst).take(head_fmts.len()));
        }
    }
    out
}


/// Compute quality metrics (SNR, NRMSE, cos_p95) for a given set of blocks
/// using the specified effective formats (post-reduction).
#[allow(dead_code)]
fn compute_quality_metrics(
    chunks: &[ChunkData],
    blocks_per_chunk: usize,
    effective_fmts: &[BlockFormat],
    is_key: bool,
) -> (f64, f64, f64) {
    // SNR, NRMSE, cos_p95
    let mut sig = 0.0f64;
    let mut noise = 0.0f64;
    let mut cos_dists: Vec<f32> = Vec::new();
    let mut blk_idx = 0usize;
    for chunk in chunks {
        let data = if is_key { &chunk.k } else { &chunk.v };
        for b in 0..blocks_per_chunk {
            let start = b * SELECT_BLOCK;
            let end = start + SELECT_BLOCK;
            if end <= data.len() && blk_idx < effective_fmts.len() {
                let blk: [f32; SELECT_BLOCK] = data[start..end].try_into().unwrap();
                let fmt = effective_fmts[blk_idx];
                let recon = fmt.apply_quant(&blk);
                let cd = cosine_distance(&blk, &recon);
                cos_dists.push(cd);
                for (&x, &xh) in blk.iter().zip(recon.iter()) {
                    sig += (x as f64) * (x as f64);
                    noise += ((x - xh) as f64) * ((x - xh) as f64);
                }
            }
            blk_idx += 1;
        }
    }
    let snr_db = if noise > 1e-30 {
        10.0 * (sig / noise).log10()
    } else {
        f64::INFINITY
    };
    let nrmse = if sig > 1e-30 {
        (noise / sig).sqrt()
    } else {
        0.0
    };
    cos_dists.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let cos_p95 = if cos_dists.len() > 0 {
        let idx = ((cos_dists.len() as f64) * 0.95) as usize;
        cos_dists[idx.min(cos_dists.len() - 1)] as f64
    } else {
        0.0
    };
    (snr_db, nrmse, cos_p95)
}

/// Compute quality metrics for BOTH worst-case and palette-4 reduction in a single pass.
/// Returns ((w1_snr, w1_nrmse, w1_cos_p95), (p4_snr, p4_nrmse, p4_cos_p95)).
/// This avoids iterating all blocks twice (once for w1, once for p4).
fn compute_quality_w1_p4(
    chunks: &[ChunkData],
    blocks_per_chunk: usize,
    w1_fmts: &[BlockFormat],
    p4_fmts: &[BlockFormat],
    is_key: bool,
) -> ((f64, f64, f64), (f64, f64, f64)) {
    let mut w1_sig = 0.0f64;
    let mut w1_noise = 0.0f64;
    let mut w1_cos: Vec<f32> = Vec::new();
    let mut p4_sig = 0.0f64;
    let mut p4_noise = 0.0f64;
    let mut p4_cos: Vec<f32> = Vec::new();
    let mut blk_idx = 0usize;
    for chunk in chunks {
        let data = if is_key { &chunk.k } else { &chunk.v };
        for b in 0..blocks_per_chunk {
            let start = b * SELECT_BLOCK;
            let end = start + SELECT_BLOCK;
            if end <= data.len() && blk_idx < w1_fmts.len() {
                let blk: [f32; SELECT_BLOCK] = data[start..end].try_into().unwrap();
                // Signal power (same for both)
                let mut blk_sig = 0.0f64;
                for &x in blk.iter() {
                    blk_sig += (x as f64) * (x as f64);
                }
                w1_sig += blk_sig;
                p4_sig += blk_sig;

                let w1_fmt = w1_fmts[blk_idx];
                let p4_fmt = p4_fmts[blk_idx];

                // If both reductions give the same format, compute quant once
                if w1_fmt == p4_fmt {
                    let recon = w1_fmt.apply_quant(&blk);
                    let cd = cosine_distance(&blk, &recon);
                    w1_cos.push(cd);
                    p4_cos.push(cd);
                    let mut n = 0.0f64;
                    for (&x, &xh) in blk.iter().zip(recon.iter()) {
                        n += ((x - xh) as f64) * ((x - xh) as f64);
                    }
                    w1_noise += n;
                    p4_noise += n;
                } else {
                    // Different formats — compute both
                    let w1_recon = w1_fmt.apply_quant(&blk);
                    let w1_cd = cosine_distance(&blk, &w1_recon);
                    w1_cos.push(w1_cd);
                    for (&x, &xh) in blk.iter().zip(w1_recon.iter()) {
                        w1_noise += ((x - xh) as f64) * ((x - xh) as f64);
                    }
                    let p4_recon = p4_fmt.apply_quant(&blk);
                    let p4_cd = cosine_distance(&blk, &p4_recon);
                    p4_cos.push(p4_cd);
                    for (&x, &xh) in blk.iter().zip(p4_recon.iter()) {
                        p4_noise += ((x - xh) as f64) * ((x - xh) as f64);
                    }
                }
            }
            blk_idx += 1;
        }
    }
    let make_result = |sig: f64, noise: f64, mut cos: Vec<f32>| -> (f64, f64, f64) {
        let snr = if noise > 1e-30 {
            10.0 * (sig / noise).log10()
        } else {
            f64::INFINITY
        };
        let nrmse = if sig > 1e-30 {
            (noise / sig).sqrt()
        } else {
            0.0
        };
        cos.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95 = if !cos.is_empty() {
            let idx = ((cos.len() as f64) * 0.95) as usize;
            cos[idx.min(cos.len() - 1)] as f64
        } else {
            0.0
        };
        (snr, nrmse, p95)
    };
    (
        make_result(w1_sig, w1_noise, w1_cos),
        make_result(p4_sig, p4_noise, p4_cos),
    )
}

/// Same as `reduce_per_head` but also returns format distribution counts
/// (number of HEADS assigned each format, indexed by `table_index()`).
fn reduce_per_head_with_dist(
    block_fmts: &[BlockFormat],
    blocks_per_chunk: usize,
    blocks_per_head: usize,
) -> (f64, [usize; 16]) {
    let mut total_bpe = 0.0f64;
    let mut total_blocks = 0usize;
    let mut head_counts = [0usize; 16]; // indexed by table_index (0..16)
    for chunk_fmts in block_fmts.chunks(blocks_per_chunk) {
        for head_fmts in chunk_fmts.chunks(blocks_per_head) {
            let worst = head_fmts
                .iter()
                .max_by_key(|f| f.table_index())
                .copied()
                .unwrap_or(BlockFormat::F16);
            total_bpe += worst.bits_per_elem() as f64 * head_fmts.len() as f64;
            total_blocks += head_fmts.len();
            head_counts[worst.table_index()] += 1;
        }
    }
    let bpe = if total_blocks > 0 {
        total_bpe / total_blocks as f64
    } else {
        16.0
    };
    (bpe, head_counts)
}

#[cfg(feature = "cuda")]
fn reduce_head_tags_with_dist(
    head_tags: &[BlockFormat],
    blocks_per_head: usize,
) -> (f64, [usize; 16]) {
    let mut total_bpe = 0.0f64;
    let mut head_counts = [0usize; 16];
    for &fmt in head_tags {
        total_bpe += fmt.bits_per_elem() as f64 * blocks_per_head as f64;
        head_counts[fmt.table_index()] += 1;
    }
    let total_blocks = head_tags.len() * blocks_per_head;
    let bpe = if total_blocks > 0 {
        total_bpe / total_blocks as f64
    } else {
        16.0
    };
    (bpe, head_counts)
}

#[cfg(feature = "cuda")]
fn palette4_effective_bpe(block_fmts: &[BlockFormat], blocks_per_head: usize) -> f64 {
    let elems_per_block = SELECT_BLOCK as f64;
    let palette_overhead_bits = (blocks_per_head * 2 + 4 * 8) as f64;
    let mut total_bits = 0.0f64;
    let mut total_elems = 0.0f64;
    for head_fmts in block_fmts.chunks(blocks_per_head) {
        for &fmt in head_fmts {
            total_bits += fmt.bits_per_elem() as f64 * elems_per_block;
            total_elems += elems_per_block;
        }
        total_bits += palette_overhead_bits;
    }
    if total_elems > 0.0 {
        total_bits / total_elems
    } else {
        16.0
    }
}

/// Collect per-block format selections for one side (K or V) across all chunks.
fn reduce_per_head_bucket_with_dist(
    block_fmts: &[BlockFormat],
    bucket_mask: &[bool],
    blocks_per_chunk: usize,
    blocks_per_head: usize,
) -> [usize; 16] {
    let mut head_counts = [0usize; 16];
    for (chunk_fmts, chunk_mask) in block_fmts
        .chunks(blocks_per_chunk)
        .zip(bucket_mask.chunks(blocks_per_chunk))
    {
        for (head_fmts, head_mask) in chunk_fmts
            .chunks(blocks_per_head)
            .zip(chunk_mask.chunks(blocks_per_head))
        {
            let mut worst: Option<BlockFormat> = None;
            for (&fmt, &in_bucket) in head_fmts.iter().zip(head_mask.iter()) {
                if in_bucket {
                    worst = Some(match worst {
                        Some(cur) if cur.table_index() >= fmt.table_index() => cur,
                        _ => fmt,
                    });
                }
            }
            if let Some(worst_fmt) = worst {
                head_counts[worst_fmt.table_index()] += 1;
            }
        }
    }
    head_counts
}

fn collect_block_formats(
    chunks: &[ChunkData],
    blocks_per_chunk: usize,
    threshold: f32,
    candidates: &[BlockFormat],
    is_key: bool,
) -> Vec<BlockFormat> {
    let mut fmts = Vec::with_capacity(chunks.len() * blocks_per_chunk);
    for chunk in chunks {
        let data = if is_key { &chunk.k } else { &chunk.v };
        let q_data = chunk.q.as_ref();
        let t = threshold;
        for b in 0..blocks_per_chunk {
            let start = b * SELECT_BLOCK;
            let end = start + SELECT_BLOCK;
            if end <= data.len() {
                let blk_raw: [f32; SELECT_BLOCK] = data[start..end].try_into().unwrap();
                let blk: [f32; SELECT_BLOCK] =
                    std::array::from_fn(|i| f32_to_f16_to_f32(blk_raw[i]));
                let fmt = if is_key {
                    if let Some(q) = q_data {
                        if end <= q.len() {
                            let q_raw: [f32; SELECT_BLOCK] = q[start..end].try_into().unwrap();
                            let q_blk: [f32; SELECT_BLOCK] =
                                std::array::from_fn(|i| f32_to_f16_to_f32(q_raw[i]));
                            cpu_select_k_qproj(&blk, &q_blk, candidates, t, t, 0.0, 1.0, 1.0)
                        } else {
                            select_format_from_candidates_k(&blk, candidates, t)
                        }
                    } else {
                        select_format_from_candidates_k(&blk, candidates, t)
                    }
                } else {
                    select_format_from_candidates(&blk, candidates, t)
                };
                fmts.push(fmt);
            }
        }
    }
    fmts
}

#[test]
#[ignore]
fn test_reduction_granularity_comparison() {
    let path = match dump_path() {
        Some(p) => p,
        None => {
            println!("kv_selection_tests: dump absent — run test_dump_kv_cache_data first.");
            return;
        }
    };

    let (header, chunks) = load_dump(&path).expect("failed to load dump");
    let elems_per_chunk = header.n_kv_head * header.chunk_size * header.head_dim;
    let blocks_per_chunk = elems_per_chunk / SELECT_BLOCK;
    let blocks_per_head = (header.chunk_size * header.head_dim) / SELECT_BLOCK;

    println!(
        "\nDump: {} layers, {} chunks, {} heads, chunk_size={}, head_dim={}",
        header.num_layers,
        chunks.len(),
        header.n_kv_head,
        header.chunk_size,
        header.head_dim,
    );
    println!(
        "  blocks_per_chunk={blocks_per_chunk}  blocks_per_head={blocks_per_head}  heads={}",
        header.n_kv_head,
    );

    let candidates = level_candidates();

    let sep = "=".repeat(130);
    let sep2 = "-".repeat(130);

    println!("\n{sep}");
    println!(
        "Reduction Granularity Comparison: per-chunk (512 blk) vs per-head (128 blk) worst-case"
    );
    println!("{sep2}");
    println!(
        "  {:>4}  {:>9}  {:>8}   {:>8} {:>8} {:>8}   {:>8} {:>8} {:>8}   {:>8} {:>8} {:>8}",
        "Mode",
        "Tier",
        "Thr",
        "K_ideal",
        "K_chunk",
        "K_head",
        "V_ideal",
        "V_chunk",
        "V_head",
        "CR_ideal",
        "CR_chunk",
        "CR_head",
    );
    println!(
        "  {:>4}  {:>9}  {:>8}   {:>8} {:>8} {:>8}   {:>8} {:>8} {:>8}   {:>8} {:>8} {:>8}",
        "", "", "", "BPE", "BPE", "BPE", "BPE", "BPE", "BPE", "", "", "",
    );
    println!("  {}", "-".repeat(126));

    // Pre-compute all levels to reuse in summary
    #[allow(dead_code)]
    struct LevelResult {
        cr_ideal: f64,
        cr_chunk: f64,
        cr_head: f64,
        k_ideal: f64,
        k_chunk: f64,
        k_head: f64,
        v_ideal: f64,
        v_chunk: f64,
        v_head: f64,
    }
    let mut results = Vec::new();

    for i in 0..10 {
        let k_fmts = collect_block_formats(
            &chunks,
            blocks_per_chunk,
            K_MAGWEIGHT_THRESHOLDS[i],
            &candidates[i].0,
            true,
            None,
        );
        let v_fmts = collect_block_formats(
            &chunks,
            blocks_per_chunk,
            V_COSINE_THRESHOLDS_PROPOSED[i],
            &candidates[i].1,
            false,
            None, // reduction-granularity test doesn't need Q2_0 promo
        );

        let k_ideal: f64 =
            k_fmts.iter().map(|f| f.bits_per_elem() as f64).sum::<f64>() / k_fmts.len() as f64;
        let v_ideal: f64 =
            v_fmts.iter().map(|f| f.bits_per_elem() as f64).sum::<f64>() / v_fmts.len() as f64;

        let k_chunk = reduce_per_chunk(&k_fmts, blocks_per_chunk);
        let v_chunk = reduce_per_chunk(&v_fmts, blocks_per_chunk);
        let k_head = reduce_per_head(&k_fmts, blocks_per_chunk, blocks_per_head);
        let v_head = reduce_per_head(&v_fmts, blocks_per_chunk, blocks_per_head);

        let cr_ideal = 16.0 / ((k_ideal + v_ideal) / 2.0);
        let cr_chunk = 16.0 / ((k_chunk + v_chunk) / 2.0);
        let cr_head = 16.0 / ((k_head + v_head) / 2.0);

        println!(
            "  C{i:<3} {:<9} {:<8.5}   {:>8.3} {:>8.3} {:>8.3}   {:>8.3} {:>8.3} {:>8.3}   {:>7.2}x {:>7.2}x {:>7.2}x",
            LEVEL_TIER[i],
            K_MAGWEIGHT_THRESHOLDS[i],
            k_ideal, k_chunk, k_head,
            v_ideal, v_chunk, v_head,
            cr_ideal, cr_chunk, cr_head,
        );

        results.push(LevelResult {
            cr_ideal,
            cr_chunk,
            cr_head,
            k_ideal,
            k_chunk,
            k_head,
            v_ideal,
            v_chunk,
            v_head,
        });
    }

    // Summary table
    println!("\n{sep2}");
    println!("Summary: CR loss from worst-case reduction vs per-block ideal");
    println!(
        "  {:>4}  {:>9}  {:>10}  {:>10}  {:>10}  {:>12}  {:>12}",
        "Mode", "Tier", "CR_ideal", "CR_chunk", "CR_head", "chunk_loss", "head_loss",
    );
    println!("  {}", "-".repeat(80));

    for i in 0..10 {
        let r = &results[i];
        let chunk_loss_pct = (1.0 - r.cr_chunk / r.cr_ideal) * 100.0;
        let head_loss_pct = (1.0 - r.cr_head / r.cr_ideal) * 100.0;

        println!(
            "  C{i:<3} {:<9}  {:>9.3}x  {:>9.3}x  {:>9.3}x  {:>+10.1}%  {:>+10.1}%",
            LEVEL_TIER[i], r.cr_ideal, r.cr_chunk, r.cr_head, chunk_loss_pct, head_loss_pct,
        );
    }

    // Head recovery = how much of the chunk loss is recovered by going per-head
    println!("\n{sep2}");
    println!("Recovery: how much compression per-head recovers vs per-chunk");
    println!(
        "  {:>4}  {:>9}  {:>10}  {:>10}  {:>12}",
        "Mode", "Tier", "CR_chunk", "CR_head", "recovery",
    );
    println!("  {}", "-".repeat(55));

    for i in 0..10 {
        let r = &results[i];
        let lost = r.cr_ideal - r.cr_chunk;
        let recovered = r.cr_head - r.cr_chunk;
        let recovery_pct = if lost.abs() > 1e-6 {
            (recovered / lost) * 100.0
        } else {
            0.0
        };

        println!(
            "  C{i:<3} {:<9}  {:>9.3}x  {:>9.3}x  {:>+10.1}%",
            LEVEL_TIER[i], r.cr_chunk, r.cr_head, recovery_pct,
        );
    }
    println!("{sep}");
}

// ============================================================================
// R16 dump (v4) validation: K, V, Q sanity checks
// ============================================================================

/// Path to the R16 dump, relative to this crate's root.
const R16_DUMP_REL_PATH: &str = "src/kv_cache/chunked/tests/data/kv_cache_r16_dump.bin";

fn r16_dump_path() -> Option<std::path::PathBuf> {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let p = base.join(R16_DUMP_REL_PATH);
    if p.exists() {
        Some(p)
    } else {
        None
    }
}

/// Helper: compute basic stats for a float slice.
fn stats(data: &[f32]) -> (f32, f32, f32, f32, usize, usize, usize) {
    let n = data.len() as f32;
    let mean = data.iter().sum::<f32>() / n;
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;
    let std_dev = variance.sqrt();
    let nan_count = data.iter().filter(|x| x.is_nan()).count();
    let inf_count = data.iter().filter(|x| x.is_infinite()).count();
    let zero_count = data.iter().filter(|&&x| x == 0.0).count();
    (mean, std_dev, min, max, nan_count, inf_count, zero_count)
}

/// Validate the R16 v4 dump: check header, K/V/Q data integrity.
///
/// Checks:
/// - File parses as v4 (has Q data)
/// - Header matches Qwen3-30B-A3B: 48 layers, 8 KV heads, head_dim=128
/// - No NaN or Inf in K, V, or Q
/// - K, V, Q are not all-zero (signal is present)
/// - Q values differ from K (Q capture is distinct data, not a copy of K)
/// - Value ranges are plausible for transformer activations
///
/// Run:
///   cargo test --release --features cuda --lib --package candle-nn \
///     kv_cache::chunked::tests::kv_selection_tests::test_validate_r16_dump \
///     -- --ignored --nocapture
#[test]
#[ignore]
fn test_validate_r16_dump() {
    let path = match r16_dump_path() {
        Some(p) => p,
        None => {
            println!("R16 dump absent — run test_dump_r16_kvq_data first. Skipping.");
            return;
        }
    };

    let (header, chunks) = load_dump(&path).expect("Failed to parse R16 dump file");

    // ── Header checks ──
    println!("=== R16 Dump Validation ===");
    println!(
        "Layers: {}, KV heads: {}, chunk_size: {}, head_dim: {}",
        header.num_layers, header.n_kv_head, header.chunk_size, header.head_dim
    );
    println!(
        "Tokens: {} (first 8: {:?})",
        header.tokens.len(),
        &header.tokens[..header.tokens.len().min(8)]
    );
    println!("Total chunks: {}", chunks.len());

    assert_eq!(
        header.num_layers, 48,
        "Expected 48 layers for Qwen3-30B-A3B"
    );
    assert_eq!(header.n_kv_head, 4, "Expected 4 KV heads");
    assert_eq!(header.head_dim, 128, "Expected head_dim=128");
    assert!(
        header.tokens.len() > 600,
        "Expected 600+ tokens, got {}",
        header.tokens.len()
    );

    let elems_per_chunk = header.n_kv_head * header.chunk_size * header.head_dim;

    // ── Per-chunk validation ──
    let mut total_k_nans = 0usize;
    let mut total_v_nans = 0usize;
    let mut total_q_nans = 0usize;
    let mut total_k_infs = 0usize;
    let mut total_v_infs = 0usize;
    let mut total_q_infs = 0usize;
    let mut all_q_zero_chunks = 0usize;
    let mut all_k_zero_chunks = 0usize;
    let mut k_eq_q_chunks = 0usize; // chunks where K == Q exactly

    for chunk in &chunks {
        assert_eq!(
            chunk.k.len(),
            elems_per_chunk,
            "K length mismatch in layer {} block {}",
            chunk.layer_idx,
            chunk.block_idx
        );
        assert_eq!(
            chunk.v.len(),
            elems_per_chunk,
            "V length mismatch in layer {} block {}",
            chunk.layer_idx,
            chunk.block_idx
        );
        let q = chunk.q.as_ref().expect("v4 dump should have Q data");
        assert_eq!(
            q.len(),
            elems_per_chunk,
            "Q length mismatch in layer {} block {}",
            chunk.layer_idx,
            chunk.block_idx
        );

        let (_, _, _, _, k_nans, k_infs, k_zeros) = stats(&chunk.k);
        let (_, _, _, _, v_nans, v_infs, _) = stats(&chunk.v);
        let (_, _, _, _, q_nans, q_infs, q_zeros) = stats(q);

        total_k_nans += k_nans;
        total_v_nans += v_nans;
        total_q_nans += q_nans;
        total_k_infs += k_infs;
        total_v_infs += v_infs;
        total_q_infs += q_infs;

        if k_zeros == elems_per_chunk {
            all_k_zero_chunks += 1;
        }
        if q_zeros == elems_per_chunk {
            all_q_zero_chunks += 1;
        }

        // Check K != Q (they should be different — K is key, Q is query)
        if chunk.k == *q {
            k_eq_q_chunks += 1;
        }
    }

    println!("\n--- Integrity ---");
    println!(
        "K NaN: {}, Inf: {}, all-zero chunks: {}",
        total_k_nans, total_k_infs, all_k_zero_chunks
    );
    println!("V NaN: {}, Inf: {}", total_v_nans, total_v_infs);
    println!(
        "Q NaN: {}, Inf: {}, all-zero chunks: {}",
        total_q_nans, total_q_infs, all_q_zero_chunks
    );
    println!(
        "K == Q chunks (exact match): {}/{}",
        k_eq_q_chunks,
        chunks.len()
    );

    assert_eq!(total_k_nans, 0, "K has NaN values");
    assert_eq!(total_v_nans, 0, "V has NaN values");
    assert_eq!(total_q_nans, 0, "Q has NaN values");
    assert_eq!(total_k_infs, 0, "K has Inf values");
    assert_eq!(total_v_infs, 0, "V has Inf values");
    assert_eq!(total_q_infs, 0, "Q has Inf values");
    assert_eq!(all_k_zero_chunks, 0, "Some K chunks are all-zero");
    assert_eq!(
        all_q_zero_chunks, 0,
        "Some Q chunks are all-zero — Q capture may have failed"
    );
    assert_eq!(
        k_eq_q_chunks, 0,
        "K == Q in some chunks — Q may be a copy of K, not actual queries"
    );

    // ── Aggregate stats across all chunks ──
    // Sample a few layers for detailed stats
    let sample_layers = [0, 1, 23, 47]; // first, second, middle, last
    println!("\n--- Per-layer sample stats (layer → K mean/std, V mean/std, Q mean/std) ---");
    for &layer in &sample_layers {
        let layer_chunks: Vec<&ChunkData> =
            chunks.iter().filter(|c| c.layer_idx == layer).collect();
        if layer_chunks.is_empty() {
            continue;
        }

        let all_k: Vec<f32> = layer_chunks
            .iter()
            .flat_map(|c| c.k.iter().cloned())
            .collect();
        let all_v: Vec<f32> = layer_chunks
            .iter()
            .flat_map(|c| c.v.iter().cloned())
            .collect();
        let all_q: Vec<f32> = layer_chunks
            .iter()
            .flat_map(|c| c.q.as_ref().unwrap().iter().cloned())
            .collect();

        let (k_mean, k_std, k_min, k_max, _, _, _) = stats(&all_k);
        let (v_mean, v_std, v_min, v_max, _, _, _) = stats(&all_v);
        let (q_mean, q_std, q_min, q_max, _, _, _) = stats(&all_q);

        println!(
            "  Layer {:2}: K [{:+.4} ± {:.4}] range [{:.3}, {:.3}]",
            layer, k_mean, k_std, k_min, k_max
        );
        println!(
            "           V [{:+.4} ± {:.4}] range [{:.3}, {:.3}]",
            v_mean, v_std, v_min, v_max
        );
        println!(
            "           Q [{:+.4} ± {:.4}] range [{:.3}, {:.3}]",
            q_mean, q_std, q_min, q_max
        );
    }

    // ── Cross-check: K and Q should have different distributions ──
    // Compute correlation between first chunk's K and Q
    let c0 = &chunks[0];
    let q0 = c0.q.as_ref().unwrap();
    let n = c0.k.len() as f64;
    let k_mean = c0.k.iter().map(|&x| x as f64).sum::<f64>() / n;
    let q_mean = q0.iter().map(|&x| x as f64).sum::<f64>() / n;
    let cov: f64 =
        c0.k.iter()
            .zip(q0.iter())
            .map(|(&k, &q)| (k as f64 - k_mean) * (q as f64 - q_mean))
            .sum::<f64>()
            / n;
    let k_var: f64 =
        c0.k.iter()
            .map(|&x| (x as f64 - k_mean).powi(2))
            .sum::<f64>()
            / n;
    let q_var: f64 = q0.iter().map(|&x| (x as f64 - q_mean).powi(2)).sum::<f64>() / n;
    let corr = if k_var > 0.0 && q_var > 0.0 {
        cov / (k_var.sqrt() * q_var.sqrt())
    } else {
        0.0
    };
    println!("\n--- K/Q correlation (layer 0, chunk 0): {:.4} ---", corr);
    // K and Q are different projection outputs — correlation should not be ~1.0
    assert!(
        corr.abs() < 0.99,
        "K and Q are too correlated ({:.4}) — Q may not be real query data",
        corr
    );

    println!("\n=== All R16 dump validation checks passed ===");
}

/// Analyse Q value significance for attention.
///
/// For each token position across all layers, computes the Q vector norm and
/// per-element contribution to attention scores (Q·K / sqrt(head_dim)).
/// Reports what fraction of Q elements are large enough to materially affect
/// attention weights.
///
/// Run:
///   cargo test --release --features cuda --lib --package candle-nn \
///     kv_cache::chunked::tests::kv_selection_tests::test_q_attention_significance \
///     -- --ignored --nocapture
#[test]
#[ignore]
fn test_q_attention_significance() {
    let path = match r16_dump_path() {
        Some(p) => p,
        None => {
            println!("R16 dump absent — run test_dump_r16_kvq_data first. Skipping.");
            return;
        }
    };

    let (header, chunks) = load_dump(&path).expect("Failed to parse R16 dump file");

    let head_dim = header.head_dim; // 128
    let n_kv_head = header.n_kv_head; // 4
    let chunk_size = header.chunk_size; // 32
    let scale = 1.0 / (head_dim as f64).sqrt(); // 1/sqrt(128) ≈ 0.0884

    println!("=== Q Attention Significance Analysis ===");
    println!(
        "head_dim={}, n_kv_head={}, chunk_size={}, scale={:.4}",
        head_dim, n_kv_head, chunk_size, scale
    );
    println!(
        "Total chunks: {}, layers: {}",
        chunks.len(),
        header.num_layers
    );

    // ── 1. Per-element |q| magnitude distribution ──
    // Collect all Q magnitudes across all chunks
    let all_q_abs: Vec<f32> = chunks
        .iter()
        .flat_map(|c| c.q.as_ref().unwrap().iter().map(|&x| x.abs()))
        .collect();
    let total_q_elems = all_q_abs.len();

    let thresholds = [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    println!(
        "\n--- Q element magnitude distribution (total: {} elements) ---",
        total_q_elems
    );
    println!("  |q| > threshold  →  count      proportion");
    for &t in &thresholds {
        let count = all_q_abs.iter().filter(|&&x| x > t).count();
        println!(
            "  |q| > {:>6.3}     →  {:>10}  ({:.4}%)",
            t,
            count,
            100.0 * count as f64 / total_q_elems as f64
        );
    }

    // ── 2. Per-head Q vector norms and sparsity per token position ──
    // For each (layer, chunk, head, token), compute Q vector L2 norm
    // and the fraction of elements carrying 90% of energy (L2²)
    let mut all_q_norms: Vec<f64> = Vec::new();
    let mut all_energy_fracs: Vec<f64> = Vec::new(); // fraction of elems for 90% energy

    for chunk in &chunks {
        let q = chunk.q.as_ref().unwrap();
        for h in 0..n_kv_head {
            for t in 0..chunk_size {
                let offset = h * chunk_size * head_dim + t * head_dim;
                let qvec = &q[offset..offset + head_dim];

                // L2 norm
                let norm_sq: f64 = qvec.iter().map(|&x| (x as f64) * (x as f64)).sum();
                let norm = norm_sq.sqrt();
                all_q_norms.push(norm);

                // Energy concentration: sort squared magnitudes descending,
                // find how many elements carry 90% of total energy
                if norm_sq > 1e-12 {
                    let mut sq_sorted: Vec<f64> =
                        qvec.iter().map(|&x| (x as f64) * (x as f64)).collect();
                    sq_sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
                    let target = 0.9 * norm_sq;
                    let mut cumsum = 0.0;
                    let mut count = 0usize;
                    for &s in &sq_sorted {
                        cumsum += s;
                        count += 1;
                        if cumsum >= target {
                            break;
                        }
                    }
                    all_energy_fracs.push(count as f64 / head_dim as f64);
                }
            }
        }
    }

    // Q norm stats
    all_q_norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_norms = all_q_norms.len();
    let norm_p50 = all_q_norms[n_norms / 2];
    let norm_p90 = all_q_norms[n_norms * 9 / 10];
    let norm_p99 = all_q_norms[n_norms * 99 / 100];
    let norm_mean = all_q_norms.iter().sum::<f64>() / n_norms as f64;
    println!(
        "\n--- Q vector norms ({} vectors, each {} dims) ---",
        n_norms, head_dim
    );
    println!(
        "  mean: {:.4}, p50: {:.4}, p90: {:.4}, p99: {:.4}",
        norm_mean, norm_p50, norm_p90, norm_p99
    );
    println!(
        "  min: {:.4}, max: {:.4}",
        all_q_norms[0],
        all_q_norms[n_norms - 1]
    );

    // Energy concentration stats
    all_energy_fracs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n_ef = all_energy_fracs.len();
    let ef_mean = all_energy_fracs.iter().sum::<f64>() / n_ef as f64;
    let ef_p50 = all_energy_fracs[n_ef / 2];
    let ef_p10 = all_energy_fracs[n_ef / 10];
    println!("\n--- Q energy concentration (fraction of dims for 90%% L2 energy) ---");
    println!(
        "  mean: {:.4}, p50: {:.4}, p10 (most concentrated): {:.4}",
        ef_mean, ef_p50, ef_p10
    );

    // ── 3. Actual attention score analysis ──
    // For each chunk, compute Q·K attention scores (each Q token attending to
    // each K token in the same chunk) and measure the distribution.
    println!("\n--- Attention score analysis (Q·K/sqrt(d) within same chunk) ---");

    let mut all_scores: Vec<f64> = Vec::new();
    let mut max_attn_weights: Vec<f64> = Vec::new(); // max softmax weight per Q token

    // Sample: layers 0, 23, 47 — to keep runtime manageable
    let sample_layers = [0, 23, 47];
    for &layer in &sample_layers {
        let layer_chunks: Vec<&ChunkData> =
            chunks.iter().filter(|c| c.layer_idx == layer).collect();

        let mut layer_scores: Vec<f64> = Vec::new();
        let mut layer_max_weights: Vec<f64> = Vec::new();

        for chunk in &layer_chunks {
            let q = chunk.q.as_ref().unwrap();
            let k = &chunk.k;

            for h in 0..n_kv_head {
                for qt in 0..chunk_size {
                    let q_off = h * chunk_size * head_dim + qt * head_dim;
                    let qvec = &q[q_off..q_off + head_dim];

                    // Compute dot product with every K token in this chunk
                    let mut scores_for_token: Vec<f64> = Vec::with_capacity(chunk_size);
                    for kt in 0..chunk_size {
                        let k_off = h * chunk_size * head_dim + kt * head_dim;
                        let kvec = &k[k_off..k_off + head_dim];
                        let dot: f64 = qvec
                            .iter()
                            .zip(kvec.iter())
                            .map(|(&q, &k)| q as f64 * k as f64)
                            .sum();
                        let score = dot * scale;
                        scores_for_token.push(score);
                        layer_scores.push(score);
                    }

                    // Softmax to get attention weights
                    let max_s = scores_for_token
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);
                    let exp_sum: f64 = scores_for_token.iter().map(|&s| (s - max_s).exp()).sum();
                    let _max_weight = (0.0f64 - max_s).exp() / exp_sum; // weight of max-score token
                                                                        // Actually compute the true max weight
                    let max_weight = scores_for_token
                        .iter()
                        .map(|&s| ((s - max_s).exp()) / exp_sum)
                        .fold(0.0f64, f64::max);
                    layer_max_weights.push(max_weight);
                }
            }
        }

        all_scores.extend(&layer_scores);
        max_attn_weights.extend(&layer_max_weights);

        // Layer summary
        layer_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        layer_max_weights.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ns = layer_scores.len();
        let nw = layer_max_weights.len();
        println!("\n  Layer {}:", layer);
        println!(
            "    Raw scores (Q·K/√d): mean={:.3}, p1={:.3}, p50={:.3}, p99={:.3}",
            layer_scores.iter().sum::<f64>() / ns as f64,
            layer_scores[ns / 100],
            layer_scores[ns / 2],
            layer_scores[ns * 99 / 100]
        );
        println!(
            "    Max attn weight per Q token: mean={:.4}, p10={:.4}, p50={:.4}, p90={:.4}",
            layer_max_weights.iter().sum::<f64>() / nw as f64,
            layer_max_weights[nw / 10],
            layer_max_weights[nw / 2],
            layer_max_weights[nw * 9 / 10]
        );

        // What fraction of Q tokens have a dominant attention weight (>0.5)?
        let dominant = layer_max_weights.iter().filter(|&&w| w > 0.5).count();
        let spread = layer_max_weights
            .iter()
            .filter(|&&w| w < 1.0 / chunk_size as f64 * 2.0)
            .count();
        println!(
            "    Dominant (max weight > 0.5): {}/{} ({:.1}%)",
            dominant,
            nw,
            100.0 * dominant as f64 / nw as f64
        );
        println!(
            "    Spread (max weight < 2/chunk_size): {}/{} ({:.1}%)",
            spread,
            nw,
            100.0 * spread as f64 / nw as f64
        );
    }

    // ── 4. Per-element Q contribution to attention ──
    // For a random Q·K pair, each element contributes q_i * k_i * scale.
    // What fraction of elements contribute >1% of the total score?
    println!("\n--- Per-element attention contribution ---");
    let sample_chunk = &chunks[chunks.len() / 2]; // middle chunk
    let q = sample_chunk.q.as_ref().unwrap();
    let k = &sample_chunk.k;
    let mut contrib_fracs: Vec<f64> = Vec::new();

    for h in 0..n_kv_head {
        for qt in 0..chunk_size {
            let q_off = h * chunk_size * head_dim + qt * head_dim;
            let k_off = h * chunk_size * head_dim + qt * head_dim; // self-attention position
            let qvec = &q[q_off..q_off + head_dim];
            let kvec = &k[k_off..k_off + head_dim];

            let contribs: Vec<f64> = qvec
                .iter()
                .zip(kvec.iter())
                .map(|(&qi, &ki)| (qi as f64 * ki as f64).abs())
                .collect();
            let total: f64 = contribs.iter().sum();
            if total > 1e-12 {
                let significant = contribs.iter().filter(|&&c| c > 0.01 * total).count();
                contrib_fracs.push(significant as f64 / head_dim as f64);
            }
        }
    }
    contrib_fracs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let nc = contrib_fracs.len();
    println!("  Fraction of dims contributing >1%% of |Q·K| score:");
    println!(
        "  mean: {:.4}, p10: {:.4}, p50: {:.4}, p90: {:.4}",
        contrib_fracs.iter().sum::<f64>() / nc as f64,
        contrib_fracs[nc / 10],
        contrib_fracs[nc / 2],
        contrib_fracs[nc * 9 / 10]
    );

    println!("\n=== Q Attention Significance Analysis Complete ===");
}

// ============================================================================
// Q-subspace-projected K error metric
// ============================================================================

/// Compute the top-k principal components of a set of Q vectors using
/// power iteration with deflation.  Returns a k×d matrix (row-major)
/// where each row is a unit-length principal component.
///
/// `q_vecs` is a slice of n vectors, each of length `d`.
fn compute_q_subspace(q_vecs: &[Vec<f32>], d: usize, k: usize) -> Vec<f32> {
    let n = q_vecs.len();
    if n == 0 || k == 0 {
        return vec![0.0; k * d];
    }

    // We'll work with f64 for numerical stability.
    // Compute Q^T Q (d×d covariance, no mean centering — we want the
    // subspace that Q occupies, not the variance from the centroid).
    let mut cov = vec![0.0f64; d * d];
    for q in q_vecs {
        for r in 0..d {
            let qr = q[r] as f64;
            for c in r..d {
                let v = qr * q[c] as f64;
                cov[r * d + c] += v;
                if c != r {
                    cov[c * d + r] += v;
                }
            }
        }
    }
    // Normalise
    let inv_n = 1.0 / n as f64;
    for v in cov.iter_mut() {
        *v *= inv_n;
    }

    // Power iteration with deflation to extract top-k eigenvectors.
    let actual_k = k.min(d);
    let mut basis = Vec::with_capacity(actual_k * d);
    let max_iter = 200;
    let tol = 1e-10;

    for _ in 0..actual_k {
        // Initialise with a random-ish vector (use first non-zero column of cov).
        let mut v = vec![0.0f64; d];
        // seed: (1, 1, 1, ...) then orthogonalise against found basis
        for vi in v.iter_mut() {
            *vi = 1.0;
        }

        for _iter in 0..max_iter {
            // Orthogonalise against already-found basis vectors
            let num_found = basis.len() / d;
            for bi in 0..num_found {
                let b_row = &basis[bi * d..(bi + 1) * d];
                let dot: f64 = v.iter().zip(b_row.iter()).map(|(&a, &b)| a * b).sum();
                for (vi, &bi_val) in v.iter_mut().zip(b_row.iter()) {
                    *vi -= dot * bi_val;
                }
            }

            // Multiply: w = cov * v
            let mut w = vec![0.0f64; d];
            for r in 0..d {
                let mut s = 0.0f64;
                for c in 0..d {
                    s += cov[r * d + c] * v[c];
                }
                w[r] = s;
            }

            // Normalise
            let norm: f64 = w.iter().map(|&x| x * x).sum::<f64>().sqrt();
            if norm < tol {
                // Eigenvalue ≈ 0; fill remaining basis with zeros
                for _ in 0..d {
                    basis.push(0.0f64);
                }
                break;
            }
            let inv = 1.0 / norm;
            for wi in w.iter_mut() {
                *wi *= inv;
            }

            // Check convergence: |w - v| < tol
            let diff: f64 = w
                .iter()
                .zip(v.iter())
                .map(|(&a, &b)| (a - b).powi(2))
                .sum::<f64>();
            v = w;
            if diff < tol * tol {
                break;
            }
        }

        // Store this basis vector
        for &vi in &v {
            basis.push(vi);
        }
    }

    // Convert to f32
    basis.iter().map(|&x| x as f32).collect()
}

/// Q-subspace-projected K error for a single 32-element sub-block.
///
/// Both numerator and denominator are block-local: the error from quantising
/// this sub-block is measured relative to this block's own contribution to the
/// Q-projected K energy.
///
///   err_b  = Σ_i (Σ_{j∈block} Q̂_ij · (K_j - K̂_j))²
///   norm_b = Σ_i (Σ_{j∈block} Q̂_ij · K_j)²
///
/// Returns err_b / norm_b.  A block where Q̂ has little weight in these 32 dims
/// will have both numerator and denominator near zero → returns 0.0 (safe to
/// compress aggressively).  A block where Q̂ concentrates weight will have a
/// large denominator and the error fraction correctly reflects attention impact.
fn q_projected_block_error(
    q_hat: &[f32], // subspace_k × head_dim, row-major
    subspace_k: usize,
    head_dim: usize,
    block_orig: &[f32; SELECT_BLOCK],
    block_recon: &[f32; SELECT_BLOCK],
    block_offset: usize, // starting element index within head_dim
) -> f32 {
    if subspace_k == 0 || head_dim == 0 {
        return 0.0;
    }

    let mut err_norm_sq = 0.0f64;
    let mut orig_norm_sq = 0.0f64;

    for i in 0..subspace_k {
        let row = &q_hat[i * head_dim..(i + 1) * head_dim];
        let block_row = &row[block_offset..block_offset + SELECT_BLOCK];

        // Error projection: Q̂ · (K - K̂) over this block's elements
        let mut proj_err = 0.0f64;
        // Original projection: Q̂ · K over this block's elements
        let mut proj_orig = 0.0f64;
        for j in 0..SELECT_BLOCK {
            let qj = block_row[j] as f64;
            proj_err += qj * (block_orig[j] - block_recon[j]) as f64;
            proj_orig += qj * block_orig[j] as f64;
        }
        err_norm_sq += proj_err * proj_err;
        orig_norm_sq += proj_orig * proj_orig;
    }

    if orig_norm_sq < 1e-30 {
        return 0.0;
    }
    (err_norm_sq / orig_norm_sq) as f32
}

/// Per-sub-block K format selection using Q-subspace-projected error.
///
/// Each of the `head_dim / 32` sub-blocks is independently assigned the most
/// aggressive format whose block-local Q-projected error ≤ `threshold`.
/// This produces heterogeneous format mixes depending on Q sensitivity per
/// dimension range.
///
/// Two-sided relevance check (applied per block):
///   relevance = ‖Q̂·K_block‖² / ‖K_block‖²
///   - If relevance < relevance_lo: Q-invisible → cheapest format (Q2_0)
///   - If relevance > relevance_hi: attention sink → sink_fmt (level-dependent)
///   - Otherwise: normal Q-projected error format scan
#[allow(dead_code)]
fn select_formats_k_qproj(
    k_full: &[f32], // head_dim elements
    q_hat: &[f32],  // subspace_k × head_dim, row-major
    subspace_k: usize,
    threshold: f32,
    relevance_lo: f32,
    relevance_hi: f32,
    sink_fmt: BlockFormat,
) -> Vec<BlockFormat> {
    select_formats_k_qproj_with_errors(
        k_full,
        q_hat,
        subspace_k,
        threshold,
        relevance_lo,
        relevance_hi,
        sink_fmt,
    )
    .0
}

/// Like select_formats_k_qproj but also returns the achieved error per block.
/// Returns (formats, errors, skip_lo_count, skip_hi_count).
fn select_formats_k_qproj_with_errors(
    k_full: &[f32], // head_dim elements
    q_hat: &[f32],  // subspace_k × head_dim, row-major
    subspace_k: usize,
    threshold: f32,
    relevance_lo: f32,
    relevance_hi: f32,
    sink_fmt: BlockFormat,
) -> (Vec<BlockFormat>, Vec<f32>, usize, usize) {
    let head_dim = k_full.len();
    let num_blocks = head_dim / SELECT_BLOCK;

    let mut formats = Vec::with_capacity(num_blocks);
    let mut errors = Vec::with_capacity(num_blocks);
    let skip_lo = 0usize;
    let skip_hi = 0usize;

    let fallback_fmt = CANDIDATES
        .iter()
        .map(|(fmt, _)| *fmt)
        .find(|&fmt| fmt != BlockFormat::BF16)
        .unwrap_or(sink_fmt);

    for b in 0..num_blocks {
        let start = b * SELECT_BLOCK;
        let blk: [f32; SELECT_BLOCK] = k_full[start..start + SELECT_BLOCK].try_into().unwrap();

        let k_energy: f64 = blk.iter().map(|&x| (x as f64) * (x as f64)).sum();
        let mut qk_energy = 0.0f64;
        for pc in 0..subspace_k {
            let row = &q_hat[pc * head_dim + start..pc * head_dim + start + SELECT_BLOCK];
            let dot: f64 = row
                .iter()
                .zip(blk.iter())
                .map(|(&q, &k)| q as f64 * k as f64)
                .sum();
            qk_energy += dot * dot;
        }
        let relevance = if k_energy > 0.0 {
            (qk_energy / k_energy) as f32
        } else {
            0.0
        };
        let _ = (relevance, relevance_lo, relevance_hi);

        let mut chosen = fallback_fmt;
        let mut chosen_err = f32::INFINITY;
        let mut least_error_fmt = fallback_fmt;
        let mut least_error = f32::INFINITY;
        for &(fmt, round_trip) in CANDIDATES {
            if fmt == BlockFormat::BF16 {
                continue;
            }
            let recon = round_trip(&blk);
            let err = q_projected_block_error(q_hat, subspace_k, head_dim, &blk, &recon, start);
            let better_least_error = err < least_error
                || (err == least_error
                    && (fmt.bits_per_elem() < least_error_fmt.bits_per_elem()
                        || (fmt.bits_per_elem() == least_error_fmt.bits_per_elem()
                            && fmt.table_index() < least_error_fmt.table_index())));
            if better_least_error {
                least_error_fmt = fmt;
                least_error = err;
            }
            if err <= threshold {
                let better = chosen_err.is_infinite()
                    || fmt.bits_per_elem() < chosen.bits_per_elem()
                    || (fmt.bits_per_elem() == chosen.bits_per_elem()
                        && (err < chosen_err
                            || (err == chosen_err && fmt.table_index() < chosen.table_index())));
                if better {
                    chosen = fmt;
                    chosen_err = err;
                }
            }
        }
        let final_fmt = if chosen_err.is_finite() {
            chosen
        } else {
            least_error_fmt
        };
        let final_err = if chosen_err.is_finite() {
            chosen_err
        } else {
            least_error
        };
        formats.push(final_fmt);
        errors.push(if final_err.is_finite() {
            final_err
        } else {
            0.0
        });
    }

    (formats, errors, skip_lo, skip_hi)
}

/// Q-subspace-projected K format selection.
///
/// This test:
/// 1. Builds the Q subspace (top-k PCA) per layer per head from the R16 dump
/// 2. Uses Q-projected K error for format selection
/// 3. Sweeps threshold levels and generates format distribution tables
/// 4. Outputs K and compression ratio tables
///
/// Run with:
///   cargo test --release --lib --package candle-nn \
///     kv_selection_tests::test_qproj_k_format_selection -- --ignored --nocapture
#[test]
#[ignore]
fn test_qproj_k_format_selection() {
    let path = match r16_dump_path() {
        Some(p) => p,
        None => {
            println!("test_qproj_k_format_selection: R16 dump absent — run test_dump_r16_kvq_data first.");
            return;
        }
    };

    let (header, chunks) = load_dump(&path).expect("failed to load R16 dump");
    println!(
        "Loaded R16 dump: {} layers, {} kv_heads, head_dim={}, chunk_size={}, {} chunks",
        header.num_layers,
        header.n_kv_head,
        header.head_dim,
        header.chunk_size,
        chunks.len()
    );

    let head_dim = header.head_dim; // 128
    let chunk_size = header.chunk_size; // 32
    let n_kv_head = header.n_kv_head; // 4
    let num_layers = header.num_layers; // 48
    let subspace_k: usize = 30; // number of principal components

    // ── Step 1: Build Q subspace per (layer, head) ──────────────────────────
    println!("\nBuilding Q subspace ({subspace_k} PCs per layer×head)...");

    // Collect Q vectors: for each (layer, head), collect all Q row vectors (head_dim floats each).
    // In the dump, Q is [n_kv_head, chunk_size, head_dim] per chunk.
    // Each token in a chunk contributes one Q vector per head.
    let mut q_vecs_by_lh: Vec<Vec<Vec<f32>>> = vec![Vec::new(); num_layers * n_kv_head];

    for chunk in &chunks {
        let q = match &chunk.q {
            Some(q) => q,
            None => continue,
        };
        let layer = chunk.layer_idx;
        for h in 0..n_kv_head {
            for t in 0..chunk_size {
                let offset = h * chunk_size * head_dim + t * head_dim;
                if offset + head_dim > q.len() {
                    break;
                }
                let qv: Vec<f32> = q[offset..offset + head_dim].to_vec();
                // Skip zero vectors (padding)
                if qv.iter().all(|&x| x == 0.0) {
                    continue;
                }
                q_vecs_by_lh[layer * n_kv_head + h].push(qv);
            }
        }
    }

    // Compute subspace for each (layer, head).
    let mut subspaces: Vec<Vec<f32>> = Vec::with_capacity(num_layers * n_kv_head);
    for lh in 0..num_layers * n_kv_head {
        let sub = compute_q_subspace(&q_vecs_by_lh[lh], head_dim, subspace_k);
        subspaces.push(sub);
    }
    // Free Q vectors
    drop(q_vecs_by_lh);

    // Report subspace energy
    println!(
        "Q subspaces built for {} layer×head pairs.",
        subspaces.len()
    );

    // ── Step 2: Profile Q-projected error for every K block × every format ──
    //
    // For each K sub-block, compute the Q-projected error at each quantisation
    // format.  This gives us the empirical error CDF per format, from which we
    // derive thresholds that produce the desired compression curve.

    println!("\nProfiling Q-projected errors across all K blocks and formats...");

    // Formats in ladder order (most aggressive → least).
    let profile_fmts = [
        BlockFormat::Q2_0,
        BlockFormat::Q3_0,
        BlockFormat::Q4_0,
        BlockFormat::Q4_1,
        BlockFormat::Q4_KS,
        BlockFormat::Q8_0,
        BlockFormat::Q8_1,
        BlockFormat::Q8_KS,
        BlockFormat::BF16,
    ];
    let n_fmts = profile_fmts.len();
    let num_sub_blocks = head_dim / SELECT_BLOCK; // 128/32 = 4

    // Collect per-format error for every K sub-block.
    let mut errors_by_fmt: Vec<Vec<f32>> = vec![Vec::new(); n_fmts];

    for chunk in &chunks {
        let layer = chunk.layer_idx;
        for h in 0..n_kv_head {
            let lh = layer * n_kv_head + h;
            let q_hat = &subspaces[lh];

            for t in 0..chunk_size {
                let k_offset = h * chunk_size * head_dim + t * head_dim;
                if k_offset + head_dim > chunk.k.len() {
                    break;
                }
                let k_full = &chunk.k[k_offset..k_offset + head_dim];

                for b in 0..num_sub_blocks {
                    let start = b * SELECT_BLOCK;
                    let blk: [f32; SELECT_BLOCK] =
                        k_full[start..start + SELECT_BLOCK].try_into().unwrap();

                    for (fi, &fmt) in profile_fmts.iter().enumerate() {
                        let recon = fmt.apply_quant(&blk);
                        let err = q_projected_block_error(
                            q_hat, subspace_k, head_dim, &blk, &recon, start,
                        );
                        errors_by_fmt[fi].push(err);
                    }
                }
            }
        }
    }

    let n_blocks = errors_by_fmt[0].len();
    println!(
        "Profiled {} K sub-blocks × {} formats = {} measurements.",
        n_blocks,
        n_fmts,
        n_blocks * n_fmts
    );

    // ── Relevance profiling: ‖Q̂·K_block‖² / ‖K_block‖² per sub-block ──────
    let mut block_relevances: Vec<f32> = Vec::with_capacity(n_blocks);
    for chunk in &chunks {
        let layer = chunk.layer_idx;
        for h in 0..n_kv_head {
            let lh = layer * n_kv_head + h;
            let q_hat = &subspaces[lh];

            for t in 0..chunk_size {
                let k_offset = h * chunk_size * head_dim + t * head_dim;
                if k_offset + head_dim > chunk.k.len() {
                    break;
                }
                let k_full = &chunk.k[k_offset..k_offset + head_dim];

                for b in 0..num_sub_blocks {
                    let start = b * SELECT_BLOCK;
                    let blk = &k_full[start..start + SELECT_BLOCK];
                    let k_energy: f64 = blk.iter().map(|&x| (x as f64) * (x as f64)).sum();
                    let mut qk_energy = 0.0f64;
                    for pc in 0..subspace_k {
                        let row =
                            &q_hat[pc * head_dim + start..pc * head_dim + start + SELECT_BLOCK];
                        let dot: f64 = row
                            .iter()
                            .zip(blk.iter())
                            .map(|(&q, &k)| q as f64 * k as f64)
                            .sum();
                        qk_energy += dot * dot;
                    }
                    let rel = if k_energy > 0.0 {
                        (qk_energy / k_energy) as f32
                    } else {
                        0.0
                    };
                    block_relevances.push(rel);
                }
            }
        }
    }
    block_relevances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let pctile_f32 = |v: &[f32], p: f64| -> f32 {
        if v.is_empty() {
            return 0.0;
        }
        let idx = ((v.len() as f64 * p) as usize).min(v.len() - 1);
        v[idx]
    };
    let rel_pcts = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99];

    println!(
        "\nBlock-level relevance (‖Q̂·K_block‖²/‖K_block‖²) across {} blocks:",
        block_relevances.len()
    );
    print!("  ");
    for &p in &rel_pcts {
        print!(
            "p{:02}={:.4}  ",
            (p * 100.0) as u32,
            pctile_f32(&block_relevances, p)
        );
    }
    println!();

    // ── Relevance histogram ────────────────────────────────────────────────
    // 20 bins from 0.0–1.0, each 0.05 wide.
    let n_bins = 20usize;
    let bin_width = 1.0f32 / n_bins as f32;
    let mut hist = vec![0usize; n_bins];
    for &r in &block_relevances {
        let bin = ((r / bin_width) as usize).min(n_bins - 1);
        hist[bin] += 1;
    }
    let total = block_relevances.len() as f64;
    let max_count = *hist.iter().max().unwrap_or(&1) as f64;

    println!(
        "\n--- Relevance histogram ({} blocks, {} bins of {:.2}) ---",
        block_relevances.len(),
        n_bins,
        bin_width
    );
    println!(
        "  {:>10}  {:>8}  {:>6}  {:>6}  {}",
        "Bin", "Count", "%", "Cum%", "Bar"
    );
    println!("  {}", "-".repeat(70));
    let mut cumulative = 0usize;
    for i in 0..n_bins {
        cumulative += hist[i];
        let pct = hist[i] as f64 / total * 100.0;
        let cum_pct = cumulative as f64 / total * 100.0;
        let bar_len = (hist[i] as f64 / max_count * 40.0).round() as usize;
        let bar: String = "█".repeat(bar_len);
        println!(
            "  [{:>4.2},{:>4.2})  {:>8}  {:>5.1}%  {:>5.1}%  {}",
            i as f32 * bin_width,
            (i + 1) as f32 * bin_width,
            hist[i],
            pct,
            cum_pct,
            bar
        );
    }

    // Relevance thresholds from histogram:
    // Low  = 0.20 — left tail, blocks where < 20% of energy is in Q subspace (3.3% of blocks)
    // High = 0.95 — right tail, attention sinks with > 95% Q-projected energy (5.1% of blocks)
    let rel_lo = 0.20f32;
    let rel_hi = 0.95f32;
    let cnt_lo = block_relevances.iter().filter(|&&r| r < rel_lo).count();
    let cnt_hi = block_relevances.iter().filter(|&&r| r > rel_hi).count();
    println!("\nRelevance thresholds (from histogram):");
    println!(
        "  LOW  = {rel_lo:.2}  — blocks below: {cnt_lo} ({:.1}%)",
        cnt_lo as f64 / total * 100.0
    );
    println!(
        "  HIGH = {rel_hi:.2}  — blocks above: {cnt_hi} ({:.1}%)",
        cnt_hi as f64 / total * 100.0
    );

    let v_auto_thresholds = V_COSINE_THRESHOLDS_PROPOSED;

    // Sort each format's errors for percentile computation.
    for errs in errors_by_fmt.iter_mut() {
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    // ── Table A: Error percentiles per format ───────────────────────────────
    let sep = "=".repeat(110);
    println!("\n{sep}");
    println!("Table A — Q-projected error percentiles per format ({n_blocks} blocks)");
    println!("  Format   BPE     p25         p50         p75         p90         p95         p99         max");
    println!("  {}", "-".repeat(100));
    let pctile = |v: &[f32], p: f64| -> f64 {
        if v.is_empty() {
            return 0.0;
        }
        let idx = ((v.len() as f64 * p) as usize).min(v.len() - 1);
        v[idx] as f64
    };
    for (fi, &fmt) in profile_fmts.iter().enumerate() {
        let e = &errors_by_fmt[fi];
        println!("  {:<6}  {:>5.1}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.6}",
            fmt, fmt.bits_per_elem(),
            pctile(e, 0.25), pctile(e, 0.50), pctile(e, 0.75),
            pctile(e, 0.90), pctile(e, 0.95), pctile(e, 0.99),
            *e.last().unwrap_or(&0.0) as f64);
    }

    // ── Table B: "What % of blocks pass at threshold T" per format ──────────
    // Pick a set of probe thresholds spanning the observed error range.
    let probes: Vec<f32> = vec![
        0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.50,
    ];
    println!("\n{sep}");
    println!("Table B — Fraction of blocks passing threshold (%) per format");
    let mut hdr = format!("  {:>8}", "Thr");
    for &fmt in &profile_fmts {
        hdr += &format!(" {:>6}", format!("{}", fmt));
    }
    println!("{hdr}");
    println!("  {}", "-".repeat(8 + 7 * n_fmts));

    for &thr in &probes {
        let mut line = format!("  {:>8.5}", thr);
        for fi in 0..n_fmts {
            let e = &errors_by_fmt[fi];
            // Binary search for the threshold position.
            let pos = e.partition_point(|&x| x <= thr);
            let pct = pos as f64 / e.len() as f64 * 100.0;
            line += &format!(" {:>6.1}", pct);
        }
        println!("{line}");
    }

    // ── Step 3: Auto-derive thresholds from the error profile ───────────────
    //
    // Strategy: define 10 compression levels by target composition.
    // For each level, pick the threshold at which the cheapest acceptable
    // format passes enough blocks.  The "border" format at each level is
    // the one that should be *dominant* — the threshold is set to its
    // median error (p50), so ~50% of blocks use it and the rest spill to
    // neighbours.
    //
    // Level → border format → threshold = p_target of that format's error CDF
    //
    //   C0: Q8_0 p05  — nearly everything passes Q8_0
    //   C1: Q8_0 p50  — most blocks Q8_0
    //   C2: Q4_1 p25  — Q8_0→Q4_1 transition
    //   C3: Q4_0 p50  — Q4_0 dominant
    //   C4: Q4_0 p90  — nearly all blocks pass Q4_0
    //   C5: Q3_0 p50  — Q3_0 dominant
    //   C6: Q3_0 p75  — Q3_0 heavy
    //   C7: Q3_0 p90  — most blocks pass Q3_0
    //   C8: Q2_0 p50  — Q2_0 dominant
    //   C9: Q2_0 p75  — Q2_0 heavy, but still bounded

    let fmt_idx = |f: BlockFormat| -> usize { profile_fmts.iter().position(|&x| x == f).unwrap() };

    let k_qproj_thresholds: [f32; 10] = [
        pctile(&errors_by_fmt[fmt_idx(BlockFormat::Q8_0)], 0.05) as f32, // C0
        pctile(&errors_by_fmt[fmt_idx(BlockFormat::Q8_0)], 0.50) as f32, // C1
        pctile(&errors_by_fmt[fmt_idx(BlockFormat::Q4_1)], 0.25) as f32, // C2
        pctile(&errors_by_fmt[fmt_idx(BlockFormat::Q4_0)], 0.50) as f32, // C3
        pctile(&errors_by_fmt[fmt_idx(BlockFormat::Q4_0)], 0.90) as f32, // C4
        pctile(&errors_by_fmt[fmt_idx(BlockFormat::Q3_0)], 0.50) as f32, // C5
        pctile(&errors_by_fmt[fmt_idx(BlockFormat::Q3_0)], 0.75) as f32, // C6
        pctile(&errors_by_fmt[fmt_idx(BlockFormat::Q3_0)], 0.90) as f32, // C7
        pctile(&errors_by_fmt[fmt_idx(BlockFormat::Q2_0)], 0.50) as f32, // C8
        pctile(&errors_by_fmt[fmt_idx(BlockFormat::Q2_0)], 0.75) as f32, // C9
    ];

    println!("\n{sep}");
    println!("Auto-derived K Q-projected thresholds (from error CDFs):");
    let derivation = [
        "Q8_0 p05", "Q8_0 p50", "Q4_1 p25", "Q4_0 p50", "Q4_0 p90", "Q3_0 p50", "Q3_0 p75",
        "Q3_0 p90", "Q2_0 p50", "Q2_0 p75",
    ];
    println!("  {:<5}  {:>12}  {}", "Level", "Threshold", "Derivation");
    println!("  {}", "-".repeat(40));
    for i in 0..10 {
        println!(
            "  C{i}     {:>12.6}  {}",
            k_qproj_thresholds[i], derivation[i]
        );
    }

    // ── Step 4: Sweep levels and collect stats ──────────────────────────────
    // Two-sided relevance thresholds derived from data above
    println!(
        "\nSweeping {} auto-derived threshold levels (rel_lo={rel_lo:.4}, rel_hi={rel_hi:.4})...",
        k_qproj_thresholds.len()
    );

    struct LevelStats {
        k_counts: [usize; 16],
        v_counts: [usize; 16],
        k_total: usize,
        v_total: usize,
        k_bpe_sum: f64,
        v_bpe_sum: f64,
        k_snr_sum: f64,     // sum of -10·log10(err) for K blocks with err > 0
        k_snr_count: usize, // number of K blocks with err > 0
        k_zero_err: usize,  // K blocks with err == 0 (infinite SNR)
        v_snr_sum: f64,
        v_snr_count: usize,
        v_zero_err: usize,
        k_skip_lo: usize, // blocks fast-tracked to Q2_0 (Q-invisible)
        k_skip_hi: usize, // blocks fast-tracked to best format (attention sinks)
    }

    let mut all_levels: Vec<LevelStats> = Vec::new();

    for (level_idx, &k_thr) in k_qproj_thresholds.iter().enumerate() {
        let v_thr = v_auto_thresholds[level_idx];

        // Sink format: F16 for C0–C5, Q8_0 for C6–C9
        let sink_fmt = if level_idx >= 6 {
            BlockFormat::Q8_0
        } else {
            BlockFormat::F16
        };

        // V max format cap: BF16 for C0–C2, Q8_0 for C3–C6, Q4_0 for C7–C9
        let v_max_bpe: f64 = if level_idx >= 7 {
            BlockFormat::Q4_0.bits_per_elem() as f64
        } else if level_idx >= 3 {
            BlockFormat::Q8_0.bits_per_elem() as f64
        } else {
            BlockFormat::BF16.bits_per_elem() as f64
        };

        let mut ls = LevelStats {
            k_counts: [0; 16],
            v_counts: [0; 16],
            k_total: 0,
            v_total: 0,
            k_bpe_sum: 0.0,
            v_bpe_sum: 0.0,
            k_snr_sum: 0.0,
            k_snr_count: 0,
            k_zero_err: 0,
            v_snr_sum: 0.0,
            v_snr_count: 0,
            v_zero_err: 0,
            k_skip_lo: 0,
            k_skip_hi: 0,
        };

        for chunk in &chunks {
            let layer = chunk.layer_idx;

            for h in 0..n_kv_head {
                let lh = layer * n_kv_head + h;
                let q_hat = &subspaces[lh];

                for t in 0..chunk_size {
                    let k_offset = h * chunk_size * head_dim + t * head_dim;
                    if k_offset + head_dim > chunk.k.len() {
                        break;
                    }
                    let k_full = &chunk.k[k_offset..k_offset + head_dim];

                    // K format selection: Q-projected metric across all sub-blocks
                    let (k_formats, k_errors, slo, shi) = select_formats_k_qproj_with_errors(
                        k_full, q_hat, subspace_k, k_thr, rel_lo, rel_hi, sink_fmt,
                    );
                    ls.k_skip_lo += slo;
                    ls.k_skip_hi += shi;
                    for (idx, &fmt) in k_formats.iter().enumerate() {
                        ls.k_counts[fmt.table_index()] += 1;
                        ls.k_bpe_sum += fmt.bits_per_elem() as f64;
                        ls.k_total += 1;
                        let err = k_errors[idx] as f64;
                        if err > 0.0 {
                            ls.k_snr_sum += -10.0 * err.log10();
                            ls.k_snr_count += 1;
                        } else {
                            ls.k_zero_err += 1;
                        }
                    }

                    // V format selection: cosine distance (unchanged from existing)
                    let v_offset = h * chunk_size * head_dim + t * head_dim;
                    if v_offset + head_dim > chunk.v.len() {
                        continue;
                    }
                    for b in 0..num_sub_blocks {
                        let start = v_offset + b * SELECT_BLOCK;
                        let end = start + SELECT_BLOCK;
                        if end > chunk.v.len() {
                            break;
                        }
                        let blk: [f32; SELECT_BLOCK] = chunk.v[start..end].try_into().unwrap();
                        let (mut fmt, v_err) = select_format_with_error(&blk, v_thr);
                        // Cap V format to level-dependent maximum
                        if fmt.bits_per_elem() as f64 > v_max_bpe {
                            // Find the most expensive format within the cap
                            let mut best_capped = BlockFormat::Q2_0;
                            for &(cf, _) in CANDIDATES {
                                if cf.bits_per_elem() as f64 <= v_max_bpe
                                    && cf.bits_per_elem() >= best_capped.bits_per_elem()
                                {
                                    best_capped = cf;
                                }
                            }
                            fmt = best_capped;
                        }
                        ls.v_counts[fmt.table_index()] += 1;
                        ls.v_bpe_sum += fmt.bits_per_elem() as f64;
                        ls.v_total += 1;
                        let ve = v_err as f64;
                        if ve > 0.0 {
                            ls.v_snr_sum += -10.0 * ve.log10();
                            ls.v_snr_count += 1;
                        } else {
                            ls.v_zero_err += 1;
                        }
                    }
                }
            }
        }

        let k_bpe = if ls.k_total > 0 {
            ls.k_bpe_sum / ls.k_total as f64
        } else {
            16.0
        };
        let v_bpe = if ls.v_total > 0 {
            ls.v_bpe_sum / ls.v_total as f64
        } else {
            16.0
        };
        let cr = 16.0 / ((k_bpe + v_bpe) / 2.0);
        let lo_pct = ls.k_skip_lo as f64 / ls.k_total.max(1) as f64 * 100.0;
        let hi_pct = ls.k_skip_hi as f64 / ls.k_total.max(1) as f64 * 100.0;
        println!("  C{level_idx}: K_bpe={k_bpe:.3}, V_bpe={v_bpe:.3}, CR={cr:.2}x  (skip_lo={:.1}%, skip_hi={:.1}%)",
            lo_pct, hi_pct);

        all_levels.push(ls);
    }

    // ── Table 1: K cache format distribution ────────────────────────────────
    println!("\n{sep}");
    println!("Q-Subspace-Projected K Format Selection  (subspace_k={subspace_k}, auto-derived thresholds)");
    println!(
        "  {} layers, {} kv_heads, head_dim={}, chunk_size={}",
        num_layers, n_kv_head, head_dim, chunk_size
    );
    println!("  K metric: ||Q̂·(K-K̂)||² / ||Q̂·K||²   V metric: cosine distance (unchanged)");

    println!("\n--- K cache format distribution (%) ---");
    println!("  {:<5} {:>10}  {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4}  {:>6} {:>6}",
        "Level", "Thr", "Q0", "Q1S", "Q2S", "Q20", "Q2A", "Q21", "Q30", "Q31", "Q40", "Q41", "Q4K", "Q80", "Q81", "Q8K", "BF", "F16", "BPE", "CR_K");
    println!("  {}", "-".repeat(120));

    for (i, ls) in all_levels.iter().enumerate() {
        let pct = |idx: usize| -> f64 {
            if ls.k_total > 0 {
                ls.k_counts[idx] as f64 / ls.k_total as f64 * 100.0
            } else {
                0.0
            }
        };
        let k_bpe = if ls.k_total > 0 {
            ls.k_bpe_sum / ls.k_total as f64
        } else {
            16.0
        };
        let k_cr = 16.0 / k_bpe;
        let mut line = format!("  C{i:<3} {:>10.6} ", k_qproj_thresholds[i]);
        for j in 0..16 {
            let p = pct(j);
            if p < 0.5 {
                line += &format!(" {:>4}", "—");
            } else {
                line += &format!(" {:>4.0}", p);
            }
        }
        line += &format!("  {:>6.3} {:>5.2}x", k_bpe, k_cr);
        println!("{line}");
    }

    // ── Table 2: V cache format distribution ────────────────────────────────
    println!("\n--- V cache format distribution (%) ---");
    println!("  {:<5} {:>10}  {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4} {:>4}  {:>6} {:>6}",
        "Level", "Thr", "Q0", "Q1S", "Q2S", "Q20", "Q2A", "Q21", "Q30", "Q31", "Q40", "Q41", "Q4K", "Q80", "Q81", "Q8K", "BF", "F16", "BPE", "CR_V");
    println!("  {}", "-".repeat(120));

    for (i, ls) in all_levels.iter().enumerate() {
        let pct = |idx: usize| -> f64 {
            if ls.v_total > 0 {
                ls.v_counts[idx] as f64 / ls.v_total as f64 * 100.0
            } else {
                0.0
            }
        };
        let v_bpe = if ls.v_total > 0 {
            ls.v_bpe_sum / ls.v_total as f64
        } else {
            16.0
        };
        let v_cr = 16.0 / v_bpe;
        let mut line = format!("  C{i:<3} {:>10.6} ", v_auto_thresholds[i]);
        for j in 0..16 {
            let p = pct(j);
            if p < 0.5 {
                line += &format!(" {:>4}", "—");
            } else {
                line += &format!(" {:>4.0}", p);
            }
        }
        line += &format!("  {:>6.3} {:>5.2}x", v_bpe, v_cr);
        println!("{line}");
    }

    // ── Table 3: Combined compression ratios with SNR ─────────────────────
    println!("\n--- Compression ratio and SNR summary ---");
    println!(
        "  {:<5}  {:>10}  {:>7}  {:>7}  {:>7}  {:>7}  {:>7}  {:>8}  {:>8}  {:>8}",
        "Level", "K_Thr", "K_BPE", "V_BPE", "Avg_BPE", "CR", "vs_F16", "K_SNR", "V_SNR", "Avg_SNR"
    );
    println!("  {}", "-".repeat(96));

    for (i, ls) in all_levels.iter().enumerate() {
        let k_bpe = if ls.k_total > 0 {
            ls.k_bpe_sum / ls.k_total as f64
        } else {
            16.0
        };
        let v_bpe = if ls.v_total > 0 {
            ls.v_bpe_sum / ls.v_total as f64
        } else {
            16.0
        };
        let avg_bpe = (k_bpe + v_bpe) / 2.0;
        let cr = 16.0 / avg_bpe;
        let savings = (1.0 - avg_bpe / 16.0) * 100.0;
        // K SNR: average over blocks with measurable error; zero-error blocks are ∞ dB
        let k_snr_str = if ls.k_snr_count > 0 {
            format!("{:.1} dB", ls.k_snr_sum / ls.k_snr_count as f64)
        } else {
            "∞ dB".to_string()
        };
        let v_snr_str = if ls.v_snr_count > 0 {
            format!("{:.1} dB", ls.v_snr_sum / ls.v_snr_count as f64)
        } else {
            "∞ dB".to_string()
        };
        // Combined average SNR (weighted by block count)
        let total_snr_count = ls.k_snr_count + ls.v_snr_count;
        let avg_snr_str = if total_snr_count > 0 {
            let avg_snr = (ls.k_snr_sum + ls.v_snr_sum) / total_snr_count as f64;
            format!("{:.1} dB", avg_snr)
        } else {
            "∞ dB".to_string()
        };
        println!("  C{i:<3}  {:>10.6}  {:>7.3}  {:>7.3}  {:>7.3}  {:>6.2}x  {:>5.1}%  {:>8}  {:>8}  {:>8}",
            k_qproj_thresholds[i], k_bpe, v_bpe, avg_bpe, cr, savings, k_snr_str, v_snr_str, avg_snr_str);
    }

    // Print SNR detail: how many blocks had zero error (infinite SNR)
    println!("\n--- SNR and relevance-skip detail ---");
    println!(
        "  {:<5}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}  {:>8}",
        "Level", "K_inf_SNR", "K_finite", "V_inf_SNR", "V_finite", "K_lo_%", "K_hi_%"
    );
    println!("  {}", "-".repeat(72));
    for (i, ls) in all_levels.iter().enumerate() {
        let lo_pct = ls.k_skip_lo as f64 / ls.k_total.max(1) as f64 * 100.0;
        let hi_pct = ls.k_skip_hi as f64 / ls.k_total.max(1) as f64 * 100.0;
        println!(
            "  C{i:<3}  {:>10}  {:>10}  {:>10}  {:>10}  {:>7.1}%  {:>7.1}%",
            ls.k_zero_err, ls.k_snr_count, ls.v_zero_err, ls.v_snr_count, lo_pct, hi_pct
        );
    }

    // ── Table 4: Compression ratios per format ──────────────────────────────
    println!("\n--- Bits per element and compression ratio per format ---");
    let fmts = [
        BlockFormat::Q2_0,
        BlockFormat::Q3_0,
        BlockFormat::Q4_0,
        BlockFormat::Q4_1,
        BlockFormat::Q4_KS,
        BlockFormat::Q8_0,
        BlockFormat::Q8_1,
        BlockFormat::Q8_KS,
        BlockFormat::BF16,
        BlockFormat::F16,
    ];
    println!("  {:<6}  {:>5}  {:>6}", "Format", "BPE", "CR");
    println!("  {}", "-".repeat(20));
    for fmt in &fmts {
        let bpe = fmt.bits_per_elem();
        let cr = 16.0 / bpe;
        println!("  {:<6}  {:>5.1}  {:>5.2}x", fmt, bpe, cr);
    }

    // ── Pasteable threshold constants ───────────────────────────────────────
    println!("\n{sep}");
    println!("// Auto-derived from Q-projected error CDFs:");
    println!("const K_QPROJ_THRESHOLDS: [f32; 10] = [");
    for i in 0..10 {
        println!(
            "    {:.6}, // C{i} — {}",
            k_qproj_thresholds[i], derivation[i]
        );
    }
    println!("];");

    println!("\n{sep}");
    println!("=== Q-Subspace K Format Selection Complete ===");
}
