//! Faithful CPU/Rust mirror of the CUDA selection kernel
//! `select_kv_format_palette4_paged` (see
//! `candle-kernels/src/quantize/select_kv_format.cuh`).
//!
//! This is a debugging/investigation tool: it reproduces the per-(chunk, head)
//! palette-4 selection algorithm in plain Rust so we can A/B against the CUDA
//! implementation when chasing chunk/V anomalies. It is intentionally NOT
//! plumbed into the live compress path.

#![allow(dead_code)]
//!
//! The algorithm is broken into phase-named helpers so each piece is testable
//! in isolation:
//!
//!   * [`per_block_amax_qrel`]               — Phase 1
//!   * [`q_relevance_quantiles`]             — quantile pass mirror
//!   * [`sink_threshold`]                    — Phase 2.5 (V-side)
//!   * [`sort_amax_desc`]                    — Phase 2
//!   * [`k_threshold_per_block`]             — Phase 3
//!   * [`compact_alive`]                     — slot compaction
//!   * [`slot_stats`]                        — per-slot p95/p80/mean/p25
//!   * [`preferred_range_outer`]             — six outer-scale candidates
//!   * [`k_threshold_scaled`]                — per-block K threshold formula
//!   * [`roundtrip_block`]                   — quant/dequant dispatch
//!   * [`search_scales_for_fmt`]             — per-fmt scale search
//!   * [`process_side`]                      — Phase 4+5 driver
//!   * [`select_palette4`]                   — public entry point
//!
//! Deviations from the CUDA kernel (audited 2026-05-28):
//!
//! * Sort tiebreak: Rust uses stable `sort_by` with idx-ascending tiebreak.
//!   The CUDA bitonic network is deterministic but its tie ordering depends
//!   on swap topology — it is not generally idx-ascending. This mirror is
//!   used to A/B against the kernel on *distinct-amax* inputs (the common
//!   case for real K/V); the tiebreak is deterministic in both, which is
//!   all the downstream slot search depends on.
//! * `qrel = 1.0` for blocks with no Q lane: matches the kernel exactly
//!   (`block_relevance` is only invoked under `has_q`). Documented in
//!   [`per_block_amax_qrel`].
//! * The kernel collapses to one outer-scale candidate for formats where
//!   `outer` cancels in the round-trip (Q4_0/1, Q5_0/1, Q8_0/1, Q2_0/1,
//!   Q3_0/1, R16). This mirror runs all six for every format — slower but
//!   exactly equivalent on those formats, and faithful on Q4_KS/Q8_KS and
//!   the INT8-scale / Q0-family formats where `outer` does not cancel.

use candle::quantized::k_quants::{
    BlockQ0, BlockQ0M2, BlockQ0M4, BlockQ0V, BlockQ0X, BlockQ1A, BlockQ1S, BlockQ2A, BlockQ2S,
    BlockQ2_0, BlockQ2_1, BlockQ3_0, BlockQ3_1, BlockQ4_0, BlockQ4_1, BlockQ4_KS, BlockQ5_0,
    BlockQ5_1, BlockQ8_0, BlockQ8_KS, BlockR16, GgmlType,
};
use candle::quantized::GgmlDType;
use half::f16;

use crate::kv_cache::QuantFormat;

const CHUNK_SIZE: usize = 32;
const HEAD_DIM: usize = 128;
const N_PALETTE: usize = 4;
const SLOT_QUOTA: usize = HEAD_DIM / N_PALETTE;
const NUM_SCALE_CANDIDATES: usize = 6;

/// Inputs to the full selection pipeline. Layout matches the CUDA kernel's
/// view of the arena (chunk × head × dim × token, row-major in that order).
pub struct SelectionInput<'a> {
    pub k_data: &'a [f32],
    pub v_data: &'a [f32],
    pub q_data: &'a [u16],
    pub k_candidates: &'a [QuantFormat],
    pub v_candidates: &'a [QuantFormat],
    pub k_threshold_hi: f32,
    pub k_threshold_lo: f32,
    pub v_threshold_hi: f32,
    pub v_threshold_lo: f32,
    pub n_chunks: usize,
    pub n_kv_head: usize,
    pub head_dim: usize,
}

/// Output for one (chunk, head). Mirrors the per-head writes of the CUDA
/// kernel (palette tags / scales / assignments / head amax).
#[derive(Debug, Clone)]
pub struct PerHeadSelection {
    pub k_pal_format: [QuantFormat; N_PALETTE],
    pub v_pal_format: [QuantFormat; N_PALETTE],
    pub k_pal_scale: [f32; N_PALETTE],
    pub v_pal_scale: [f32; N_PALETTE],
    pub k_assignments: [u8; HEAD_DIM],
    pub v_assignments: [u8; HEAD_DIM],
    pub k_head_amax: f32,
    pub v_head_amax: f32,
}

pub struct SelectionOutput {
    pub heads: Vec<PerHeadSelection>,
}

/// Public entry point. `head_id = chunk * n_kv_head + head_idx` keys into
/// `output.heads`.
pub fn select_palette4(input: SelectionInput<'_>) -> SelectionOutput {
    assert_eq!(input.head_dim, HEAD_DIM, "head_dim must be 128");
    let total_heads = input.n_chunks * input.n_kv_head;
    let per_head_floats = HEAD_DIM * CHUNK_SIZE;
    let expected = total_heads * per_head_floats;
    assert_eq!(input.k_data.len(), expected, "k_data length mismatch");
    assert_eq!(input.v_data.len(), expected, "v_data length mismatch");
    assert_eq!(input.q_data.len(), expected, "q_data length mismatch");

    let mut heads = Vec::with_capacity(total_heads);
    for c in 0..input.n_chunks {
        for h in 0..input.n_kv_head {
            let base = (c * input.n_kv_head + h) * per_head_floats;
            let k_block = &input.k_data[base..base + per_head_floats];
            let v_block = &input.v_data[base..base + per_head_floats];
            let q_block = &input.q_data[base..base + per_head_floats];
            heads.push(select_one_head(
                k_block,
                v_block,
                q_block,
                input.k_candidates,
                input.v_candidates,
                input.k_threshold_hi,
                input.k_threshold_lo,
                input.v_threshold_hi,
                input.v_threshold_lo,
            ));
        }
    }
    SelectionOutput { heads }
}

/// Deterministic per-block hash perturbation added to amax during Phase 1
/// to break sort ties. Magnitude lives in the `2^-24 .. 2^-23` decade
/// (~6e-8), far below any real activation amax. Matches the CUDA kernel
/// jitter at `select_kv_format.cuh` Phase 1 byte-for-byte (Knuth golden-ratio
/// hash, xor mix, then construct a small float by replacing the exponent).
#[inline]
pub fn amax_tie_jitter(block_idx: usize) -> f32 {
    let h = (block_idx as u32).wrapping_mul(2654435761u32) ^ 0x9e3779b9u32;
    f32::from_bits((h & 0x007fffffu32) | 0x33800000u32)
}

/// Per-block stats produced by Phase 1: per-(dim) max(|K|), max(|V|), per-block
/// q-relevance (`Σq²k² / Σk²`), and per-block mean of Q across the 32 tokens.
#[derive(Debug, Clone)]
pub struct PerBlockStats {
    pub amax_k: [f32; HEAD_DIM],
    pub amax_v: [f32; HEAD_DIM],
    pub qrel_k: [f32; HEAD_DIM],
    pub q_mean: [f32; HEAD_DIM],
    pub k_head_amax: f32,
    pub v_head_amax: f32,
}

/// Phase 1: per-block amax / q-relevance / Q-mean over (dim, token).
///
/// `k_block[d * 32 + t]` is K at (dim d, token t); same for V and Q. Q is
/// f16-encoded as a u16 bit-pattern (matches the in-arena R16 layout).
///
/// `qrel_k[b] = 1.0` when no Q lane in the block is non-zero — exactly
/// matching the kernel's `has_q == false` short-circuit (`block_relevance`
/// is only invoked when `has_q`). The K threshold path then sees a flat
/// relevance signal and the per-block kthresh collapses toward the
/// geometric mean `sqrt(lo·hi)` (no z-scoring kicks in because
/// `q_relevance_quantiles` returns spread = 0 when no block has Q).
pub fn per_block_amax_qrel(k_block: &[f32], v_block: &[f32], q_block_f16: &[u16]) -> PerBlockStats {
    debug_assert_eq!(k_block.len(), HEAD_DIM * CHUNK_SIZE);
    debug_assert_eq!(v_block.len(), HEAD_DIM * CHUNK_SIZE);
    debug_assert_eq!(q_block_f16.len(), HEAD_DIM * CHUNK_SIZE);

    let mut amax_k = [0.0f32; HEAD_DIM];
    let mut amax_v = [0.0f32; HEAD_DIM];
    let mut qrel_k = [1.0f32; HEAD_DIM];
    let mut q_mean = [0.0f32; HEAD_DIM];
    let mut k_head_amax = 0.0f32;
    let mut v_head_amax = 0.0f32;

    for d in 0..HEAD_DIM {
        let base = d * CHUNK_SIZE;
        let mut ak = 0.0f32;
        let mut av = 0.0f32;
        let mut qsum = 0.0f32;
        let mut qk2_sum = 0.0f32;
        let mut k2_sum = 0.0f32;
        let mut any_q = false;
        for t in 0..CHUNK_SIZE {
            let kv = k_block[base + t];
            let vv = v_block[base + t];
            let qv = f16::from_bits(q_block_f16[base + t]).to_f32();
            ak = ak.max(kv.abs());
            av = av.max(vv.abs());
            qsum += qv;
            let k2 = kv * kv;
            qk2_sum += qv * qv * k2;
            k2_sum += k2;
            if qv != 0.0 {
                any_q = true;
            }
        }
        // Per-block hash jitter (~5e-8 magnitude) to break sort ties on
        // tied amax. Without this, partial-tail chunks whose unused
        // token positions are zero-padded produce many near-equal small
        // amax values; the bitonic sort then drifts toward
        // near-monotonic block-index order and the per-palette claim
        // phase ends up assigning long contiguous dim ranges to one
        // palette (the chunk-1/tail clustering we observed in the
        // dump). The jitter is far below any real amax magnitude
        // (typical V activations live around 1e-2 to 1e1), so format
        // search and partition selection on real data are unaffected.
        amax_k[d] = ak + amax_tie_jitter(d);
        amax_v[d] = av + amax_tie_jitter(d);
        q_mean[d] = qsum * (1.0 / CHUNK_SIZE as f32);
        qrel_k[d] = if any_q {
            if k2_sum == 0.0 {
                0.0
            } else {
                qk2_sum / k2_sum
            }
        } else {
            1.0
        };
        k_head_amax = k_head_amax.max(ak);
        v_head_amax = v_head_amax.max(av);
    }

    PerBlockStats {
        amax_k,
        amax_v,
        qrel_k,
        q_mean,
        k_head_amax,
        v_head_amax,
    }
}

/// Per-(chunk, head) q-relevance median and spread.
///
/// Mirrors `approximate_q_relevance_quantiles`: 64-bin histogram of
/// per-block q-relevance values restricted to blocks that actually carry Q
/// (any non-zero Q lane). Spread = 2 · 1.4426950408889634 · IQR. When fewer
/// than 2 samples or zero range, both outputs are 0 — exactly matching the
/// kernel's early-out path (later code substitutes `sqrt(lo·hi)`).
pub fn q_relevance_quantiles(k_block: &[f32], q_block_f16: &[u16]) -> (f32, f32) {
    debug_assert_eq!(k_block.len(), HEAD_DIM * CHUNK_SIZE);
    debug_assert_eq!(q_block_f16.len(), HEAD_DIM * CHUNK_SIZE);

    const HIST_BINS: usize = 64;
    let mut samples = Vec::with_capacity(HEAD_DIM);
    for d in 0..HEAD_DIM {
        let base = d * CHUNK_SIZE;
        let mut qk2_sum = 0.0f32;
        let mut k2_sum = 0.0f32;
        let mut any_q = false;
        for t in 0..CHUNK_SIZE {
            let kv = k_block[base + t];
            let qv = f16::from_bits(q_block_f16[base + t]).to_f32();
            if qv != 0.0 {
                any_q = true;
            }
            qk2_sum += qv * qv * kv * kv;
            k2_sum += kv * kv;
        }
        if any_q {
            let qr = if k2_sum == 0.0 { 0.0 } else { qk2_sum / k2_sum };
            samples.push(qr);
        }
    }

    if samples.len() <= 1 {
        return (0.0, 0.0);
    }
    let mut head_min = samples[0];
    let mut head_max = samples[0];
    for &s in &samples[1..] {
        if s < head_min {
            head_min = s;
        }
        if s > head_max {
            head_max = s;
        }
    }
    if !(head_max > head_min) || head_max <= 0.0 {
        return (0.0, 0.0);
    }

    let inv_range = (HIST_BINS as f32 - 1.0) / (head_max - head_min);
    let mut hist = [0i32; HIST_BINS];
    for &s in &samples {
        let mut bin = ((s - head_min) * inv_range).floor() as i32;
        if bin < 0 {
            bin = 0;
        }
        if bin >= HIST_BINS as i32 {
            bin = HIST_BINS as i32 - 1;
        }
        hist[bin as usize] += 1;
    }

    let sample_count = samples.len() as i32;
    let target_q1 = (0.25 * (sample_count - 1) as f32).floor() as i32;
    let target_median = (0.50 * (sample_count - 1) as f32).floor() as i32;
    let target_q3 = (0.75 * (sample_count - 1) as f32).floor() as i32;

    let mut accum = 0;
    let mut q1_bin = 0i32;
    let mut median_bin = 0i32;
    let mut q3_bin = HIST_BINS as i32 - 1;
    let mut found_q1 = false;
    let mut found_m = false;
    for b in 0..HIST_BINS {
        accum += hist[b];
        if !found_q1 && accum > target_q1 {
            q1_bin = b as i32;
            found_q1 = true;
        }
        if !found_m && accum > target_median {
            median_bin = b as i32;
            found_m = true;
        }
        if accum > target_q3 {
            q3_bin = b as i32;
            break;
        }
    }
    let scale = (head_max - head_min) / (HIST_BINS as f32 - 1.0);
    let q1_v = head_min + q1_bin as f32 * scale;
    let median_v = head_min + median_bin as f32 * scale;
    let q3_v = head_min + q3_bin as f32 * scale;
    let iqr = (q3_v - q1_v).max(1.0e-8);
    let spread = iqr * 2.0 * std::f32::consts::LOG2_E;
    (median_v, spread)
}

/// Phase 2.5 sink statistics — sink_weight max, and the V-side squared
/// effective threshold the search loop compares against.
#[derive(Debug, Clone, Copy)]
pub struct SinkStats {
    pub sink_weight_max: f32,
    pub v_thr_sq: f32,
}

/// Phase 2.5: per-token sink scores → tanh-clamped sink weight → max →
/// effective V threshold squared.
pub fn sink_threshold(
    k_block: &[f32],
    q_mean: &[f32; HEAD_DIM],
    v_threshold_lo: f32,
    v_threshold_hi: f32,
) -> SinkStats {
    debug_assert_eq!(k_block.len(), HEAD_DIM * CHUNK_SIZE);

    let mut sink_score = [0.0f32; CHUNK_SIZE];
    for t in 0..CHUNK_SIZE {
        let mut s = 0.0f32;
        for d in 0..HEAD_DIM {
            s += q_mean[d] * k_block[d * CHUNK_SIZE + t];
        }
        sink_score[t] = s * (1.0f32 / (HEAD_DIM as f32).sqrt());
    }

    let mut mu = 0.0f32;
    for &s in &sink_score {
        mu += s;
    }
    mu /= CHUNK_SIZE as f32;
    let mut var = 0.0f32;
    for &s in &sink_score {
        let d = s - mu;
        var += d * d;
    }
    var /= CHUNK_SIZE as f32;
    let sigma = var.sqrt();
    let safe_sigma = sigma.max(1.0e-8);

    let mut w_max = 0.0f32;
    for &s in &sink_score {
        let z = (s - mu) / safe_sigma;
        let w = 0.0f32.max(z.tanh());
        if w > w_max {
            w_max = w;
        }
    }

    let v_thr_eff = v_threshold_lo + w_max * (v_threshold_hi - v_threshold_lo);
    SinkStats {
        sink_weight_max: w_max,
        v_thr_sq: v_thr_eff * v_thr_eff,
    }
}

/// Phase 2: sort 128 blocks by amax descending (tie-break by original index
/// ascending). Returns the sorted block-index sequence — `idx[0]` is the
/// block with largest amax.
///
/// Deviation: the CUDA bitonic network does not guarantee idx-ascending
/// tiebreak (it is deterministic but topology-defined). On distinct-amax
/// inputs both implementations produce the same order; on tied inputs we
/// produce a deterministic-but-not-kernel-identical order. This is
/// acceptable for the chunk-1 V debugging use case because the slot
/// search downstream only depends on amax-descending order, not on the
/// specific tiebreak.
pub fn sort_amax_desc(amax: &[f32; HEAD_DIM]) -> [u16; HEAD_DIM] {
    let mut idx: [u16; HEAD_DIM] = std::array::from_fn(|i| i as u16);
    idx.sort_by(|&a, &b| {
        let av = amax[a as usize];
        let bv = amax[b as usize];
        let prim = bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal);
        if prim == std::cmp::Ordering::Equal {
            a.cmp(&b)
        } else {
            prim
        }
    });
    idx
}

/// Phase 3: per-block K threshold.
///
/// `kthresh[b] = k_threshold_scaled(qrel_k[b], q_med, q_spread)` if
/// `q_spread > 1e-8`, otherwise `sqrt(lo * hi)`.
pub fn k_threshold_per_block(
    qrel_k: &[f32; HEAD_DIM],
    q_med: f32,
    q_spread: f32,
    k_threshold_lo: f32,
    k_threshold_hi: f32,
) -> [f32; HEAD_DIM] {
    let mut out = [0.0f32; HEAD_DIM];
    let fallback = (k_threshold_lo * k_threshold_hi).sqrt();
    let usable = q_spread > 1.0e-8;
    for b in 0..HEAD_DIM {
        out[b] = if usable {
            k_threshold_scaled(k_threshold_lo, k_threshold_hi, qrel_k[b], q_med, q_spread)
        } else {
            fallback
        };
    }
    out
}

/// Per-block K threshold formula.
///
/// ```text
/// z          = (qrel − q_median) / max(q_spread, 1e-8)
/// multiplier = exp(−z)
/// scaled     = sqrt(lo · hi) · multiplier
/// out        = clamp(scaled, hi, lo)
/// ```
///
/// If `q_relevance` or `q_median` is NaN the result would be NaN under
/// pure IEEE arithmetic; we substitute the geometric-mean fallback so
/// downstream comparisons stay well-behaved (the kernel's f16 round-trip
/// for `qrel_k` masks NaN to a deterministic bit-pattern; this guard
/// matches the spirit but not the bit-pattern of that masking).
pub fn k_threshold_scaled(
    threshold_lo: f32,
    threshold_hi: f32,
    q_relevance: f32,
    q_median: f32,
    q_spread: f32,
) -> f32 {
    let base = (threshold_lo * threshold_hi).sqrt();
    if !q_relevance.is_finite() || !q_median.is_finite() {
        return threshold_hi.max(threshold_lo.min(base));
    }
    let safe_spread = q_spread.max(1.0e-8);
    let z = (q_relevance - q_median) / safe_spread;
    let multiplier = (-z).exp();
    let scaled = base * multiplier;
    threshold_hi.max(threshold_lo.min(scaled))
}

/// Six outer-scale candidates (matches CUDA `preferred_range`).
pub fn preferred_range_outer(
    idx: usize,
    slot_amax: f32,
    safe_p95: f32,
    safe_p80: f32,
    slot_mean: f32,
    safe_p25: f32,
) -> f32 {
    match idx {
        0 => 1.0,
        1 => 1.0 / slot_amax,
        2 => 1.0 / safe_p95,
        3 => 1.0 / safe_p80,
        4 => 1.0 / slot_mean,
        5 => 1.0 / safe_p25,
        _ => 1.0,
    }
}

/// Compact the sort order down to the currently-alive entries.
pub fn compact_alive(sorted: &[u16; HEAD_DIM], alive: &[bool; HEAD_DIM]) -> Vec<u16> {
    let mut out = Vec::with_capacity(HEAD_DIM);
    for &b in sorted.iter() {
        if alive[b as usize] {
            out.push(b);
        }
    }
    out
}

/// Per-slot amax distribution stats. Mirrors `s_slot_*` smem writes.
#[derive(Debug, Clone, Copy)]
pub struct SlotStats {
    pub slot_amax: f32,
    pub safe_p95: f32,
    pub safe_p80: f32,
    pub slot_mean: f32,
    pub safe_p25: f32,
}

/// Slot stats given alive entries in amax-desc order. `amax_sorted` is
/// indexed by sort position; `idx_compact` lists alive block IDs in sort
/// order. Matches the CUDA single-thread stat walk exactly (same target
/// formulae, same fallback to slot_amax_raw when target isn't hit).
pub fn slot_stats(amax: &[f32; HEAD_DIM], idx_compact: &[u16]) -> SlotStats {
    let lc = idx_compact.len();
    if lc == 0 {
        return SlotStats {
            slot_amax: 1.0e-8,
            safe_p95: 1.0e-8,
            safe_p80: 1.0e-8,
            slot_mean: 1.0e-8,
            safe_p25: 1.0e-8,
        };
    }
    let slot_amax_raw = amax[idx_compact[0] as usize];
    let mut sum = 0.0f32;
    let mut p95_val = slot_amax_raw;
    let mut p80_val = slot_amax_raw;
    let mut p25_val = slot_amax_raw;
    let p95_tgt = 1.max((lc + 19) / 20) as i32;
    let p80_tgt = 1.max((lc + 4) / 5) as i32;
    let p25_tgt = 1.max((3 * lc) / 4) as i32;
    let mut cnt = 0i32;
    for &b in idx_compact {
        let av = amax[b as usize];
        sum += av;
        cnt += 1;
        if cnt == p95_tgt {
            p95_val = av;
        }
        if cnt == p80_tgt {
            p80_val = av;
        }
        if cnt == p25_tgt {
            p25_val = av;
        }
    }
    let mean = sum / lc as f32;
    SlotStats {
        slot_amax: slot_amax_raw.max(1.0e-8),
        safe_p95: p95_val.max(1.0e-8),
        safe_p80: p80_val.max(1.0e-8),
        slot_mean: mean.max(1.0e-8),
        safe_p25: p25_val.max(1.0e-8),
    }
}

/// Map QuantFormat to a `GgmlDType` we can dispatch round-trip through.
///
/// Unsupported formats and why:
///
/// * `R16` — K-source carrier format that stores Q activations alongside
///   K values. It is never a *quant target* in the candidate ladders, so
///   its round-trip semantics differ from every other format (the `q[]`
///   field would round-trip as zero on V data).
/// * `Q8_1` — `BlockQ8_1::to_float` is `unimplemented!()` in
///   `candle-core` (Q8_1 is a vec-dot intermediate, not a storage
///   format). Q8_1 appears in production candidate ladders for K-side
///   C1–C7 but its kernel-side round-trip uses a CUDA-only dequant path
///   that has no Rust analogue. Returning `None` here lets the search
///   skip it cleanly.
fn ggml_supported(fmt: QuantFormat) -> Option<GgmlDType> {
    match fmt {
        QuantFormat::R16 | QuantFormat::Q8_1 => None,
        _ => Some(fmt.to_ggml_dtype()),
    }
}

/// Round-trip 32 floats through `fmt` after pre-scaling by `outer`. Returns
/// the dequantized vector pre-divided by `outer` (so the caller compares
/// against the original directly). Returns `None` for formats not mirrored
/// (currently `R16` only — see [`ggml_supported`]).
pub fn roundtrip_block(
    fmt: QuantFormat,
    src: &[f32; CHUNK_SIZE],
    outer: f32,
) -> Option<[f32; CHUNK_SIZE]> {
    let dtype = ggml_supported(fmt)?;
    let mut scaled = [0.0f32; CHUNK_SIZE];
    for i in 0..CHUNK_SIZE {
        scaled[i] = src[i] * outer;
    }
    let mut recon = [0.0f32; CHUNK_SIZE];
    match dtype {
        GgmlDType::Q4_0 => roundtrip_via::<BlockQ4_0>(&scaled, &mut recon),
        GgmlDType::Q4_1 => roundtrip_via::<BlockQ4_1>(&scaled, &mut recon),
        GgmlDType::Q5_0 => roundtrip_via::<BlockQ5_0>(&scaled, &mut recon),
        GgmlDType::Q5_1 => roundtrip_via::<BlockQ5_1>(&scaled, &mut recon),
        GgmlDType::Q8_0 => roundtrip_via::<BlockQ8_0>(&scaled, &mut recon),
        // Q8_1 filtered by ggml_supported — to_float is unimplemented.
        GgmlDType::Q8_1 => return None,
        GgmlDType::Q4_KS => roundtrip_via::<BlockQ4_KS>(&scaled, &mut recon),
        GgmlDType::Q8_KS => roundtrip_via::<BlockQ8_KS>(&scaled, &mut recon),
        GgmlDType::Q2_0 => roundtrip_via::<BlockQ2_0>(&scaled, &mut recon),
        GgmlDType::Q3_0 => roundtrip_via::<BlockQ3_0>(&scaled, &mut recon),
        GgmlDType::Q2_1 => roundtrip_via::<BlockQ2_1>(&scaled, &mut recon),
        GgmlDType::Q3_1 => roundtrip_via::<BlockQ3_1>(&scaled, &mut recon),
        GgmlDType::Q0 => roundtrip_via::<BlockQ0>(&scaled, &mut recon),
        GgmlDType::Q1_S => roundtrip_via::<BlockQ1S>(&scaled, &mut recon),
        GgmlDType::Q2_S => roundtrip_via::<BlockQ2S>(&scaled, &mut recon),
        GgmlDType::Q2_A => roundtrip_via::<BlockQ2A>(&scaled, &mut recon),
        GgmlDType::Q0_V => roundtrip_via::<BlockQ0V>(&scaled, &mut recon),
        GgmlDType::Q1_A => roundtrip_via::<BlockQ1A>(&scaled, &mut recon),
        GgmlDType::Q0_X => roundtrip_via::<BlockQ0X>(&scaled, &mut recon),
        GgmlDType::Q0_M2 => roundtrip_via::<BlockQ0M2>(&scaled, &mut recon),
        GgmlDType::Q0_M4 => roundtrip_via::<BlockQ0M4>(&scaled, &mut recon),
        GgmlDType::R16 => roundtrip_via::<BlockR16>(&scaled, &mut recon),
        _ => return None,
    }
    let inv_outer = if outer != 0.0 { 1.0 / outer } else { 0.0 };
    let mut out = [0.0f32; CHUNK_SIZE];
    for i in 0..CHUNK_SIZE {
        out[i] = recon[i] * inv_outer;
    }
    Some(out)
}

fn roundtrip_via<B: GgmlType>(scaled: &[f32; CHUNK_SIZE], recon: &mut [f32; CHUNK_SIZE]) {
    let mut blk = [B::zeros()];
    B::from_float(scaled, &mut blk);
    B::to_float(&blk, recon);
}

/// Mean of the four largest absolute errors across the 32 lanes.
/// Matches `mean_top4_abs_error_warp` (degenerate-equal case returns the
/// shared value, not zero).
fn mean_top4_abs_error(orig: &[f32; CHUNK_SIZE], recon: &[f32; CHUNK_SIZE]) -> f32 {
    let mut errs = [0.0f32; CHUNK_SIZE];
    for i in 0..CHUNK_SIZE {
        errs[i] = (orig[i] - recon[i]).abs();
    }
    let mut sum = 0.0f32;
    let mut local = errs;
    for _ in 0..4 {
        let mut m = local[0];
        for &v in local.iter() {
            if v > m {
                m = v;
            }
        }
        let mut first_lane = 0usize;
        for (i, &v) in local.iter().enumerate() {
            if v == m {
                first_lane = i;
                break;
            }
        }
        sum += m;
        local[first_lane] = f32::MIN;
    }
    sum * 0.25
}

fn mean_squared_error(orig: &[f32; CHUNK_SIZE], recon: &[f32; CHUNK_SIZE]) -> f32 {
    let mut s = 0.0f32;
    for i in 0..CHUNK_SIZE {
        let d = orig[i] - recon[i];
        s += d * d;
    }
    s * (1.0 / CHUNK_SIZE as f32)
}

/// Per-(fmt, scale) summary recorded during the search. Matches the kernel's
/// per-(warp, si) accumulator state once tid 0 has folded the warp slots.
#[derive(Debug, Clone, Copy)]
pub struct ScaleResult {
    pub fmt: QuantFormat,
    pub outer: f32,
    pub pass_count: usize,
    pub pass_mask: u128,
    /// Max pass_metric across ALL alive blocks (passing AND failing) — the
    /// quantity the kernel's fallback bookkeeping tracks.
    pub max_pass_metric: f32,
}

/// Per-fmt search output: the best quota-hitting (fmt, scale) (if any), and
/// the lowest-aerr (fmt, scale) overall — i.e. the per-fmt contribution to
/// the cross-format fallback. Mirrors the kernel's `s_best_*` / `s_fallback_*`
/// updates: best is conditional on `total >= 32`, fallback is unconditional.
#[derive(Debug, Clone, Copy, Default)]
pub struct PerFmtSearch {
    pub best: Option<ScaleResult>,
    pub fallback: Option<ScaleResult>,
}

/// Per-fmt scale search. Returns:
///
/// * `best`: lowest-aerr (fmt, scale) among scales that hit the 32-block
///   quota (`None` if no scale passes).
/// * `fallback`: lowest-aerr (fmt, scale) overall (always populated if the
///   format dispatches).
///
/// Returns `None` only if the format has no round-trip dispatch (R16).
#[allow(clippy::too_many_arguments)]
pub fn search_scales_for_fmt(
    fmt: QuantFormat,
    is_k: bool,
    data: &[f32],
    idx_compact: &[u16],
    stats: SlotStats,
    kthresh: &[f32; HEAD_DIM],
    inv_head_amax: f32,
    inv_head_amax_sq: f32,
    v_thr_sq: f32,
) -> Option<PerFmtSearch> {
    ggml_supported(fmt)?;
    let mut best: Option<ScaleResult> = None;
    let mut fallback: Option<ScaleResult> = None;
    for si in 0..NUM_SCALE_CANDIDATES {
        let outer = preferred_range_outer(
            si,
            stats.slot_amax,
            stats.safe_p95,
            stats.safe_p80,
            stats.slot_mean,
            stats.safe_p25,
        );
        let mut pass_count = 0usize;
        let mut pass_mask: u128 = 0;
        let mut max_pm = 0.0f32;
        for &b in idx_compact {
            let bu = b as usize;
            let mut orig = [0.0f32; CHUNK_SIZE];
            orig.copy_from_slice(&data[bu * CHUNK_SIZE..(bu + 1) * CHUNK_SIZE]);
            let recon = roundtrip_block(fmt, &orig, outer)?;
            let (pass_metric, thr) = if is_k {
                let e = mean_top4_abs_error(&orig, &recon);
                (e * inv_head_amax, kthresh[bu])
            } else {
                let mse = mean_squared_error(&orig, &recon);
                (mse * inv_head_amax_sq, v_thr_sq)
            };
            if pass_metric > max_pm {
                max_pm = pass_metric;
            }
            if pass_metric <= thr {
                pass_count += 1;
                pass_mask |= 1u128 << bu;
            }
        }
        let cand = ScaleResult {
            fmt,
            outer,
            pass_count,
            pass_mask,
            max_pass_metric: max_pm,
        };
        fallback = Some(match fallback {
            None => cand,
            Some(prev) => {
                if cand.max_pass_metric < prev.max_pass_metric {
                    cand
                } else {
                    prev
                }
            }
        });
        if pass_count >= SLOT_QUOTA {
            best = Some(match best {
                None => cand,
                Some(prev) => {
                    if cand.max_pass_metric < prev.max_pass_metric {
                        cand
                    } else {
                        prev
                    }
                }
            });
        }
    }
    Some(PerFmtSearch { best, fallback })
}

/// Phase 4+5 driver for one side (K or V). Returns per-slot format/scale
/// and per-dim palette assignments.
#[allow(clippy::too_many_arguments)]
pub fn process_side(
    data: &[f32],
    amax: &[f32; HEAD_DIM],
    idx_sorted: &[u16; HEAD_DIM],
    candidates: &[QuantFormat],
    head_amax: f32,
    is_k: bool,
    kthresh: &[f32; HEAD_DIM],
    v_thr_sq: f32,
) -> ([QuantFormat; N_PALETTE], [f32; N_PALETTE], [u8; HEAD_DIM]) {
    let safe_head_amax = head_amax.max(1.0e-8);
    let inv_head_amax = 1.0 / safe_head_amax;
    let inv_head_amax_sq = inv_head_amax * inv_head_amax;
    let fallback_last = *candidates.last().expect("non-empty candidate set");

    let mut alive = [true; HEAD_DIM];
    let mut slot_fmt = [fallback_last; N_PALETTE];
    let mut slot_scale = [1.0f32; N_PALETTE];
    let mut assignments = [255u8; HEAD_DIM];

    for s in 0..N_PALETTE {
        let idx_compact = compact_alive(idx_sorted, &alive);
        if idx_compact.is_empty() {
            slot_fmt[s] = fallback_last;
            slot_scale[s] = 1.0;
            continue;
        }
        let stats = slot_stats(amax, &idx_compact);

        let mut best: Option<ScaleResult> = None;
        let mut fallback: Option<ScaleResult> = None;
        for &fmt in candidates {
            if let Some(b) = best {
                if b.pass_count >= SLOT_QUOTA {
                    break;
                }
            }
            let per_fmt = match search_scales_for_fmt(
                fmt,
                is_k,
                data,
                &idx_compact,
                stats,
                kthresh,
                inv_head_amax,
                inv_head_amax_sq,
                v_thr_sq,
            ) {
                Some(r) => r,
                None => continue,
            };
            if let Some(fb) = per_fmt.fallback {
                fallback = Some(match fallback {
                    None => fb,
                    Some(prev) => {
                        if fb.max_pass_metric < prev.max_pass_metric {
                            fb
                        } else {
                            prev
                        }
                    }
                });
            }
            if let Some(bw) = per_fmt.best {
                best = Some(match best {
                    None => bw,
                    Some(prev) => {
                        if bw.max_pass_metric < prev.max_pass_metric {
                            bw
                        } else {
                            prev
                        }
                    }
                });
            }
        }

        let winner = best.or(fallback);
        let (chosen_fmt, chosen_scale, pass_mask) = match winner {
            Some(w) => (w.fmt, w.outer, w.pass_mask),
            None => (fallback_last, 1.0f32, 0u128),
        };
        slot_fmt[s] = chosen_fmt;
        slot_scale[s] = chosen_scale;

        // Pass 1: claim passing blocks in sort order. Pass 2: fill any
        // remaining quota from the front of the still-alive set. Matches
        // the kernel's claim_passing_blocks_from_mask + warp-0 fill.
        let mut claimed = 0usize;
        for &b in &idx_compact {
            if claimed >= SLOT_QUOTA {
                break;
            }
            let bit = (pass_mask >> b as u32) & 1u128;
            if bit != 0 {
                assignments[b as usize] = s as u8;
                alive[b as usize] = false;
                claimed += 1;
            }
        }
        if claimed < SLOT_QUOTA {
            for &b in &idx_compact {
                if claimed >= SLOT_QUOTA {
                    break;
                }
                if alive[b as usize] {
                    assignments[b as usize] = s as u8;
                    alive[b as usize] = false;
                    claimed += 1;
                }
            }
        }
    }

    for d in 0..HEAD_DIM {
        if assignments[d] == 255 {
            assignments[d] = (N_PALETTE - 1) as u8;
        }
    }
    (slot_fmt, slot_scale, assignments)
}

fn select_one_head(
    k_block: &[f32],
    v_block: &[f32],
    q_block_f16: &[u16],
    k_candidates: &[QuantFormat],
    v_candidates: &[QuantFormat],
    k_threshold_hi: f32,
    k_threshold_lo: f32,
    v_threshold_hi: f32,
    v_threshold_lo: f32,
) -> PerHeadSelection {
    let stats = per_block_amax_qrel(k_block, v_block, q_block_f16);
    let (q_med, q_spread) = q_relevance_quantiles(k_block, q_block_f16);
    let sink = sink_threshold(k_block, &stats.q_mean, v_threshold_lo, v_threshold_hi);
    let kidx = sort_amax_desc(&stats.amax_k);
    let vidx = sort_amax_desc(&stats.amax_v);
    let kthresh = k_threshold_per_block(
        &stats.qrel_k,
        q_med,
        q_spread,
        k_threshold_lo,
        k_threshold_hi,
    );

    let (k_pal_format, k_pal_scale, k_assignments) = process_side(
        k_block,
        &stats.amax_k,
        &kidx,
        k_candidates,
        stats.k_head_amax,
        true,
        &kthresh,
        sink.v_thr_sq,
    );
    let (v_pal_format, v_pal_scale, v_assignments) = process_side(
        v_block,
        &stats.amax_v,
        &vidx,
        v_candidates,
        stats.v_head_amax,
        false,
        &kthresh,
        sink.v_thr_sq,
    );

    PerHeadSelection {
        k_pal_format,
        v_pal_format,
        k_pal_scale,
        v_pal_scale,
        k_assignments,
        v_assignments,
        k_head_amax: stats.k_head_amax,
        v_head_amax: stats.v_head_amax,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::chunked::sampled_selection::params::{
        production_adaptive_candidates, PRODUCTION_K_QREL_HIGH_THRESHOLDS,
        PRODUCTION_K_QREL_LOW_THRESHOLDS, PRODUCTION_V_QREL_HIGH_THRESHOLDS,
        PRODUCTION_V_QREL_LOW_THRESHOLDS,
    };
    use crate::kv_cache::KvFormat;

    fn make_block(values: impl Fn(usize, usize) -> f32) -> Vec<f32> {
        let mut buf = vec![0.0f32; HEAD_DIM * CHUNK_SIZE];
        for d in 0..HEAD_DIM {
            for t in 0..CHUNK_SIZE {
                buf[d * CHUNK_SIZE + t] = values(d, t);
            }
        }
        buf
    }

    fn deterministic_uniform(seed: u64, idx: usize) -> f32 {
        let mut x = seed
            .wrapping_mul(0x9E3779B97F4A7C15)
            .wrapping_add(idx as u64);
        x ^= x >> 30;
        x = x.wrapping_mul(0xBF58476D1CE4E5B9);
        x ^= x >> 27;
        x = x.wrapping_mul(0x94D049BB133111EB);
        x ^= x >> 31;
        let mantissa = (x & 0x7FFF_FFFF) as f32 / 2147483647.0;
        mantissa * 2.0 - 1.0
    }

    /// Box-Muller from two uniforms → ~ N(0, 1).
    fn deterministic_gauss(seed: u64, idx: usize) -> f32 {
        let u1 = (deterministic_uniform(seed, idx * 2) + 1.0) * 0.5;
        let u2 = (deterministic_uniform(seed, idx * 2 + 1) + 1.0) * 0.5;
        let u1 = u1.clamp(1e-7, 1.0 - 1e-7);
        let r = (-2.0_f32 * u1.ln()).sqrt();
        let th = 2.0 * std::f32::consts::PI * u2;
        r * th.cos()
    }

    fn quant_from_kv(kv: KvFormat) -> QuantFormat {
        match kv {
            KvFormat::Quantized(q) => q,
            KvFormat::Float(_) => panic!("expected quantized format"),
        }
    }

    #[test]
    fn amax_tie_jitter_breaks_zero_block_ties() {
        // Pure zero K and V — every block's true amax is 0; only the
        // jitter survives. The sort must then produce a non-monotonic
        // order; without jitter it would be the bitonic network's
        // tie-break which biases toward original-index order.
        let k = vec![0.0f32; HEAD_DIM * CHUNK_SIZE];
        let v = vec![0.0f32; HEAD_DIM * CHUNK_SIZE];
        let q = vec![0u16; HEAD_DIM * CHUNK_SIZE];
        let stats = per_block_amax_qrel(&k, &v, &q);
        // Every block must be strictly positive (jitter is in 2^-24..2^-23)
        // and pairwise distinct, so sort_amax_desc gives a unique permutation.
        let mut seen = std::collections::HashSet::new();
        for d in 0..HEAD_DIM {
            assert!(stats.amax_v[d] > 0.0, "jitter zero at d={d}");
            assert!(
                stats.amax_v[d] < 1e-6,
                "jitter too large at d={d}: {}",
                stats.amax_v[d]
            );
            assert!(
                seen.insert(stats.amax_v[d].to_bits()),
                "duplicate jitter at d={d}"
            );
        }
        let order = sort_amax_desc(&stats.amax_v);
        // The sorted indices must not be the identity permutation (which
        // is the failure mode the jitter exists to prevent).
        let identity_count = (0..HEAD_DIM).filter(|&i| order[i] as usize == i).count();
        assert!(
            identity_count < HEAD_DIM / 2,
            "post-jitter sort still near-identity: {identity_count}/{HEAD_DIM} fixed points"
        );
    }

    #[test]
    fn amax_tie_jitter_negligible_vs_real_amax() {
        // Jitter must be small enough that it doesn't perturb real amax
        // values meaningfully — anything ≥ 1e-4 magnitude should win
        // unchanged. Confirms the format-search path is unaffected by
        // the tie-breaker on real data.
        for b in 0..HEAD_DIM {
            assert!(amax_tie_jitter(b) < 1e-6);
            assert!(amax_tie_jitter(b) > 0.0);
        }
    }

    #[test]
    fn per_block_amax_known_values() {
        let k = make_block(|d, t| (d as f32 + 1.0) * t as f32);
        let v = make_block(|d, t| -(d as f32 + 1.0) * t as f32);
        let q = vec![0u16; HEAD_DIM * CHUNK_SIZE];
        let stats = per_block_amax_qrel(&k, &v, &q);
        for d in 0..HEAD_DIM {
            let expected = (d as f32 + 1.0) * 31.0;
            assert!(
                (stats.amax_k[d] - expected).abs() < 1e-4,
                "K amax mismatch at d={d}: got {}, want {}",
                stats.amax_k[d],
                expected
            );
            assert!(
                (stats.amax_v[d] - expected).abs() < 1e-4,
                "V amax mismatch at d={d}: got {}, want {}",
                stats.amax_v[d],
                expected
            );
        }
        let expected_head = 128.0 * 31.0;
        assert!((stats.k_head_amax - expected_head).abs() < 1e-3);
        assert!((stats.v_head_amax - expected_head).abs() < 1e-3);
    }

    #[test]
    fn sort_desc_with_tiebreak() {
        let mut a = [0.0f32; HEAD_DIM];
        for d in 0..HEAD_DIM {
            a[d] = (d / 2) as f32;
        }
        let idx = sort_amax_desc(&a);
        assert_eq!(idx[0], 126);
        assert_eq!(idx[1], 127);
        assert_eq!(idx[2], 124);
        assert_eq!(idx[3], 125);
        assert_eq!(idx[HEAD_DIM - 2], 0);
        assert_eq!(idx[HEAD_DIM - 1], 1);
    }

    #[test]
    fn slot_stats_quantiles() {
        let mut a = [0.0f32; HEAD_DIM];
        for d in 0..HEAD_DIM {
            a[d] = (128 - d) as f32;
        }
        let idx = sort_amax_desc(&a);
        let alive = [true; HEAD_DIM];
        let compact = compact_alive(&idx, &alive);
        let s = slot_stats(&a, &compact);
        assert_eq!(s.slot_amax, 128.0);
        assert_eq!(s.safe_p95, 122.0);
        assert_eq!(s.safe_p80, 103.0);
        assert_eq!(s.safe_p25, 33.0);
        assert!((s.slot_mean - 64.5).abs() < 1e-3);
    }

    #[test]
    fn preferred_range_idx_table() {
        let amax = 10.0;
        let p95 = 8.0;
        let p80 = 5.0;
        let mean = 4.0;
        let p25 = 2.0;
        assert_eq!(preferred_range_outer(0, amax, p95, p80, mean, p25), 1.0);
        assert!((preferred_range_outer(1, amax, p95, p80, mean, p25) - 0.1).abs() < 1e-7);
        assert!((preferred_range_outer(2, amax, p95, p80, mean, p25) - 0.125).abs() < 1e-7);
        assert!((preferred_range_outer(3, amax, p95, p80, mean, p25) - 0.2).abs() < 1e-7);
        assert!((preferred_range_outer(4, amax, p95, p80, mean, p25) - 0.25).abs() < 1e-7);
        assert!((preferred_range_outer(5, amax, p95, p80, mean, p25) - 0.5).abs() < 1e-7);
    }

    #[test]
    fn roundtrip_quantize_dispatch_q4_0() {
        let mut src = [0.0f32; CHUNK_SIZE];
        for i in 0..CHUNK_SIZE {
            src[i] = (i as f32 - 16.0) * 0.1;
        }
        let out = roundtrip_block(QuantFormat::Q4_0, &src, 1.0).unwrap();
        for i in 0..CHUNK_SIZE {
            let e = (src[i] - out[i]).abs();
            assert!(e < 0.25, "Q4_0 element {i} error {e} too large");
        }
    }

    #[test]
    fn roundtrip_quantize_dispatch_q8_0() {
        let mut src = [0.0f32; CHUNK_SIZE];
        for i in 0..CHUNK_SIZE {
            src[i] = (i as f32 - 16.0) * 0.1;
        }
        let out = roundtrip_block(QuantFormat::Q8_0, &src, 1.0).unwrap();
        for i in 0..CHUNK_SIZE {
            let e = (src[i] - out[i]).abs();
            assert!(e < 0.02, "Q8_0 element {i} error {e} too large");
        }
    }

    fn ramp_src() -> [f32; CHUNK_SIZE] {
        let mut src = [0.0f32; CHUNK_SIZE];
        for i in 0..CHUNK_SIZE {
            src[i] = (i as f32 - 16.0) * 0.05;
        }
        src
    }

    /// Per-format reconstruction-error tolerance. These are loose ceilings
    /// chosen to detect dispatch wiring bugs, not to assert codec quality.
    fn fmt_error_bound(fmt: QuantFormat) -> f32 {
        match fmt {
            QuantFormat::Q8_0 | QuantFormat::Q8_1 | QuantFormat::Q8_KS => 0.05,
            QuantFormat::Q5_0 | QuantFormat::Q5_1 => 0.1,
            QuantFormat::Q4_0 | QuantFormat::Q4_1 | QuantFormat::Q4_KS => 0.25,
            QuantFormat::Q3_0 | QuantFormat::Q3_1 => 0.4,
            QuantFormat::Q2_0 | QuantFormat::Q2_1 | QuantFormat::Q2_S | QuantFormat::Q2_A => 0.8,
            QuantFormat::Q1_S | QuantFormat::Q1_A => 1.6,
            QuantFormat::Q0
            | QuantFormat::Q0_V
            | QuantFormat::Q0_X
            | QuantFormat::Q0_M2
            | QuantFormat::Q0_M4 => 2.0,
            QuantFormat::R16 => 0.01,
        }
    }

    fn run_roundtrip(fmt: QuantFormat) {
        let src = ramp_src();
        let out = roundtrip_block(fmt, &src, 1.0)
            .unwrap_or_else(|| panic!("{:?} dispatch returned None", fmt));
        let bound = fmt_error_bound(fmt);
        let mut max_err = 0.0f32;
        for i in 0..CHUNK_SIZE {
            let e = (src[i] - out[i]).abs();
            if e > max_err {
                max_err = e;
            }
        }
        assert!(
            max_err < bound,
            "{:?} max error {} exceeds bound {}",
            fmt,
            max_err,
            bound
        );
    }

    #[test]
    fn roundtrip_q4_1() {
        run_roundtrip(QuantFormat::Q4_1);
    }
    #[test]
    fn roundtrip_q5_0() {
        run_roundtrip(QuantFormat::Q5_0);
    }
    #[test]
    fn roundtrip_q5_1() {
        run_roundtrip(QuantFormat::Q5_1);
    }
    #[test]
    fn roundtrip_q4_ks() {
        run_roundtrip(QuantFormat::Q4_KS);
    }
    #[test]
    fn roundtrip_q8_ks() {
        run_roundtrip(QuantFormat::Q8_KS);
    }
    #[test]
    fn roundtrip_q2_0() {
        run_roundtrip(QuantFormat::Q2_0);
    }
    #[test]
    fn roundtrip_q3_0() {
        run_roundtrip(QuantFormat::Q3_0);
    }
    #[test]
    fn roundtrip_q2_1() {
        run_roundtrip(QuantFormat::Q2_1);
    }
    #[test]
    fn roundtrip_q3_1() {
        run_roundtrip(QuantFormat::Q3_1);
    }
    #[test]
    fn roundtrip_q0() {
        run_roundtrip(QuantFormat::Q0);
    }
    #[test]
    fn roundtrip_q1_s() {
        run_roundtrip(QuantFormat::Q1_S);
    }
    #[test]
    fn roundtrip_q2_s() {
        run_roundtrip(QuantFormat::Q2_S);
    }
    #[test]
    fn roundtrip_q2_a() {
        run_roundtrip(QuantFormat::Q2_A);
    }
    #[test]
    fn roundtrip_q0_v() {
        run_roundtrip(QuantFormat::Q0_V);
    }
    #[test]
    fn roundtrip_q1_a() {
        run_roundtrip(QuantFormat::Q1_A);
    }
    #[test]
    fn roundtrip_q0_x() {
        run_roundtrip(QuantFormat::Q0_X);
    }
    #[test]
    fn roundtrip_q0_m2() {
        run_roundtrip(QuantFormat::Q0_M2);
    }
    #[test]
    fn roundtrip_q0_m4() {
        run_roundtrip(QuantFormat::Q0_M4);
    }

    #[test]
    fn roundtrip_unsupported_format_returns_none() {
        let src = ramp_src();
        // R16 carries Q activations in the `q[]` field; round-tripping
        // would lose them.
        assert!(roundtrip_block(QuantFormat::R16, &src, 1.0).is_none());
        // Q8_1::to_float is unimplemented in candle-core.
        assert!(roundtrip_block(QuantFormat::Q8_1, &src, 1.0).is_none());
    }

    #[test]
    fn k_threshold_scaled_matches_kernel_formula() {
        let lo = 0.05f32;
        let hi = 0.01f32;
        let base = (lo * hi).sqrt();

        // q_spread tiny → collapses to safe_spread=1e-8, multiplier ≈ exp(huge)
        // or exp(-huge); clamp pins it. Pick q_med = q_relevance → z=0 →
        // multiplier=1 → scaled=base. Geom-mean of lo/hi.
        let r = k_threshold_scaled(lo, hi, 0.05, 0.05, 1.0e-12);
        assert!(
            (r - base).abs() < 1e-7,
            "tiny-spread, q_med=qrel: got {r}, want {base}"
        );

        // z = 0 → multiplier = 1 → result = base
        let r = k_threshold_scaled(lo, hi, 0.1, 0.1, 0.05);
        assert!((r - base).abs() < 1e-7, "z=0: got {r}, want {base}");

        // q_med large vs qrel small: z = (small − large) / spread = strongly
        // negative → multiplier = exp(+big) → scaled huge → clamp to lo.
        let r = k_threshold_scaled(lo, hi, 0.001, 0.5, 0.05);
        assert!((r - lo).abs() < 1e-7, "z<<0: got {r}, want lo={lo}");

        // q_med small vs qrel large: z >> 0 → multiplier = exp(-big) ≈ 0 →
        // scaled ≈ 0 → clamp to hi.
        let r = k_threshold_scaled(lo, hi, 0.5, 0.001, 0.05);
        assert!((r - hi).abs() < 1e-7, "z>>0: got {r}, want hi={hi}");

        // Hand-computed mid case: lo=0.05, hi=0.01, base=sqrt(5e-4)≈0.022360680.
        // qrel=0.06, q_med=0.05, spread=0.02 → z=0.5 → mult=exp(-0.5)≈0.6065307.
        // scaled = 0.022360680 * 0.6065307 ≈ 0.013561242, in [hi=0.01, lo=0.05].
        let r = k_threshold_scaled(0.05, 0.01, 0.06, 0.05, 0.02);
        let expected_base = (0.05f32 * 0.01).sqrt();
        let expected = expected_base * (-0.5f32).exp();
        assert!(
            (r - expected).abs() < 1e-6,
            "mid case: got {r}, want {expected}"
        );
    }

    #[test]
    fn q_relevance_quantiles_matches_kernel_histogram() {
        // Synthesize 128 blocks with known qrel distribution. We need:
        //  - k_block: non-zero per (d, t) so k2_sum > 0.
        //  - q_block: per-dim Q value chosen so qrel_d takes a known value.
        // qrel = Σ q² k² / Σ k² ⇒ when k is constant across t, qrel = mean(q²).
        // Use k = 1.0 everywhere; q[d, t] = sqrt(d / 127) * sign(t-15.5).
        // Then qrel[d] = mean(q²) = d / 127, evenly spaced 0..1.
        let mut k = vec![0.0f32; HEAD_DIM * CHUNK_SIZE];
        let mut q = vec![0u16; HEAD_DIM * CHUNK_SIZE];
        for d in 0..HEAD_DIM {
            let target = d as f32 / 127.0;
            let qv = target.sqrt();
            for t in 0..CHUNK_SIZE {
                k[d * CHUNK_SIZE + t] = 1.0;
                let sign = if t < 16 { 1.0 } else { -1.0 };
                let q_actual = qv * sign;
                q[d * CHUNK_SIZE + t] = f16::from_f32(q_actual).to_bits();
            }
        }
        let (median, spread) = q_relevance_quantiles(&k, &q);
        // Ground truth median ≈ 0.5; q1 ≈ 0.25; q3 ≈ 0.75; iqr ≈ 0.5;
        // spread ≈ 0.5 * 2 * 1.4427 ≈ 1.4427. Bin resolution = 1/63 ≈ 0.016.
        // f16 quantization adds a few % on top — slack 0.05 captures both.
        assert!((median - 0.5).abs() < 0.05, "median {median} not near 0.5");
        let expected_spread = 0.5f32 * 2.0 * std::f32::consts::LOG2_E;
        assert!(
            (spread - expected_spread).abs() < 0.1,
            "spread {spread} not near {expected_spread}"
        );
    }

    #[test]
    fn sink_threshold_zero_k_collapses_to_lo() {
        let k = vec![0.0f32; HEAD_DIM * CHUNK_SIZE];
        let q_mean = [0.0f32; HEAD_DIM];
        let stats = sink_threshold(&k, &q_mean, 0.05, 0.005);
        // sink_score all zero → mu=0, sigma=0, safe_sigma=1e-8, z=0,
        // tanh(0)=0, w_max=0 → v_thr_eff = lo.
        assert_eq!(stats.sink_weight_max, 0.0);
        let expected_eff = 0.05f32;
        let expected_sq = expected_eff * expected_eff;
        assert!(
            (stats.v_thr_sq - expected_sq).abs() < 1e-9,
            "v_thr_sq {} not near {} (= lo²)",
            stats.v_thr_sq,
            expected_sq
        );
    }

    #[test]
    fn sink_threshold_aligned_k_spikes_threshold() {
        // q_mean is a unit-ish vector along dim 0; K has one token (t=5)
        // with a huge value along dim 0 — that token's sink_score will
        // be a strong outlier vs the other 31 tokens (which dot to 0).
        let mut k = vec![0.0f32; HEAD_DIM * CHUNK_SIZE];
        let spike_t = 5;
        // Dim 0, token `spike_t` — the row-major offset is just the token index.
        k[spike_t] = 100.0;
        let mut q_mean = [0.0f32; HEAD_DIM];
        q_mean[0] = 1.0;
        let lo = 0.05f32;
        let hi = 0.005f32;
        let stats = sink_threshold(&k, &q_mean, lo, hi);
        // 31 tokens at sink_score=0, one at ~100/sqrt(128). mu = score/32,
        // sigma = sqrt(score² * 31/32²); z for the spike = (score - mu)/sigma
        // = (31/32 * score) / (sqrt(31)/32 * score) = sqrt(31) ≈ 5.57.
        // tanh(5.57) ≈ 0.9999... → w_max ≈ 1.0 → v_thr_eff ≈ hi.
        assert!(
            stats.sink_weight_max > 0.99,
            "sink_weight_max {} should be near 1.0 for aligned spike",
            stats.sink_weight_max
        );
        let expected_eff = lo + stats.sink_weight_max * (hi - lo);
        let expected_sq = expected_eff * expected_eff;
        assert!(
            (stats.v_thr_sq - expected_sq).abs() < 1e-9,
            "v_thr_sq {} not the lerp",
            stats.v_thr_sq
        );
        // v_thr_eff should have lerped much closer to hi than to lo.
        assert!(
            expected_eff < (lo + hi) * 0.5,
            "v_thr_eff {expected_eff} did not lerp toward hi"
        );
    }

    #[test]
    fn select_palette4_smoke() {
        let n_chunks = 2;
        let n_kv_head = 2;
        let total = n_chunks * n_kv_head * HEAD_DIM * CHUNK_SIZE;
        let mut k = vec![0.0f32; total];
        let mut v = vec![0.0f32; total];
        for i in 0..total {
            k[i] = deterministic_uniform(0xA17EBA1ABCDEF000, i);
            v[i] = deterministic_uniform(0xB17EBA1A12345600, i);
        }
        let q = vec![0u16; total];
        let k_cands = [QuantFormat::Q4_0, QuantFormat::Q8_0];
        let v_cands = [QuantFormat::Q4_0, QuantFormat::Q8_0];
        let input = SelectionInput {
            k_data: &k,
            v_data: &v,
            q_data: &q,
            k_candidates: &k_cands,
            v_candidates: &v_cands,
            k_threshold_hi: 0.01,
            k_threshold_lo: 0.05,
            v_threshold_hi: 0.005,
            v_threshold_lo: 0.05,
            n_chunks,
            n_kv_head,
            head_dim: HEAD_DIM,
        };
        let out = select_palette4(input);
        assert_eq!(out.heads.len(), n_chunks * n_kv_head);
        for (head_id, h) in out.heads.iter().enumerate() {
            for d in 0..HEAD_DIM {
                assert!(h.k_assignments[d] < N_PALETTE as u8);
                assert!(h.v_assignments[d] < N_PALETTE as u8);
            }
            let mut k_counts = [0usize; N_PALETTE];
            let mut v_counts = [0usize; N_PALETTE];
            for d in 0..HEAD_DIM {
                k_counts[h.k_assignments[d] as usize] += 1;
                v_counts[h.v_assignments[d] as usize] += 1;
            }
            for s in 0..N_PALETTE {
                assert_eq!(k_counts[s], SLOT_QUOTA);
                assert_eq!(v_counts[s], SLOT_QUOTA);
            }
            let base = head_id * HEAD_DIM * CHUNK_SIZE;
            let k_slice = &k[base..base + HEAD_DIM * CHUNK_SIZE];
            let v_slice = &v[base..base + HEAD_DIM * CHUNK_SIZE];
            let expected_k = k_slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let expected_v = v_slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            assert!((h.k_head_amax - expected_k).abs() < 1e-5);
            assert!((h.v_head_amax - expected_v).abs() < 1e-5);
        }
    }

    #[test]
    fn select_palette4_clustering_resists() {
        let n_chunks = 1;
        let n_kv_head = 1;
        let total = HEAD_DIM * CHUNK_SIZE;
        let mut v = vec![0.0f32; total];
        for d in 0..HEAD_DIM {
            let mut mixed = 0u32;
            let mut x = d as u32;
            for _ in 0..7 {
                mixed = (mixed << 1) | (x & 1);
                x >>= 1;
            }
            let mag = (mixed as f32 + 1.0) / 128.0;
            for t in 0..CHUNK_SIZE {
                let phase = if t % 2 == 0 { 1.0 } else { -1.0 };
                v[d * CHUNK_SIZE + t] =
                    mag * phase * (1.0 + 0.1 * deterministic_uniform(0xC0FFEE, d * 32 + t));
            }
        }
        let mut k = vec![0.0f32; total];
        for i in 0..total {
            k[i] = 0.01 * deterministic_uniform(0xDEAD, i);
        }
        let q = vec![0u16; total];
        let cands = [QuantFormat::Q4_0, QuantFormat::Q8_0];
        let input = SelectionInput {
            k_data: &k,
            v_data: &v,
            q_data: &q,
            k_candidates: &cands,
            v_candidates: &cands,
            k_threshold_hi: 0.01,
            k_threshold_lo: 0.05,
            v_threshold_hi: 0.005,
            v_threshold_lo: 0.05,
            n_chunks,
            n_kv_head,
            head_dim: HEAD_DIM,
        };
        let out = select_palette4(input);
        let v_assign = &out.heads[0].v_assignments;
        let mut transitions = 0usize;
        for d in 1..HEAD_DIM {
            if v_assign[d] != v_assign[d - 1] {
                transitions += 1;
            }
        }
        assert!(
            transitions > 60,
            "V assignments too clustered: {transitions} transitions"
        );
    }

    /// Synth a single-head V dataset where all 128 blocks pass the low-BPE
    /// candidate at outer=1.0. With smooth small-amplitude data (well below
    /// the V threshold scaled by head_amax), Q8_0 trivially passes the quota
    /// for every block in slot 0.
    #[test]
    fn process_side_quota_fulfilled_in_priority_order() {
        let total = HEAD_DIM * CHUNK_SIZE;
        let mut v = vec![0.0f32; total];
        // Smooth small amplitude — Q8_0 reconstructs ~1e-3 error, MSE
        // normalized by head_amax² is tiny.
        for d in 0..HEAD_DIM {
            for t in 0..CHUNK_SIZE {
                v[d * CHUNK_SIZE + t] = 0.5 * ((d as f32 * 0.03 + t as f32 * 0.1).sin())
                    + 0.1 * deterministic_uniform(0x1234, d * 32 + t);
            }
        }
        let stats = per_block_amax_qrel(&v, &v, &vec![0u16; total]);
        let idx = sort_amax_desc(&stats.amax_v);
        let kthresh = [0.05f32; HEAD_DIM];
        // Generous v_thr_sq so all blocks pass.
        let v_thr_sq = 1.0f32;
        let cands = [QuantFormat::Q8_0];
        let (slot_fmt, _slot_scale, assigns) = process_side(
            &v,
            &stats.amax_v,
            &idx,
            &cands,
            stats.v_head_amax,
            false,
            &kthresh,
            v_thr_sq,
        );
        assert_eq!(slot_fmt[0], QuantFormat::Q8_0);
        // Slot 0 should claim the top-32 amax blocks, i.e. idx[0..32].
        let top32: std::collections::HashSet<u16> = idx.iter().take(32).copied().collect();
        for d in 0..HEAD_DIM {
            if top32.contains(&(d as u16)) {
                assert_eq!(
                    assigns[d], 0,
                    "dim {d} (top-32 amax) should be in slot 0 but is {}",
                    assigns[d]
                );
            }
        }
    }

    /// Construct a side where no candidate passes the threshold for even
    /// the slot quota — the fallback path must engage and still fill 32
    /// blocks. We use an extreme-low threshold and small candidate set.
    #[test]
    fn process_side_fallback_when_no_format_fits() {
        let total = HEAD_DIM * CHUNK_SIZE;
        let mut v = vec![0.0f32; total];
        for i in 0..total {
            v[i] = deterministic_gauss(0xCAFE, i) * 2.0;
        }
        let stats = per_block_amax_qrel(&v, &v, &vec![0u16; total]);
        let idx = sort_amax_desc(&stats.amax_v);
        let kthresh = [0.0f32; HEAD_DIM];
        // Effectively-zero V threshold so no block can pass.
        let v_thr_sq = 1.0e-20f32;
        let cands = [QuantFormat::Q4_0, QuantFormat::Q8_0];
        let (slot_fmt, _, assigns) = process_side(
            &v,
            &stats.amax_v,
            &idx,
            &cands,
            stats.v_head_amax,
            false,
            &kthresh,
            v_thr_sq,
        );
        // All four slots should still have valid (one of the candidate)
        // formats from fallback selection.
        for s in 0..N_PALETTE {
            assert!(
                slot_fmt[s] == QuantFormat::Q4_0 || slot_fmt[s] == QuantFormat::Q8_0,
                "slot {s} fmt {:?} not in candidate set",
                slot_fmt[s]
            );
        }
        let mut counts = [0usize; N_PALETTE];
        for d in 0..HEAD_DIM {
            assert!(assigns[d] < N_PALETTE as u8);
            counts[assigns[d] as usize] += 1;
        }
        for s in 0..N_PALETTE {
            assert_eq!(counts[s], SLOT_QUOTA, "slot {s} did not get 32 blocks");
        }
    }

    #[test]
    fn v_threshold_squared_comparison() {
        // Build a V-side scenario where the metric (mse/head_amax²) for a
        // specific block straddles a chosen v_thr_sq. We control orig vs
        // recon directly by handcrafting a tiny test of the inner predicate.
        // mean_squared_error * inv_head_amax_sq <= v_thr_sq is the V check.
        let head_amax = 1.0f32;
        let inv_head_amax_sq = 1.0;
        let mut orig = [0.0f32; CHUNK_SIZE];
        let mut recon_just_above = [0.0f32; CHUNK_SIZE];
        let mut recon_just_below = [0.0f32; CHUNK_SIZE];
        for i in 0..CHUNK_SIZE {
            orig[i] = 0.5;
            recon_just_above[i] = 0.5 + 0.11;
            recon_just_below[i] = 0.5 + 0.09;
        }
        let mse_above = mean_squared_error(&orig, &recon_just_above);
        let mse_below = mean_squared_error(&orig, &recon_just_below);
        // v_thr_eff = 0.1; v_thr_sq = 0.01. mse_below = 0.0081 (< 0.01),
        // mse_above = 0.0121 (> 0.01).
        let v_thr_sq = 0.01f32;
        assert!(
            mse_above * inv_head_amax_sq > v_thr_sq,
            "above case: mse {mse_above} should exceed v_thr_sq {v_thr_sq}"
        );
        assert!(
            mse_below * inv_head_amax_sq <= v_thr_sq,
            "below case: mse {mse_below} should be <= v_thr_sq {v_thr_sq}"
        );
        let _ = head_amax;
    }

    #[test]
    fn production_c5_candidates_all_round_trip() {
        let (k_cands, v_cands) = production_adaptive_candidates(5);
        for kv in k_cands.iter().chain(v_cands.iter()) {
            let q = quant_from_kv(*kv);
            let src = ramp_src();
            let out = roundtrip_block(q, &src, 1.0)
                .unwrap_or_else(|| panic!("C5 candidate {:?} round-trip None", q));
            let mut max_err = 0.0f32;
            for i in 0..CHUNK_SIZE {
                let e = (src[i] - out[i]).abs();
                if e > max_err {
                    max_err = e;
                }
            }
            assert!(
                max_err < fmt_error_bound(q),
                "C5 {:?}: max_err {} >= bound {}",
                q,
                max_err,
                fmt_error_bound(q)
            );
        }
    }

    #[test]
    fn select_palette4_against_real_c5_thresholds() {
        let n_chunks = 1;
        let n_kv_head = 1;
        let total = HEAD_DIM * CHUNK_SIZE;
        let mut k = vec![0.0f32; total];
        let mut v = vec![0.0f32; total];
        for i in 0..total {
            k[i] = deterministic_gauss(0xC5A11, i) * 1.5;
            v[i] = deterministic_gauss(0xC5A12, i) * 0.5;
        }
        let mut q = vec![0u16; total];
        for i in 0..total {
            let qv = deterministic_uniform(0xC5A13, i) * 0.1;
            q[i] = f16::from_f32(qv).to_bits();
        }
        let (k_cands_kv, v_cands_kv) = production_adaptive_candidates(5);
        let k_cands: Vec<QuantFormat> = k_cands_kv.iter().map(|f| quant_from_kv(*f)).collect();
        let v_cands: Vec<QuantFormat> = v_cands_kv.iter().map(|f| quant_from_kv(*f)).collect();
        let input = SelectionInput {
            k_data: &k,
            v_data: &v,
            q_data: &q,
            k_candidates: &k_cands,
            v_candidates: &v_cands,
            k_threshold_hi: PRODUCTION_K_QREL_HIGH_THRESHOLDS[5],
            k_threshold_lo: PRODUCTION_K_QREL_LOW_THRESHOLDS[5],
            v_threshold_hi: PRODUCTION_V_QREL_HIGH_THRESHOLDS[5],
            v_threshold_lo: PRODUCTION_V_QREL_LOW_THRESHOLDS[5],
            n_chunks,
            n_kv_head,
            head_dim: HEAD_DIM,
        };
        let out = select_palette4(input);
        assert_eq!(out.heads.len(), 1);
        let h = &out.heads[0];
        let mut k_counts = [0usize; N_PALETTE];
        let mut v_counts = [0usize; N_PALETTE];
        for d in 0..HEAD_DIM {
            assert!(h.k_assignments[d] < N_PALETTE as u8);
            assert!(h.v_assignments[d] < N_PALETTE as u8);
            k_counts[h.k_assignments[d] as usize] += 1;
            v_counts[h.v_assignments[d] as usize] += 1;
        }
        for s in 0..N_PALETTE {
            assert_eq!(k_counts[s], SLOT_QUOTA, "K slot {s} count");
            assert_eq!(v_counts[s], SLOT_QUOTA, "V slot {s} count");
        }
        let expected_k = k.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let expected_v = v.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!((h.k_head_amax - expected_k).abs() < 1e-5);
        assert!((h.v_head_amax - expected_v).abs() < 1e-5);
        for s in 0..N_PALETTE {
            assert!(
                k_cands.contains(&h.k_pal_format[s]),
                "K slot {s} fmt {:?} not in C5 K candidates",
                h.k_pal_format[s]
            );
            assert!(
                v_cands.contains(&h.v_pal_format[s]),
                "V slot {s} fmt {:?} not in C5 V candidates",
                h.v_pal_format[s]
            );
            assert!(h.k_pal_scale[s].is_finite() && h.k_pal_scale[s] > 0.0);
            assert!(h.v_pal_scale[s].is_finite() && h.v_pal_scale[s] > 0.0);
        }
    }
}
