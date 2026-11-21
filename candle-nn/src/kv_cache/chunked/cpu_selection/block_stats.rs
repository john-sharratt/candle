//! Computes the per-block statistics the format search reads: amax, query
//! relevance, the head scale and the flatness test.
//!
//! The head scale is the head's amax capped at a multiple of its 95th percentile,
//! so one sink element cannot loosen every error threshold in the chunk that holds
//! it. Flatness is judged against one INT8 step of that scale, since a block whose
//! variation is below one step has nothing to keep, and Q0 is offered only those
//! blocks because it stores a block's mean alone and keeps none of its variation.
//! Blocks that carry no query leave their relevance at 1.0 and the quantile
//! early-out returns zero spread, so the K threshold falls back to the geometric
//! mean. Amax values get a deterministic jitter far below any real magnitude, so
//! tied blocks sort into a stable order and do not cluster into one palette. The
//! sink statistic gives the V threshold a squared effective value that the search
//! compares each block's error against.

use super::*;

/// Formats offered only flat blocks (`flat_only`). Q0 stores a block's mean and
/// nothing else, so it keeps none of the variation between the block's tokens;
/// the error metrics judge an error against the head's scale and would pass it
/// on a dim whose tokens vary by a small fraction of the head's range.
pub(super) fn flat_only(fmt: QuantFormat) -> bool {
    fmt == QuantFormat::Q0
}

/// The scale the error metrics divide by (`head_scale`): the head's amax capped
/// at [`HEAD_SCALE_P95_CAP`] times its p95 |x|, so a sink token's few large
/// elements cannot loosen every threshold in the chunk that holds them.
pub fn head_scale(amax: f32, p95: f32) -> f32 {
    amax.min(HEAD_SCALE_P95_CAP * p95).max(1.0e-8)
}

/// 95th percentile of `|x|` over a head's elements, as the kernel's percentile
/// pass computes it: sqrt-spaced bins over `[0, amax]`, answered as the upper
/// edge of the bin the percentile falls in, so it never understates the true
/// value. Every element of `data` is a valid token.
pub fn head_p95(data: &[f32], amax: f32) -> f32 {
    let amax_safe = if amax > 1.0e-8 { amax } else { 1.0 };
    let inv_amax = 1.0 / amax_safe;
    let mut hist = [0usize; P95_HIST_BINS];
    for &x in data {
        let bin = ((x.abs() * inv_amax).sqrt() * (P95_HIST_BINS - 1) as f32).floor() as usize;
        hist[bin.min(P95_HIST_BINS - 1)] += 1;
    }
    let target = (0.95 * (data.len() - 1) as f32).floor() as usize;
    let mut accum = 0usize;
    let mut p95_bin = P95_HIST_BINS - 1;
    for (b, &count) in hist.iter().enumerate() {
        accum += count;
        if accum > target {
            p95_bin = b;
            break;
        }
    }
    let edge = (p95_bin + 1) as f32 / (P95_HIST_BINS - 1) as f32;
    amax_safe.min(amax_safe * edge * edge)
}

/// Is `block` flat — its standard deviation below one INT8 step of `head_norm`?
/// The variance is E[x^2] - mean^2 from the first and second moments, as the
/// kernel computes it.
pub(super) fn block_is_flat(block: &[f32; CHUNK_SIZE], head_norm: f32) -> bool {
    let n_inv = 1.0 / CHUNK_SIZE as f32;
    let s1: f32 = block.iter().sum();
    let s2: f32 = block.iter().map(|x| x * x).sum();
    let mean = s1 * n_inv;
    let var = s2 * n_inv - mean * mean;
    let step = head_norm * (1.0 / FLAT_BLOCK_STEPS);
    var <= step * step
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
    pub k_head_p95: f32,
    pub v_head_p95: f32,
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
        k_head_p95: head_p95(k_block, k_head_amax),
        v_head_p95: head_p95(v_block, v_head_amax),
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
    // Not `head_max <= head_min`: the samples are ratios, so a NaN can reach
    // here, and it must take this early return rather than flow into
    // `inv_range`. `partial_cmp != Greater` rejects the unordered case the way
    // the negated `>` did.
    if head_max.partial_cmp(&head_min) != Some(Ordering::Greater) || head_max <= 0.0 {
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
    for (b, &count) in hist.iter().enumerate() {
        accum += count;
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
