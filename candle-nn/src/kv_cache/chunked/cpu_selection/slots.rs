//! Orders blocks by amax, derives the per-block K threshold and the slot
//! statistics the palette search reads.
//!
//! Blocks sort by amax with ties broken by original index, so the order is
//! deterministic. The kernel's sorting network does not guarantee that tie order,
//! so on tied inputs this order can differ from the kernel's, while the slot search
//! depends only on the amax order. The K threshold is a geometric mean of its
//! bounds scaled by how the block's query relevance sits against the chunk's
//! median, with a z-score taken over the quantile spread and clamped to the bounds.
//! A NaN in the relevance or median takes the geometric-mean value, so the
//! comparisons that follow stay defined. Slot statistics are walked over the blocks
//! still alive, in amax order, with the same target formulae the kernel uses.

use super::*;

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
    let p95_tgt = 1.max(lc.div_ceil(20)) as i32;
    let p80_tgt = 1.max(lc.div_ceil(5)) as i32;
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
