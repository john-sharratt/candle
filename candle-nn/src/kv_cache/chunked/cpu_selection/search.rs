//! Searches, for each candidate format, the outer scale that fits the most blocks,
//! and the error each format leaves on a block.
//!
//! A block round-trips through a format after it is pre-scaled by the outer scale,
//! and the caller compares the result against the original. Each format yields two
//! results: the lowest-error scale among those that hit the 32-block quota, when
//! one does, and the lowest-error scale overall, which is always available and
//! serves as the fallback. A flat-only format cannot pass a block that is not flat,
//! since the kernel skips its round trip, so a flat-only format's error covers only
//! the blocks it was offered and it is not a fallback for the whole slot. The
//! error metric is the mean of the four largest absolute errors across the 32 lanes,
//! and the search also records the maximum pass metric over every block, so the
//! fallback bookkeeping matches the kernel's.

use super::*;

/// Round-trip 32 floats through `fmt` on side `is_k` after pre-scaling by
/// `outer` (`block_round_trip`). Returns the dequantized vector pre-divided
/// by `outer` (so the caller compares against the original directly), or
/// `None` for a format that is never a quant target (R16, the active K
/// carrier: its Q-capture half has no round trip).
pub fn roundtrip_block(
    fmt: QuantFormat,
    src: &[f32; CHUNK_SIZE],
    outer: f32,
    is_k: bool,
) -> Option<[f32; CHUNK_SIZE]> {
    let scaled: [f32; CHUNK_SIZE] = std::array::from_fn(|i| src[i] * outer);
    let recon = block_round_trip(fmt.to_ggml_dtype(), &scaled, is_k)?;
    let inv_outer = if outer != 0.0 { 1.0 / outer } else { 0.0 };
    Some(recon.map(|r| r * inv_outer))
}

/// Mean of the four largest absolute errors across the 32 lanes.
/// Matches `mean_top4_abs_error_warp` (degenerate-equal case returns the
/// shared value, not zero).
pub(super) fn mean_top4_abs_error(orig: &[f32; CHUNK_SIZE], recon: &[f32; CHUNK_SIZE]) -> f32 {
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

pub(super) fn mean_squared_error(orig: &[f32; CHUNK_SIZE], recon: &[f32; CHUNK_SIZE]) -> f32 {
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
    head_norm: f32,
) -> Option<PerFmtSearch> {
    block_round_trip(fmt.to_ggml_dtype(), &[0.0; CHUNK_SIZE], is_k)?;
    let mut best: Option<ScaleResult> = None;
    let mut fallback: Option<ScaleResult> = None;
    // A flat-only format that skipped any block cannot take the whole slot, so
    // it is not a fallback candidate; it can still win outright on its passing
    // blocks. Set by the first scale's pass over the blocks, which is the same
    // set for every scale.
    let mut skipped_nonflat = false;
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
            // A flat-only format cannot pass a block that is not flat: the
            // kernel skips the round trip, and so does this. Its error then
            // covers only the blocks it was offered.
            if flat_only(fmt) && !block_is_flat(&orig, head_norm) {
                skipped_nonflat = true;
                continue;
            }
            let recon = roundtrip_block(fmt, &orig, outer, is_k)?;
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
        if !skipped_nonflat {
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
        }
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
