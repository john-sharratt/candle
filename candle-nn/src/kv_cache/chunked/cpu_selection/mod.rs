//! A CPU mirror of the CUDA palette-4 format selection, used to compare against the
//! kernel when chasing chunk-level anomalies.
//!
//! The mirror repeats the kernel's per-(chunk, head) selection in plain Rust, so
//! its output can be compared with the GPU's on the same data. It is a debugging
//! and investigation tool and is not part of the live compression path. Its
//! constants and the head dimension it is written for are fixed here because a
//! differential mirror benefits from fixed array sizes and exact asserts, and the
//! GPU path accepts any compiled shape that divides by the palette count.
//!
//! - `block_stats`: per-block amax, query relevance, flatness and the head scale.
//! - `search`: the per-format scale search and round-trip error.
//! - `side`: the two-pass slot assignment for one side of the cache.
//! - `slots`: amax ordering, the K threshold and slot statistics.
//!
//! Tests:
//!
//! - `tests_roundtrip`: per-format round-trip checks.
//! - `tests_select`: palette selection outcomes on synthetic blocks.
//! - `tests_stats`: block statistics against hand-computed values.

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

use half::f16;

use super::block_round_trip::block_round_trip;
use crate::kv_cache::QuantFormat;
use std::cmp::Ordering;

const CHUNK_SIZE: usize = 32;

/// The head dimension this mirror is written for — a property of *this file*,
/// not of the algorithm or of the live path. The selection operates on
/// 32-element blocks, so head dimension only sets how many blocks a palette
/// band covers; the GPU path takes any compiled shape that divides by
/// `N_PALETTE` (128 and 256 today). Fixing it here buys stack arrays and
/// exact-size asserts, which is what a differential debugging mirror wants.
const HEAD_DIM: usize = 128;

const N_PALETTE: usize = 4;

const SLOT_QUOTA: usize = HEAD_DIM / N_PALETTE;

const NUM_SCALE_CANDIDATES: usize = 6;

/// Histogram bins of the head |x| percentile pass (`QREL_HIST_BINS`).
const P95_HIST_BINS: usize = 64;

/// A block whose standard deviation is below `head scale / FLAT_BLOCK_STEPS`
/// carries less than one INT8 step of variation (`FLAT_BLOCK_STEPS`).
const FLAT_BLOCK_STEPS: f32 = 127.0;

/// The head scale is capped at this many times the head's p95 |x|
/// (`HEAD_SCALE_P95_CAP`).
const HEAD_SCALE_P95_CAP: f32 = 8.0;

/// Inputs to the full selection pipeline, in the CUDA kernel's logical block
/// view (chunk × head × dim × token, row-major in that order): each 32-element
/// block is one dim's tokens. Float arenas hold `[t][pd]` bands in memory; the
/// kernel reads them into this view (`float_band_elem`).
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
    assert_eq!(
        input.head_dim, HEAD_DIM,
        "this mirror is built for head_dim {HEAD_DIM}; the live path is not \
         restricted to it (see the constant)"
    );
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
        head_scale(stats.k_head_amax, stats.k_head_p95),
        true,
        &kthresh,
        sink.v_thr_sq,
    );
    let (v_pal_format, v_pal_scale, v_assignments) = process_side(
        v_block,
        &stats.amax_v,
        &vidx,
        v_candidates,
        head_scale(stats.v_head_amax, stats.v_head_p95),
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

mod block_stats;
mod search;
mod side;
mod slots;
#[cfg(test)]
mod tests_roundtrip;
#[cfg(test)]
mod tests_select;
#[cfg(test)]
mod tests_stats;

pub use block_stats::*;
pub use search::*;
pub use side::*;
pub use slots::*;
