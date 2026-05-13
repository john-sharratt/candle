// SPDX-License-Identifier: MIT
// Q0_V Quantization: per-block encoder.
//
// Pipeline:
//
//   Step 1: Compute the actual block (centroid, scale).
//
//   Step 2: Pick scale_idx (32 entries) and centroid_idx (8 entries within
//           the chosen scale row) from the constant-memory codebooks. These
//           are tiny scans run with FMA + fast reciprocal.
//
//   Step 3: Normalise the block into the curve-table space:
//             target_scaled[lane] = (xi − chosen_centroid) / scale_baked
//           where scale_baked = scale_norm / 127 (the f16-stored value).
//           After this, target_scaled is directly comparable to the i8
//           curve_table values — no /127 needed in the inner loop.
//
//   Step 4: Peak-bin curve search.
//
//     A — Find target peak lane: the lane of max |target_scaled[lane]|, via
//         a warp argmax (~10 shfl).
//
//     B — The 256 curves are pre-sorted by their own peak lane (stored in
//         q0_v_peak_curve_indices_<side>; bin offsets in
//         q0_v_peak_bin_offsets_<side>). Score only curves whose peak lane
//         is in {peak−1, peak, peak+1} (cyclic), where the peak ±1 window
//         absorbs i8 quantisation slop in either curve or target.
//
//     C — Pick the lowest-L2² curve in the ±1 window.
//
//   Step 5: Pack (curve_idx, scale_idx, centroid_idx) into the 2-byte block.
//
// Operates on outer-normalised input — i.e. (raw / head_amax), so xi is in
// [-1, +1].
//
// IS_K selects between the K-side and V-side calibrated table sets at compile
// time. K and V are calibrated separately (different codebooks) because their
// statistical distributions differ.
//
// CPU reference: `candle-core/src/quantized/k_quants.rs::encode_block_q0_v`.

#pragma once

#include "q0_v_tables.cuh"

namespace q0_v_detail {

// Warp-cooperative reductions (full mask).
__device__ __forceinline__ float warp_sum32(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        x += __shfl_xor_sync(0xffffffff, x, off, 32);
    return x;
}
__device__ __forceinline__ float warp_max32(float x) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, off, 32));
    return x;
}

// Warp argmax of |x| over the 32 lanes; returns the lane index of the max
// (broadcast to all lanes). Tie-broken by lower lane.
__device__ __forceinline__ int warp_argmax_abs32(float x) {
    const float a = fabsf(x);
    const float m = warp_max32(a);
    // Lanes that hold the max raise their lane bit; pick the lowest such lane.
    const unsigned mask = __ballot_sync(0xffffffff, a == m);
    return __ffs(mask) - 1;
}

// Score `target_scaled` against a 32-element i8 curve, return the warp-wide
// L2² error. Inner loop has zero constant multiplications: target_scaled is
// in i8 [-127, +127] space (since we pre-baked /127 into scale), so the
// curve values are read raw and subtracted directly.
__device__ __forceinline__ float score_curve(
    float target_scaled, const int8_t* __restrict__ curve, int lane)
{
    const float cv = (float)curve[lane];
    const float d  = target_scaled - cv;
    return warp_sum32(d * d);
}

// Q0VTablesStatic<IS_K> / Q0VTablesRuntime are defined in q0_v_tables.cuh
// (shared between encoder and decoder so block_q0_v.cuh can also use them).

// =============================================================================
// Steps 1–3: shared between hierarchical and flat encoders
// =============================================================================
// Computes the per-lane `target_scaled` value (block normalised into
// curve-table i8 space) and picks the best (scale_idx, centroid_idx) pair
// from the codebook. Both encode variants reuse this prologue verbatim.
template <typename Tables>
__device__ __forceinline__ void compute_target_and_indices(
    float xi, const Tables& tbl,
    float& target_scaled, int& best_scale_idx, int& best_centroid_idx)
{
    // ── Step 1: actual (centroid, scale) of the block ──
    const float sum_x = warp_sum32(xi);
    const float actual_centroid = sum_x * (1.0f / 32.0f);
    const float dev = fabsf(xi - actual_centroid);
    const float actual_scale = warp_max32(dev);

    // ── Step 2a: pick scale_idx (32 entries) ──
    best_scale_idx = 0;
    float best_scale_err = 1e30f;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        const float scale_baked = __half2float(__ushort_as_half(tbl.scale_bits(i)));
        const float s = scale_baked * 127.0f;
        const float err = fabsf(actual_scale - s);
        if (err < best_scale_err) { best_scale_err = err; best_scale_idx = i; }
    }
    const float chosen_scale_baked = __half2float(__ushort_as_half(tbl.scale_bits(best_scale_idx)));

    // ── Step 2b: pick centroid_idx (16 entries from chosen scale row) ──
    best_centroid_idx = 0;
    float best_centroid_err = 1e30f;
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        const float c = __half2float(__ushort_as_half(tbl.centroid_bits(best_scale_idx, j)));
        const float err = fabsf(actual_centroid - c);
        if (err < best_centroid_err) { best_centroid_err = err; best_centroid_idx = j; }
    }
    const float chosen_centroid = __half2float(__ushort_as_half(
        tbl.centroid_bits(best_scale_idx, best_centroid_idx)));

    // ── Step 3: normalise into curve-table space ──
    const float inv_scale  = __frcp_rn(chosen_scale_baked);
    const float negc_invs  = -chosen_centroid * inv_scale;
    target_scaled          = __fmaf_rn(xi, inv_scale, negc_invs);
    (void)best_scale_err; (void)best_centroid_err;
}

// =============================================================================
// Step 4: hierarchical curve search — Stage A (8 bucket reps) + Stage B
//         (16 phases of best bucket) + peak-bin refinement (±1 lane window).
// =============================================================================
// Curve table is now 128 entries (7-bit curve_idx) laid out as 8 buckets ×
// 16 phases (bucket-major). Stage A scans 8 buckets (was 16, when the table
// held 256 curves). Stage B/C are unchanged.
template <typename Tables>
__device__ __forceinline__ int find_best_curve_hierarchical(
    float target_scaled, int lane, const Tables& tbl)
{
    // ── Stage A — score 8 bucket representatives ──
    int   best_bucket = 0;
    float best_bucket_err = 1e30f;
    #pragma unroll
    for (int b = 0; b < 8; b++) {
        const float err = score_curve(target_scaled, tbl.curve(b * 16), lane);
        if (err < best_bucket_err) { best_bucket_err = err; best_bucket = b; }
    }
    (void)best_bucket_err;

    // ── Stage B — score 16 phases of best_bucket ──
    int   best_curve_idx = best_bucket << 4;
    float best_curve_err = 1e30f;
    {
        const int base = best_bucket << 4;
        #pragma unroll
        for (int p = 0; p < 16; p++) {
            const int c = base + p;
            const float err = score_curve(target_scaled, tbl.curve(c), lane);
            if (err < best_curve_err) { best_curve_err = err; best_curve_idx = c; }
        }
    }

    // ── Stage C: peak-bin refinement ──
    const int peak_lane = warp_argmax_abs32(target_scaled);
    #pragma unroll
    for (int dp = -1; dp <= 1; dp++) {
        const int bin   = (peak_lane + dp + 32) & 31;
        const int start = (int)tbl.peak_off(bin);
        const int end   = (int)tbl.peak_off(bin + 1);
        for (int k = start; k < end; k++) {
            const int   c   = (int)tbl.peak_idx(k);
            const float err = score_curve(target_scaled, tbl.curve(c), lane);
            if (err < best_curve_err) { best_curve_err = err; best_curve_idx = c; }
        }
    }
    return best_curve_idx;
}


// =============================================================================
// Top-level per-block encoder (generic over `Tables`).
// =============================================================================
// Combines the shared Steps 1–3 with the hierarchical curve search. Production
// (`Q0VTablesStatic<IS_K>`) and runtime (`Q0VTablesRuntime`) callers both use
// this — the only difference is which struct gets passed in.
//
// Returns the three indexes in lane 0; values in other lanes are undefined.
template <typename Tables>
__device__ __forceinline__ void per_block_encode_generic(
    float xi, int lane, const Tables& tbl,
    int& curve_idx, int& scale_idx, int& centroid_idx)
{
    float target_scaled;
    compute_target_and_indices(xi, tbl, target_scaled, scale_idx, centroid_idx);
    curve_idx = find_best_curve_hierarchical(target_scaled, lane, tbl);
}

// Backward-compat wrapper for IS_K-templated callers (production path).
// Empty `Q0VTablesStatic<IS_K>` instance is zero-sized; the compiler inlines
// every method to a direct __constant__-memory load.
template <bool IS_K>
__device__ __forceinline__ void per_block_encode(
    float xi, int lane,
    int& curve_idx, int& scale_idx, int& centroid_idx)
{
    Q0VTablesStatic<IS_K> tbl;
    per_block_encode_generic(xi, lane, tbl, curve_idx, scale_idx, centroid_idx);
}

// New 16-bit packing (curve 7b, scale 5b, centroid 4b):
//   bits[0..6]   = curve_idx
//   bits[7..11]  = scale_idx
//   bits[12..15] = centroid_idx
// Byte view: lo = curve | (scale & 1) << 7;  hi = (scale >> 1) | (centroid << 4)
__device__ __forceinline__ void q0_v_pack_block(
    block_q0_v* __restrict__ dst, int curve_idx, int scale_idx, int centroid_idx)
{
    const unsigned bits =
        ((unsigned)(curve_idx)    & 0x7Fu)
      | (((unsigned)(scale_idx)   & 0x1Fu) << 7)
      | (((unsigned)(centroid_idx) & 0x0Fu) << 12);
    dst->lo = (uint8_t)(bits & 0xFFu);
    dst->hi = (uint8_t)((bits >> 8) & 0xFFu);
}

}  // namespace q0_v_detail

// Per-block encoder. Each warp encodes one block. lane = element index 0..31.
template <bool IS_K>
__device__ __forceinline__ void quantize_block_q0_v_core(
    float xi, block_q0_v* __restrict__ dst)
{
    const int lane = threadIdx.x & 31;
    int curve_idx = 0, scale_idx = 0, centroid_idx = 0;
    q0_v_detail::per_block_encode<IS_K>(xi, lane, curve_idx, scale_idx, centroid_idx);
    if (lane == 0) {
        q0_v_detail::q0_v_pack_block(dst, curve_idx, scale_idx, centroid_idx);
    }
}

// Runtime-tables variant: same encoder logic, codebook supplied at launch
// time via `Q0VTablesRuntime`. Used by the curve-selection diagnostic where
// the caller swaps the curve table per iteration. The runtime tables must
// match the new layout (128 curves = 8 buckets × 16 phases, 32 scales,
// 32×16 centroids, 128-entry peak permutation, 33-entry peak offsets) so
// the hierarchical Stage A + B + peak-bin path runs unchanged.
__device__ __forceinline__ void quantize_block_q0_v_core_runtime(
    float xi, block_q0_v* __restrict__ dst,
    const q0_v_detail::Q0VTablesRuntime& tbl)
{
    const int lane = threadIdx.x & 31;
    int curve_idx = 0, scale_idx = 0, centroid_idx = 0;
    q0_v_detail::per_block_encode_generic(xi, lane, tbl, curve_idx, scale_idx, centroid_idx);
    if (lane == 0) {
        q0_v_detail::q0_v_pack_block(dst, curve_idx, scale_idx, centroid_idx);
    }
}

template <bool IS_K = false>
__device__ __forceinline__ void quantize_block_q0_v_vec(
    const float* __restrict__ src, block_q0_v* __restrict__ dst)
{
    const int lane = threadIdx.x & 31;
    quantize_block_q0_v_core<IS_K>(src[lane], dst);
}

template <bool IS_K = false>
__device__ __forceinline__ void quantize_block_q0_v(
    const float* __restrict__ src, block_q0_v* __restrict__ dst)
{
    const int lane = threadIdx.x & 31;
    quantize_block_q0_v_core<IS_K>(src[lane], dst);
}

template <bool IS_K = false, int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q0_v(
    const float* __restrict__ src, block_q0_v* __restrict__ dst, int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x & 31;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    for (int blk = warp_id + blockIdx.x * warps_per_block;
         blk < num_blocks; blk += warps_per_block * gridDim.x) {
        quantize_block_q0_v_core<IS_K>(src[blk * QK_Q0_V + lane], dst + blk);
    }
}
