// SPDX-License-Identifier: MIT
// Q0_V Quantization: per-block encoder.
//
// Pipeline:
//
//   Step 1: Compute the actual block (centroid, scale).
//
//   Step 2: Pick scale_idx (32 entries) and centroid_idx (16 entries within
//           the chosen scale row), one entry per lane and a warp argmin each.
//
//   Step 3: Normalise the block into the curve-table space:
//             target_scaled[lane] = (xi − chosen_centroid) / scale_baked
//           where scale_baked = scale_norm / 127 (the f16-stored value).
//           After this, target_scaled is directly comparable to the i8
//           curve values — no /127 needed in the inner loop.
//
//   Step 4: Curve search. The production codebooks score all 128 curves
//           exactly, as 64 (base, phase) correlations across the lanes
//           (`find_best_curve_exhaustive`); a codebook supplied at launch is
//           searched hierarchically (`find_best_curve_hierarchical`).
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

// Q0VTablesStatic<IS_K>, Q0VTablesRuntime and Q0VDecodeTables<IS_K> are
// defined in q0_v_tables.cuh, shared with the decoder in block_q0_v.cuh.

// The warp's hardware reductions (`redux.sync`, sm_80+ — every target this
// builds for) answer a max or min over the 32 lanes in one instruction where
// a shuffle butterfly takes five rounds. They reduce unsigned integers, so a
// float is compared through its bits: for non-negative floats the bit
// patterns order exactly as the values do.

/// Max of a non-negative float over the warp, exactly (every lane).
__device__ __forceinline__ float warp_max32_nonneg(float x) {
    return __uint_as_float(__reduce_max_sync(0xffffffff, __float_as_uint(x)));
}

/// The lane holding the smallest non-negative `err`, the lowest such lane on
/// a tie — what a scan of the lanes in order with a strict `<` keeps (every
/// lane).
__device__ __forceinline__ int warp_argmin_lane_nonneg(float err) {
    const uint32_t bits = __float_as_uint(err);
    const uint32_t lo = __reduce_min_sync(0xffffffff, bits);
    return __ffs(__ballot_sync(0xffffffff, bits == lo)) - 1;
}

/// Warp argmin of (err, idx), lexicographic, for an err of any sign: the
/// smallest error, and among equal errors the lowest index (every lane).
/// The float maps to an unsigned key that orders as the float does — sign
/// bit set for positives, all bits flipped for negatives — after `+ 0.f`
/// turns a −0 into +0, since the two compare equal and must tie.
__device__ __forceinline__ int warp_argmin_idx(float err, int idx) {
    const uint32_t u = __float_as_uint(__fadd_rn(err, 0.f));
    const uint32_t key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    const uint32_t lo = __reduce_min_sync(0xffffffff, key);
    return (int)__reduce_min_sync(0xffffffff, key == lo ? (uint32_t)idx : 0xffffffffu);
}

// =============================================================================
// Steps 1–3: shared by the production and runtime-table encoders
// =============================================================================
// Computes the per-lane `target_scaled` value (block normalised into
// curve-table i8 space) and picks the best (scale_idx, centroid_idx) pair
// from the codebook. The scale and centroid scans run one entry per lane.
//
// Every operation is an explicitly rounded intrinsic, so the compiler cannot
// contract a multiply into a following add: `encode_block_q0_v` in
// k_quants.rs performs the same operations in the same order and produces
// the same bytes (the encode oracle test compares them).
template <typename Tables>
__device__ __forceinline__ void compute_target_and_indices(
    float xi, const Tables& tbl,
    float& target_scaled, int& best_scale_idx, int& best_centroid_idx)
{
    const int lane = threadIdx.x & 31;

    // ── Step 1: actual (centroid, scale) of the block ──
    const float sum_x = warp_sum32(xi);
    const float actual_centroid = __fmul_rn(sum_x, 1.0f / 32.0f);
    const float dev = fabsf(__fsub_rn(xi, actual_centroid));
    const float actual_scale = warp_max32_nonneg(dev);

    // ── Step 2a: scale_idx — lane i scores scale entry i ──
    {
        const float scale_baked = __half2float(__ushort_as_half(tbl.scale_bits(lane)));
        const float err = fabsf(__fsub_rn(actual_scale, __fmul_rn(scale_baked, 127.0f)));
        best_scale_idx = warp_argmin_lane_nonneg(err);
    }
    const float chosen_scale_baked = __half2float(__ushort_as_half(tbl.scale_bits(best_scale_idx)));

    // ── Step 2b: centroid_idx — lane j < 16 scores entry j of the chosen row ──
    {
        const int j = lane & 15;
        const float c = __half2float(__ushort_as_half(tbl.centroid_bits(best_scale_idx, j)));
        const float err = lane < 16 ? fabsf(__fsub_rn(actual_centroid, c)) : INFINITY;
        best_centroid_idx = warp_argmin_lane_nonneg(err);
    }
    const float chosen_centroid = __half2float(__ushort_as_half(
        tbl.centroid_bits(best_scale_idx, best_centroid_idx)));

    // ── Step 3: normalise into curve-table space ──
    const float inv_scale  = __frcp_rn(chosen_scale_baked);
    const float negc_invs  = __fmul_rn(-chosen_centroid, inv_scale);
    target_scaled          = __fmaf_rn(xi, inv_scale, negc_invs);
}

// =============================================================================
// Step 4 (production codebooks): exhaustive curve search
// =============================================================================
// A curve's squared error against the target is
//
//   Σ (t − curve)² = Σ t² + E(curve) − 2 R(curve),   R = Σ t · curve,
//
// and Σ t² is the same for every curve, so the best curve is the argmin of
// E − 2R. The production codebooks are signed rotations of four base curves
// (q0_v_tables.cuh, pinned by q0_v_curve_structure.rs), so E depends only on
// the base, and a bucket's negation (buckets 4–7) negates R exactly. The 128
// curves are therefore 64 (base, phase) correlations, each scored for both
// signs. A lane runs two of them start to finish (pairs lane and lane + 32),
// every one of the 128 candidates is scored, and one warp argmin picks the
// winner. The previous search spent a five-shuffle warp reduction per
// scored curve.
//
// A pair reads its 32 base values from the float copy of the doubled row
// (element e of phase p is entry e + 2p, and 2p is even, so the run is 16
// aligned float2 loads), and E from the per-base energy table.
//
// R is four interleaved FMA chains — element e accumulates into chain e mod 4,
// `acc[e & 3] = fma(t[e], v, acc[e & 3])` in element order — summed as
// (acc0 + acc1) + (acc2 + acc3). Four chains of eight rather than one of 32
// is a quarter of the dependent latency, and the reference sums the same way,
// so each score is the reference's to the bit. For a negated curve every
// chain, and so the sum, is the positive one negated (round-to-nearest is
// sign-symmetric), and fma(2, R, E) is exactly the reference's fma(−2, −R, E).
template <bool IS_K>
__device__ __forceinline__ int find_best_curve_exhaustive(float target_scaled, int lane)
{
    float t[32];
    #pragma unroll
    for (int e = 0; e < 32; ++e) t[e] = __shfl_sync(0xffffffff, target_scaled, e, 32);

    float best_err = INFINITY;
    int   best_idx = 0;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int k = lane + 32 * h;          // pair (base k >> 4, phase k & 15)
        const int p = k & 15;
        const float2* row = reinterpret_cast<const float2*>(Q0VDecodeTables<IS_K>::base2f(k >> 4)) + p;
        float acc[4] = { 0.f, 0.f, 0.f, 0.f };
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float2 v = __ldg(row + i);
            acc[(2 * i) & 3]     = __fmaf_rn(t[2 * i], v.x, acc[(2 * i) & 3]);
            acc[(2 * i + 1) & 3] = __fmaf_rn(t[2 * i + 1], v.y, acc[(2 * i + 1) & 3]);
        }
        const float r = __fadd_rn(__fadd_rn(acc[0], acc[1]), __fadd_rn(acc[2], acc[3]));
        const float e = (float)Q0VDecodeTables<IS_K>::energy(k >> 4);  // ≤ 32 · 127², exact
        const float err_pos = __fmaf_rn(-2.f, r, e);
        const float err_neg = __fmaf_rn(2.f, r, e);
        // The positive curve's index is k; its negation's is k + 64. A tie
        // keeps the positive one, the lower index.
        const bool neg = err_neg < err_pos;
        const float err = neg ? err_neg : err_pos;
        const int   idx = neg ? k + 64 : k;
        if (err < best_err || (err == best_err && idx < best_idx)) { best_err = err; best_idx = idx; }
    }
    return warp_argmin_idx(best_err, best_idx);
}

// =============================================================================
// Step 4 (runtime codebooks): hierarchical curve search — Stage A (8 bucket
//         reps) + Stage B (16 phases of best bucket) + peak-bin refinement
//         (±1 lane window).
// =============================================================================
// A codebook supplied at launch carries no guarantee of the rotated-base
// structure the exhaustive search depends on, so it is searched curve by
// curve: 8 buckets × 16 phases (bucket-major), scored by warp reduction.
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
// Top-level per-block encoders
// =============================================================================
// Steps 1–3 are shared; the production codebooks search exhaustively, the
// runtime ones hierarchically. Both return the three indexes in every lane.
template <typename Tables>
__device__ __forceinline__ void per_block_encode_runtime(
    float xi, int lane, const Tables& tbl,
    int& curve_idx, int& scale_idx, int& centroid_idx)
{
    float target_scaled;
    compute_target_and_indices(xi, tbl, target_scaled, scale_idx, centroid_idx);
    curve_idx = find_best_curve_hierarchical(target_scaled, lane, tbl);
}

template <bool IS_K>
__device__ __forceinline__ void per_block_encode(
    float xi, int lane,
    int& curve_idx, int& scale_idx, int& centroid_idx)
{
    Q0VTablesStatic<IS_K> tbl;
    float target_scaled;
    compute_target_and_indices(xi, tbl, target_scaled, scale_idx, centroid_idx);
    curve_idx = find_best_curve_exhaustive<IS_K>(target_scaled, lane);
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
    q0_v_detail::per_block_encode_runtime(xi, lane, tbl, curve_idx, scale_idx, centroid_idx);
    if (lane == 0) {
        q0_v_detail::q0_v_pack_block(dst, curve_idx, scale_idx, centroid_idx);
    }
}

// IS_K has no default in any encoder: a block must be encoded under the
// codebook of the side that will decode it, and a caller that does not know
// its side must not compile.
template <bool IS_K>
__device__ __forceinline__ void quantize_block_q0_v_vec(
    const float* __restrict__ src, block_q0_v* __restrict__ dst)
{
    const int lane = threadIdx.x & 31;
    quantize_block_q0_v_core<IS_K>(src[lane], dst);
}

template <bool IS_K>
__device__ __forceinline__ void quantize_block_q0_v(
    const float* __restrict__ src, block_q0_v* __restrict__ dst)
{
    const int lane = threadIdx.x & 31;
    quantize_block_q0_v_core<IS_K>(src[lane], dst);
}

template <bool IS_K, int BLOCKS_PER_WARP = 1>
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
