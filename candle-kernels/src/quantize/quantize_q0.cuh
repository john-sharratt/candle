// SPDX-License-Identifier: MIT
// Q0 sub-bit palette quantization family (all blocks = 32 elements)
//   Q0:    0.03 BPE — one INT8 constant (block mean rounded to [-127,127])
//   (Q0_V moved to quantize_q0_v.cuh — pattern-indexed structured format)
//   (Q1_A moved to quantize_q1_a.cuh — sibling of Q1_S)
//   Q0_X:  0.50 BPE — INT8 bulk anchor + 5-bit outlier idx + 3-bit outlier delta
//   Q0_M2: 0.75 BPE — two INT8 constants + 8-bit quartet mask (Lloyd ×4)
//   Q0_M4: 2.00 BPE — four INT8 constants + 32-bit pair mask (Lloyd ×5)
//
// No per-block scale. Range is provided entirely by the palette outer scale.
// Encode convention: centroid = round(x * 127), x in [-1, 1] after outer applied.
// Decode convention: x = centroid / 127.0f, then divide by outer.
//
// NOTE: Include via quantize.cuh which defines the block structs.
#pragma once

// ---------------------------------------------------------------------------
// Warp-wide reduction helpers (all 32 lanes, full mask)
// ---------------------------------------------------------------------------

__device__ __forceinline__ float q0_warp_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xffffffff, val, offset, 32);
    return val;
}

__device__ __forceinline__ float q0_warp_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, offset, 32));
    return val;
}

__device__ __forceinline__ float q0_warp_min(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fminf(val, __shfl_xor_sync(0xffffffff, val, offset, 32));
    return val;
}

// Encode a float centroid (already in [-1,1] after outer applied) to INT8.
__device__ __forceinline__ int8_t q0_encode_centroid(float val) {
    return (int8_t)__float2int_rn(fmaxf(-127.0f, fminf(127.0f, val * 127.0f)));
}

// ---------------------------------------------------------------------------
// Q0: single INT8 constant — block mean rounded to [-127, 127]
//
// MSE-optimal constant for a set of values is their mean, so no search needed.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void quantize_block_q0_core(
    float xi, float sum_x, block_q0* dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    if (lane == 0) {
        dst->centroid = q0_encode_centroid(sum_x * (1.0f / 32.0f));
    }
}

__device__ __forceinline__ void quantize_block_q0_vec(
    const float* __restrict__ src, block_q0* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const float xi = src[lane];
    quantize_block_q0_core(xi, q0_warp_sum(xi), dst);
}

__device__ __forceinline__ void quantize_block_q0(
    const float* __restrict__ src, block_q0* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const float xi = src[lane];
    quantize_block_q0_core(xi, q0_warp_sum(xi), dst);
}

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q0(
    const float* __restrict__ src, block_q0* __restrict__ dst, int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    for (int blk = warp_id + blockIdx.x * warps_per_block;
         blk < num_blocks;
         blk += warps_per_block * gridDim.x) {
        const float xi = src[blk * QK_Q0 + lane];
        quantize_block_q0_core(xi, q0_warp_sum(xi), dst + blk);
    }
}

// Q0_V is implemented in quantize_q0_v.cuh — it's a structured pattern-
// indexed format with group-level state (sign_pattern, shape, magnitude
// templates) rather than a fixed-split bipartition. See that file plus
// docs/archived/q0_v.md for the design.

// ---------------------------------------------------------------------------
// Q0_X: INT8 bulk anchor + one outlier escape (no Lloyd, single pass)
//
// Targets blocks that are nearly constant with one anomalous element:
// 31 lanes near the bulk mean, 1 lane (attention sink, content boundary,
// content-type spike) significantly off. The outlier is escaped via a
// signed 3-bit coarse delta scaled by Q0_X_S_OUTLIER.
//
// Pipeline:
//   1. bulk_anchor = round(mean(x) * 127), clamped to [-127, 127]
//   2. residual[i] = round(x[i] * 127) - bulk_anchor
//   3. outlier_idx = argmax(|residual|)  (first lane on ties)
//   4. outlier_delta = clamp(round(residual[outlier_idx] / S_OUTLIER), -4, 3)
//
// Cheaper than every other Q0_* family format — no centroid refinement.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void quantize_block_q0_x_core(
    float xi, block_q0_x* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;

    // 1. bulk_anchor = INT8-encoded block mean (full INT8 range)
    const float sum_x = q0_warp_sum(xi);
    const float mean  = sum_x * (1.0f / 32.0f);
    const int bulk_anchor_int = max(-127, min(127, __float2int_rn(mean * 127.0f)));
    const int8_t bulk_anchor  = (int8_t)bulk_anchor_int;

    // 2. Per-lane INT8 residual (target value − bulk_anchor)
    const int x_i8     = max(-127, min(127, __float2int_rn(xi * 127.0f)));
    const int residual = x_i8 - bulk_anchor_int;

    // 3. argmax(|residual|) → outlier_idx (first lane with the max)
    const int     abs_r   = abs(residual);
    const float   fmax    = q0_warp_max((float)abs_r);
    const int     max_int = (int)fmax;
    const unsigned ballot = __ballot_sync(0xffffffff, abs_r == max_int);
    const int     outlier_idx = __ffs(ballot) - 1;

    // 4. Outlier residual → signed 3-bit delta in [-4, 3]
    const int residual_at_outlier = __shfl_sync(0xffffffff, residual, outlier_idx, 32);
    const int delta_raw     = __float2int_rn((float)residual_at_outlier / (float)Q0_X_S_OUTLIER);
    const int outlier_delta = max(-4, min(3, delta_raw));

    // 5. Pack: byte 0 = bulk_anchor, byte 1 = [delta:3 | idx:5]
    if (lane == 0) {
        const uint8_t packed_idx   = (uint8_t)(outlier_idx & 0x1F);
        const uint8_t packed_delta = (uint8_t)((outlier_delta & 0x07) << 5);
        dst->bulk_anchor    = bulk_anchor;
        dst->outlier_packed = packed_idx | packed_delta;
    }
}

__device__ __forceinline__ void quantize_block_q0_x_vec(
    const float* __restrict__ src, block_q0_x* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    quantize_block_q0_x_core(src[lane], dst);
}
__device__ __forceinline__ void quantize_block_q0_x(
    const float* __restrict__ src, block_q0_x* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    quantize_block_q0_x_core(src[lane], dst);
}
template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q0_x(
    const float* __restrict__ src, block_q0_x* __restrict__ dst, int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    for (int blk = warp_id + blockIdx.x * warps_per_block;
         blk < num_blocks; blk += warps_per_block * gridDim.x)
        quantize_block_q0_x_core(src[blk * QK_Q0_X + lane], dst + blk);
}

// ---------------------------------------------------------------------------
// Q0_M2: two INT8 constants + 8-bit quartet mask  (Lloyd ×4)
//
// 32 elements → 8 quartets of 4.  Each quartet is assigned as a unit to
// c0 or c1 based on the quartet mean vs the two centroids.
//
// Lloyd's iterations operate at quartet granularity to match the format's
// constrained reconstruction (4 elements share one centroid). Per-element
// Lloyd would optimise a different objective and produce centroids that
// don't match the actual encoding.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void quantize_block_q0_m2_core(
    float xi, block_q0_m2* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;

    // Quartet mean: XOR-reduce within each group of 4 (all 4 lanes hold the
    // same value afterwards).
    float qsum = xi;
    qsum += __shfl_xor_sync(0xffffffff, qsum, 1, 32);
    qsum += __shfl_xor_sync(0xffffffff, qsum, 2, 32);
    const float qt_mean = qsum * 0.25f;

    // Initialise centroids from min / max of the 8 quartet means.
    // (The 4-way duplication across lanes is harmless under min/max.)
    float c0 = q0_warp_min(qt_mean);
    float c1 = q0_warp_max(qt_mean);

    // Lloyd iterations at quartet granularity. The centroid update sums
    // element values (xi) over lanes whose quartet is assigned to that
    // cluster — equal-size quartets make this equivalent to the mean of
    // assigned qt_means, but uses the existing all-lanes reduction path.
    #pragma unroll
    for (int iter = 0; iter < 4; iter++) {
        const int qt_assign = (fabsf(qt_mean - c1) < fabsf(qt_mean - c0)) ? 1 : 0;

        const float s0 = q0_warp_sum((qt_assign == 0) ? xi   : 0.0f);
        const float n0 = q0_warp_sum((qt_assign == 0) ? 1.0f : 0.0f);
        const float s1 = q0_warp_sum((qt_assign == 1) ? xi   : 0.0f);
        const float n1 = q0_warp_sum((qt_assign == 1) ? 1.0f : 0.0f);

        if (n0 > 0.0f) c0 = s0 / n0;
        if (n1 > 0.0f) c1 = s1 / n1;
    }

    // Final quartet assignment against the converged centroids.
    const int qt_assign = (fabsf(qt_mean - c1) < fabsf(qt_mean - c0)) ? 1 : 0;

    // Build 8-bit mask: bit i = assignment for quartet i (i = lane/4)
    uint32_t qmask = (uint32_t)qt_assign << (lane / 4);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        qmask |= __shfl_xor_sync(0xffffffff, qmask, offset, 32);

    if (lane == 0) {
        dst->centroid[0] = q0_encode_centroid(c0);
        dst->centroid[1] = q0_encode_centroid(c1);
        dst->qmask = (uint8_t)(qmask & 0xFF);
    }
}

__device__ __forceinline__ void quantize_block_q0_m2_vec(
    const float* __restrict__ src, block_q0_m2* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    quantize_block_q0_m2_core(src[lane], dst);
}
__device__ __forceinline__ void quantize_block_q0_m2(
    const float* __restrict__ src, block_q0_m2* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    quantize_block_q0_m2_core(src[lane], dst);
}
template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q0_m2(
    const float* __restrict__ src, block_q0_m2* __restrict__ dst, int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    for (int blk = warp_id + blockIdx.x * warps_per_block;
         blk < num_blocks; blk += warps_per_block * gridDim.x)
        quantize_block_q0_m2_core(src[blk * QK_Q0_M2 + lane], dst + blk);
}

// ---------------------------------------------------------------------------
// Q0_M4: four INT8 constants + 32-bit pair mask  (Lloyd ×5)
//
// 32 elements → 16 pairs of 2. Each pair is assigned as a unit to one of
// four centroids; qmask bits [2i+1 : 2i] hold the index for pair i.
//
// Lloyd's iterations operate at pair granularity to match the format's
// constrained reconstruction (2 elements share one centroid). Centroids
// initialised as equally-spaced from min to max of the 16 pair means.
// ---------------------------------------------------------------------------

__device__ __forceinline__ void quantize_block_q0_m4_core(
    float xi, block_q0_m4* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;

    // Pair mean: XOR-reduce within each group of 2 (both lanes hold result).
    float psum = xi;
    psum += __shfl_xor_sync(0xffffffff, psum, 1, 32);
    const float pair_mean = psum * 0.5f;

    // Initialise four centroids equally-spaced across the pair-mean range.
    const float vmin  = q0_warp_min(pair_mean);
    const float vmax  = q0_warp_max(pair_mean);
    const float step  = (vmax - vmin) * (1.0f / 3.0f);

    float c[4];
    c[0] = vmin;
    c[1] = vmin + step;
    c[2] = vmin + 2.0f * step;
    c[3] = vmax;

    // Lloyd iterations at pair granularity. Centroid update sums xi over
    // lanes whose pair is assigned to that cluster — equivalent to the
    // mean of assigned pair_means under the equal-size-pair invariant.
    #pragma unroll
    for (int iter = 0; iter < 5; iter++) {
        // Per-pair assignment to nearest centroid
        int pair_assign = 0;
        float best_d = fabsf(pair_mean - c[0]);
        for (int k = 1; k < 4; k++) {
            const float d = fabsf(pair_mean - c[k]);
            if (d < best_d) { best_d = d; pair_assign = k; }
        }

        // Recompute centroids (empty cluster → keep previous)
        for (int k = 0; k < 4; k++) {
            const float sk = q0_warp_sum((pair_assign == k) ? xi   : 0.0f);
            const float nk = q0_warp_sum((pair_assign == k) ? 1.0f : 0.0f);
            if (nk > 0.0f) c[k] = sk / nk;
        }
    }

    // Final pair assignment against the converged centroids.
    int pair_assign = 0;
    float pair_best_d = fabsf(pair_mean - c[0]);
    for (int k = 1; k < 4; k++) {
        const float d = fabsf(pair_mean - c[k]);
        if (d < pair_best_d) { pair_best_d = d; pair_assign = k; }
    }

    // Build 32-bit mask: bits [2i+1:2i] = pair_assign for pair i (i = lane/2)
    uint32_t qmask = (uint32_t)pair_assign << (2 * (lane / 2));
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        qmask |= __shfl_xor_sync(0xffffffff, qmask, offset, 32);

    if (lane == 0) {
        dst->centroid[0] = q0_encode_centroid(c[0]);
        dst->centroid[1] = q0_encode_centroid(c[1]);
        dst->centroid[2] = q0_encode_centroid(c[2]);
        dst->centroid[3] = q0_encode_centroid(c[3]);
        dst->qmask = qmask;
    }
}

__device__ __forceinline__ void quantize_block_q0_m4_vec(
    const float* __restrict__ src, block_q0_m4* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    quantize_block_q0_m4_core(src[lane], dst);
}
__device__ __forceinline__ void quantize_block_q0_m4(
    const float* __restrict__ src, block_q0_m4* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    quantize_block_q0_m4_core(src[lane], dst);
}
template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q0_m4(
    const float* __restrict__ src, block_q0_m4* __restrict__ dst, int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;
    for (int blk = warp_id + blockIdx.x * warps_per_block;
         blk < num_blocks; blk += warps_per_block * gridDim.x)
        quantize_block_q0_m4_core(src[blk * QK_Q0_M4 + lane], dst + blk);
}
