// SPDX-License-Identifier: MIT

// ============================================================================
// K-QUANT DEQUANTIZATION (QK_K = 256) - output to compute_t
// These are larger blocks - use limited unrolling to avoid PTX bloat
// K-quants are the primary format for production LLMs (llama.cpp)
// 
// VECTORIZATION STRATEGY:
// - Use float2 stores for float path (64-bit coalesced writes)
// - Use half2 intrinsics for half path (2 ops/instruction)
// - Limit outer loop unrolling to prevent PTX bloat
//
// TEMPLATING STRATEGY for K-quants:
// K-quants have 256 elements with 16-element sub-blocks, each with own scale.
// Core functions take: qs, scales array, d/dmin, element count N
// N must be multiple of 16 (sub-block size) for Q2_K/Q6_K
// N must be multiple of 64 (super-block size) for Q4_K/Q5_K
// ============================================================================

// ============================================================================
// Q2_K: 256 elements, 2-bit quantization with scales
// Sub-block size: 16 elements, each with packed scale byte (4-bit d, 4-bit m)
//
// WARP-PARALLEL: 256 elements / 32 lanes = 8 elements per lane
// Scale lookup: scales[elem_idx / 16] for 16-element sub-blocks
//
// LUT OPTIMIZATION:
// Q2_K formula: dl * val - ml where val ∈ [0,3] (only 4 values!)
// We can use a 4-entry LUT per subblock, but since there are 16 subblocks
// with different scales, a full LUT approach is complex.
// Instead, use FMA: dl * val + (-ml) for efficiency.
// ============================================================================

#pragma once

// Core templated dequant - generic output (warp-parallel with FMA)
// N must be multiple of 16, processes N/16 sub-blocks
template <int N = QK_K, typename compute_t>
__device__ __forceinline__ void dequantize_q2_K(
    const uint8_t* __restrict__ qs,
    const uint8_t* __restrict__ scales,
    const float d,
    const float dmin,
    compute_t* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    const int lane_id = threadIdx.x % WARP_SIZE;
    constexpr int ELEMS_PER_LANE = N / WARP_SIZE;
    
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; ++i) {
        const int elem_idx = lane_id + i * WARP_SIZE;
        const int subblock = elem_idx / 16;
        const uint8_t sc = scales[subblock];
        const float dl = d * (sc & 0xF);
        const float neg_ml = -(dmin * (sc >> 4));  // Pre-negate for FMA
        
        const int byte_idx = elem_idx / 4;
        const int shift = (elem_idx & 3) * 2;
        const uint8_t q = qs[byte_idx];
        const int val = (q >> shift) & 3;
        // FMA: dl * val + neg_ml = dl * val - ml
        dst[elem_idx] = from_f32<compute_t>(__fmaf_rn(dl, float(val), neg_ml));
    }
}

// Core templated dequant - float output (warp-parallel with FMA)
template <int N = QK_K>
__device__ __forceinline__ void dequantize_q2_K(
    const uint8_t* __restrict__ qs,
    const uint8_t* __restrict__ scales,
    const float d,
    const float dmin,
    float* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    const int lane_id = threadIdx.x % WARP_SIZE;
    constexpr int ELEMS_PER_LANE = N / WARP_SIZE;
    
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; ++i) {
        const int elem_idx = lane_id + i * WARP_SIZE;
        const int subblock = elem_idx / 16;
        const uint8_t sc = scales[subblock];
        const float dl = d * (sc & 0xF);
        const float neg_ml = -(dmin * (sc >> 4));
        
        const int byte_idx = elem_idx / 4;
        const int shift = (elem_idx & 3) * 2;
        const uint8_t q = qs[byte_idx];
        const int val = (q >> shift) & 3;
        dst[elem_idx] = __fmaf_rn(dl, float(val), neg_ml);
    }
}

// Block wrappers
template <typename compute_t>
__device__ __forceinline__ void dequantize_block_q2_K(
    const block_q2_K* __restrict__ src,
    compute_t* __restrict__ dst) {
    dequantize_q2_K<QK_K>(src->qs, src->scales, __low2float(src->dm), __high2float(src->dm), dst);
}

__device__ __forceinline__ void dequantize_block_q2_K(
    const block_q2_K* __restrict__ src,
    float* __restrict__ dst) {
    dequantize_q2_K<QK_K>(src->qs, src->scales, __low2float(src->dm), __high2float(src->dm), dst);
}
