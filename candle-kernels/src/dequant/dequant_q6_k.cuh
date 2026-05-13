// SPDX-License-Identifier: MIT

// ============================================================================
// Q6_K: 256 elements, 6-bit quantization
// Sub-block size: 16 elements, each with scale from scales array
// Complex bit packing: 4 bits in ql, 2 bits in qh
//
// WARP-PARALLEL: Process 32 elements (2 subblocks) at once
// Lanes 0-15 handle subblock j, lanes 16-31 handle subblock j+1
// Each pair of subblocks has its own scale from scales[j] and scales[j+1]
// ============================================================================

#pragma once

// Core templated dequant - generic output (warp-parallel, 2 subblocks)
template <int N = 32, typename compute_t>
__device__ __forceinline__ void dequantize_q6_K_dual_subblock(
    const uint8_t* __restrict__ ql,
    const uint8_t* __restrict__ qh,
    const float dl0,
    const float dl1,
    const int subblock_offset,
    compute_t* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    const int lane_id = threadIdx.x % WARP_SIZE;
    constexpr int ELEMS_PER_LANE = N / WARP_SIZE;
    
    // Pre-compute -32 * dl for FMA: dl * q + (-32 * dl)
    const float dl = (lane_id < 16) ? dl0 : dl1;
    const float neg_32_dl = -32.0f * dl;
    
    #pragma unroll
    for (int iter = 0; iter < ELEMS_PER_LANE; ++iter) {
        const int local_idx = lane_id + iter * WARP_SIZE;
        const int idx = subblock_offset + local_idx;
        const int ql_idx = idx / 2;
        const int qh_idx = idx / 4;
        
        const uint8_t ql_byte = ql[ql_idx];
        const uint8_t qh_byte = qh[qh_idx];
        
        const int shift_l = (idx & 1) * 4;
        const int shift_h = (idx & 3) * 2;
        
        const int q = ((ql_byte >> shift_l) & 0xF) | (((qh_byte >> shift_h) & 3) << 4);
        // Use FMA: dl * q + neg_32_dl = dl * (q - 32)
        dst[local_idx] = from_f32<compute_t>(__fmaf_rn(dl, float(q), neg_32_dl));
    }
}

// Core templated dequant - float output (warp-parallel, 2 subblocks)
template <int N = 32>
__device__ __forceinline__ void dequantize_q6_K_dual_subblock(
    const uint8_t* __restrict__ ql,
    const uint8_t* __restrict__ qh,
    const float dl0,
    const float dl1,
    const int subblock_offset,
    float* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    const int lane_id = threadIdx.x % WARP_SIZE;
    constexpr int ELEMS_PER_LANE = N / WARP_SIZE;
    
    // Pre-compute -32 * dl for FMA
    const float dl = (lane_id < 16) ? dl0 : dl1;
    const float neg_32_dl = -32.0f * dl;
    
    #pragma unroll
    for (int iter = 0; iter < ELEMS_PER_LANE; ++iter) {
        const int local_idx = lane_id + iter * WARP_SIZE;
        const int idx = subblock_offset + local_idx;
        const int ql_idx = idx / 2;
        const int qh_idx = idx / 4;
        
        const uint8_t ql_byte = ql[ql_idx];
        const uint8_t qh_byte = qh[qh_idx];
        
        const int shift_l = (idx & 1) * 4;
        const int shift_h = (idx & 3) * 2;
        
        const int q = ((ql_byte >> shift_l) & 0xF) | (((qh_byte >> shift_h) & 3) << 4);
        dst[local_idx] = __fmaf_rn(dl, float(q), neg_32_dl);
    }
}

// Block wrapper - generic (processes 2 subblocks per iteration)
template <typename compute_t>
__device__ __forceinline__ void dequantize_block_q6_K(
    const block_q6_K* __restrict__ src,
    compute_t* __restrict__ dst) {
    const float d = __half2float(src->d);
    
    // 256 elements / 32 per iteration = 8 iterations
    #pragma unroll 1
    for (int j = 0; j < QK_K/32; ++j) {
        const float dl0 = d * src->scales[j * 2];
        const float dl1 = d * src->scales[j * 2 + 1];
        dequantize_q6_K_dual_subblock<32>(src->ql, src->qh, dl0, dl1, j * 32, dst + j * 32);
    }
}

// Block wrapper - float (processes 2 subblocks per iteration)
__device__ __forceinline__ void dequantize_block_q6_K(
    const block_q6_K* __restrict__ src,
    float* __restrict__ dst) {
    const float d = __half2float(src->d);
    
    // 256 elements / 32 per iteration = 8 iterations
    #pragma unroll 1
    for (int j = 0; j < QK_K/32; ++j) {
        const float dl0 = d * src->scales[j * 2];
        const float dl1 = d * src->scales[j * 2 + 1];
        dequantize_q6_K_dual_subblock<32>(src->ql, src->qh, dl0, dl1, j * 32, dst + j * 32);
    }
}
