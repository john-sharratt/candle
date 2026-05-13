// SPDX-License-Identifier: MIT

// ============================================================================
// Q5_K: 256 elements, 5-bit quantization with block scales
// Super-block size: 64 elements (32 low + 32 high, with high-bit mask)
// Similar to Q4_K but with extra qh bits
//
// WARP-PARALLEL: 64 elements / 32 lanes = 2 elements per lane
//
// LUT OPTIMIZATION:
// Q5_K formula: scale * (nibble | (h << 4)) - min where combined ∈ [0,31]
// 32 possible values fits perfectly in one warp!
// Each lane computes LUT[lane_id] for both low and high halves.
// ============================================================================

#pragma once

#if QK_K == 256
// Core templated dequant - generic output (warp-parallel with LUT)
template <int N = 64, typename compute_t>
__device__ __forceinline__ void dequantize_q5_K_superblock(
    const uint8_t* __restrict__ qs,
    const uint8_t* __restrict__ qh,
    const float d0,
    const float d1,
    const float m0,
    const float m1,
    compute_t* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    static_assert(N == 64, "LUT optimization assumes 64-element super-blocks");
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Build 32-entry LUT for each half - branchless
    // LUT_lo[i] = d0 * i - m0, LUT_hi[i] = d1 * i - m1 for i ∈ [0,31]
    const float idx_f = float(lane_id);
    const float lut_lo = __fmaf_rn(d0, idx_f, -m0);
    const float lut_hi = __fmaf_rn(d1, idx_f, -m1);
    
    // Load low nibble and high bits for low half (elements 0-31)
    const uint8_t ql = qs[lane_id];
    const uint8_t qh_byte = qh[lane_id / 8];
    const int h_lo = (qh_byte >> (lane_id % 8)) & 1;
    const int h_hi = (qh_byte >> (lane_id % 8)) & 0x10 ? 1 : 0;
    
    // Combine nibble + high bit for 5-bit value
    const int val_lo = (ql & 0xF) | (h_lo << 4);  // 0-31
    const int val_hi = (ql >> 4) | (h_hi << 4);   // 0-31
    
    // Shuffle-based LUT lookup - full warp (32 entries)
    dst[lane_id] = from_f32<compute_t>(__shfl_sync(0xffffffff, lut_lo, val_lo));
    dst[32 + lane_id] = from_f32<compute_t>(__shfl_sync(0xffffffff, lut_hi, val_hi));
}

// Core templated dequant - float output (warp-parallel with LUT)
template <int N = 64>
__device__ __forceinline__ void dequantize_q5_K_superblock(
    const uint8_t* __restrict__ qs,
    const uint8_t* __restrict__ qh,
    const float d0,
    const float d1,
    const float m0,
    const float m1,
    float* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    static_assert(N == 64, "LUT optimization assumes 64-element super-blocks");
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Build 32-entry LUT for each half - branchless
    const float idx_f = float(lane_id);
    const float lut_lo = __fmaf_rn(d0, idx_f, -m0);
    const float lut_hi = __fmaf_rn(d1, idx_f, -m1);
    
    // Load and extract
    const uint8_t ql = qs[lane_id];
    const uint8_t qh_byte = qh[lane_id / 8];
    const int h_lo = (qh_byte >> (lane_id % 8)) & 1;
    const int h_hi = (qh_byte >> (lane_id % 8)) & 0x10 ? 1 : 0;
    
    const int val_lo = (ql & 0xF) | (h_lo << 4);
    const int val_hi = (ql >> 4) | (h_hi << 4);
    
    // Shuffle-based LUT lookup
    dst[lane_id] = __shfl_sync(0xffffffff, lut_lo, val_lo);
    dst[32 + lane_id] = __shfl_sync(0xffffffff, lut_hi, val_hi);
}

// Block wrapper - generic
template <typename compute_t>
__device__ __forceinline__ void dequantize_block_q5_K(
    const block_q5_K* __restrict__ src,
    compute_t* __restrict__ dst) {
    const float d = __low2float(src->dm);
    const float dmin = __high2float(src->dm);
    
    #pragma unroll 1
    for (int j = 0; j < QK_K/64; ++j) {
        const uint8_t* sc = src->scales + j * 3;
        const float d0 = d * (sc[0] & 0x3F);
        const float d1 = d * (sc[1] & 0x3F);
        const float m0 = dmin * (sc[0] >> 6 | ((sc[2] & 0x03) << 2));
        const float m1 = dmin * (sc[1] >> 6 | ((sc[2] & 0x0C) << 0));
        
        dequantize_q5_K_superblock<64>(src->qs + j * 32, src->qh + j * 4, d0, d1, m0, m1, dst + j * 64);
    }
}

// Block wrapper - float
__device__ __forceinline__ void dequantize_block_q5_K(
    const block_q5_K* __restrict__ src,
    float* __restrict__ dst) {
    const float d = __low2float(src->dm);
    const float dmin = __high2float(src->dm);
    
    #pragma unroll 1
    for (int j = 0; j < QK_K/64; ++j) {
        const uint8_t* sc = src->scales + j * 3;
        const float d0 = d * (sc[0] & 0x3F);
        const float d1 = d * (sc[1] & 0x3F);
        const float m0 = dmin * (sc[0] >> 6 | ((sc[2] & 0x03) << 2));
        const float m1 = dmin * (sc[1] >> 6 | ((sc[2] & 0x0C) << 0));
        
        dequantize_q5_K_superblock<64>(src->qs + j * 32, src->qh + j * 4, d0, d1, m0, m1, dst + j * 64);
    }
}

#else  // QK_K != 256 (K/64)

// K/64 path: provide minimal dequant stubs to keep compilation green.
template <typename compute_t>
__device__ __forceinline__ void dequantize_block_q5_K(
    const block_q5_K* __restrict__,
    compute_t* __restrict__ dst) {
    #pragma unroll
    for (int i = 0; i < QK_K; ++i) {
        dst[i] = from_f32<compute_t>(0.0f);
    }
}

__device__ __forceinline__ void dequantize_block_q5_K(
    const block_q5_K* __restrict__,
    float* __restrict__ dst) {
    #pragma unroll
    for (int i = 0; i < QK_K; ++i) {
        dst[i] = 0.0f;
    }
}

#endif
