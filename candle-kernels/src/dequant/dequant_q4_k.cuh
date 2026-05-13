// SPDX-License-Identifier: MIT

// ============================================================================
// Q4_K: 256 elements, 4-bit quantization with block scales
// THIS IS THE MOST COMMON FORMAT - HEAVILY OPTIMIZED
// Super-block size: 64 elements (32 low nibbles + 32 high nibbles)
// Scale packing: 3 bytes per 64-element super-block
//
// WARP-PARALLEL: 64 elements / 32 lanes = 2 elements per lane
// Low 32 elements use d0/m0, high 32 elements use d1/m1
//
// LUT OPTIMIZATION:
// Instead of computing scale * nibble - min for each element (64 FMAs),
// we build a 16-entry LUT per half (32 FMAs total), then lookup via shuffle.
// Lanes 0-15 each hold LUT[lane_id] = scale * lane_id - min
// Any lane can lookup nibble N by shuffling from lane N.
// ============================================================================

#pragma once

#if QK_K == 256
// ============================================================================
// HELPER: Extract scale and min from packed Q4_K scales array
// ============================================================================
// Q4_K scales are packed into 12 bytes for 8 sub-blocks (256 elements / 32 = 8)
// Each sub-block has a 6-bit scale (d) and 6-bit min (m)
//
// For j < 4: straightforward extraction
// For j >= 4: bits are split across multiple bytes
//
// This matches the llama.cpp/GGML reference implementation.
static inline __device__ void get_scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63; 
        m = q[j + 4] & 63;
    } else {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

// Core templated dequant - generic output (warp-parallel with LUT)
template <int N = 64, typename compute_t>
__device__ __forceinline__ void dequantize_q4_K_superblock(
    const uint8_t* __restrict__ qs,
    const float d0,
    const float d1,
    const float m0,
    const float m1,
    compute_t* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    static_assert(N == 64, "LUT optimization assumes 64-element super-blocks");
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Build LUT in registers - branchless, all lanes compute
    // LUT_lo[i] = d0 * i - m0, LUT_hi[i] = d1 * i - m1
    // lane_id & 15 maps lanes 16-31 to 0-15 (redundant but avoids divergence)
    const int lut_idx = lane_id & 15;
    const float idx_f = float(lut_idx);
    const float lut_lo = __fmaf_rn(d0, idx_f, -m0);
    const float lut_hi = __fmaf_rn(d1, idx_f, -m1);
    
    // Load once, extract both nibbles
    const uint8_t q = qs[lane_id];
    const int nibble_lo = q & 0xF;
    const int nibble_hi = q >> 4;
    
    // Shuffle-based LUT lookup - low half (elements 0-31)
    dst[lane_id] = from_f32<compute_t>(__shfl_sync(0xffffffff, lut_lo, nibble_lo));
    
    // Shuffle-based LUT lookup - high half (elements 32-63)
    dst[32 + lane_id] = from_f32<compute_t>(__shfl_sync(0xffffffff, lut_hi, nibble_hi));
}

// Core templated dequant - float output (warp-parallel with LUT)
template <int N = 64>
__device__ __forceinline__ void dequantize_q4_K_superblock(
    const uint8_t* __restrict__ qs,
    const float d0,
    const float d1,
    const float m0,
    const float m1,
    float* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    static_assert(N == 64, "LUT optimization assumes 64-element super-blocks");
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Build LUT in registers - branchless, all lanes compute
    // lane_id & 15 maps lanes 16-31 to 0-15 (redundant but avoids divergence)
    const int lut_idx = lane_id & 15;
    const float idx_f = float(lut_idx);
    const float lut_lo = __fmaf_rn(d0, idx_f, -m0);
    const float lut_hi = __fmaf_rn(d1, idx_f, -m1);
    
    // Load once, extract both nibbles
    const uint8_t q = qs[lane_id];
    const int nibble_lo = q & 0xF;
    const int nibble_hi = q >> 4;
    
    // Shuffle-based LUT lookup
    dst[lane_id] = __shfl_sync(0xffffffff, lut_lo, nibble_lo);
    dst[32 + lane_id] = __shfl_sync(0xffffffff, lut_hi, nibble_hi);
}

// Core templated dequant - half output (warp-parallel with LUT)
template <int N = 64>
__device__ __forceinline__ void dequantize_q4_K_superblock(
    const uint8_t* __restrict__ qs,
    const half2 d0_2,
    const half2 d1_2,
    const half2 neg_m0_2,
    const half2 neg_m1_2,
    __half* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    static_assert(N == 64, "LUT optimization assumes 64-element super-blocks");
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Extract scalar values for LUT computation
    const float d0 = __low2float(d0_2);
    const float d1 = __low2float(d1_2);
    const float neg_m0 = __low2float(neg_m0_2);
    const float neg_m1 = __low2float(neg_m1_2);
    
    // Build LUT in registers - branchless, all lanes compute
    // Note: neg_m means we ADD it (formula is scale * nibble + neg_m)
    // lane_id & 15 maps lanes 16-31 to 0-15 (redundant but avoids divergence)
    const int lut_idx = lane_id & 15;
    const float idx_f = float(lut_idx);
    const float lut_lo = __fmaf_rn(d0, idx_f, neg_m0);
    const float lut_hi = __fmaf_rn(d1, idx_f, neg_m1);
    
    // Load once, extract both nibbles
    const uint8_t q = qs[lane_id];
    const int nibble_lo = q & 0xF;
    const int nibble_hi = q >> 4;
    
    // Shuffle-based LUT lookup
    dst[lane_id] = __float2half(__shfl_sync(0xffffffff, lut_lo, nibble_lo));
    dst[32 + lane_id] = __float2half(__shfl_sync(0xffffffff, lut_hi, nibble_hi));
}

// Core templated dequant - FP8 output (warp-parallel with LUT)
template <int N = 64>
__device__ __forceinline__ void dequantize_q4_K_superblock(
    const uint8_t* __restrict__ qs,
    const float d0,
    const float d1,
    const float m0,
    const float m1,
    __nv_fp8_e4m3* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    static_assert(N == 64, "LUT optimization assumes 64-element super-blocks");
    const int lane_id = threadIdx.x % WARP_SIZE;
    
    // Build LUT in registers - branchless, all lanes compute
    // lane_id & 15 maps lanes 16-31 to 0-15 (redundant but avoids divergence)
    //
    // FP8 e4m3 range: [-448, 448]. Q4_K dequant: d * nibble - m where nibble ∈ [0,15]
    // For well-quantized LLM weights, dequantized values are typically in [-2, 2] range
    // (weights are normalized). Values exceeding ±448 would indicate pathological
    // quantization. __nv_fp8_e4m3() saturates gracefully - no NaN/inf, just clips.
    const int lut_idx = lane_id & 15;
    const float idx_f = float(lut_idx);
    const float lut_lo = __fmaf_rn(d0, idx_f, -m0);
    const float lut_hi = __fmaf_rn(d1, idx_f, -m1);
    
    // Load once, extract both nibbles
    const uint8_t q = qs[lane_id];
    const int nibble_lo = q & 0xF;
    const int nibble_hi = q >> 4;
    
    // Shuffle-based LUT lookup
    dst[lane_id] = __nv_fp8_e4m3(__shfl_sync(0xffffffff, lut_lo, nibble_lo));
    dst[32 + lane_id] = __nv_fp8_e4m3(__shfl_sync(0xffffffff, lut_hi, nibble_hi));
}

// Block wrapper - generic
template <typename compute_t>
__device__ __forceinline__ void dequantize_block_q4_K(
    const block_q4_K* __restrict__ src,
    compute_t* __restrict__ dst) {
    const float d = __low2float(src->dm);
    const float dmin = __high2float(src->dm);
    
    #pragma unroll 1
    for (int j = 0; j < QK_K/64; ++j) {  // j = 0,1,2,3 for 4 superblocks
        const int is = 2 * j;  // scale index: 0,2,4,6
        
        // Decode scales for this 64-element superblock (2 sub-blocks of 32)
        uint8_t sc0, m0, sc1, m1;
        get_scale_min_k4(is + 0, src->scales, sc0, m0);
        get_scale_min_k4(is + 1, src->scales, sc1, m1);
        
        const float d0 = d * sc0;
        const float d1 = d * sc1;
        const float min0 = dmin * m0;
        const float min1 = dmin * m1;
        
        dequantize_q4_K_superblock<64>(src->qs + j * 32, d0, d1, min0, min1, dst + j * 64);
    }
}

// Block wrapper - float
__device__ __forceinline__ void dequantize_block_q4_K(
    const block_q4_K* __restrict__ src,
    float* __restrict__ dst) {
    const float d = __low2float(src->dm);
    const float dmin = __high2float(src->dm);
    
    #pragma unroll 1
    for (int j = 0; j < QK_K/64; ++j) {
        const int is = 2 * j;
        
        uint8_t sc0, m0, sc1, m1;
        get_scale_min_k4(is + 0, src->scales, sc0, m0);
        get_scale_min_k4(is + 1, src->scales, sc1, m1);
        
        const float d0 = d * sc0;
        const float d1 = d * sc1;
        const float min0 = dmin * m0;
        const float min1 = dmin * m1;
        
        dequantize_q4_K_superblock<64>(src->qs + j * 32, d0, d1, min0, min1, dst + j * 64);
    }
}

// Block wrapper - half
__device__ __forceinline__ void dequantize_block_q4_K(
    const block_q4_K* __restrict__ src,
    half* __restrict__ dst) {
    const float d = __low2float(src->dm);
    const float dmin = __high2float(src->dm);
    
    #pragma unroll 1
    for (int j = 0; j < QK_K/64; ++j) {
        const int is = 2 * j;
        
        uint8_t sc0, m0, sc1, m1;
        get_scale_min_k4(is + 0, src->scales, sc0, m0);
        get_scale_min_k4(is + 1, src->scales, sc1, m1);
        
        const half2 d0_2 = __half2half2(__float2half(d * sc0));
        const half2 d1_2 = __half2half2(__float2half(d * sc1));
        const half2 neg_m0_2 = __half2half2(__float2half(-(dmin * m0)));
        const half2 neg_m1_2 = __half2half2(__float2half(-(dmin * m1)));
        
        dequantize_q4_K_superblock<64>(src->qs + j * 32, d0_2, d1_2, neg_m0_2, neg_m1_2, dst + j * 64);
    }
}

// Block wrapper - FP8
__device__ __forceinline__ void dequantize_block_q4_K(
    const block_q4_K* __restrict__ src,
    __nv_fp8_e4m3* __restrict__ dst) {
    const float d = __low2float(src->dm);
    const float dmin = __high2float(src->dm);
    
    #pragma unroll 1
    for (int j = 0; j < QK_K/64; ++j) {
        const int is = 2 * j;
        
        uint8_t sc0, m0, sc1, m1;
        get_scale_min_k4(is + 0, src->scales, sc0, m0);
        get_scale_min_k4(is + 1, src->scales, sc1, m1);
        
        const float d0 = d * sc0;
        const float d1 = d * sc1;
        const float min0 = dmin * m0;
        const float min1 = dmin * m1;
        
        dequantize_q4_K_superblock<64>(src->qs + j * 32, d0, d1, min0, min1, dst + j * 64);
    }
}

#else  // QK_K != 256 (K/64)

// K/64 path: provide minimal dequant stubs to keep compilation green.
// These are not used by the Marlin matmul path.
template <typename compute_t>
__device__ __forceinline__ void dequantize_block_q4_K(
    const block_q4_K* __restrict__,
    compute_t* __restrict__ dst) {
    #pragma unroll
    for (int i = 0; i < QK_K; ++i) {
        dst[i] = from_f32<compute_t>(0.0f);
    }
}

__device__ __forceinline__ void dequantize_block_q4_K(
    const block_q4_K* __restrict__,
    float* __restrict__ dst) {
    #pragma unroll
    for (int i = 0; i < QK_K; ++i) {
        dst[i] = 0.0f;
    }
}

__device__ __forceinline__ void dequantize_block_q4_K(
    const block_q4_K* __restrict__,
    half* __restrict__ dst) {
    #pragma unroll
    for (int i = 0; i < QK_K; ++i) {
        dst[i] = __float2half(0.0f);
    }
}

__device__ __forceinline__ void dequantize_block_q4_K(
    const block_q4_K* __restrict__,
    __nv_fp8_e4m3* __restrict__ dst) {
    #pragma unroll
    for (int i = 0; i < QK_K; ++i) {
        dst[i] = __nv_fp8_e4m3(0.0f);
    }
}

#endif
