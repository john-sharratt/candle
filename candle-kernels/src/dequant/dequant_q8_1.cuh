// SPDX-License-Identifier: MIT

// ============================================================================
// Q8_1 DEQUANTIZATION (for Y vectors) - templated output type
// 32 elements per block, similar to Q8_0 but with sum field
//
// WARP-PARALLEL: All functions use lane_id for parallel dequantization
// ============================================================================

#pragma once

// Core templated dequant - generic output (warp-parallel)
template <int N = QK8_1, typename compute_t>
__device__ __forceinline__ void dequantize_q8_1(
    const int8_t* __restrict__ qs,
    const __half d,
    compute_t* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    const int lane_id = threadIdx.x % WARP_SIZE;
    const float df = __half2float(d);
    constexpr int ELEMS_PER_LANE = N / WARP_SIZE;
    
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; ++i) {
        const int elem_idx = lane_id + i * WARP_SIZE;
        dst[elem_idx] = from_f32<compute_t>(qs[elem_idx] * df);
    }
}

// Core templated dequant - float output (warp-parallel)
template <int N = QK8_1>
__device__ __forceinline__ void dequantize_q8_1(
    const int8_t* __restrict__ qs,
    const __half d,
    float* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    const int lane_id = threadIdx.x % WARP_SIZE;
    const float df = __half2float(d);
    constexpr int ELEMS_PER_LANE = N / WARP_SIZE;
    
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; ++i) {
        const int elem_idx = lane_id + i * WARP_SIZE;
        dst[elem_idx] = qs[elem_idx] * df;
    }
}

// Core templated dequant - half output (warp-parallel)
template <int N = QK8_1>
__device__ __forceinline__ void dequantize_q8_1(
    const int8_t* __restrict__ qs,
    const __half d,
    __half* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    const int lane_id = threadIdx.x % WARP_SIZE;
    const float df = __half2float(d);
    constexpr int ELEMS_PER_LANE = N / WARP_SIZE;
    
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; ++i) {
        const int elem_idx = lane_id + i * WARP_SIZE;
        dst[elem_idx] = __float2half(qs[elem_idx] * df);
    }
}

// Core templated dequant - FP8 output (warp-parallel)
template <int N = QK8_1>
__device__ __forceinline__ void dequantize_q8_1(
    const int8_t* __restrict__ qs,
    const __half d,
    __nv_fp8_e4m3* __restrict__ dst) {
    static_assert(N % WARP_SIZE == 0, "N must be multiple of WARP_SIZE");
    const int lane_id = threadIdx.x % WARP_SIZE;
    const float df = __half2float(d);
    constexpr int ELEMS_PER_LANE = N / WARP_SIZE;
    
    #pragma unroll
    for (int i = 0; i < ELEMS_PER_LANE; ++i) {
        const int elem_idx = lane_id + i * WARP_SIZE;
        dst[elem_idx] = __nv_fp8_e4m3(qs[elem_idx] * df);
    }
}

// Block wrappers - extract qs/d from block_q8_1 and call core functions
template <typename compute_t>
__device__ __forceinline__ void dequantize_block_q8_1(
    const block_q8_1* __restrict__ src,
    compute_t* __restrict__ dst) {
    dequantize_q8_1<QK8_1>(src->qs, __low2half(src->ds), dst);
}

__device__ __forceinline__ void dequantize_block_q8_1(
    const block_q8_1* __restrict__ src,
    float* __restrict__ dst) {
    dequantize_q8_1<QK8_1>(src->qs, __low2half(src->ds), dst);
}

__device__ __forceinline__ void dequantize_block_q8_1(
    const block_q8_1* __restrict__ src,
    __half* __restrict__ dst) {
    dequantize_q8_1<QK8_1>(src->qs, __low2half(src->ds), dst);
}

__device__ __forceinline__ void dequantize_block_q8_1(
    const block_q8_1* __restrict__ src,
    __nv_fp8_e4m3* __restrict__ dst) {
    dequantize_q8_1<QK8_1>(src->qs, __low2half(src->ds), dst);
}

// Legacy alias for compatibility
__device__ __forceinline__ void dequantize_q8_1_to_half(
    const block_q8_1* __restrict__ src,
    half* __restrict__ dst) {
    dequantize_block_q8_1(src, dst);
}
