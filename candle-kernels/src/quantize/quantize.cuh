// SPDX-License-Identifier: MIT
// Quantize - Unified quantization header for float -> quantized formats
//
// This header provides warp-cooperative quantization kernels for converting
// floating point tensors to various quantized formats on GPU.
//
// Prerequisites: Block types (block_q8_0, block_q8_1, etc.) and
// constants (QK8_0, QK8_1, QK4_0, WARP_SIZE, QK_K) must be defined before including.
//
// Usage:
//   #include "quantize/quantize.cuh"
//
//   // Single block quantization (one warp, 32 threads):
//   quantize_block_q8_0(float_ptr, block_q8_0_ptr);
//   quantize_block_q4_K(float_ptr, block_q4_K_ptr);
//
//   // Multi-block quantization (grid of warps):
//   quantize_blocks_q8_0<1>(src, dst, num_blocks);
//   quantize_blocks_q4_K<1>(src, dst, num_blocks);
//
// All kernels:
//   - Are warp-cooperative (32 threads per quantization block)
//   - Use warp shuffle for reductions (no shared memory needed)
//   - Support arbitrary number of blocks via grid-stride loops
//
// Supported formats:
//   Standard (32 elements per block):
//     Q4_0: 4-bit symmetric, scale only
//     Q4_1: 4-bit asymmetric, scale + min
//     Q5_0: 5-bit symmetric, scale only
//     Q5_1: 5-bit asymmetric, scale + min
//     Q8_0: 8-bit symmetric, scale only
//     Q8_1: 8-bit symmetric, scale + sum
//
//   K-quant (256 elements per super-block):
//     Q2_K: 2-bit with hierarchical scales
//     Q3_K: 3-bit with hierarchical scales
//     Q4_K: 4-bit with hierarchical scales
//     Q5_K: 5-bit with hierarchical scales
//     Q6_K: 6-bit with hierarchical scales
//     Q8_K: 8-bit with block sums
//
//   AWQ (Activation-aware Weight Quantization):
//     Q_AWQ: 4-bit with scale + zero, group size 128
//     Q_AWQ_G64: 4-bit with scale + zero, group size 64

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// =============================================================================
// WARP REDUCTION PRIMITIVES
// =============================================================================

// Warp-level maximum reduction using shuffle
__device__ __forceinline__ float quantize_warp_reduce_max(float x) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP_SIZE));
    }
    return x;
}

// Warp-level sum reduction using shuffle
__device__ __forceinline__ float quantize_warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP_SIZE);
    }
    return x;
}

// Warp-level minimum reduction using shuffle
__device__ __forceinline__ float quantize_warp_reduce_min(float x) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        x = fminf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP_SIZE));
    }
    return x;
}

// =============================================================================
// FORMAT-SPECIFIC IMPLEMENTATIONS
// =============================================================================

// Standard 32-element block formats
#include "quantize_q4_0.cuh"
#include "quantize_q4_1.cuh"
#include "quantize_q5_0.cuh"
#include "quantize_q5_1.cuh"
#include "quantize_q8_0.cuh"
#include "quantize_q8_1.cuh"

// KS (attention-sink sub-block) formats
#include "quantize_q4_ks.cuh"
#include "quantize_q8_ks.cuh"

// Simple low-bit formats (2-bit and 3-bit symmetric)
#include "quantize_q2_0.cuh"
#include "quantize_q3_0.cuh"

// Ultra-low-bit and FP8-scale formats
#include "quantize_q0.cuh"
#include "quantize_q0_v.cuh"
#include "quantize_q1_s.cuh"
#include "quantize_q1_a.cuh"
#include "quantize_q2_s.cuh"
#include "quantize_q2_a.cuh"
#include "quantize_q2_1.cuh"
#include "quantize_q3_1.cuh"

// R16: Raw F16 with Q-capture space
#include "quantize_r16.cuh"

// K-quant 256-element super-block formats
#include "quantize_q2_k.cuh"
#include "quantize_q3_k.cuh"
#include "quantize_q4_k.cuh"
#include "quantize_q5_k.cuh"
#include "quantize_q6_k.cuh"
#include "quantize_q8_k.cuh"

// AWQ formats - only include when 80-byte padded structs are available
// (defined in quantize_kernels.cu, not compatible with blocks.cuh)
#ifdef CANDLE_AWQ_QUANTIZE_PADDED
#include "quantize_q_awq.cuh"
#include "quantize_q_awq_g64.cuh"
#endif

// =============================================================================
// KERNEL ENTRY POINTS - Standard formats
// =============================================================================

extern "C" __global__ void quantize_tensor_q4_0(
    const float* __restrict__ src,
    block_q4_0* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q4_0<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q4_1(
    const float* __restrict__ src,
    block_q4_1* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q4_1<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q5_0(
    const float* __restrict__ src,
    block_q5_0* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q5_0<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q5_1(
    const float* __restrict__ src,
    block_q5_1* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q5_1<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q8_0(
    const float* __restrict__ src,
    block_q8_0* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q8_0<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q8_1(
    const float* __restrict__ src,
    block_q8_1* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q8_1<1>(src, dst, num_blocks);
}

// Q2_0 and Q3_0: 2-bit and 3-bit symmetric formats
extern "C" __global__ void quantize_tensor_q2_0(
    const float* __restrict__ src,
    block_q2_0* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q2_0<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q3_0(
    const float* __restrict__ src,
    block_q3_0* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q3_0<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q0(
    const float* __restrict__ src,
    block_q0* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q0<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q0_v(
    const float* __restrict__ src,
    block_q0_v* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q0_v<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q1_a(
    const float* __restrict__ src,
    block_q1_a* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q1_a<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q0_x(
    const float* __restrict__ src,
    block_q0_x* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q0_x<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q0_m2(
    const float* __restrict__ src,
    block_q0_m2* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q0_m2<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q0_m4(
    const float* __restrict__ src,
    block_q0_m4* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q0_m4<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q1_s(
    const float* __restrict__ src,
    block_q1_s* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q1_s<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q2_s(
    const float* __restrict__ src,
    block_q2_s* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q2_s<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q2_a(
    const float* __restrict__ src,
    block_q2_a* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q2_a<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q2_1(
    const float* __restrict__ src,
    block_q2_1* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q2_1<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q3_1(
    const float* __restrict__ src,
    block_q3_1* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q3_1<1>(src, dst, num_blocks);
}

// =============================================================================
// KERNEL ENTRY POINTS - K-quant formats
// =============================================================================

extern "C" __global__ void quantize_tensor_q2_K(
    const float* __restrict__ src,
    block_q2_K* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q2_K<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q3_K(
    const float* __restrict__ src,
    block_q3_K* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q3_K<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q4_K(
    const float* __restrict__ src,
    block_q4_K* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q4_K<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q5_K(
    const float* __restrict__ src,
    block_q5_K* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q5_K<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q6_K(
    const float* __restrict__ src,
    block_q6_K* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q6_K<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q8_K(
    const float* __restrict__ src,
    block_q8_K* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q8_K<1>(src, dst, num_blocks);
}

// =============================================================================
// KERNEL ENTRY POINTS - AWQ formats (only with padded structs)
// =============================================================================
#ifdef CANDLE_AWQ_QUANTIZE_PADDED
extern "C" __global__ void quantize_tensor_q_awq(
    const float* __restrict__ src,
    block_q_awq* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q_awq<1>(src, dst, num_blocks);
}

extern "C" __global__ void quantize_tensor_q_awq_g64(
    const float* __restrict__ src,
    block_q_awq_g64* __restrict__ dst,
    int num_blocks) {
    quantize_blocks_q_awq_g64<1>(src, dst, num_blocks);
}
#endif // CANDLE_AWQ_QUANTIZE_PADDED

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

__host__ __device__ __forceinline__ int get_num_q8_blocks(int num_elements) {
    return (num_elements + QK8_0 - 1) / QK8_0;
}

__host__ __device__ __forceinline__ int get_num_q4_blocks(int num_elements) {
    return (num_elements + QK4_0 - 1) / QK4_0;
}

__host__ __device__ __forceinline__ int get_num_qk_blocks(int num_elements) {
    return (num_elements + QK_K - 1) / QK_K;
}

__host__ __device__ __forceinline__ int get_num_awq_blocks(int num_elements) {
    return (num_elements + QK_Q_AWQ - 1) / QK_Q_AWQ;
}

__host__ __device__ __forceinline__ int get_num_awq_g64_blocks(int num_elements) {
    return (num_elements + QK_Q_AWQ_G64 - 1) / QK_Q_AWQ_G64;
}
