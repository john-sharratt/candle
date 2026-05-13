// SPDX-License-Identifier: MIT
#pragma once

// =============================================================================
// UNIFIED DEQUANTIZATION HEADER
// =============================================================================
// Includes all format-specific dequant implementations and provides unified
// dequantize_block() overloads for generic code.
//
// Usage:
//   #include "dequant/dequant.cuh"
//   dequantize_block(src_blocks, dst_ptr);  // dispatches based on block type
//
// Output element count per call:
//   - Q4_0, Q4_1, Q5_0, Q5_1, Q8_0: 64 elements (dual-block)
//   - Q8_1: 32 elements (single block for Y vector)
//   - Q2_K, Q3_K, Q4_K, Q5_K, Q6_K: 256 elements (full K-quant block)
// =============================================================================

// Prerequisites: WARP_SIZE, from_f32<>, block types
// These should be available from parent headers (math.cuh, types etc.)

// Format-specific implementations
#include "dequant_q4_0.cuh"
#include "dequant_q4_1.cuh"
#include "dequant_q5_0.cuh"
#include "dequant_q5_1.cuh"
#include "dequant_q8_0.cuh"
#include "dequant_q8_1.cuh"
#include "dequant_q2_k.cuh"
#include "dequant_q3_k.cuh"
#include "dequant_q4_k.cuh"
#include "dequant_q5_k.cuh"
#include "dequant_q6_k.cuh"

// K/128 AWQ format implementations
#include "dequant_q_awq.cuh"
#include "dequant_q_awq_g64.cuh"

// =============================================================================
// UNIFIED DEQUANTIZE DISPATCH
// =============================================================================
// Overloaded functions to call the correct dequantize_block_* based on
// block type. This enables generic code to dequantize any quantized format.
// Note: Using function overloading, not template specialization.

// Q4_0 overloads
template <typename compute_t>
__device__ __forceinline__ void dequantize_block(
    const block_q4_0* __restrict__ src, compute_t* __restrict__ dst) {
    dequantize_block_q4_0(src, dst);
}

// Q4_1 overloads
template <typename compute_t>
__device__ __forceinline__ void dequantize_block(
    const block_q4_1* __restrict__ src, compute_t* __restrict__ dst) {
    dequantize_block_q4_1(src, dst);
}

// Q5_0 overloads
template <typename compute_t>
__device__ __forceinline__ void dequantize_block(
    const block_q5_0* __restrict__ src, compute_t* __restrict__ dst) {
    dequantize_block_q5_0(src, dst);
}

// Q5_1 overloads
template <typename compute_t>
__device__ __forceinline__ void dequantize_block(
    const block_q5_1* __restrict__ src, compute_t* __restrict__ dst) {
    dequantize_block_q5_1(src, dst);
}

// Q8_0 overloads
template <typename compute_t>
__device__ __forceinline__ void dequantize_block(
    const block_q8_0* __restrict__ src, compute_t* __restrict__ dst) {
    dequantize_block_q8_0(src, dst);
}

// Q8_1 overloads (Y vector activation format)
template <typename compute_t>
__device__ __forceinline__ void dequantize_block(
    const block_q8_1* __restrict__ src, compute_t* __restrict__ dst) {
    dequantize_block_q8_1(src, dst);
}

// K-quant overloads
template <typename compute_t>
__device__ __forceinline__ void dequantize_block(
    const block_q2_K* __restrict__ src, compute_t* __restrict__ dst) {
    dequantize_block_q2_K(src, dst);
}

template <typename compute_t>
__device__ __forceinline__ void dequantize_block(
    const block_q3_K* __restrict__ src, compute_t* __restrict__ dst) {
    dequantize_block_q3_K(src, dst);
}

template <typename compute_t>
__device__ __forceinline__ void dequantize_block(
    const block_q4_K* __restrict__ src, compute_t* __restrict__ dst) {
    dequantize_block_q4_K(src, dst);
}

template <typename compute_t>
__device__ __forceinline__ void dequantize_block(
    const block_q5_K* __restrict__ src, compute_t* __restrict__ dst) {
    dequantize_block_q5_K(src, dst);
}

template <typename compute_t>
__device__ __forceinline__ void dequantize_block(
    const block_q6_K* __restrict__ src, compute_t* __restrict__ dst) {
    dequantize_block_q6_K(src, dst);
}

// =============================================================================
// K/128 AWQ FORMAT OVERLOADS
// =============================================================================

// Q_AWQ overloads (AWQ 4-bit with group size 128)
template <typename compute_t>
__device__ __forceinline__ void dequantize_block(
    const block_c_q_awq_k128* __restrict__ src, compute_t* __restrict__ dst) {
    dequantize_block_q_awq(src, dst);
}

// Q_AWQ_G64 overloads (AWQ 4-bit with group size 64)
template <typename compute_t>
__device__ __forceinline__ void dequantize_block(
    const block_c_q_awq_g64_k128* __restrict__ src, compute_t* __restrict__ dst) {
    dequantize_block_q_awq_g64(src, dst);
}
