// SPDX-License-Identifier: MIT
// Batched fused transpose+quantize for KV cache.
// Transforms [chunk, head, token, dim] float -> [chunk, head, dim] quant blocks.
//
// Supports multiple input dtypes (F32, F16, BF16, FP8) with inline conversion.
// Reuses quantization logic from quantize_q*.cuh headers.
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>
#include "../blocks.cuh"
#include "quantize.cuh"  // Reuse existing quantization primitives

// =============================================================================
// DTYPE CONVERSION HELPERS
// =============================================================================

// Input dtype codes. Values MUST match `GgmlDType` / `QType` / `ArenaFormat`
// so that a single integer survives the Rust→CUDA round-trip without needing
// a per-call translation table. Unify them here to avoid the F8E4M3=3
// collision with R16=3 that would otherwise exist between this local enum
// and GgmlDType::R16.
enum SrcDType {
    SDTYPE_F32     = 0,   // GgmlDType::F32
    SDTYPE_F16     = 1,   // GgmlDType::F16
    SDTYPE_BF16    = 2,   // GgmlDType::BF16
    SDTYPE_R16     = 3,   // GgmlDType::R16   (128B block: 64B F16 + 64B Q-capture)
    SDTYPE_F8E4M3  = 34,  // GgmlDType::F8E4M3
};

// Load from typed pointer and convert to float
template<int SDTYPE>
__device__ __forceinline__ float load_convert(const void* ptr, int idx) {
    if constexpr (SDTYPE == SDTYPE_F32) {
        return reinterpret_cast<const float*>(ptr)[idx];
    } else if constexpr (SDTYPE == SDTYPE_F16) {
        return __half2float(reinterpret_cast<const __half*>(ptr)[idx]);
    } else if constexpr (SDTYPE == SDTYPE_BF16) {
        return __bfloat162float(reinterpret_cast<const __nv_bfloat16*>(ptr)[idx]);
    } else if constexpr (SDTYPE == SDTYPE_F8E4M3) {
        return float(reinterpret_cast<const __nv_fp8_e4m3*>(ptr)[idx]);
    } else if constexpr (SDTYPE == SDTYPE_R16) {
        // R16: 128 bytes per 32-element block (64B F16 data + 64B Q-capture)
        // Within each block, F16 values are contiguous at offset 0.
        const int blk = idx / 32;
        const int within = idx % 32;
        const __half* f16_ptr = reinterpret_cast<const __half*>(
            reinterpret_cast<const char*>(ptr) + blk * 128
        );
        return __half2float(f16_ptr[within]);
    }
    return 0.0f; // Should not reach here
}

// =============================================================================
// BATCHED KERNEL - MULTI-DTYPE
// =============================================================================
// Loads strided data into shared memory with inline dtype conversion,
// then calls existing quantize_block_* functions.

template<typename BlockT, int QTYPE, int SDTYPE>
__global__ void transpose_quant_batch_typed(
    const void* __restrict__ src, void* __restrict__ dst,
    const int* __restrict__ src_offsets, const int* __restrict__ dst_offsets,
    int num_chunks, int n_head, int chunk_size, int head_dim
) {
    __shared__ float sdata[32];  // Stage strided data here (always f32 for quantization)
    
    const int idx = blockIdx.x;
    const int blocks_per_chunk = n_head * head_dim;
    const int chunk_idx = idx / blocks_per_chunk;
    const int within = idx % blocks_per_chunk;
    const int h = within / head_dim;
    const int d = within % head_dim;
    const int lane = threadIdx.x;
    
    if (chunk_idx >= num_chunks || h >= n_head) return;
    
    // Load strided data into shared memory with dtype conversion (transpose)
    const int src_elems = n_head * chunk_size * head_dim;
    const int src_off = src_offsets ? src_offsets[chunk_idx] : chunk_idx * src_elems;
    // R16 uses dims-first layout: block[dim] holds chunk_size tokens
    int src_idx;
    if constexpr (SDTYPE == SDTYPE_R16) {
        src_idx = src_off + h * chunk_size * head_dim + d * chunk_size + lane;
    } else {
        src_idx = src_off + h * chunk_size * head_dim + d + lane * head_dim;
    }
    
    sdata[lane] = load_convert<SDTYPE>(src, src_idx);  // Strided load + convert -> contiguous in smem
    __syncwarp();
    
    // Destination block
    const int dst_blk = dst_offsets
        ? (dst_offsets[chunk_idx] / sizeof(BlockT)) + h * head_dim + d
        : chunk_idx * blocks_per_chunk + h * head_dim + d;
    BlockT* out = &((BlockT*)dst)[dst_blk];
    
    // Call existing tested quantize functions (they expect contiguous data)
    if constexpr (QTYPE == 3) quantize_block_r16(sdata, out);
    else if constexpr (QTYPE == 7) quantize_block_q8_0(sdata, out);
    else if constexpr (QTYPE == 8) quantize_block_q8_1(sdata, out);
    else if constexpr (QTYPE == 10) quantize_block_q8_ks(sdata, out);
    else if constexpr (QTYPE == 12) quantize_block_q5_0(sdata, out);
    else if constexpr (QTYPE == 13) quantize_block_q5_1(sdata, out);
    else if constexpr (QTYPE == 15) quantize_block_q4_0(sdata, out);
    else if constexpr (QTYPE == 16) quantize_block_q4_1(sdata, out);
    else if constexpr (QTYPE == 17) quantize_block_q4_ks(sdata, out);
    else if constexpr (QTYPE == 19) quantize_block_q3_0(sdata, out);
    else if constexpr (QTYPE == 20) quantize_block_q3_1(sdata, out);
    else if constexpr (QTYPE == 22) quantize_block_q2_0(sdata, out);
    else if constexpr (QTYPE == 23) quantize_block_q2_1(sdata, out);
    else if constexpr (QTYPE == 25) quantize_block_q2_s(sdata, out);
    else if constexpr (QTYPE == 26) quantize_block_q2_a(sdata, out);
    else if constexpr (QTYPE == 27) quantize_block_q1_s(sdata, out);
    else if constexpr (QTYPE == 33) quantize_block_q0(sdata, out);
    else if constexpr (QTYPE == 28) quantize_block_q0_v(sdata, out);
    else if constexpr (QTYPE == 29) quantize_block_q1_a(sdata, out);
    else if constexpr (QTYPE == 30) quantize_block_q0_x(sdata, out);
    else if constexpr (QTYPE == 31) quantize_block_q0_m2(sdata, out);
    else if constexpr (QTYPE == 32) quantize_block_q0_m4(sdata, out);
}

// Legacy f32-only kernel for backwards compatibility
template<typename BlockT, int QTYPE>
__global__ void transpose_quant_batch(
    const float* __restrict__ src, void* __restrict__ dst,
    const int* __restrict__ src_offsets, const int* __restrict__ dst_offsets,
    int num_chunks, int n_head, int chunk_size, int head_dim
) {
    __shared__ float sdata[32];  // Stage strided data here
    
    const int idx = blockIdx.x;
    const int blocks_per_chunk = n_head * head_dim;
    const int chunk_idx = idx / blocks_per_chunk;
    const int within = idx % blocks_per_chunk;
    const int h = within / head_dim;
    const int d = within % head_dim;
    const int lane = threadIdx.x;
    
    if (chunk_idx >= num_chunks || h >= n_head) return;
    
    // Load strided data into shared memory (transpose)
    const int src_elems = n_head * chunk_size * head_dim;
    const int src_off = src_offsets ? src_offsets[chunk_idx] : chunk_idx * src_elems;
    const float* src_ptr = src + src_off + h * chunk_size * head_dim + d;
    sdata[lane] = src_ptr[lane * head_dim];  // Strided load -> contiguous in smem
    __syncwarp();
    
    // Destination block
    const int dst_blk = dst_offsets
        ? (dst_offsets[chunk_idx] / sizeof(BlockT)) + h * head_dim + d
        : chunk_idx * blocks_per_chunk + h * head_dim + d;
    BlockT* out = &((BlockT*)dst)[dst_blk];
    
    // Call existing tested quantize functions (they expect contiguous data)
    if constexpr (QTYPE == 3) quantize_block_r16(sdata, out);
    else if constexpr (QTYPE == 7) quantize_block_q8_0(sdata, out);
    else if constexpr (QTYPE == 8) quantize_block_q8_1(sdata, out);
    else if constexpr (QTYPE == 10) quantize_block_q8_ks(sdata, out);
    else if constexpr (QTYPE == 12) quantize_block_q5_0(sdata, out);
    else if constexpr (QTYPE == 13) quantize_block_q5_1(sdata, out);
    else if constexpr (QTYPE == 15) quantize_block_q4_0(sdata, out);
    else if constexpr (QTYPE == 16) quantize_block_q4_1(sdata, out);
    else if constexpr (QTYPE == 17) quantize_block_q4_ks(sdata, out);
    else if constexpr (QTYPE == 19) quantize_block_q3_0(sdata, out);
    else if constexpr (QTYPE == 20) quantize_block_q3_1(sdata, out);
    else if constexpr (QTYPE == 22) quantize_block_q2_0(sdata, out);
    else if constexpr (QTYPE == 23) quantize_block_q2_1(sdata, out);
    else if constexpr (QTYPE == 25) quantize_block_q2_s(sdata, out);
    else if constexpr (QTYPE == 26) quantize_block_q2_a(sdata, out);
    else if constexpr (QTYPE == 27) quantize_block_q1_s(sdata, out);
    else if constexpr (QTYPE == 33) quantize_block_q0(sdata, out);
    else if constexpr (QTYPE == 28) quantize_block_q0_v(sdata, out);
    else if constexpr (QTYPE == 29) quantize_block_q1_a(sdata, out);
    else if constexpr (QTYPE == 30) quantize_block_q0_x(sdata, out);
    else if constexpr (QTYPE == 31) quantize_block_q0_m2(sdata, out);
    else if constexpr (QTYPE == 32) quantize_block_q0_m4(sdata, out);
}

// =============================================================================
// DISPATCHER - MULTI-DTYPE VERSION
// =============================================================================

extern "C" void run_quantize_transposed_batched_typed(
    const void* src, void* dst,
    const int* src_offsets, const int* dst_offsets,
    int32_t num_chunks, int32_t n_head, int32_t chunk_size, int32_t head_dim,
    int32_t qtype, int32_t src_dtype
) {
    if (num_chunks == 0) return;
    dim3 grid(num_chunks * n_head * head_dim), block(32);
    
    // Dispatch on both qtype and src_dtype
    #define L(T,Q,S) transpose_quant_batch_typed<T,Q,S><<<grid,block>>>(src,dst,src_offsets,dst_offsets,num_chunks,n_head,chunk_size,head_dim)
    // Dispatch src_dtype by its canonical GgmlDType value.
    #define LQ(T,Q) switch(src_dtype) { \
        case SDTYPE_F32:    L(T,Q,SDTYPE_F32);    break; \
        case SDTYPE_F16:    L(T,Q,SDTYPE_F16);    break; \
        case SDTYPE_BF16:   L(T,Q,SDTYPE_BF16);   break; \
        case SDTYPE_R16:    L(T,Q,SDTYPE_R16);    break; \
        case SDTYPE_F8E4M3: L(T,Q,SDTYPE_F8E4M3); break; \
    }
    switch (qtype) {
        case 3: LQ(block_r16,3); break;
        case 7: LQ(block_q8_0,7); break;
        case 8: LQ(block_q8_1,8); break;
        case 10: LQ(block_q8_ks,10); break;
        case 12: LQ(block_q5_0,12); break;
        case 13: LQ(block_q5_1,13); break;
        case 15: LQ(block_q4_0,15); break;
        case 16: LQ(block_q4_1,16); break;
        case 17: LQ(block_q4_ks,17); break;
        case 19: LQ(block_q3_0,19); break;
        case 20: LQ(block_q3_1,20); break;
        case 22: LQ(block_q2_0,22); break;
        case 23: LQ(block_q2_1,23); break;
        case 25: LQ(block_q2_s,25); break;
        case 26: LQ(block_q2_a,26); break;
        case 27: LQ(block_q1_s,27); break;
        case 33: LQ(block_q0,33); break;
        case 28: LQ(block_q0_v,28); break;
        case 29: LQ(block_q1_a,29); break;
        case 30: LQ(block_q0_x,30); break;
        case 31: LQ(block_q0_m2,31); break;
        case 32: LQ(block_q0_m4,32); break;
    }
    #undef LQ
    #undef L
}

// =============================================================================
// DISPATCHER - LEGACY F32 VERSION
// =============================================================================

extern "C" void run_quantize_transposed_batched(
    const float* src, void* dst,
    const int* src_offsets, const int* dst_offsets,
    int32_t num_chunks, int32_t n_head, int32_t chunk_size, int32_t head_dim,
    int32_t qtype
) {
    if (num_chunks == 0) return;
    dim3 grid(num_chunks * n_head * head_dim), block(32);
    
    #define L(T,Q) transpose_quant_batch<T,Q><<<grid,block>>>(src,dst,src_offsets,dst_offsets,num_chunks,n_head,chunk_size,head_dim)
    switch (qtype) {        
        case 3: L(block_r16,3); break;
        case 7: L(block_q8_0,7); break;
        case 8: L(block_q8_1,8); break;
        case 10: L(block_q8_ks,10); break;
        case 12: L(block_q5_0,12); break;
        case 13: L(block_q5_1,13); break;
        case 15: L(block_q4_0,15); break;
        case 16: L(block_q4_1,16); break;
        case 17: L(block_q4_ks,17); break;
        case 19: L(block_q3_0,19); break;
        case 20: L(block_q3_1,20); break;
        case 22: L(block_q2_0,22); break;
        case 23: L(block_q2_1,23); break;
        case 25: L(block_q2_s,25); break;
        case 26: L(block_q2_a,26); break;
        case 27: L(block_q1_s,27); break;
        case 33: L(block_q0,33); break;
        case 28: L(block_q0_v,28); break;
        case 29: L(block_q1_a,29); break;
        case 30: L(block_q0_x,30); break;
        case 31: L(block_q0_m2,31); break;
        case 32: L(block_q0_m4,32); break;
    }
    #undef L
}
