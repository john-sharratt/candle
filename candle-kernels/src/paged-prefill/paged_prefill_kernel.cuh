/*
 * ============================================================================
 * PAGED PREFILL ATTENTION KERNEL
 * ============================================================================
 * 
 * High-performance Flash Attention kernel with paged KV cache support for
 * LLM inference serving. Optimized for NVIDIA Ada Lovelace (SM89) and
 * Ampere (SM80+) architectures.
 *
 * ============================================================================
 * FEATURE COMPARISON WITH INDUSTRY IMPLEMENTATIONS
 * ============================================================================
 *
 * ┌──────────────────────┬──────┬───────┬────────┬───────────┬──────┬─────────┐
 * │ Feature              │ This │ FA2   │ FA3    │ FlashInfer│ vLLM │ cuDNN   │
 * ├──────────────────────┼──────┼───────┼────────┼───────────┼──────┼─────────┤
 * │ Online Softmax       │  ✓   │   ✓  │   ✓   │     ✓     │  ✓   │    ✓    │
 * │ Paged KV Cache       │  ✓   │   ✗  │   ✗   │     ✓     │  ✓   │    ✗    │
 * │ Tensor Cores (MMA)   │  ✓   │   ✓  │   ✓   │     ✓     │  ✗   │    ✓    │
 * │ Async Copy (cp.async)│  ✓   │   ✓  │   ✓   │     ✓     │  ✗   │    ✓    │
 * │ Multi-stage Pipeline │ 2-3  │   2   │  2+   │    2-3     │  1   │   2+    │
 * │ GQA Support          │  ✓   │   ✓  │   ✓   │     ✓     │  ✓   │    ✓    │
 * │ Register-based P     │  ✓   │   ✗  │   ✓   │  Partial  │  ✗   │    ✗    │
 * │ Prefill + KV Write   │  ✓   │   ✗  │   ✗   │     ✓     │  ✗  │    ✗    │
 * │ WGMMA (Hopper)       │  ✗   │   ✗  │   ✓   │     ✗     │  ✗  │    ✓    │
 * │ TMA (Hopper)         │  ✗   │   ✗  │   ✓   │     ✗     │  ✗  │    ✓    │
 * └──────────────────────┴──────┴───────┴───────┴───────────┴──────┴─────────┘
 *
 * Performance: ~7000+ tokens/sec on RTX 4090 (HEAD_DIM=64, batch inference)
 *
 * ============================================================================
 * ARCHITECTURE REQUIREMENTS
 * ============================================================================
 *
 * Minimum: SM80 (Ampere) for cp.async and m16n8k16 MMA
 * Optimal: SM89 (Ada Lovelace) - kernel is tuned for this architecture
 * Note:    SM90 (Hopper) WGMMA/TMA not supported - use FlashAttention-3
 *
 * ============================================================================
 * KEY OPTIMIZATIONS
 * ============================================================================
 *
 * 1. REGISTER-BASED SOFTMAX
 *    P values kept in registers, redistributed via warp shuffles.
 *    Eliminates 4KB shared memory and associated sync points.
 *
 * 2. FOLDED BETA COMPUTATION
 *    Standard:  p = exp(s - local_max) * exp(local_max - m_new)
 *    Optimized: p = exp(s - m_new)
 *    Saves 2 exp() calls and 8 multiplies per tile iteration.
 *
 * 3. MULTI-STAGE ASYNC PIPELINE
 *    Triple-buffered K/V loading with cp.async for latency hiding.
 *    Automatic fallback to double-buffer if shared memory constrained.
 *
 * 4. BANK-CONFLICT-FREE SHARED MEMORY
 *    +4 padding on output accumulator stride eliminates 8-way conflicts.
 *
 * 5. REVERSE K-TILE ITERATION
 *    Process K-tiles last-to-first. First iteration handles edge masking;
 *    remaining tiles skip bounds checks for full-tile fast path.
 *
 * ============================================================================
 * THREAD MAPPING (m16n8k16 MMA Layout)
 * ============================================================================
 *
 * Each warp (32 threads) processes a 16×16 score tile:
 *
 *   groupID = lane >> 2   (0-7)  → Determines output ROWS
 *   tid     = lane & 3    (0-3)  → Determines output COLUMNS
 *
 * Thread (groupID, tid) owns:
 *   Rows: groupID, groupID+8 (2 rows per 16-row subtile)
 *   Cols: tid*2, tid*2+1, tid*2+8, tid*2+9 (4 columns)
 *
 * For BLOCK_M=32: 2 subtiles × 2 rows = 4 rows per thread
 *
 * ============================================================================
 * SHARED MEMORY LAYOUT
 * ============================================================================
 *
 * Total shared memory budget: ~100KB (default) or ~164KB (extended)
 *
 * ┌─────────────────────────────────────────────────────────────────────────┐
 * │ Buffer              │ Size Formula                │ HD64    │ HD128    │
 * ├─────────────────────┼─────────────────────────────┼─────────┼──────────┤
 * │ smem_k[STAGES]      │ STAGES × TILE_K × HD × 2B   │ 8-12KB  │ 16-24KB  │
 * │ smem_v[STAGES]      │ STAGES × TILE_K × HD × 2B   │ 8-12KB  │ 16-24KB  │
 * │ smem_q              │ WARPS_TC × BLOCK_M × HD × 2B│ 16KB    │ 16KB     │
 * │ O accumulators      │ per-thread registers         │ ~0KB    │ ~0KB      │
 * │ Metadata arrays     │ ~1KB                        │ ~1KB    │ ~1KB     │
 * └─────────────────────┴─────────────────────────────┴─────────┴──────────┘
 *
 * WARPS_TC stays conservative as HEAD_DIM grows to control register pressure:
 *   HD64:  WARPS_TC=4, HD128: WARPS_TC=2, HD256: WARPS_TC=1
 *
 * ============================================================================
 * USAGE
 * ============================================================================
 *
 * launch_paged_prefill_chunks<T, HEAD_DIM, WARPS_PER_BLOCK, TILE_K>(
 *     q_ptr, k_ptr, v_ptr,
 *     per_head_table, chunk_meta,
 *     cu_seqlens_q, q_lens, kv_lens,
 *     o_ptr,
 *     total_q, batch_size, n_head, n_kv_head,
 *     arena_chunks, max_blocks,
 *     softmax_scale, has_prefix
 * );
 *
 * ============================================================================
 * REVISION HISTORY
 * ============================================================================
 *
 * v5 - Register-based P values, double-buffered K/V/Q rings
 *
 * ============================================================================
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <math.h>
#include <stdint.h>
#include <type_traits>
#include "../fast_exp.cuh"
#include "../arena_table.cuh"
#include "../paged-decode/slot_types.cuh"
#include "../device_caps.cuh"
#include "../convert/convert_all.cuh"

using namespace nvcuda;

// ============================================================================
// CONSTANTS
// ============================================================================

// IEEE 754 negative infinity for float (avoids double→float narrowing warning)
#define NEG_INF_F __int_as_float(0xff800000)

// MMA instruction dimensions (m16n8k16)
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 8;
constexpr int WMMA_K = 16;

// ============================================================================
// TYPE CONVERSION UTILITIES
// ============================================================================
// to_f32 and from_f32 are now provided by convert/convert.cuh (via convert_all.cuh)

/// Zero value for half/bfloat16/float
template <typename T>
__device__ __forceinline__ T zero_val();

// Float zero value
template <>
__device__ __forceinline__ float zero_val<float>() {
    return 0.f;
}

template <>
__device__ __forceinline__ __half zero_val<__half>() {
    return __float2half_rn(0.f);
}

template <>
__device__ __forceinline__ __nv_bfloat16 zero_val<__nv_bfloat16>() {
    return __float2bfloat16_rn(0.f);
}

// FP8 zero value
template <>
__device__ __forceinline__ __nv_fp8_e4m3 zero_val<__nv_fp8_e4m3>() {
    __nv_fp8_e4m3 result;
    *reinterpret_cast<uint8_t*>(&result) = 0;
    return result;
}

// ============================================================================
// ROPE HELPERS
// ============================================================================

/// Look up precomputed cos/sin pair for RoPE from the cos_sin table.
/// rope_cs layout: rope_cs[pos * HEAD_DIM + d_idx * 2] = cos,
///                 rope_cs[pos * HEAD_DIM + d_idx * 2 + 1] = sin.
template <int HEAD_DIM>
__device__ __forceinline__ void rope_cos_sin(
    int pos, int d_idx, const float* __restrict__ rope_cs, float& cos_v, float& sin_v
) {
    const float* entry = rope_cs + (int64_t)pos * HEAD_DIM + d_idx * 2;
    cos_v = __ldg(entry);
    sin_v = __ldg(entry + 1);
}

// ============================================================================
// KV LOAD HELPERS - Direct same-type copy from arena to shared memory
// ============================================================================

/// Load 8 elements from KV arena to smem. Direct copy path (no conversion).
template <typename T, bool USE_TC>
__device__ __forceinline__ void load_kv_chunk(
    T* __restrict__ k_dst,           // Shared memory destination for K (8 elements)
    T* __restrict__ v_dst,           // Shared memory destination for V (8 elements)
    const T* __restrict__ k_src,     // Global memory source for K
    const T* __restrict__ v_src,     // Global memory source for V
    int64_t src_offset               // Offset into k_src/v_offset
) {
    constexpr int ELEMS_PER_CP = 8;
    constexpr int CP_SIZE = ELEMS_PER_CP * sizeof(T);  // 16 bytes for FP16/BF16, 8 bytes for FP8
    
    if constexpr (USE_TC && CP_SIZE >= 16) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;"
            :: "r"((uint32_t)__cvta_generic_to_shared(k_dst)), 
               "l"(&k_src[src_offset]), 
               "n"(CP_SIZE));
        asm volatile("cp.async.cg.shared.global [%0], [%1], %2;"
            :: "r"((uint32_t)__cvta_generic_to_shared(v_dst)), 
               "l"(&v_src[src_offset]), 
               "n"(CP_SIZE));
    } else {
        // Non-TC path: vectorized copy based on type size
        if constexpr (CP_SIZE == 32) {
            // F32: 32 bytes = 8× uint32_t
            const uint32_t* ks = reinterpret_cast<const uint32_t*>(&k_src[src_offset]);
            const uint32_t* vs = reinterpret_cast<const uint32_t*>(&v_src[src_offset]);
            uint32_t* kd = reinterpret_cast<uint32_t*>(k_dst);
            uint32_t* vd = reinterpret_cast<uint32_t*>(v_dst);
            kd[0] = ks[0]; kd[1] = ks[1]; kd[2] = ks[2]; kd[3] = ks[3];
            kd[4] = ks[4]; kd[5] = ks[5]; kd[6] = ks[6]; kd[7] = ks[7];
            vd[0] = vs[0]; vd[1] = vs[1]; vd[2] = vs[2]; vd[3] = vs[3];
            vd[4] = vs[4]; vd[5] = vs[5]; vd[6] = vs[6]; vd[7] = vs[7];
        } else if constexpr (CP_SIZE == 16) {
            // F16/BF16: 16 bytes = 4× uint32_t
            const uint32_t* ks = reinterpret_cast<const uint32_t*>(&k_src[src_offset]);
            const uint32_t* vs = reinterpret_cast<const uint32_t*>(&v_src[src_offset]);
            uint32_t* kd = reinterpret_cast<uint32_t*>(k_dst);
            uint32_t* vd = reinterpret_cast<uint32_t*>(v_dst);
            kd[0] = ks[0]; kd[1] = ks[1]; kd[2] = ks[2]; kd[3] = ks[3];
            vd[0] = vs[0]; vd[1] = vs[1]; vd[2] = vs[2]; vd[3] = vs[3];
        } else {
            // FP8 or other: element-by-element copy (no alignment guarantee)
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_CP; ++i) {
                k_dst[i] = k_src[src_offset + i];
                v_dst[i] = v_src[src_offset + i];
            }
        }
    }
}

/// Load 8 elements from arena with format conversion (per-head resolved).
/// For dtype formats: uses vectorized loads when arena format matches T, else converts.
/// For quant formats: token-oriented dequantization using dequant_element_hybrid.
///
/// Parameters:
///   k_dst, v_dst: Shared memory destinations (8 elements each, type T)
///   k_head_ptr, v_head_ptr: Pre-resolved byte pointers to this head's K/V data
///   k_chunk_byte_stride, v_chunk_byte_stride: Bytes between consecutive chunks for this head
///   chunk_idx, within_chunk: Position within arena
///   dim_offset: Starting dimension offset (0..HEAD_DIM in steps of 8)
///   blocks_per_dim: CHUNK_SIZE / 32 for token-oriented quant layout
template <typename T, int HEAD_DIM, bool USE_TC = false>
__device__ __forceinline__ void load_kv_chunk_arena(
    T* k_dst,
    T* v_dst,
    const char* k_head_ptr,
    const char* v_head_ptr,
    int64_t k_chunk_byte_stride,
    int64_t v_chunk_byte_stride,
    int k_fmt,
    int v_fmt,
    int k_chunk_idx,
    int v_chunk_idx,
    int within_chunk,
    int dim_offset,
    int blocks_per_dim,
    float k_scale,
    float v_scale
) {
    constexpr int ELEMS_PER_CP = 8;
    constexpr int T_FORMAT = type_to_arena_format<T>();
    constexpr int CP_BYTES = ELEMS_PER_CP * sizeof(T);
    
    // K and V may live in different format families (e.g. R16: K=quant, V=float),
    // so load each side according to its own format independently.
    int k_elem_size = ArenaFormat::float_elem_size(k_fmt);
    int v_elem_size = ArenaFormat::float_elem_size(v_fmt);

    // --- K loading ---
    if (k_elem_size > 0) {
        const int64_t k_within_off = (int64_t)within_chunk * (int64_t)HEAD_DIM + dim_offset;
        if (k_fmt == T_FORMAT) {
            const T* k_src = reinterpret_cast<const T*>(k_head_ptr + (int64_t)k_chunk_idx * k_chunk_byte_stride) + k_within_off;
            if constexpr (USE_TC && CP_BYTES >= 16) {
                cp_async_cg_16<true>(k_dst, k_src);
                if constexpr (CP_BYTES > 16) {
                    cp_async_cg_16<true>((char*)k_dst + 16, (const char*)k_src + 16);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < ELEMS_PER_CP; ++i) k_dst[i] = k_src[i];
            }
        } else {
            const char* k_base = k_head_ptr + (int64_t)k_chunk_idx * k_chunk_byte_stride + k_within_off * k_elem_size;
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_CP; ++i)
                k_dst[i] = from_f32<T>(arena_load_element(k_base + i * k_elem_size, k_fmt, k_scale));
        }
    } else {
        int k_block_bytes = get_quant_block_bytes(k_fmt);
        const char* k_head = k_head_ptr + (int64_t)k_chunk_idx * k_chunk_byte_stride;
        int block_within_dim = within_chunk / 32;
        int elem_in_block = within_chunk % 32;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_CP; ++i) {
            int dim = dim_offset + i;
            int64_t block_idx = (int64_t)dim * blocks_per_dim + block_within_dim;
            const char* k_block = k_head + block_idx * k_block_bytes;
            k_dst[i] = from_f32<T>(dequant_element_hybrid<float>(k_block, elem_in_block, k_fmt, k_scale));
        }
    }

    // --- V loading ---
    if (v_elem_size > 0) {
        const int64_t v_within_off = (int64_t)within_chunk * (int64_t)HEAD_DIM + dim_offset;
        if (v_fmt == T_FORMAT) {
            const T* v_src = reinterpret_cast<const T*>(v_head_ptr + (int64_t)v_chunk_idx * v_chunk_byte_stride) + v_within_off;
            if constexpr (USE_TC && CP_BYTES >= 16) {
                cp_async_cg_16<true>(v_dst, v_src);
                if constexpr (CP_BYTES > 16) {
                    cp_async_cg_16<true>((char*)v_dst + 16, (const char*)v_src + 16);
                }
            } else {
                #pragma unroll
                for (int i = 0; i < ELEMS_PER_CP; ++i) v_dst[i] = v_src[i];
            }
        } else {
            const char* v_base = v_head_ptr + (int64_t)v_chunk_idx * v_chunk_byte_stride + v_within_off * v_elem_size;
            #pragma unroll
            for (int i = 0; i < ELEMS_PER_CP; ++i)
                v_dst[i] = from_f32<T>(arena_load_element(v_base + i * v_elem_size, v_fmt, v_scale));
        }
    } else {
        int v_block_bytes = get_quant_block_bytes(v_fmt);
        const char* v_head = v_head_ptr + (int64_t)v_chunk_idx * v_chunk_byte_stride;
        int block_within_dim = within_chunk / 32;
        int elem_in_block = within_chunk % 32;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_CP; ++i) {
            int dim = dim_offset + i;
            int64_t block_idx = (int64_t)dim * blocks_per_dim + block_within_dim;
            const char* v_block = v_head + block_idx * v_block_bytes;
            v_dst[i] = from_f32<T>(dequant_element_hybrid<float>(v_block, elem_in_block, v_fmt, v_scale));
        }
    }
}

/// Compute palette index and rank-within-palette for global dim `tid`.
/// pal_map: 32 bytes, 2 bits per dim, little-endian packed.
__device__ __forceinline__ void prefill_pal_rank(
    const uint8_t* pal_map, int tid, int* out_p, int* out_rank)
{
    int my_p = (pal_map[tid >> 2] >> (2 * (tid & 3))) & 0x3;
    const uint32_t* pm = (const uint32_t*)pal_map;
    int word_idx = tid >> 4;
    int partial  = tid & 15;

    auto match = [my_p](uint32_t w) -> uint32_t {
        uint32_t b0 = w & 0x55555555u;
        uint32_t b1 = (w >> 1) & 0x55555555u;
        uint32_t m0 = (my_p & 1) ? b0 : (~b0 & 0x55555555u);
        uint32_t m1 = (my_p & 2) ? b1 : (~b1 & 0x55555555u);
        return m0 & m1;
    };

    int rank = 0;
    for (int i = 0; i < word_idx; i++) {
        rank += __popc(match(pm[i]));
    }
    if (partial > 0) {
        uint32_t m = match(pm[word_idx]);
        m &= (1u << (partial * 2)) - 1;
        rank += __popc(m);
    }

    *out_p    = my_p;
    *out_rank = rank;
}

/// Helper to get quant block size in bytes
__device__ __forceinline__ int get_quant_block_bytes(int fmt) {
    switch (fmt) {
        case ArenaFormat::Q4_0: return 18;
        case ArenaFormat::Q4_1: return 20;
        case ArenaFormat::Q5_0: return 22;
        case ArenaFormat::Q5_1: return 24;
        case ArenaFormat::Q8_0: return 34;
        case ArenaFormat::Q8_1: return 36;
        case ArenaFormat::Q4_KS: return 20;
        case ArenaFormat::Q8_KS: return 36;
        case ArenaFormat::Q2_0: return 10;
        case ArenaFormat::Q3_0: return 14;
        case ArenaFormat::R16: return 128;
        case ArenaFormat::Q0: return 1;
        case ArenaFormat::Q1_S: return 5;
        case ArenaFormat::Q2_S: return 9;
        case ArenaFormat::Q2_A: return 10;
        case ArenaFormat::Q2_1: return 12;
        case ArenaFormat::Q3_1: return 16;
        case ArenaFormat::Q0_V: return 2;
        case ArenaFormat::Q1_A: return 6;
        case ArenaFormat::Q0_X: return 2;
        case ArenaFormat::Q0_M2: return 3;
        case ArenaFormat::Q0_M4: return 8;
        // Non-arena format reached arena block addressing: fail loud (a guessed size
        // would silently corrupt memory). K-quants/AWQ/q8a128 are not arena-storable.
        // printf+__trap matches arena_load_element's idiom in this file.
        default:
            printf("get_quant_block_bytes: non-arena format %d (not arena-storable)\n", fmt);
            __trap();
            return 0;
    }
}

/// Helper to load a single float from dtype arena
__device__ __forceinline__ float arena_load_element(const char* ptr, int fmt, float scale) {
    switch (fmt) {
        case ArenaFormat::F32: return *reinterpret_cast<const float*>(ptr) / scale;
        case ArenaFormat::F16: return __half2float(*reinterpret_cast<const __half*>(ptr)) / scale;
        case ArenaFormat::R16: return __half2float(*reinterpret_cast<const __half*>(ptr)) / scale;
        case ArenaFormat::BF16: return __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(ptr)) / scale;
        case ArenaFormat::F8E4M3: {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
            __nv_fp8_storage_t s = *reinterpret_cast<const __nv_fp8_storage_t*>(ptr);
            return __half2float(__nv_cvt_fp8_to_halfraw(s, __NV_E4M3)) / scale;
#else
            uint8_t byte = *reinterpret_cast<const uint8_t*>(ptr);
            __nv_fp8_e4m3 fp8_val;
            *reinterpret_cast<uint8_t*>(&fp8_val) = byte;
            return to_f32<__nv_fp8_e4m3>(fp8_val) / scale;
#endif
        }
        default:
            printf("arena_load_element: unhandled format %d\n", fmt);
            __trap();
            return 0.f;
    }
}

// ============================================================================
// KV STORE HELPERS - Direct same-type storage (no conversion needed)
// ============================================================================

/// Store 1 element to KV arena (same type, no conversion)
template <typename T>
__device__ __forceinline__ void store_kv_element(
    T* dst,
    T src_val
) {
    *dst = src_val;
}

/// Store 8 elements from smem to arena (vectorized same-type copy)
template <typename T>
__device__ __forceinline__ void store_kv_chunk(
    T* k_dst,           // Arena destination for K (8 elements)
    T* v_dst,           // Arena destination for V (8 elements)
    const T* k_src,     // Shared memory source for K
    const T* v_src      // Shared memory source for V
) {
    constexpr int COPY_SIZE = 8 * sizeof(T);  // Total bytes to copy
    
    if constexpr (COPY_SIZE == 32) {
        // F32: 32 bytes = 2× uint4 (128-bit) stores
        const uint4* ks = reinterpret_cast<const uint4*>(k_src);
        const uint4* vs = reinterpret_cast<const uint4*>(v_src);
        uint4* kd = reinterpret_cast<uint4*>(k_dst);
        uint4* vd = reinterpret_cast<uint4*>(v_dst);
        kd[0] = ks[0];
        kd[1] = ks[1];
        vd[0] = vs[0];
        vd[1] = vs[1];
    } else if constexpr (COPY_SIZE == 16) {
        // F16/BF16: 16 bytes = single uint4 (128-bit) store
        const uint4* ks = reinterpret_cast<const uint4*>(k_src);
        const uint4* vs = reinterpret_cast<const uint4*>(v_src);
        uint4* kd = reinterpret_cast<uint4*>(k_dst);
        uint4* vd = reinterpret_cast<uint4*>(v_dst);
        *kd = *ks;
        *vd = *vs;
    } else {
        // FP8 or other: element-by-element copy (no alignment guarantee)
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            k_dst[i] = k_src[i];
            v_dst[i] = v_src[i];
        }
    }
}

/// Store 8 elements from smem to arena with format conversion (per-head resolved).
/// Only supports dtype formats (quant writes require separate quantization pass).
/// q_src: optional Q projection for R16 Q-capture (nullptr when k_fmt != R16).
template <typename T, typename Q_T, int HEAD_DIM>
__device__ __forceinline__ void store_kv_chunk_arena(
    char* k_head_ptr,
    char* v_head_ptr,
    const T* k_src,
    const T* v_src,
    const Q_T* q_src,
    int k_fmt,
    int v_fmt,
    int k_chunk_idx,
    int v_chunk_idx,
    int within_chunk,
    int dim_offset,
    int64_t k_chunk_byte_stride,
    int64_t v_chunk_byte_stride
) {
    constexpr int ELEMS_PER_CP = 8;
    constexpr int T_FORMAT = type_to_arena_format<T>();
    
    int k_elem_size = ArenaFormat::float_elem_size(k_fmt);
    int v_elem_size = ArenaFormat::float_elem_size(v_fmt);
    
    // V must be a dtype format; K may be R16 (elem_size=0) or dtype (elem_size>0)
    if (v_elem_size <= 0) return;
    if (k_elem_size <= 0 && k_fmt != ArenaFormat::R16) return;
    
    const int64_t within_off = (int64_t)within_chunk * (int64_t)HEAD_DIM + dim_offset;
    
    // K write
    if (k_fmt == ArenaFormat::R16) {
        // R16 block layout: 128 bytes = 64B F16 d[32] + 64B u16 q[32]
        // Dims-first layout: block[dim] holds 32 tokens for that dimension.
        //   byte offset = dim * 128 + token * 2  (for d[])
        //                 dim * 128 + 64 + token * 2  (for q[])
        // k_chunk_byte_stride is already in bytes for R16 (HEAD_DIM * 128)
        char* chunk_base = k_head_ptr + (int64_t)k_chunk_idx * k_chunk_byte_stride;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_CP; ++i) {
            int dim = dim_offset + i;
            char* blk_base = chunk_base + (int64_t)dim * 128;
            // d[within_chunk]: K value as F16
            __half k_val = __float2half(to_f32<T>(k_src[i]));
            *reinterpret_cast<__half*>(blk_base + within_chunk * 2) = k_val;
            // q[within_chunk]: Q-capture value as F16
            *reinterpret_cast<__half*>(blk_base + 64 + within_chunk * 2) = __float2half(to_f32<Q_T>(q_src[i]));
        }
    } else if (k_fmt == T_FORMAT) {
        T* k_dst = reinterpret_cast<T*>(k_head_ptr + (int64_t)k_chunk_idx * k_chunk_byte_stride) + within_off;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_CP; ++i) k_dst[i] = k_src[i];
    } else {
        char* k_ptr = k_head_ptr + (int64_t)k_chunk_idx * k_chunk_byte_stride + within_off * k_elem_size;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_CP; ++i)
            arena_store_element(k_ptr + i * k_elem_size, to_f32<T>(k_src[i]), k_fmt);
    }
    // V write (always dtype format)
    if (v_fmt == T_FORMAT) {
        T* v_dst = reinterpret_cast<T*>(v_head_ptr + (int64_t)v_chunk_idx * v_chunk_byte_stride) + within_off;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_CP; ++i) v_dst[i] = v_src[i];
    } else {
        char* v_ptr = v_head_ptr + (int64_t)v_chunk_idx * v_chunk_byte_stride + within_off * v_elem_size;
        #pragma unroll
        for (int i = 0; i < ELEMS_PER_CP; ++i)
            arena_store_element(v_ptr + i * v_elem_size, to_f32<T>(v_src[i]), v_fmt);
    }
}

/// Helper to store a single float to dtype arena
__device__ __forceinline__ void arena_store_element(char* ptr, float val, int fmt) {
    switch (fmt) {
        case ArenaFormat::F32: *reinterpret_cast<float*>(ptr) = val; break;
        case ArenaFormat::F16: *reinterpret_cast<__half*>(ptr) = __float2half(val); break;
        case ArenaFormat::BF16: *reinterpret_cast<__nv_bfloat16*>(ptr) = __float2bfloat16(val); break;
        case ArenaFormat::F8E4M3: {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
            __nv_fp8_storage_t s = __nv_cvt_float_to_fp8(val, __NV_SATFINITE, __NV_E4M3);
            *reinterpret_cast<__nv_fp8_storage_t*>(ptr) = s;
#else
            __nv_fp8_e4m3 fp8_val = from_f32<__nv_fp8_e4m3>(val);
            *reinterpret_cast<uint8_t*>(ptr) = *reinterpret_cast<const uint8_t*>(&fp8_val);
#endif
            break;
        }
        default: break;
    }
}

// ============================================================================
// FAST MATH - Using shared fast_exp.cuh library
// ============================================================================
// See ../fast_exp.cuh for the fast exponential implementation.
// Use fast_exp::exp<float, fast_exp::Softmax>() for softmax operations.

// ============================================================================
// VECTOR TYPE UTILITIES
// ============================================================================

/// Vector type traits for half2/bfloat162 operations
template <typename T>
struct vec2_traits;

template <>
struct vec2_traits<__half> {
    using type = __half2;
    static __device__ __forceinline__ __half2 make(float a, float b) {
        return __floats2half2_rn(a, b);
    }
    static __device__ __forceinline__ __half2 splat(float x) {
        return __float2half2_rn(x);
    }
    static __device__ __forceinline__ __half2 mul(__half2 a, __half2 b) {
        return __hmul2(a, b);
    }
    static __device__ __forceinline__ __half2 fma(__half2 a, __half2 b, __half2 c) {
        return __hfma2(a, b, c);
    }
};

template <>
struct vec2_traits<__nv_bfloat16> {
    using type = __nv_bfloat162;
    static __device__ __forceinline__ __nv_bfloat162 make(float a, float b) {
        return __floats2bfloat162_rn(a, b);
    }
    static __device__ __forceinline__ __nv_bfloat162 splat(float x) {
        return __float2bfloat162_rn(x);
    }
    static __device__ __forceinline__ __nv_bfloat162 mul(__nv_bfloat162 a, __nv_bfloat162 b) {
        return __hmul2(a, b);
    }
    static __device__ __forceinline__ __nv_bfloat162 fma(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) {
        return __hfma2(a, b, c);
    }
};

// FP8 vec2 traits - uses uint16_t to hold two FP8 values
template <>
struct vec2_traits<__nv_fp8_e4m3> {
    using type = uint16_t;
    static __device__ __forceinline__ uint16_t make(float a, float b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        __nv_fp8_storage_t sa = __nv_cvt_halfraw_to_fp8(__float2half(a), __NV_SATFINITE, __NV_E4M3);
        __nv_fp8_storage_t sb = __nv_cvt_halfraw_to_fp8(__float2half(b), __NV_SATFINITE, __NV_E4M3);
        return ((uint16_t)sa) | (((uint16_t)sb) << 8);
#else
        __nv_fp8_e4m3 fa = from_f32<__nv_fp8_e4m3>(a);
        __nv_fp8_e4m3 fb = from_f32<__nv_fp8_e4m3>(b);
        uint8_t ba = *reinterpret_cast<const uint8_t*>(&fa);
        uint8_t bb = *reinterpret_cast<const uint8_t*>(&fb);
        return ((uint16_t)ba) | (((uint16_t)bb) << 8);
#endif
    }
    static __device__ __forceinline__ uint16_t splat(float x) {
        return make(x, x);
    }
    static __device__ __forceinline__ float2 to_float2(const __nv_fp8_e4m3* p) {
        // Scalar loads for alignment safety
        return make_float2(to_f32<__nv_fp8_e4m3>(p[0]), to_f32<__nv_fp8_e4m3>(p[1]));
    }
};

/// Pack two 16-bit values into uint32 for MMA operand
template <typename T>
__device__ __forceinline__ uint32_t __pack_half2(T a, T b);

// Float dummy - tensor cores not used for float, but template instantiation requires this
template <>
__device__ __forceinline__ uint32_t __pack_half2<float>(float a, float b) {
    (void)a; (void)b;
    return 0;  // Not actually used - float uses scalar fallback path
}

template <>
__device__ __forceinline__ uint32_t __pack_half2<__half>(__half a, __half b) {
    uint32_t result;
    asm("mov.b32 %0, {%1,%2};" : "=r"(result)
        : "h"(*reinterpret_cast<const unsigned short*>(&a)),
          "h"(*reinterpret_cast<const unsigned short*>(&b)));
    return result;
}

template <>
__device__ __forceinline__ uint32_t __pack_half2<__nv_bfloat16>(__nv_bfloat16 a, __nv_bfloat16 b) {
    uint32_t result;
    asm("mov.b32 %0, {%1,%2};" : "=r"(result)
        : "h"(*reinterpret_cast<const unsigned short*>(&a)),
          "h"(*reinterpret_cast<const unsigned short*>(&b)));
    return result;
}

// FP8 pack - for MMA we convert to half first
template <>
__device__ __forceinline__ uint32_t __pack_half2<__nv_fp8_e4m3>(__nv_fp8_e4m3 a, __nv_fp8_e4m3 b) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __nv_fp8_storage_t sa = *reinterpret_cast<const __nv_fp8_storage_t*>(&a);
    __nv_fp8_storage_t sb = *reinterpret_cast<const __nv_fp8_storage_t*>(&b);
    __half ha = __nv_cvt_fp8_to_halfraw(sa, __NV_E4M3);
    __half hb = __nv_cvt_fp8_to_halfraw(sb, __NV_E4M3);
    uint32_t result;
    asm("mov.b32 %0, {%1,%2};" : "=r"(result)
        : "h"(*reinterpret_cast<const unsigned short*>(&ha)),
          "h"(*reinterpret_cast<const unsigned short*>(&hb)));
    return result;
#else
    __half ha = __float2half(to_f32<__nv_fp8_e4m3>(a));
    __half hb = __float2half(to_f32<__nv_fp8_e4m3>(b));
    uint32_t result;
    asm("mov.b32 %0, {%1,%2};" : "=r"(result)
        : "h"(*reinterpret_cast<const unsigned short*>(&ha)),
          "h"(*reinterpret_cast<const unsigned short*>(&hb)));
    return result;
#endif
}

// ============================================================================
// VECTORIZED LOAD/STORE
// ============================================================================

/// Load VEC elements from memory, converting to float32
template <typename T, int VEC>
struct VecLoad {
    __device__ __forceinline__ static void load(const T* p, float out[VEC]) {
        #pragma unroll
        for (int i = 0; i < VEC; ++i) out[i] = to_f32<T>(p[i]);
    }
};

// Specializations use scalar loads to avoid alignment requirements
template <>
struct VecLoad<__half, 1> {
    __device__ __forceinline__ static void load(const __half* p, float out[1]) {
        out[0] = __half2float(p[0]);
    }
};

template <>
struct VecLoad<__half, 2> {
    __device__ __forceinline__ static void load(const __half* p, float out[2]) {
        out[0] = __half2float(p[0]);
        out[1] = __half2float(p[1]);
    }
};

template <>
struct VecLoad<__half, 4> {
    __device__ __forceinline__ static void load(const __half* p, float out[4]) {
        out[0] = __half2float(p[0]);
        out[1] = __half2float(p[1]);
        out[2] = __half2float(p[2]);
        out[3] = __half2float(p[3]);
    }
};

template <>
struct VecLoad<__half, 8> {
    __device__ __forceinline__ static void load(const __half* p, float out[8]) {
        VecLoad<__half, 4>::load(p, out);
        VecLoad<__half, 4>::load(p + 4, out + 4);
    }
};

template <>
struct VecLoad<__nv_bfloat16, 1> {
    __device__ __forceinline__ static void load(const __nv_bfloat16* p, float out[1]) {
        out[0] = __bfloat162float(p[0]);
    }
};

template <>
struct VecLoad<__nv_bfloat16, 2> {
    __device__ __forceinline__ static void load(const __nv_bfloat16* p, float out[2]) {
        out[0] = __bfloat162float(p[0]);
        out[1] = __bfloat162float(p[1]);
    }
};

template <>
struct VecLoad<__nv_bfloat16, 4> {
    __device__ __forceinline__ static void load(const __nv_bfloat16* p, float out[4]) {
        out[0] = __bfloat162float(p[0]);
        out[1] = __bfloat162float(p[1]);
        out[2] = __bfloat162float(p[2]);
        out[3] = __bfloat162float(p[3]);
    }
};

template <>
struct VecLoad<__nv_bfloat16, 8> {
    __device__ __forceinline__ static void load(const __nv_bfloat16* p, float out[8]) {
        VecLoad<__nv_bfloat16, 4>::load(p, out);
        VecLoad<__nv_bfloat16, 4>::load(p + 4, out + 4);
    }
};

// ============================================================================
// FP8 OPTIMIZED VECTOR LOAD HELPERS
// Use wider loads (32-bit, 64-bit) for better memory throughput
// ============================================================================

// Convert 4 packed FP8 values (uint32_t) to 4 floats
__device__ __forceinline__ void fp8x4_to_float4(uint32_t packed, float& f0, float& f1, float& f2, float& f3) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __nv_fp8_storage_t s0 = (packed >>  0) & 0xFF;
    __nv_fp8_storage_t s1 = (packed >>  8) & 0xFF;
    __nv_fp8_storage_t s2 = (packed >> 16) & 0xFF;
    __nv_fp8_storage_t s3 = (packed >> 24) & 0xFF;
    f0 = __half2float(__nv_cvt_fp8_to_halfraw(s0, __NV_E4M3));
    f1 = __half2float(__nv_cvt_fp8_to_halfraw(s1, __NV_E4M3));
    f2 = __half2float(__nv_cvt_fp8_to_halfraw(s2, __NV_E4M3));
    f3 = __half2float(__nv_cvt_fp8_to_halfraw(s3, __NV_E4M3));
#else
    __nv_fp8_e4m3 v0, v1, v2, v3;
    *reinterpret_cast<uint8_t*>(&v0) = (packed >>  0) & 0xFF;
    *reinterpret_cast<uint8_t*>(&v1) = (packed >>  8) & 0xFF;
    *reinterpret_cast<uint8_t*>(&v2) = (packed >> 16) & 0xFF;
    *reinterpret_cast<uint8_t*>(&v3) = (packed >> 24) & 0xFF;
    f0 = to_f32<__nv_fp8_e4m3>(v0);
    f1 = to_f32<__nv_fp8_e4m3>(v1);
    f2 = to_f32<__nv_fp8_e4m3>(v2);
    f3 = to_f32<__nv_fp8_e4m3>(v3);
#endif
}

// Convert 8 packed FP8 values (uint64_t / 2x uint32_t) to 8 floats  
__device__ __forceinline__ void fp8x8_to_float8(uint32_t lo, uint32_t hi, float out[8]) {
    fp8x4_to_float4(lo, out[0], out[1], out[2], out[3]);
    fp8x4_to_float4(hi, out[4], out[5], out[6], out[7]);
}

// Convert 16 packed FP8 values (uint4 = 4x uint32_t) to 16 floats
__device__ __forceinline__ void fp8x16_to_float16(uint4 packed, float out[16]) {
    fp8x4_to_float4(packed.x, out[0],  out[1],  out[2],  out[3]);
    fp8x4_to_float4(packed.y, out[4],  out[5],  out[6],  out[7]);
    fp8x4_to_float4(packed.z, out[8],  out[9],  out[10], out[11]);
    fp8x4_to_float4(packed.w, out[12], out[13], out[14], out[15]);
}

// FP8 VecLoad specializations - scalar loads for alignment safety
template <>
struct VecLoad<__nv_fp8_e4m3, 1> {
    __device__ __forceinline__ static void load(const __nv_fp8_e4m3* p, float out[1]) {
        out[0] = to_f32<__nv_fp8_e4m3>(p[0]);
    }
};

template <>
struct VecLoad<__nv_fp8_e4m3, 2> {
    __device__ __forceinline__ static void load(const __nv_fp8_e4m3* p, float out[2]) {
        // Scalar loads for alignment safety
        out[0] = to_f32<__nv_fp8_e4m3>(p[0]);
        out[1] = to_f32<__nv_fp8_e4m3>(p[1]);
    }
};

template <>
struct VecLoad<__nv_fp8_e4m3, 4> {
    __device__ __forceinline__ static void load(const __nv_fp8_e4m3* p, float out[4]) {
        // Scalar loads for alignment safety
        out[0] = to_f32<__nv_fp8_e4m3>(p[0]);
        out[1] = to_f32<__nv_fp8_e4m3>(p[1]);
        out[2] = to_f32<__nv_fp8_e4m3>(p[2]);
        out[3] = to_f32<__nv_fp8_e4m3>(p[3]);
    }
};

template <>
struct VecLoad<__nv_fp8_e4m3, 8> {
    __device__ __forceinline__ static void load(const __nv_fp8_e4m3* p, float out[8]) {
        // Scalar loads for alignment safety
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            out[i] = to_f32<__nv_fp8_e4m3>(p[i]);
        }
    }
};

// VecLoad for 16 FP8 values
template <>
struct VecLoad<__nv_fp8_e4m3, 16> {
    __device__ __forceinline__ static void load(const __nv_fp8_e4m3* p, float out[16]) {
        // Scalar loads for alignment safety
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            out[i] = to_f32<__nv_fp8_e4m3>(p[i]);
        }
    }
};

/// 128-bit load/store helpers
__device__ __forceinline__ uint4 ld_u4(const void* p) {
    return *reinterpret_cast<const uint4*>(p);
}

__device__ __forceinline__ void st_u4(void* p, const uint4& v) {
    *reinterpret_cast<uint4*>(p) = v;
}

/// Vectorized 32-bit store of two 16-bit elements
template <typename T>
__device__ __forceinline__ void store_vec2(T* dst, float v0, float v1) {
    T t0 = from_f32<T>(v0);
    T t1 = from_f32<T>(v1);
    if constexpr (sizeof(T) == 2) {
        uint32_t packed = (reinterpret_cast<uint16_t&>(t1) << 16) |
                           reinterpret_cast<uint16_t&>(t0);
        *reinterpret_cast<uint32_t*>(dst) = packed;
    } else {
        dst[0] = t0;
        dst[1] = t1;
    }
}

// ============================================================================
// ASYNC COPY PRIMITIVES (SM80+)
// ============================================================================

/**
 * Wait for N outstanding async copy groups to complete.
 * USE_TC=true uses cp.async instructions, USE_TC=false is no-op (sync loads).
 */
template <int N, bool USE_TC>
__device__ __forceinline__ void cp_async_wait() {
    if constexpr (USE_TC) {
        static_assert(N >= 0 && N <= 8, "N must be 0-8");
        if constexpr (N == 0) asm volatile("cp.async.wait_group 0;" ::);
        else if constexpr (N == 1) asm volatile("cp.async.wait_group 1;" ::);
        else if constexpr (N == 2) asm volatile("cp.async.wait_group 2;" ::);
        else if constexpr (N == 3) asm volatile("cp.async.wait_group 3;" ::);
        else if constexpr (N == 4) asm volatile("cp.async.wait_group 4;" ::);
    }
}

/// 16-byte async copy from global to shared memory
template <bool USE_TC, typename T = void>
__device__ __forceinline__ void cp_async_16(void* smem_dst, const void* gmem_src) {
    // Check 16-byte alignment (required for cp.async and vectorized copy)
    bool aligned16 = (reinterpret_cast<uintptr_t>(gmem_src) & 0xF) == 0;
    
    bool smem_aligned16 = (reinterpret_cast<uintptr_t>(smem_dst) & 0xF) == 0;
    
    if constexpr (USE_TC) {
        if (aligned16 && smem_aligned16) {
            uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_dst));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                         :: "r"(smem_addr), "l"(gmem_src));
        } else {
            // Unaligned: fall back to scalar copy
            const uint8_t* src = reinterpret_cast<const uint8_t*>(gmem_src);
            uint8_t* dst = reinterpret_cast<uint8_t*>(smem_dst);
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                dst[i] = src[i];
            }
        }
    } else {
        if (aligned16 || (reinterpret_cast<uintptr_t>(gmem_src) & 0x3) == 0) {
            // At least 4-byte aligned: use vectorized copy
            const uint32_t* src = reinterpret_cast<const uint32_t*>(gmem_src);
            uint32_t* dst = reinterpret_cast<uint32_t*>(smem_dst);
            dst[0] = src[0]; dst[1] = src[1]; dst[2] = src[2]; dst[3] = src[3];
        } else {
            // Scalar fallback for unaligned access
            const uint8_t* src = reinterpret_cast<const uint8_t*>(gmem_src);
            uint8_t* dst = reinterpret_cast<uint8_t*>(smem_dst);
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                dst[i] = src[i];
            }
        }
    }
}

/// Commit current async copy group
template <bool USE_TC>
__device__ __forceinline__ void cp_async_commit() {
    if constexpr (USE_TC) {
        asm volatile("cp.async.commit_group;" ::);
    }
}

/// Wait for all async copies
template <bool USE_TC>
__device__ __forceinline__ void cp_async_wait0() {
    if constexpr (USE_TC) {
        asm volatile("cp.async.wait_group 0;" ::);
    }
}

// cp_async_cg_16 is now provided by convert_all.cuh

// ============================================================================
// FP8 KV CACHE LOAD/STORE HELPERS (Vectorized)
// ============================================================================

/// Convert 4 packed FP8 → 2 half2
__device__ __forceinline__ void cvt_fp8x4_to_half2x2(
    __half2& out0, __half2& out1,
    uint32_t fp8x4
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    // SM89+ has native FP8 conversion
    // Convert 4 FP8 to 4 half via intermediate
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&fp8x4);
    __half h0 = __nv_cvt_fp8_to_halfraw(bytes[0], __NV_E4M3);
    __half h1 = __nv_cvt_fp8_to_halfraw(bytes[1], __NV_E4M3);
    __half h2 = __nv_cvt_fp8_to_halfraw(bytes[2], __NV_E4M3);
    __half h3 = __nv_cvt_fp8_to_halfraw(bytes[3], __NV_E4M3);
    out0 = __halves2half2(h0, h1);
    out1 = __halves2half2(h2, h3);
#else
    // Fallback: scalar conversion
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&fp8x4);
    float f0 = to_f32<__nv_fp8_e4m3>(*reinterpret_cast<__nv_fp8_e4m3*>(&bytes[0]));
    float f1 = to_f32<__nv_fp8_e4m3>(*reinterpret_cast<__nv_fp8_e4m3*>(&bytes[1]));
    float f2 = to_f32<__nv_fp8_e4m3>(*reinterpret_cast<__nv_fp8_e4m3*>(&bytes[2]));
    float f3 = to_f32<__nv_fp8_e4m3>(*reinterpret_cast<__nv_fp8_e4m3*>(&bytes[3]));
    out0 = __floats2half2_rn(f0, f1);
    out1 = __floats2half2_rn(f2, f3);
#endif
}

/// Convert 2 half2 → 4 packed FP8
__device__ __forceinline__ uint32_t cvt_half2x2_to_fp8x4(
    __half2 in0, __half2 in1
) {
    // Extract individual halves using intrinsics (__half2 doesn't have .x/.y members)
    __half in0_lo = __low2half(in0);
    __half in0_hi = __high2half(in0);
    __half in1_lo = __low2half(in1);
    __half in1_hi = __high2half(in1);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    uint32_t result;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
    bytes[0] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&in0_lo), __NV_SATFINITE, __NV_E4M3);
    bytes[1] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&in0_hi), __NV_SATFINITE, __NV_E4M3);
    bytes[2] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&in1_lo), __NV_SATFINITE, __NV_E4M3);
    bytes[3] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&in1_hi), __NV_SATFINITE, __NV_E4M3);
    return result;
#else
    uint32_t result;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
    *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[0]) = from_f32<__nv_fp8_e4m3>(__half2float(in0_lo));
    *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[1]) = from_f32<__nv_fp8_e4m3>(__half2float(in0_hi));
    *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[2]) = from_f32<__nv_fp8_e4m3>(__half2float(in1_lo));
    *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[3]) = from_f32<__nv_fp8_e4m3>(__half2float(in1_hi));
    return result;
#endif
}

/// Vectorized FP8→F16 load (8 elements per thread)
/// k_base_offset and v_base_offset may differ when K and V are in different arenas.
template <int HD>
__device__ __forceinline__ void load_kv_vec_fp8_to_f16(
    __half* __restrict__ smem_k,
    __half* __restrict__ smem_v,
    const __nv_fp8_e4m3* __restrict__ k_arena,
    const __nv_fp8_e4m3* __restrict__ v_arena,
    int64_t k_base_offset,
    int64_t v_base_offset,
    int tid,
    int block_dim
) {
    static_assert(HD % 8 == 0, "HD must be multiple of 8 for vectorized FP8 load");
    
    constexpr int ELEMS_PER_THREAD = 8;
    const __nv_fp8_e4m3* k_base = k_arena + k_base_offset;
    const __nv_fp8_e4m3* v_base = v_arena + v_base_offset;
    
    // Check alignment once before the loop (need 8-byte alignment for uint2 loads)
    bool aligned = ((reinterpret_cast<uintptr_t>(k_base) & 0x7) == 0) &&
                   ((reinterpret_cast<uintptr_t>(v_base) & 0x7) == 0);
    
    if (aligned) {
        // Fast vectorized path: 8-byte loads
        #pragma unroll
        for (int d = tid * ELEMS_PER_THREAD; d < HD; d += block_dim * ELEMS_PER_THREAD) {
            const uint32_t* k_src = reinterpret_cast<const uint32_t*>(k_base + d);
            const uint32_t* v_src = reinterpret_cast<const uint32_t*>(v_base + d);
            uint32_t k_fp8_01 = k_src[0];
            uint32_t k_fp8_23 = k_src[1];
            uint32_t v_fp8_01 = v_src[0];
            uint32_t v_fp8_23 = v_src[1];
            
            // Convert FP8x4 -> half2x2
            __half2 k_h2_0, k_h2_1, k_h2_2, k_h2_3;
            __half2 v_h2_0, v_h2_1, v_h2_2, v_h2_3;
            cvt_fp8x4_to_half2x2(k_h2_0, k_h2_1, k_fp8_01);
            cvt_fp8x4_to_half2x2(k_h2_2, k_h2_3, k_fp8_23);
            cvt_fp8x4_to_half2x2(v_h2_0, v_h2_1, v_fp8_01);
            cvt_fp8x4_to_half2x2(v_h2_2, v_h2_3, v_fp8_23);
            
            // Store as half2 (vectorized store to smem)
            __half2* k_dst = reinterpret_cast<__half2*>(smem_k + d);
            __half2* v_dst = reinterpret_cast<__half2*>(smem_v + d);
            k_dst[0] = k_h2_0; k_dst[1] = k_h2_1; k_dst[2] = k_h2_2; k_dst[3] = k_h2_3;
            v_dst[0] = v_h2_0; v_dst[1] = v_h2_1; v_dst[2] = v_h2_2; v_dst[3] = v_h2_3;
        }
    } else {
        // Scalar fallback for unaligned access
        #pragma unroll
        for (int d = tid * ELEMS_PER_THREAD; d < HD; d += block_dim * ELEMS_PER_THREAD) {
            const __nv_fp8_e4m3* k_ptr = k_base + d;
            const __nv_fp8_e4m3* v_ptr = v_base + d;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                smem_k[d + i] = from_f32<__half>(to_f32<__nv_fp8_e4m3>(k_ptr[i]));
                smem_v[d + i] = from_f32<__half>(to_f32<__nv_fp8_e4m3>(v_ptr[i]));
            }
        }
    }
}

/// Vectorized F16→FP8 store (8 elements per thread)
/// k_base_offset and v_base_offset may differ when K and V are in different arenas.
template <int HD>
__device__ __forceinline__ void store_kv_vec_f16_to_fp8(
    __nv_fp8_e4m3* __restrict__ k_arena,
    __nv_fp8_e4m3* __restrict__ v_arena,
    const __half* __restrict__ smem_k,
    const __half* __restrict__ smem_v,
    int64_t k_base_offset,
    int64_t v_base_offset,
    int tid,
    int block_dim
) {
    static_assert(HD % 8 == 0, "HD must be multiple of 8 for vectorized FP8 store");
    
    constexpr int ELEMS_PER_THREAD = 8;
    __nv_fp8_e4m3* k_base = k_arena + k_base_offset;
    __nv_fp8_e4m3* v_base = v_arena + v_base_offset;
    
    // Check alignment once before the loop (need 8-byte alignment for uint2 stores)
    bool aligned = ((reinterpret_cast<uintptr_t>(k_base) & 0x7) == 0) &&
                   ((reinterpret_cast<uintptr_t>(v_base) & 0x7) == 0);
    
    if (aligned) {
        // Fast vectorized path: 8-byte stores
        #pragma unroll
        for (int d = tid * ELEMS_PER_THREAD; d < HD; d += block_dim * ELEMS_PER_THREAD) {
            // Load 8 F16 values as 4 half2 (vectorized load from smem)
            const __half2* k_src = reinterpret_cast<const __half2*>(smem_k + d);
            const __half2* v_src = reinterpret_cast<const __half2*>(smem_v + d);
            __half2 k_h2_0 = k_src[0], k_h2_1 = k_src[1], k_h2_2 = k_src[2], k_h2_3 = k_src[3];
            __half2 v_h2_0 = v_src[0], v_h2_1 = v_src[1], v_h2_2 = v_src[2], v_h2_3 = v_src[3];
            
            // Convert half2x2 -> FP8x4
            uint32_t k_fp8_01 = cvt_half2x2_to_fp8x4(k_h2_0, k_h2_1);
            uint32_t k_fp8_23 = cvt_half2x2_to_fp8x4(k_h2_2, k_h2_3);
            uint32_t v_fp8_01 = cvt_half2x2_to_fp8x4(v_h2_0, v_h2_1);
            uint32_t v_fp8_23 = cvt_half2x2_to_fp8x4(v_h2_2, v_h2_3);
            
            // Store as uint32_t (vectorized 8-byte store)
            uint32_t* k_dst = reinterpret_cast<uint32_t*>(k_base + d);
            uint32_t* v_dst = reinterpret_cast<uint32_t*>(v_base + d);
            k_dst[0] = k_fp8_01; k_dst[1] = k_fp8_23;
            v_dst[0] = v_fp8_01; v_dst[1] = v_fp8_23;
        }
    } else {
        // Scalar fallback for unaligned access
        #pragma unroll
        for (int d = tid * ELEMS_PER_THREAD; d < HD; d += block_dim * ELEMS_PER_THREAD) {
            __nv_fp8_e4m3* k_dst = k_base + d;
            __nv_fp8_e4m3* v_dst = v_base + d;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                k_dst[i] = from_f32<__nv_fp8_e4m3>(__half2float(smem_k[d + i]));
                v_dst[i] = from_f32<__nv_fp8_e4m3>(__half2float(smem_v[d + i]));
            }
        }
    }
}

/// Load K/V tile from FP8 arena to F16 smem (vectorized, 8-byte loads) — per-head resolved
template <int HD, int TILE_K>
__device__ __forceinline__ void load_kv_tile_from_fp8_arena_vec(
    __half* __restrict__ smem_k,
    __half* __restrict__ smem_v,
    const Palette4PerHeadEntry* __restrict__ per_head_table,
    const ChunkMeta* cm_batch,
    const int64_t* hg_batch,
    int k_start,
    int tile_len,
    int kv_head_idx,
    int n_kv_head,
    int arena_chunks,
    int max_blocks,
    int tid,
    int block_dim
) {
    const int64_t head_stride = (int64_t)CHUNK_SIZE * (int64_t)HD;
    
    // Process each position in tile
    for (int t = 0; t < tile_len; ++t) {
        int k_pos = k_start + t;
        int logical_block = chunk_div(k_pos);
        int within_block = chunk_mod(k_pos);
        
        if (logical_block >= max_blocks) {
            // Zero fill this row
            for (int d = tid; d < HD; d += block_dim) {
                smem_k[t * HD + d] = __float2half(0.f);
                smem_v[t * HD + d] = __float2half(0.f);
            }
            continue;
        }
        
        int64_t global_chunk_id = head_gid_k(hg_batch, logical_block, kv_head_idx, n_kv_head);
        int64_t global_v_chunk_id = head_gid_v(hg_batch, logical_block, kv_head_idx, n_kv_head);
        if (global_chunk_id < 0) {
            for (int d = tid; d < HD; d += block_dim) {
                smem_k[t * HD + d] = __float2half(0.f);
                smem_v[t * HD + d] = __float2half(0.f);
            }
            continue;
        }
        
        int k_arena_idx = (int)(global_chunk_id / (int64_t)arena_chunks);
        int k_chunk_idx = (int)(global_chunk_id % (int64_t)arena_chunks);
        int v_arena_idx = (int)(global_v_chunk_id / (int64_t)arena_chunks);
        int v_chunk_idx = (int)(global_v_chunk_id % (int64_t)arena_chunks);
        
        PerHeadTableEntry ph_k = per_head_lookup(per_head_table, k_arena_idx, kv_head_idx, n_kv_head);
        PerHeadTableEntry ph_v = per_head_lookup(per_head_table, v_arena_idx, kv_head_idx, n_kv_head);
        const __nv_fp8_e4m3* k_arena = reinterpret_cast<const __nv_fp8_e4m3*>(per_head_k_ptr(ph_k));
        const __nv_fp8_e4m3* v_arena = reinterpret_cast<const __nv_fp8_e4m3*>(per_head_v_ptr(ph_v));
        int64_t k_base = (int64_t)k_chunk_idx * ph_k.k_chunk_byte_stride +
                       (int64_t)within_block * (int64_t)HD;
        int64_t v_base = (int64_t)v_chunk_idx * ph_v.v_chunk_byte_stride +
                       (int64_t)within_block * (int64_t)HD;
        
        // Vectorized load for this row — separate K and V base offsets
        load_kv_vec_fp8_to_f16<HD>(
            smem_k + t * HD, smem_v + t * HD,
            k_arena, v_arena, k_base, v_base, tid, block_dim);
    }
    
    // Zero-fill remaining positions (vectorized)
    for (int t = tile_len; t < TILE_K; ++t) {
        __half2* k_dst = reinterpret_cast<__half2*>(smem_k + t * HD);
        __half2* v_dst = reinterpret_cast<__half2*>(smem_v + t * HD);
        __half2 zero = __float2half2_rn(0.f);
        for (int d = tid; d < HD / 2; d += block_dim) {
            k_dst[d] = zero;
            v_dst[d] = zero;
        }
    }
}

/// Store single K/V position from F16 smem to FP8 arena — per-head resolved
/// Accepts separate K/V arena indices and chunk indices for independent K/V addressing.
template <int HD>
__device__ __forceinline__ void store_kv_position_to_fp8_arena_vec(
    const Palette4PerHeadEntry* __restrict__ per_head_table,
    const __half* __restrict__ smem_k,
    const __half* __restrict__ smem_v,
    int k_arena_idx,
    int v_arena_idx,
    int k_chunk_idx,
    int v_chunk_idx,
    int within_block,
    int kv_head_idx,
    int n_kv_head,
    int tid,
    int block_dim
) {
    PerHeadTableEntry ph_k = per_head_lookup(per_head_table, k_arena_idx, kv_head_idx, n_kv_head);
    PerHeadTableEntry ph_v = per_head_lookup(per_head_table, v_arena_idx, kv_head_idx, n_kv_head);
    const int64_t head_stride = (int64_t)CHUNK_SIZE * (int64_t)HD;
    
    __nv_fp8_e4m3* k_arena = reinterpret_cast<__nv_fp8_e4m3*>(per_head_k_ptr_mut(ph_k));
    __nv_fp8_e4m3* v_arena = reinterpret_cast<__nv_fp8_e4m3*>(per_head_v_ptr_mut(ph_v));
    
    int64_t k_base = (int64_t)k_chunk_idx * (ph_k.k_chunk_byte_stride > 0 ? ph_k.k_chunk_byte_stride : head_stride) +
                   (int64_t)within_block * (int64_t)HD;
    int64_t v_base = (int64_t)v_chunk_idx * (ph_v.v_chunk_byte_stride > 0 ? ph_v.v_chunk_byte_stride : head_stride) +
                   (int64_t)within_block * (int64_t)HD;
    
    store_kv_vec_f16_to_fp8<HD>(k_arena, v_arena, smem_k, smem_v, k_base, v_base, tid, block_dim);
}

// ============================================================================
// MMA INSTRUCTION WRAPPERS
// ============================================================================

/// m16n8k16 MMA with F32 accumulation (auto-selects f16/bf16 opcode)
template <typename T>
__device__ __forceinline__ void mma_sync_m16n8k16_row_col_f32(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
        );
    } else {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
        );
    }
}

/**
 * Execute m16n8k32 MMA with FP8 E4M3 inputs and float32 accumulation.
 * Available on SM89+ (Ada Lovelace, Hopper).
 *
 * A operand: 16×32 (4 uint32 registers, 16 FP8 values each)
 * B operand: 32×8  (2 uint32 registers, 16 FP8 values each)
 * D output:  16×8  (4 float registers, accumulated)
 *
 * Note: This processes 32 K-dimension elements per MMA (2× the F16 version)
 */
__device__ __forceinline__ void mma_sync_m16n8k32_row_col_f32_fp8(
    float& d0, float& d1, float& d2, float& d3,
    uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
    uint32_t b0, uint32_t b1
) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    // SM89+ (Ada): FP8 E4M3 tensor core instruction
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1)
    );
#else
    // Fallback: no FP8 tensor cores available
    (void)d0; (void)d1; (void)d2; (void)d3;
    (void)a0; (void)a1; (void)a2; (void)a3;
    (void)b0; (void)b1;
#endif
}

/// 2×half2 → packed FP8x4
__device__ __forceinline__ uint32_t cvt_f16x4_to_fp8x4(__half2 h01, __half2 h23) {
    // Extract individual halves using intrinsics (__half2 doesn't have .x/.y members)
    __half h0 = __low2half(h01);
    __half h1 = __high2half(h01);
    __half h2 = __low2half(h23);
    __half h3 = __high2half(h23);
    uint32_t result;
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
    bytes[0] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h0), __NV_SATFINITE, __NV_E4M3);
    bytes[1] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h1), __NV_SATFINITE, __NV_E4M3);
    bytes[2] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h2), __NV_SATFINITE, __NV_E4M3);
    bytes[3] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h3), __NV_SATFINITE, __NV_E4M3);
#else
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
    *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[0]) = from_f32<__nv_fp8_e4m3>(__half2float(h0));
    *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[1]) = from_f32<__nv_fp8_e4m3>(__half2float(h1));
    *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[2]) = from_f32<__nv_fp8_e4m3>(__half2float(h2));
    *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[3]) = from_f32<__nv_fp8_e4m3>(__half2float(h3));
#endif
    return result;
}

/// Pack 4 FP8 from smem into uint32 (assumes 4-byte alignment - for smem hot paths)
__device__ __forceinline__ uint32_t pack_fp8x4(
    const __nv_fp8_e4m3* ptr
) {
    // smem accesses at multiples of 4 bytes are always aligned
    return *reinterpret_cast<const uint32_t*>(ptr);
}

/// Pack 4 FP8 from gmem into uint32 (with alignment check - for gmem cold paths)
__device__ __forceinline__ uint32_t pack_fp8x4_safe(
    const __nv_fp8_e4m3* ptr
) {
    // Check if pointer is 4-byte aligned
    if ((reinterpret_cast<uintptr_t>(ptr) & 0x3) == 0) {
        return *reinterpret_cast<const uint32_t*>(ptr);
    } else {
        // Unaligned: use byte-wise loads
        uint32_t result;
        uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
        bytes[0] = *reinterpret_cast<const uint8_t*>(&ptr[0]);
        bytes[1] = *reinterpret_cast<const uint8_t*>(&ptr[1]);
        bytes[2] = *reinterpret_cast<const uint8_t*>(&ptr[2]);
        bytes[3] = *reinterpret_cast<const uint8_t*>(&ptr[3]);
        return result;
    }
}

/// 4×float → packed FP8x4 (for P values)
__device__ __forceinline__ uint32_t cvt_f32x4_to_fp8x4(float f0, float f1, float f2, float f3) {
    uint32_t result;
    uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
    __half h0 = __float2half(f0);
    __half h1 = __float2half(f1);
    __half h2 = __float2half(f2);
    __half h3 = __float2half(f3);
    bytes[0] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h0), __NV_SATFINITE, __NV_E4M3);
    bytes[1] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h1), __NV_SATFINITE, __NV_E4M3);
    bytes[2] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h2), __NV_SATFINITE, __NV_E4M3);
    bytes[3] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&h3), __NV_SATFINITE, __NV_E4M3);
#else
    *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[0]) = from_f32<__nv_fp8_e4m3>(f0);
    *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[1]) = from_f32<__nv_fp8_e4m3>(f1);
    *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[2]) = from_f32<__nv_fp8_e4m3>(f2);
    *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[3]) = from_f32<__nv_fp8_e4m3>(f3);
#endif
    return result;
}

/**
 * Redistribute P values from m16n8k16 output layout to m16n8k32 A operand layout.
 * 
 * Input layout (what each tid owns from S=Q×K^T m16n8k16 output):
 *   tid owns k positions: tid*2, tid*2+1, tid*2+8, tid*2+9
 * 
 * Output layout (what each tid needs for m16n8k32 P×V A operand):
 *   tid needs k positions: tid*4, tid*4+1, tid*4+2, tid*4+3
 * 
 * Cost: 4 shuffles per subtile = 8 shuffles per row = 16 total for 2 rows
 */
__device__ __forceinline__ void redistribute_p_for_fp8_pv(
    // Output: 4 consecutive k values for m16n8k32
    float& out0, float& out1, float& out2, float& out3,
    // Input: P values at k = tid*2, tid*2+1, tid*2+8, tid*2+9
    float p_k0, float p_k1, float p_k8, float p_k9,
    int lane  // 0-31
) {
    const int tid = lane & 3;       // 0-3 within group
    const int groupID = lane >> 2;  // 0-7
    const int base_lane = groupID * 4;
    
    // Source thread mapping for m16n8k32 A operand:
    // tid=0 needs k[0,1,2,3]:   src0=0 (k[0,1]), src1=1 (k[2,3])
    // tid=1 needs k[4,5,6,7]:   src0=2 (k[4,5]), src1=3 (k[6,7])
    // tid=2 needs k[8,9,10,11]: src0=0 (k[8,9]), src1=1 (k[10,11])
    // tid=3 needs k[12,13,14,15]: src0=2 (k[12,13]), src1=3 (k[14,15])
    
    // Correct source mapping:
    // tid=0: needs k[0-3], get k[0,1] from tid=0, k[2,3] from tid=1
    // tid=1: needs k[4-7], get k[4,5] from tid=2, k[6,7] from tid=3
    // tid=2: needs k[8-11], get k[8,9] from tid=0 (p_k8,p_k9), k[10,11] from tid=1
    // tid=3: needs k[12-15], get k[12,13] from tid=2 (p_k8,p_k9), k[14,15] from tid=3
    
    // Which source thread pair to use
    const int src_pair = (tid & 1);  // 0 → threads 0,1; 1 → threads 2,3
    const int src_a = src_pair * 2;       // 0 or 2
    const int src_b = src_pair * 2 + 1;   // 1 or 3
    
    // IMPORTANT: selection (low-k vs high-k) must happen AFTER shuffles.
    // If we select before shuffling, source lanes with tid<2 would never
    // contribute their p_k8/p_k9 (they would only send p_k0/p_k1), which
    // breaks tid>=2 destinations.
    const bool use_high_k = (tid >= 2);

    // Gather low-k from the chosen source lanes.
    const float a_k0 = __shfl_sync(0xffffffff, p_k0, base_lane + src_a);
    const float a_k1 = __shfl_sync(0xffffffff, p_k1, base_lane + src_a);
    const float b_k0 = __shfl_sync(0xffffffff, p_k0, base_lane + src_b);
    const float b_k1 = __shfl_sync(0xffffffff, p_k1, base_lane + src_b);

    // Gather high-k from the chosen source lanes.
    const float a_k8 = __shfl_sync(0xffffffff, p_k8, base_lane + src_a);
    const float a_k9 = __shfl_sync(0xffffffff, p_k9, base_lane + src_a);
    const float b_k8 = __shfl_sync(0xffffffff, p_k8, base_lane + src_b);
    const float b_k9 = __shfl_sync(0xffffffff, p_k9, base_lane + src_b);

    // Select the correct quartet.
    out0 = use_high_k ? a_k8 : a_k0;
    out1 = use_high_k ? a_k9 : a_k1;
    out2 = use_high_k ? b_k8 : b_k0;
    out3 = use_high_k ? b_k9 : b_k1;
}

// ============================================================================
// FP8 TENSOR CORE COMPUTE FUNCTIONS
// ============================================================================

// FP8 E4M3: max ~448, min ~1.95e-3. MMA accumulates in F32.
// Well-conditioned models (post-LayerNorm Q/K in [-1,1]) work fine.
// If NaN/Inf observed: add FP8_PRESCALE or use E5M2 format.

/**
 * Compute 16×16 QK^T using FP8 m16n8k32 MMA (2× throughput vs F16).
 * Q in smem type T, K in FP8 smem. When T=FP8, Q is directly packed.
 * When T=F16/BF16, Q is converted to FP8 on the fly.
 *
 * CRITICAL: m16n8k32 A operand has INTERLEAVED row layout per PTX ISA:
 *   a[0] bits[7:0]   = A[groupID, col0]
 *   a[0] bits[15:8]  = A[groupID+8, col0]
 *   a[0] bits[23:16] = A[groupID, col1]
 *   a[0] bits[31:24] = A[groupID+8, col1]
 */
template <typename T, int HEAD_DIM>
__device__ __forceinline__ void compute_qk_16x16_fp8(
    float& s0_a, float& s1_a, float& s2_a, float& s3_a,
    float& s0_b, float& s1_b, float& s2_b, float& s3_b,
    const T* __restrict__ smem_q,      // Q in T (F16 or BF16) [16][HEAD_DIM]
    const __nv_fp8_e4m3* __restrict__ smem_k,  // K in FP8 [16][HEAD_DIM]
    int lane
) {
    const int groupID = lane >> 2;  // 0-7, selects row pair (r, r+8)
    const int tid = lane & 3;       // 0-3, selects K-column group

    // Initialize MMA accumulators
    float mma1_d0 = 0.f, mma1_d1 = 0.f, mma1_d2 = 0.f, mma1_d3 = 0.f;
    float mma2_d0 = 0.f, mma2_d1 = 0.f, mma2_d2 = 0.f, mma2_d3 = 0.f;

    const int row0 = groupID;
    const int row1 = groupID + 8;

    // Process HEAD_DIM in chunks of 32 (FP8 MMA k-dimension)
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; d += 32) {
        // =========================================
        // Load Q fragment and convert to FP8 with INTERLEAVED row layout
        // A operand for m16n8k32 (CORRECT layout per PTX ISA):
        //   a[0]: rows {groupID, groupID+8} × cols {tid*4+0, tid*4+1}
        //   a[1]: rows {groupID, groupID+8} × cols {tid*4+2, tid*4+3}
        //   a[2]: rows {groupID, groupID+8} × cols {tid*4+16, tid*4+17}
        //   a[3]: rows {groupID, groupID+8} × cols {tid*4+18, tid*4+19}
        // =========================================
        uint32_t q_regs[4];
        {
            // Column bases for the 4 registers
            const int c0 = tid * 4;       // cols 0,1 for a[0]
            const int c2 = tid * 4 + 2;   // cols 2,3 for a[1]
            const int c16 = tid * 4 + 16; // cols 16,17 for a[2]
            const int c18 = tid * 4 + 18; // cols 18,19 for a[3]
            
            // Helper to pack 2 cols × 2 rows into FP8x4
            // PTX ISA m16n8k32 A operand layout (per byte index i=0,1,2,3):
            //   byte[0] = A[groupID,   tid*2]     (row0, col0)
            //   byte[1] = A[groupID,   tid*2+1]   (row0, col1)
            //   byte[2] = A[groupID+8, tid*2]     (row1, col0)
            //   byte[3] = A[groupID+8, tid*2+1]   (row1, col1)
            auto pack_interleaved = [&](int col_base) -> uint32_t {
                uint32_t result;
                uint8_t* bytes = reinterpret_cast<uint8_t*>(&result);
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
                if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
                    // FP8 path: Q already in FP8, just pack with interleaved layout
                    bytes[0] = *reinterpret_cast<const uint8_t*>(&smem_q[row0 * HEAD_DIM + d + col_base]);
                    bytes[1] = *reinterpret_cast<const uint8_t*>(&smem_q[row0 * HEAD_DIM + d + col_base + 1]);
                    bytes[2] = *reinterpret_cast<const uint8_t*>(&smem_q[row1 * HEAD_DIM + d + col_base]);
                    bytes[3] = *reinterpret_cast<const uint8_t*>(&smem_q[row1 * HEAD_DIM + d + col_base + 1]);
                } else if constexpr (std::is_same_v<T, __half>) {
                    // F16 path: convert half → FP8
                    __half q_r0_c0 = smem_q[row0 * HEAD_DIM + d + col_base];
                    __half q_r0_c1 = smem_q[row0 * HEAD_DIM + d + col_base + 1];
                    __half q_r1_c0 = smem_q[row1 * HEAD_DIM + d + col_base];
                    __half q_r1_c1 = smem_q[row1 * HEAD_DIM + d + col_base + 1];
                    bytes[0] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&q_r0_c0), __NV_SATFINITE, __NV_E4M3);
                    bytes[1] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&q_r0_c1), __NV_SATFINITE, __NV_E4M3);
                    bytes[2] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&q_r1_c0), __NV_SATFINITE, __NV_E4M3);
                    bytes[3] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&q_r1_c1), __NV_SATFINITE, __NV_E4M3);
                } else {
                    // BF16 path: bf16 → float → half → fp8
                    __half q_r0_c0 = __float2half(to_f32(smem_q[row0 * HEAD_DIM + d + col_base]));
                    __half q_r0_c1 = __float2half(to_f32(smem_q[row0 * HEAD_DIM + d + col_base + 1]));
                    __half q_r1_c0 = __float2half(to_f32(smem_q[row1 * HEAD_DIM + d + col_base]));
                    __half q_r1_c1 = __float2half(to_f32(smem_q[row1 * HEAD_DIM + d + col_base + 1]));
                    bytes[0] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&q_r0_c0), __NV_SATFINITE, __NV_E4M3);
                    bytes[1] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&q_r0_c1), __NV_SATFINITE, __NV_E4M3);
                    bytes[2] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&q_r1_c0), __NV_SATFINITE, __NV_E4M3);
                    bytes[3] = __nv_cvt_halfraw_to_fp8(*reinterpret_cast<__half_raw*>(&q_r1_c1), __NV_SATFINITE, __NV_E4M3);
                }
#else
                // Fallback for pre-SM89
                *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[0]) = from_f32<__nv_fp8_e4m3>(to_f32(smem_q[row0 * HEAD_DIM + d + col_base]));
                *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[1]) = from_f32<__nv_fp8_e4m3>(to_f32(smem_q[row0 * HEAD_DIM + d + col_base + 1]));
                *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[2]) = from_f32<__nv_fp8_e4m3>(to_f32(smem_q[row1 * HEAD_DIM + d + col_base]));
                *reinterpret_cast<__nv_fp8_e4m3*>(&bytes[3]) = from_f32<__nv_fp8_e4m3>(to_f32(smem_q[row1 * HEAD_DIM + d + col_base + 1]));
#endif
                return result;
            };
            
            q_regs[0] = pack_interleaved(c0);   // cols tid*4+0, tid*4+1
            q_regs[1] = pack_interleaved(c2);   // cols tid*4+2, tid*4+3
            q_regs[2] = pack_interleaved(c16);  // cols tid*4+16, tid*4+17
            q_regs[3] = pack_interleaved(c18);  // cols tid*4+18, tid*4+19
        }

        // =========================================
        // Load K^T fragments (B operand) - layout is correct
        // B operand for m16n8k32 (32×8, column-major):
        // For K^T where K is [N_keys][HEAD_DIM], K^T[k, n] = K[n, k]
        //   b[0] = K[groupID, d + tid*4 : d + tid*4+3]
        //   b[1] = K[groupID, d + tid*4+16 : d + tid*4+19]
        // =========================================
        
        // K^T for output cols 0-7 (K rows 0-7)
        uint32_t kt1_regs[2];
        {
            const int k_row = groupID;  // K row = output column in S
            const int k_col0 = tid * 4;
            const int k_col1 = tid * 4 + 16;
            kt1_regs[0] = pack_fp8x4(&smem_k[k_row * HEAD_DIM + d + k_col0]);
            kt1_regs[1] = pack_fp8x4(&smem_k[k_row * HEAD_DIM + d + k_col1]);
        }

        // K^T for output cols 8-15 (K rows 8-15)
        uint32_t kt2_regs[2];
        {
            const int k_row = groupID + 8;  // K row 8-15
            const int k_col0 = tid * 4;
            const int k_col1 = tid * 4 + 16;
            kt2_regs[0] = pack_fp8x4(&smem_k[k_row * HEAD_DIM + d + k_col0]);
            kt2_regs[1] = pack_fp8x4(&smem_k[k_row * HEAD_DIM + d + k_col1]);
        }

        // Execute FP8 MMAs
        mma_sync_m16n8k32_row_col_f32_fp8(mma1_d0, mma1_d1, mma1_d2, mma1_d3,
                                          q_regs[0], q_regs[1], q_regs[2], q_regs[3],
                                          kt1_regs[0], kt1_regs[1]);
        mma_sync_m16n8k32_row_col_f32_fp8(mma2_d0, mma2_d1, mma2_d2, mma2_d3,
                                          q_regs[0], q_regs[1], q_regs[2], q_regs[3],
                                          kt2_regs[0], kt2_regs[1]);
    }

    // Map MMA outputs to caller's variables
    s0_a = mma1_d0; s1_a = mma1_d1; s2_a = mma1_d2; s3_a = mma1_d3;
    s0_b = mma2_d0; s1_b = mma2_d1; s2_b = mma2_d2; s3_b = mma2_d3;
}

/**
 * Compute P×V using FP8 m16n8k32 MMA. P in F32 regs, V in FP8 smem.
 * Uses warp shuffles to redistribute P from m16n8k16 layout to m16n8k32.
 * Output: out0_* = row groupID, out1_* = row groupID+8, cols tid*2, tid*2+1, +8, +9.
 */
template <int HEAD_DIM>
__device__ __forceinline__ void compute_pv_fp8(
    float& out0_a, float& out0_b, float& out0_c, float& out0_d,
    float& out1_a, float& out1_b, float& out1_c, float& out1_d,
    // P values from m16n8k16 output: row0 owns k[tid*2, tid*2+1, tid*2+8, tid*2+9]
    float p_r0_st0_0, float p_r0_st0_1, float p_r0_st0_4, float p_r0_st0_5,
    float p_r0_st1_0, float p_r0_st1_1, float p_r0_st1_4, float p_r0_st1_5,
    // P values from m16n8k16 output: row1 owns same k pattern
    float p_r1_st0_2, float p_r1_st0_3, float p_r1_st0_6, float p_r1_st0_7,
    float p_r1_st1_2, float p_r1_st1_3, float p_r1_st1_6, float p_r1_st1_7,
    const __nv_fp8_e4m3* __restrict__ smem_v,  // V in FP8 [TILE_K][HEAD_DIM]
    int v_col_base,
    int k_len,
    int lane
) {
    const int tid = lane & 3;
    const int groupID = lane >> 2;

    // =========================================================================
    // Step 1: Redistribute P values via warp shuffles (ONE TIME, 16 shuffles)
    // =========================================================================
    // For m16n8k32, each tid needs 4 consecutive k values: [tid*4, tid*4+3]
    // Input has: tid owns k = [tid*2, tid*2+1, tid*2+8, tid*2+9]
    
    float p_row0_k[8];  // k[tid*4..tid*4+3] for subtile 0 and subtile 1
    float p_row1_k[8];
    
    // Subtile 0 (k=0..15): redistribute row0's P values
    redistribute_p_for_fp8_pv(
        p_row0_k[0], p_row0_k[1], p_row0_k[2], p_row0_k[3],
        p_r0_st0_0, p_r0_st0_1, p_r0_st0_4, p_r0_st0_5, lane);
    
    // Row1 (groupID+8): same pattern with r1 values
    redistribute_p_for_fp8_pv(
        p_row1_k[0], p_row1_k[1], p_row1_k[2], p_row1_k[3],
        p_r1_st0_2, p_r1_st0_3, p_r1_st0_6, p_r1_st0_7, lane);
    
    // Subtile 1 (k=16..31)
    if (k_len > 16) {
        redistribute_p_for_fp8_pv(
            p_row0_k[4], p_row0_k[5], p_row0_k[6], p_row0_k[7],
            p_r0_st1_0, p_r0_st1_1, p_r0_st1_4, p_r0_st1_5, lane);
        redistribute_p_for_fp8_pv(
            p_row1_k[4], p_row1_k[5], p_row1_k[6], p_row1_k[7],
            p_r1_st1_2, p_r1_st1_3, p_r1_st1_6, p_r1_st1_7, lane);
    } else {
        p_row0_k[4] = p_row0_k[5] = p_row0_k[6] = p_row0_k[7] = 0.f;
        p_row1_k[4] = p_row1_k[5] = p_row1_k[6] = p_row1_k[7] = 0.f;
    }

    // =========================================================================
    // Step 2: Pack P into FP8 m16n8k32 A operand format (ONE TIME)
    // =========================================================================
    // m16n8k32 A operand layout:
    //   a[0] = row0, k[tid*4..tid*4+3]      (4 FP8 values from subtile 0)
    //   a[1] = row1, k[tid*4..tid*4+3]      (4 FP8 values from subtile 0)
    //   a[2] = row0, k[tid*4+16..tid*4+19]  (4 FP8 values from subtile 1)
    //   a[3] = row1, k[tid*4+16..tid*4+19]  (4 FP8 values from subtile 1)
    
    uint32_t p_regs[4];
    p_regs[0] = cvt_f32x4_to_fp8x4(p_row0_k[0], p_row0_k[1], p_row0_k[2], p_row0_k[3]);
    p_regs[1] = cvt_f32x4_to_fp8x4(p_row1_k[0], p_row1_k[1], p_row1_k[2], p_row1_k[3]);
    p_regs[2] = cvt_f32x4_to_fp8x4(p_row0_k[4], p_row0_k[5], p_row0_k[6], p_row0_k[7]);
    p_regs[3] = cvt_f32x4_to_fp8x4(p_row1_k[4], p_row1_k[5], p_row1_k[6], p_row1_k[7]);

    // =========================================================================
    // Step 3: Load V and execute FP8 MMAs for 16 output columns
    // =========================================================================
    // Need TWO MMAs to cover 16 V columns (matching the BF16 path output):
    //   MMA1: V columns v_col_base + groupID (outputs d0,d1 → out0_a,out0_b and d2,d3 → out1_a,out1_b)
    //   MMA2: V columns v_col_base + groupID + 8 (outputs d0,d1 → out0_c,out0_d and d2,d3 → out1_c,out1_d)
    //
    // B operand (32×8, col-major): 
    //   b[0] = V[k=tid*4..tid*4+3, col=groupID]
    //   b[1] = V[k=tid*4+16..tid*4+19, col=groupID]
    
    const int k_base0 = tid * 4;
    const int k_base1 = tid * 4 + 16;
    
    // Helper lambda to load V fragment for a given column offset
    auto load_v_fragment = [&](uint32_t v_regs_out[2], int col_offset) {
        const int v_col = v_col_base + groupID + col_offset;
        uint8_t* r0 = reinterpret_cast<uint8_t*>(&v_regs_out[0]);
        uint8_t* r1 = reinterpret_cast<uint8_t*>(&v_regs_out[1]);
        
        if (k_base0 + 3 < k_len) {
            *reinterpret_cast<__nv_fp8_e4m3*>(&r0[0]) = smem_v[k_base0 * HEAD_DIM + v_col];
            *reinterpret_cast<__nv_fp8_e4m3*>(&r0[1]) = smem_v[(k_base0 + 1) * HEAD_DIM + v_col];
            *reinterpret_cast<__nv_fp8_e4m3*>(&r0[2]) = smem_v[(k_base0 + 2) * HEAD_DIM + v_col];
            *reinterpret_cast<__nv_fp8_e4m3*>(&r0[3]) = smem_v[(k_base0 + 3) * HEAD_DIM + v_col];
        } else {
            *reinterpret_cast<__nv_fp8_e4m3*>(&r0[0]) = (k_base0 < k_len) ? smem_v[k_base0 * HEAD_DIM + v_col] : zero_val<__nv_fp8_e4m3>();
            *reinterpret_cast<__nv_fp8_e4m3*>(&r0[1]) = (k_base0 + 1 < k_len) ? smem_v[(k_base0 + 1) * HEAD_DIM + v_col] : zero_val<__nv_fp8_e4m3>();
            *reinterpret_cast<__nv_fp8_e4m3*>(&r0[2]) = (k_base0 + 2 < k_len) ? smem_v[(k_base0 + 2) * HEAD_DIM + v_col] : zero_val<__nv_fp8_e4m3>();
            *reinterpret_cast<__nv_fp8_e4m3*>(&r0[3]) = (k_base0 + 3 < k_len) ? smem_v[(k_base0 + 3) * HEAD_DIM + v_col] : zero_val<__nv_fp8_e4m3>();
        }
        
        if (k_base1 + 3 < k_len) {
            *reinterpret_cast<__nv_fp8_e4m3*>(&r1[0]) = smem_v[k_base1 * HEAD_DIM + v_col];
            *reinterpret_cast<__nv_fp8_e4m3*>(&r1[1]) = smem_v[(k_base1 + 1) * HEAD_DIM + v_col];
            *reinterpret_cast<__nv_fp8_e4m3*>(&r1[2]) = smem_v[(k_base1 + 2) * HEAD_DIM + v_col];
            *reinterpret_cast<__nv_fp8_e4m3*>(&r1[3]) = smem_v[(k_base1 + 3) * HEAD_DIM + v_col];
        } else {
            *reinterpret_cast<__nv_fp8_e4m3*>(&r1[0]) = (k_base1 < k_len) ? smem_v[k_base1 * HEAD_DIM + v_col] : zero_val<__nv_fp8_e4m3>();
            *reinterpret_cast<__nv_fp8_e4m3*>(&r1[1]) = (k_base1 + 1 < k_len) ? smem_v[(k_base1 + 1) * HEAD_DIM + v_col] : zero_val<__nv_fp8_e4m3>();
            *reinterpret_cast<__nv_fp8_e4m3*>(&r1[2]) = (k_base1 + 2 < k_len) ? smem_v[(k_base1 + 2) * HEAD_DIM + v_col] : zero_val<__nv_fp8_e4m3>();
            *reinterpret_cast<__nv_fp8_e4m3*>(&r1[3]) = (k_base1 + 3 < k_len) ? smem_v[(k_base1 + 3) * HEAD_DIM + v_col] : zero_val<__nv_fp8_e4m3>();
        }
    };
    
    // MMA1: V columns v_col_base + groupID (produces outputs for cols tid*2, tid*2+1)
    uint32_t v_regs1[2];
    load_v_fragment(v_regs1, 0);
    
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
    mma_sync_m16n8k32_row_col_f32_fp8(d0, d1, d2, d3,
                                       p_regs[0], p_regs[1], p_regs[2], p_regs[3],
                                       v_regs1[0], v_regs1[1]);
    
    // m16n8k32 output layout:
    //   d0 = O[groupID, 0], d1 = O[groupID, 1]
    //   d2 = O[groupID+8, 0], d3 = O[groupID+8, 1]
    // But we need to map to thread's output columns (tid*2, tid*2+1)
    // Since MMA produces same values for all threads with same groupID,
    // each thread writes to its designated columns via smem_out indexing.
    out0_a = d0; out0_b = d1;  // row groupID
    out1_a = d2; out1_b = d3;  // row groupID+8
    
    // MMA2: V columns v_col_base + groupID + 8 (produces outputs for cols tid*2+8, tid*2+9)
    uint32_t v_regs2[2];
    load_v_fragment(v_regs2, 8);
    
    d0 = 0.f; d1 = 0.f; d2 = 0.f; d3 = 0.f;
    mma_sync_m16n8k32_row_col_f32_fp8(d0, d1, d2, d3,
                                       p_regs[0], p_regs[1], p_regs[2], p_regs[3],
                                       v_regs2[0], v_regs2[1]);
    
    out0_c = d0; out0_d = d1;  // row groupID, cols +8
    out1_c = d2; out1_d = d3;  // row groupID+8, cols +8
}

// ============================================================================
// QK^T COMPUTATION
// ============================================================================

/**
 * Compute 16×16 QK^T tile using two m16n8k16 MMAs.
 *
 * Thread output mapping (per MMA natural layout):
 *   groupID = lane >> 2 (0-7): owns rows groupID, groupID+8
 *   tid = lane & 3 (0-3): owns columns tid*2, tid*2+1, tid*2+8, tid*2+9
 *
 * Output variables:
 *   s0_a = S[groupID, tid*2]        s0_b = S[groupID, tid*2+8]
 *   s1_a = S[groupID, tid*2+1]      s1_b = S[groupID, tid*2+9]
 *   s2_a = S[groupID+8, tid*2]      s2_b = S[groupID+8, tid*2+8]
 *   s3_a = S[groupID+8, tid*2+1]    s3_b = S[groupID+8, tid*2+9]
 *
 * @tparam USE_TC  true=tensor cores, false=scalar fallback
 */
template <typename T, int HEAD_DIM, bool USE_TC>
__device__ __forceinline__ void compute_qk_16x16(
    float& s0_a, float& s1_a, float& s2_a, float& s3_a,
    float& s0_b, float& s1_b, float& s2_b, float& s3_b,
    const T* __restrict__ smem_q,
    const T* __restrict__ smem_k,
    int lane
) {
    if constexpr (USE_TC) {
        const int groupID = lane >> 2;
        const int tid = lane & 3;

        // Initialize MMA accumulators
        float mma1_d0 = 0.f, mma1_d1 = 0.f, mma1_d2 = 0.f, mma1_d3 = 0.f;
        float mma2_d0 = 0.f, mma2_d1 = 0.f, mma2_d2 = 0.f, mma2_d3 = 0.f;

        // Process HEAD_DIM in chunks of 16 (MMA k-dimension)
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 16) {
            // Load Q fragment (A operand) — 16 rows × 16 cols of T arranged as
            // 4 8×8 sub-matrices: (r0-7,c0-7), (r8-15,c0-7), (r0-7,c8-15),
            // (r8-15,c8-15). Each sub-matrix lands in q_regs[i] in that order
            // (matching mma.m16n8k16.row.col.f32 A-operand layout).
            uint32_t q_regs[4];
            if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
                // ldmatrix.x4: 32 lanes, each provides one row-start address
                // for one of the 4 sub-matrices; each lane receives 4 uint32
                // (one per matrix) holding 2 packed fp16/bf16 elements.
                int mat         = lane >> 3;            // 0..3 → which sub-matrix
                int row_in_mat  = lane & 7;             // 0..7 → row within matrix
                int addr_row    = row_in_mat + ((mat & 1) << 3);
                int addr_col    = d + ((mat >> 1) << 3);
                uint32_t s_addr = __cvta_generic_to_shared(
                    &smem_q[addr_row * HEAD_DIM + addr_col]);
                asm volatile(
                    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
                    "{%0, %1, %2, %3}, [%4];\n"
                    : "=r"(q_regs[0]), "=r"(q_regs[1]),
                      "=r"(q_regs[2]), "=r"(q_regs[3])
                    : "r"(s_addr));
            } else {
                int row0 = groupID, row1 = groupID + 8;
                int col0 = tid * 2, col1 = tid * 2 + 1;
                int col2 = tid * 2 + 8, col3 = tid * 2 + 9;

                q_regs[0] = __pack_half2<T>(smem_q[row0 * HEAD_DIM + d + col0],
                                            smem_q[row0 * HEAD_DIM + d + col1]);
                q_regs[1] = __pack_half2<T>(smem_q[row1 * HEAD_DIM + d + col0],
                                            smem_q[row1 * HEAD_DIM + d + col1]);
                q_regs[2] = __pack_half2<T>(smem_q[row0 * HEAD_DIM + d + col2],
                                            smem_q[row0 * HEAD_DIM + d + col3]);
                q_regs[3] = __pack_half2<T>(smem_q[row1 * HEAD_DIM + d + col2],
                                            smem_q[row1 * HEAD_DIM + d + col3]);
            }

            // Load K^T fragment for cols 0-7 (B operand, transposed)
            uint32_t kt1_regs[2];
            {
                int k0 = tid * 2, k1 = tid * 2 + 1;
                int k2 = tid * 2 + 8, k3 = tid * 2 + 9;
                int n = groupID;  // K row 0-7

                kt1_regs[0] = __pack_half2<T>(smem_k[n * HEAD_DIM + d + k0],
                                              smem_k[n * HEAD_DIM + d + k1]);
                kt1_regs[1] = __pack_half2<T>(smem_k[n * HEAD_DIM + d + k2],
                                              smem_k[n * HEAD_DIM + d + k3]);
            }

            // Load K^T fragment for cols 8-15 (B operand)
            uint32_t kt2_regs[2];
            {
                int k0 = tid * 2, k1 = tid * 2 + 1;
                int k2 = tid * 2 + 8, k3 = tid * 2 + 9;
                int n = groupID + 8;  // K row 8-15

                kt2_regs[0] = __pack_half2<T>(smem_k[n * HEAD_DIM + d + k0],
                                              smem_k[n * HEAD_DIM + d + k1]);
                kt2_regs[1] = __pack_half2<T>(smem_k[n * HEAD_DIM + d + k2],
                                              smem_k[n * HEAD_DIM + d + k3]);
            }

            // Execute MMAs
            mma_sync_m16n8k16_row_col_f32<T>(mma1_d0, mma1_d1, mma1_d2, mma1_d3,
                                             q_regs[0], q_regs[1], q_regs[2], q_regs[3],
                                             kt1_regs[0], kt1_regs[1]);
            mma_sync_m16n8k16_row_col_f32<T>(mma2_d0, mma2_d1, mma2_d2, mma2_d3,
                                             q_regs[0], q_regs[1], q_regs[2], q_regs[3],
                                             kt2_regs[0], kt2_regs[1]);
        }

        // Map MMA outputs to caller's variables
        s0_a = mma1_d0; s1_a = mma1_d1; s2_a = mma1_d2; s3_a = mma1_d3;
        s0_b = mma2_d0; s1_b = mma2_d1; s2_b = mma2_d2; s3_b = mma2_d3;
        return;
    }

    // Scalar fallback path
    const int groupID = lane >> 2;
    const int tid = lane & 3;
    const int r_low = groupID, r_high = groupID + 8;
    const int col_a0 = tid * 2, col_a1 = tid * 2 + 1;
    const int col_b0 = tid * 2 + 8, col_b1 = tid * 2 + 9;

    s0_a = s1_a = s2_a = s3_a = 0.f;
    s0_b = s1_b = s2_b = s3_b = 0.f;

    for (int d = 0; d < HEAD_DIM; ++d) {
        float q_low = to_f32(smem_q[r_low * HEAD_DIM + d]);
        float q_high = to_f32(smem_q[r_high * HEAD_DIM + d]);
        float k_a0 = to_f32(smem_k[col_a0 * HEAD_DIM + d]);
        float k_a1 = to_f32(smem_k[col_a1 * HEAD_DIM + d]);
        float k_b0 = to_f32(smem_k[col_b0 * HEAD_DIM + d]);
        float k_b1 = to_f32(smem_k[col_b1 * HEAD_DIM + d]);

        s0_a += q_low * k_a0;  s1_a += q_low * k_a1;
        s2_a += q_high * k_a0; s3_a += q_high * k_a1;
        s0_b += q_low * k_b0;  s1_b += q_low * k_b1;
        s2_b += q_high * k_b0; s3_b += q_high * k_b1;
    }
}

/// QK^T dispatch: FP8 uses m16n8k32 (2× throughput), else m16n8k16.
/// Q_T: query type (can be different from T for mixed precision FP8 KV)
/// T: KV type
template <typename Q_T, typename T, int HEAD_DIM, bool USE_TC>
__device__ __forceinline__ void compute_qk_16x16_dispatch(
    float& s0_a, float& s1_a, float& s2_a, float& s3_a,
    float& s0_b, float& s1_b, float& s2_b, float& s3_b,
    const Q_T* __restrict__ smem_q,
    const T* __restrict__ smem_k,
    int lane
) {
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        // FP8 KV path: use FP8 tensor cores on SM89+
        // Q_T can be BF16/F16 (mixed precision) or FP8 (uniform)
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        if constexpr (USE_TC) {
            // FP8 tensor core path - Q will be converted to FP8 in compute_qk_16x16_fp8
            compute_qk_16x16_fp8<Q_T, HEAD_DIM>(
                s0_a, s1_a, s2_a, s3_a,
                s0_b, s1_b, s2_b, s3_b,
                smem_q, smem_k, lane);
        } else {
#endif
            // Non-TC fallback: convert to float on the fly
            const int groupID = lane >> 2;
            const int tid = lane & 3;
            const int r_low = groupID, r_high = groupID + 8;
            const int col_a0 = tid * 2, col_a1 = tid * 2 + 1;
            const int col_b0 = tid * 2 + 8, col_b1 = tid * 2 + 9;
            
            s0_a = s1_a = s2_a = s3_a = 0.f;
            s0_b = s1_b = s2_b = s3_b = 0.f;
            
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                float q_low = to_f32(smem_q[r_low * HEAD_DIM + d]);
                float q_high = to_f32(smem_q[r_high * HEAD_DIM + d]);
                float k_a0 = to_f32(smem_k[col_a0 * HEAD_DIM + d]);
                float k_a1 = to_f32(smem_k[col_a1 * HEAD_DIM + d]);
                float k_b0 = to_f32(smem_k[col_b0 * HEAD_DIM + d]);
                float k_b1 = to_f32(smem_k[col_b1 * HEAD_DIM + d]);
                
                s0_a += q_low * k_a0;  s1_a += q_low * k_a1;
                s2_a += q_high * k_a0; s3_a += q_high * k_a1;
                s0_b += q_low * k_b0;  s1_b += q_low * k_b1;
                s2_b += q_high * k_b0; s3_b += q_high * k_b1;
            }
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        }
#endif
    } else {
        // Standard F16/BF16 path - Q_T and T should be the same
        static_assert(std::is_same_v<Q_T, T>, "Non-FP8 path requires Q_T == T");
        compute_qk_16x16<T, HEAD_DIM, USE_TC>(
            s0_a, s1_a, s2_a, s3_a,
            s0_b, s1_b, s2_b, s3_b,
            smem_q, smem_k, lane);
    }
}

// ============================================================================
// P×V COMPUTATION WITH REGISTER-BASED P VALUES
// ============================================================================

/**
 * Compute P×V with P in registers (no smem). Uses warp shuffles to gather P.
 * out0_* = row groupID (4 cols), out1_* = row groupID+8 (4 cols).
 */
template <typename T, int HEAD_DIM, bool USE_TC>
__device__ __forceinline__ void compute_pv_from_regs(
    float& out0_a, float& out0_b, float& out0_c, float& out0_d,
    float& out1_a, float& out1_b, float& out1_c, float& out1_d,
    float p_r0_st0_0, float p_r0_st0_1, float p_r0_st0_4, float p_r0_st0_5,
    float p_r0_st1_0, float p_r0_st1_1, float p_r0_st1_4, float p_r0_st1_5,
    float p_r1_st0_2, float p_r1_st0_3, float p_r1_st0_6, float p_r1_st0_7,
    float p_r1_st1_2, float p_r1_st1_3, float p_r1_st1_6, float p_r1_st1_7,
    const T* __restrict__ smem_v,
    int v_col_base,
    int k_len,
    int lane
) {
    const int tid = lane & 3;
    const int groupID = lane >> 2;
    const int col_a = v_col_base + tid * 2;
    const int col_b = v_col_base + tid * 2 + 1;
    const int col_c = v_col_base + tid * 2 + 8;
    const int col_d = v_col_base + tid * 2 + 9;

    if constexpr (USE_TC) {
        // MMA path: P layout matches MMA A operand - no shuffles needed
        const bool has_st1 = (k_len > 16);

        // Pack P values into MMA A operands
        uint32_t p_st0[4], p_st1[4];
        p_st0[0] = __pack_half2<T>(from_f32<T>(p_r0_st0_0), from_f32<T>(p_r0_st0_1));
        p_st0[1] = __pack_half2<T>(from_f32<T>(p_r1_st0_2), from_f32<T>(p_r1_st0_3));
        p_st0[2] = __pack_half2<T>(from_f32<T>(p_r0_st0_4), from_f32<T>(p_r0_st0_5));
        p_st0[3] = __pack_half2<T>(from_f32<T>(p_r1_st0_6), from_f32<T>(p_r1_st0_7));

        if (has_st1) {
            p_st1[0] = __pack_half2<T>(from_f32<T>(p_r0_st1_0), from_f32<T>(p_r0_st1_1));
            p_st1[1] = __pack_half2<T>(from_f32<T>(p_r1_st1_2), from_f32<T>(p_r1_st1_3));
            p_st1[2] = __pack_half2<T>(from_f32<T>(p_r0_st1_4), from_f32<T>(p_r0_st1_5));
            p_st1[3] = __pack_half2<T>(from_f32<T>(p_r1_st1_6), from_f32<T>(p_r1_st1_7));
        }

        // Load V fragments
        uint32_t v1_st0[2], v2_st0[2], v1_st1[2], v2_st1[2];
        {
            int k0 = tid * 2, k1 = k0 + 1, k2 = k0 + 8, k3 = k2 + 1;
            T b0 = (k0 < k_len) ? smem_v[k0 * HEAD_DIM + v_col_base + groupID] : zero_val<T>();
            T b1 = (k1 < k_len) ? smem_v[k1 * HEAD_DIM + v_col_base + groupID] : zero_val<T>();
            T b2 = (k2 < k_len) ? smem_v[k2 * HEAD_DIM + v_col_base + groupID] : zero_val<T>();
            T b3 = (k3 < k_len) ? smem_v[k3 * HEAD_DIM + v_col_base + groupID] : zero_val<T>();
            v1_st0[0] = __pack_half2<T>(b0, b1);
            v1_st0[1] = __pack_half2<T>(b2, b3);

            b0 = (k0 < k_len) ? smem_v[k0 * HEAD_DIM + v_col_base + groupID + 8] : zero_val<T>();
            b1 = (k1 < k_len) ? smem_v[k1 * HEAD_DIM + v_col_base + groupID + 8] : zero_val<T>();
            b2 = (k2 < k_len) ? smem_v[k2 * HEAD_DIM + v_col_base + groupID + 8] : zero_val<T>();
            b3 = (k3 < k_len) ? smem_v[k3 * HEAD_DIM + v_col_base + groupID + 8] : zero_val<T>();
            v2_st0[0] = __pack_half2<T>(b0, b1);
            v2_st0[1] = __pack_half2<T>(b2, b3);
        }

        if (has_st1) {
            int k0 = 16 + tid * 2, k1 = k0 + 1, k2 = k0 + 8, k3 = k2 + 1;
            T b0 = (k0 < k_len) ? smem_v[k0 * HEAD_DIM + v_col_base + groupID] : zero_val<T>();
            T b1 = (k1 < k_len) ? smem_v[k1 * HEAD_DIM + v_col_base + groupID] : zero_val<T>();
            T b2 = (k2 < k_len) ? smem_v[k2 * HEAD_DIM + v_col_base + groupID] : zero_val<T>();
            T b3 = (k3 < k_len) ? smem_v[k3 * HEAD_DIM + v_col_base + groupID] : zero_val<T>();
            v1_st1[0] = __pack_half2<T>(b0, b1);
            v1_st1[1] = __pack_half2<T>(b2, b3);

            b0 = (k0 < k_len) ? smem_v[k0 * HEAD_DIM + v_col_base + groupID + 8] : zero_val<T>();
            b1 = (k1 < k_len) ? smem_v[k1 * HEAD_DIM + v_col_base + groupID + 8] : zero_val<T>();
            b2 = (k2 < k_len) ? smem_v[k2 * HEAD_DIM + v_col_base + groupID + 8] : zero_val<T>();
            b3 = (k3 < k_len) ? smem_v[k3 * HEAD_DIM + v_col_base + groupID + 8] : zero_val<T>();
            v2_st1[0] = __pack_half2<T>(b0, b1);
            v2_st1[1] = __pack_half2<T>(b2, b3);
        }

        // Execute MMAs
        float d0, d1, d2, d3;

        d0 = out0_a; d1 = out0_b; d2 = out1_a; d3 = out1_b;
        mma_sync_m16n8k16_row_col_f32<T>(d0, d1, d2, d3,
            p_st0[0], p_st0[1], p_st0[2], p_st0[3], v1_st0[0], v1_st0[1]);
        out0_a = d0; out0_b = d1; out1_a = d2; out1_b = d3;

        d0 = out0_c; d1 = out0_d; d2 = out1_c; d3 = out1_d;
        mma_sync_m16n8k16_row_col_f32<T>(d0, d1, d2, d3,
            p_st0[0], p_st0[1], p_st0[2], p_st0[3], v2_st0[0], v2_st0[1]);
        out0_c = d0; out0_d = d1; out1_c = d2; out1_d = d3;

        if (has_st1) {
            d0 = out0_a; d1 = out0_b; d2 = out1_a; d3 = out1_b;
            mma_sync_m16n8k16_row_col_f32<T>(d0, d1, d2, d3,
                p_st1[0], p_st1[1], p_st1[2], p_st1[3], v1_st1[0], v1_st1[1]);
            out0_a = d0; out0_b = d1; out1_a = d2; out1_b = d3;

            d0 = out0_c; d1 = out0_d; d2 = out1_c; d3 = out1_d;
            mma_sync_m16n8k16_row_col_f32<T>(d0, d1, d2, d3,
                p_st1[0], p_st1[1], p_st1[2], p_st1[3], v2_st1[0], v2_st1[1]);
            out0_c = d0; out0_d = d1; out1_c = d2; out1_d = d3;
        }
        return;
    }

    // Scalar fallback: use shuffles to gather P from all threads in group
    for (int src_tid = 0; src_tid < 4; ++src_tid) {
        int src_lane = groupID * 4 + src_tid;

        float p0 = __shfl_sync(0xffffffffu, p_r0_st0_0, src_lane);
        float p1 = __shfl_sync(0xffffffffu, p_r0_st0_1, src_lane);
        float p4 = __shfl_sync(0xffffffffu, p_r0_st0_4, src_lane);
        float p5 = __shfl_sync(0xffffffffu, p_r0_st0_5, src_lane);
        float p2 = __shfl_sync(0xffffffffu, p_r1_st0_2, src_lane);
        float p3 = __shfl_sync(0xffffffffu, p_r1_st0_3, src_lane);
        float p6 = __shfl_sync(0xffffffffu, p_r1_st0_6, src_lane);
        float p7 = __shfl_sync(0xffffffffu, p_r1_st0_7, src_lane);

        int k0 = src_tid * 2, k1 = k0 + 1, k8 = k0 + 8, k9 = k8 + 1;

        if (k0 < k_len) {
            float va = to_f32(smem_v[k0 * HEAD_DIM + col_a]);
            float vb = to_f32(smem_v[k0 * HEAD_DIM + col_b]);
            float vc = to_f32(smem_v[k0 * HEAD_DIM + col_c]);
            float vd = to_f32(smem_v[k0 * HEAD_DIM + col_d]);
            out0_a += p0 * va; out0_b += p0 * vb; out0_c += p0 * vc; out0_d += p0 * vd;
            out1_a += p2 * va; out1_b += p2 * vb; out1_c += p2 * vc; out1_d += p2 * vd;
        }
        if (k1 < k_len) {
            float va = to_f32(smem_v[k1 * HEAD_DIM + col_a]);
            float vb = to_f32(smem_v[k1 * HEAD_DIM + col_b]);
            float vc = to_f32(smem_v[k1 * HEAD_DIM + col_c]);
            float vd = to_f32(smem_v[k1 * HEAD_DIM + col_d]);
            out0_a += p1 * va; out0_b += p1 * vb; out0_c += p1 * vc; out0_d += p1 * vd;
            out1_a += p3 * va; out1_b += p3 * vb; out1_c += p3 * vc; out1_d += p3 * vd;
        }
        if (k8 < k_len) {
            float va = to_f32(smem_v[k8 * HEAD_DIM + col_a]);
            float vb = to_f32(smem_v[k8 * HEAD_DIM + col_b]);
            float vc = to_f32(smem_v[k8 * HEAD_DIM + col_c]);
            float vd = to_f32(smem_v[k8 * HEAD_DIM + col_d]);
            out0_a += p4 * va; out0_b += p4 * vb; out0_c += p4 * vc; out0_d += p4 * vd;
            out1_a += p6 * va; out1_b += p6 * vb; out1_c += p6 * vc; out1_d += p6 * vd;
        }
        if (k9 < k_len) {
            float va = to_f32(smem_v[k9 * HEAD_DIM + col_a]);
            float vb = to_f32(smem_v[k9 * HEAD_DIM + col_b]);
            float vc = to_f32(smem_v[k9 * HEAD_DIM + col_c]);
            float vd = to_f32(smem_v[k9 * HEAD_DIM + col_d]);
            out0_a += p5 * va; out0_b += p5 * vb; out0_c += p5 * vc; out0_d += p5 * vd;
            out1_a += p7 * va; out1_b += p7 * vb; out1_c += p7 * vc; out1_d += p7 * vd;
        }
    }

    // Process subtile 1 (k=16..31)
    if (k_len > 16) {
        for (int src_tid = 0; src_tid < 4; ++src_tid) {
            int src_lane = groupID * 4 + src_tid;

            float p0 = __shfl_sync(0xffffffffu, p_r0_st1_0, src_lane);
            float p1 = __shfl_sync(0xffffffffu, p_r0_st1_1, src_lane);
            float p4 = __shfl_sync(0xffffffffu, p_r0_st1_4, src_lane);
            float p5 = __shfl_sync(0xffffffffu, p_r0_st1_5, src_lane);
            float p2 = __shfl_sync(0xffffffffu, p_r1_st1_2, src_lane);
            float p3 = __shfl_sync(0xffffffffu, p_r1_st1_3, src_lane);
            float p6 = __shfl_sync(0xffffffffu, p_r1_st1_6, src_lane);
            float p7 = __shfl_sync(0xffffffffu, p_r1_st1_7, src_lane);

            int k0 = 16 + src_tid * 2, k1 = k0 + 1, k8 = k0 + 8, k9 = k8 + 1;

            if (k0 < k_len) {
                float va = to_f32(smem_v[k0 * HEAD_DIM + col_a]);
                float vb = to_f32(smem_v[k0 * HEAD_DIM + col_b]);
                float vc = to_f32(smem_v[k0 * HEAD_DIM + col_c]);
                float vd = to_f32(smem_v[k0 * HEAD_DIM + col_d]);
                out0_a += p0 * va; out0_b += p0 * vb; out0_c += p0 * vc; out0_d += p0 * vd;
                out1_a += p2 * va; out1_b += p2 * vb; out1_c += p2 * vc; out1_d += p2 * vd;
            }
            if (k1 < k_len) {
                float va = to_f32(smem_v[k1 * HEAD_DIM + col_a]);
                float vb = to_f32(smem_v[k1 * HEAD_DIM + col_b]);
                float vc = to_f32(smem_v[k1 * HEAD_DIM + col_c]);
                float vd = to_f32(smem_v[k1 * HEAD_DIM + col_d]);
                out0_a += p1 * va; out0_b += p1 * vb; out0_c += p1 * vc; out0_d += p1 * vd;
                out1_a += p3 * va; out1_b += p3 * vb; out1_c += p3 * vc; out1_d += p3 * vd;
            }
            if (k8 < k_len) {
                float va = to_f32(smem_v[k8 * HEAD_DIM + col_a]);
                float vb = to_f32(smem_v[k8 * HEAD_DIM + col_b]);
                float vc = to_f32(smem_v[k8 * HEAD_DIM + col_c]);
                float vd = to_f32(smem_v[k8 * HEAD_DIM + col_d]);
                out0_a += p4 * va; out0_b += p4 * vb; out0_c += p4 * vc; out0_d += p4 * vd;
                out1_a += p6 * va; out1_b += p6 * vb; out1_c += p6 * vc; out1_d += p6 * vd;
            }
            if (k9 < k_len) {
                float va = to_f32(smem_v[k9 * HEAD_DIM + col_a]);
                float vb = to_f32(smem_v[k9 * HEAD_DIM + col_b]);
                float vc = to_f32(smem_v[k9 * HEAD_DIM + col_c]);
                float vd = to_f32(smem_v[k9 * HEAD_DIM + col_d]);
                out0_a += p5 * va; out0_b += p5 * vb; out0_c += p5 * vc; out0_d += p5 * vd;
                out1_a += p7 * va; out1_b += p7 * vb; out1_c += p7 * vc; out1_d += p7 * vd;
            }
        }
    }
}

/// P×V dispatch: FP8 uses FP8 tensor cores (2× throughput), else F16/BF16.
template <typename T, int HEAD_DIM, bool USE_TC>
__device__ __forceinline__ void compute_pv_from_regs_dispatch(
    float& out0_a, float& out0_b, float& out0_c, float& out0_d,
    float& out1_a, float& out1_b, float& out1_c, float& out1_d,
    float p_r0_st0_0, float p_r0_st0_1, float p_r0_st0_4, float p_r0_st0_5,
    float p_r0_st1_0, float p_r0_st1_1, float p_r0_st1_4, float p_r0_st1_5,
    float p_r1_st0_2, float p_r1_st0_3, float p_r1_st0_6, float p_r1_st0_7,
    float p_r1_st1_2, float p_r1_st1_3, float p_r1_st1_6, float p_r1_st1_7,
    const T* __restrict__ smem_v,
    int v_col_base,
    int k_len,
    int lane
) {
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        // FP8 path: use compute_pv_fp8 on SM89+
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        if constexpr (USE_TC) {
            compute_pv_fp8<HEAD_DIM>(
                out0_a, out0_b, out0_c, out0_d,
                out1_a, out1_b, out1_c, out1_d,
                p_r0_st0_0, p_r0_st0_1, p_r0_st0_4, p_r0_st0_5,
                p_r0_st1_0, p_r0_st1_1, p_r0_st1_4, p_r0_st1_5,
                p_r1_st0_2, p_r1_st0_3, p_r1_st0_6, p_r1_st0_7,
                p_r1_st1_2, p_r1_st1_3, p_r1_st1_6, p_r1_st1_7,
                smem_v, v_col_base, k_len, lane);
        } else
#endif
        {
            // Scalar fallback: convert FP8 V to float on the fly
            const int tid = lane & 3;
            const int groupID = lane >> 2;
            const int col_a = v_col_base + tid * 2;
            const int col_b = v_col_base + tid * 2 + 1;
            const int col_c = v_col_base + tid * 2 + 8;
            const int col_d = v_col_base + tid * 2 + 9;
            
            // Subtile 0 (k=0..15)
            #pragma unroll
            for (int src_tid = 0; src_tid < 4; ++src_tid) {
                int src_lane = groupID * 4 + src_tid;
                
                float p0 = __shfl_sync(0xffffffffu, p_r0_st0_0, src_lane);
                float p1 = __shfl_sync(0xffffffffu, p_r0_st0_1, src_lane);
                float p4 = __shfl_sync(0xffffffffu, p_r0_st0_4, src_lane);
                float p5 = __shfl_sync(0xffffffffu, p_r0_st0_5, src_lane);
                float p2 = __shfl_sync(0xffffffffu, p_r1_st0_2, src_lane);
                float p3 = __shfl_sync(0xffffffffu, p_r1_st0_3, src_lane);
                float p6 = __shfl_sync(0xffffffffu, p_r1_st0_6, src_lane);
                float p7 = __shfl_sync(0xffffffffu, p_r1_st0_7, src_lane);
                
                int k0 = src_tid * 2, k1 = k0 + 1, k8 = k0 + 8, k9 = k8 + 1;
                
                if (k0 < k_len) {
                    float va = to_f32(smem_v[k0 * HEAD_DIM + col_a]);
                    float vb = to_f32(smem_v[k0 * HEAD_DIM + col_b]);
                    float vc = to_f32(smem_v[k0 * HEAD_DIM + col_c]);
                    float vd = to_f32(smem_v[k0 * HEAD_DIM + col_d]);
                    out0_a += p0 * va; out0_b += p0 * vb; out0_c += p0 * vc; out0_d += p0 * vd;
                    out1_a += p2 * va; out1_b += p2 * vb; out1_c += p2 * vc; out1_d += p2 * vd;
                }
                if (k1 < k_len) {
                    float va = to_f32(smem_v[k1 * HEAD_DIM + col_a]);
                    float vb = to_f32(smem_v[k1 * HEAD_DIM + col_b]);
                    float vc = to_f32(smem_v[k1 * HEAD_DIM + col_c]);
                    float vd = to_f32(smem_v[k1 * HEAD_DIM + col_d]);
                    out0_a += p1 * va; out0_b += p1 * vb; out0_c += p1 * vc; out0_d += p1 * vd;
                    out1_a += p3 * va; out1_b += p3 * vb; out1_c += p3 * vc; out1_d += p3 * vd;
                }
                if (k8 < k_len) {
                    float va = to_f32(smem_v[k8 * HEAD_DIM + col_a]);
                    float vb = to_f32(smem_v[k8 * HEAD_DIM + col_b]);
                    float vc = to_f32(smem_v[k8 * HEAD_DIM + col_c]);
                    float vd = to_f32(smem_v[k8 * HEAD_DIM + col_d]);
                    out0_a += p4 * va; out0_b += p4 * vb; out0_c += p4 * vc; out0_d += p4 * vd;
                    out1_a += p6 * va; out1_b += p6 * vb; out1_c += p6 * vc; out1_d += p6 * vd;
                }
                if (k9 < k_len) {
                    float va = to_f32(smem_v[k9 * HEAD_DIM + col_a]);
                    float vb = to_f32(smem_v[k9 * HEAD_DIM + col_b]);
                    float vc = to_f32(smem_v[k9 * HEAD_DIM + col_c]);
                    float vd = to_f32(smem_v[k9 * HEAD_DIM + col_d]);
                    out0_a += p5 * va; out0_b += p5 * vb; out0_c += p5 * vc; out0_d += p5 * vd;
                    out1_a += p7 * va; out1_b += p7 * vb; out1_c += p7 * vc; out1_d += p7 * vd;
                }
            }
            
            // Subtile 1 (k=16..31)
            if (k_len > 16) {
                #pragma unroll
                for (int src_tid = 0; src_tid < 4; ++src_tid) {
                    int src_lane = groupID * 4 + src_tid;
                    
                    float p0 = __shfl_sync(0xffffffffu, p_r0_st1_0, src_lane);
                    float p1 = __shfl_sync(0xffffffffu, p_r0_st1_1, src_lane);
                    float p4 = __shfl_sync(0xffffffffu, p_r0_st1_4, src_lane);
                    float p5 = __shfl_sync(0xffffffffu, p_r0_st1_5, src_lane);
                    float p2 = __shfl_sync(0xffffffffu, p_r1_st1_2, src_lane);
                    float p3 = __shfl_sync(0xffffffffu, p_r1_st1_3, src_lane);
                    float p6 = __shfl_sync(0xffffffffu, p_r1_st1_6, src_lane);
                    float p7 = __shfl_sync(0xffffffffu, p_r1_st1_7, src_lane);
                    
                    int k0 = 16 + src_tid * 2, k1 = k0 + 1, k8 = k0 + 8, k9 = k8 + 1;
                    
                    if (k0 < k_len) {
                        float va = to_f32(smem_v[k0 * HEAD_DIM + col_a]);
                        float vb = to_f32(smem_v[k0 * HEAD_DIM + col_b]);
                        float vc = to_f32(smem_v[k0 * HEAD_DIM + col_c]);
                        float vd = to_f32(smem_v[k0 * HEAD_DIM + col_d]);
                        out0_a += p0 * va; out0_b += p0 * vb; out0_c += p0 * vc; out0_d += p0 * vd;
                        out1_a += p2 * va; out1_b += p2 * vb; out1_c += p2 * vc; out1_d += p2 * vd;
                    }
                    if (k1 < k_len) {
                        float va = to_f32(smem_v[k1 * HEAD_DIM + col_a]);
                        float vb = to_f32(smem_v[k1 * HEAD_DIM + col_b]);
                        float vc = to_f32(smem_v[k1 * HEAD_DIM + col_c]);
                        float vd = to_f32(smem_v[k1 * HEAD_DIM + col_d]);
                        out0_a += p1 * va; out0_b += p1 * vb; out0_c += p1 * vc; out0_d += p1 * vd;
                        out1_a += p3 * va; out1_b += p3 * vb; out1_c += p3 * vc; out1_d += p3 * vd;
                    }
                    if (k8 < k_len) {
                        float va = to_f32(smem_v[k8 * HEAD_DIM + col_a]);
                        float vb = to_f32(smem_v[k8 * HEAD_DIM + col_b]);
                        float vc = to_f32(smem_v[k8 * HEAD_DIM + col_c]);
                        float vd = to_f32(smem_v[k8 * HEAD_DIM + col_d]);
                        out0_a += p4 * va; out0_b += p4 * vb; out0_c += p4 * vc; out0_d += p4 * vd;
                        out1_a += p6 * va; out1_b += p6 * vb; out1_c += p6 * vc; out1_d += p6 * vd;
                    }
                    if (k9 < k_len) {
                        float va = to_f32(smem_v[k9 * HEAD_DIM + col_a]);
                        float vb = to_f32(smem_v[k9 * HEAD_DIM + col_b]);
                        float vc = to_f32(smem_v[k9 * HEAD_DIM + col_c]);
                        float vd = to_f32(smem_v[k9 * HEAD_DIM + col_d]);
                        out0_a += p5 * va; out0_b += p5 * vb; out0_c += p5 * vc; out0_d += p5 * vd;
                        out1_a += p7 * va; out1_b += p7 * vb; out1_c += p7 * vc; out1_d += p7 * vd;
                    }
                }
            }
        }
    } else {
        // Standard F16/BF16 path
        compute_pv_from_regs<T, HEAD_DIM, USE_TC>(
            out0_a, out0_b, out0_c, out0_d,
            out1_a, out1_b, out1_c, out1_d,
            p_r0_st0_0, p_r0_st0_1, p_r0_st0_4, p_r0_st0_5,
            p_r0_st1_0, p_r0_st1_1, p_r0_st1_4, p_r0_st1_5,
            p_r1_st0_2, p_r1_st0_3, p_r1_st0_6, p_r1_st0_7,
            p_r1_st1_2, p_r1_st1_3, p_r1_st1_6, p_r1_st1_7,
            smem_v, v_col_base, k_len, lane);
    }
}

// ============================================================================
// MAIN KERNEL
// ============================================================================

/**
 * Paged prefill attention forward pass kernel.
 *
 * Implements Flash Attention algorithm with paged KV cache support:
 * 1. Load Q tile once into shared memory
 * 2. Stream K/V tiles with async pipelining
 * 3. Compute QK^T scores with tensor core MMA
 * 4. Apply causal mask and prefix validity
 * 5. Online softmax (running max/sum)
 * 6. Compute P×V and accumulate outputs
 * 7. Write K/V to paged arena during prefill
 *
 * @tparam Q_T          Query element type (defaults to T, but can be different for mixed precision)
 * @tparam T            KV element type (__half, __nv_bfloat16, or __nv_fp8_e4m3)
 * @tparam O            Output type (same as T for non-FP8, __nv_bfloat16 for FP8)
 * @tparam HEAD_DIM     Head dimension (32, 64, 128, or 256)
 * @tparam WARPS_PER_BLOCK  Warps per thread block
 * @tparam TILE_K       K-tile size (typically 32)
 * @tparam BLOCK_M      Q-tile size (typically 32)
 * @tparam HAS_PREFIX   Whether prefix KV cache exists
 * @tparam WARPS_TC     Active warps for tensor core compute
 * @tparam USE_TC       Use tensor cores (SM80+) vs scalar fallback
 * @tparam NUM_STAGES   Pipeline ring-buffer stages — always 2 (double-buffer);
 *                      triple-buffering was removed (it capped occupancy at
 *                      1 block/SM on shared-constrained SMs for no benefit).
 */

// Helper: cp.async one R16 palette (d[] halfs) in [dim][position] order.
// SUB_HD = HEAD_DIM / N_PALETTE, TILE_K positions per tile.
template <typename T, bool USE_TC, int HEAD_DIM, int SUB_HD, int TILE_K>
__device__ __forceinline__ void r16_cp_async_palette(
    const char* head_r, T* dst_buf, int d_base, int in_blk_base, int tid, int block_dim)
{
    constexpr int ELEMS_PER_CP_T  = 16 / (int)sizeof(T);
    constexpr int POS_GROUPS      = TILE_K / ELEMS_PER_CP_T;
    constexpr int R16_BLOCK_BYTES = 128;
    int total_items = SUB_HD * POS_GROUPS;
    for (int idx = tid; idx < total_items; idx += block_dim) {
        int ld      = idx / POS_GROUPS;
        int pg      = idx % POS_GROUPS;
        int pos_off = pg * ELEMS_PER_CP_T;
        const T* src = reinterpret_cast<const T*>(
            head_r + (int64_t)ld * R16_BLOCK_BYTES
                   + (int64_t)(in_blk_base + pos_off) * (int64_t)sizeof(T));
        T* dst = &dst_buf[(d_base + ld) * TILE_K + pos_off];
        cp_async_cg_16<USE_TC>(dst, src);
    }
}

// Helper: in-place transpose smem buffer from [dim][position] to [position][dim] layout.
// Contains two __syncthreads() calls (before and after write-back).
// MIN_BLOCK_DIM: the smallest blockDim.x this will be called with.  Used to size
// the per-thread register array exactly: ceil(TOTAL_ELEMS / MIN_BLOCK_DIM) slots.
// Callers must compute MIN_BLOCK_DIM from their launch configuration.
template <typename T, int HEAD_DIM, int TILE_K, int MIN_BLOCK_DIM>
__device__ __forceinline__ void transpose_smem_dim_pos(T* buf, int tid, int block_dim)
{
    constexpr int TOTAL_ELEMS = HEAD_DIM * TILE_K;
    constexpr int MAX_ITERS = (TOTAL_ELEMS + MIN_BLOCK_DIM - 1) / MIN_BLOCK_DIM;
    T regs[MAX_ITERS];
    int n_mine = 0;
    for (int idx = tid; idx < TOTAL_ELEMS; idx += block_dim) {
        int d = idx / TILE_K;
        int i = idx % TILE_K;
        regs[n_mine++] = buf[d * TILE_K + i];
    }
    __syncthreads();
    n_mine = 0;
    for (int idx = tid; idx < TOTAL_ELEMS; idx += block_dim) {
        int d = idx / TILE_K;
        int i = idx % TILE_K;
        buf[i * HEAD_DIM + d] = regs[n_mine++];
    }
    __syncthreads();
}

// Palette-aware transpose: [palette-ordered dim][position] → [position][global dim].
// Data in buf is laid out as palette 0's SUB_HD dims, then palette 1's, etc.
// pal_map is the original packed 2-bit-per-dim palette map.
template <typename T, int HEAD_DIM, int TILE_K, int MIN_BLOCK_DIM>
__device__ __forceinline__ void transpose_smem_dim_pos_pal_nosync(
    T* buf, const uint8_t* pal_map, int tid, int block_dim)
{
    constexpr int TOTAL_ELEMS = HEAD_DIM * TILE_K;
    constexpr int SUB_HD = HEAD_DIM / 4;
    constexpr int MAX_ITERS = (TOTAL_ELEMS + MIN_BLOCK_DIM - 1) / MIN_BLOCK_DIM;
    T regs[MAX_ITERS];
    int n_mine = 0;
    for (int idx = tid; idx < TOTAL_ELEMS; idx += block_dim) {
        int d = idx / TILE_K;
        int i = idx % TILE_K;
        int p, rank;
        prefill_pal_rank(pal_map, d, &p, &rank);
        int buf_d = p * SUB_HD + rank;
        regs[n_mine++] = buf[buf_d * TILE_K + i];
    }
    __syncthreads();
    n_mine = 0;
    for (int idx = tid; idx < TOTAL_ELEMS; idx += block_dim) {
        int d = idx / TILE_K;
        int i = idx % TILE_K;
        buf[i * HEAD_DIM + d] = regs[n_mine++];
    }
}

// Same transpose, but for the hoisted table form used in the hot prefix path:
// each byte stores palette id in bits[7:6] and rank-within-palette in bits[5:0].
template <typename T, int HEAD_DIM, int TILE_K, int MIN_BLOCK_DIM>
__device__ __forceinline__ void transpose_smem_dim_pos_ranked_pal_nosync(
    T* buf, const uint8_t* pal_table, int tid, int block_dim, const float* scales)
{
    constexpr int TOTAL_ELEMS = HEAD_DIM * TILE_K;
    constexpr int SUB_HD = HEAD_DIM / 4;
    constexpr int MAX_ITERS = (TOTAL_ELEMS + MIN_BLOCK_DIM - 1) / MIN_BLOCK_DIM;
    T regs[MAX_ITERS];
    int n_mine = 0;
    for (int idx = tid; idx < TOTAL_ELEMS; idx += block_dim) {
        int d = idx / TILE_K;
        int i = idx % TILE_K;
        uint8_t pe = pal_table[d];
        int p = (pe >> 6) & 0x3;
        int rank = pe & 0x3f;
        int buf_d = p * SUB_HD + rank;
        regs[n_mine++] = buf[buf_d * TILE_K + i];
    }
    __syncthreads();
    n_mine = 0;
    for (int idx = tid; idx < TOTAL_ELEMS; idx += block_dim) {
        int d = idx / TILE_K;
        int i = idx % TILE_K;
        uint8_t pe = pal_table[d];
        int p = (pe >> 6) & 0x3;
        float scale = scales[p];
        if (scale != 1.0f) {
            buf[i * HEAD_DIM + d] = from_f32<T>(to_f32<T>(regs[n_mine]) / scale);
        } else {
            buf[i * HEAD_DIM + d] = regs[n_mine];
        }
        ++n_mine;
    }
}

template <typename T, int HEAD_DIM, int TILE_K, int MIN_BLOCK_DIM>
__device__ __forceinline__ void transpose_smem_dim_pos_pal(
    T* buf, const uint8_t* pal_map, int tid, int block_dim)
{
    transpose_smem_dim_pos_pal_nosync<T, HEAD_DIM, TILE_K, MIN_BLOCK_DIM>(buf, pal_map, tid, block_dim);
    __syncthreads();
}

template <typename Q_T, typename T, typename O, int HEAD_DIM, int WARPS_PER_BLOCK, int TILE_K, int BLOCK_M,
          bool HAS_PREFIX, int WARPS_TC, bool USE_TC, int NUM_STAGES = 2>
__global__ void __launch_bounds__(128, (HEAD_DIM <= 64) ? 8 : 4)
paged_prefill_attn_fwd_chunks_kernel(
    const Q_T* __restrict__ q,
    const T* __restrict__ k_packed,
    const T* __restrict__ v_packed,
    const uint8_t* __restrict__ headers_ptr,
    const uint32_t* __restrict__ cu_seqlens_q,
    const uint32_t* __restrict__ q_lens,
    const uint32_t* __restrict__ kv_lens,
    O* __restrict__ out,
    int batch_size,
    int n_head,
    int n_kv_head,
    int max_blocks,
    float softmax_scale,
    int total_q,
    const uint32_t* __restrict__ rope_offsets,
    const float* __restrict__ rope_cs,
    int rope_interleaved,
    const uint32_t* __restrict__ write_offset_shifts
) {
    static_assert(HEAD_DIM % 32 == 0, "HEAD_DIM must be multiple of 32");
    static_assert(HEAD_DIM >= 32 && HEAD_DIM <= 256, "HEAD_DIM must be 32-256");
    static_assert(BLOCK_M == 32, "BLOCK_M must be 32");
    static_assert(TILE_K <= 32, "Per-tile validity is packed into a uint32_t mask; TILE_K must be ≤ 32");
    static_assert(CHUNK_SIZE % TILE_K == 0,
                  "TILE_K must divide CHUNK_SIZE so each tile maps to a single chunk slice "
                  "(required by the hoisted palette-table scheme — one slice per tile load).");

    constexpr int HEAD_DIM_CHUNK = (HEAD_DIM >= 64 && HEAD_DIM % 64 == 0) ? 64 : 32;
    constexpr int NUM_HEAD_CHUNKS = HEAD_DIM / HEAD_DIM_CHUNK;
    constexpr int VEC = HEAD_DIM_CHUNK / WARP_SIZE;
    static_assert(VEC == 1 || VEC == 2, "VEC must be 1 or 2");

    constexpr int ROWS_PER_BATCH = 16;
    constexpr int NUM_ROW_BATCHES = BLOCK_M / ROWS_PER_BATCH;

    int tid = (int)threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;
    int batch_idx = (int)blockIdx.z;
    const SlotHeader& slot_hdr = get_slot_header(headers_ptr, batch_idx);

    if (batch_idx >= batch_size) return;

    // ========================================================================
    // SHARED MEMORY LAYOUT
    //   Per-row state is packed:
    //     - Active / write-active rows: one bit per row in a uint32_t mask.
    //     - q_pos:        uint16_t  (q_pos < q_len, fits)
    //     - write slot:   uint8_t  (within-block offset < CHUNK_SIZE = 32)
    //     - slice index:  uint16_t
    //     - row_max_k is recomputed via the row_max_k() lambda; not stored.
    //   Per-tile state (HAS_PREFIX):
    //     - Validity is a single uint32_t mask (one bit per tile slot).
    //     - Palette rank/format for every (stage, head_dim) is precomputed once
    //       per tile load into s_pal_table_{k,v}, eliminating per-element
    //       prefill_pal_rank() calls in the hot transpose / element-wise paths.
    // ========================================================================

    // Batch scalars
    __shared__ int s_q_start, s_q_end, s_q_len, s_kv_len, s_prefix_len;
    __shared__ int s_max_k_max;

    // K/V tile buffers
    constexpr bool USE_FP8 = std::is_same_v<T, __nv_fp8_e4m3>;
    // K/V/Q rings live in dynamic shared memory (double-buffered; the >48 KB
    // budget on SM80+ is opted into via size + cudaFuncSetAttribute in
    // launch_paged_prefill_chunks). Contiguous layout, each region 128-B aligned
    // (per-stage TILE_K·HEAD_DIM·sizeof(T) is a multiple of 128 for every
    // supported HEAD_DIM/TILE_K, so the offsets below stay aligned):
    //   [ smem_k : NUM_STAGES × TILE_K·HEAD_DIM × T   ]
    //   [ smem_v : NUM_STAGES × TILE_K·HEAD_DIM × T   ]
    //   [ smem_q : WARPS_TC   × BLOCK_M·HEAD_DIM × Q_T ]
    extern __shared__ __align__(128) unsigned char prefill_smem[];
    T (*smem_k)[TILE_K * HEAD_DIM] =
        reinterpret_cast<T (*)[TILE_K * HEAD_DIM]>(prefill_smem);
    T (*smem_v)[TILE_K * HEAD_DIM] =
        reinterpret_cast<T (*)[TILE_K * HEAD_DIM]>(
            prefill_smem + sizeof(T) * NUM_STAGES * TILE_K * HEAD_DIM);
    // Q tile (loaded once per block)
    Q_T* smem_q = reinterpret_cast<Q_T*>(
        prefill_smem + sizeof(T) * 2 * NUM_STAGES * TILE_K * HEAD_DIM);

    // Packed per-row metadata
    __shared__ uint32_t s_row_active_mask;        // bit r set iff row r is active
    __shared__ uint32_t s_write_active_mask;      // bit r set iff row r needs KV write
    __shared__ uint16_t s_q_pos[BLOCK_M];         // q_pos < q_len, fits uint16
    __shared__ uint8_t  s_write_in_blk[BLOCK_M];  // < CHUNK_SIZE = 32, fits uint8
    __shared__ uint16_t s_write_slice_idx[BLOCK_M];

    // Packed per-tile validity (HAS_PREFIX only)
    [[maybe_unused]] __shared__ uint32_t smem_valid_mask;
    [[maybe_unused]] __shared__ int32_t  smem_rope_pos[TILE_K];

    // Per-stage postprocess metadata (HAS_PREFIX only)
    [[maybe_unused]] __shared__ uint8_t smem_postprocess_fmt_k[NUM_STAGES][N_PALETTE];
    [[maybe_unused]] __shared__ uint8_t smem_postprocess_fmt_v[NUM_STAGES][N_PALETTE];
    [[maybe_unused]] __shared__ uint8_t smem_postprocess_any[NUM_STAGES];
    [[maybe_unused]] __shared__ float   smem_k_scale[NUM_STAGES][N_PALETTE];
    [[maybe_unused]] __shared__ float   smem_v_scale[NUM_STAGES][N_PALETTE];

    // Hoisted palette tables (HAS_PREFIX only).
    //   Each byte: bits[7:6] = palette id (0..3), bits[5:0] = rank within palette (0..63).
    //   Populated cooperatively once per tile load (TILE_K divides CHUNK_SIZE
    //   ⇒ each tile lies within a single chunk slice).
    [[maybe_unused]] __shared__ uint8_t s_pal_table_k[NUM_STAGES][HEAD_DIM];
    [[maybe_unused]] __shared__ uint8_t s_pal_table_v[NUM_STAGES][HEAD_DIM];
    // Second hoisted tables for the STRADDLE case. Under cum-token addressing a
    // tile can straddle two slices (any prefix containing a partial chunk), and
    // the two slices' pal_maps may differ — routing the second slice's bytes
    // through the first slice's table reads the wrong (palette, block) with the
    // wrong outer scale. These hold the routing for the slice covering the
    // tile's LAST prefix position; the cold path selects table by each
    // position's resolved slice. Tiles spanning ≥3 slices (several tiny slices
    // in one 32-token window) resolve their middle-slice positions with a
    // per-element prefill_pal_rank fallback.
    [[maybe_unused]] __shared__ uint8_t s_pal_table_k2[NUM_STAGES][HEAD_DIM];
    [[maybe_unused]] __shared__ uint8_t s_pal_table_v2[NUM_STAGES][HEAD_DIM];

    // ========================================================================
    // ACCESSORS — hide bit ops, recompute max_k on demand
    // ========================================================================
    auto row_active   = [&](int r) -> bool { return (s_row_active_mask   >> r) & 1u; };
    auto write_active = [&](int r) -> bool { return (s_write_active_mask >> r) & 1u; };
    auto row_max_k = [&](int r) -> int {
        int mk = s_prefix_len + (int)s_q_pos[r] + 1;
        if (mk > s_kv_len) mk = s_kv_len;
        if (mk < 0) mk = 0;
        return mk;
    };
    auto pal_palette = [](uint8_t v) -> int { return (v >> 6) & 0x3; };
    auto pal_rank    = [](uint8_t v) -> int { return v & 0x3F; };

    // ========================================================================
    // BATCH SETUP
    // ========================================================================
    if (tid == 0) {
        s_q_start = (int)cu_seqlens_q[batch_idx];
        s_q_end = (int)cu_seqlens_q[batch_idx + 1];
        s_q_len = (int)q_lens[batch_idx];
        s_kv_len = (int)kv_lens[batch_idx];
        if constexpr (HAS_PREFIX) {
            int pl = s_kv_len - s_q_len;
            s_prefix_len = (pl < 0) ? 0 : pl;
        } else {
            s_prefix_len = 0;
        }
        s_row_active_mask = 0u;
        s_write_active_mask = 0u;
    }
    __syncthreads();

    int q_start = s_q_start;
    int q_end = s_q_end;
    int q_len = s_q_len;
    int kv_len = s_kv_len;
    int prefix_len = s_prefix_len;

    int t_base = q_start + (int)blockIdx.x * BLOCK_M;
    if (t_base >= q_end || t_base >= total_q) return;

    // ========================================================================
    // GQA HEAD MAPPING
    // ========================================================================
    int num_groups = n_head / n_kv_head;
    if (num_groups <= 0) num_groups = 1;
    int head_blocks_per_kv = (num_groups + WARPS_TC - 1) / WARPS_TC;
    int kv_head_idx = (int)blockIdx.y / head_blocks_per_kv;
    if (kv_head_idx >= n_kv_head) return;
    int head_block = (int)blockIdx.y - kv_head_idx * head_blocks_per_kv;
    int head_base = kv_head_idx * num_groups + head_block * WARPS_TC;
    int head_idx = head_base + warp_id;
    int kv_group_end = kv_head_idx * num_groups + num_groups;
    bool active = (warp_id < WARPS_TC) && (head_idx < n_head) && (head_idx < kv_group_end);
    constexpr int COMPUTE_THREADS = WARPS_TC * 32;
    const bool has_post_helper = HAS_PREFIX && ((int)blockDim.x > COMPUTE_THREADS);
    const bool helper_warp = has_post_helper && (warp_id >= WARPS_TC);
    // Exact blockDim.x for this kernel instantiation — used to size the per-thread
    // regs array in the R16 transpose helpers so it's neither too small (overflow)
    // nor wastefully large (LMEM pressure).
    constexpr int PREFILL_BLOCK_DIM = HAS_PREFIX && (WARPS_TC < WARPS_PER_BLOCK)
        ? (WARPS_TC + 1) * 32
        : WARPS_TC * 32;

    // ========================================================================
    // PER-ROW METADATA INIT
    //   atomicOr is race-free here because each lane writes a unique bit.
    // ========================================================================
    if (tid < BLOCK_M) {
        int t = t_base + tid;
        bool row_active_v = (t < q_end) && (t < total_q);
        int q_pos = row_active_v ? (t - q_start) : 0;
        int max_k = prefix_len + q_pos + 1;
        if (max_k > kv_len) max_k = kv_len;
        if (max_k < 0) max_k = 0;

        s_q_pos[tid] = (uint16_t)q_pos;
        if (row_active_v) atomicOr(&s_row_active_mask, 1u << tid);

        bool write_active_v = false;
        int write_in_blk = 0;
        int write_slice_idx = 0;

        if (row_active_v && q_len > 0 && kv_len > 0) {
            int abs_pos = prefix_len + q_pos;
            if (abs_pos >= 0 && abs_pos < kv_len) {
                // Cum-token addressing: position_map covers [0, kv_len)
                // including the write region (filled by host
                // `extend_for_write_region`).  write_offset_shifts is no
                // longer needed — the host's slice layout already
                // places the writer where it belongs.
                int blk, in_blk;
                resolve_pos(slot_hdr, abs_pos, blk, in_blk);
                (void)write_offset_shifts;
                if (blk < (int)slot_hdr.n_slices) {
                    write_active_v = true;
                    write_in_blk = in_blk;
                    write_slice_idx = blk;
                }
            }
        }
        if (write_active_v) atomicOr(&s_write_active_mask, 1u << tid);
        s_write_in_blk[tid] = (uint8_t)write_in_blk;
        s_write_slice_idx[tid] = (uint16_t)write_slice_idx;
    }
    __syncthreads();

    // Compute max_k across all rows on the fly (row_max_k is cheap).
    if (tid < 32) {
        int v = (tid < BLOCK_M && row_active(tid)) ? row_max_k(tid) : 0;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            v = max(v, __shfl_xor_sync(0xffffffffu, v, off));
        if (tid == 0) s_max_k_max = v;
    }
    __syncthreads();

    int max_k_max = s_max_k_max;

    // Early exit for empty sequences
    if (q_len <= 0 || kv_len <= 0 || max_k_max <= 0) {
        if (active) {
            for (int r = 0; r < BLOCK_M; ++r) {
                if (!row_active(r)) continue;
                int t = t_base + r;
                if (t >= total_q) continue;
                int64_t row_base = ((int64_t)t * (int64_t)n_head + (int64_t)head_idx) * (int64_t)HEAD_DIM;
                for (int d = lane; d < HEAD_DIM; d += WARP_SIZE) {
                    out[row_base + d] = from_f32<O>(0.f);
                }
            }
        }
        return;
    }

    // ========================================================================
    // LOAD Q TILE
    // ========================================================================
    if (warp_id < WARPS_TC) {
        constexpr int VEC_BYTES = 16;
        constexpr int Q_ELEMS_PER_VEC = VEC_BYTES / sizeof(Q_T);
        constexpr int Q_VECS_PER_TOKEN = HEAD_DIM / Q_ELEMS_PER_VEC;
        constexpr int Q_TOTAL_VECS = BLOCK_M * Q_VECS_PER_TOKEN;
        constexpr int Q_VECS_LOG2 = (Q_VECS_PER_TOKEN == 2) ? 1 : (Q_VECS_PER_TOKEN == 4) ? 2 :
                                  (Q_VECS_PER_TOKEN == 8) ? 3 : (Q_VECS_PER_TOKEN == 16) ? 4 : 0;

        #pragma unroll 1
        for (int v = lane; v < Q_TOTAL_VECS; v += WARP_SIZE) {
            int row = v >> Q_VECS_LOG2;
            int vv = v & (Q_VECS_PER_TOKEN - 1);
            int elem_off = vv * Q_ELEMS_PER_VEC;
            int t_row = t_base + row;
            Q_T* q_dst = &smem_q[warp_id * BLOCK_M * HEAD_DIM + row * HEAD_DIM + elem_off];

            if (t_row < q_end && t_row < total_q && row_active(row)) {
                const Q_T* qsrc = q + ((int64_t)t_row * (int64_t)n_head + (int64_t)head_idx) * (int64_t)HEAD_DIM + elem_off;
                cp_async_16<USE_TC>(q_dst, qsrc);
            } else {
                #pragma unroll
                for (int i = 0; i < Q_ELEMS_PER_VEC; ++i) q_dst[i] = zero_val<Q_T>();
            }
        }
    }
    cp_async_commit<USE_TC>();
    cp_async_wait0<USE_TC>();
    __syncthreads();

    // ========================================================================
    // FUSED ROPE on Q
    // ========================================================================
    {
        const uint32_t rope_base = rope_offsets[batch_idx];
        if (warp_id < WARPS_TC) {
            Q_T* wq = smem_q + warp_id * BLOCK_M * HEAD_DIM;
            constexpr int HALF_DIM = HEAD_DIM / 2;
            #pragma unroll 1
            for (int r = 0; r < BLOCK_M; ++r) {
                if (!row_active(r)) continue;
                const int rope_pos = prefix_len + (int)s_q_pos[r] + (int)rope_base;
                Q_T* __restrict__ row = wq + r * HEAD_DIM;
                if (!rope_interleaved) {
                    for (int d = lane; d < HALF_DIM; d += WARP_SIZE) {
                        float x = to_f32(row[d]);
                        float y = to_f32(row[d + HALF_DIM]);
                        float cos_v, sin_v;
                        rope_cos_sin<HEAD_DIM>(rope_pos, d, rope_cs, cos_v, sin_v);
                        row[d]            = from_f32<Q_T>(x * cos_v - y * sin_v);
                        row[d + HALF_DIM] = from_f32<Q_T>(x * sin_v + y * cos_v);
                    }
                } else {
                    for (int d = lane; d < HALF_DIM; d += WARP_SIZE) {
                        float x = to_f32(row[d * 2]);
                        float y = to_f32(row[d * 2 + 1]);
                        float cos_v, sin_v;
                        rope_cos_sin<HEAD_DIM>(rope_pos, d, rope_cs, cos_v, sin_v);
                        row[d * 2]     = from_f32<Q_T>(x * cos_v - y * sin_v);
                        row[d * 2 + 1] = from_f32<Q_T>(x * sin_v + y * cos_v);
                    }
                }
            }
        }
        __syncthreads();
    }

    const Q_T* warp_smem_q = smem_q + warp_id * BLOCK_M * HEAD_DIM;

    constexpr int CP_SIZE_T = USE_FP8 ? 8 : 16;
    constexpr int ELEMS_PER_CP_T = CP_SIZE_T / sizeof(T);
    constexpr int TOTAL_CPS_PER_ROW = HEAD_DIM / ELEMS_PER_CP_T;
    constexpr int CPS_LOG2 = (TOTAL_CPS_PER_ROW == 2) ? 1 : (TOTAL_CPS_PER_ROW == 4) ? 2 :
                             (TOTAL_CPS_PER_ROW == 8) ? 3 : (TOTAL_CPS_PER_ROW == 16) ? 4 : 0;

    const int tid_in_warp = lane & 3;
    const int groupID = lane >> 2;

    float m0, m1, m2, m3;
    float l0, l1, l2, l3;
    constexpr int VT_ITERS_PER_CHUNK = HEAD_DIM_CHUNK / 16;
    float o_reg[NUM_ROW_BATCHES][NUM_HEAD_CHUNKS][VT_ITERS_PER_CHUNK][8];

    // ========================================================================
    // MAIN ATTENTION LOOP
    // ========================================================================
    if constexpr (!HAS_PREFIX) {
        // -------- NO PREFIX PATH --------

        auto load_tile = [&](int k0_ld, int stage) {
            int k1_ld = k0_ld + TILE_K;
            if (k1_ld > max_k_max) k1_ld = max_k_max;
            int tile_len_ld = k1_ld - k0_ld;
            if (tile_len_ld <= 0) return;

            T* kdst = smem_k[stage];
            T* vdst = smem_v[stage];
            int total_cps = tile_len_ld * TOTAL_CPS_PER_ROW;

            for (int cp_idx = tid; cp_idx < total_cps; cp_idx += (int)blockDim.x) {
                int i = cp_idx >> CPS_LOG2;
                int cp_in_row = cp_idx & (TOTAL_CPS_PER_ROW - 1);
                int d = cp_in_row * ELEMS_PER_CP_T;
                int tok = q_start + (k0_ld + i);
                int64_t base = ((int64_t)tok * (int64_t)n_kv_head + (int64_t)kv_head_idx) * (int64_t)HEAD_DIM;
                load_kv_chunk<T, USE_TC>(
                    &kdst[i * HEAD_DIM + d], &vdst[i * HEAD_DIM + d],
                    k_packed, v_packed, base + d);
            }
        };

        m0 = m1 = m2 = m3 = NEG_INF_F;
        l0 = l1 = l2 = l3 = 0.f;

        if (active) {
            #pragma unroll
            for (int rb = 0; rb < NUM_ROW_BATCHES; ++rb)
                #pragma unroll
                for (int hc = 0; hc < NUM_HEAD_CHUNKS; ++hc)
                    #pragma unroll
                    for (int vs = 0; vs < VT_ITERS_PER_CHUNK; ++vs)
                        #pragma unroll
                        for (int c = 0; c < 8; ++c) o_reg[rb][hc][vs][c] = 0.f;
        }
        __syncthreads();

        const int n_tiles = (max_k_max + TILE_K - 1) / TILE_K;
        const int last_tile_k0 = (n_tiles - 1) * TILE_K;
        const bool last_tile_partial = (max_k_max % TILE_K) != 0;
        const int last_tile_len = last_tile_partial ? (max_k_max % TILE_K) : TILE_K;

        load_tile(last_tile_k0, 0);
        cp_async_commit<USE_TC>();

        int tiles_loaded = 1;
        if (last_tile_k0 - TILE_K >= 0) {
            load_tile(last_tile_k0 - TILE_K, 1);
            cp_async_commit<USE_TC>();
            tiles_loaded = 2;
        }
        if (tiles_loaded >= NUM_STAGES) cp_async_wait<NUM_STAGES - 1, USE_TC>();
        else                            cp_async_wait<0, USE_TC>();
        __syncthreads();

        int cur_stage = 0;
        int k0 = last_tile_k0;

        for (; k0 >= 0; k0 -= TILE_K) {
            const int tile_len = (k0 == last_tile_k0) ? last_tile_len : TILE_K;
            const bool is_full_tile = (tile_len == TILE_K);
            const int k1 = k0 + tile_len;

            // KV writeback (un-rotated). Identity palette.
            if (head_block == 0) {
                int abs_block_begin = (int)s_q_pos[0];
                int abs_block_end = abs_block_begin + BLOCK_M;
                int write_abs_begin = (k0 > abs_block_begin) ? k0 : abs_block_begin;
                int write_abs_end = (k1 < abs_block_end) ? k1 : abs_block_end;

                if (write_abs_begin < write_abs_end) {
                    int first_q_head = kv_head_idx * num_groups;
                    for (int abs_pos = write_abs_begin; abs_pos < write_abs_end; ++abs_pos) {
                        int r = abs_pos - abs_block_begin;
                        if (!row_active(r) || !write_active(r)) continue;

                        int i_in_tile = abs_pos - k0;
                        const uint8_t* w_slice = get_slice<HEAD_DIM>(slot_hdr.slices_ptr, (int)s_write_slice_idx[r], n_kv_head);
                        const uint8_t* w_head = get_head<HEAD_DIM>(w_slice, kv_head_idx);
                        int src_base = i_in_tile * HEAD_DIM;
                        const Q_T* q_row = q + ((int64_t)(q_start + (int)s_q_pos[r]) * (int64_t)n_head + (int64_t)first_q_head) * (int64_t)HEAD_DIM;
                        constexpr int CHUNK = 8;
                        constexpr int SUB_HD = HEAD_DIM / N_PALETTE;
                        constexpr int VEC_LIMIT = (HEAD_DIM / CHUNK) * CHUNK;
                        for (int d = tid * CHUNK; d < VEC_LIMIT; d += (int)blockDim.x * CHUNK) {
                            int p = d / SUB_HD;
                            int local_d = d - p * SUB_HD;
                            store_kv_chunk_arena<T, Q_T, SUB_HD>(
                                (char*)kvhead_k_ptr<HEAD_DIM>(w_head, p),
                                (char*)kvhead_v_ptr<HEAD_DIM>(w_head, p),
                                &smem_k[cur_stage][src_base + d],
                                &smem_v[cur_stage][src_base + d],
                                &q_row[d],
                                kvhead_k_fmt<HEAD_DIM>(w_head, p),
                                kvhead_v_fmt<HEAD_DIM>(w_head, p),
                                0, 0, (int)s_write_in_blk[r],
                                local_d, 0, 0);
                        }
                        if constexpr (HEAD_DIM % CHUNK != 0) {
                            for (int d = VEC_LIMIT + tid; d < HEAD_DIM; d += (int)blockDim.x) {
                                int p = d / SUB_HD;
                                int local_d = d - p * SUB_HD;
                                store_kv_chunk_arena<T, Q_T, SUB_HD>(
                                    (char*)kvhead_k_ptr<HEAD_DIM>(w_head, p),
                                    (char*)kvhead_v_ptr<HEAD_DIM>(w_head, p),
                                    &smem_k[cur_stage][src_base + d],
                                    &smem_v[cur_stage][src_base + d],
                                    &q_row[d],
                                    kvhead_k_fmt<HEAD_DIM>(w_head, p),
                                    kvhead_v_fmt<HEAD_DIM>(w_head, p),
                                    0, 0, (int)s_write_in_blk[r],
                                    local_d, 0, 0);
                            }
                        }
                    }
                }
            }
            __syncthreads();

            // Fused RoPE on K (no-prefix path)
            {
                const uint32_t rope_base = rope_offsets[batch_idx];
                constexpr int HALF_DIM = HEAD_DIM / 2;
                const int flat_total = tile_len * HALF_DIM;
                if (!rope_interleaved) {
                    for (int flat = tid; flat < flat_total; flat += (int)blockDim.x) {
                        const int i = flat & (HALF_DIM - 1);
                        const int tok = flat >> ((HALF_DIM == 16) ? 4 : (HALF_DIM == 32) ? 5 :
                                                 (HALF_DIM == 64) ? 6 : 7);
                        const int rope_pos = k0 + tok + (int)rope_base;
                        T* row = &smem_k[cur_stage][tok * HEAD_DIM];
                        float x = to_f32(row[i]);
                        float y = to_f32(row[i + HALF_DIM]);
                        float cos_v, sin_v;
                        rope_cos_sin<HEAD_DIM>(rope_pos, i, rope_cs, cos_v, sin_v);
                        row[i]            = from_f32<T>(x * cos_v - y * sin_v);
                        row[i + HALF_DIM] = from_f32<T>(x * sin_v + y * cos_v);
                    }
                } else {
                    for (int flat = tid; flat < flat_total; flat += (int)blockDim.x) {
                        const int i = flat & (HALF_DIM - 1);
                        const int tok = flat >> ((HALF_DIM == 16) ? 4 : (HALF_DIM == 32) ? 5 :
                                                 (HALF_DIM == 64) ? 6 : 7);
                        const int rope_pos = k0 + tok + (int)rope_base;
                        T* row = &smem_k[cur_stage][tok * HEAD_DIM];
                        float x = to_f32(row[i * 2]);
                        float y = to_f32(row[i * 2 + 1]);
                        float cos_v, sin_v;
                        rope_cos_sin<HEAD_DIM>(rope_pos, i, rope_cs, cos_v, sin_v);
                        row[i * 2]     = from_f32<T>(x * cos_v - y * sin_v);
                        row[i * 2 + 1] = from_f32<T>(x * sin_v + y * cos_v);
                    }
                }
                __syncthreads();
            }

            // -------- Row-batch compute --------
            #pragma unroll
            for (int row_batch = 0; row_batch < NUM_ROW_BATCHES; ++row_batch) {
                const int row_start = row_batch * ROWS_PER_BATCH;
                float& m_li0 = (row_batch == 0) ? m0 : m2;
                float& m_li1 = (row_batch == 0) ? m1 : m3;
                float& l_li0 = (row_batch == 0) ? l0 : l2;
                float& l_li1 = (row_batch == 0) ? l1 : l3;

                float scores_reg[2][8];
                if (active) {
                    for (int kt = 0; kt < TILE_K; kt += 16) {
                        const int r0 = row_start + groupID;
                        const int r1 = row_start + groupID + 8;
                        const int col_a0 = kt + tid_in_warp * 2;
                        const int col_a1 = kt + tid_in_warp * 2 + 1;
                        const int col_b0 = kt + tid_in_warp * 2 + 8;
                        const int col_b1 = kt + tid_in_warp * 2 + 9;
                        const int sub_tile = kt >> 4;

                        float s0_a, s1_a, s2_a, s3_a, s0_b, s1_b, s2_b, s3_b;
                        compute_qk_16x16_dispatch<Q_T, T, HEAD_DIM, USE_TC>(
                            s0_a, s1_a, s2_a, s3_a, s0_b, s1_b, s2_b, s3_b,
                            warp_smem_q + row_start * HEAD_DIM,
                            &smem_k[cur_stage][kt * HEAD_DIM], lane);

                        s0_a *= softmax_scale; s1_a *= softmax_scale;
                        s2_a *= softmax_scale; s3_a *= softmax_scale;
                        s0_b *= softmax_scale; s1_b *= softmax_scale;
                        s2_b *= softmax_scale; s3_b *= softmax_scale;

                        int k_pos_a0 = k0 + col_a0, k_pos_a1 = k0 + col_a1;
                        int k_pos_b0 = k0 + col_b0, k_pos_b1 = k0 + col_b1;

                        bool r0_valid = (r0 < BLOCK_M) & row_active(r0);
                        bool r1_valid = (r1 < BLOCK_M) & row_active(r1);
                        int max_k_r0 = r0_valid ? row_max_k(r0) : 0;
                        int max_k_r1 = r1_valid ? row_max_k(r1) : 0;

                        if (is_full_tile) {
                            s0_a = (r0_valid & (k_pos_a0 < max_k_r0)) ? s0_a : NEG_INF_F;
                            s1_a = (r0_valid & (k_pos_a1 < max_k_r0)) ? s1_a : NEG_INF_F;
                            s0_b = (r0_valid & (k_pos_b0 < max_k_r0)) ? s0_b : NEG_INF_F;
                            s1_b = (r0_valid & (k_pos_b1 < max_k_r0)) ? s1_b : NEG_INF_F;
                            s2_a = (r1_valid & (k_pos_a0 < max_k_r1)) ? s2_a : NEG_INF_F;
                            s3_a = (r1_valid & (k_pos_a1 < max_k_r1)) ? s3_a : NEG_INF_F;
                            s2_b = (r1_valid & (k_pos_b0 < max_k_r1)) ? s2_b : NEG_INF_F;
                            s3_b = (r1_valid & (k_pos_b1 < max_k_r1)) ? s3_b : NEG_INF_F;
                        } else {
                            s0_a = (r0_valid & (k_pos_a0 < max_k_r0) & (col_a0 < tile_len)) ? s0_a : NEG_INF_F;
                            s1_a = (r0_valid & (k_pos_a1 < max_k_r0) & (col_a1 < tile_len)) ? s1_a : NEG_INF_F;
                            s0_b = (r0_valid & (k_pos_b0 < max_k_r0) & (col_b0 < tile_len)) ? s0_b : NEG_INF_F;
                            s1_b = (r0_valid & (k_pos_b1 < max_k_r0) & (col_b1 < tile_len)) ? s1_b : NEG_INF_F;
                            s2_a = (r1_valid & (k_pos_a0 < max_k_r1) & (col_a0 < tile_len)) ? s2_a : NEG_INF_F;
                            s3_a = (r1_valid & (k_pos_a1 < max_k_r1) & (col_a1 < tile_len)) ? s3_a : NEG_INF_F;
                            s2_b = (r1_valid & (k_pos_b0 < max_k_r1) & (col_b0 < tile_len)) ? s2_b : NEG_INF_F;
                            s3_b = (r1_valid & (k_pos_b1 < max_k_r1) & (col_b1 < tile_len)) ? s3_b : NEG_INF_F;
                        }

                        scores_reg[sub_tile][0] = s0_a; scores_reg[sub_tile][1] = s1_a;
                        scores_reg[sub_tile][2] = s2_a; scores_reg[sub_tile][3] = s3_a;
                        scores_reg[sub_tile][4] = s0_b; scores_reg[sub_tile][5] = s1_b;
                        scores_reg[sub_tile][6] = s2_b; scores_reg[sub_tile][7] = s3_b;
                    }
                }

                float alpha0 = 1.f, alpha1 = 1.f;
                float p0_st0 = 0, p1_st0 = 0, p4_st0 = 0, p5_st0 = 0;
                float p0_st1 = 0, p1_st1 = 0, p4_st1 = 0, p5_st1 = 0;
                float p2_st0 = 0, p3_st0 = 0, p6_st0 = 0, p7_st0 = 0;
                float p2_st1 = 0, p3_st1 = 0, p6_st1 = 0, p7_st1 = 0;

                if (active) {
                    const int r0 = row_start + groupID;
                    const int r1 = row_start + groupID + 8;
                    const bool row0_active = row_active(r0);
                    const bool row1_active = row_active(r1);

                    {
                        float local_max_r0 = fmaxf(fmaxf(scores_reg[0][0], scores_reg[0][1]),
                                                   fmaxf(scores_reg[0][4], scores_reg[0][5]));
                        local_max_r0 = fmaxf(local_max_r0,
                                             fmaxf(fmaxf(scores_reg[1][0], scores_reg[1][1]),
                                                   fmaxf(scores_reg[1][4], scores_reg[1][5])));
                        local_max_r0 = fmaxf(local_max_r0, __shfl_xor_sync(0xffffffffu, local_max_r0, 1));
                        local_max_r0 = fmaxf(local_max_r0, __shfl_xor_sync(0xffffffffu, local_max_r0, 2));

                        float m_prev = m_li0;
                        float m_new = fmaxf(m_prev, local_max_r0);
                        float new_alpha0 = fast_exp::exp<float, fast_exp::Softmax>(m_prev - m_new);

                        float4 exp_st0 = fast_exp::exp4<float, fast_exp::Softmax>(make_float4(
                            scores_reg[0][0] - m_new, scores_reg[0][1] - m_new,
                            scores_reg[0][4] - m_new, scores_reg[0][5] - m_new));
                        float4 exp_st1 = fast_exp::exp4<float, fast_exp::Softmax>(make_float4(
                            scores_reg[1][0] - m_new, scores_reg[1][1] - m_new,
                            scores_reg[1][4] - m_new, scores_reg[1][5] - m_new));
                        p0_st0 = exp_st0.x; p1_st0 = exp_st0.y; p4_st0 = exp_st0.z; p5_st0 = exp_st0.w;
                        p0_st1 = exp_st1.x; p1_st1 = exp_st1.y; p4_st1 = exp_st1.z; p5_st1 = exp_st1.w;

                        float local_sum_r0 = exp_st0.x + exp_st0.y + exp_st0.z + exp_st0.w +
                                             exp_st1.x + exp_st1.y + exp_st1.z + exp_st1.w;
                        local_sum_r0 += __shfl_xor_sync(0xffffffffu, local_sum_r0, 1);
                        local_sum_r0 += __shfl_xor_sync(0xffffffffu, local_sum_r0, 2);

                        alpha0 = row0_active ? new_alpha0 : 1.f;
                        m_li0 = row0_active ? m_new : m_li0;
                        float l_new0 = l_li0 * new_alpha0 + local_sum_r0;
                        l_li0 = row0_active ? fmaxf(l_new0, 1e-30f) : l_li0;
                    }

                    {
                        float local_max_r1 = fmaxf(fmaxf(scores_reg[0][2], scores_reg[0][3]),
                                                   fmaxf(scores_reg[0][6], scores_reg[0][7]));
                        local_max_r1 = fmaxf(local_max_r1,
                                             fmaxf(fmaxf(scores_reg[1][2], scores_reg[1][3]),
                                                   fmaxf(scores_reg[1][6], scores_reg[1][7])));
                        local_max_r1 = fmaxf(local_max_r1, __shfl_xor_sync(0xffffffffu, local_max_r1, 1));
                        local_max_r1 = fmaxf(local_max_r1, __shfl_xor_sync(0xffffffffu, local_max_r1, 2));

                        float m_prev = m_li1;
                        float m_new = fmaxf(m_prev, local_max_r1);
                        float new_alpha1 = fast_exp::exp<float, fast_exp::Softmax>(m_prev - m_new);

                        float4 exp_r1_st0 = fast_exp::exp4<float, fast_exp::Softmax>(make_float4(
                            scores_reg[0][2] - m_new, scores_reg[0][3] - m_new,
                            scores_reg[0][6] - m_new, scores_reg[0][7] - m_new));
                        float4 exp_r1_st1 = fast_exp::exp4<float, fast_exp::Softmax>(make_float4(
                            scores_reg[1][2] - m_new, scores_reg[1][3] - m_new,
                            scores_reg[1][6] - m_new, scores_reg[1][7] - m_new));
                        p2_st0 = exp_r1_st0.x; p3_st0 = exp_r1_st0.y; p6_st0 = exp_r1_st0.z; p7_st0 = exp_r1_st0.w;
                        p2_st1 = exp_r1_st1.x; p3_st1 = exp_r1_st1.y; p6_st1 = exp_r1_st1.z; p7_st1 = exp_r1_st1.w;

                        float local_sum_r1 = exp_r1_st0.x + exp_r1_st0.y + exp_r1_st0.z + exp_r1_st0.w +
                                             exp_r1_st1.x + exp_r1_st1.y + exp_r1_st1.z + exp_r1_st1.w;
                        local_sum_r1 += __shfl_xor_sync(0xffffffffu, local_sum_r1, 1);
                        local_sum_r1 += __shfl_xor_sync(0xffffffffu, local_sum_r1, 2);

                        alpha1 = row1_active ? new_alpha1 : 1.f;
                        m_li1 = row1_active ? m_new : m_li1;
                        float l_new1 = l_li1 * new_alpha1 + local_sum_r1;
                        l_li1 = row1_active ? fmaxf(l_new1, 1e-30f) : l_li1;
                    }

                    constexpr int V_UNROLL = (HEAD_DIM <= 64) ? 4 : (HEAD_DIM <= 128) ? 2 : 1;

                    #pragma unroll 1
                    for (int head_chunk = 0; head_chunk < NUM_HEAD_CHUNKS; ++head_chunk) {
                        const int vt_start = head_chunk * HEAD_DIM_CHUNK;
                        const int vt_end = vt_start + HEAD_DIM_CHUNK;

                        #pragma unroll V_UNROLL
                        for (int vt = vt_start; vt < vt_end; vt += 16) {
                            float pv0_a = 0, pv0_b = 0, pv0_c = 0, pv0_d = 0;
                            float pv1_a = 0, pv1_b = 0, pv1_c = 0, pv1_d = 0;

                            compute_pv_from_regs_dispatch<T, HEAD_DIM, USE_TC>(
                                pv0_a, pv0_b, pv0_c, pv0_d, pv1_a, pv1_b, pv1_c, pv1_d,
                                p0_st0, p1_st0, p4_st0, p5_st0,
                                p0_st1, p1_st1, p4_st1, p5_st1,
                                p2_st0, p3_st0, p6_st0, p7_st0,
                                p2_st1, p3_st1, p6_st1, p7_st1,
                                &smem_v[cur_stage][0], vt, tile_len, lane);

                            const int vt_slot = (vt - vt_start) >> 4;
                            float* o_acc = o_reg[row_batch][head_chunk][vt_slot];
                            o_acc[0] = fmaf(o_acc[0], alpha0, pv0_a);
                            o_acc[1] = fmaf(o_acc[1], alpha0, pv0_b);
                            o_acc[2] = fmaf(o_acc[2], alpha0, pv0_c);
                            o_acc[3] = fmaf(o_acc[3], alpha0, pv0_d);
                            o_acc[4] = fmaf(o_acc[4], alpha1, pv1_a);
                            o_acc[5] = fmaf(o_acc[5], alpha1, pv1_b);
                            o_acc[6] = fmaf(o_acc[6], alpha1, pv1_c);
                            o_acc[7] = fmaf(o_acc[7], alpha1, pv1_d);
                        }
                    }
                }
            }

            __syncthreads();

            int prefetch_k0 = k0 - NUM_STAGES * TILE_K;
            if (prefetch_k0 >= 0) {
                load_tile(prefetch_k0, cur_stage);
                cp_async_commit<USE_TC>();
            }
            int next_k0 = k0 - TILE_K;
            if (next_k0 >= 0) {
                cp_async_wait<NUM_STAGES - 1, USE_TC>();
                __syncthreads();
            }
            cur_stage = (cur_stage + 1) % NUM_STAGES;
        }

        // Final writeback
        __syncthreads();
        if (active) {
            auto write_row = [&](int r, int row_batch, bool second_row, float l_val) {
                if (row_active(r)) {
                    int t = t_base + r;
                    if (t < total_q) {
                        float inv_l = (l_val > 0.f) ? __fdividef(1.f, l_val) : 0.f;
                        int64_t row_base = ((int64_t)t * (int64_t)n_head + (int64_t)head_idx) * (int64_t)HEAD_DIM;
                        #pragma unroll
                        for (int head_chunk = 0; head_chunk < NUM_HEAD_CHUNKS; ++head_chunk) {
                            const int vt_start = head_chunk * HEAD_DIM_CHUNK;
                            const int vt_end = vt_start + HEAD_DIM_CHUNK;
                            #pragma unroll
                            for (int vt = vt_start; vt < vt_end; vt += 16) {
                                const int vt_slot = (vt - vt_start) >> 4;
                                float* o_acc = o_reg[row_batch][head_chunk][vt_slot];
                                int col01 = vt + tid_in_warp * 2;
                                int col23 = vt + tid_in_warp * 2 + 8;
                                if (!second_row) {
                                    out[row_base + col01]     = from_f32<O>(o_acc[0] * inv_l);
                                    out[row_base + col01 + 1] = from_f32<O>(o_acc[1] * inv_l);
                                    out[row_base + col23]     = from_f32<O>(o_acc[2] * inv_l);
                                    out[row_base + col23 + 1] = from_f32<O>(o_acc[3] * inv_l);
                                } else {
                                    out[row_base + col01]     = from_f32<O>(o_acc[4] * inv_l);
                                    out[row_base + col01 + 1] = from_f32<O>(o_acc[5] * inv_l);
                                    out[row_base + col23]     = from_f32<O>(o_acc[6] * inv_l);
                                    out[row_base + col23 + 1] = from_f32<O>(o_acc[7] * inv_l);
                                }
                            }
                        }
                    }
                }
            };
            write_row(groupID, 0, false, l0);
            write_row(groupID + 8, 0, true, l1);
            write_row(16 + groupID, 1, false, l2);
            write_row(16 + groupID + 8, 1, true, l3);
        }
        __syncthreads();

    } else {
        // ====================================================================
        // HAS_PREFIX PATH
        // ====================================================================
        m0 = m1 = m2 = m3 = NEG_INF_F;
        l0 = l1 = l2 = l3 = 0.f;

        if (active) {
            #pragma unroll
            for (int rb = 0; rb < NUM_ROW_BATCHES; ++rb)
                #pragma unroll
                for (int hc = 0; hc < NUM_HEAD_CHUNKS; ++hc)
                    #pragma unroll
                    for (int vs = 0; vs < VT_ITERS_PER_CHUNK; ++vs)
                        #pragma unroll
                        for (int c = 0; c < 8; ++c) o_reg[rb][hc][vs][c] = 0.f;
        }
        __syncthreads();

        // ---- load_tile_prefix: populate the hoisted palette tables for this
        //      tile, then issue cp.async loads using those tables. ----
        auto load_tile_prefix = [&](int k0_ld, int stage) {
            int k1_ld = k0_ld + TILE_K;
            if (k1_ld > max_k_max) k1_ld = max_k_max;
            int tile_len_ld = k1_ld - k0_ld;
            if (tile_len_ld <= 0) return;

            T* kdst = smem_k[stage];
            T* vdst = smem_v[stage];
            constexpr int BLOCKS_PER_DIM = CHUNK_SIZE / 32;
            constexpr int SUB_HD = HEAD_DIM / N_PALETTE;
            constexpr int T_FORMAT = type_to_arena_format<T>();
            int total_cps = tile_len_ld * TOTAL_CPS_PER_ROW;

            // Clear postprocess flags and reset scales to 1.0
            if (tid == 0) {
                smem_postprocess_any[stage] = 0;
                for (int p = 0; p < N_PALETTE; ++p) {
                    smem_postprocess_fmt_k[stage][p] = 0;
                    smem_postprocess_fmt_v[stage][p] = 0;
                    smem_k_scale[stage][p] = 1.0f;
                    smem_v_scale[stage][p] = 1.0f;
                }
            }

            // Populate the hoisted palette tables for this tile's slices once.
            // Under cum-token addressing the slice covering the tile's start
            // is the one position_map[k0_ld] resolves to. A tile can straddle
            // two slices whose pal_maps differ (any prefix with a partial
            // chunk), so a SECOND table is built from the slice covering the
            // tile's LAST prefix position; the cold path selects per position.
            int blk_for_tables = 0, _in_blk_unused = 0;
            if (k0_ld < prefix_len) {
                resolve_pos(slot_hdr, k0_ld, blk_for_tables, _in_blk_unused);
            }
            bool tile_has_prefix = (k0_ld < prefix_len) && (blk_for_tables < (int)slot_hdr.n_slices);
            int blk_for_tables2 = blk_for_tables;
            if (tile_has_prefix) {
                const uint8_t* slice_pt = get_slice<HEAD_DIM>(slot_hdr.slices_ptr, blk_for_tables, n_kv_head);
                const uint8_t* r_head_pt = get_head<HEAD_DIM>(slice_pt, kv_head_idx);
                const uint8_t* k_pal_map = kvhead_k_pal_map<HEAD_DIM>(r_head_pt);
                const uint8_t* v_pal_map = kvhead_v_pal_map<HEAD_DIM>(r_head_pt);
                for (int d = tid; d < HEAD_DIM; d += (int)blockDim.x) {
                    int p_k, rank_k, p_v, rank_v;
                    prefill_pal_rank(k_pal_map, d, &p_k, &rank_k);
                    prefill_pal_rank(v_pal_map, d, &p_v, &rank_v);
                    s_pal_table_k[stage][d] = (uint8_t)(((p_k & 0x3) << 6) | (rank_k & 0x3F));
                    s_pal_table_v[stage][d] = (uint8_t)(((p_v & 0x3) << 6) | (rank_v & 0x3F));
                }
                // Straddle table: resolve the slice of the tile's last prefix
                // position; when it differs, build the second table from ITS
                // pal_maps (same cooperative build, +one 128-B smem table per
                // side). Non-straddling tiles skip this entirely.
                int last_ld = k0_ld + TILE_K;
                if (last_ld > (int)prefix_len) last_ld = (int)prefix_len;
                last_ld -= 1;
                if (last_ld > k0_ld) {
                    int _ib2 = 0;
                    resolve_pos(slot_hdr, last_ld, blk_for_tables2, _ib2);
                }
                if (blk_for_tables2 != blk_for_tables
                    && blk_for_tables2 < (int)slot_hdr.n_slices) {
                    const uint8_t* slice2 = get_slice<HEAD_DIM>(slot_hdr.slices_ptr, blk_for_tables2, n_kv_head);
                    const uint8_t* head2 = get_head<HEAD_DIM>(slice2, kv_head_idx);
                    const uint8_t* k_map2 = kvhead_k_pal_map<HEAD_DIM>(head2);
                    const uint8_t* v_map2 = kvhead_v_pal_map<HEAD_DIM>(head2);
                    for (int d = tid; d < HEAD_DIM; d += (int)blockDim.x) {
                        int p_k2, rank_k2, p_v2, rank_v2;
                        prefill_pal_rank(k_map2, d, &p_k2, &rank_k2);
                        prefill_pal_rank(v_map2, d, &p_v2, &rank_v2);
                        s_pal_table_k2[stage][d] = (uint8_t)(((p_k2 & 0x3) << 6) | (rank_k2 & 0x3F));
                        s_pal_table_v2[stage][d] = (uint8_t)(((p_v2 & 0x3) << 6) | (rank_v2 & 0x3F));
                    }
                }
            }
            __syncthreads();  // tables visible to all threads

            bool r16_k_handled = false, r16_v_handled = false;
            bool quant_k_handled = false, quant_v_handled = false;

            // ---- R16 cp.async / quant scalar staging fast path ----
            if (tile_len_ld == TILE_K && k0_ld + TILE_K <= prefix_len) {
                int blk, in_blk_base;
                resolve_pos(slot_hdr, k0_ld, blk, in_blk_base);
                if (blk < (int)slot_hdr.n_slices) {
                    const uint8_t* slice = get_slice<HEAD_DIM>(slot_hdr.slices_ptr, blk, n_kv_head);
                    uint32_t bv  = slice_len(slice);
                    uint32_t off = slice_offset(slice);
                    if (in_blk_base + TILE_K <= (int)(off + bv)) {
                        const uint8_t* r_head = get_head<HEAD_DIM>(slice, kv_head_idx);

                        // Per-palette format classification.
                        // We support three fast-path modes per side:
                        //   - all_r16:    every palette is R16  → r16_cp_async_palette
                        //   - all_quant:  every palette is quantized (formats may differ
                        //                  across palettes) → per-palette staged cp.async
                        //   - none:        falls through to the cold path below
                        // Mixed quant+R16 or quant+float in the same side still goes cold.
                        int k_fmt_pal[N_PALETTE];
                        int v_fmt_pal[N_PALETTE];
                        bool all_k_r16 = true, all_k_quant = true;
                        bool all_v_r16 = true, all_v_quant = true;
                        for (int p = 0; p < N_PALETTE; ++p) {
                            int kfmt = kvhead_k_fmt<HEAD_DIM>(r_head, p);
                            int vfmt = kvhead_v_fmt<HEAD_DIM>(r_head, p);
                            k_fmt_pal[p] = kfmt;
                            v_fmt_pal[p] = vfmt;
                            if (kfmt != ArenaFormat::R16)                      all_k_r16   = false;
                            if (ArenaFormat::float_elem_size(kfmt) != 0)       all_k_quant = false;
                            if (vfmt != ArenaFormat::R16)                      all_v_r16   = false;
                            if (ArenaFormat::float_elem_size(vfmt) != 0)       all_v_quant = false;
                        }

                        // Per-palette staging layout for quantized fast path:
                        //   region_off[p]    = byte offset of palette p's slab in smem_k_raw
                        //   block_bytes[p]   = real arena block size for palette p
                        //   padded_stride[p] = block_bytes rounded up to 4 (smem alignment)
                        // A dim with (palette p, rank r) lives at
                        //   smem_k_raw + region_off[p] + r * padded_stride[p].
                        // Total slab = sum_p SUB_HD * padded_stride[p]. For HEAD_DIM=128,
                        // worst case (all Q8_1/Q8_KS = 36B) = 4 * 32 * 36 = 4.5 KB,
                        // which fits in TILE_K * HEAD_DIM * sizeof(T) = 8 KB (f16/TILE_K=32).
                        auto compute_region_off = [](const int* fmt_pal, int* region_off) {
                            int acc = 0;
                            #pragma unroll
                            for (int p = 0; p < N_PALETTE; ++p) {
                                region_off[p] = acc;
                                int bb = get_quant_block_bytes(fmt_pal[p]);
                                int ps = (bb + 3) & ~3;
                                acc += SUB_HD * ps;
                            }
                        };

                        // K fast path
                        if (all_k_r16) {
                            for (int p = 0; p < N_PALETTE; ++p) {
                                if (tid == 0) smem_postprocess_fmt_k[stage][p] = ArenaFormat::R16;
                                r16_cp_async_palette<T, USE_TC, HEAD_DIM, SUB_HD, TILE_K>(
                                    (const char*)kvhead_k_ptr<HEAD_DIM>(r_head, p),
                                    kdst, p * SUB_HD, in_blk_base, tid, (int)blockDim.x);
                            }
                            r16_k_handled = true;
                        } else if (all_k_quant) {
                            int k_blk_within = in_blk_base / 32;
                            int k_block_bytes_pal[N_PALETTE];
                            int k_padded_stride_pal[N_PALETTE];
                            int k_region_off_pal[N_PALETTE];
                            #pragma unroll
                            for (int p = 0; p < N_PALETTE; ++p) {
                                int bb = get_quant_block_bytes(k_fmt_pal[p]);
                                k_block_bytes_pal[p]   = bb;
                                k_padded_stride_pal[p] = (bb + 3) & ~3;
                            }
                            compute_region_off(k_fmt_pal, k_region_off_pal);
                            if (tid == 0) {
                                for (int p = 0; p < N_PALETTE; ++p)
                                    smem_postprocess_fmt_k[stage][p] = (uint8_t)k_fmt_pal[p];
                            }
                            char* smem_k_raw = reinterpret_cast<char*>(smem_k[stage]);
                            // Per-dim, route to the palette's slab. Adjacent dims may use
                            // different formats once palette maps are shuffled.
                            for (int d = tid; d < HEAD_DIM; d += (int)blockDim.x) {
                                uint8_t pe = s_pal_table_k[stage][d];
                                int p       = pal_palette(pe);
                                int local_d = pal_rank(pe);
                                int bb      = k_block_bytes_pal[p];
                                int ps      = k_padded_stride_pal[p];
                                const char* pal = (const char*)kvhead_k_ptr<HEAD_DIM>(r_head, p);
                                const char* src = pal + ((int64_t)local_d * BLOCKS_PER_DIM + k_blk_within) * bb;
                                char* dst = smem_k_raw + k_region_off_pal[p] + local_d * ps;
                                int o = 0;
                                for (; o + 16 <= bb; o += 16)
                                    cp_async_16<USE_TC>(dst + o, src + o);
                                for (; o < bb; ++o) dst[o] = src[o];
                            }
                            quant_k_handled = true;
                        }

                        // V fast path
                        if (all_v_r16) {
                            for (int p = 0; p < N_PALETTE; ++p) {
                                if (tid == 0) smem_postprocess_fmt_v[stage][p] = ArenaFormat::R16;
                                r16_cp_async_palette<T, USE_TC, HEAD_DIM, SUB_HD, TILE_K>(
                                    (const char*)kvhead_v_ptr<HEAD_DIM>(r_head, p),
                                    vdst, p * SUB_HD, in_blk_base, tid, (int)blockDim.x);
                            }
                            r16_v_handled = true;
                        } else if (all_v_quant) {
                            int v_blk_within = in_blk_base / 32;
                            int v_block_bytes_pal[N_PALETTE];
                            int v_padded_stride_pal[N_PALETTE];
                            int v_region_off_pal[N_PALETTE];
                            #pragma unroll
                            for (int p = 0; p < N_PALETTE; ++p) {
                                int bb = get_quant_block_bytes(v_fmt_pal[p]);
                                v_block_bytes_pal[p]   = bb;
                                v_padded_stride_pal[p] = (bb + 3) & ~3;
                            }
                            compute_region_off(v_fmt_pal, v_region_off_pal);
                            if (tid == 0) {
                                for (int p = 0; p < N_PALETTE; ++p)
                                    smem_postprocess_fmt_v[stage][p] = (uint8_t)v_fmt_pal[p];
                            }
                            char* smem_v_raw = reinterpret_cast<char*>(smem_v[stage]);
                            for (int d = tid; d < HEAD_DIM; d += (int)blockDim.x) {
                                uint8_t pe = s_pal_table_v[stage][d];
                                int p       = pal_palette(pe);
                                int local_d = pal_rank(pe);
                                int bb      = v_block_bytes_pal[p];
                                int ps      = v_padded_stride_pal[p];
                                const char* pal = (const char*)kvhead_v_ptr<HEAD_DIM>(r_head, p);
                                const char* src = pal + ((int64_t)local_d * BLOCKS_PER_DIM + v_blk_within) * bb;
                                char* dst = smem_v_raw + v_region_off_pal[p] + local_d * ps;
                                int o = 0;
                                for (; o + 16 <= bb; o += 16)
                                    cp_async_16<USE_TC>(dst + o, src + o);
                                for (; o < bb; ++o) dst[o] = src[o];
                            }
                            quant_v_handled = true;
                        }

                        if (tid == 0) {
                            uint8_t any = 0;
                            if (r16_k_handled)   any |= 0x01;
                            if (r16_v_handled)   any |= 0x02;
                            if (quant_k_handled) any |= 0x04;
                            if (quant_v_handled) any |= 0x08;
                            smem_postprocess_any[stage] = any;
                            for (int p = 0; p < N_PALETTE; ++p) {
                                smem_k_scale[stage][p] = kvhead_k_scale<HEAD_DIM>(r_head, p);
                                smem_v_scale[stage][p] = kvhead_v_scale<HEAD_DIM>(r_head, p);
                            }
                        }

                        if ((r16_k_handled || quant_k_handled) && (r16_v_handled || quant_v_handled)) return;
                    }
                }
            }

            // ---- Cold path: per-element load with palette-table lookup ----
            for (int cp_idx = tid; cp_idx < total_cps; cp_idx += (int)blockDim.x) {
                int i = cp_idx >> CPS_LOG2;
                int cp_in_row = cp_idx & (TOTAL_CPS_PER_ROW - 1);
                int d = cp_in_row * ELEMS_PER_CP_T;
                int k_pos = k0_ld + i;

                T* k_dst = &kdst[i * HEAD_DIM + d];
                T* v_dst = &vdst[i * HEAD_DIM + d];

                if (k_pos < prefix_len) {
                    int blk, in_blk;
                    resolve_pos(slot_hdr, k_pos, blk, in_blk);
                    bool valid = false;
                    if (blk < (int)slot_hdr.n_slices) {
                        const uint8_t* slice = get_slice<HEAD_DIM>(slot_hdr.slices_ptr, blk, n_kv_head);
                        uint32_t bv  = slice_len(slice);
                        uint32_t off = slice_offset(slice);
                        if (in_blk >= (int)off && in_blk < (int)(off + bv)) {
                            valid = true;
                            const uint8_t* r_head = get_head<HEAD_DIM>(slice, kv_head_idx);

                            // K
                            if (!r16_k_handled && !quant_k_handled) {
                                // Cold path must resolve palette/rank per element because
                                // adjacent global dims can land in different palettes once
                                // the destination map is shuffled. Hoist per-palette
                                // metadata once (only N_PALETTE entries) so the per-element
                                // loop only does table lookups indexed by palette id.
                                const char* k_head_r_pal[N_PALETTE];
                                int         k_fmt_pal[N_PALETTE];
                                int         k_elem_size_pal[N_PALETTE];
                                int         k_block_bytes_pal[N_PALETTE];
                                float       k_scale_pal[N_PALETTE];
                                #pragma unroll
                                for (int p = 0; p < N_PALETTE; ++p) {
                                    k_head_r_pal[p]     = (const char*)kvhead_k_ptr<HEAD_DIM>(r_head, p);
                                    int kf              = kvhead_k_fmt<HEAD_DIM>(r_head, p);
                                    k_fmt_pal[p]        = kf;
                                    int es              = ArenaFormat::float_elem_size(kf);
                                    k_elem_size_pal[p]  = es;
                                    k_block_bytes_pal[p] = (es > 0) ? 0 : get_quant_block_bytes(kf);
                                    k_scale_pal[p]      = kvhead_k_scale<HEAD_DIM>(r_head, p);
                                }
                                int blk_within  = in_blk >> 5;   // /32
                                int elem_in_blk = in_blk & 31;   // %32
                                // Routing must come from THIS position's slice: select the
                                // hoisted table matching the resolved slice (first-slice or
                                // straddle table — one smem byte read per element, same cost
                                // as the single-slice case). Only a ≥3-slice tile's middle
                                // positions miss both tables and pay the per-element
                                // prefill_pal_rank fallback.
                                const uint8_t* k_tab = (blk == blk_for_tables)
                                    ? s_pal_table_k[stage]
                                    : (blk == blk_for_tables2) ? s_pal_table_k2[stage]
                                                               : nullptr;
                                const uint8_t* k_pal_map_cur =
                                    kvhead_k_pal_map<HEAD_DIM>(r_head);
                                #pragma unroll
                                for (int j = 0; j < ELEMS_PER_CP_T; ++j) {
                                    int p_k, local_d_k;
                                    if (k_tab != nullptr) {
                                        uint8_t pe_k = k_tab[d + j];
                                        p_k       = pal_palette(pe_k);
                                        local_d_k = pal_rank(pe_k);
                                    } else {
                                        prefill_pal_rank(k_pal_map_cur, d + j, &p_k, &local_d_k);
                                    }
                                    const char* k_head_r = k_head_r_pal[p_k];
                                    int k_fmt       = k_fmt_pal[p_k];
                                    int k_elem_size = k_elem_size_pal[p_k];
                                    float k_scale   = k_scale_pal[p_k];
                                    if (k_elem_size > 0) {
                                        const int64_t k_off = (int64_t)in_blk * (int64_t)SUB_HD + local_d_k;
                                        if (k_fmt == T_FORMAT && k_scale == 1.0f) {
                                            k_dst[j] = reinterpret_cast<const T*>(k_head_r)[k_off];
                                        } else {
                                            k_dst[j] = from_f32<T>(arena_load_element(k_head_r + k_off * k_elem_size, k_fmt, k_scale));
                                        }
                                    } else {
                                        int k_block_bytes = k_block_bytes_pal[p_k];
                                        int64_t bidx = (int64_t)local_d_k * BLOCKS_PER_DIM + blk_within;
                                        k_dst[j] = from_f32<T>(dequant_element_hybrid<float>(k_head_r + bidx * k_block_bytes, elem_in_blk, k_fmt, k_scale));
                                    }
                                }
                            }

                            // V
                            if (!r16_v_handled && !quant_v_handled) {
                                // Same per-element resolution for shuffled/mixed palette V,
                                // with per-palette metadata hoisted out of the unrolled loop.
                                const char* v_head_r_pal[N_PALETTE];
                                int         v_fmt_pal[N_PALETTE];
                                int         v_elem_size_pal[N_PALETTE];
                                int         v_block_bytes_pal[N_PALETTE];
                                float       v_scale_pal[N_PALETTE];
                                #pragma unroll
                                for (int p = 0; p < N_PALETTE; ++p) {
                                    v_head_r_pal[p]     = (const char*)kvhead_v_ptr<HEAD_DIM>(r_head, p);
                                    int vf              = kvhead_v_fmt<HEAD_DIM>(r_head, p);
                                    v_fmt_pal[p]        = vf;
                                    int es              = ArenaFormat::float_elem_size(vf);
                                    v_elem_size_pal[p]  = es;
                                    v_block_bytes_pal[p] = (es > 0) ? 0 : get_quant_block_bytes(vf);
                                    v_scale_pal[p]      = kvhead_v_scale<HEAD_DIM>(r_head, p);
                                }
                                int blk_within_v  = in_blk >> 5;
                                int elem_in_blk_v = in_blk & 31;
                                // Same slice-matched table selection as the K side above.
                                const uint8_t* v_tab = (blk == blk_for_tables)
                                    ? s_pal_table_v[stage]
                                    : (blk == blk_for_tables2) ? s_pal_table_v2[stage]
                                                               : nullptr;
                                const uint8_t* v_pal_map_cur =
                                    kvhead_v_pal_map<HEAD_DIM>(r_head);
                                #pragma unroll
                                for (int j = 0; j < ELEMS_PER_CP_T; ++j) {
                                    int p_v, local_d_v;
                                    if (v_tab != nullptr) {
                                        uint8_t pe_v = v_tab[d + j];
                                        p_v       = pal_palette(pe_v);
                                        local_d_v = pal_rank(pe_v);
                                    } else {
                                        prefill_pal_rank(v_pal_map_cur, d + j, &p_v, &local_d_v);
                                    }
                                    const char* v_head_r = v_head_r_pal[p_v];
                                    int v_fmt       = v_fmt_pal[p_v];
                                    int v_elem_size = v_elem_size_pal[p_v];
                                    float v_scale   = v_scale_pal[p_v];
                                    if (v_elem_size > 0) {
                                        const int64_t v_off = (int64_t)in_blk * (int64_t)SUB_HD + local_d_v;
                                        if (v_fmt == T_FORMAT && v_scale == 1.0f) {
                                            v_dst[j] = reinterpret_cast<const T*>(v_head_r)[v_off];
                                        } else {
                                            v_dst[j] = from_f32<T>(arena_load_element(v_head_r + v_off * v_elem_size, v_fmt, v_scale));
                                        }
                                    } else {
                                        int v_block_bytes = v_block_bytes_pal[p_v];
                                        int64_t bidx = (int64_t)local_d_v * BLOCKS_PER_DIM + blk_within_v;
                                        v_dst[j] = from_f32<T>(dequant_element_hybrid<float>(v_head_r + bidx * v_block_bytes, elem_in_blk_v, v_fmt, v_scale));
                                    }
                                }
                            }
                        }
                    }
                    if (!valid) {
                        #pragma unroll
                        for (int j = 0; j < ELEMS_PER_CP_T; ++j) {
                            if (!r16_k_handled && !quant_k_handled) k_dst[j] = zero_val<T>();
                            if (!r16_v_handled && !quant_v_handled) v_dst[j] = zero_val<T>();
                        }
                    }
                } else {
                    // Current token: cp.async from k_packed/v_packed
                    int tok = q_start + (k_pos - prefix_len);
                    int64_t base = ((int64_t)tok * (int64_t)n_kv_head + (int64_t)kv_head_idx) * (int64_t)HEAD_DIM;
                    load_kv_chunk<T, USE_TC>(k_dst, v_dst, k_packed, v_packed, base + d);
                }
            }
        };

        // ---- Postprocess: dequant + R16 transpose using hoisted tables ----
        auto postprocess_quant_tile_prefix = [&](int stage, int worker_tid, int worker_threads, bool warp_scope_sync) {
            uint8_t any = smem_postprocess_any[stage];
            bool any_k_quant = (any & 0x04) != 0;
            bool any_v_quant = (any & 0x08) != 0;
            if (!any_k_quant && !any_v_quant) return;

            auto worker_sync = [&]() {
                if (warp_scope_sync) __syncwarp();
                else __syncthreads();
            };

            constexpr int MAX_DS_PER_THREAD = (HEAD_DIM + 31) / 32;
            constexpr int POST_SUB_HD = HEAD_DIM / N_PALETTE;

            // Helper: derive per-palette region offsets from the format table.
            // Mirrors the staging layout written by the fast quant load path.
            auto compute_region_off = [](const uint8_t* fmt_pal_smem, int* region_off, int* padded_stride) {
                int acc = 0;
                #pragma unroll
                for (int p = 0; p < N_PALETTE; ++p) {
                    region_off[p] = acc;
                    int bb = get_quant_block_bytes((int)fmt_pal_smem[p]);
                    int ps = (bb + 3) & ~3;
                    padded_stride[p] = ps;
                    acc += POST_SUB_HD * ps;
                }
            };

            if (any_k_quant) {
                int k_region_off[N_PALETTE];
                int k_padded_stride[N_PALETTE];
                compute_region_off(smem_postprocess_fmt_k[stage], k_region_off, k_padded_stride);
                const char* smem_k_raw = reinterpret_cast<const char*>(smem_k[stage]);
                T* k_out = smem_k[stage];
                alignas(4) char k_blk_reg[MAX_DS_PER_THREAD][36];
                int k_fmt_slot[MAX_DS_PER_THREAD];

                int slot = 0;
                for (int d = worker_tid; d < HEAD_DIM; d += worker_threads) {
                    uint8_t pe = s_pal_table_k[stage][d];
                    int p       = pal_palette(pe);
                    int local_d = pal_rank(pe);
                    int kf      = (int)smem_postprocess_fmt_k[stage][p];
                    int bb      = get_quant_block_bytes(kf);
                    int ps      = k_padded_stride[p];
                    const char* staged = smem_k_raw + k_region_off[p] + local_d * ps;
                    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(staged);
                    uint32_t* dst32 = reinterpret_cast<uint32_t*>(k_blk_reg[slot]);
                    const int words = bb >> 2;
                    const int tail = bb & 3;
                    #pragma unroll
                    for (int w = 0; w < 9; ++w) if (w < words) dst32[w] = src32[w];
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) if (b < tail) k_blk_reg[slot][words * 4 + b] = staged[words * 4 + b];
                    k_fmt_slot[slot] = kf;
                    ++slot;
                }
                worker_sync();
                slot = 0;
                for (int d = worker_tid; d < HEAD_DIM; d += worker_threads) {
                    int kf = k_fmt_slot[slot];
                    int p_ks = pal_palette(s_pal_table_k[stage][d]);
                    float ks = smem_k_scale[stage][p_ks];
                    #pragma unroll
                    for (int pos = 0; pos < TILE_K; ++pos)
                        k_out[pos * HEAD_DIM + d] = from_f32<T>(dequant_element_hybrid<float>(k_blk_reg[slot], pos, kf, ks));
                    ++slot;
                }
                worker_sync();
            }

            if (any_v_quant) {
                int v_region_off[N_PALETTE];
                int v_padded_stride[N_PALETTE];
                compute_region_off(smem_postprocess_fmt_v[stage], v_region_off, v_padded_stride);
                const char* smem_v_raw = reinterpret_cast<const char*>(smem_v[stage]);
                T* v_out = smem_v[stage];
                alignas(4) char v_blk_reg[MAX_DS_PER_THREAD][36];
                int v_fmt_slot[MAX_DS_PER_THREAD];

                int slot = 0;
                for (int d = worker_tid; d < HEAD_DIM; d += worker_threads) {
                    uint8_t pe = s_pal_table_v[stage][d];
                    int p       = pal_palette(pe);
                    int local_d = pal_rank(pe);
                    int vf      = (int)smem_postprocess_fmt_v[stage][p];
                    int bb      = get_quant_block_bytes(vf);
                    int ps      = v_padded_stride[p];
                    const char* staged = smem_v_raw + v_region_off[p] + local_d * ps;
                    const uint32_t* src32 = reinterpret_cast<const uint32_t*>(staged);
                    uint32_t* dst32 = reinterpret_cast<uint32_t*>(v_blk_reg[slot]);
                    const int words = bb >> 2;
                    const int tail = bb & 3;
                    #pragma unroll
                    for (int w = 0; w < 9; ++w) if (w < words) dst32[w] = src32[w];
                    #pragma unroll
                    for (int b = 0; b < 4; ++b) if (b < tail) v_blk_reg[slot][words * 4 + b] = staged[words * 4 + b];
                    v_fmt_slot[slot] = vf;
                    ++slot;
                }
                worker_sync();
                slot = 0;
                for (int d = worker_tid; d < HEAD_DIM; d += worker_threads) {
                    int vf = v_fmt_slot[slot];
                    int p_vs = pal_palette(s_pal_table_v[stage][d]);
                    float vs = smem_v_scale[stage][p_vs];
                    #pragma unroll
                    for (int pos = 0; pos < TILE_K; ++pos)
                        v_out[pos * HEAD_DIM + d] = from_f32<T>(dequant_element_hybrid<float>(v_blk_reg[slot], pos, vf, vs));
                    ++slot;
                }
                worker_sync();
            }
        };

        // R16 transpose consumes the hoisted s_pal_table_{k,v}[stage] directly.
        auto postprocess_r16_tile_prefix = [&](int stage) {
            uint8_t any = smem_postprocess_any[stage];
            bool any_k_r16 = (any & 0x01) != 0;
            bool any_v_r16 = (any & 0x02) != 0;
            if (any_k_r16 || any_v_r16) {
                if (any_k_r16) transpose_smem_dim_pos_ranked_pal_nosync<T, HEAD_DIM, TILE_K, PREFILL_BLOCK_DIM>(
                    smem_k[stage], s_pal_table_k[stage], tid, (int)blockDim.x, smem_k_scale[stage]);
                if (any_v_r16) transpose_smem_dim_pos_ranked_pal_nosync<T, HEAD_DIM, TILE_K, PREFILL_BLOCK_DIM>(
                    smem_v[stage], s_pal_table_v[stage], tid, (int)blockDim.x, smem_v_scale[stage]);
                __syncthreads();
            }
        };

        auto postprocess_tile_prefix = [&](int stage) {
            postprocess_quant_tile_prefix(stage, tid, (int)blockDim.x, false);
            postprocess_r16_tile_prefix(stage);
        };

        const int n_tiles_prefix = (max_k_max + TILE_K - 1) / TILE_K;
        const int last_tile_k0_prefix = (n_tiles_prefix - 1) * TILE_K;
        const bool last_tile_partial_prefix = (max_k_max % TILE_K) != 0;
        const int last_tile_len_prefix = last_tile_partial_prefix ? (max_k_max % TILE_K) : TILE_K;

        load_tile_prefix(last_tile_k0_prefix, 0);
        cp_async_commit<USE_TC>();

        int tiles_loaded = 1;
        if (last_tile_k0_prefix - TILE_K >= 0) {
            load_tile_prefix(last_tile_k0_prefix - TILE_K, 1);
            cp_async_commit<USE_TC>();
            tiles_loaded = 2;
        }
        if (tiles_loaded >= NUM_STAGES) cp_async_wait<NUM_STAGES - 1, USE_TC>();
        else                            cp_async_wait<0, USE_TC>();
        __syncthreads();
        postprocess_tile_prefix(0);

        int cur_stage = 0;

        for (int k0 = last_tile_k0_prefix; k0 >= 0; k0 -= TILE_K) {
            const int tile_len = (k0 == last_tile_k0_prefix) ? last_tile_len_prefix : TILE_K;
            const int k1 = k0 + tile_len;

            // ---- Build validity bitmask + rope positions for current tile ----
            // Warp 0 handles all tile_len ≤ 32 positions (one lane per position)
            // and writes the validity mask in a single store via __ballot_sync,
            // avoiding shared-memory atomic contention on a single word.
            if (warp_id == 0) {
                const uint32_t rope_base_v = rope_offsets[batch_idx];
                int i = lane;
                bool is_valid = false;
                int rope_p = 0;
                if (i < tile_len) {
                    int k_pos = k0 + i;
                    if (k_pos < prefix_len) {
                        int blk, in_blk;
                        resolve_pos(slot_hdr, k_pos, blk, in_blk);
                        if (blk < (int)slot_hdr.n_slices) {
                            const uint8_t* slice = get_slice<HEAD_DIM>(slot_hdr.slices_ptr, blk, n_kv_head);
                            uint32_t bv  = slice_len(slice);
                            uint32_t off = slice_offset(slice);
                            if (in_blk >= (int)off && in_blk < (int)(off + bv)) {
                                is_valid = true;
                                rope_p = (int)slice_rope(slice) + (in_blk - (int)off);
                            }
                        }
                    } else {
                        is_valid = true;
                        rope_p = k_pos + (int)rope_base_v;
                    }
                }
                uint32_t bits = __ballot_sync(0xffffffffu, is_valid);
                if (lane == 0) smem_valid_mask = bits;
                if (is_valid) smem_rope_pos[i] = rope_p;
            }
            __syncthreads();

            // Local accessor for per-position validity
            auto pos_valid = [&](int i) -> bool { return (smem_valid_mask >> i) & 1u; };

            // ---- KV writeback (un-rotated). New tokens only. Identity palette. ----
            if (head_block == 0) {
                int first_q_head2 = kv_head_idx * num_groups;
                for (int r = 0; r < BLOCK_M; ++r) {
                    if (!row_active(r) || !write_active(r)) continue;
                    int abs_pos = prefix_len + (int)s_q_pos[r];
                    if (abs_pos >= k0 && abs_pos < k1) {
                        int i_write = abs_pos - k0;
                        const uint8_t* w_slice2 = get_slice<HEAD_DIM>(slot_hdr.slices_ptr, (int)s_write_slice_idx[r], n_kv_head);
                        const uint8_t* w_head2 = get_head<HEAD_DIM>(w_slice2, kv_head_idx);
                        int src_base = i_write * HEAD_DIM;
                        const Q_T* q_row2 = q + ((int64_t)(q_start + (int)s_q_pos[r]) * (int64_t)n_head + (int64_t)first_q_head2) * (int64_t)HEAD_DIM;
                        constexpr int CHUNK = 8;
                        constexpr int SUB_HD = HEAD_DIM / N_PALETTE;
                        constexpr int VEC_LIMIT = (HEAD_DIM / CHUNK) * CHUNK;
                        for (int d = tid * CHUNK; d < VEC_LIMIT; d += (int)blockDim.x * CHUNK) {
                            int p = d / SUB_HD;
                            int local_d = d - p * SUB_HD;
                            store_kv_chunk_arena<T, Q_T, SUB_HD>(
                                (char*)kvhead_k_ptr<HEAD_DIM>(w_head2, p),
                                (char*)kvhead_v_ptr<HEAD_DIM>(w_head2, p),
                                &smem_k[cur_stage][src_base + d],
                                &smem_v[cur_stage][src_base + d],
                                &q_row2[d],
                                kvhead_k_fmt<HEAD_DIM>(w_head2, p),
                                kvhead_v_fmt<HEAD_DIM>(w_head2, p),
                                0, 0, (int)s_write_in_blk[r],
                                local_d, 0, 0);
                        }
                        if constexpr (HEAD_DIM % CHUNK != 0) {
                            for (int d = VEC_LIMIT + tid; d < HEAD_DIM; d += (int)blockDim.x) {
                                int p = d / SUB_HD;
                                int local_d = d - p * SUB_HD;
                                store_kv_chunk_arena<T, Q_T, SUB_HD>(
                                    (char*)kvhead_k_ptr<HEAD_DIM>(w_head2, p),
                                    (char*)kvhead_v_ptr<HEAD_DIM>(w_head2, p),
                                    &smem_k[cur_stage][src_base + d],
                                    &smem_v[cur_stage][src_base + d],
                                    &q_row2[d],
                                    kvhead_k_fmt<HEAD_DIM>(w_head2, p),
                                    kvhead_v_fmt<HEAD_DIM>(w_head2, p),
                                    0, 0, (int)s_write_in_blk[r],
                                    local_d, 0, 0);
                            }
                        }
                    }
                }
            }
            __syncthreads();

            // ---- Fused RoPE on K (precomputed positions in smem_rope_pos) ----
            {
                constexpr int HALF_DIM = HEAD_DIM / 2;
                const int flat_total = tile_len * HALF_DIM;
                if (!rope_interleaved) {
                    for (int flat = tid; flat < flat_total; flat += (int)blockDim.x) {
                        const int d = flat & (HALF_DIM - 1);
                        const int tok = flat >> ((HALF_DIM == 16) ? 4 : (HALF_DIM == 32) ? 5 :
                                                 (HALF_DIM == 64) ? 6 : 7);
                        const int rope_pos = smem_rope_pos[tok];
                        T* row = &smem_k[cur_stage][tok * HEAD_DIM];
                        float x = to_f32(row[d]);
                        float y = to_f32(row[d + HALF_DIM]);
                        float cos_v, sin_v;
                        rope_cos_sin<HEAD_DIM>(rope_pos, d, rope_cs, cos_v, sin_v);
                        row[d]            = from_f32<T>(x * cos_v - y * sin_v);
                        row[d + HALF_DIM] = from_f32<T>(x * sin_v + y * cos_v);
                    }
                } else {
                    for (int flat = tid; flat < flat_total; flat += (int)blockDim.x) {
                        const int d = flat & (HALF_DIM - 1);
                        const int tok = flat >> ((HALF_DIM == 16) ? 4 : (HALF_DIM == 32) ? 5 :
                                                 (HALF_DIM == 64) ? 6 : 7);
                        const int rope_pos = smem_rope_pos[tok];
                        T* row = &smem_k[cur_stage][tok * HEAD_DIM];
                        float x = to_f32(row[d * 2]);
                        float y = to_f32(row[d * 2 + 1]);
                        float cos_v, sin_v;
                        rope_cos_sin<HEAD_DIM>(rope_pos, d, rope_cs, cos_v, sin_v);
                        row[d * 2]     = from_f32<T>(x * cos_v - y * sin_v);
                        row[d * 2 + 1] = from_f32<T>(x * sin_v + y * cos_v);
                    }
                }
                __syncthreads();
            }

            int next_k0 = k0 - TILE_K;
            int next_stage = (cur_stage + 1) % NUM_STAGES;
            bool overlap_next_quant = false;
            if (next_k0 >= 0) {
                cp_async_wait<NUM_STAGES - 1, USE_TC>();
                __syncthreads();
                overlap_next_quant = has_post_helper && ((smem_postprocess_any[next_stage] & 0x0c) != 0);
                if (overlap_next_quant) {
                    if (helper_warp) postprocess_quant_tile_prefix(next_stage, lane, 32, true);
                } else {
                    postprocess_tile_prefix(next_stage);
                }
            }

            // ---- Row-batch compute (uses pos_valid()) ----
            #pragma unroll
            for (int row_batch = 0; row_batch < NUM_ROW_BATCHES; ++row_batch) {
                const int row_start = row_batch * ROWS_PER_BATCH;
                float& m_li0 = (row_batch == 0) ? m0 : m2;
                float& m_li1 = (row_batch == 0) ? m1 : m3;
                float& l_li0 = (row_batch == 0) ? l0 : l2;
                float& l_li1 = (row_batch == 0) ? l1 : l3;

                float scores_reg[2][8];

                if (active) {
                    for (int kt = 0; kt < TILE_K; kt += 16) {
                        const int r0 = row_start + groupID;
                        const int r1 = row_start + groupID + 8;
                        const int col_a0 = kt + tid_in_warp * 2;
                        const int col_a1 = kt + tid_in_warp * 2 + 1;
                        const int col_b0 = kt + tid_in_warp * 2 + 8;
                        const int col_b1 = kt + tid_in_warp * 2 + 9;
                        const int sub_tile = kt >> 4;

                        float s0_a, s1_a, s2_a, s3_a, s0_b, s1_b, s2_b, s3_b;
                        compute_qk_16x16_dispatch<Q_T, T, HEAD_DIM, USE_TC>(
                            s0_a, s1_a, s2_a, s3_a, s0_b, s1_b, s2_b, s3_b,
                            warp_smem_q + row_start * HEAD_DIM,
                            &smem_k[cur_stage][kt * HEAD_DIM], lane);

                        s0_a *= softmax_scale; s1_a *= softmax_scale;
                        s2_a *= softmax_scale; s3_a *= softmax_scale;
                        s0_b *= softmax_scale; s1_b *= softmax_scale;
                        s2_b *= softmax_scale; s3_b *= softmax_scale;

                        int k_pos_a0 = k0 + col_a0, k_pos_a1 = k0 + col_a1;
                        int k_pos_b0 = k0 + col_b0, k_pos_b1 = k0 + col_b1;

                        bool valid_a0 = (col_a0 < tile_len) && (k_pos_a0 >= prefix_len || pos_valid(col_a0));
                        bool valid_a1 = (col_a1 < tile_len) && (k_pos_a1 >= prefix_len || pos_valid(col_a1));
                        bool valid_b0 = (col_b0 < tile_len) && (k_pos_b0 >= prefix_len || pos_valid(col_b0));
                        bool valid_b1 = (col_b1 < tile_len) && (k_pos_b1 >= prefix_len || pos_valid(col_b1));

                        bool r0_valid = (r0 < BLOCK_M) & row_active(r0);
                        bool r1_valid = (r1 < BLOCK_M) & row_active(r1);
                        int max_k_r0 = r0_valid ? row_max_k(r0) : 0;
                        int max_k_r1 = r1_valid ? row_max_k(r1) : 0;

                        s0_a = (r0_valid & (k_pos_a0 < max_k_r0) & valid_a0) ? s0_a : NEG_INF_F;
                        s1_a = (r0_valid & (k_pos_a1 < max_k_r0) & valid_a1) ? s1_a : NEG_INF_F;
                        s0_b = (r0_valid & (k_pos_b0 < max_k_r0) & valid_b0) ? s0_b : NEG_INF_F;
                        s1_b = (r0_valid & (k_pos_b1 < max_k_r0) & valid_b1) ? s1_b : NEG_INF_F;
                        s2_a = (r1_valid & (k_pos_a0 < max_k_r1) & valid_a0) ? s2_a : NEG_INF_F;
                        s3_a = (r1_valid & (k_pos_a1 < max_k_r1) & valid_a1) ? s3_a : NEG_INF_F;
                        s2_b = (r1_valid & (k_pos_b0 < max_k_r1) & valid_b0) ? s2_b : NEG_INF_F;
                        s3_b = (r1_valid & (k_pos_b1 < max_k_r1) & valid_b1) ? s3_b : NEG_INF_F;

                        scores_reg[sub_tile][0] = s0_a; scores_reg[sub_tile][1] = s1_a;
                        scores_reg[sub_tile][2] = s2_a; scores_reg[sub_tile][3] = s3_a;
                        scores_reg[sub_tile][4] = s0_b; scores_reg[sub_tile][5] = s1_b;
                        scores_reg[sub_tile][6] = s2_b; scores_reg[sub_tile][7] = s3_b;
                    }
                }

                float alpha0 = 1.f, alpha1 = 1.f;
                float p0_st0 = 0, p1_st0 = 0, p4_st0 = 0, p5_st0 = 0;
                float p0_st1 = 0, p1_st1 = 0, p4_st1 = 0, p5_st1 = 0;
                float p2_st0 = 0, p3_st0 = 0, p6_st0 = 0, p7_st0 = 0;
                float p2_st1 = 0, p3_st1 = 0, p6_st1 = 0, p7_st1 = 0;

                if (active) {
                    const int r0 = row_start + groupID;
                    const int r1 = row_start + groupID + 8;
                    const bool row0_active = row_active(r0);
                    const bool row1_active = row_active(r1);

                    {
                        float local_max_r0 = fmaxf(fmaxf(scores_reg[0][0], scores_reg[0][1]),
                                                   fmaxf(scores_reg[0][4], scores_reg[0][5]));
                        local_max_r0 = fmaxf(local_max_r0,
                                             fmaxf(fmaxf(scores_reg[1][0], scores_reg[1][1]),
                                                   fmaxf(scores_reg[1][4], scores_reg[1][5])));
                        local_max_r0 = fmaxf(local_max_r0, __shfl_xor_sync(0xffffffffu, local_max_r0, 1));
                        local_max_r0 = fmaxf(local_max_r0, __shfl_xor_sync(0xffffffffu, local_max_r0, 2));

                        float m_prev = m_li0;
                        float m_new = fmaxf(m_prev, local_max_r0);
                        float new_alpha0 = fast_exp::exp<float, fast_exp::Softmax>(m_prev - m_new);

                        float4 exp_st0 = fast_exp::exp4<float, fast_exp::Softmax>(make_float4(
                            scores_reg[0][0] - m_new, scores_reg[0][1] - m_new,
                            scores_reg[0][4] - m_new, scores_reg[0][5] - m_new));
                        float4 exp_st1 = fast_exp::exp4<float, fast_exp::Softmax>(make_float4(
                            scores_reg[1][0] - m_new, scores_reg[1][1] - m_new,
                            scores_reg[1][4] - m_new, scores_reg[1][5] - m_new));
                        p0_st0 = exp_st0.x; p1_st0 = exp_st0.y; p4_st0 = exp_st0.z; p5_st0 = exp_st0.w;
                        p0_st1 = exp_st1.x; p1_st1 = exp_st1.y; p4_st1 = exp_st1.z; p5_st1 = exp_st1.w;

                        float local_sum_r0 = exp_st0.x + exp_st0.y + exp_st0.z + exp_st0.w +
                                             exp_st1.x + exp_st1.y + exp_st1.z + exp_st1.w;
                        local_sum_r0 += __shfl_xor_sync(0xffffffffu, local_sum_r0, 1);
                        local_sum_r0 += __shfl_xor_sync(0xffffffffu, local_sum_r0, 2);

                        alpha0 = row0_active ? new_alpha0 : 1.f;
                        m_li0 = row0_active ? m_new : m_li0;
                        float l_new0 = l_li0 * new_alpha0 + local_sum_r0;
                        l_li0 = row0_active ? fmaxf(l_new0, 1e-30f) : l_li0;
                    }

                    {
                        float local_max_r1 = fmaxf(fmaxf(scores_reg[0][2], scores_reg[0][3]),
                                                   fmaxf(scores_reg[0][6], scores_reg[0][7]));
                        local_max_r1 = fmaxf(local_max_r1,
                                             fmaxf(fmaxf(scores_reg[1][2], scores_reg[1][3]),
                                                   fmaxf(scores_reg[1][6], scores_reg[1][7])));
                        local_max_r1 = fmaxf(local_max_r1, __shfl_xor_sync(0xffffffffu, local_max_r1, 1));
                        local_max_r1 = fmaxf(local_max_r1, __shfl_xor_sync(0xffffffffu, local_max_r1, 2));

                        float m_prev = m_li1;
                        float m_new = fmaxf(m_prev, local_max_r1);
                        float new_alpha1 = fast_exp::exp<float, fast_exp::Softmax>(m_prev - m_new);

                        float4 exp_r1_st0 = fast_exp::exp4<float, fast_exp::Softmax>(make_float4(
                            scores_reg[0][2] - m_new, scores_reg[0][3] - m_new,
                            scores_reg[0][6] - m_new, scores_reg[0][7] - m_new));
                        float4 exp_r1_st1 = fast_exp::exp4<float, fast_exp::Softmax>(make_float4(
                            scores_reg[1][2] - m_new, scores_reg[1][3] - m_new,
                            scores_reg[1][6] - m_new, scores_reg[1][7] - m_new));
                        p2_st0 = exp_r1_st0.x; p3_st0 = exp_r1_st0.y; p6_st0 = exp_r1_st0.z; p7_st0 = exp_r1_st0.w;
                        p2_st1 = exp_r1_st1.x; p3_st1 = exp_r1_st1.y; p6_st1 = exp_r1_st1.z; p7_st1 = exp_r1_st1.w;

                        float local_sum_r1 = exp_r1_st0.x + exp_r1_st0.y + exp_r1_st0.z + exp_r1_st0.w +
                                             exp_r1_st1.x + exp_r1_st1.y + exp_r1_st1.z + exp_r1_st1.w;
                        local_sum_r1 += __shfl_xor_sync(0xffffffffu, local_sum_r1, 1);
                        local_sum_r1 += __shfl_xor_sync(0xffffffffu, local_sum_r1, 2);

                        alpha1 = row1_active ? new_alpha1 : 1.f;
                        m_li1 = row1_active ? m_new : m_li1;
                        float l_new1 = l_li1 * new_alpha1 + local_sum_r1;
                        l_li1 = row1_active ? fmaxf(l_new1, 1e-30f) : l_li1;
                    }

                    constexpr int V_UNROLL = (HEAD_DIM <= 64) ? 4 : (HEAD_DIM <= 128) ? 2 : 1;

                    #pragma unroll 1
                    for (int head_chunk = 0; head_chunk < NUM_HEAD_CHUNKS; ++head_chunk) {
                        const int vt_start = head_chunk * HEAD_DIM_CHUNK;
                        const int vt_end = vt_start + HEAD_DIM_CHUNK;

                        #pragma unroll V_UNROLL
                        for (int vt = vt_start; vt < vt_end; vt += 16) {
                            float pv0_a = 0, pv0_b = 0, pv0_c = 0, pv0_d = 0;
                            float pv1_a = 0, pv1_b = 0, pv1_c = 0, pv1_d = 0;

                            compute_pv_from_regs_dispatch<T, HEAD_DIM, USE_TC>(
                                pv0_a, pv0_b, pv0_c, pv0_d, pv1_a, pv1_b, pv1_c, pv1_d,
                                p0_st0, p1_st0, p4_st0, p5_st0,
                                p0_st1, p1_st1, p4_st1, p5_st1,
                                p2_st0, p3_st0, p6_st0, p7_st0,
                                p2_st1, p3_st1, p6_st1, p7_st1,
                                &smem_v[cur_stage][0], vt, tile_len, lane);

                            const int vt_slot = (vt - vt_start) >> 4;
                            float* o_acc = o_reg[row_batch][head_chunk][vt_slot];
                            o_acc[0] = fmaf(o_acc[0], alpha0, pv0_a);
                            o_acc[1] = fmaf(o_acc[1], alpha0, pv0_b);
                            o_acc[2] = fmaf(o_acc[2], alpha0, pv0_c);
                            o_acc[3] = fmaf(o_acc[3], alpha0, pv0_d);
                            o_acc[4] = fmaf(o_acc[4], alpha1, pv1_a);
                            o_acc[5] = fmaf(o_acc[5], alpha1, pv1_b);
                            o_acc[6] = fmaf(o_acc[6], alpha1, pv1_c);
                            o_acc[7] = fmaf(o_acc[7], alpha1, pv1_d);
                        }
                    }
                }
            }
            __syncthreads();

            if (next_k0 >= 0 && overlap_next_quant) {
                postprocess_r16_tile_prefix(next_stage);
            }

            int prefetch_k0 = k0 - NUM_STAGES * TILE_K;
            if (prefetch_k0 >= 0) {
                load_tile_prefix(prefetch_k0, cur_stage);
                cp_async_commit<USE_TC>();
            }

            cur_stage = (cur_stage + 1) % NUM_STAGES;
        }

        // Final writeback
        __syncthreads();
        if (active) {
            auto write_row = [&](int r, int row_batch, bool second_row, float l_val) {
                if (row_active(r)) {
                    int t = t_base + r;
                    if (t < total_q) {
                        float inv_l = (l_val > 0.f) ? __fdividef(1.f, l_val) : 0.f;
                        int64_t row_base = ((int64_t)t * (int64_t)n_head + (int64_t)head_idx) * (int64_t)HEAD_DIM;
                        #pragma unroll
                        for (int head_chunk = 0; head_chunk < NUM_HEAD_CHUNKS; ++head_chunk) {
                            const int vt_start = head_chunk * HEAD_DIM_CHUNK;
                            const int vt_end = vt_start + HEAD_DIM_CHUNK;
                            #pragma unroll
                            for (int vt = vt_start; vt < vt_end; vt += 16) {
                                const int vt_slot = (vt - vt_start) >> 4;
                                float* o_acc = o_reg[row_batch][head_chunk][vt_slot];
                                int col01 = vt + tid_in_warp * 2;
                                int col23 = vt + tid_in_warp * 2 + 8;
                                if (!second_row) {
                                    out[row_base + col01]     = from_f32<O>(o_acc[0] * inv_l);
                                    out[row_base + col01 + 1] = from_f32<O>(o_acc[1] * inv_l);
                                    out[row_base + col23]     = from_f32<O>(o_acc[2] * inv_l);
                                    out[row_base + col23 + 1] = from_f32<O>(o_acc[3] * inv_l);
                                } else {
                                    out[row_base + col01]     = from_f32<O>(o_acc[4] * inv_l);
                                    out[row_base + col01 + 1] = from_f32<O>(o_acc[5] * inv_l);
                                    out[row_base + col23]     = from_f32<O>(o_acc[6] * inv_l);
                                    out[row_base + col23 + 1] = from_f32<O>(o_acc[7] * inv_l);
                                }
                            }
                        }
                    }
                }
            };
            write_row(groupID, 0, false, l0);
            write_row(groupID + 8, 0, true, l1);
            write_row(16 + groupID, 1, false, l2);
            write_row(16 + groupID + 8, 1, true, l3);
        }
        __syncthreads();
    }
}

// ============================================================================
// LAUNCH WRAPPER
// ============================================================================

/// Launch paged prefill kernel. Auto-selects WARPS_TC, USE_TC, and buffer depth.
/// Q_T: Query type (for mixed precision, e.g. BF16 Q with FP8 KV)
/// T: KV type
/// O: Output type
template <typename Q_T, typename T, typename O, int HEAD_DIM, int WARPS_PER_BLOCK, int TILE_K>
inline void launch_paged_prefill_chunks(
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    const uint8_t* headers_ptr,          // SlotHeader[batch_size] — per-slot metadata (slot_types.cuh)
    const uint32_t* cu_seqlens_q,
    const uint32_t* q_lens,
    const uint32_t* kv_lens,
    void* o_ptr,
    int32_t total_q,
    int32_t batch_size,
    int32_t n_head,
    int32_t n_kv_head,
    int32_t max_blocks,
    float softmax_scale,
    bool has_prefix,
    // RoPE position offsets per batch element [batch_size].
    // Always required: token r in batch b gets pos = prefix_len + r + rope_offsets[b].
    // Pass zeros([batch_size]) for natural positions 0..seq_len-1. V is never rotated.
    const uint32_t* rope_offsets,
    const float* rope_cs,  // Precomputed cos/sin table [max_pos * HEAD_DIM]
    int rope_interleaved,    // 0=non-interleaved half-split (Qwen/GPT2), 1=interleaved adjacent-pairs (Llama)
    const uint32_t* write_offset_shifts = nullptr, // Per-batch write position shift [batch_size], nullable
    cudaStream_t stream = nullptr
) {
    int num_groups = n_head / n_kv_head;
    if (num_groups <= 0) num_groups = 1;
    constexpr int BLOCK_M = 32;

    // Keep WARPS_TC conservative as HEAD_DIM grows to bound register pressure.
    constexpr int MAX_WARPS_FOR_SMEM = (HEAD_DIM >= 256) ? 1 :
                                       (HEAD_DIM >= 128) ? 2 : 4;
    constexpr int WARPS_TC_COMPUTED = (WARPS_PER_BLOCK > MAX_WARPS_FOR_SMEM) ?
                                       MAX_WARPS_FOR_SMEM : WARPS_PER_BLOCK;

    int head_blocks_per_kv_tc = (num_groups + WARPS_TC_COMPUTED - 1) / WARPS_TC_COMPUTED;

    // Grid configuration
    uint32_t q_upper = (uint32_t)(max_blocks * CHUNK_SIZE);
    uint32_t grid_x = (uint32_t)((q_upper + (uint32_t)BLOCK_M - 1u) / (uint32_t)BLOCK_M);
    if (grid_x == 0) grid_x = 1;
    dim3 grid(grid_x, (uint32_t)(n_kv_head * head_blocks_per_kv_tc), (uint32_t)batch_size);
    int prefix_block_warps = has_prefix ? ((WARPS_TC_COMPUTED < WARPS_PER_BLOCK) ? (WARPS_TC_COMPUTED + 1) : WARPS_TC_COMPUTED)
                                       : WARPS_TC_COMPUTED;
    dim3 block_tc((uint32_t)(prefix_block_warps * 32), 1, 1);

    // Detect SM80+ for tensor cores (via shared device caps cache)
    constexpr bool HEAD_DIM_OK_FOR_TC = (HEAD_DIM % 32 == 0);
    const auto& caps = get_device_caps();
    bool sm80_or_later = (caps.sm_version >= 800);
    bool use_tc = HEAD_DIM_OK_FOR_TC && sm80_or_later;
    
    // One-time per-instantiation: query smem requirements and configure extended smem.
    // Although prefill is not called every decode step, caching avoids redundancy on
    // multi-prompt batches and keeps the pattern consistent with the decode launcher.
    // Dynamic-smem byte counts for the K/V/Q rings (declared extern in the
    // kernel). Depends only on the stage count — not on HAS_PREFIX / USE_TC.
#define PREFILL_DYN_BYTES(NSV) \
    ( (size_t)sizeof(T)   * 2 * (NSV) * TILE_K * HEAD_DIM \
    + (size_t)sizeof(Q_T) * WARPS_TC_COMPUTED * BLOCK_M * HEAD_DIM )
    constexpr size_t DYN_DOUBLE = PREFILL_DYN_BYTES(2);

    static size_t s_smem_double_actual = 0;
    static bool   s_configured = false;
    if (!s_configured) {
        // sharedSizeBytes is now just the small static bookkeeping; total smem a
        // launch needs is that plus the dynamic ring.
        cudaFuncAttributes attrs;
        cudaFuncGetAttributes(&attrs, paged_prefill_attn_fwd_chunks_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK,
            TILE_K, BLOCK_M, true, WARPS_TC_COMPUTED, true, 2>);
        s_smem_double_actual = attrs.sharedSizeBytes;
        cudaFuncGetAttributes(&attrs, paged_prefill_attn_fwd_chunks_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK,
            TILE_K, BLOCK_M, false, WARPS_TC_COMPUTED, true, 2>);
        if (attrs.sharedSizeBytes > s_smem_double_actual) s_smem_double_actual = attrs.sharedSizeBytes;
        s_smem_double_actual += DYN_DOUBLE;

        // Force double-buffering. Triple-buffering's third K/V ring stage
        // (~65 KB/block) exceeds what shared-constrained SMs — e.g. consumer
        // Blackwell at ~100 KB/SM — can fit more than ONE block of, capping
        // occupancy at 1 block/SM. Double-buffering (~49 KB/block) keeps the
        // load/compute overlap while letting 2 blocks co-reside, which this
        // latency-bound kernel benefits from more than the deeper pipeline.
        // cudaFuncSetAttribute takes the *dynamic* cap (DYN_*), not the total.
        if (s_smem_double_actual > caps.smem_default) {
            cudaFuncSetAttribute(
                paged_prefill_attn_fwd_chunks_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, TILE_K, BLOCK_M,
                    true, WARPS_TC_COMPUTED, true, 2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DYN_DOUBLE);
            cudaFuncSetAttribute(
                paged_prefill_attn_fwd_chunks_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, TILE_K, BLOCK_M,
                    true, WARPS_TC_COMPUTED, false, 2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DYN_DOUBLE);
            cudaFuncSetAttribute(
                paged_prefill_attn_fwd_chunks_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, TILE_K, BLOCK_M,
                    false, WARPS_TC_COMPUTED, true, 2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DYN_DOUBLE);
            cudaFuncSetAttribute(
                paged_prefill_attn_fwd_chunks_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, TILE_K, BLOCK_M,
                    false, WARPS_TC_COMPUTED, false, 2>,
                cudaFuncAttributeMaxDynamicSharedMemorySize, (int)DYN_DOUBLE);
            cudaGetLastError();
        }
        s_configured = true;
    }
    // Launch kernel (double-buffered — see the buffer-selection note above).
    #define LAUNCH_KERNEL(HAS_PREFIX, USE_TC_VAL, NUM_STAGES_VAL) \
        paged_prefill_attn_fwd_chunks_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, TILE_K, BLOCK_M, \
            HAS_PREFIX, WARPS_TC_COMPUTED, USE_TC_VAL, NUM_STAGES_VAL>\
            <<<grid, block_tc, PREFILL_DYN_BYTES(NUM_STAGES_VAL), stream>>>( \
            (const Q_T*)q_ptr, (const T*)k_ptr, (const T*)v_ptr, \
            headers_ptr, cu_seqlens_q, q_lens, kv_lens, \
            (O*)o_ptr, (int)batch_size, (int)n_head, (int)n_kv_head, \
            (int)max_blocks, softmax_scale, (int)total_q, rope_offsets, rope_cs, rope_interleaved, write_offset_shifts)

    if (has_prefix) {
        if (use_tc) { LAUNCH_KERNEL(true, true, 2); }
        else        { LAUNCH_KERNEL(true, false, 2); }
    } else {
        if (use_tc) { LAUNCH_KERNEL(false, true, 2); }
        else        { LAUNCH_KERNEL(false, false, 2); }
    }

    #undef LAUNCH_KERNEL
    #undef PREFILL_DYN_BYTES

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "PAGED PREFILL KERNEL LAUNCH FAILED: %s (grid=%d,%d,%d block=%d n_head=%d n_kv=%d hd=%d total_q=%d)\n",
                cudaGetErrorString(err), grid.x, grid.y, grid.z, block_tc.x,
                n_head, n_kv_head, HEAD_DIM, total_q);
    }
}
