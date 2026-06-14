#pragma once

// =============================================================================
// UNIFIED CONVERT HEADER
// =============================================================================
// Include this single header to get all format conversion capabilities.
// Supports float (F32, F16, BF16, FP8) and quantized (Q4_0/1, Q5_0/1, Q8_0/1) formats.
//
// LAYOUT DIFFERENCE:
// - Float formats (F32, F16, BF16, FP8): Channel-oriented, contiguous head vectors
//   Memory: [tok0_dim0, tok0_dim1, ..., tok0_dimN, tok1_dim0, ...]
//
// - Quant formats (Q4_0/1, Q5_0/1, Q8_0/1): Token-oriented blocking
//   Each 32-element block contains 32 consecutive TOKENS for a single DIMENSION.
//   Memory: [dim0_tok0-31, dim0_tok32-63, ..., dim1_tok0-31, ...]
//   This isolates outlier channels (1-2% of dims with 10-100x magnitude) so each
//   block gets its own scale, preserving effective precision across all dims.
//
// USAGE:
//   #include "convert/convert_all.cuh"
//   
//   // In kernel - use ArenaAccessor for clean index-based access:
//   ArenaAccessor accessor(arena_base, format, chunk_stride, head_stride, blocks_per_dim);
//   accessor.load_head_scaled<T, HEAD_DIM, USE_TC>(dst, chunk_idx, head_idx, within_idx, lane, scale);
// =============================================================================

// Common header with base template, scalar converters, block definitions
#include "convert.cuh"

// Dtype block converters
#include "block_f32.cuh"
#include "block_f16.cuh"
#include "block_bf16.cuh"
#include "block_fp8.cuh"

// Quant block converters
#include "block_q4_0.cuh"
#include "block_q4_1.cuh"
#include "block_q5_0.cuh"
#include "block_q5_1.cuh"
#include "block_q8_0.cuh"
#include "block_q8_1.cuh"
#include "block_q4_ks.cuh"
#include "block_q8_ks.cuh"
#include "block_q2_0.cuh"
#include "block_q3_0.cuh"
#include "block_q0.cuh"
#include "block_q0_v.cuh"
#include "block_q1_a.cuh"
#include "block_q0_x.cuh"
#include "block_q0_m2.cuh"
#include "block_q0_m4.cuh"
#include "block_q1_s.cuh"
#include "block_q2_s.cuh"
#include "block_q2_a.cuh"
#include "block_q2_1.cuh"
#include "block_q3_1.cuh"
#include "block_r16.cuh"

// =============================================================================
// RUNTIME FORMAT DISPATCH - ALL FORMATS
// Hot-path formats (F16, R16, Q4_0, Q8_0) are inlined at the call site.
// All others go through a __noinline__ slow path to reduce I-cache pressure.
// =============================================================================

// IS_K is threaded as a default-false template parameter so existing callers
// compile unchanged (they get V-side Q0_V tables). K-side decode call sites
// (e.g. paged-decode K, attention K-pass) explicitly pass IS_K=true so the
// Q0_V case uses the K-side calibrated tables.
template <typename DstType, int BLOCK_SIZE, bool IS_K = false>
__device__ __noinline__ void load_block_convert_slow(
    DstType* dst,
    const void* src,
    int format,
    int lane,
    float scale
) {
    switch (format) {
        case ArenaFormat::F32:
            BlockConverter<block_f32, DstType>::load(dst, reinterpret_cast<const block_f32*>(src), lane, scale);
            return;
        case ArenaFormat::BF16:
            BlockConverter<block_bf16, DstType>::load(dst, reinterpret_cast<const block_bf16*>(src), lane, scale);
            return;
        case ArenaFormat::F8E4M3:
            BlockConverter<block_fp8_e4m3, DstType>::load(dst, reinterpret_cast<const block_fp8_e4m3*>(src), lane, scale);
            return;
        case ArenaFormat::Q4_1:
            BlockConverter<block_q4_1, DstType>::load(dst, reinterpret_cast<const block_q4_1*>(src), lane, scale);
            return;
        case ArenaFormat::Q5_0:
            BlockConverter<block_q5_0, DstType>::load(dst, reinterpret_cast<const block_q5_0*>(src), lane, scale);
            return;
        case ArenaFormat::Q5_1:
            BlockConverter<block_q5_1, DstType>::load(dst, reinterpret_cast<const block_q5_1*>(src), lane, scale);
            return;
        case ArenaFormat::Q8_1:
            BlockConverter<block_q8_1, DstType>::load(dst, reinterpret_cast<const block_q8_1*>(src), lane, scale);
            return;
        case ArenaFormat::Q4_KS:
            BlockConverter<block_q4_ks, DstType>::load(dst, reinterpret_cast<const block_q4_ks*>(src), lane, scale);
            return;
        case ArenaFormat::Q8_KS:
            BlockConverter<block_q8_ks, DstType>::load(dst, reinterpret_cast<const block_q8_ks*>(src), lane, scale);
            return;
        case ArenaFormat::Q2_0:
            BlockConverter<block_q2_0, DstType>::load(dst, reinterpret_cast<const block_q2_0*>(src), lane, scale);
            return;
        case ArenaFormat::Q3_0:
            BlockConverter<block_q3_0, DstType>::load(dst, reinterpret_cast<const block_q3_0*>(src), lane, scale);
            return;
        case ArenaFormat::Q0:
            BlockConverter<block_q0, DstType>::load(dst, reinterpret_cast<const block_q0*>(src), lane, scale);
            return;
        case ArenaFormat::Q1_S:
            BlockConverter<block_q1_s, DstType>::load(dst, reinterpret_cast<const block_q1_s*>(src), lane, scale);
            return;
        case ArenaFormat::Q2_S:
            BlockConverter<block_q2_s, DstType>::load(dst, reinterpret_cast<const block_q2_s*>(src), lane, scale);
            return;
        case ArenaFormat::Q2_A:
            BlockConverter<block_q2_a, DstType>::load(dst, reinterpret_cast<const block_q2_a*>(src), lane, scale);
            return;
        case ArenaFormat::Q2_1:
            BlockConverter<block_q2_1, DstType>::load(dst, reinterpret_cast<const block_q2_1*>(src), lane, scale);
            return;
        case ArenaFormat::Q3_1:
            BlockConverter<block_q3_1, DstType>::load(dst, reinterpret_cast<const block_q3_1*>(src), lane, scale);
            return;
        case ArenaFormat::Q0_V:
            q0_v_load_block_typed<DstType, IS_K>(dst, reinterpret_cast<const block_q0_v*>(src), lane, scale);
            return;
        case ArenaFormat::Q1_A:
            BlockConverter<block_q1_a, DstType>::load(dst, reinterpret_cast<const block_q1_a*>(src), lane, scale);
            return;
        case ArenaFormat::Q0_X:
            BlockConverter<block_q0_x, DstType>::load(dst, reinterpret_cast<const block_q0_x*>(src), lane, scale);
            return;
        case ArenaFormat::Q0_M2:
            BlockConverter<block_q0_m2, DstType>::load(dst, reinterpret_cast<const block_q0_m2*>(src), lane, scale);
            return;
        case ArenaFormat::Q0_M4:
            BlockConverter<block_q0_m4, DstType>::load(dst, reinterpret_cast<const block_q0_m4*>(src), lane, scale);
            return;
        default:
            dst[lane] = from_float<DstType>(0.f);
            return;
    }
}

template <typename DstType, int BLOCK_SIZE, bool IS_K = false>
__device__ __forceinline__ void load_block_convert(
    DstType* dst,
    const void* src,
    int format,
    int lane,
    float scale
) {
    // Hot-path formats inlined at call site
    if (format == ArenaFormat::F16) {
        BlockConverter<block_f16, DstType>::load(dst, reinterpret_cast<const block_f16*>(src), lane, scale);
    } else if (format == ArenaFormat::BF16) {
        BlockConverter<block_bf16, DstType>::load(dst, reinterpret_cast<const block_bf16*>(src), lane, scale);
    } else if (format == ArenaFormat::R16) {
        BlockConverter<block_r16, DstType>::load(dst, reinterpret_cast<const block_r16*>(src), lane, scale);
    } else if (format == ArenaFormat::Q4_0) {
        BlockConverter<block_q4_0, DstType>::load(dst, reinterpret_cast<const block_q4_0*>(src), lane, scale);
    } else if (format == ArenaFormat::Q8_0) {
        BlockConverter<block_q8_0, DstType>::load(dst, reinterpret_cast<const block_q8_0*>(src), lane, scale);
    } else {
        load_block_convert_slow<DstType, BLOCK_SIZE, IS_K>(dst, src, format, lane, scale);
    }
}

template <typename SrcType, int BLOCK_SIZE>
__device__ __forceinline__ void store_block_convert(
    void* dst,
    const SrcType* src,
    int format,
    int lane
) {
    switch (format) {
        // Float formats
        case ArenaFormat::F32:
            BlockConverter<block_f32, SrcType>::store(reinterpret_cast<block_f32*>(dst), src, lane);
            return;
        case ArenaFormat::F16:
            BlockConverter<block_f16, SrcType>::store(reinterpret_cast<block_f16*>(dst), src, lane);
            return;
        case ArenaFormat::R16:
            BlockConverter<block_r16, SrcType>::store(reinterpret_cast<block_r16*>(dst), src, lane);
            return;
        case ArenaFormat::BF16:
            BlockConverter<block_bf16, SrcType>::store(reinterpret_cast<block_bf16*>(dst), src, lane);
            return;
        case ArenaFormat::F8E4M3:
            BlockConverter<block_fp8_e4m3, SrcType>::store(reinterpret_cast<block_fp8_e4m3*>(dst), src, lane);
            return;
        
        // Quant formats - quantization not implemented (read-only)
        default:
            return;
    }
}

template <typename DstType, int BLOCK_SIZE, bool IS_K = false>
__device__ __forceinline__ void load_block_convert_all(
    DstType* dst,
    const void* src,
    int format,
    int lane,
    float scale
) {
    load_block_convert<DstType, BLOCK_SIZE, IS_K>(dst, src, format, lane, scale);
}

// =============================================================================
// SINGLE-ELEMENT DEQUANT DISPATCH (for token-oriented quant loading)
// =============================================================================
// Extracts and dequantizes a single element from a quantized block.
// Used when gathering one element from each of HEAD_DIM blocks.
//
// F16 is intentionally not handled here: it has `float_elem_size > 0` so
// callers route it through the float load path, not this function. R16 is
// handled here because its `float_elem_size == 0` pushes it through the quant
// path even though it carries float values.

// dequant_element_slow: all 22 formats, __noinline__ for compact code.
// IS_K threaded as default-false for back-compat; Q0_V dispatches by IS_K.
template <typename T, bool IS_K = false>
__device__ __noinline__ T dequant_element_slow(const void* block_ptr, int idx, int format, float scale) {
    switch (format) {
        case ArenaFormat::R16:
            return BlockConverter<block_r16, T>::load_element(
                reinterpret_cast<const block_r16*>(block_ptr), idx, scale);
        case ArenaFormat::Q4_0:
            return BlockConverter<block_q4_0, T>::load_element(
                reinterpret_cast<const block_q4_0*>(block_ptr), idx, scale);
        case ArenaFormat::Q4_1:
            return BlockConverter<block_q4_1, T>::load_element(
                reinterpret_cast<const block_q4_1*>(block_ptr), idx, scale);
        case ArenaFormat::Q5_0:
            return BlockConverter<block_q5_0, T>::load_element(
                reinterpret_cast<const block_q5_0*>(block_ptr), idx, scale);
        case ArenaFormat::Q5_1:
            return BlockConverter<block_q5_1, T>::load_element(
                reinterpret_cast<const block_q5_1*>(block_ptr), idx, scale);
        case ArenaFormat::Q8_0:
            return BlockConverter<block_q8_0, T>::load_element(
                reinterpret_cast<const block_q8_0*>(block_ptr), idx, scale);
        case ArenaFormat::Q8_1:
            return BlockConverter<block_q8_1, T>::load_element(
                reinterpret_cast<const block_q8_1*>(block_ptr), idx, scale);
        case ArenaFormat::Q4_KS:
            return BlockConverter<block_q4_ks, T>::load_element(
                reinterpret_cast<const block_q4_ks*>(block_ptr), idx, scale);
        case ArenaFormat::Q8_KS:
            return BlockConverter<block_q8_ks, T>::load_element(
                reinterpret_cast<const block_q8_ks*>(block_ptr), idx, scale);
        case ArenaFormat::Q3_0:
            return BlockConverter<block_q3_0, T>::load_element(
                reinterpret_cast<const block_q3_0*>(block_ptr), idx, scale);
        case ArenaFormat::Q3_1:
            return BlockConverter<block_q3_1, T>::load_element(
                reinterpret_cast<const block_q3_1*>(block_ptr), idx, scale);
        case ArenaFormat::Q2_0:
            return BlockConverter<block_q2_0, T>::load_element(
                reinterpret_cast<const block_q2_0*>(block_ptr), idx, scale);
        case ArenaFormat::Q2_1:
            return BlockConverter<block_q2_1, T>::load_element(
                reinterpret_cast<const block_q2_1*>(block_ptr), idx, scale);
        case ArenaFormat::Q2_A:
            return BlockConverter<block_q2_a, T>::load_element(
                reinterpret_cast<const block_q2_a*>(block_ptr), idx, scale);
        case ArenaFormat::Q2_S:
            return BlockConverter<block_q2_s, T>::load_element(
                reinterpret_cast<const block_q2_s*>(block_ptr), idx, scale);
        case ArenaFormat::Q1_S:
            return BlockConverter<block_q1_s, T>::load_element(
                reinterpret_cast<const block_q1_s*>(block_ptr), idx, scale);
        case ArenaFormat::Q0:
            return BlockConverter<block_q0, T>::load_element(
                reinterpret_cast<const block_q0*>(block_ptr), idx, scale);
        case ArenaFormat::Q0_V:
            return q0_v_load_element_typed<T, IS_K>(
                reinterpret_cast<const block_q0_v*>(block_ptr), idx, scale);
        case ArenaFormat::Q1_A:
            return BlockConverter<block_q1_a, T>::load_element(
                reinterpret_cast<const block_q1_a*>(block_ptr), idx, scale);
        case ArenaFormat::Q0_X:
            return BlockConverter<block_q0_x, T>::load_element(
                reinterpret_cast<const block_q0_x*>(block_ptr), idx, scale);
        case ArenaFormat::Q0_M2:
            return BlockConverter<block_q0_m2, T>::load_element(
                reinterpret_cast<const block_q0_m2*>(block_ptr), idx, scale);
        case ArenaFormat::Q0_M4:
            return BlockConverter<block_q0_m4, T>::load_element(
                reinterpret_cast<const block_q0_m4*>(block_ptr), idx, scale);
        default:
            __trap();  // unhandled format — a programming error (no valid arena hits this)
            return T(0);
    }
}

// dequant_element_hybrid: hot-5 fast path (R16, Q4_0, Q8_0, Q4_KS, Q8_KS)
// inlined + __noinline__ slow path for all others. Preferred for prefill.
// IS_K threaded through to the slow-path delegation so Q0_V picks the
// correct K/V calibrated tables.
template <typename T, bool IS_K = false>
__device__ __forceinline__ T dequant_element_hybrid(const void* block_ptr, int idx, int format, float scale) {
    // Fast path: most common formats inlined
    switch (format) {
        case ArenaFormat::R16:
            return BlockConverter<block_r16, T>::load_element(
                reinterpret_cast<const block_r16*>(block_ptr), idx, scale);
        case ArenaFormat::Q4_0:
            return BlockConverter<block_q4_0, T>::load_element(
                reinterpret_cast<const block_q4_0*>(block_ptr), idx, scale);
        case ArenaFormat::Q8_0:
            return BlockConverter<block_q8_0, T>::load_element(
                reinterpret_cast<const block_q8_0*>(block_ptr), idx, scale);
        case ArenaFormat::Q4_KS:
            return BlockConverter<block_q4_ks, T>::load_element(
                reinterpret_cast<const block_q4_ks*>(block_ptr), idx, scale);
        case ArenaFormat::Q8_KS:
            return BlockConverter<block_q8_ks, T>::load_element(
                reinterpret_cast<const block_q8_ks*>(block_ptr), idx, scale);
        default:
            // Delegate cold path to __noinline__ slow function
            return dequant_element_slow<T, IS_K>(block_ptr, idx, format, scale);
    }
}

// dequant_element_inline: all 22 formats fully inlined at the call site.
// Use for aggressive latency requirements (e.g., select_kv_format). IS_K
// threaded as default-false; Q0_V dispatches by IS_K.
template <typename T, bool IS_K = false>
__device__ __forceinline__ T dequant_element_inline(const void* block_ptr, int idx, int format, float scale) {
    switch (format) {
        case ArenaFormat::R16:
            return BlockConverter<block_r16, T>::load_element(
                reinterpret_cast<const block_r16*>(block_ptr), idx, scale);
        case ArenaFormat::Q4_0:
            return BlockConverter<block_q4_0, T>::load_element(
                reinterpret_cast<const block_q4_0*>(block_ptr), idx, scale);
        case ArenaFormat::Q4_1:
            return BlockConverter<block_q4_1, T>::load_element(
                reinterpret_cast<const block_q4_1*>(block_ptr), idx, scale);
        case ArenaFormat::Q5_0:
            return BlockConverter<block_q5_0, T>::load_element(
                reinterpret_cast<const block_q5_0*>(block_ptr), idx, scale);
        case ArenaFormat::Q5_1:
            return BlockConverter<block_q5_1, T>::load_element(
                reinterpret_cast<const block_q5_1*>(block_ptr), idx, scale);
        case ArenaFormat::Q8_0:
            return BlockConverter<block_q8_0, T>::load_element(
                reinterpret_cast<const block_q8_0*>(block_ptr), idx, scale);
        case ArenaFormat::Q8_1:
            return BlockConverter<block_q8_1, T>::load_element(
                reinterpret_cast<const block_q8_1*>(block_ptr), idx, scale);
        case ArenaFormat::Q4_KS:
            return BlockConverter<block_q4_ks, T>::load_element(
                reinterpret_cast<const block_q4_ks*>(block_ptr), idx, scale);
        case ArenaFormat::Q8_KS:
            return BlockConverter<block_q8_ks, T>::load_element(
                reinterpret_cast<const block_q8_ks*>(block_ptr), idx, scale);
        case ArenaFormat::Q3_0:
            return BlockConverter<block_q3_0, T>::load_element(
                reinterpret_cast<const block_q3_0*>(block_ptr), idx, scale);
        case ArenaFormat::Q3_1:
            return BlockConverter<block_q3_1, T>::load_element(
                reinterpret_cast<const block_q3_1*>(block_ptr), idx, scale);
        case ArenaFormat::Q2_0:
            return BlockConverter<block_q2_0, T>::load_element(
                reinterpret_cast<const block_q2_0*>(block_ptr), idx, scale);
        case ArenaFormat::Q2_1:
            return BlockConverter<block_q2_1, T>::load_element(
                reinterpret_cast<const block_q2_1*>(block_ptr), idx, scale);
        case ArenaFormat::Q2_A:
            return BlockConverter<block_q2_a, T>::load_element(
                reinterpret_cast<const block_q2_a*>(block_ptr), idx, scale);
        case ArenaFormat::Q2_S:
            return BlockConverter<block_q2_s, T>::load_element(
                reinterpret_cast<const block_q2_s*>(block_ptr), idx, scale);
        case ArenaFormat::Q1_S:
            return BlockConverter<block_q1_s, T>::load_element(
                reinterpret_cast<const block_q1_s*>(block_ptr), idx, scale);
        case ArenaFormat::Q0:
            return BlockConverter<block_q0, T>::load_element(
                reinterpret_cast<const block_q0*>(block_ptr), idx, scale);
        case ArenaFormat::Q0_V:
            return q0_v_load_element_typed<T, IS_K>(
                reinterpret_cast<const block_q0_v*>(block_ptr), idx, scale);
        case ArenaFormat::Q1_A:
            return BlockConverter<block_q1_a, T>::load_element(
                reinterpret_cast<const block_q1_a*>(block_ptr), idx, scale);
        case ArenaFormat::Q0_X:
            return BlockConverter<block_q0_x, T>::load_element(
                reinterpret_cast<const block_q0_x*>(block_ptr), idx, scale);
        case ArenaFormat::Q0_M2:
            return BlockConverter<block_q0_m2, T>::load_element(
                reinterpret_cast<const block_q0_m2*>(block_ptr), idx, scale);
        case ArenaFormat::Q0_M4:
            return BlockConverter<block_q0_m4, T>::load_element(
                reinterpret_cast<const block_q0_m4*>(block_ptr), idx, scale);
        default:
            __trap();  // unhandled format — a programming error (no valid arena hits this)
            return T(0);
    }
}

// =============================================================================
// HELPER FOR 16-BYTE ASYNC COPY (SM80+)
// =============================================================================

template <bool USE_TC>
__device__ __forceinline__ void cp_async_cg_16(void* dst, const void* src) {
    if constexpr (USE_TC) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst))),
                        "l"(src));
    } else {
        *reinterpret_cast<float4*>(dst) = *reinterpret_cast<const float4*>(src);
    }
}

// =============================================================================
// ARENA ACCESSOR - CLEAN INDEX-BASED ACCESS ABSTRACTION
// =============================================================================
// Handles both dtype (channel-oriented) and quant (token-oriented) formats.
//
// For dtype formats: contiguous head vectors, traditional linear access
// For quant formats: token-oriented blocking where each block has 32 tokens × 1 dim

struct ArenaAccessor {
    const char* base;
    int format;
    int64_t chunk_stride;
    int64_t head_stride;
    int blocks_per_dim;  // CHUNK_SIZE / 32, for token-oriented quant layout
    int64_t chunk_byte_stride;  // Bytes between consecutive chunks for this head (0 = derive from element strides)
    
    __device__ __forceinline__ ArenaAccessor(
        const char* arena_base,
        int arena_format,
        int64_t arena_chunk_stride,
        int64_t arena_head_stride,
        int arena_blocks_per_dim = 1,  // Default 1 for backward compat (CHUNK_SIZE=32)
        int64_t arena_chunk_byte_stride = 0  // 0 = derive from element strides (backward compat)
    ) : base(arena_base), format(arena_format), 
        chunk_stride(arena_chunk_stride), head_stride(arena_head_stride),
        blocks_per_dim(arena_blocks_per_dim), chunk_byte_stride(arena_chunk_byte_stride) {}
    
    // Element offset for dtype formats (channel-oriented)
    template <int HEAD_DIM>
    __device__ __forceinline__ int64_t elem_offset(
        int chunk_idx,
        int head_idx,
        int within_chunk
    ) const {
        return (int64_t)chunk_idx * chunk_stride + 
               (int64_t)head_idx * head_stride +
               (int64_t)within_chunk * (int64_t)HEAD_DIM;
    }
    
    __device__ __forceinline__ const char* byte_ptr(int64_t elem_off) const {
        int elem_size = ArenaFormat::float_elem_size(format);
        if (elem_size > 0) {
            return base + elem_off * elem_size;
        } else {
            constexpr int BLOCK_ELEMS = 32;
            int block_idx = elem_off / BLOCK_ELEMS;
            int block_bytes = get_quant_block_bytes(format);
            return base + (int64_t)block_idx * block_bytes;
        }
    }
    
    __device__ __forceinline__ int get_quant_block_bytes(int fmt) const {
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
            case ArenaFormat::Q0_V:  return 2;
            case ArenaFormat::Q1_A:  return 6;
            case ArenaFormat::Q0_X:  return 2;
            case ArenaFormat::Q0_M2: return 3;
            case ArenaFormat::Q0_M4: return 8;
            default: return 32;
        }
    }
    
    template <typename T, int HEAD_DIM, bool USE_TC>
    __device__ __forceinline__ void load_head_scaled(
        T* dst,
        int chunk_idx,
        int head_idx,
        int within_chunk,
        int lane,
        float scale
    ) const {
        constexpr int T_FORMAT = type_to_arena_format<T>();

        // Check if format is a quant format (needs token-oriented loading)
        int elem_size = ArenaFormat::float_elem_size(format);
        if (elem_size <= 0) {
            // Quant format: use token-oriented loading
            load_head_quant_token_oriented<T, HEAD_DIM>(dst, chunk_idx, head_idx, within_chunk, lane, scale);
            return;
        }

        // Dtype format: use traditional channel-oriented loading
        int64_t elem_off = elem_offset<HEAD_DIM>(chunk_idx, head_idx, within_chunk);

        if (format == T_FORMAT && scale == 1.0f) {
            const T* src = reinterpret_cast<const T*>(base) + elem_off;
            load_head_fast<T, HEAD_DIM, USE_TC>(dst, src, lane);
            return;
        }

        load_head_convert_dtype<T, HEAD_DIM, USE_TC>(dst, elem_off, lane, scale);
    }
    
private:
    // =========================================================================
    // DTYPE FAST PATH - Same-type, direct copy with async
    // =========================================================================
    template <typename T, int HEAD_DIM, bool USE_TC>
    __device__ __forceinline__ void load_head_fast(
        T* dst,
        const T* src,
        int lane
    ) const {
        if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
            constexpr int ELEMS_PER_COPY = 16;
            constexpr int COPIES = HEAD_DIM / ELEMS_PER_COPY;
            constexpr int UNROLL = (COPIES <= 8) ? 2 : 1;
            #pragma unroll UNROLL
            for (int c = lane; c < COPIES; c += 32) {
                cp_async_cg_16<USE_TC>(dst + c * ELEMS_PER_COPY, src + c * ELEMS_PER_COPY);
            }
        } else if constexpr (std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) {
            constexpr int ELEMS_PER_COPY = 8;
            constexpr int COPIES = HEAD_DIM / ELEMS_PER_COPY;
            constexpr int UNROLL = (COPIES <= 8) ? 2 : 1;
            #pragma unroll UNROLL
            for (int c = lane; c < COPIES; c += 32) {
                cp_async_cg_16<USE_TC>(dst + c * ELEMS_PER_COPY, src + c * ELEMS_PER_COPY);
            }
        } else {
            constexpr int ELEMS_PER_COPY = 4;
            constexpr int COPIES = HEAD_DIM / ELEMS_PER_COPY;
            constexpr int UNROLL = (COPIES <= 16) ? 2 : 1;
            #pragma unroll UNROLL
            for (int c = lane; c < COPIES; c += 32) {
                cp_async_cg_16<USE_TC>(dst + c * ELEMS_PER_COPY, src + c * ELEMS_PER_COPY);
            }
        }
    }
    
    // =========================================================================
    // DTYPE CONVERT PATH - Convert between float formats (F32/F16/BF16/FP8)
    // =========================================================================
    template <typename T, int HEAD_DIM, bool USE_TC>
    __device__ __forceinline__ void load_head_convert_dtype(
        T* dst,
        int64_t elem_off,
        int lane,
        float scale
    ) const {
        constexpr int BLOCK_SIZE = (HEAD_DIM >= 32) ? 32 : HEAD_DIM;
        constexpr int NUM_BLOCKS = HEAD_DIM / BLOCK_SIZE;

        int elem_size = ArenaFormat::float_elem_size(format);
        const char* src = base + elem_off * elem_size;

        // Each lane converts element `lane` of a BLOCK_SIZE-wide band. When the
        // band is narrower than the warp (a sub-head band, e.g. SUB_HEAD_DIM=16
        // at HEAD_DIM=64), lanes >= BLOCK_SIZE own no element — guard them so the
        // per-lane converter doesn't read `src->data[lane]` / write `dst[lane]`
        // past the band (which runs off the arena row → illegal address for the
        // final token). No-op when BLOCK_SIZE == 32 (full-width bands).
        if (lane < BLOCK_SIZE) {
            #pragma unroll
            for (int b = 0; b < NUM_BLOCKS; ++b) {
                load_block_convert_all<T, BLOCK_SIZE>(
                    dst + b * BLOCK_SIZE,
                    src + b * BLOCK_SIZE * elem_size,
                    format,
                    lane,
                    scale
                );
            }
        }
    }
    
    // =========================================================================
    // QUANT TOKEN-ORIENTED PATH - Gather from HEAD_DIM blocks
    // =========================================================================
    // Token-oriented layout: each block contains 32 consecutive TOKENS for 1 DIM.
    // Block indexing within a chunk's head:
    //   block_idx = dim * blocks_per_dim + (token_in_chunk / 32)
    //   elem_in_block = token_in_chunk % 32
    //
    // Memory layout per chunk (n_kv_head heads, HEAD_DIM dims, blocks_per_dim blocks):
    //   [h0_d0_b0][h0_d0_b1]...[h0_d1_b0]...[h0_d(H-1)_b(B-1)][h1_d0_b0]...
    // Where h=head, d=dim, b=block_within_dim, H=HEAD_DIM, B=blocks_per_dim
    //
    // For HEAD_DIM=128, we need to read from 128 different blocks.
    // Each warp lane handles HEAD_DIM/32 dimensions.
    template <typename BlockT, typename T, int HEAD_DIM>
    __device__ __forceinline__ void load_head_quant_token_oriented_typed(
        T* dst,
        int chunk_idx,
        int head_idx,
        int within_chunk,
        int lane,
        float scale
    ) const {
        constexpr int DIMS_PER_LANE = HEAD_DIM / 32;
        constexpr int BLOCK_BYTES = sizeof(BlockT);

        const int elem_in_block = within_chunk & 31;
        const int block_within_dim = within_chunk >> 5;
        const int64_t block_head_stride = (int64_t)HEAD_DIM * blocks_per_dim;

        const char* head_base = (chunk_byte_stride > 0)
            ? (base + (int64_t)chunk_idx * chunk_byte_stride
                    + (int64_t)head_idx * block_head_stride * BLOCK_BYTES)
            : (base + ((int64_t)chunk_idx * ((chunk_stride / head_stride) * block_head_stride)
                      + (int64_t)head_idx * block_head_stride) * BLOCK_BYTES);

        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; ++i) {
            const int dim = lane + i * 32;
            const int64_t block_idx = (int64_t)dim * blocks_per_dim + block_within_dim;
            const BlockT* block_ptr = reinterpret_cast<const BlockT*>(head_base + block_idx * BLOCK_BYTES);
            dst[dim] = BlockConverter<BlockT, T>::load_element(block_ptr, elem_in_block, scale);
        }
    }

    template <typename T, int HEAD_DIM>
    __device__ __forceinline__ void load_head_quant_token_oriented(
        T* dst,
        int chunk_idx,
        int head_idx,
        int within_chunk,
        int lane,
        float scale
    ) const {
        // Switch once on format, then run the full palette-span load.
        // This amortizes format dispatch across all dimensions in the run.
        switch (format) {
            case ArenaFormat::R16:
                load_head_quant_token_oriented_typed<block_r16, T, HEAD_DIM>(
                    dst, chunk_idx, head_idx, within_chunk, lane, scale);
                return;
            case ArenaFormat::Q4_0:
                load_head_quant_token_oriented_typed<block_q4_0, T, HEAD_DIM>(
                    dst, chunk_idx, head_idx, within_chunk, lane, scale);
                return;
            case ArenaFormat::Q8_0:
                load_head_quant_token_oriented_typed<block_q8_0, T, HEAD_DIM>(
                    dst, chunk_idx, head_idx, within_chunk, lane, scale);
                return;
            case ArenaFormat::Q4_KS:
                load_head_quant_token_oriented_typed<block_q4_ks, T, HEAD_DIM>(
                    dst, chunk_idx, head_idx, within_chunk, lane, scale);
                return;
            case ArenaFormat::Q8_KS:
                load_head_quant_token_oriented_typed<block_q8_ks, T, HEAD_DIM>(
                    dst, chunk_idx, head_idx, within_chunk, lane, scale);
                return;
            default:
                break;
        }

        // Fallback for less-common formats keeps the generic element path.
        constexpr int DIMS_PER_LANE = HEAD_DIM / 32;
        const int block_bytes = get_quant_block_bytes(format);
        const int elem_in_block = within_chunk & 31;
        const int block_within_dim = within_chunk >> 5;
        const int64_t block_head_stride = (int64_t)HEAD_DIM * blocks_per_dim;

        const char* head_base = (chunk_byte_stride > 0)
            ? (base + (int64_t)chunk_idx * chunk_byte_stride
                    + (int64_t)head_idx * block_head_stride * block_bytes)
            : (base + ((int64_t)chunk_idx * ((chunk_stride / head_stride) * block_head_stride)
                      + (int64_t)head_idx * block_head_stride) * block_bytes);

        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; ++i) {
            const int dim = lane + i * 32;
            const int64_t block_idx = (int64_t)dim * blocks_per_dim + block_within_dim;
            const char* block_ptr = head_base + block_idx * block_bytes;
            dst[dim] = dequant_element_inline<T>(block_ptr, elem_in_block, format, scale);
        }
    }

public:
    // =========================================================================
    // LOAD HEAD INT8 READ-THROUGH (per-dim block scales) — for V skip-dequant
    // =========================================================================
    // Reads a head's int8 values straight from a native-INT8 arena with NO FP
    // round-trip, writing each dim's per-(dim,block) scale into out_dim_scale[]
    // so the consumer (the INT8 PV, §1A) applies one scale per output dim.
    // Three pieces, same shape as load_head_scaled's quant path:
    //   - is_int8_readthrough_format(fmt) — the eligibility predicate;
    //   - load_head_int8_readthrough_typed<BlockT> — the TYPED worker, keyed on
    //     the block struct so BLOCK_BYTES = sizeof(BlockT) and the element
    //     extraction (BlockInt8<BlockT>) compile away (no magic sizes, no
    //     duplicated unpack);
    //   - load_head_int8_readthrough<HEAD_DIM> — the void format DISPATCHER that
    //     switch()es the runtime format to the worker (the generic home for the
    //     switch, mirroring load_head_quant_token_oriented). No bool to drop:
    //     the caller gates with is_int8_readthrough_format and supplies the FP
    //     path (load_head_scaled) for non-passthrough formats.
    //
    //   dst[dim]          : centered int8 for this token (palette-local order)
    //   out_dim_scale[dim]: FP32 scale s with dequant == dst[dim] * s, already
    //                       divided by in_scale (matching load_head_scaled's
    //                       `/ scale`)
    //   in_scale          : per-palette scale from the head metadata (v_scale_p)
    //
    // K never uses this (RoPE needs FP).

    // True for the formats the dispatcher/worker handle. The caller gates the
    // read-through on this; keep in sync with the switch in the dispatcher below.
    static __device__ __forceinline__ bool is_int8_readthrough_format(int fmt) {
        return fmt == ArenaFormat::Q8_0  || fmt == ArenaFormat::Q4_0
            || fmt == ArenaFormat::Q5_0  || fmt == ArenaFormat::Q2_0
            || fmt == ArenaFormat::Q3_0  || fmt == ArenaFormat::Q4_KS
            || fmt == ArenaFormat::Q8_KS || fmt == ArenaFormat::Q8_1
            || fmt == ArenaFormat::Q2_S  || fmt == ArenaFormat::Q1_S
            || fmt == ArenaFormat::Q1_A  || fmt == ArenaFormat::Q0
            || fmt == ArenaFormat::Q0_M2 || fmt == ArenaFormat::Q0_M4
            || fmt == ArenaFormat::Q0_X;
    }

    template <typename BlockT, int HEAD_DIM>
    __device__ __forceinline__ void load_head_int8_readthrough_typed(
        int8_t* dst,
        float* out_dim_scale,
        int chunk_idx,
        int head_idx,
        int within_chunk,
        int lane,
        float in_scale
    ) const {
        constexpr int DIMS_PER_LANE = HEAD_DIM / 32;
        constexpr int BLOCK_BYTES = sizeof(BlockT);
        const int elem_in_block = within_chunk & 31;
        const int block_within_dim = within_chunk >> 5;
        const int64_t block_head_stride = (int64_t)HEAD_DIM * blocks_per_dim;
        const float inv = 1.0f / in_scale;

        const char* head_base = (chunk_byte_stride > 0)
            ? (base + (int64_t)chunk_idx * chunk_byte_stride
                    + (int64_t)head_idx * block_head_stride * BLOCK_BYTES)
            : (base + ((int64_t)chunk_idx * ((chunk_stride / (head_stride > 0 ? head_stride : 1)) * block_head_stride)
                      + (int64_t)head_idx * block_head_stride) * BLOCK_BYTES);

        #pragma unroll
        for (int i = 0; i < DIMS_PER_LANE; ++i) {
            const int dim = lane + i * 32;
            const int64_t block_idx = (int64_t)dim * blocks_per_dim + block_within_dim;
            const BlockT* blk = reinterpret_cast<const BlockT*>(head_base + block_idx * BLOCK_BYTES);
            const Int8Sample smp = BlockInt8<BlockT>::load(blk, elem_in_block);
            dst[dim] = smp.v;
            out_dim_scale[dim] = smp.s * inv;
        }
    }

    // Format dispatcher — the generic home for the read-through switch, mirroring
    // how load_head_scaled routes through load_head_quant_token_oriented's
    // switch(format). Switches the RUNTIME format ONCE to the typed worker.
    // Returns void (no dropped bool): the caller gates on
    // is_int8_readthrough_format and supplies the FP path for non-passthrough
    // formats, so the default below is unreachable.
    template <int HEAD_DIM>
    __device__ __forceinline__ void load_head_int8_readthrough(
        int8_t* dst,
        float* out_dim_scale,
        int chunk_idx,
        int head_idx,
        int within_chunk,
        int lane,
        float in_scale
    ) const {
        switch (format) {
            case ArenaFormat::Q8_0:  load_head_int8_readthrough_typed<block_q8_0,  HEAD_DIM>(dst, out_dim_scale, chunk_idx, head_idx, within_chunk, lane, in_scale); return;
            case ArenaFormat::Q4_0:  load_head_int8_readthrough_typed<block_q4_0,  HEAD_DIM>(dst, out_dim_scale, chunk_idx, head_idx, within_chunk, lane, in_scale); return;
            case ArenaFormat::Q5_0:  load_head_int8_readthrough_typed<block_q5_0,  HEAD_DIM>(dst, out_dim_scale, chunk_idx, head_idx, within_chunk, lane, in_scale); return;
            case ArenaFormat::Q2_0:  load_head_int8_readthrough_typed<block_q2_0,  HEAD_DIM>(dst, out_dim_scale, chunk_idx, head_idx, within_chunk, lane, in_scale); return;
            case ArenaFormat::Q3_0:  load_head_int8_readthrough_typed<block_q3_0,  HEAD_DIM>(dst, out_dim_scale, chunk_idx, head_idx, within_chunk, lane, in_scale); return;
            case ArenaFormat::Q4_KS: load_head_int8_readthrough_typed<block_q4_ks, HEAD_DIM>(dst, out_dim_scale, chunk_idx, head_idx, within_chunk, lane, in_scale); return;
            case ArenaFormat::Q8_KS: load_head_int8_readthrough_typed<block_q8_ks, HEAD_DIM>(dst, out_dim_scale, chunk_idx, head_idx, within_chunk, lane, in_scale); return;
            case ArenaFormat::Q8_1:  load_head_int8_readthrough_typed<block_q8_1,  HEAD_DIM>(dst, out_dim_scale, chunk_idx, head_idx, within_chunk, lane, in_scale); return;
            case ArenaFormat::Q2_S:  load_head_int8_readthrough_typed<block_q2_s,  HEAD_DIM>(dst, out_dim_scale, chunk_idx, head_idx, within_chunk, lane, in_scale); return;
            case ArenaFormat::Q1_S:  load_head_int8_readthrough_typed<block_q1_s,  HEAD_DIM>(dst, out_dim_scale, chunk_idx, head_idx, within_chunk, lane, in_scale); return;
            case ArenaFormat::Q1_A:  load_head_int8_readthrough_typed<block_q1_a,  HEAD_DIM>(dst, out_dim_scale, chunk_idx, head_idx, within_chunk, lane, in_scale); return;
            case ArenaFormat::Q0:    load_head_int8_readthrough_typed<block_q0,    HEAD_DIM>(dst, out_dim_scale, chunk_idx, head_idx, within_chunk, lane, in_scale); return;
            case ArenaFormat::Q0_M2: load_head_int8_readthrough_typed<block_q0_m2, HEAD_DIM>(dst, out_dim_scale, chunk_idx, head_idx, within_chunk, lane, in_scale); return;
            case ArenaFormat::Q0_M4: load_head_int8_readthrough_typed<block_q0_m4, HEAD_DIM>(dst, out_dim_scale, chunk_idx, head_idx, within_chunk, lane, in_scale); return;
            case ArenaFormat::Q0_X:  load_head_int8_readthrough_typed<block_q0_x,  HEAD_DIM>(dst, out_dim_scale, chunk_idx, head_idx, within_chunk, lane, in_scale); return;
            default: return;  // unreachable: caller gates on is_int8_readthrough_format
        }
    }
};

// =============================================================================
// ARENA ACCESSOR (MUTABLE) - For writing to arenas
// =============================================================================
// Supports dtype formats only (quant formats require separate quantization pass).

struct ArenaAccessorMut {
    char* base;
    int format;
    int64_t chunk_stride;
    int64_t head_stride;
    
    __device__ __forceinline__ ArenaAccessorMut(
        char* arena_base,
        int arena_format,
        int64_t arena_chunk_stride,
        int64_t arena_head_stride
    ) : base(arena_base), format(arena_format), 
        chunk_stride(arena_chunk_stride), head_stride(arena_head_stride) {}
    
    // Element offset for dtype formats (channel-oriented)
    template <int HEAD_DIM>
    __device__ __forceinline__ int64_t elem_offset(
        int chunk_idx,
        int head_idx,
        int within_chunk
    ) const {
        return (int64_t)chunk_idx * chunk_stride + 
               (int64_t)head_idx * head_stride +
               (int64_t)within_chunk * (int64_t)HEAD_DIM;
    }
    
    // Store a head vector with format conversion (dtype only)
    // SrcType: type of source data (e.g., BF16 from model)
    // HEAD_DIM: head dimension
    template <typename SrcType, int HEAD_DIM>
    __device__ __forceinline__ void store_head(
        const SrcType* src,
        int chunk_idx,
        int head_idx,
        int within_chunk,
        int lane
    ) {
        constexpr int SRC_FORMAT = type_to_arena_format<SrcType>();
        int64_t elem_off = elem_offset<HEAD_DIM>(chunk_idx, head_idx, within_chunk);
        int elem_size = ArenaFormat::float_elem_size(format);
        
        // Assert dtype format (quant not supported for writes)
        // In debug builds this helps catch misconfigured arenas
        // if (elem_size <= 0) return;  // Silent fail for quant formats
        
        if (format == SRC_FORMAT) {
            // Same type: direct copy
            SrcType* dst = reinterpret_cast<SrcType*>(base) + elem_off;
            store_head_fast<SrcType, HEAD_DIM>(dst, src, lane);
        } else {
            // Different dtype: use converter
            store_head_convert<SrcType, HEAD_DIM>(elem_off, src, lane);
        }
    }
    
private:
    // =========================================================================
    // DTYPE FAST PATH - Same-type, direct vectorized copy
    // =========================================================================
    template <typename T, int HEAD_DIM>
    __device__ __forceinline__ void store_head_fast(
        T* dst,
        const T* src,
        int lane
    ) {
        constexpr int VEC = HEAD_DIM / 32;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            dst[lane * VEC + j] = src[lane * VEC + j];
        }
    }
    
    // =========================================================================
    // DTYPE CONVERT PATH - Convert between float formats
    // =========================================================================
    template <typename SrcType, int HEAD_DIM>
    __device__ __forceinline__ void store_head_convert(
        int64_t elem_off,
        const SrcType* src,
        int lane
    ) {
        constexpr int BLOCK_SIZE = 32;
        constexpr int NUM_BLOCKS = HEAD_DIM / BLOCK_SIZE;
        
        int elem_size = ArenaFormat::float_elem_size(format);
        char* dst = base + elem_off * elem_size;
        
        #pragma unroll
        for (int b = 0; b < NUM_BLOCKS; ++b) {
            store_block_convert<SrcType, BLOCK_SIZE>(
                dst + b * BLOCK_SIZE * elem_size,
                src + b * BLOCK_SIZE,
                format,
                lane
            );
        }
    }
};

