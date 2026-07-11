#pragma once
// ============================================================================
// Arena KV store helpers — shared by the FP16 prefill kernel and the INT8
// prefix-attention prefill kernel. Both write a prefill's fresh tokens into
// the writer chunks through these (unrotated K, R16 Q-capture preserved).
// Requires ../arena_table.cuh and ../convert/convert_all.cuh (ArenaFormat,
// to_f32 / from_f32, type_to_arena_format) to be included first.
// ============================================================================

#include <stdint.h>

/// Store a single float to a dtype arena element.
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

/// Store 8 elements from a source row to the arena with format conversion
/// (per-head-palette resolved). Only supports dtype formats for V; K may be
/// R16 (with Q-capture co-store) or a dtype format. Quant formats are
/// produced by the separate seal-time quantization pass, never here.
/// q_src: Q projection for R16 Q-capture (ignored when k_fmt != R16).
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
