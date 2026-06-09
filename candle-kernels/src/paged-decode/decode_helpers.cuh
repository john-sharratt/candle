#pragma once
// ============================================================================
// decode_helpers.cuh - shared decode helpers.
//   vec2 load, cp.async commit/wait, RoPE (rope_cos_sin / apply_rope_*),
//   arena scatter (write_regs_to_arena / write_regs_to_r16), and the
//   write-length commit kernel. Shared by the INT8 decode kernel and the
//   legacy V2 paged-decode kernel.
// ============================================================================

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>
#include <type_traits>

#include "../fast_exp.cuh"
#include "../arena_table.cuh"
#include "../simple/warp_reduce.cuh"
#include "../convert/convert_all.cuh"
#include "slot_types.cuh"           // Slot buffer byte-layout accessors
#include "pal_iter.cuh"             // PalIter — palette-aware dimension iterator

// ============================================================================
// Local helper surface for v2 decode (previously pulled from the legacy header)
// ============================================================================

template <typename T> struct vec2_traits;

template <> struct vec2_traits<__half> {
    using vec_type = __half2;
    static __device__ __forceinline__ float2 to_float2(vec_type v) { return __half22float2(v); }
};

template <> struct vec2_traits<__nv_bfloat16> {
    using vec_type = __nv_bfloat162;
    static __device__ __forceinline__ float2 to_float2(vec_type v) { return __bfloat1622float2(v); }
};

template <> struct vec2_traits<float> {
    using vec_type = float2;
    static __device__ __forceinline__ float2 to_float2(vec_type v) { return v; }
};

template <> struct vec2_traits<__nv_fp8_e4m3> {
    static __device__ __forceinline__ float2 to_float2(const __nv_fp8_e4m3* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 890)
        __nv_fp8_storage_t sa = *reinterpret_cast<const __nv_fp8_storage_t*>(&p[0]);
        __nv_fp8_storage_t sb = *reinterpret_cast<const __nv_fp8_storage_t*>(&p[1]);
        return make_float2(
            __half2float(__nv_cvt_fp8_to_halfraw(sa, __NV_E4M3)),
            __half2float(__nv_cvt_fp8_to_halfraw(sb, __NV_E4M3))
        );
#else
        return make_float2(to_f32(p[0]), to_f32(p[1]));
#endif
    }
};

template <typename T>
__device__ __forceinline__ float2 load_vec2(const T* ptr) {
    if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        return vec2_traits<__nv_fp8_e4m3>::to_float2(ptr);
    } else {
        using traits = vec2_traits<T>;
        return traits::to_float2(*reinterpret_cast<const typename traits::vec_type*>(ptr));
    }
}

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

template <bool USE_TC>
__device__ __forceinline__ void cp_async_commit() {
    if constexpr (USE_TC) {
        asm volatile("cp.async.commit_group;" ::);
    }
}

template <int HEAD_DIM>
__device__ __forceinline__ void rope_cos_sin(
    int pos, int d_idx, const float* __restrict__ rope_cs, float& cos_v, float& sin_v
) {
    const float* entry = rope_cs + (int64_t)pos * HEAD_DIM + d_idx * 2;
    cos_v = __ldg(entry);
    sin_v = __ldg(entry + 1);
}

template <int VEC, int HEAD_DIM>
__device__ __forceinline__ void apply_rope_rotary_f32(float* regs, int lane, int pos, const float* __restrict__ rope_cs) {
    const int pair_lane = lane ^ 16;
    float pair_regs[VEC];
    #pragma unroll
    for (int j = 0; j < VEC; ++j) pair_regs[j] = __shfl_sync(0xffffffff, regs[j], pair_lane);
    const float sign = (lane & 16) ? 1.f : -1.f;
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
        float cos_v, sin_v;
        rope_cos_sin<HEAD_DIM>(pos, (lane & 15) * VEC + j, rope_cs, cos_v, sin_v);
        regs[j] = regs[j] * cos_v + sign * pair_regs[j] * sin_v;
    }
}

template <int VEC, int HEAD_DIM>
__device__ __forceinline__ void apply_rope_interleaved_f32(float* regs, int lane, int pos, const float* __restrict__ rope_cs) {
    const int base_idx = lane * VEC;
    if constexpr (VEC == 1) {
        float cos_v, sin_v;
        rope_cos_sin<HEAD_DIM>(pos, lane / 2, rope_cs, cos_v, sin_v);
        float partner = __shfl_sync(0xffffffff, regs[0], lane ^ 1);
        const float sign = (lane & 1) ? 1.f : -1.f;
        regs[0] = regs[0] * cos_v + sign * partner * sin_v;
    } else {
        static_assert(VEC % 2 == 0,
            "Interleaved RoPE requires even VEC (HEAD_DIM must be 32 or a multiple of 64)");
        #pragma unroll
        for (int j = 0; j < VEC; j += 2) {
            int pair_idx = (base_idx + j) / 2;
            float cos_v, sin_v;
            rope_cos_sin<HEAD_DIM>(pos, pair_idx, rope_cs, cos_v, sin_v);
            float x = regs[j], y = regs[j + 1];
            regs[j]     = x * cos_v - y * sin_v;
            regs[j + 1] = x * sin_v + y * cos_v;
        }
    }
}

template <int VEC>
__device__ __forceinline__ void write_regs_to_arena(
    char* arena_base, int64_t elem_offset_base, int lane, int elem_size,
    int arena_fmt, const float* regs
) {
    char* dst = arena_base + (elem_offset_base + (int64_t)lane * VEC) * elem_size;
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
        char* p = dst + j * elem_size;
        if (arena_fmt == ArenaFormat::F32) {
            *reinterpret_cast<float*>(p) = regs[j];
        } else if (arena_fmt == ArenaFormat::F16) {
            *reinterpret_cast<__half*>(p) = __float2half(regs[j]);
        } else if (arena_fmt == ArenaFormat::BF16) {
            *reinterpret_cast<__nv_bfloat16*>(p) = __float2bfloat16(regs[j]);
        } else if (arena_fmt == ArenaFormat::F8E4M3) {
            *reinterpret_cast<__nv_fp8_e4m3*>(p) = __nv_fp8_e4m3(regs[j]);
        }
    }
}

template <int VEC>
__device__ __forceinline__ void write_regs_to_r16(
    char* arena_base, int64_t chunk_byte_offset, int within_chunk, int lane,
    const float* k_regs, const float* q_regs
) {
    #pragma unroll
    for (int j = 0; j < VEC; ++j) {
        int dim = lane * VEC + j;
        char* blk_base = arena_base + chunk_byte_offset + (int64_t)dim * 128;
        *reinterpret_cast<__half*>(blk_base + within_chunk * 2) = __float2half(k_regs[j]);
        *reinterpret_cast<__half*>(blk_base + 64 + within_chunk * 2) = __float2half(q_regs[j]);
    }
}

template <int HEAD_DIM>
__global__ void commit_decode_write_len_kernel(
    const uint8_t* headers_ptr,
    int num_active_slots,
    int n_kv_head
) {
    int slot_idx = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (slot_idx >= num_active_slots) return;

    const SlotHeader& slot = get_slot_header(headers_ptr, slot_idx);
    if (slot.n_slices == 0 || slot.write_slice >= slot.n_slices) return;

    uint8_t* write_slice_ptr =
        get_slice_mut<HEAD_DIM>(slot.slices_ptr, (int)slot.write_slice, n_kv_head);
    const uint16_t ws_offset = slice_offset(write_slice_ptr);
    const uint16_t ws_len = slice_len(write_slice_ptr);

#ifndef NDEBUG
    assert((int)ws_offset >= 0 && (int)ws_offset <= CHUNK_SIZE);
    assert((int)ws_len >= 0 && (int)ws_len <= CHUNK_SIZE);
#endif

    if ((int)ws_offset + (int)ws_len < CHUNK_SIZE) {
        slice_increment_len(write_slice_ptr);
    }
}

// ============================================================================
// Minimum blocks per SM based on warp count and register pressure.
// WARPS=8  (256 threads): target 3 blocks/SM for good occupancy.
// WARPS=16 (512 threads): target 2 blocks/SM.
