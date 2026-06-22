#pragma once

// q8a128 quantize: typed activations (f16/bf16/f32) → block_q8a128.
//
// block_q8a128 is the contiguous q8 ACTIVATION twin of the q8_1 weight block,
// per-128: 128 elements share ONE {scale, sum}, stored at ds[0]. Layout is
// half2 ds[4] (ds[0] = the tile's {scale, sum}; ds[1..3] are 16-byte-alignment
// pad) + a 16-byte-aligned int8 qs[128] run for wide cp.async. The int8 matmul
// folds one activation scale per 128-K MMA accumulation, so per-128 is the
// granularity the kernel produces.
//
// This is a bandwidth-bound streaming kernel, so it is fully vectorized:
//   - ONE warp per 128-tile; lane t owns 4 contiguous elements [t*4, t*4+4).
//     The whole warp (32 lanes × 4 elems) is one 128-element group, so a
//     full-width `shfl_xor` (5 butterfly steps) reduces amax/Σx across the tile.
//   - 16-byte vector loads (float4 / 2×half2) — naturally aligned.
//   - one char4 (int32) store of the 4 quants instead of 4 byte writes.
// Σx (ds[0].y) is the raw activation sum used by the INT8 matmul's Q4_K min
// correction; it is unused for plain dequant. Its f16 value is invariant to the
// reduction order (order differences are ~100× below the f16 ULP at these sums).

#include "../blocks.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Load 4 contiguous elements of T as floats via a single 16/8-byte vector load.
template <typename T>
__device__ __forceinline__ void q8a128_load4(const T* p, float& a, float& b, float& c, float& d);

template <>
__device__ __forceinline__ void q8a128_load4<float>(const float* p, float& a, float& b, float& c, float& d) {
    const float4 v = *reinterpret_cast<const float4*>(p);
    a = v.x; b = v.y; c = v.z; d = v.w;
}
template <>
__device__ __forceinline__ void q8a128_load4<__half>(const __half* p, float& a, float& b, float& c, float& d) {
    const __half2* h = reinterpret_cast<const __half2*>(p);
    const float2 lo = __half22float2(h[0]);
    const float2 hi = __half22float2(h[1]);
    a = lo.x; b = lo.y; c = hi.x; d = hi.y;
}
template <>
__device__ __forceinline__ void q8a128_load4<__nv_bfloat16>(const __nv_bfloat16* p, float& a, float& b, float& c, float& d) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(p);
    const float2 lo = __bfloat1622float2(h[0]);
    const float2 hi = __bfloat1622float2(h[1]);
    a = lo.x; b = lo.y; c = hi.x; d = hi.y;
}

template <typename T>
__global__ void quantize_q8a128_kernel(
    const T* __restrict__ act, block_q8a128* __restrict__ out, int rows, int cols)
{
    const int total_tiles = (int)(((int64_t)rows * cols) / 128);
    const int total_warps = (gridDim.x * blockDim.x) >> 5;
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;
    // per-128: the whole warp (32 lanes × 4 elems) is ONE 128-element group with one scale.

    uint8_t* obytes = reinterpret_cast<uint8_t*>(out);
    for (int tile = warp; tile < total_tiles; tile += total_warps) {
        const int64_t base = (int64_t)tile * 128 + (int64_t)lane * 4;
        float x0, x1, x2, x3;
        q8a128_load4<T>(act + base, x0, x1, x2, x3);

        float amax = fmaxf(fmaxf(fabsf(x0), fabsf(x1)), fmaxf(fabsf(x2), fabsf(x3)));
        float s = x0 + x1 + x2 + x3;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, 32));
            s += __shfl_xor_sync(0xffffffff, s, off, 32);
        }
        const float id = (amax != 0.f) ? 127.f / amax : 0.f;

        // q8a1024 flat-grouped placement: qs and ds de-interleaved into the tile's
        // super-block slot (see blocks.cuh). Same quant math as the old AoS block.
        *reinterpret_cast<char4*>(obytes + q8a1024_qs_off(tile) + lane * 4) = make_char4(
            (int8_t)__float2int_rn(x0 * id),
            (int8_t)__float2int_rn(x1 * id),
            (int8_t)__float2int_rn(x2 * id),
            (int8_t)__float2int_rn(x3 * id));
        if (lane == 0) {
            // One (scale, sum) per 128-element tile (per-128). Stored at the tile slot's first half2.
            half2* ds = reinterpret_cast<half2*>(obytes + q8a1024_ds_off(tile));
            ds[0] = make_half2(__float2half_rn(amax / 127.f), __float2half_rn(s));
        }
    }
}
