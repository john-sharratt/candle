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
// ds[0].y is the activation sum used by the INT8 matmul's affine min correction
// (unused for plain dequant). It is stored **normalised by amax** — Σx/amax, not
// Σx — and the matmul rebuilds Σx as `ds.y * ds.x * 127`. See "the sum is
// normalised" in blocks.cuh for why. Its f16 value is invariant to the reduction
// order (order differences are ~100× below the f16 ULP).

#include "../blocks.cuh"
#include "q8a128_tile.cuh"
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

// `sum_norm`: 0 stores the raw Σx, 1 stores Σx/amax (see blocks.cuh). A RUNTIME
// argument, uniform across the grid and read by one lane per 128-tile, so it
// costs a predicated select on a store that happens once per tile — and, unlike
// a template parameter, it leaves the kernel count where it was.
template <typename T>
__global__ void quantize_q8a128_kernel(
    const T* __restrict__ act, block_q8a128* __restrict__ out, int rows, int cols,
    int sum_norm)
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
        emit_q8a128_tile(obytes, tile, lane, x0, x1, x2, x3, sum_norm);
    }
}
