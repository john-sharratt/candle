#pragma once

// q8a128 dequant: block_q8a128 → f32/f16/bf16 (vectorized bulk path).
//
// Decode formula: x[i] = ds[0].x · qs[i] — per-128, one scale per tile. This is
// the vectorized bulk form of the per-element trait
// BlockConverter<block_q8a128, T>::load_element (scale divisor 1) in
// convert/block_q8a128.cuh — the same formula, expressed once scalar (trait,
// used by the attention/convert gather) and once vectorized (here, used for
// whole-tensor dequant), exactly as the q8_1 twin splits dequant_q8_1.cuh vs
// convert/block_q8_1.cuh. q8a128_dequant_exact pins the two byte-identical.
//
// One warp per 128-tile, lane t owns 4 contiguous elements [t*4, t*4+4); all 32
// lanes share the tile's single scale ds[0].x. qs loaded as one char4, output
// written as one 16/8-byte vector store.

#include "../blocks.cuh"
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Store 4 floats as 4 contiguous elements of T via a single 16/8-byte vector store.
template <typename T>
__device__ __forceinline__ void q8a128_store4(T* p, float a, float b, float c, float d);

template <>
__device__ __forceinline__ void q8a128_store4<float>(float* p, float a, float b, float c, float d) {
    *reinterpret_cast<float4*>(p) = make_float4(a, b, c, d);
}
template <>
__device__ __forceinline__ void q8a128_store4<__half>(__half* p, float a, float b, float c, float d) {
    __half2* h = reinterpret_cast<__half2*>(p);
    h[0] = __floats2half2_rn(a, b);
    h[1] = __floats2half2_rn(c, d);
}
template <>
__device__ __forceinline__ void q8a128_store4<__nv_bfloat16>(__nv_bfloat16* p, float a, float b, float c, float d) {
    __nv_bfloat162* h = reinterpret_cast<__nv_bfloat162*>(p);
    h[0] = __floats2bfloat162_rn(a, b);
    h[1] = __floats2bfloat162_rn(c, d);
}

template <typename T>
__global__ void dequantize_q8a128_kernel(
    const block_q8a128* __restrict__ in, T* __restrict__ out, int rows, int cols)
{
    const int total_tiles = (int)(((int64_t)rows * cols) / 128);
    const int total_warps = (gridDim.x * blockDim.x) >> 5;
    const int warp = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane = threadIdx.x & 31;

    const uint8_t* ibytes = reinterpret_cast<const uint8_t*>(in);
    for (int tile = warp; tile < total_tiles; tile += total_warps) {
        // q8a1024 flat-grouped: qs and ds de-interleaved into the tile's slot. One scale per 128.
        const half2* ds = reinterpret_cast<const half2*>(ibytes + q8a1024_ds_off(tile));
        const float scale = __half2float(ds[0].x);
        const char4 q = *reinterpret_cast<const char4*>(ibytes + q8a1024_qs_off(tile) + lane * 4);
        const int64_t base = (int64_t)tile * 128 + (int64_t)lane * 4;
        q8a128_store4<T>(out + base, scale * q.x, scale * q.y, scale * q.z, scale * q.w);
    }
}
