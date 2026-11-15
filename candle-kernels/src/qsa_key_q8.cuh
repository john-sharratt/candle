// =============================================================================
// QSA index keys in Q8_0 — the device definition
// =============================================================================
// A completed index key is stored as Q8_0 along its channels: blocks of
// QSA_Q8_BLOCK channels, an int8 code a channel and one f16 scale a block
// (`docs/qsa_index_q8.md` §3).
//
//     amax = max|x| over the block
//     d    = f16( amax · (1/127) )
//     id   = (d == 0) ? 0 : 127 / amax
//     q[c] = int8( rint( x[c] · id ) ),  clamped to ±127
//     x̂[c] = q[c] · f32(d)
//
// A block whose scale rounds to f16 zero stores all-zero codes. The host
// definition is `qwen4exp/key_q8.rs`; this one runs under the archive's fast
// math and is gated within one code of it — a key is quantized once and read
// back by the scale that was stored, so nothing reproduces these bytes.
//
// On the card a chunk of keys is two planes: the codes, then the scales
// row-major as `[rows, nb]` with `nb = ⌈d / 32⌉`.

#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#define QSA_Q8_BLOCK 32

/// Scales a `d`-wide key carries.
__host__ __device__ __forceinline__ int qsa_q8_blocks(int d)
{
    return (d + QSA_Q8_BLOCK - 1) / QSA_Q8_BLOCK;
}

/// A block's stored scale and the multiplier its codes are made with.
struct QsaQ8Quant {
    unsigned short d;
    float id;
};

__device__ __forceinline__ QsaQ8Quant qsa_q8_quant(float amax)
{
    QsaQ8Quant q;
    q.d = __half_as_ushort(__float2half_rn(amax * (1.0f / 127.0f)));
    q.id = (q.d & 0x7fff) != 0 ? 127.0f / amax : 0.0f;
    return q;
}

__device__ __forceinline__ signed char qsa_q8_code(float x, float id)
{
    return (signed char)max(-127, min(127, __float2int_rn(x * id)));
}

/// A stored scale as a float.
__device__ __forceinline__ float qsa_q8_scale(unsigned short bits)
{
    return __half2float(__ushort_as_half(bits));
}

/// Four consecutive codes of one word, dequantized.
__device__ __forceinline__ float4 qsa_q8_word(int w, float scale)
{
    float4 v;
    v.x = (float)(signed char)(w) * scale;
    v.y = (float)(signed char)(w >> 8) * scale;
    v.z = (float)(signed char)(w >> 16) * scale;
    v.w = (float)(w >> 24) * scale;
    return v;
}
