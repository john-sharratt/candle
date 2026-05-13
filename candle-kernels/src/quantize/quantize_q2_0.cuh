// SPDX-License-Identifier: MIT
// Q2_0 Quantization: float -> 2-bit symmetric with per-block scale
//
// Q2_0 format stores:
//   - d (half): scale factor, d = amax / 1.5
//   - qs[8] (uint8_t): packed 2-bit values, 4 quants per byte
//
// Block size: 32 elements packed into 8 bytes + 2 byte scale = 10 bytes
// Compression: 2.5 bits per element (vs 32 bits for float)
//
// Quantization levels: 0..3, centered at 1.5
//   decode: x = d * (q - 1.5)   (q=0 → -1.5d, q=1 → -0.5d, q=2 → +0.5d, q=3 → +1.5d)
//   encode: q = clamp(round(x * (1.5/amax) + 1.5), 0, 3)
//
// NOTE: Include via quantize.cuh which defines warp reduce functions.

#pragma once

// =============================================================================
// OPTIMIZED SINGLE-BLOCK QUANTIZATION (32 elements, vectorized)
// =============================================================================
// 8 threads load float4 each (8 x 4 = 32 floats). Each thread encodes 4 quants
// into a single byte using 2-bit packing.

__device__ __forceinline__ void quantize_block_q2_0_vec(
    const float* __restrict__ src,
    block_q2_0* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;

    float4 v;
    float local_max = 0.0f;

    if (lane < 8) {
        v = reinterpret_cast<const float4*>(src)[lane];
        local_max = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                         fmaxf(fabsf(v.z), fabsf(v.w)));
    }

    // Reduce max across first 8 lanes, then broadcast to all 32
    float amax = local_max;
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1)
        amax = fmaxf(amax, __shfl_xor_sync(0xff, amax, offset, 8));
    amax = __shfl_sync(0xffffffff, amax, 0, 32);

    if (lane < 8) {
        const float id = (amax != 0.0f) ? 1.5f / amax : 0.0f;

        // Encode 4 floats into one 2-bit-packed byte
        auto encode = [&](float x) -> uint8_t {
            return (uint8_t)fminf(3.0f, fmaxf(0.0f, roundf(x * id + 1.5f)));
        };

        uint8_t q0 = encode(v.x);
        uint8_t q1 = encode(v.y);
        uint8_t q2 = encode(v.z);
        uint8_t q3 = encode(v.w);

        // Pack 4 x 2-bit values into one byte
        dst->qs[lane] = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6);
    }

    if (lane == 0)
        dst->d = __float2half_rn(amax / 1.5f);
}

// =============================================================================
// SCALAR FALLBACK (one element per thread, full-warp reduction)
// Used by transpose_batch.cuh for fused transpose+quantize.
// =============================================================================

__device__ __forceinline__ void quantize_block_q2_0(
    const float* __restrict__ src,
    block_q2_0* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const float xi = src[lane];

    float amax = fabsf(xi);
    amax = quantize_warp_reduce_max(amax);

    const float d = amax / 1.5f;
    const float id = (amax != 0.0f) ? 1.5f / amax : 0.0f;

    // q = clamp(round(xi * id + 1.5), 0, 3)
    const uint8_t q2 = (uint8_t)fminf(3.0f, fmaxf(0.0f, roundf(xi * id + 1.5f)));

    // Pack 4 lanes' 2-bit values into one byte using warp shuffle.
    // Group of 4 lanes (lane/4) → byte lane/4.
    // Each lane places its quant at bit position (lane%4)*2.
    // Use XOR-reduce within group of 4 to collect all 4 contributions.
    uint8_t packed = (q2 & 3) << (2 * (lane & 3));
    // Two XOR-shuffle steps to OR together 4 contributions:
    packed |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)packed, 1, 32);
    packed |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)packed, 2, 32);

    // Only the first lane in each group of 4 writes its byte
    if ((lane & 3) == 0) {
        dst->qs[lane >> 2] = packed;
    }

    if (lane == 0) {
        dst->d = __float2half_rn(d);
    }
}

// =============================================================================
// MULTI-BLOCK QUANTIZATION (grid-stride, vectorized)
// =============================================================================

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q2_0(
    const float* __restrict__ src,
    block_q2_0* __restrict__ dst,
    int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    for (int blk = warp_id + blockIdx.x * warps_per_block;
         blk < num_blocks;
         blk += warps_per_block * gridDim.x) {

        const float* block_src = src + blk * QK2_0;
        block_q2_0*  block_dst = dst + blk;

        float4 v;
        float local_max = 0.0f;

        if (lane < 8) {
            v = reinterpret_cast<const float4*>(block_src)[lane];
            local_max = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)),
                             fmaxf(fabsf(v.z), fabsf(v.w)));
        }

        float amax = local_max;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1)
            amax = fmaxf(amax, __shfl_xor_sync(0xff, amax, offset, 8));
        amax = __shfl_sync(0xffffffff, amax, 0, 32);

        if (lane < 8) {
            const float id = (amax != 0.0f) ? 1.5f / amax : 0.0f;

            auto encode = [&](float x) -> uint8_t {
                return (uint8_t)fminf(3.0f, fmaxf(0.0f, roundf(x * id + 1.5f)));
            };

            block_dst->qs[lane] = encode(v.x) | (encode(v.y) << 2)
                                              | (encode(v.z) << 4)
                                              | (encode(v.w) << 6);
        }

        if (lane == 0)
            block_dst->d = __float2half_rn(amax / 1.5f);
    }
}
