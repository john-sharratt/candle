// SPDX-License-Identifier: MIT
// R16 Quantization: float -> F16 with reserved Q-capture space
//
// R16 format: [ half d[32] | uint16_t q[32] ] = 128 bytes
//   d[i] = float_to_half(xi)   (lossless F16 conversion)
//   q[i] = 0                   (reserved for Q capture)
//
// This is a "pseudo-quantization" that simply converts F32 to F16 and
// zero-fills the Q space. It's designed for use as a high-fidelity
// KV cache format that can later capture query values.
//
// NOTE: Include via quantize.cuh which defines warp reduce functions.

#pragma once

// =============================================================================
// SINGLE-BLOCK QUANTIZATION (32 elements, one per thread)
// =============================================================================

__device__ __forceinline__ void quantize_block_r16(
    const float* __restrict__ src,
    block_r16* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    dst->d[lane] = __float2half_rn(src[lane]);
    dst->q[lane] = 0;  // Zero-fill Q space
}

// =============================================================================
// MULTI-BLOCK QUANTIZATION (grid-stride)
// =============================================================================

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_r16(
    const float* __restrict__ src,
    block_r16* __restrict__ dst,
    int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    for (int blk = warp_id + blockIdx.x * warps_per_block;
         blk < num_blocks;
         blk += warps_per_block * gridDim.x)
    {
        dst[blk].d[lane] = __float2half_rn(src[blk * QK_R16 + lane]);
        dst[blk].q[lane] = 0;
    }
}
