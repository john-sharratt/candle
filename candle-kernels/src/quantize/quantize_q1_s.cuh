// SPDX-License-Identifier: MIT
// Q1_S Quantization: float -> 1-bit symmetric with INT8 scale
// Block: { int8_t scale; uint8_t qs[4]; } = 5 bytes = 1.25 BPE
// scale = round(mean(|x|) * 127). Each element: 1 bit (sign), reconstructed as sign ? +d : -d
// where d = scale / 127.0. Outer palette scale provides range; no FP8 used.
#pragma once

__device__ __forceinline__ void quantize_block_q1_s_vec(
    const float* __restrict__ src,
    block_q1_s* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;

    float4 v;
    float local_sum = 0.0f;
    if (lane < 8) {
        v = reinterpret_cast<const float4*>(src)[lane];
        local_sum = fabsf(v.x) + fabsf(v.y) + fabsf(v.z) + fabsf(v.w);
    }

    float sum_abs = local_sum;
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1)
        sum_abs += __shfl_xor_sync(0xff, sum_abs, offset, 8);
    const float mean_abs = __shfl_sync(0xffffffff, sum_abs, 0, 32) / 32.0f;

    if (lane < 8) {
        uint8_t bits = 0;
        bits |= (v.x >= 0.0f ? 1u : 0u) << 0;
        bits |= (v.y >= 0.0f ? 1u : 0u) << 1;
        bits |= (v.z >= 0.0f ? 1u : 0u) << 2;
        bits |= (v.w >= 0.0f ? 1u : 0u) << 3;
        uint8_t partner_bits = (uint8_t)__shfl_xor_sync(0xff, (int)bits, 1, 8);
        if ((lane & 1) == 0) {
            dst->qs[lane >> 1] = (bits & 0xF) | ((partner_bits & 0xF) << 4);
        }
    }

    if (lane == 0)
        dst->scale = (int8_t)__float2int_rn(fminf(127.0f, mean_abs * 127.0f));
}

__device__ __forceinline__ void quantize_block_q1_s(
    const float* __restrict__ src,
    block_q1_s* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const float xi = src[lane];

    float sum_abs = fabsf(xi);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_abs += __shfl_xor_sync(0xffffffff, sum_abs, offset, 32);
    const float mean_abs = sum_abs / 32.0f;

    const uint8_t sign_bit = (xi >= 0.0f) ? 1u : 0u;
    uint8_t packed = sign_bit << (lane & 7);
    #pragma unroll
    for (int offset = 1; offset < 8; offset <<= 1)
        packed |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)packed, offset, 32);

    if ((lane & 7) == 0)
        dst->qs[lane >> 3] = packed;

    if (lane == 0)
        dst->scale = (int8_t)__float2int_rn(fminf(127.0f, mean_abs * 127.0f));
}

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q1_s(
    const float* __restrict__ src,
    block_q1_s* __restrict__ dst,
    int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    for (int blk = warp_id + blockIdx.x * warps_per_block;
         blk < num_blocks;
         blk += warps_per_block * gridDim.x) {

        const float* block_src = src + blk * QK1_S;
        block_q1_s* block_dst = dst + blk;

        const float xi = block_src[lane];

        float sum_abs = fabsf(xi);
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            sum_abs += __shfl_xor_sync(0xffffffff, sum_abs, offset, 32);
        const float mean_abs = sum_abs / 32.0f;

        const uint8_t sign_bit = (xi >= 0.0f) ? 1u : 0u;
        uint8_t packed = sign_bit << (lane & 7);
        #pragma unroll
        for (int offset = 1; offset < 8; offset <<= 1)
            packed |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)packed, offset, 32);

        if ((lane & 7) == 0)
            block_dst->qs[lane >> 3] = packed;

        if (lane == 0)
            block_dst->scale = (int8_t)__float2int_rn(fminf(127.0f, mean_abs * 127.0f));
    }
}
