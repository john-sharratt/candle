// SPDX-License-Identifier: MIT
// Q1_A Quantization: float -> 1-bit asymmetric with separate amplitude per sign
// Block: { int8_t scale_pos; int8_t scale_neg; uint8_t qs[4]; } = 6 bytes = 1.50 BPE
// scale_pos = round(mean(x_i for x_i >= 0) * 127)
// scale_neg = round(mean(|x_i| for x_i <  0) * 127)
// Each element: 1 sign bit. Reconstruction: sign ? +scale_pos/127 : -scale_neg/127.
// Outer palette scale provides range; no FP8 used.
//
// Structure mirrors quantize_block_q1_s but with two amplitudes and a
// __ballot_sync-driven sign mask (matches the Q0_M4 ballot path).
#pragma once

__device__ __forceinline__ void quantize_block_q1_a_core(
    float xi,
    uint32_t qmask,
    float sum_pos,
    float sum_neg,
    block_q1_a* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;

    if (lane == 0) {
        const int n_pos = __popc(qmask);
        const int n_neg = 32 - n_pos;
        const float mean_pos = (n_pos > 0) ? (sum_pos / (float)n_pos) : 0.0f;
        const float mean_neg = (n_neg > 0) ? (sum_neg / (float)n_neg) : 0.0f;

        const int sp = __float2int_rn(mean_pos * 127.0f);
        const int sn = __float2int_rn(mean_neg * 127.0f);
        dst->scale_pos = (int8_t)max(0, min(127, sp));
        dst->scale_neg = (int8_t)max(0, min(127, sn));

        // 32 sign bits → 4 bytes. Bit set = positive (or zero).
        dst->qs[0] = (uint8_t)( qmask        & 0xFF);
        dst->qs[1] = (uint8_t)((qmask >>  8) & 0xFF);
        dst->qs[2] = (uint8_t)((qmask >> 16) & 0xFF);
        dst->qs[3] = (uint8_t)((qmask >> 24) & 0xFF);
    }
}

__device__ __forceinline__ void quantize_block_q1_a_vec(
    const float* __restrict__ src,
    block_q1_a* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const float xi = src[lane];

    const bool is_pos = (xi >= 0.0f);
    const uint32_t qmask = __ballot_sync(0xffffffff, is_pos);
    const float sum_pos = q0_warp_sum(is_pos ? xi  : 0.0f);
    const float sum_neg = q0_warp_sum(is_pos ? 0.0f : -xi);

    quantize_block_q1_a_core(xi, qmask, sum_pos, sum_neg, dst);
}

__device__ __forceinline__ void quantize_block_q1_a(
    const float* __restrict__ src,
    block_q1_a* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const float xi = src[lane];

    const bool is_pos = (xi >= 0.0f);
    const uint32_t qmask = __ballot_sync(0xffffffff, is_pos);
    const float sum_pos = q0_warp_sum(is_pos ? xi  : 0.0f);
    const float sum_neg = q0_warp_sum(is_pos ? 0.0f : -xi);

    quantize_block_q1_a_core(xi, qmask, sum_pos, sum_neg, dst);
}

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q1_a(
    const float* __restrict__ src,
    block_q1_a* __restrict__ dst,
    int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane    = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    for (int blk = warp_id + blockIdx.x * warps_per_block;
         blk < num_blocks;
         blk += warps_per_block * gridDim.x) {

        const float xi = src[blk * QK1_A + lane];

        const bool is_pos = (xi >= 0.0f);
        const uint32_t qmask = __ballot_sync(0xffffffff, is_pos);
        const float sum_pos = q0_warp_sum(is_pos ? xi  : 0.0f);
        const float sum_neg = q0_warp_sum(is_pos ? 0.0f : -xi);

        quantize_block_q1_a_core(xi, qmask, sum_pos, sum_neg, dst + blk);
    }
}
