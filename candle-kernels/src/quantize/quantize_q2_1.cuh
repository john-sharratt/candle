// SPDX-License-Identifier: MIT
// Q2_1 Quantization: float -> 2-bit asymmetric with F16 scale + F16 min
// Block: { half2 dm; uint8_t qs[8]; } = 12 bytes = 3.00 BPE
// Like Q4_1 but 2-bit: q * d + m, q in [0,3]
#pragma once

__device__ __forceinline__ void quantize_block_q2_1_vec(
    const float* __restrict__ src,
    block_q2_1* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;

    float4 v;
    float local_max = -3.402823466e+38f;
    float local_min = 3.402823466e+38f;
    if (lane < 8) {
        v = reinterpret_cast<const float4*>(src)[lane];
        local_max = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
        local_min = fminf(fminf(v.x, v.y), fminf(v.z, v.w));
    }

    float vmax = local_max, vmin = local_min;
    #pragma unroll
    for (int offset = 4; offset > 0; offset >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(0xff, vmax, offset, 8));
        vmin = fminf(vmin, __shfl_xor_sync(0xff, vmin, offset, 8));
    }
    vmax = __shfl_sync(0xffffffff, vmax, 0, 32);
    vmin = __shfl_sync(0xffffffff, vmin, 0, 32);

    const float d = (vmax - vmin) / 3.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

    if (lane < 8) {
        auto encode = [&](float x) -> uint8_t {
            return (uint8_t)fminf(3.0f, fmaxf(0.0f, roundf((x - vmin) * id)));
        };
        dst->qs[lane] = encode(v.x) | (encode(v.y) << 2)
                        | (encode(v.z) << 4) | (encode(v.w) << 6);
    }
    if (lane == 0)
        dst->dm = __halves2half2(__float2half_rn(d), __float2half_rn(vmin));
}

__device__ __forceinline__ void quantize_block_q2_1(
    const float* __restrict__ src,
    block_q2_1* __restrict__ dst)
{
    const int lane = threadIdx.x % WARP_SIZE;
    const float xi = src[lane];

    float vmax = xi, vmin = xi;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        vmax = fmaxf(vmax, __shfl_xor_sync(0xffffffff, vmax, offset, 32));
        vmin = fminf(vmin, __shfl_xor_sync(0xffffffff, vmin, offset, 32));
    }

    const float d = (vmax - vmin) / 3.0f;
    const float id = (d != 0.0f) ? 1.0f / d : 0.0f;
    const uint8_t q2 = (uint8_t)fminf(3.0f, fmaxf(0.0f, roundf((xi - vmin) * id)));

    uint8_t packed = (q2 & 3) << (2 * (lane & 3));
    packed |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)packed, 1, 32);
    packed |= (uint8_t)__shfl_xor_sync(0xffffffff, (int)packed, 2, 32);

    if ((lane & 3) == 0)
        dst->qs[lane >> 2] = packed;

    if (lane == 0)
        dst->dm = __halves2half2(__float2half_rn(d), __float2half_rn(vmin));
}

template <int BLOCKS_PER_WARP = 1>
__device__ __forceinline__ void quantize_blocks_q2_1(
    const float* __restrict__ src,
    block_q2_1* __restrict__ dst,
    int num_blocks)
{
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warps_per_block = blockDim.x / WARP_SIZE;

    for (int blk = warp_id + blockIdx.x * warps_per_block;
         blk < num_blocks;
         blk += warps_per_block * gridDim.x) {

        const float* block_src = src + blk * QK2_1;
        block_q2_1* block_dst = dst + blk;

        float4 v;
        float local_max = -3.402823466e+38f, local_min = 3.402823466e+38f;
        if (lane < 8) {
            v = reinterpret_cast<const float4*>(block_src)[lane];
            local_max = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
            local_min = fminf(fminf(v.x, v.y), fminf(v.z, v.w));
        }
        float vmax = local_max, vmin = local_min;
        #pragma unroll
        for (int offset = 4; offset > 0; offset >>= 1) {
            vmax = fmaxf(vmax, __shfl_xor_sync(0xff, vmax, offset, 8));
            vmin = fminf(vmin, __shfl_xor_sync(0xff, vmin, offset, 8));
        }
        vmax = __shfl_sync(0xffffffff, vmax, 0, 32);
        vmin = __shfl_sync(0xffffffff, vmin, 0, 32);

        const float d = (vmax - vmin) / 3.0f;
        const float id = (d != 0.0f) ? 1.0f / d : 0.0f;

        if (lane < 8) {
            auto encode = [&](float x) -> uint8_t {
                return (uint8_t)fminf(3.0f, fmaxf(0.0f, roundf((x - vmin) * id)));
            };
            block_dst->qs[lane] = encode(v.x) | (encode(v.y) << 2)
                                  | (encode(v.z) << 4) | (encode(v.w) << 6);
        }
        if (lane == 0)
            block_dst->dm = __halves2half2(__float2half_rn(d), __float2half_rn(vmin));
    }
}
