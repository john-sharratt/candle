#pragma once

// KO weight dequant: the lane-major per-128 KO chunk tensor → f32 [nrows × ncols]
// (row-major). Inverse of quantize_ko.cuh, byte-identical to the CPU reference
// `dequant_ko` in candle-core/src/quantized/ko_quant.rs: W = scale·q + min, with the
// quant `q` reassembled from the ql nibble plus the Q5 hi-bit / Q6 crumb streams.
//
// Mirror of the quantize warp/lane mapping: one warp per 1024-weight chunk, lane =
// r*4 + q3; lane (r, q3) reconstructs its 32 weights — lo run out[sub*32 + q3*4 + 0..3],
// hi run out[sub*32 + 16 + q3*4 + 0..3] — and scatters them into the row-major output.
// scale·q + min is forced to a rounded mul + rounded add (__fmul_rn/__fadd_rn) so the
// result matches the CPU's separate mul/add bit-for-bit (no fma contraction).
//
// Bandwidth-bound: the dominant traffic is the f32 output, written as 16-byte-aligned
// float4 stores; the ql run is read as one aligned uint32 (the Q5 hi / Q6 crumb streams
// stay scalar).

#include "../blocks.cuh"
#include <cuda_fp16.h>
#include <stdint.h>

template <int CRUMB_BYTES, int HI_BYTES>
__global__ void dequantize_ko_affine_kernel(
    const uint8_t* __restrict__ chunk, float* __restrict__ out, int nrows, int ncols)
{
    const int CRUMB_BASE = 512;
    const int HI_BASE = 512 + CRUMB_BYTES;
    const int DM_BASE = 512 + CRUMB_BYTES + HI_BYTES;
    const int CHUNK_BYTES = DM_BASE + 32;
    const int k_blocks = ncols / 128;
    const int row_groups = nrows / 8;
    const int total_chunks = k_blocks * row_groups;
    const int total_warps = (gridDim.x * blockDim.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int r = lane >> 2;
    const int q3 = lane & 3;

    for (int chunk_idx = (int)(((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
         chunk_idx < total_chunks; chunk_idx += total_warps) {
        const int k_blk = chunk_idx / row_groups;
        const int g = chunk_idx % row_groups;
        const int64_t obase = (int64_t)(g * 8 + r) * ncols + (int64_t)k_blk * 128;
        const int cbase = chunk_idx * CHUNK_BYTES;

        const half2 dm = *(const half2*)(chunk + cbase + DM_BASE + r * 4);
        const float scale = __half2float(dm.x);
        const float mn = __half2float(dm.y);

        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const uint32_t ql4 = *(const uint32_t*)(chunk + cbase + lane * 16 + sub * 4);
            uint32_t cr0 = 0, cr1 = 0, hb = 0;
            if (CRUMB_BYTES > 0) {
                const int off = cbase + CRUMB_BASE + lane * 8 + sub * 2;
                cr0 = chunk[off];
                cr1 = chunk[off + 1];
            }
            if (HI_BYTES > 0) {
                hb = chunk[cbase + HI_BASE + lane * 4 + sub];
            }
            float lo[4], hi[4];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const uint32_t qlb = (ql4 >> (8 * i)) & 0xFF;
                uint32_t qlo = qlb & 0xF;
                uint32_t qhi = (qlb >> 4) & 0xF;
                if (CRUMB_BYTES > 0) {
                    qlo |= ((cr0 >> (2 * i)) & 0x3) << 4;
                    qhi |= ((cr1 >> (2 * i)) & 0x3) << 4;
                }
                if (HI_BYTES > 0) {
                    qlo |= (((hb & 0xF) >> i) & 1) << 4;
                    qhi |= ((((hb >> 4) & 0xF) >> i) & 1) << 4;
                }
                lo[i] = __fadd_rn(__fmul_rn(scale, (float)qlo), mn);
                hi[i] = __fadd_rn(__fmul_rn(scale, (float)qhi), mn);
            }
            // Streaming stores: the f32 output is write-once, never re-read here.
            __stcs((float4*)(out + obase + sub * 32 + q3 * 4), make_float4(lo[0], lo[1], lo[2], lo[3]));
            __stcs((float4*)(out + obase + sub * 32 + 16 + q3 * 4), make_float4(hi[0], hi[1], hi[2], hi[3]));
        }
    }
}

// Q8_KO: symmetric, min = 0. lo quant in b_frag[0] [0,512), hi quant in b_frag[1] [512,1024).
__global__ void dequantize_q8_ko_kernel(
    const uint8_t* __restrict__ chunk, float* __restrict__ out, int nrows, int ncols)
{
    const int CHUNK_BYTES = 1024 + 32;
    const int k_blocks = ncols / 128;
    const int row_groups = nrows / 8;
    const int total_chunks = k_blocks * row_groups;
    const int total_warps = (gridDim.x * blockDim.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int r = lane >> 2;
    const int q3 = lane & 3;

    for (int chunk_idx = (int)(((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
         chunk_idx < total_chunks; chunk_idx += total_warps) {
        const int k_blk = chunk_idx / row_groups;
        const int g = chunk_idx % row_groups;
        const int64_t obase = (int64_t)(g * 8 + r) * ncols + (int64_t)k_blk * 128;
        const int cbase = chunk_idx * CHUNK_BYTES;
        const float scale = __half2float(*(const half*)(chunk + cbase + 1024 + r * 4));

        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const uchar4 b0 = *(const uchar4*)(chunk + cbase + lane * 16 + sub * 4);
            const uchar4 b1 = *(const uchar4*)(chunk + cbase + 512 + lane * 16 + sub * 4);
            float4 lo, hi;
            lo.x = __fmul_rn(scale, (float)(int8_t)b0.x);
            lo.y = __fmul_rn(scale, (float)(int8_t)b0.y);
            lo.z = __fmul_rn(scale, (float)(int8_t)b0.z);
            lo.w = __fmul_rn(scale, (float)(int8_t)b0.w);
            hi.x = __fmul_rn(scale, (float)(int8_t)b1.x);
            hi.y = __fmul_rn(scale, (float)(int8_t)b1.y);
            hi.z = __fmul_rn(scale, (float)(int8_t)b1.z);
            hi.w = __fmul_rn(scale, (float)(int8_t)b1.w);
            __stcs((float4*)(out + obase + sub * 32 + q3 * 4), lo);
            __stcs((float4*)(out + obase + sub * 32 + 16 + q3 * 4), hi);
        }
    }
}
