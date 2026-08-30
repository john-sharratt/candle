#pragma once

// KO weight quantize: F32 [nrows × ncols] (row-major) → the lane-major per-128 KO
// chunk tensor the int8 KO matmul reads. Byte-identical to the CPU reference
// `quantize_ko` in candle-core/src/quantized/ko_quant.rs — same affine `(scale, min)`
// per 128-K-per-row, same lane-major pack, so a GPU-vs-CPU byte compare must match.
//
// One warp per 1024-weight chunk (8 rows × 128 K). lane = r*4 + q3 with r = lane>>2
// (row in the group, 0..7) and q3 = lane&3 (which quarter, 0..3). Each lane owns 32
// weights of its row — for sub∈0..4 the "lo" run w[sub*32 + q3*4 + 0..3] and the "hi"
// run w[sub*32 + 16 + q3*4 + 0..3] — and the 4 q3-lanes of a row cooperate via a
// width-4 shfl butterfly for the per-row min/max (or amax for Q8).
//
// Bandwidth-bound, so memory traffic is fully vectorized: each lane's two 4-element
// runs are 16-byte-aligned float4 loads, and its four ql bytes per sub are one aligned
// uint32 store (the low-traffic Q5 hi / Q6 crumb streams stay scalar).

#include "../blocks.cuh"
#include <cuda_fp16.h>
#include <stdint.h>

// The affine KO widths that carry a 512 B `ql` plane: Q4_KO <15,0,0>, Q5_KO <31,0,128>,
// Q6_KO <63,256,0>. CRUMB_BYTES carries bits 4-5 (Q6), HI_BYTES carries bit 4 (Q5); both zero
// for Q4. Q2_KO and Q3_KO have NO `ql` plane — their value starts at the crumb, so their bit
// shifts differ and they have their own kernels below rather than instantiating this template.
// The full plane stack and the shift rule live in `ko_quant::KoPlanes` (Rust), which is the
// authority both sides are byte-checked against.
template <int MAXQ, int CRUMB_BYTES, int HI_BYTES>
__global__ void quantize_ko_affine_kernel(
    const float* __restrict__ w, uint8_t* __restrict__ ob, int nrows, int ncols)
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

    for (int chunk = (int)(((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
         chunk < total_chunks; chunk += total_warps) {
        const int k_blk = chunk / row_groups;
        const int g = chunk % row_groups;
        const int64_t wbase = (int64_t)(g * 8 + r) * ncols + (int64_t)k_blk * 128;
        const int cbase = chunk * CHUNK_BYTES;

        float4 vlo[4], vhi[4]; // [sub], 4 contiguous weights each
        float mn = INFINITY, mx = -INFINITY;
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            // Plain (cached) loads: the affine formats are read-heavy (small output), so
            // keeping the input in L2 — adjacent chunks share row cache lines — beats the
            // evict-first __ldcs, which measurably regressed Q4 here.
            float4 a = *(const float4*)(w + wbase + sub * 32 + q3 * 4);
            float4 b = *(const float4*)(w + wbase + sub * 32 + 16 + q3 * 4);
            vlo[sub] = a;
            vhi[sub] = b;
            mn = fminf(mn, fminf(fminf(a.x, a.y), fminf(a.z, a.w)));
            mn = fminf(mn, fminf(fminf(b.x, b.y), fminf(b.z, b.w)));
            mx = fmaxf(mx, fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)));
            mx = fmaxf(mx, fmaxf(fmaxf(b.x, b.y), fmaxf(b.z, b.w)));
        }
        // Width-4 butterfly over the row's 4 q3-lanes (consecutive lanes r*4+0..3).
        mn = fminf(mn, __shfl_xor_sync(0xffffffff, mn, 2));
        mn = fminf(mn, __shfl_xor_sync(0xffffffff, mn, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        const float scale = fmaxf((mx - mn) / (float)MAXQ, 1e-12f);

        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            // CPU uses (w - mn) / scale (division, not reciprocal-mul) — match it.
            const float4 a = vlo[sub], b = vhi[sub];
            int ql0 = min(max((int)roundf((a.x - mn) / scale), 0), MAXQ);
            int ql1 = min(max((int)roundf((a.y - mn) / scale), 0), MAXQ);
            int ql2 = min(max((int)roundf((a.z - mn) / scale), 0), MAXQ);
            int ql3 = min(max((int)roundf((a.w - mn) / scale), 0), MAXQ);
            int qh0 = min(max((int)roundf((b.x - mn) / scale), 0), MAXQ);
            int qh1 = min(max((int)roundf((b.y - mn) / scale), 0), MAXQ);
            int qh2 = min(max((int)roundf((b.z - mn) / scale), 0), MAXQ);
            int qh3 = min(max((int)roundf((b.w - mn) / scale), 0), MAXQ);
            // ql: low nibble = lo quant, high nibble = hi quant; 4 bytes → one uint32.
            uint32_t packed =
                  ((uint32_t)((ql0 & 0xF) | ((qh0 & 0xF) << 4)))
                | ((uint32_t)((ql1 & 0xF) | ((qh1 & 0xF) << 4)) << 8)
                | ((uint32_t)((ql2 & 0xF) | ((qh2 & 0xF) << 4)) << 16)
                | ((uint32_t)((ql3 & 0xF) | ((qh3 & 0xF) << 4)) << 24);
            *(uint32_t*)(ob + cbase + lane * 16 + sub * 4) = packed;

            if (CRUMB_BYTES > 0) { // Q6: bits 4-5 (2 bits) of each quant, 4 packed per byte.
                uint8_t cr0 = (uint8_t)(((ql0 >> 4) & 0x3) | (((ql1 >> 4) & 0x3) << 2)
                                        | (((ql2 >> 4) & 0x3) << 4) | (((ql3 >> 4) & 0x3) << 6));
                uint8_t cr1 = (uint8_t)(((qh0 >> 4) & 0x3) | (((qh1 >> 4) & 0x3) << 2)
                                        | (((qh2 >> 4) & 0x3) << 4) | (((qh3 >> 4) & 0x3) << 6));
                ob[cbase + CRUMB_BASE + lane * 8 + sub * 2] = cr0;
                ob[cbase + CRUMB_BASE + lane * 8 + sub * 2 + 1] = cr1;
            }
            if (HI_BYTES > 0) { // Q5: bit 4 (1 bit) of each quant, lo nibble + hi nibble.
                uint8_t hb0 = (uint8_t)(((ql0 >> 4) & 1) | (((ql1 >> 4) & 1) << 1)
                                        | (((ql2 >> 4) & 1) << 2) | (((ql3 >> 4) & 1) << 3));
                uint8_t hb1 = (uint8_t)(((qh0 >> 4) & 1) | (((qh1 >> 4) & 1) << 1)
                                        | (((qh2 >> 4) & 1) << 2) | (((qh3 >> 4) & 1) << 3));
                ob[cbase + HI_BASE + lane * 4 + sub] = (uint8_t)(hb0 | (hb1 << 4));
            }
        }
        if (q3 == 0) { // one (scale, min) per row at dm[r].
            *(half2*)(ob + cbase + DM_BASE + r * 4) =
                __halves2half2(__float2half_rn(scale), __float2half_rn(mn));
        }
    }
}

// Q8_KO: symmetric 8-bit (min = 0). b_frag[0] region [0,512) holds the lo quants,
// b_frag[1] region [512,1024) the hi quants, both at lane*16 + sub*4 + i. dm = (scale, 0).
__global__ void quantize_q8_ko_kernel(
    const float* __restrict__ w, uint8_t* __restrict__ ob, int nrows, int ncols)
{
    const int CHUNK_BYTES = 1024 + 32;
    const int k_blocks = ncols / 128;
    const int row_groups = nrows / 8;
    const int total_chunks = k_blocks * row_groups;
    const int total_warps = (gridDim.x * blockDim.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int r = lane >> 2;
    const int q3 = lane & 3;

    for (int chunk = (int)(((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
         chunk < total_chunks; chunk += total_warps) {
        const int k_blk = chunk / row_groups;
        const int g = chunk % row_groups;
        const int64_t wbase = (int64_t)(g * 8 + r) * ncols + (int64_t)k_blk * 128;
        const int cbase = chunk * CHUNK_BYTES;

        float4 vlo[4], vhi[4];
        float amax = 0.f;
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            // Q8_KO writes the most output (full 1024B/chunk), so it is write-bound:
            // streaming (evict-first) loads free L2 for the heavy stores and measurably
            // help here — the opposite trade-off from the read-heavy affine kernel.
            float4 a = __ldcs((const float4*)(w + wbase + sub * 32 + q3 * 4));
            float4 b = __ldcs((const float4*)(w + wbase + sub * 32 + 16 + q3 * 4));
            vlo[sub] = a;
            vhi[sub] = b;
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(a.z), fabsf(a.w))));
            amax = fmaxf(amax, fmaxf(fmaxf(fabsf(b.x), fabsf(b.y)), fmaxf(fabsf(b.z), fabsf(b.w))));
        }
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 2));
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, 1));
        const float scale = fmaxf(amax / 127.0f, 1e-12f);
        const float id = 1.0f / scale; // CPU uses id = 1/scale then w*id — match it.

        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            const float4 a = vlo[sub], b = vhi[sub];
            uchar4 q0, q1;
            q0.x = (uint8_t)(int8_t)min(max((int)roundf(a.x * id), -127), 127);
            q0.y = (uint8_t)(int8_t)min(max((int)roundf(a.y * id), -127), 127);
            q0.z = (uint8_t)(int8_t)min(max((int)roundf(a.z * id), -127), 127);
            q0.w = (uint8_t)(int8_t)min(max((int)roundf(a.w * id), -127), 127);
            q1.x = (uint8_t)(int8_t)min(max((int)roundf(b.x * id), -127), 127);
            q1.y = (uint8_t)(int8_t)min(max((int)roundf(b.y * id), -127), 127);
            q1.z = (uint8_t)(int8_t)min(max((int)roundf(b.z * id), -127), 127);
            q1.w = (uint8_t)(int8_t)min(max((int)roundf(b.w * id), -127), 127);
            *(uchar4*)(ob + cbase + lane * 16 + sub * 4) = q0;
            *(uchar4*)(ob + cbase + 512 + lane * 16 + sub * 4) = q1;
        }
        if (q3 == 0) {
            *(half2*)(ob + cbase + 1024 + r * 4) =
                __halves2half2(__float2half_rn(scale), __float2half_rn(0.0f));
        }
    }
}

// Q2_KO: 2-bit affine (per-128 scale, min). value 0..3 stored as crumbs — cr0 (4 low-half
// values) / cr1 (4 high-half) per (lane, sub) at lane*8 + sub*2, each crumb at bit 2j. 288 B
// chunk (256 crumb + 32 dm). Byte-identical to the CPU reference `quantize_q2_ko`. Same
// butterfly min/max over the row's 4 q3-lanes as the affine kernel; no 512 B ql region.
__global__ void quantize_q2_ko_kernel(
    const float* __restrict__ w, uint8_t* __restrict__ ob, int nrows, int ncols)
{
    const int DM_BASE = 256;
    const int CHUNK_BYTES = DM_BASE + 32; // 288
    const int k_blocks = ncols / 128;
    const int row_groups = nrows / 8;
    const int total_chunks = k_blocks * row_groups;
    const int total_warps = (gridDim.x * blockDim.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int r = lane >> 2;
    const int q3 = lane & 3;

    for (int chunk = (int)(((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
         chunk < total_chunks; chunk += total_warps) {
        const int k_blk = chunk / row_groups;
        const int g = chunk % row_groups;
        const int64_t wbase = (int64_t)(g * 8 + r) * ncols + (int64_t)k_blk * 128;
        const int cbase = chunk * CHUNK_BYTES;

        float4 vlo[4], vhi[4];
        float mn = INFINITY, mx = -INFINITY;
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            float4 a = *(const float4*)(w + wbase + sub * 32 + q3 * 4);
            float4 b = *(const float4*)(w + wbase + sub * 32 + 16 + q3 * 4);
            vlo[sub] = a;
            vhi[sub] = b;
            mn = fminf(mn, fminf(fminf(a.x, a.y), fminf(a.z, a.w)));
            mn = fminf(mn, fminf(fminf(b.x, b.y), fminf(b.z, b.w)));
            mx = fmaxf(mx, fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)));
            mx = fmaxf(mx, fmaxf(fmaxf(b.x, b.y), fmaxf(b.z, b.w)));
        }
        mn = fminf(mn, __shfl_xor_sync(0xffffffff, mn, 2));
        mn = fminf(mn, __shfl_xor_sync(0xffffffff, mn, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        const float scale = fmaxf((mx - mn) / 3.0f, 1e-12f);

        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            // CPU uses (w - mn) / scale (division, not reciprocal-mul) — match it.
            const float4 a = vlo[sub], b = vhi[sub];
            int q0 = min(max((int)roundf((a.x - mn) / scale), 0), 3);
            int q1 = min(max((int)roundf((a.y - mn) / scale), 0), 3);
            int q2 = min(max((int)roundf((a.z - mn) / scale), 0), 3);
            int q3v = min(max((int)roundf((a.w - mn) / scale), 0), 3);
            int h0 = min(max((int)roundf((b.x - mn) / scale), 0), 3);
            int h1 = min(max((int)roundf((b.y - mn) / scale), 0), 3);
            int h2 = min(max((int)roundf((b.z - mn) / scale), 0), 3);
            int h3 = min(max((int)roundf((b.w - mn) / scale), 0), 3);
            uint8_t cr0 = (uint8_t)((q0 & 3) | ((q1 & 3) << 2) | ((q2 & 3) << 4) | ((q3v & 3) << 6));
            uint8_t cr1 = (uint8_t)((h0 & 3) | ((h1 & 3) << 2) | ((h2 & 3) << 4) | ((h3 & 3) << 6));
            ob[cbase + lane * 8 + sub * 2] = cr0;
            ob[cbase + lane * 8 + sub * 2 + 1] = cr1;
        }
        if (q3 == 0) { // one (scale, min) per row at dm[r].
            *(half2*)(ob + cbase + DM_BASE + r * 4) =
                __halves2half2(__float2half_rn(scale), __float2half_rn(mn));
        }
    }
}

// Q3_KO: 3-bit affine (per-128 scale, min), value 0..7. Two planes and no ql — bits 0-1 in the
// 256 B crumb region at `lane*8 + sub*2` (byte-identical to Q2_KO's) and bit 2 in the 128 B hi
// region at `256 + lane*4 + sub` (byte-identical to Q5_KO's 5th-bit region: low nibble = the 4
// low-half values, high nibble = the 4 high-half). 416 B chunk (256 crumb + 128 hi + 32 dm).
// Byte-identical to CPU `ko_quant::quantize_ko(.., Q3_KO)`.
__global__ void quantize_q3_ko_kernel(
    const float* __restrict__ w, uint8_t* __restrict__ ob, int nrows, int ncols)
{
    const int HI_BASE = 256;
    const int DM_BASE = 384;
    const int CHUNK_BYTES = DM_BASE + 32; // 416
    const int k_blocks = ncols / 128;
    const int row_groups = nrows / 8;
    const int total_chunks = k_blocks * row_groups;
    const int total_warps = (gridDim.x * blockDim.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int r = lane >> 2;
    const int q3 = lane & 3;

    for (int chunk = (int)(((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5);
         chunk < total_chunks; chunk += total_warps) {
        const int k_blk = chunk / row_groups;
        const int g = chunk % row_groups;
        const int64_t wbase = (int64_t)(g * 8 + r) * ncols + (int64_t)k_blk * 128;
        const int cbase = chunk * CHUNK_BYTES;

        float4 vlo[4], vhi[4];
        float mn = INFINITY, mx = -INFINITY;
        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            float4 a = *(const float4*)(w + wbase + sub * 32 + q3 * 4);
            float4 b = *(const float4*)(w + wbase + sub * 32 + 16 + q3 * 4);
            vlo[sub] = a;
            vhi[sub] = b;
            mn = fminf(mn, fminf(fminf(a.x, a.y), fminf(a.z, a.w)));
            mn = fminf(mn, fminf(fminf(b.x, b.y), fminf(b.z, b.w)));
            mx = fmaxf(mx, fmaxf(fmaxf(a.x, a.y), fmaxf(a.z, a.w)));
            mx = fmaxf(mx, fmaxf(fmaxf(b.x, b.y), fmaxf(b.z, b.w)));
        }
        mn = fminf(mn, __shfl_xor_sync(0xffffffff, mn, 2));
        mn = fminf(mn, __shfl_xor_sync(0xffffffff, mn, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 2));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffff, mx, 1));
        const float scale = fmaxf((mx - mn) / 7.0f, 1e-12f);

        #pragma unroll
        for (int sub = 0; sub < 4; ++sub) {
            // CPU uses (w - mn) / scale (division, not reciprocal-mul) — match it.
            const float4 a = vlo[sub], b = vhi[sub];
            int q[4], h[4];
            q[0] = min(max((int)roundf((a.x - mn) / scale), 0), 7);
            q[1] = min(max((int)roundf((a.y - mn) / scale), 0), 7);
            q[2] = min(max((int)roundf((a.z - mn) / scale), 0), 7);
            q[3] = min(max((int)roundf((a.w - mn) / scale), 0), 7);
            h[0] = min(max((int)roundf((b.x - mn) / scale), 0), 7);
            h[1] = min(max((int)roundf((b.y - mn) / scale), 0), 7);
            h[2] = min(max((int)roundf((b.z - mn) / scale), 0), 7);
            h[3] = min(max((int)roundf((b.w - mn) / scale), 0), 7);
            uint8_t cr0 = 0, cr1 = 0, hb0 = 0, hb1 = 0;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                cr0 |= (uint8_t)((q[j] & 3) << (2 * j));
                cr1 |= (uint8_t)((h[j] & 3) << (2 * j));
                hb0 |= (uint8_t)(((q[j] >> 2) & 1) << j);
                hb1 |= (uint8_t)(((h[j] >> 2) & 1) << j);
            }
            ob[cbase + lane * 8 + sub * 2] = cr0;
            ob[cbase + lane * 8 + sub * 2 + 1] = cr1;
            ob[cbase + HI_BASE + lane * 4 + sub] = (uint8_t)(hb0 | (hb1 << 4));
        }
        if (q3 == 0) { // one (scale, min) per row at dm[r].
            *(half2*)(ob + cbase + DM_BASE + r * 4) =
                __halves2half2(__float2half_rn(scale), __float2half_rn(mn));
        }
    }
}
