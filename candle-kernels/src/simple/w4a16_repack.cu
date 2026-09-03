// =============================================================================
// W4A16 → Q4_KO REPACK — fused, batched byte permutation
// =============================================================================
// Converts compressed-tensors `pack-quantized` int4 experts (8 nibbles per
// int32 along K, +8 offset already applied; one bf16 scale per 128) into the
// lane-major Q4_KO chunk layout the int8 KO matmul reads. Pure data movement —
// no weight arithmetic — so the import stays bit-exact by construction: the
// nibbles are re-grouped, never re-rounded, and the per-group (scale, min) is
// stored as (f16(s), f16(−8·s)).
//
// Batched over every expert of a tensor in ONE launch (grid.y = expert,
// grid.x = chunk). Each thread emits one aligned u32 of the chunk's 512-byte
// ql plane: its four output bytes share (lane, sub), so their four low-nibble
// sources are four CONSECUTIVE columns — one 16-bit quad of a single input
// word — and likewise the high nibbles from +16 columns. Two u32 loads, a
// nibble spread, one u32 store; threads 0..7 additionally emit the chunk's
// eight (scale, min) f16 pairs.
//
// A bf16 scale outside the f16-exact range would make the store lossy; the
// kernel counts such groups into `violations` (atomicAdd) and the host
// refuses the conversion when the counter is nonzero — the same contract the
// CPU importer enforces by bailing per group.
//
// Layout agreement with `ko_quant::pack_q4_ko` (the CPU reference) is pinned
// by the harness's byte-identity gate (`convert_bench`), not assumed.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Spread a 16-bit quad of nibbles n0..n3 into the low nibble of each byte of
// a u32 (little-endian byte order).
static __device__ __forceinline__ uint32_t spread_nibbles(uint32_t q) {
    return (q & 0x000Fu) | ((q & 0x00F0u) << 4) | ((q & 0x0F00u) << 8) | ((q & 0xF000u) << 12);
}

extern "C" __global__ void w4a16_repack_q4ko_kernel(
    const uint32_t* __restrict__ words,  // [n_experts, nrows*ncols/8]
    const uint16_t* __restrict__ scales, // [n_experts, nrows*ncols/128] bf16 bits
    uint8_t* __restrict__ out,           // [n_experts, chunks*544]
    int* __restrict__ violations,        // f16-exactness failures (see header)
    int nrows,
    int ncols)
{
    const int e = blockIdx.y;
    const int chunk = blockIdx.x; // = k_blk * row_groups + g, the CPU pack's order
    const int row_groups = nrows >> 3;
    const int k_blk = chunk / row_groups;
    const int g = chunk % row_groups;

    const uint32_t* w = words + (size_t)e * ((size_t)nrows * ncols / 8);
    uint8_t* oc = out + ((size_t)e * gridDim.x + chunk) * 544;

    // Thread t emits output u32 t: bytes j = 4t..4t+3 share lane = t/4 and
    // sub = t%4 (since lane = j/16, sub = (j%16)/4).
    const int t = threadIdx.x;
    const int lane = t >> 2;
    const int sub = t & 3;
    const int r = lane >> 2;
    const int q3 = lane & 3;
    const int row = (g << 3) + r;
    const int col_lo = (k_blk << 7) + (sub << 5) + (q3 << 2); // 4-aligned

    const size_t elo = (size_t)row * ncols + col_lo;
    const uint32_t lq = (w[elo >> 3] >> ((elo & 7) * 4)) & 0xFFFFu;
    const size_t ehi = elo + 16;
    const uint32_t hq = (w[ehi >> 3] >> ((ehi & 7) * 4)) & 0xFFFFu;
    reinterpret_cast<uint32_t*>(oc)[t] = spread_nibbles(lq) | (spread_nibbles(hq) << 4);

    if (t < 8) {
        const int row2 = (g << 3) + t;
        const uint16_t sb = scales[(size_t)e * ((size_t)nrows * (ncols >> 7))
                                   + (size_t)row2 * (ncols >> 7) + k_blk];
        const float s = __uint_as_float(((uint32_t)sb) << 16); // bf16 → f32, exact
        const __half hs = __float2half_rn(s);
        if (__half2float(hs) != s) atomicAdd(violations, 1);
        const __half hm = __float2half_rn(-8.0f * s);
        const uint32_t dm = (uint32_t)__half_as_ushort(hs)
                          | ((uint32_t)__half_as_ushort(hm) << 16);
        reinterpret_cast<uint32_t*>(oc + 512)[t] = dm;
    }
}

extern "C" void run_w4a16_repack_q4ko(
    const uint32_t* words,
    const uint16_t* scales,
    uint8_t* out,
    int* violations,
    int n_experts,
    int nrows,
    int ncols,
    void* stream)
{
    if (n_experts <= 0) return;
    const int chunks = (nrows >> 3) * (ncols >> 7);
    dim3 grid(chunks, n_experts);
    w4a16_repack_q4ko_kernel<<<grid, 128, 0, (cudaStream_t)stream>>>(
        words, scales, out, violations, nrows, ncols);
}
