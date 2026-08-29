// =============================================================================
// prov_sign_pack — GPU sign(Q) bit-pack for the wide-Q provenance signature
// =============================================================================
//
// PURPOSE
// -------
// The provenance signature stored per sealed token is `sign(Q)` folded across
// layers. The old path D2H'd the full R16 K/Q/V for every layer (48 separate
// blocking `memcpy_dtov`s per scope), expanded f16→f32 on the CPU, then signed
// and folded — moving megabytes to produce a few hundred bytes.
//
// This kernel does the read + sign + bit-pack entirely on the GPU, for every
// (layer, block, head, palette) sub-band in ONE launch. It emits one u32 of
// sign bits per (sub-band, token). The host then only D2H's those packed bits
// (16× smaller than the f16 gather) and XOR-folds them — a few KB, a few ms.
//
// GRID / BLOCK
// ------------
//   Grid:  (n_warps, 1, 1)   n_warps = n_layers × n_r16_blocks × n_kv_head × N_PALETTE
//   Block: (CHUNK_SIZE, 1, 1) = 32 threads, one per token → one hardware warp.
//
// INPUT
// -----
// q_ptrs[warp]: byte address of the R16 K chunk for this (layer,block,head,palette)
//   triple — Q is co-located in that chunk (block_r16 layout, see gather_r16_kv):
//     group d spans bytes [d*128, d*128+128): K in [0,64), Q in [64,128).
//     Q[token][d] = q_ptr + d*128 + 64 + token*2   (f16)
//   For fixed d, consecutive tokens (threads) are 2 bytes apart → coalesced read.
//
// OUTPUT
// ------
// out[warp * CHUNK_SIZE + token]: u64 with bit d set iff Q[token][d] >= 0, for
//   d in 0..sub_head_dim. `sub_head_dim = head_dim / N_PALETTE` must be <= 64
//   (one u64 per palette sub-band); the host packs palette p's bits into global
//   head-dims [p*sub_head_dim, (p+1)*sub_head_dim).
//
// The word is 64 bits because that is the width of a PHYSICAL band, which is
// what lets a 256-wide head take this path at all. `N_PALETTE` describes the R16
// arena layout — at head_dim 256 a head is stored as 4 bands of 64 dims — so the
// band cannot be narrowed to fit a u32 without re-banding the arena. Packing 8
// bands of 32 instead reads only the low half of each physical band and lays
// band p at p*32 rather than p*64: half of every signature dropped and the rest
// dim-permuted. Widening the word keeps the packing bit-identical to the CPU
// fold. A head_dim 128 band (32 dims) leaves the upper 32 bits clear, and the
// host's `bit_off = p * sub_head_dim` placement is already general over both.
//
// =============================================================================

#include "blocks.cuh"

// Sub-bands (warps) handled per CUDA block. The kernel is DRAM-bandwidth bound,
// so this mainly trims launch/grid overhead and lifts occupancy; block =
// (CHUNK_SIZE, WARPS_PER_BLOCK) threads, each row a token, each column a warp.
#define PROV_WARPS_PER_BLOCK 8

__global__ void prov_sign_pack_kernel(
    const int64_t* __restrict__ q_ptrs,
    uint64_t* __restrict__ out,
    int n_warps,
    int sub_head_dim
) {
    const int warp  = blockIdx.x * blockDim.y + threadIdx.y;
    if (warp >= n_warps) return;
    const int token = threadIdx.x;  // 0 .. CHUNK_SIZE-1
    const int64_t base = q_ptrs[warp];

    uint64_t bits = 0ull;
    for (int d = 0; d < sub_head_dim; d++) {
        // Q lives at +64 within each 128-byte dim group; token*2 → coalesced.
        const __half q = *(const __half*)(base + (int64_t)d * 128 + 64 + (int64_t)token * 2);
        // Bit set iff value >= 0 — identical convention to WideQSig::from_band
        // (`band[i] >= 0.0`), which correctly handles -0.0 (bit set) and NaN
        // (bit clear); the f16→f32 cast is exact for the comparison.
        if (__half2float(q) >= 0.0f) {
            bits |= (1ull << d);
        }
    }
    out[warp * CHUNK_SIZE + token] = bits;
}

extern "C" void run_prov_sign_pack(
    const int64_t* q_ptrs,
    void*          out,
    int            n_warps,
    int            sub_head_dim,
    cudaStream_t   stream
) {
    // sub_head_dim > 64 can't pack into a u64 — caller falls back to the CPU path.
    if (n_warps <= 0 || sub_head_dim <= 0 || sub_head_dim > 64) return;
    const int wpb = PROV_WARPS_PER_BLOCK;
    const int grid = (n_warps + wpb - 1) / wpb;
    dim3 block(CHUNK_SIZE, wpb);
    prov_sign_pack_kernel<<<grid, block, 0, stream>>>(
        q_ptrs, (uint64_t*)out, n_warps, sub_head_dim
    );
}
