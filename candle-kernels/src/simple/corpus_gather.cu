// =============================================================================
// corpus_gather.cu — fused gather of the compressed-corpus HOT two-region cache.
//
// The decode wave selects a per-session top-k of compressed entries, then must
// assemble the selected rows of the gallery's hot cache — `nope_i8`
// `[len, 448]` u8, `nope_scale` `[len, 14]` f32, `rope_bf` `[len, 64]` bf16, and
// `pos` `[len]` u32 — into ONE contiguous block the attention kernel walks. The
// eager path did four `index_select`s per session plus four `cat`s across
// sessions; this gathers all four regions for a session's `k` gids in a SINGLE
// launch, writing straight into the shared output at the session's row offset,
// so the wave neither re-lists per region nor concatenates.
//
// One block per gathered row (`grid.x = k`); the block's threads copy that
// row's four regions as 32-bit words (all region widths are 4-byte multiples:
// 448/56/128/4 bytes).
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Region widths in 32-bit words (NOPE_DIM=448 u8, NOPE_BANDS=14 f32,
// ROPE_DIM=64 bf16, pos=1 u32).
#define NOPE_W 112 // 448 bytes / 4
#define SCALE_W 14 // 14 f32
#define ROPE_W 32  // 64 bf16 / 2

extern "C" __global__ void corpus_gather_rows_kernel(
    const uint32_t* __restrict__ nope_i8,    // [len, 112] (u8[448] as u32)
    const uint32_t* __restrict__ nope_scale, // [len, 14]  (f32)
    const uint32_t* __restrict__ rope_bf,    // [len, 32]  (bf16[64] as u32)
    const uint32_t* __restrict__ pos,        // [len]
    const uint32_t* __restrict__ gids,       // [k]
    uint32_t* __restrict__ out_nope,         // [total, 112]
    uint32_t* __restrict__ out_scale,        // [total, 14]
    uint32_t* __restrict__ out_rope,         // [total, 32]
    uint32_t* __restrict__ out_pos,          // [total]
    int k,
    int row_offset
) {
    int j = blockIdx.x;
    if (j >= k) return;
    uint32_t gid = gids[j];
    int dst = row_offset + j;
    int t = threadIdx.x;
    int nt = blockDim.x;

    const uint32_t* sn = nope_i8 + (int64_t)gid * NOPE_W;
    uint32_t* dn = out_nope + (int64_t)dst * NOPE_W;
    for (int w = t; w < NOPE_W; w += nt) dn[w] = sn[w];

    const uint32_t* ss = nope_scale + (int64_t)gid * SCALE_W;
    uint32_t* ds = out_scale + (int64_t)dst * SCALE_W;
    for (int w = t; w < SCALE_W; w += nt) ds[w] = ss[w];

    const uint32_t* sr = rope_bf + (int64_t)gid * ROPE_W;
    uint32_t* dr = out_rope + (int64_t)dst * ROPE_W;
    for (int w = t; w < ROPE_W; w += nt) dr[w] = sr[w];

    if (t == 0) out_pos[dst] = pos[gid];
}

// Batched across all HOT decode sessions in ONE launch: session `h` gathers its
// `cnt` rows from its own gallery (region base addresses packed in `ptrs`:
// `[nope|scale|rope|pos]`, each `n_hot`) at `gids[gid_start + j]`, writing to
// `out_*[out_off + j]` (`gid_start`/`out_off`/`cnt` packed in `meta`, each
// `n_hot`). `ROWS_PER_BLOCK` warps per block, one warp per gathered row, each
// region `uint4`-vectorized (widest = NOPE, 448 B = 28×`uint4`). Replaces the
// per-session launch loop (0.39 waves/SM); the packed metadata + a host-cached
// pointer table keep the launch's host cost O(1).
#define ROWS_PER_BLOCK 4

extern "C" __global__ void corpus_gather_rows_batched_kernel(
    const uint64_t* __restrict__ ptrs, // [5*n_hot] nope|scale|rope|pos|gids base addrs
    const uint32_t* __restrict__ meta, // [2*n_hot] out_off|cnt
    uint32_t* __restrict__ out_nope,   // [total, 112]
    uint32_t* __restrict__ out_scale,  // [total, 14]
    uint32_t* __restrict__ out_rope,   // [total, 32]
    uint32_t* __restrict__ out_pos,    // [total]
    int n_hot
) {
    int h = blockIdx.y;
    int warp = threadIdx.x >> 5;   // 0..ROWS_PER_BLOCK-1
    int lane = threadIdx.x & 31;
    int j = blockIdx.x * ROWS_PER_BLOCK + warp; // row within session
    if (j >= (int)meta[1 * n_hot + h]) return;

    uint32_t gid = ((const uint32_t*)ptrs[4 * n_hot + h])[j];
    int dst = (int)meta[0 * n_hot + h] + j;

    const uint4* sn = (const uint4*)((const uint32_t*)ptrs[0 * n_hot + h] + (int64_t)gid * NOPE_W);
    uint4* dn = (uint4*)(out_nope + (int64_t)dst * NOPE_W);
    for (int w = lane; w < NOPE_W / 4; w += 32) dn[w] = sn[w];

    const uint32_t* ss = (const uint32_t*)ptrs[1 * n_hot + h] + (int64_t)gid * SCALE_W;
    uint32_t* ds = out_scale + (int64_t)dst * SCALE_W;
    for (int w = lane; w < SCALE_W; w += 32) ds[w] = ss[w];

    const uint4* sr = (const uint4*)((const uint32_t*)ptrs[2 * n_hot + h] + (int64_t)gid * ROPE_W);
    uint4* dr = (uint4*)(out_rope + (int64_t)dst * ROPE_W);
    for (int w = lane; w < ROPE_W / 4; w += 32) dr[w] = sr[w];

    if (lane == 0) out_pos[dst] = ((const uint32_t*)ptrs[3 * n_hot + h])[gid];
}

extern "C" int32_t run_corpus_gather_rows_batched(
    const int64_t* ptrs,
    const uint32_t* meta,
    uint32_t* out_nope,
    uint32_t* out_scale,
    uint32_t* out_rope,
    uint32_t* out_pos,
    int32_t n_hot,
    int32_t max_k,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (n_hot <= 0 || max_k <= 0) return 0;
    dim3 blocks((max_k + ROWS_PER_BLOCK - 1) / ROWS_PER_BLOCK, n_hot);
    corpus_gather_rows_batched_kernel<<<blocks, ROWS_PER_BLOCK * 32, 0, stream>>>(
        (const uint64_t*)ptrs, meta, out_nope, out_scale, out_rope, out_pos, n_hot);
    return (int32_t)cudaGetLastError();
}

extern "C" int32_t run_corpus_gather_rows(
    const uint32_t* nope_i8,
    const uint32_t* nope_scale,
    const uint32_t* rope_bf,
    const uint32_t* pos,
    const uint32_t* gids,
    uint32_t* out_nope,
    uint32_t* out_scale,
    uint32_t* out_rope,
    uint32_t* out_pos,
    int32_t k,
    int32_t row_offset,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (k <= 0) return 0;
    // 128 threads covers the widest region (NOPE_W=112) in one stride.
    corpus_gather_rows_kernel<<<k, 128, 0, stream>>>(
        nope_i8, nope_scale, rope_bf, pos, gids,
        out_nope, out_scale, out_rope, out_pos, k, row_offset);
    return (int32_t)cudaGetLastError();
}
