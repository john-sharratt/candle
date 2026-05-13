// SPDX-License-Identifier: MIT
// Arena compaction kernels — format-agnostic byte copy + GID patch.
//
// Design: docs/arena-compact-kernel-design.md
//
// Kernel 1 (copy):  persistent-block work-stealing. A fixed grid of blocks
//                   atomically grab moves from a counter. Each block copies
//                   one move at a time, then grabs the next. Single launch,
//                   mixed strides, full SM occupancy.
//
// Kernel 2 (patch): rewrites GPU block_table entries from src_gid → dst_gid
//                   using sorted src_gids + binary search.

#include <cuda_runtime.h>
#include <stdint.h>

// =============================================================================
// DATA TYPES
// =============================================================================

struct CompactMove {
    void*       dst;
    const void* src;
    uint32_t    stride_bytes;
    uint32_t    _pad;          // align to 24 bytes → nice for coalescing
};

// =============================================================================
// KERNEL 1 — arena_compact_copy
// =============================================================================
//
// Grid:  <<<num_moves, blockDim>>>
// One block per move. Each block reads its move's stride_bytes and copies
// via direct uint4 loads/stores. Single launch handles mixed strides.
// Requires src/dst 16-byte aligned (guaranteed by cudaMalloc).

__launch_bounds__(128, 8)
__global__ void arena_compact_copy(
    const CompactMove* __restrict__ moves)
{
    const CompactMove& m = moves[blockIdx.x];
    const int stride_u4  = m.stride_bytes / 16;

    const uint4* __restrict__ src = (const uint4*)m.src;
    uint4* __restrict__ dst       = (uint4*)m.dst;

    for (int i = threadIdx.x; i < stride_u4; i += blockDim.x) {
        dst[i] = src[i];
    }
}

// =============================================================================
// KERNEL 2 — arena_compact_patch
// =============================================================================
//
// Grid: <<<ceil(num_entries / 256), 256>>>
// Each thread binary-searches sorted src_gids for its block_table entry.

__global__ void arena_compact_patch(
    int32_t*       __restrict__ block_table,
    int            num_entries,
    const int32_t* __restrict__ src_gids,
    const int32_t* __restrict__ dst_gids,
    int            num_moves)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_entries) return;

    int32_t entry = block_table[idx];
    if (entry < 0) return;  // -1 = empty slot

    int lo = 0, hi = num_moves;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (src_gids[mid] < entry) lo = mid + 1; else hi = mid;
    }
    if (lo < num_moves && src_gids[lo] == entry)
        block_table[idx] = dst_gids[lo];
}

// =============================================================================
// C DISPATCHERS
// =============================================================================

extern "C" void run_arena_compact_copy(
    const CompactMove* moves,
    int32_t num_moves,
    int32_t block_dim,
    cudaStream_t stream)
{
    if (num_moves <= 0 || block_dim <= 0) return;
    arena_compact_copy<<<num_moves, block_dim, 0, stream>>>(moves);
}

extern "C" void run_arena_compact_patch(
    int32_t*       block_table,
    int32_t        num_entries,
    const int32_t* src_gids,
    const int32_t* dst_gids,
    int32_t        num_moves,
    cudaStream_t   stream)
{
    if (num_entries <= 0 || num_moves <= 0) return;
    int grid = (num_entries + 255) / 256;
    arena_compact_patch<<<grid, 256, 0, stream>>>(
        block_table, num_entries, src_gids, dst_gids, num_moves);
}
