// =============================================================================
// Compressed-index expansion: per-slot {offset, count} → the dense index matrix
// =============================================================================
// Every decode slot's corpus selection is the dense range [offset, offset+count)
// of the fleet-wide gathered block — strictly ascending, which is the
// compressed-index contract the paged decode kernel documents. The matrix the
// kernel reads is therefore fully determined by two small per-slot numbers:
//
//     comp_idx[i][k] = k < count[i] ? offset[i] + k : 0xFFFFFFFF
//
// Expressed with tensor ops that is an `arange`, a `broadcast_add`, a
// `broadcast_lt`, a `full` and a `where_cond` — five launches — on top of two
// pageable uploads for the offsets and the counts, every layer of every step.
// The whole thing is ~30 words of input and a few kilobytes of output, so all
// six of those launches are pure overhead around arithmetic a single thread
// could do.
//
// This kernel does it in ONE launch from a staged descriptor (hot-path
// invariant 2b), and emits `comp_cnt` in the same pass because the decode
// kernel wants it as a device array anyway — so the counts never make a
// separate trip.
//
// Descriptor layout, STRUCT-OF-ARRAYS, 2 u32 per slot:
//     [0*n + i]  slot i's offset into the gathered block
//     [1*n + i]  slot i's selected entry count
//
// SoA rather than the interleaved form the other tables use: both halves are
// read by every thread of a row, so a warp's loads of `offset` coalesce instead
// of striding over the counts. grid.y indexes the slot, grid.x tiles its row.

#include <cuda_runtime.h>

#define COMP_IDX_SLOT_WORDS 2

extern "C" __global__ void comp_idx_build_kernel(
    const unsigned int* __restrict__ desc,
    unsigned int* __restrict__ idx,
    unsigned int* __restrict__ cnt,
    int n,
    int max_sel)
{
    const int i = blockIdx.y;
    if (i >= n) return;
    const unsigned int off = desc[i];
    const unsigned int c = desc[n + i];
    // One thread of the whole row republishes the count as a device array.
    if (blockIdx.x == 0 && threadIdx.x == 0) cnt[i] = c;

    const int stride = (int)(gridDim.x * blockDim.x);
    for (int k = (int)(blockIdx.x * blockDim.x + threadIdx.x); k < max_sel; k += stride) {
        // Past the slot's count the row is padding: the kernel's sentinel, not a
        // clamped index, so an out-of-range read can never masquerade as entry 0.
        idx[(long long)i * max_sel + k] = (k < (int)c) ? (off + (unsigned int)k) : 0xFFFFFFFFu;
    }
}

extern "C" void run_comp_idx_build(
    const unsigned int* desc,
    unsigned int* idx,
    unsigned int* cnt,
    int n,
    int max_sel,
    void* stream)
{
    if (n <= 0 || max_sel <= 0) return;
    const int threads = 256;
    int tiles = (max_sel + threads - 1) / threads;
    // A row is at most a few thousand entries; the grid-stride loop covers any
    // remainder rather than launching a block per 256 of them.
    if (tiles > 32) tiles = 32;
    dim3 grid(tiles, n, 1);
    comp_idx_build_kernel<<<grid, threads, 0, (cudaStream_t)stream>>>(
        desc, idx, cnt, n, max_sel);
}
