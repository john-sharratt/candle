// =============================================================================
// warm_lift_scatter — one pinned ring slot into its destination arena slots
// =============================================================================
//
// The warm lift (docs/warm_lift.md) copies a turn's K/V from host RAM into fresh
// VRAM arena slots. The host packs a slot of the pinned ring with the bands'
// payloads, 16-byte aligned, and writes the slot's descriptor table behind them;
// this kernel reads both straight from the slot — mapped pinned memory, at its
// host address — and writes every run to its destination. Nothing is uploaded.
//
// A warp takes one run at a time, grid-stride (`scatter_copy.cuh`): lanes 0-2
// read the run's three table words and the warp shares them, then the warp
// copies the run with four 16-byte `.cv` loads in flight per lane. The grid is
// sized by the caller to fill the device; a run's length decides nothing about
// it.
//
//   Grid:  (blocks, 1, 1), blocks chosen by the caller
//   Block: (SCATTER_THREADS, 1, 1)

#include <cstdint>
#include <cuda_runtime.h>

#include "scatter_copy.cuh"

// One run of a slot: its payload at `slot + src_off`, `bytes` long, to `dst`.
struct WarmLiftEntry {
    uint64_t src_off;
    uint64_t dst;
    uint64_t bytes;
};

__global__ void __launch_bounds__(SCATTER_THREADS, SCATTER_BLOCKS_PER_SM)
warm_lift_scatter_kernel(
    const uint8_t* __restrict__ slot,
    const WarmLiftEntry* __restrict__ table,
    int n_entries
) {
    const int lane = threadIdx.x & (SCATTER_WARP - 1);
    const int warps = scatter_warp_count();
    for (int e = scatter_warp_id(); e < n_entries; e += warps) {
        // One table row is three words; three lanes fetch them, each once.
        const uint64_t* row = reinterpret_cast<const uint64_t*>(table + e);
        unsigned long long word = 0ull;
        if (lane < 3) {
            word = LoadCv::ld8(row + lane);
        }
        const unsigned long long src_off = __shfl_sync(0xffffffffu, word, 0);
        const unsigned long long dst = __shfl_sync(0xffffffffu, word, 1);
        const unsigned long long bytes = __shfl_sync(0xffffffffu, word, 2);
        warp_copy<LoadCv>(
            slot + src_off,
            reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(dst)),
            bytes,
            lane
        );
    }
}

extern "C" void run_warm_lift_scatter(
    const uint8_t*       slot,
    const WarmLiftEntry* table,
    int                  n_entries,
    int                  blocks,
    cudaStream_t         stream
) {
    if (n_entries <= 0 || blocks <= 0) return;
    // No more warps than runs: a warp with no run would only exit.
    const int warps_per_block = SCATTER_THREADS / SCATTER_WARP;
    const int needed = (n_entries + warps_per_block - 1) / warps_per_block;
    const int grid = blocks < needed ? blocks : needed;
    warm_lift_scatter_kernel<<<grid, SCATTER_THREADS, 0, stream>>>(slot, table, n_entries);
}
