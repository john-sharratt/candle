// Stamp and clear the sampler's resident per-row count tables.
//
// The sampling kernel reads `counts[row * vocab + token]` densely, while one
// dispatch holds a few hundred nonzero entries. The table stays all zero
// between dispatches: a dispatch writes its entries in, and once the sampling
// kernel has been queued the same offsets are written back to zero. Each
// entry is one independent store, so the entries are spread over the grid.

#include <cuda_runtime.h>
#include <stdint.h>

// `table[offsets[i]] = values[i]`, or `= 0` when `values` is null.
extern "C" __global__ void stamp_counts_kernel(
    const uint32_t* __restrict__ offsets,
    const uint32_t* __restrict__ values,
    uint32_t n,
    uint32_t* __restrict__ table
) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
        table[offsets[i]] = values != nullptr ? values[i] : 0u;
    }
}

// Launch `stamp_counts_kernel` over `n` entries on `stream`. Offsets must be
// distinct, so no two threads store to one entry.
extern "C" void run_stamp_counts(
    const uint32_t* offsets,
    const uint32_t* values,
    uint32_t n,
    uint32_t* table,
    void* stream
) {
    if (n == 0) {
        return;
    }
    const uint32_t block = 256;
    const uint32_t grid = (n + block - 1) / block;
    stamp_counts_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(offsets, values, n, table);
}
