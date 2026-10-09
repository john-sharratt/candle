#pragma once
// The narrow int8 dense entry's launch geometry and shared-memory layout, shared by the
// kernel (`dense_narrow.cuh`), which carves its dynamic shared memory by it, and its
// launcher (`dispatcher.cu`), which sizes the launch by it. See the entry for the
// schedule these describe.
//
// One block owns one 8-row output tile — one KO weight chunk per K tile — and `warps`
// warps; warp `w` walks the contiguous K-tile range `[w·per, min(K_tiles, (w+1)·per))`,
// `per = ceil(K_tiles / warps)`. Dynamic shared memory, in order:
//
//   ring   — `warps × narrow_ahead(chunk) × chunk` bytes: each warp's weight chunks in
//            flight, one slot per chunk;
//   slots  — one float2 per lane per K tile past warp 0's range: the per-tile folds of
//            warps 1.., which warp 0 adds to its own running sum in tile order.

// The most tokens one narrow launch carries: the m16n8k32 MMA's first eight rows.
#define NARROW_MAX_ROWS 8
// The most warps one narrow block runs (its launch bounds).
#define NARROW_MAX_WARPS 16
// The dynamic shared memory the launcher opts every narrow kernel into, and the most any
// launch it accepts may use: under the 99 KiB per-block opt-in of sm_86, sm_89 and sm_120.
#define NARROW_SMEM_CAP (96 * 1024)

// Weight chunks one warp keeps in flight: about 4 KiB of whatever the format, at least four
// chunks (the Q8_KO chunk is 1,056 bytes) and at most eight (Q2_KO's is 288).
__host__ __device__ constexpr int narrow_ahead(int chunk_bytes) {
    return 4096 / chunk_bytes < 4 ? 4 : (4096 / chunk_bytes > 8 ? 8 : 4096 / chunk_bytes);
}

// K tiles each warp walks (the last warps' ranges may be short or empty).
__host__ __device__ constexpr int narrow_tiles_per_warp(int k_tiles, int warps) {
    return (k_tiles + warps - 1) / warps;
}

// Bytes of the weight ring.
__host__ __device__ constexpr int narrow_ring_bytes(int chunk_bytes, int warps) {
    return warps * narrow_ahead(chunk_bytes) * chunk_bytes;
}

// K tiles whose folds go through the slots: every tile past warp 0's range.
__host__ __device__ constexpr int narrow_slot_tiles(int k_tiles, int warps) {
    return k_tiles > narrow_tiles_per_warp(k_tiles, warps)
               ? k_tiles - narrow_tiles_per_warp(k_tiles, warps)
               : 0;
}

// Bytes of the fold slots: a float2 per lane per slotted tile.
__host__ __device__ constexpr int narrow_slot_bytes(int k_tiles, int warps) {
    return narrow_slot_tiles(k_tiles, warps) * 32 * 8;
}

// The launch's dynamic shared memory.
__host__ __device__ constexpr int narrow_smem_bytes(int chunk_bytes, int k_tiles, int warps) {
    return narrow_ring_bytes(chunk_bytes, warps) + narrow_slot_bytes(k_tiles, warps);
}
