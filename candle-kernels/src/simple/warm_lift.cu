// =============================================================================
// warm_lift_scatter — one pinned ring slot into its destination arena slots
// =============================================================================
//
// The warm lift (docs/warm_lift.md) copies a turn's K/V from host RAM into fresh
// VRAM arena slots. The host packs a slot of the pinned ring with the bands'
// payloads, 16-byte aligned, and writes the slot's descriptor table behind them;
// this kernel reads both straight from the slot — mapped pinned memory, at its
// host address — and writes every band to its destination. Nothing is uploaded.
//
// The copy loop is the expert path's (`moe_live_ahead_piece`,
// candle-kernels/src/quantized/kernel.cuh): four 16-byte loads in flight per
// thread before their stores, which is what pulls pinned memory at the link's
// rate (docs/moe_live_dispatch_design.md §0.10.1 — one load at a time, 2.2 GB/s).
//
// The loads are `.cv`, not `.cg`: the host rewrites a ring slot in place every
// time it reuses it, and `.cg` may be served from an L2 line an earlier batch's
// read of the same address left behind.
//
//   Grid:  (WARM_LIFT_BLOCKS, 1, 1) — bands taken in turn by block
//   Block: (WARM_LIFT_THREADS, 1, 1)

#include <cstdint>
#include <cuda_runtime.h>

#define WARM_LIFT_BLOCKS 8
#define WARM_LIFT_THREADS 128
#define WARM_LIFT_IN_FLIGHT 4

// One band of a slot: its payload at `slot + src_off`, `bytes` long, to `dst`.
struct WarmLiftEntry {
    uint64_t src_off;
    uint64_t dst;
    uint64_t bytes;
};

__device__ __forceinline__ uint4 warm_lift_ld16(const uint8_t* p) {
    uint4 v;
    asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
    return v;
}

__device__ __forceinline__ unsigned long long warm_lift_ld8(const uint64_t* p) {
    unsigned long long v;
    asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(v) : "l"(p));
    return v;
}

__device__ __forceinline__ unsigned char warm_lift_ld1(const uint8_t* p) {
    unsigned short v;
    asm volatile("ld.global.cv.u8 %0, [%1];" : "=h"(v) : "l"(p));
    return (unsigned char)v;
}

__global__ void warm_lift_scatter_kernel(
    const uint8_t* __restrict__ slot,
    const WarmLiftEntry* __restrict__ table,
    int n_entries
) {
    const int tid = threadIdx.x;
    for (int e = blockIdx.x; e < n_entries; e += gridDim.x) {
        const uint64_t* word = reinterpret_cast<const uint64_t*>(table + e);
        const unsigned long long src_off = warm_lift_ld8(word);
        const unsigned long long dst_addr = warm_lift_ld8(word + 1);
        const unsigned long long bytes = warm_lift_ld8(word + 2);
        const uint8_t* src = slot + src_off;
        uint8_t* dst = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(dst_addr));
        // The host places every source 16-byte aligned; a destination or a
        // length that is not takes the byte copy, which is correct for any.
        const bool wide = ((dst_addr | bytes | (unsigned long long)(uintptr_t)src) & 15ull) == 0ull;
        if (wide) {
            const long long units = (long long)(bytes / 16ull);
            for (long long b = tid; b < units; b += WARM_LIFT_THREADS * WARM_LIFT_IN_FLIGHT) {
                uint4 v[WARM_LIFT_IN_FLIGHT];
                #pragma unroll
                for (int u = 0; u < WARM_LIFT_IN_FLIGHT; ++u) {
                    const long long idx = b + (long long)u * WARM_LIFT_THREADS;
                    if (idx < units) {
                        v[u] = warm_lift_ld16(src + (size_t)idx * 16);
                    }
                }
                #pragma unroll
                for (int u = 0; u < WARM_LIFT_IN_FLIGHT; ++u) {
                    const long long idx = b + (long long)u * WARM_LIFT_THREADS;
                    if (idx < units) {
                        *reinterpret_cast<uint4*>(dst + (size_t)idx * 16) = v[u];
                    }
                }
            }
        } else {
            for (unsigned long long i = tid; i < bytes; i += WARM_LIFT_THREADS) {
                dst[i] = warm_lift_ld1(src + i);
            }
        }
    }
}

extern "C" void run_warm_lift_scatter(
    const uint8_t*       slot,
    const WarmLiftEntry* table,
    int                  n_entries,
    cudaStream_t         stream
) {
    if (n_entries <= 0) return;
    warm_lift_scatter_kernel<<<WARM_LIFT_BLOCKS, WARM_LIFT_THREADS, 0, stream>>>(
        slot, table, n_entries
    );
}
