// =============================================================================
// scatter_copy.cuh — one warp copies one run of bytes, wide and in flight
// =============================================================================
//
// The copy both scatter kernels are made of (`kv_migrate.cu`, `warm_lift.cu`):
// a plan of many short runs — a KV band is ~1-4 KB — each `src → dst`, `bytes`
// long. A run that short is latency-bound, not bandwidth-bound: one block per
// run leaves most of its threads idle and pays one round trip per run, which is
// what held the warm lift to ~0.5 GB/s over PCIe.
//
// So a WARP owns a run, and each lane keeps `SCATTER_IN_FLIGHT` 16-byte loads
// in flight before it stores them: a warp moves 32 x 4 x 16 = 2 KB a round
// trip, about one band, and a grid of many warps keeps enough round trips in
// flight to cover the latency. Runs are taken grid-stride by warp, so the grid
// is sized to fill the device, not to the plan.
//
// `Load` picks how a source byte is read:
//   - `LoadCv` — `ld.global.cv`, for host memory the host rewrites in place
//     (the warm lift's pinned ring): `.cg` may be served from an L2 line an
//     earlier read of the same address left behind.
//   - `LoadCg` — `ld.global.cg`, for device memory read once (every byte of a
//     migration is read once, so L1 would only be polluted).
//
// A run whose source, destination or length is not 16-byte aligned takes the
// byte copy, which is correct for any.

#pragma once

#include <cstdint>

#define SCATTER_WARP 32
#define SCATTER_IN_FLIGHT 4
// Threads a block, and blocks resident per SM: six blocks of 256 fill an SM of
// 1,536 threads (sm_86, sm_89, sm_120) — 48 warps, each its own round trips.
// `__launch_bounds__` holds the kernels to the registers that allows.
#define SCATTER_THREADS 256
#define SCATTER_BLOCKS_PER_SM 6

struct LoadCv {
    static __device__ __forceinline__ uint4 ld16(const uint8_t* p) {
        uint4 v;
        asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
        return v;
    }
    static __device__ __forceinline__ unsigned char ld1(const uint8_t* p) {
        unsigned short v;
        asm volatile("ld.global.cv.u8 %0, [%1];" : "=h"(v) : "l"(p));
        return (unsigned char)v;
    }
    static __device__ __forceinline__ unsigned long long ld8(const uint64_t* p) {
        unsigned long long v;
        asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(v) : "l"(p));
        return v;
    }
};

struct LoadCg {
    static __device__ __forceinline__ uint4 ld16(const uint8_t* p) {
        uint4 v;
        asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];"
                     : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
        return v;
    }
    static __device__ __forceinline__ unsigned char ld1(const uint8_t* p) {
        return *p;
    }
    static __device__ __forceinline__ unsigned long long ld8(const uint64_t* p) {
        return *p;
    }
};

// Copy `bytes` from `src` to `dst`, by the whole warp; `lane` is the caller's
// lane. Every lane of the warp must call it with the same arguments.
template <typename Load>
__device__ __forceinline__ void warp_copy(
    const uint8_t* src, uint8_t* dst, unsigned long long bytes, int lane
) {
    const bool wide =
        (((unsigned long long)(uintptr_t)src | (unsigned long long)(uintptr_t)dst | bytes) & 15ull)
        == 0ull;
    // 32-bit unit indices: a run is a KV band or a row of one, far below 2^32
    // sixteen-byte units, and 64-bit arithmetic on every index cost the kernels
    // a fifth of their registers — the difference between four and six
    // resident blocks.
    if (wide) {
        const uint4* s = reinterpret_cast<const uint4*>(src);
        uint4* d = reinterpret_cast<uint4*>(dst);
        const unsigned int units = (unsigned int)(bytes >> 4);
        const unsigned int step = SCATTER_WARP * SCATTER_IN_FLIGHT;
        for (unsigned int b = lane; b < units; b += step) {
            uint4 v[SCATTER_IN_FLIGHT];
            #pragma unroll
            for (int u = 0; u < SCATTER_IN_FLIGHT; ++u) {
                const unsigned int idx = b + u * SCATTER_WARP;
                if (idx < units) {
                    v[u] = Load::ld16(reinterpret_cast<const uint8_t*>(s + idx));
                }
            }
            #pragma unroll
            for (int u = 0; u < SCATTER_IN_FLIGHT; ++u) {
                const unsigned int idx = b + u * SCATTER_WARP;
                if (idx < units) {
                    d[idx] = v[u];
                }
            }
        }
    } else {
        const unsigned int n = (unsigned int)bytes;
        for (unsigned int i = lane; i < n; i += SCATTER_WARP) {
            dst[i] = Load::ld1(src + i);
        }
    }
}

// The warp's index across the grid, and how many warps the grid holds.
__device__ __forceinline__ int scatter_warp_id() {
    return (int)((blockIdx.x * blockDim.x + threadIdx.x) / SCATTER_WARP);
}
__device__ __forceinline__ int scatter_warp_count() {
    return (int)((gridDim.x * blockDim.x) / SCATTER_WARP);
}
