#pragma once
// =============================================================================
// cp_async.cuh — cp.async primitives, LoadDescriptor, named barriers.
//
// Reuses cp_async_commit / cp_async_wait from the v2 stack
// (paged_decode_kernel.cuh). This file only adds:
//   - cp_async_cg_16_raw: a non-templated 16-byte cp.async helper for the
//     loader role.
//   - LoadDescriptor: descriptor queue entry consumed by loader_role.
//   - bar_sync / bar_arrive: named-barrier wrappers.
// =============================================================================

#include <cuda_runtime.h>
#include <cstdint>

namespace fused_attn {

__device__ __forceinline__ void cp_async_cg_16_raw(
    void*       dst_smem,
    const void* src_global
) {
    uint32_t smem_int = static_cast<uint32_t>(__cvta_generic_to_shared(dst_smem));
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16;\n"
        :
        : "r"(smem_int), "l"(src_global)
        : "memory"
    );
}

// LoadDescriptor — describes one cp.async transfer. The loader pool walks a
// queue of these. Each entry encodes:
//   - source/destination addresses
//   - byte count (must be multiple of 16)
//   - barrier IDs for the pre-load free-slot wait and post-load ready signal
struct alignas(16) LoadDescriptor {
    const void* src_vram;       // 8 B
    void*       dst_smem;       // 8 B
    uint32_t    bytes;          // 4 B (must be multiple of 16)
    uint8_t     free_barrier;   // 1 B (0xFF = none)
    uint8_t     ready_barrier;  // 1 B (0xFF = none)
    uint8_t     sync_count;     // 1 B (participants for the named barrier)
    uint8_t     _pad;           // 1 B
};

static constexpr uint8_t BARRIER_NONE = 0xFF;

// Named barrier wrappers (CUDA's bar.sync / bar.arrive — 16 named barriers per
// CTA). The fused kernel uses these for warp-specialized coordination across
// loader / dequant / consumer pools.
__device__ __forceinline__ void bar_sync(int barrier_id, int participants) {
    asm volatile("bar.sync %0, %1;" :: "r"(barrier_id), "r"(participants) : "memory");
}

__device__ __forceinline__ void bar_arrive(int barrier_id, int participants) {
    asm volatile("bar.arrive %0, %1;" :: "r"(barrier_id), "r"(participants) : "memory");
}

namespace bar_id {
    static constexpr int W_OR_KV_LOADED   = 0;
    static constexpr int W_OR_KV_CONSUMED = 1;
    static constexpr int INT8_READY       = 2;
    static constexpr int INT8_CONSUMED    = 3;
    static constexpr int PHASE_2_TO_3     = 4;
    static constexpr int PHASE_3_TO_4     = 5;
}

} // namespace fused_attn
