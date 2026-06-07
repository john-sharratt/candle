#pragma once
// =============================================================================
// loader_role.cuh — pure cp.async dispatch driven by descriptor queue.
//
// 2 loader warps (W0, W1) round-robin the LoadDescriptor queue. For each
// descriptor: optional pre-load barrier wait, then issue cp.async chunks of
// 16 bytes per lane until `bytes` is fully covered, then commit + wait
// + raise the post-load ready barrier.
// =============================================================================

#include "../paged-decode/paged_decode_kernel.cuh"  // cp_async_commit / cp_async_wait
#include "model_descriptor.cuh"
#include "smem_arena.cuh"
#include "cp_async.cuh"

namespace fused_attn {

template<typename Cfg, typename Arch>
__device__ void loader_role(
    int                     warp_in_pool,
    int                     lane,
    const LoadDescriptor*   queue,
    int                     n_queue_entries
) {
    // Round-robin partition: warp 0 takes even entries, warp 1 takes odd.
    for (int i = warp_in_pool; i < n_queue_entries; i += 2) {
        const LoadDescriptor& desc = queue[i];

        if (desc.free_barrier != BARRIER_NONE) {
            bar_sync(desc.free_barrier, desc.sync_count);
        }

        int n_chunks = desc.bytes / 16;
        for (int c = lane; c < n_chunks; c += 32) {
            void*       dst_chunk = static_cast<char*>(desc.dst_smem) + c * 16;
            const void* src_chunk = static_cast<const char*>(desc.src_vram) + c * 16;
            cp_async_cg_16_raw(dst_chunk, src_chunk);
        }

        // Reuse v2's commit/wait helpers (they live in paged_decode_kernel.cuh).
        cp_async_commit</*USE_TC=*/true>();
        cp_async_wait<tile::N_PIPELINE_STAGES - 1, /*USE_TC=*/true>();

        if (desc.ready_barrier != BARRIER_NONE) {
            bar_arrive(desc.ready_barrier, desc.sync_count);
        }
    }

    cp_async_wait<0, /*USE_TC=*/true>();
}

} // namespace fused_attn
