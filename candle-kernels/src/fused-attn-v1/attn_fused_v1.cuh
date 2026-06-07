#pragma once
// =============================================================================
// attn_fused_v1.cuh — top-level fused QKV + attention kernel.
//
// Grid: (num_active_slots, n_kv_head)
// Block: 256 threads (8 warps): W0/W1=loaders, W2/W3=dequant, W4..W7=consumers
// =============================================================================

#include "../paged-decode/slot_types.cuh"
#include "../arena_table.cuh"
#include "model_descriptor.cuh"
#include "arch_traits.cuh"
#include "smem_arena.cuh"
#include "cp_async.cuh"
#include "loader_role.cuh"
#include "dequant_role.cuh"
#include "consumer_role.cuh"

namespace fused_attn {

// build_load_queue — populate the descriptor queue.
//
// Emits one entry per K-chunk for the W_qkv source bytes. The loader pool
// processes this serially; consumers wait via INT8_READY barriers.
//
// Currently emits only Phase 2 entries (W_qkv per K-chunk). Phase 4 KV tile
// loads are intentionally absent; the dequant role's Phase 4 path uses
// ArenaAccessor directly to pull from K/V cache without going through the
// LoadDescriptor queue. A future revision could add Phase 4 entries to use
// cp.async for KV tile data and reduce dequant-role latency.
template<typename Cfg, typename Arch, typename Q_T>
__device__ int build_load_queue(
    const Q_T*         /*activations*/,
    const uint8_t*     w_qkv_q4,
    const void*        /*w_qkv_scales*/,
    const SlotHeader&  /*slot*/,
    int                /*slot_idx*/,
    int                /*kv_head_idx*/,
    int                /*n_kv_head*/,
    LoadDescriptor*    queue
) {
    constexpr int N_K_CHUNKS = Cfg::D_MODEL / Arch::MMA_K;
    constexpr int N_TILE     = tile::W_QKV_TILE_N;
    // Per K-chunk × per N-tile: 32 K-elements × N_TILE N-dims × 18 B/Q4_0 block / 32 elements
    //   = N_TILE * 18 bytes (one Q4_0 block per N-dim covering 32 K-elems)
    constexpr int W_BYTES_PER_LOAD = N_TILE * 18;

    int count = 0;

    // For each K-chunk, emit one load. The host is expected to lay out
    // w_qkv_q4 as [K-chunk][N-tile][N-dim within tile][Q4_0 block 18 bytes],
    // i.e. K-chunk-major. This stub assumes one descriptor covers one
    // K-chunk's full N range; in practice the kernel works in N-tile
    // granularity which would emit N_K_CHUNKS * N_N_TILES entries.
    //
    // Until N-tiling is wired through the consumer's K-chunk loop, emit
    // one entry per K-chunk covering the first N-tile only. This is
    // intentionally incomplete and only safe when the dispatch returns
    // cudaErrorNotSupported (i.e. the kernel never executes from Rust).
    for (int k_chunk = 0; k_chunk < N_K_CHUNKS; ++k_chunk) {
        if (count >= 256) break;  // MAX_QUEUE_ENTRIES bound
        LoadDescriptor& d = queue[count++];
        d.src_vram = w_qkv_q4 + (int64_t)k_chunk * W_BYTES_PER_LOAD;
        // dst_smem points into one of the 2 staging slots; the loader
        // alternates stages.
        d.dst_smem = nullptr;  // filled by loader at issue time
        d.bytes = W_BYTES_PER_LOAD;
        d.free_barrier = bar_id::W_OR_KV_CONSUMED;
        d.ready_barrier = bar_id::W_OR_KV_LOADED;
        d.sync_count = 4 * 32;  // dequant + loader pools (2+2 warps)
    }
    return count;
}

template<typename Q_T, typename O, typename Cfg, typename Arch>
__global__ __launch_bounds__(256, 1)
void fused_qkv_attn_kernel(
    const Q_T*     activations,
    const uint8_t* w_qkv_q4,
    const void*    w_qkv_scales,
    const uint8_t* headers_ptr,
    O*             out,
    int            num_active_slots,
    int            n_q_head,
    int            n_kv_head,
    float          softmax_scale,
    const float*   rope_cs,
    int            sliding_window_size
) {
    int tid  = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;

    int slot_idx    = blockIdx.x;
    int kv_head_idx = blockIdx.y;

    if (slot_idx >= num_active_slots || kv_head_idx >= n_kv_head) return;

    const SlotHeader& slot = get_slot_header(headers_ptr, slot_idx);
    if (slot.n_slices == 0) {
        // Match v2's empty-slot zero-write behaviour.
        if (warp >= 4) {
            int wpool = warp - 4;
            constexpr int HEAD_DIM = Cfg::HEAD_DIM;
            constexpr int OUT_DIMS_PER_WARP = HEAD_DIM / 4;
            int heads_per_group = n_q_head / n_kv_head;
            if (heads_per_group <= 0) heads_per_group = 1;
            // Each warp covers OUT_DIMS_PER_WARP dims of every Q head in the group.
            for (int h = 0; h < heads_per_group; ++h) {
                int head_idx = kv_head_idx * heads_per_group + h;
                if (head_idx >= n_q_head) break;
                int dim_base = wpool * OUT_DIMS_PER_WARP;
                int64_t out_base = ((int64_t)slot_idx * (int64_t)n_q_head
                    + (int64_t)head_idx) * (int64_t)HEAD_DIM;
                for (int d = lane; d < OUT_DIMS_PER_WARP; d += 32) {
                    out[out_base + dim_base + d] = (O)0;
                }
            }
        }
        return;
    }

    uint8_t* write_slice_ptr = get_slice_mut<Cfg::HEAD_DIM>(
        slot.slices_ptr, (int)slot.write_slice, n_kv_head);

    const uint32_t ws_rope    = slice_rope(write_slice_ptr);
    const uint16_t ws_len     = slice_len(write_slice_ptr);
    const uint32_t q_rope_pos = ws_rope + ws_len;

    static_assert(smem_arena_fits_default<Cfg>(),
        "Smem arena exceeds 48 KB default; v1 supports only the default budget.");

    __shared__ SmemArena<Cfg> arena;

    // Build load queue (currently empty for the placeholder build).
    constexpr int MAX_QUEUE_ENTRIES = 256;
    __shared__ LoadDescriptor queue[MAX_QUEUE_ENTRIES];
    __shared__ int            queue_count;

    if (tid == 0) {
        queue_count = build_load_queue<Cfg, Arch, Q_T>(
            activations, w_qkv_q4, w_qkv_scales,
            slot, slot_idx, kv_head_idx, n_kv_head,
            queue);
    }
    __syncthreads();

    int kv_len = (int)q_rope_pos + 1;
    if (kv_len > (int)slot.n_slices * CHUNK_SIZE)
        kv_len = (int)slot.n_slices * CHUNK_SIZE;
    int n_kv_tiles = (kv_len + tile::TILE_N - 1) / tile::TILE_N;

    if (warp < 2) {
        loader_role<Cfg, Arch>(
            /*warp_in_pool=*/warp, lane, queue, queue_count);
    } else if (warp < 4) {
        dequant_role<Cfg, Arch, Q_T>(
            /*is_k_warp=*/(warp == 2),
            lane, arena, rope_cs, q_rope_pos, n_kv_tiles,
            write_slice_ptr, slot.slices_ptr,
            (int)slot.write_slice, (int)slot.n_slices,
            kv_head_idx, n_kv_head, slot_idx,
            activations, n_q_head);
    } else {
        consumer_role<Cfg, Arch, O, Q_T>(
            /*warp_in_pool=*/(warp - 4),
            lane, arena, rope_cs, q_rope_pos,
            softmax_scale, sliding_window_size,
            n_kv_tiles, slot_idx, n_q_head, out,
            activations, num_active_slots);
    }
}

template<typename Q_T, typename O, typename Cfg, int SM_VERSION>
cudaError_t launch_fused_qkv_attn(
    const Q_T*     activations,
    const uint8_t* w_qkv_q4,
    const void*    w_qkv_scales,
    const uint8_t* headers_ptr,
    O*             out,
    int            num_active_slots,
    int            n_q_head,
    int            n_kv_head,
    float          softmax_scale,
    const float*   rope_cs,
    int            sliding_window_size,
    cudaStream_t   stream = nullptr
) {
    using Arch = ArchTraits<SM_VERSION>;

    if (n_q_head != Cfg::N_Q_HEADS) return cudaErrorInvalidConfiguration;
    if (n_kv_head != Cfg::N_KV_HEADS) return cudaErrorInvalidConfiguration;

    dim3 grid(num_active_slots, n_kv_head);
    dim3 block(256);

    fused_qkv_attn_kernel<Q_T, O, Cfg, Arch><<<grid, block, 0, stream>>>(
        activations, w_qkv_q4, w_qkv_scales,
        headers_ptr, out,
        num_active_slots, n_q_head, n_kv_head,
        softmax_scale, rope_cs,
        sliding_window_size);

    return cudaGetLastError();
}

} // namespace fused_attn
