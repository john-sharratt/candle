#pragma once
// =============================================================================
// dequant_role.cuh — W2/W3 control flow.
//
// W2 (is_k_warp=true):   K path with RoPE + per-(t,p) re-quant scale
// W3 (is_k_warp=false):  V path, no RoPE
//
// Phase 2: dequant W_qkv Q4 -> INT8 staging (per K-chunk, double-buffered).
// Phase 3: K_new RoPE + scatter to arena (W2) / V_new scatter (W3).
//          Tile-0 fast path: K_new/V_new also re-quantized into smem_int8 stage 0.
// Phase 4: per-tile dequant via dequant_kv_tile_K / V (skip tile 0 — already in smem).
//
// Reuses v2's write_regs_to_arena / write_regs_to_r16 for phase 3 scatter.
// =============================================================================

#include "../paged-decode/paged_decode_kernel.cuh"  // for write_regs_to_arena, write_regs_to_r16, ROPE helpers
#include "../paged-decode/slot_types.cuh"
#include "../arena_table.cuh"
#include "model_descriptor.cuh"
#include "smem_arena.cuh"
#include "dequant_store.cuh"
#include "cp_async.cuh"
#include "rope.cuh"

namespace fused_attn {

template<typename Cfg, typename Arch, typename Q_T>
__device__ void dequant_role(
    bool                    is_k_warp,
    int                     lane,
    SmemArena<Cfg>&         arena,
    const float*            rope_cs_table,
    uint32_t                q_rope_pos,
    int                     n_kv_tiles,
    uint8_t*                write_slice_ptr,
    uint64_t                slices_ptr,
    int                     /*write_slice_idx*/,
    int                     n_slices,
    int                     kv_head_idx,
    int                     n_kv_head,
    int                     /*slot_idx*/,
    const Q_T*              /*q_for_r16_capture*/,
    int                     n_q_head
) {
    using namespace tile;
    constexpr int HEAD_DIM = Cfg::HEAD_DIM;
    constexpr int VEC      = HEAD_DIM / 32;
    constexpr int N_K_CHUNKS = Cfg::D_MODEL / Arch::MMA_K;

    // ───────────────────────────────────────────────────────────────────
    // PHASE 2: dequant W_qkv -> INT8 staging
    //
    // For v1 we keep the W_qkv staging body as a placeholder loop. The actual
    // Q4_0 -> INT8 conversion mirrors ArenaAccessor::load_head_int8_readthrough
    // but with the weight tile shape (32 K-elems x W_QKV_TILE_N output dims).
    // ───────────────────────────────────────────────────────────────────
    for (int k_chunk = 0; k_chunk < N_K_CHUNKS; ++k_chunk) {
        bar_sync(bar_id::W_OR_KV_LOADED, /*participants=*/2 * 32 + 2 * 32);

        // ── W_qkv Q4→INT8 dequant for this K-chunk ──────────────────────
        // The loader pool has placed a Q4_0-packed K-chunk × W_QKV_TILE_N
        // slice into arena.phase12.w_q4_src[stage]. Each Q4_0 block is 18 B
        // (FP16 scale + 16 packed nibble bytes covering 32 elements).
        //
        // For the K-chunk × W_QKV_TILE_N tile (32 × 128 INT8 = 4096 bytes):
        //   - Each Q4_0 block covers 32 K-elements × 1 N-element OR equiv.
        //     depending on how the host laid out W_qkv. Here we assume host
        //     packs as (K-major within block, N-major across blocks):
        //       block_idx = n_dim * (K-chunks) + k_chunk_local
        //     Each block: 32 K-elements for that one N-dim.
        //
        // 64 threads (2 dequant warps) cover W_QKV_TILE_N = 128 N-dims, so
        // each thread handles 2 N-dims per K-chunk.
        int stage = k_chunk % N_W_STAGING_STAGES;
        int local_tid = is_k_warp ? lane : (lane + 32); // 0..63
        constexpr int N_TILE = tile::W_QKV_TILE_N;
        constexpr int THREADS = 64;
        constexpr int DIMS_PER_THREAD = (N_TILE + THREADS - 1) / THREADS;

        const uint8_t* q4_src = arena.as_phase12().w_q4_src[stage];
        int8_t* int8_dst = arena.as_phase12().w_staging_int8[stage];

        #pragma unroll
        for (int dn = 0; dn < DIMS_PER_THREAD; ++dn) {
            int n_dim = local_tid * DIMS_PER_THREAD + dn;
            if (n_dim >= N_TILE) break;

            // One Q4_0 block covers 32 K-elements for this n_dim.
            // Block layout: { __half d; uint8_t qs[16]; } — 18 bytes.
            const uint8_t* blk = q4_src + (int64_t)n_dim * 18;
            __half d_h;
            memcpy(&d_h, blk, sizeof(__half));
            float blk_scale = __half2float(d_h);

            // Write per-32-block scale (one per N-dim for this K-chunk).
            if (dn == 0 || (n_dim & 31) == 0) {
                int scale_block = n_dim / 32;
                arena.as_phase12().w_staging_scales[stage][scale_block] = blk_scale;
            }

            // Unpack 32 nibbles → 32 centered INT8 (range [-8,7]).
            // Layout in w_staging_int8: [stage][k=0..31, n=0..N_TILE-1] flat
            // with K-major rows of N_TILE bytes each. So byte at offset
            // [k*N_TILE + n].
            #pragma unroll
            for (int k = 0; k < 32; ++k) {
                uint8_t b = blk[2 + (k >> 1)];
                int nibble = (k & 1) ? (int)(b >> 4) : (int)(b & 0xF);
                int8_t centered = (int8_t)(nibble - 8);
                int8_dst[(int64_t)k * N_TILE + n_dim] = centered;
            }
        }

        bar_arrive(bar_id::INT8_READY, /*participants=*/4 * 32 + 2 * 32);
        bar_sync(bar_id::INT8_CONSUMED, /*participants=*/4 * 32 + 2 * 32);
    }

    bar_sync(bar_id::PHASE_2_TO_3, /*participants=*/8 * 32);

    // ───────────────────────────────────────────────────────────────────
    // PHASE 3: K_new / V_new RoPE + arena scatter
    // ───────────────────────────────────────────────────────────────────
    {
        const uint16_t ws_offset = slice_offset(write_slice_ptr);
        const uint16_t ws_len    = slice_len(write_slice_ptr);
        const int      within    = (int)ws_offset + (int)ws_len;

        if (within < CHUNK_SIZE) {
            const uint8_t* head_ptr = get_head<HEAD_DIM>(write_slice_ptr, kv_head_idx);

            constexpr int LANES_PER_PAL = 32 / Cfg::N_PALETTE;  // = 8
            constexpr int SUB_HEAD_DIM  = HEAD_DIM / Cfg::N_PALETTE;
            int pal        = lane / LANES_PER_PAL;
            int local_lane = lane % LANES_PER_PAL;

            if (is_k_warp) {
                uint64_t k_ptr_p = kvhead_k_ptr<HEAD_DIM>(head_ptr, pal);
                int      k_fmt   = kvhead_k_fmt<HEAD_DIM>(head_ptr, pal);
                if (k_ptr_p != 0) {
                    char* k_arena = (char*)(uintptr_t)k_ptr_p;
                    float k_regs[VEC];
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j) {
                        int dim = lane * VEC + j;
                        k_regs[j] = arena.as_phase12().k_new_fp32[dim];
                    }

                    apply_rope_dispatch<HEAD_DIM, VEC, Cfg::ROPE_STYLE,
                                        Cfg::ROPE_INTERLEAVED>(
                        k_regs, lane, (int)q_rope_pos, rope_cs_table);

                    if (k_fmt == ArenaFormat::R16) {
                        int heads_per_group = n_q_head / n_kv_head;
                        if (heads_per_group < 1) heads_per_group = 1;
                        // [Q-capture deferred: q_regs[VEC] from q_for_r16_capture.]
                        float q_regs[VEC] = {};
                        write_regs_to_r16<VEC>(k_arena, /*chunk_byte_offset=*/0,
                                                within, local_lane, k_regs, q_regs);
                    } else {
                        int k_esz = ArenaFormat::float_elem_size(k_fmt);
                        if (k_esz > 0) {
                            int64_t eo = (int64_t)within * SUB_HEAD_DIM;
                            write_regs_to_arena<VEC>(k_arena, eo, local_lane,
                                                      k_esz, k_fmt, k_regs);
                        }
                    }

                    // [Phase B: K_new INT8 re-quant for tile-0 fast path.]
                }
            } else {
                uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(head_ptr, pal);
                int      v_fmt   = kvhead_v_fmt<HEAD_DIM>(head_ptr, pal);
                if (v_ptr_p != 0) {
                    char* v_arena = (char*)(uintptr_t)v_ptr_p;
                    float v_regs[VEC];
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j) {
                        int dim = lane * VEC + j;
                        v_regs[j] = arena.as_phase12().v_new_fp32[dim];
                    }

                    int v_esz = ArenaFormat::float_elem_size(v_fmt);
                    if (v_esz > 0) {
                        int64_t eo = (int64_t)within * SUB_HEAD_DIM;
                        write_regs_to_arena<VEC>(v_arena, eo, local_lane,
                                                  v_esz, v_fmt, v_regs);
                    }
                    // [Phase B: V_new INT8 re-quant for tile-0 fast path.]
                }
            }
        }
    }

    bar_sync(bar_id::PHASE_3_TO_4, /*participants=*/8 * 32);

    // ───────────────────────────────────────────────────────────────────
    // PHASE 4: per-tile K/V dequant
    // ───────────────────────────────────────────────────────────────────
    for (int t_tile = 0; t_tile < n_kv_tiles; ++t_tile) {
        if (t_tile == 0) {
            // Tile 0: fast path uses the K_new/V_new written above. (The
            // re-quant into smem_int8 is gated on Phase B; for now signal
            // ready and let consumer MMA over zeros.)
            bar_arrive(bar_id::INT8_READY, /*participants=*/4 * 32 + 2 * 32);
            bar_sync(bar_id::INT8_CONSUMED, /*participants=*/4 * 32 + 2 * 32);
            continue;
        }

        int stage = t_tile % N_PIPELINE_STAGES;
        int k_base = t_tile * TILE_N;
        int my_slice_idx = chunk_div(k_base);
        int tile_within_base = chunk_mod(k_base);

        if (my_slice_idx >= n_slices) {
            bar_arrive(bar_id::INT8_READY, /*participants=*/4 * 32 + 2 * 32);
            bar_sync(bar_id::INT8_CONSUMED, /*participants=*/4 * 32 + 2 * 32);
            continue;
        }

        const uint8_t* sl       = get_slice<HEAD_DIM>(slices_ptr, my_slice_idx, n_kv_head);
        const uint8_t* head_ptr = get_head<HEAD_DIM>(sl, kv_head_idx);

        bar_sync(bar_id::W_OR_KV_LOADED, /*participants=*/2 * 32 + 2 * 32);

        if (is_k_warp) {
            int8_t* dst_int8 = &arena.as_phase4().smem_int8_K[stage][0][0];
            float*  scales_out = &arena.as_phase4().smem_scale_K_post[stage][0][0];
            const uint32_t* rope_pos = &arena.as_phase4().k_rope_positions[stage][0];

            dequant_kv_tile_K<Cfg>(
                head_ptr, t_tile, tile_within_base,
                dst_int8, /*dst_dim_stride=*/TILE_N,
                scales_out, rope_cs_table, rope_pos, lane, /*warp_in_pool=*/0);
        } else {
            int8_t* dst_int8 = &arena.as_phase4().smem_int8_V[stage][0][0];
            float*  scales_per_token = &arena.as_phase4().smem_scale_V[stage][0];

            dequant_kv_tile_V<Cfg>(
                head_ptr, t_tile, tile_within_base,
                dst_int8, /*dst_token_stride=*/HEAD_DIM,
                scales_per_token, lane, /*warp_in_pool=*/0);
        }

        bar_arrive(bar_id::INT8_READY, /*participants=*/4 * 32 + 2 * 32);
        bar_sync(bar_id::INT8_CONSUMED, /*participants=*/4 * 32 + 2 * 32);
    }
}

} // namespace fused_attn
