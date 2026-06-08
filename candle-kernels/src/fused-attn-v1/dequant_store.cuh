#pragma once
// =============================================================================
// dequant_store.cuh — per-tile K/V dequant primitives for Phase 4.
//
// dequant_kv_tile_K: load K from arena (per-palette), apply RoPE in registers,
//   compute fresh per-palette INT8 scale, re-quant, write k-major to smem.
//
// dequant_kv_tile_V: load V from arena into mn-major INT8, scales come from
//   the per-token V scale tensor (loaded via cp.async by the loader pool).
//
// Reuses ArenaAccessor::load_head_int8_readthrough (per-dim block scales) from
// convert_all.cuh and v2's apply_rope_*_f32 helpers.
// =============================================================================

#include "../convert/convert_all.cuh"
#include "../arena_table.cuh"
#include "../paged-decode/slot_types.cuh"
#include "../paged-decode/pal_iter.cuh"
#include "model_descriptor.cuh"
#include "rope.cuh"
#include "softmax_state.cuh"  // for warp_reduce_max

namespace fused_attn {

// -----------------------------------------------------------------------------
// dequant_kv_tile_K
// -----------------------------------------------------------------------------
template<typename Cfg>
__device__ void dequant_kv_tile_K(
    const uint8_t*  head_ptr,
    int             /*tile_idx*/,
    int             tile_within_chunk_base,
    int8_t*         dst_int8_kmajor,    // [HEAD_DIM][TILE_N]
    int             dst_dim_stride,     // = TILE_N
    float*          out_scales_post,    // [TILE_N][N_PALETTE]
    const float*    rope_cs_table,
    const uint32_t* rope_positions_per_token, // [TILE_N]
    int             lane,
    int             /*warp_in_pool*/
) {
    constexpr int HEAD_DIM     = Cfg::HEAD_DIM;
    constexpr int N_PAL        = Cfg::N_PALETTE;
    constexpr int SUB_HEAD_DIM = HEAD_DIM / N_PAL;
    constexpr int VEC          = HEAD_DIM / 32;
    constexpr int TILE_N       = tile::TILE_N;
    constexpr int BLOCKS_PER_DIM = CHUNK_SIZE / 32;

    // Per-palette geometry inside the head (palette p covers SUB_HEAD_DIM dims).
    constexpr int64_t sub_head_stride = (int64_t)SUB_HEAD_DIM * CHUNK_SIZE;

    #pragma unroll 1
    for (int t = 0; t < TILE_N; ++t) {
        int within = tile_within_chunk_base + t;

        // ── Load + dequant FP32 in registers, palette-by-palette ─────────────
        float k_regs[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) k_regs[j] = 0.f;

        #pragma unroll
        for (int p = 0; p < N_PAL; ++p) {
            uint64_t k_ptr_p   = kvhead_k_ptr<HEAD_DIM>(head_ptr, p);
            int      k_fmt     = kvhead_k_fmt<HEAD_DIM>(head_ptr, p);
            float    k_scale_p = kvhead_k_scale<HEAD_DIM>(head_ptr, p);
            if (k_ptr_p == 0) continue;

            ArenaAccessor k_acc(
                (const char*)(uintptr_t)k_ptr_p, k_fmt,
                sub_head_stride, sub_head_stride,
                BLOCKS_PER_DIM, 0);

            // Load this palette's SUB_HEAD_DIM dims as FP32 into a small scratch
            // and route into the lane's k_regs[] entries that map to palette p.
            float pal_buf[SUB_HEAD_DIM];
            k_acc.template load_head_scaled<float, SUB_HEAD_DIM, /*USE_TC=*/false>(
                pal_buf, 0, 0, within, lane, k_scale_p);
            // pal_buf is laid out by sub-head dim 0..SUB_HEAD_DIM-1.
            // Lane covers VEC dims of the full HEAD_DIM (lane*VEC + j).
            // Fold each lane-owned dim that maps to palette p into k_regs[j].
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                int dim_full = lane * VEC + j;
                int pal_full = dim_full / SUB_HEAD_DIM;
                if (pal_full == p) {
                    int sub = dim_full - p * SUB_HEAD_DIM;
                    k_regs[j] = pal_buf[sub];
                }
            }
        }

        // ── Apply RoPE per-token ────────────────────────────────────────────
        apply_rope_dispatch<HEAD_DIM, VEC, Cfg::ROPE_STYLE, Cfg::ROPE_INTERLEAVED>(
            k_regs, lane, (int)rope_positions_per_token[t], rope_cs_table);

        // ── Compute per-palette max-abs, derive new INT8 scale ─────────────
        float max_abs_per_pal[N_PAL] = {};
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int dim_full = lane * VEC + j;
            int pal_full = dim_full / SUB_HEAD_DIM;
            float av = fabsf(k_regs[j]);
            if (av > max_abs_per_pal[pal_full]) max_abs_per_pal[pal_full] = av;
        }
        float inv_scale[N_PAL];
        #pragma unroll
        for (int p = 0; p < N_PAL; ++p) {
            float rmax = warp_reduce_max(max_abs_per_pal[p]);
            float new_scale = (rmax > 0.f) ? (rmax / 127.f) : 1.f;
            inv_scale[p] = 1.f / new_scale;
            if (lane == 0) {
                out_scales_post[t * N_PAL + p] = new_scale;
            }
        }

        // ── Re-quantize INT8, write k-major dst[dim][token] ─────────────────
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int dim_full = lane * VEC + j;
            int pal_full = dim_full / SUB_HEAD_DIM;
            float scaled = k_regs[j] * inv_scale[pal_full];
            float clamped = fminf(fmaxf(scaled, -127.f), 127.f);
            int8_t q8 = (int8_t)__float2int_rn(clamped);
            dst_int8_kmajor[dim_full * dst_dim_stride + t] = q8;
        }
    }
}

// -----------------------------------------------------------------------------
// dequant_kv_tile_V — no RoPE, mn-major output
// -----------------------------------------------------------------------------
template<typename Cfg>
__device__ void dequant_kv_tile_V(
    const uint8_t*  head_ptr,
    int             /*tile_idx*/,
    int             tile_within_chunk_base,
    int8_t*         dst_int8_mnmajor,   // [TILE_N][HEAD_DIM]
    int             dst_token_stride,    // = HEAD_DIM
    float*          out_scales_per_token, // [TILE_N]
    int             lane,
    int             /*warp_in_pool*/
) {
    constexpr int HEAD_DIM     = Cfg::HEAD_DIM;
    constexpr int N_PAL        = Cfg::N_PALETTE;
    constexpr int SUB_HEAD_DIM = HEAD_DIM / N_PAL;
    constexpr int VEC          = HEAD_DIM / 32;
    constexpr int TILE_N       = tile::TILE_N;
    constexpr int BLOCKS_PER_DIM = CHUNK_SIZE / 32;
    constexpr int64_t sub_head_stride = (int64_t)SUB_HEAD_DIM * CHUNK_SIZE;

    #pragma unroll 1
    for (int t = 0; t < TILE_N; ++t) {
        int within = tile_within_chunk_base + t;

        // Load V as FP32 (per-palette), then re-quant per-token (token scale).
        float v_regs[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) v_regs[j] = 0.f;

        #pragma unroll
        for (int p = 0; p < N_PAL; ++p) {
            uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(head_ptr, p);
            int      v_fmt   = kvhead_v_fmt<HEAD_DIM>(head_ptr, p);
            float    v_scale_p = kvhead_v_scale<HEAD_DIM>(head_ptr, p);
            if (v_ptr_p == 0) continue;

            ArenaAccessor v_acc(
                (const char*)(uintptr_t)v_ptr_p, v_fmt,
                sub_head_stride, sub_head_stride,
                BLOCKS_PER_DIM, 0);

            float pal_buf[SUB_HEAD_DIM];
            v_acc.template load_head_scaled<float, SUB_HEAD_DIM, /*USE_TC=*/false>(
                pal_buf, 0, 0, within, lane, v_scale_p);

            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                int dim_full = lane * VEC + j;
                int pal_full = dim_full / SUB_HEAD_DIM;
                if (pal_full == p) {
                    int sub = dim_full - p * SUB_HEAD_DIM;
                    v_regs[j] = pal_buf[sub];
                }
            }
        }

        // Compute per-token scale (single warp-collective max-abs).
        float max_abs = 0.f;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float av = fabsf(v_regs[j]);
            if (av > max_abs) max_abs = av;
        }
        max_abs = warp_reduce_max(max_abs);
        float scale_t = (max_abs > 0.f) ? (max_abs / 127.f) : 1.f;
        float inv = 1.f / scale_t;
        if (lane == 0) out_scales_per_token[t] = scale_t;

        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            int dim_full = lane * VEC + j;
            float scaled = v_regs[j] * inv;
            float clamped = fminf(fmaxf(scaled, -127.f), 127.f);
            int8_t q8 = (int8_t)__float2int_rn(clamped);
            dst_int8_mnmajor[t * dst_token_stride + dim_full] = q8;
        }
    }
}

} // namespace fused_attn
