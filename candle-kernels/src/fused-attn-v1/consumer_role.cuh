#pragma once
// =============================================================================
// consumer_role.cuh — W4..W7 control flow (4 consumer warps).
//
// Phase 2: activation FP→INT8 quant + INT8 MMA QKV projection.
//          For each K-chunk along D_MODEL, accumulate INT32 partial; track
//          parallel FP32 scale-product track; combine at K-chunk close.
// Phase 3: route owned dims by descriptor → Q (RoPE+quant), K_new (smem),
//          V_new (smem).
// Phase 4: attention loop. Reuses the INT8 MMA QK^T + INT8 PV path proven
//          in int8_decode_kernel.cuh.
//
// UNVERIFIED LANE MAPPING — design's m16n8k32 C-fragment lane→dim assumption
// is `(lane%4)*2 + (r>>1)` for column and `(lane/4) + (r&1)*8` for row.
// These need GPU-iteration validation against numerical reference.
// =============================================================================

#include "../paged-decode/paged_decode_kernel.cuh"
#include "../convert/convert_all.cuh"
#include "model_descriptor.cuh"
#include "arch_traits.cuh"
#include "smem_arena.cuh"
#include "mma_wrappers.cuh"
#include "rope.cuh"
#include "softmax_state.cuh"
#include "cp_async.cuh"

namespace fused_attn {

template<typename Cfg, typename Arch, typename O, typename Q_T>
__device__ void consumer_role(
    int                     warp_in_pool,    // 0..3
    int                     lane,
    SmemArena<Cfg>&         arena,
    const float*            rope_cs_table,
    uint32_t                q_rope_pos,
    float                   softmax_scale,
    int                     sliding_window_size,
    int                     n_kv_tiles,
    int                     slot_idx,
    int                     n_q_head,
    O*                      out,
    const Q_T*              activations,    // [num_active_slots, D_MODEL]
    int                     num_active_slots
) {
    using namespace tile;
    constexpr int HEAD_DIM        = Cfg::HEAD_DIM;
    constexpr int D_MODEL         = Cfg::D_MODEL;
    constexpr int N_PAL           = Cfg::N_PALETTE;
    constexpr int Q_OUTPUT_DIM    = Cfg::Q_OUTPUT_DIM;
    constexpr int K_END           = Q_OUTPUT_DIM + Cfg::K_OUTPUT_DIM;
    constexpr int TOTAL_OUT       = Cfg::TOTAL_OUTPUT_DIM;
    constexpr int N_DIMS_PER_WARP = TOTAL_OUT / 4;
    constexpr int N_K_CHUNKS      = D_MODEL / Arch::MMA_K;
    constexpr int N_TILES_PER_WARP = (N_DIMS_PER_WARP + Arch::MMA_N - 1) / Arch::MMA_N;
    constexpr int VEC             = HEAD_DIM / 32;

    int my_n_dim_start = warp_in_pool * N_DIMS_PER_WARP;

    // ───────────────────────────────────────────────────────────────────
    // PHASE 2 PREP: activation FP→INT8 quant.
    //
    // Each consumer warp covers D_MODEL/4 dims of activations. Per 32-element
    // block: max-abs across 32 lanes (via shfl_xor_sync 1/2/4/8/16) → scale
    // = max/127 → INT8 → store to smem.
    // ───────────────────────────────────────────────────────────────────
    constexpr int ACT_DIMS_PER_WARP = D_MODEL / 4;
    constexpr int ACT_BLOCKS_PER_WARP = ACT_DIMS_PER_WARP / 32;
    {
        const Q_T* act_base = activations + (int64_t)slot_idx * D_MODEL
                                          + (int64_t)warp_in_pool * ACT_DIMS_PER_WARP;
        #pragma unroll
        for (int b = 0; b < ACT_BLOCKS_PER_WARP; ++b) {
            // Each lane reads 1 element of this 32-element block.
            float v = to_f32<Q_T>(act_base[b * 32 + lane]);
            float my_abs = fabsf(v);
            float blk_max = my_abs;
            blk_max = fmaxf(blk_max, __shfl_xor_sync(0xffffffff, blk_max, 1));
            blk_max = fmaxf(blk_max, __shfl_xor_sync(0xffffffff, blk_max, 2));
            blk_max = fmaxf(blk_max, __shfl_xor_sync(0xffffffff, blk_max, 4));
            blk_max = fmaxf(blk_max, __shfl_xor_sync(0xffffffff, blk_max, 8));
            blk_max = fmaxf(blk_max, __shfl_xor_sync(0xffffffff, blk_max, 16));
            float s = blk_max / 127.f;
            if (s == 0.f) s = 1.f;
            float inv = 1.f / s;
            float vq = fminf(fmaxf(v * inv, -127.f), 127.f);
            int8_t q = (int8_t)__float2int_rn(vq);
            int dst_idx = warp_in_pool * ACT_DIMS_PER_WARP + b * 32 + lane;
            arena.as_phase12().activations_int8[dst_idx] = q;
            if (lane == 0) {
                int blk_idx = (warp_in_pool * ACT_DIMS_PER_WARP + b * 32) / 32;
                arena.as_phase12().activations_scales[blk_idx] = s;
            }
        }
    }
    __syncthreads();

    // ───────────────────────────────────────────────────────────────────
    // PHASE 2: QKV projection — INT8 MMA over K-chunks of D_MODEL.
    //
    // For each K-chunk (32 contraction dims), we MMA against W_qkv staging.
    // The N-axis is tiled at W_QKV_TILE_N granularity to fit smem; for each
    // K-chunk we step through N-tiles, accumulating into per-N_tile FP32
    // partial outputs.
    // ───────────────────────────────────────────────────────────────────
    constexpr int N_N_TILES = (TOTAL_OUT + tile::W_QKV_TILE_N - 1) / tile::W_QKV_TILE_N;

    // FP32 accumulator: [n_tile_idx][C_REGS_PER_THREAD]. For decode (M=1),
    // only row 0 is meaningful; we keep all 4 c regs anyway since the MMA
    // produces them.
    float fp32_output[N_N_TILES][Arch::C_REGS_PER_THREAD] = {};

    for (int k_chunk = 0; k_chunk < N_K_CHUNKS; ++k_chunk) {
        // Wait for loader+dequant to deliver this k_chunk's INT8 W staging.
        bar_sync(bar_id::INT8_READY, /*participants=*/4 * 32 + 2 * 32);

        // A fragment: activations for this k_chunk (32 INT8) — same for all
        // consumer warps, broadcast via shfl from the lanes that own the
        // appropriate dims. Per m16n8k32 layout: lane t row 0 holds cols
        // (t%4)*4..(t%4)*4+3 in a[0] and (t%4)*4+16..(t%4)*4+19 in a[2].
        //
        // We pull from the smem activation int8 directly: for one K-chunk,
        // the 32 INT8 activations are at [k_chunk*32..k_chunk*32+31].
        uint32_t a_frag[4];
        const int8_t* a_smem = &arena.as_phase12()
            .activations_int8[k_chunk * Arch::MMA_K];
        // Lane t<4 takes 4 contiguous bytes for cols (t%4)*4..+3 in a[0].
        // Lane t≥4 also reads (filling rows 1-7 of A; values irrelevant for M=1).
        a_frag[0] = *reinterpret_cast<const uint32_t*>(a_smem + (lane & 3) * 4);
        a_frag[1] = 0;  // row 8-15 unused
        a_frag[2] = *reinterpret_cast<const uint32_t*>(a_smem + (lane & 3) * 4 + 16);
        a_frag[3] = 0;
        float scale_A = arena.as_phase12().activations_scales[k_chunk];

        // Iterate over N-tiles. For each W_QKV_TILE_N slice, do MMAs.
        // N_TILES_PER_WARP across warps splits the output dim equally; here
        // each consumer warp handles a slice of W_QKV_TILE_N output dims.
        int stage = k_chunk % N_W_STAGING_STAGES;

        #pragma unroll
        for (int n_tile = 0; n_tile < N_TILES_PER_WARP; ++n_tile) {
            int n_offset = my_n_dim_start + n_tile * Arch::MMA_N;
            if (n_offset >= TOTAL_OUT) break;
            int n_within_tile = n_offset % tile::W_QKV_TILE_N;

            // B fragment: W_qkv[k_chunk*32..+31][n_offset..n_offset+7] from staging.
            // w_staging_int8 layout: [stage][32 K-rows × W_QKV_TILE_N N-cols] flattened
            // with K-major (per stage, 32 rows of W_QKV_TILE_N bytes each).
            const int8_t* b_smem = &arena.as_phase12()
                .w_staging_int8[stage][n_within_tile * Arch::MMA_K];
            float scale_B = arena.as_phase12()
                .w_staging_scales[stage][n_within_tile / 32];

            uint32_t b_frag[2];
            b_frag[0] = *reinterpret_cast<const uint32_t*>(b_smem + (lane & 7) * Arch::MMA_K + (lane >> 3) * 4);
            b_frag[1] = *reinterpret_cast<const uint32_t*>(b_smem + (lane & 7) * Arch::MMA_K + (lane >> 3) * 4 + 16);

            int32_t c_p[4] = {0, 0, 0, 0};
            mma_int8_m16n8k32(c_p, a_frag, b_frag, c_p);

            // Scale to FP32 with combined activation×weight scale and accumulate.
            float s = scale_A * scale_B;
            int n_acc_idx = (n_offset - my_n_dim_start) / Arch::MMA_N;
            if (n_acc_idx < N_N_TILES) {
                #pragma unroll
                for (int r = 0; r < Arch::C_REGS_PER_THREAD; ++r) {
                    fp32_output[n_acc_idx][r] += s * (float)c_p[r];
                }
            }
        }

        bar_arrive(bar_id::INT8_CONSUMED, /*participants=*/4 * 32 + 2 * 32);
    }

    bar_sync(bar_id::PHASE_2_TO_3, /*participants=*/8 * 32);

    // ───────────────────────────────────────────────────────────────────
    // PHASE 3: route owned dims by descriptor.
    //
    // Lane→dim mapping (PTX m16n8k32 C-fragment):
    //   m_in_tile = (lane / 4) + (r & 1) * 8
    //   n_in_tile = (lane % 4) * 2 + (r >> 1)
    // For decode (single token), only m_in_tile==0 is valid: lanes 0-3 with
    // r in {0,2} give n=0..7 covering one MMA's N tile.
    //
    // For each owned (n_tile, r), determine if dim falls in Q/K/V section
    // and route accordingly.
    // ───────────────────────────────────────────────────────────────────
    int8_t  q_int8_buf[N_N_TILES][Arch::C_REGS_PER_THREAD];
    float   scale_Q[N_PAL] = {0.f, 0.f, 0.f, 0.f};
    float   q_max_per_pal[N_PAL] = {0.f, 0.f, 0.f, 0.f};

    // First pass: identify Q dims, accumulate per-palette max-abs (warp-scope).
    #pragma unroll
    for (int n_tile = 0; n_tile < N_TILES_PER_WARP; ++n_tile) {
        int n_offset = my_n_dim_start + n_tile * Arch::MMA_N;
        if (n_offset >= TOTAL_OUT) break;

        if (n_offset < Q_OUTPUT_DIM) {
            #pragma unroll
            for (int r = 0; r < Arch::C_REGS_PER_THREAD; ++r) {
                int m_in_tile = (lane / 4) + (r & 1) * 8;
                if (m_in_tile != 0) continue;
                int n_in_tile = (lane % 4) * 2 + (r >> 1);
                int dim = n_offset + n_in_tile;
                int dim_in_head = dim % HEAD_DIM;
                int pal = dim_in_head / Cfg::DIMS_PER_PALETTE;
                float av = fabsf(fp32_output[n_tile][r]);
                if (av > q_max_per_pal[pal]) q_max_per_pal[pal] = av;
            }
        } else if (n_offset < K_END) {
            // K dims: write FP32 to k_new_fp32 smem. Indexed by kv_offset.
            int kv_offset = n_offset - Q_OUTPUT_DIM;
            #pragma unroll
            for (int r = 0; r < Arch::C_REGS_PER_THREAD; ++r) {
                int m_in_tile = (lane / 4) + (r & 1) * 8;
                if (m_in_tile != 0) continue;
                int n_in_tile = (lane % 4) * 2 + (r >> 1);
                arena.as_phase12().k_new_fp32[kv_offset + n_in_tile]
                    = fp32_output[n_tile][r];
            }
        } else {
            int kv_offset = n_offset - K_END;
            #pragma unroll
            for (int r = 0; r < Arch::C_REGS_PER_THREAD; ++r) {
                int m_in_tile = (lane / 4) + (r & 1) * 8;
                if (m_in_tile != 0) continue;
                int n_in_tile = (lane % 4) * 2 + (r >> 1);
                arena.as_phase12().v_new_fp32[kv_offset + n_in_tile]
                    = fp32_output[n_tile][r];
            }
        }
    }

    // Reduce per-palette max across the warp (only lanes that contributed
    // matter, but warp_reduce_max is fine as zeros for non-Q lanes).
    #pragma unroll
    for (int p = 0; p < N_PAL; ++p) {
        float m = q_max_per_pal[p];
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 1));
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 2));
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 4));
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 8));
        m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, 16));
        scale_Q[p] = m / 127.f;
        if (scale_Q[p] == 0.f) scale_Q[p] = 1.f;
    }

    // Second pass: apply RoPE to Q dims, quantize.
    //
    // RoPE on a register-distributed Q is non-trivial because the C-fragment
    // layout doesn't match the v2 RoPE helper's expected lane/VEC layout.
    // For now we skip RoPE on the fused-Q path and route through smem to
    // re-gather lane-distributed FP32 Q in v2 layout, then apply RoPE.
    //
    // SMEM scratch: reuse k_new_fp32 region briefly (it's already populated
    // for the same number of dims; we use a separate buffer).
    //
    // Phase B note: implementing this properly requires either an alternate
    // RoPE helper that consumes the C-fragment layout directly, or a smem
    // round-trip. We sketch the round-trip path below but keep it gated until
    // we add a dedicated Q smem buffer to Phase12View.
    //
    // For now: write Q in C-fragment order to a temporary smem (placeholder
    // — would require allocating buffer; commented out), apply RoPE in v2
    // layout, re-quantize.
    //
    // Iteration 4 placeholder: skip RoPE on Q, quantize C-fragment FP32
    // directly. RoPE will be applied to K_new in Phase 3 (dequant role) so
    // K-side is correct; Q-side mismatch will produce wrong attention output
    // until the smem round-trip is implemented.

    #pragma unroll
    for (int n_tile = 0; n_tile < N_TILES_PER_WARP; ++n_tile) {
        int n_offset = my_n_dim_start + n_tile * Arch::MMA_N;
        if (n_offset >= Q_OUTPUT_DIM) continue;
        #pragma unroll
        for (int r = 0; r < Arch::C_REGS_PER_THREAD; ++r) {
            int m_in_tile = (lane / 4) + (r & 1) * 8;
            if (m_in_tile != 0) continue;
            int n_in_tile = (lane % 4) * 2 + (r >> 1);
            int dim = n_offset + n_in_tile;
            int dim_in_head = dim % HEAD_DIM;
            int pal = dim_in_head / Cfg::DIMS_PER_PALETTE;
            float scaled = fp32_output[n_tile][r] / scale_Q[pal];
            float clamped = fminf(fmaxf(scaled, -127.f), 127.f);
            q_int8_buf[n_tile][r] = (int8_t)__float2int_rn(clamped);
        }
    }

    bar_sync(bar_id::PHASE_3_TO_4, /*participants=*/8 * 32);

    // ───────────────────────────────────────────────────────────────────
    // PHASE 4: attention — same shape as int8_decode_kernel's Phase 4 but
    // with INT8 Q taken from q_int8_buf and INT8 K/V taken from
    // arena.as_phase4().smem_int8_K / smem_int8_V (populated by dequant role).
    //
    // For brevity we skip the full attention loop here; in a real
    // implementation we would lift the iter-3 Phase 4 logic into a shared
    // helper and call it from both int8_decode_kernel and consumer_role.
    //
    // Phase B placeholder: use OnlineSoftmaxState scaffold and emit zeros.
    // ───────────────────────────────────────────────────────────────────
    OnlineSoftmaxState softmax_state;
    softmax_state.init();

    constexpr int OUT_DIMS_PER_WARP = HEAD_DIM / 4;
    float out_accum[OUT_DIMS_PER_WARP / 32 > 0 ? OUT_DIMS_PER_WARP / 32 : 1] = {};

    for (int t_tile = 0; t_tile < n_kv_tiles; ++t_tile) {
        bar_sync(bar_id::INT8_READY, /*participants=*/4 * 32 + 2 * 32);

        // Phase B TODO: full INT8 MMA QK^T + softmax + INT8 PV here. Borrow
        // the structure from int8_decode_kernel.cuh's process_tile.
        //
        // For now we keep the running accumulator at zero and just step the
        // softmax state so the bar_arrive sequence completes.
        float dummy_logit = -1e38f;
        float dummy_arr[1] = {dummy_logit};
        softmax_state.update(dummy_arr);

        if constexpr (Cfg::USE_SLIDING_WINDOW) {
            // sliding_window guard, unused while attention loop is stubbed
            (void)sliding_window_size;
        }

        bar_arrive(bar_id::INT8_CONSUMED, /*participants=*/4 * 32 + 2 * 32);
    }

    // Output writeback: lanes 0..7 of the consumer pool together cover
    // HEAD_DIM dims of one Q head. For decode (single token, n_q_head heads
    // packed across CTAs and warps), we write to out[slot, head, dim].
    //
    // Phase B placeholder: write zeros of correct shape. Real implementation
    // applies softmax_state.normalizer() and uses out_accum.
    if (warp_in_pool < n_q_head / Cfg::N_KV_HEADS) {
        int heads_per_group = n_q_head / Cfg::N_KV_HEADS;
        if (heads_per_group < 1) heads_per_group = 1;
        // Each consumer warp's slice of HEAD_DIM dims for one head pair.
        // Without a finished attention loop we just zero out the output region.
        int head_idx_base = (slot_idx % 1) * heads_per_group; // placeholder
        (void)head_idx_base; (void)out;
    }
    (void)num_active_slots;
    (void)q_int8_buf; (void)scale_Q;
    (void)q_rope_pos; (void)softmax_scale; (void)rope_cs_table;
    (void)out_accum;
}

} // namespace fused_attn
