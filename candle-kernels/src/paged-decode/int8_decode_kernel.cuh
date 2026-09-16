#pragma once
// =============================================================================
// int8_decode_kernel.cuh — v2-API-compatible decode-attention kernel (Track A).
//
// Drop-in for v2's paged_decode_attn, computing the QK^T as an INT8 m16n8k32 MMA
// per palette (lane-collective INT8 dot fallback for head dims whose palette
// isn't 32-wide) and the PV in INT8, with a per-32-token tile-batched softmax
// and the §1A V read-through. Same slot-header / paged-arena interface as v2.
//
// # When this kernel faults and the address does not say where
//
// The kernels are built *without* `--generate-line-info`, so a fault here
// surfaces as a bare device address rather than a file and line. Rebuild with
// the `kernel-lineinfo` feature for the debugging session:
//
//     cargo test -p candle-transformers --features cuda,kernel-lineinfo ...
//
// then drop it again. It is off by default because it is two thirds of the
// compiled archive — 175 KB of `.text` against 592 KB of debug sections in a
// measured cubin — and every CUDA test binary links a copy. See
// `candle-kernels/build_utils.rs`.
// =============================================================================

#include <assert.h>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>
#include <type_traits>

#include "../fast_exp.cuh"
#include "../arena_table.cuh"
#include "../simple/warp_reduce.cuh"
#include "../convert/convert_all.cuh"
#include "../blocks.cuh"
#include "slot_types.cuh"
#include "pal_iter.cuh"
// Shared decode helpers (vec2_traits, load_vec2, cp_async_*, RoPE, scatter,
// write-len commit) — formerly inline in the V2 paged_decode_kernel.cuh.
#include "decode_helpers.cuh"
#include "int8_decode_emit.cuh"
#include "../mma/mma_wrappers.cuh"
// QSA block-sparse selection: `sel.entries == nullptr` for every model whose
// attention layers read the whole causal prefix, which is all of them but
// Qwen3.8-Flash-Next's 12 full-attention layers.
#include "../qsa_select.cuh"
// Wide-head (HEAD_DIM 256) decode on the INT8 tensor cores: one slice per
// tile, the group's query heads as the MMA's M rows.
#include "int8_decode_tile_kernel.cuh"

namespace fused_attn {

/// Ceiling on the split-KV factor: the partial pool is sized to it and the
/// combine compacts a row's partials through shared arrays of this length.
constexpr int MAX_SPLITS = 256;

// QK^T is computed with the m16n8k32 INT8 MMA when a palette spans exactly 32
// dims (HEAD_DIM==128 → SUB_HEAD_DIM==32); for other head dims (e.g. hd64) it
// falls back to the lane-collective INT8 dot — see USE_MMA_QK below. The PV is
// INT8 throughout. V is read straight through from native-int8 arenas where the
// format allows (§1A); otherwise it is dequantized to FP and re-quantized to
// int8 for the PV.
template <typename Q_T, typename T, typename O,
          int HEAD_DIM, int WARPS_PER_BLOCK,
          int TILE_K = 32, int NUM_STAGES = 2,
          bool USE_TC = false, bool ROPE_INTERLEAVED = false>
__device__ __forceinline__ void int8_decode_attn_impl(
    const Q_T* __restrict__ q,
    const uint8_t* __restrict__ headers_ptr,
    O* __restrict__ out,
    int num_active_slots,
    int n_q_head,
    int n_kv_head,
    float softmax_scale,
    const T* __restrict__ k_new,
    const T* __restrict__ v_new,
    const float* __restrict__ rope_cs,
    float* __restrict__ partial_acc,   // split-KV: [slot*n_q_head+qh][split][HEAD_DIM] un-normalized ΣwV; nullptr → write final
    float* __restrict__ partial_ml,    // split-KV: [slot*n_q_head+qh][split][2] = (m, l)
    QsaSel sel                         // QSA: one selection row per SLOT (decode is one query per slot)
) {
    constexpr int VEC = HEAD_DIM / WARP_SIZE;
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be multiple of 32");
    static_assert(VEC <= 8, "HEAD_DIM must be <= 256");
    static_assert(NUM_STAGES >= 1 && NUM_STAGES <= 3, "NUM_STAGES must be 1-3");
    static_assert(CHUNK_SIZE % WARPS_PER_BLOCK == 0,
        "CHUNK_SIZE must be a multiple of WARPS_PER_BLOCK");

    int slot_idx = (int)blockIdx.x;
    int kv_head_idx = (int)blockIdx.y;
    int split_idx = (int)blockIdx.z;
    int num_splits = (int)gridDim.z;
    int tid = (int)threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    if (slot_idx >= num_active_slots || kv_head_idx >= n_kv_head) return;

    // Emit a warp's result: in split-KV mode (partial_acc != nullptr) write the
    // un-normalized partial (ΣwV, m, l) for this split; otherwise normalize and
    // write the final output. The combine kernel merges the per-split partials.
    auto emit_result = [&](int qh, const float* oreg, float mval, float lval, bool active) {
        if (!active) return;
        if (partial_acc != nullptr) {
            int64_t base = ((int64_t)slot_idx * n_q_head + qh) * num_splits + split_idx;
            float* acc = partial_acc + base * HEAD_DIM;
            #pragma unroll
            for (int j = 0; j < VEC; ++j) acc[lane * VEC + j] = oreg[j];
            if (lane == 0) { partial_ml[base * 2] = mval; partial_ml[base * 2 + 1] = lval; }
        } else {
            float inv_l = __fdividef(1.f, fmaxf(lval, 1e-10f));
            O* out_ptr = out + ((int64_t)slot_idx * (int64_t)n_q_head + (int64_t)qh) * (int64_t)HEAD_DIM;
            #pragma unroll
            for (int j = 0; j < VEC; ++j) out_ptr[lane * VEC + j] = from_f32<O>(oreg[j] * inv_l);
        }
    };

    const SlotHeader& slot = get_slot_header(headers_ptr, slot_idx);
    // Positions end at the write slice (see `resolve_pos`): the chunks after
    // it are empty capacity whose `rope` nothing keeps current, and a
    // rope-ordered search that reached one would take it for the owner of
    // positions the writer holds.
    const uint32_t n_slices  = min(slot.n_slices, slot.write_slice + 1u);
    const uint32_t write_slice_idx = slot.write_slice;
    const uint64_t slices_ptr = slot.slices_ptr;

    if (n_slices == 0) {
        int heads_per_group = n_q_head / n_kv_head;
        if (heads_per_group <= 0) heads_per_group = 1;
        int head_idx = kv_head_idx * heads_per_group + warp;
        bool warp_active = (warp < heads_per_group) && (head_idx < n_q_head);
        float zero_reg[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) zero_reg[j] = 0.f;
        emit_result(head_idx, zero_reg, -1e38f, 0.f, warp_active);
        return;
    }

    uint8_t* write_slice_ptr = get_slice_mut<HEAD_DIM>(slices_ptr, (int)write_slice_idx, n_kv_head);
    const uint16_t ws_offset = slice_offset(write_slice_ptr);
    const uint16_t ws_len    = slice_len(write_slice_ptr);
    const uint32_t ws_rope   = slice_rope(write_slice_ptr);

    // ─── Fused KV scatter (warp 0 only) ────────────────────────────────
    {
        const int within = (int)ws_offset + (int)ws_len;
        constexpr int LANES_PER_PAL = WARP_SIZE / N_PALETTE;
        constexpr int SUB_HEAD_DIM  = HEAD_DIM / N_PALETTE;

        if (warp == 0 && within < CHUNK_SIZE) {
            const uint8_t* head_ptr = get_head<HEAD_DIM>(write_slice_ptr, kv_head_idx);
            int pal = lane / LANES_PER_PAL;
            int local_lane = lane % LANES_PER_PAL;
            uint64_t k_ptr_p = kvhead_k_ptr<HEAD_DIM>(head_ptr, pal);
            uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(head_ptr, pal);
            int k_fmt = kvhead_k_fmt<HEAD_DIM>(head_ptr, pal);
            int v_fmt = kvhead_v_fmt<HEAD_DIM>(head_ptr, pal);
            if (k_ptr_p != 0) {
                char* k_arena = (char*)(uintptr_t)k_ptr_p;
                char* v_arena = (char*)(uintptr_t)v_ptr_p;
                int k_esz = ArenaFormat::float_elem_size(k_fmt);
                int v_esz = ArenaFormat::float_elem_size(v_fmt);
                int64_t src_base = ((int64_t)slot_idx * (int64_t)n_kv_head + (int64_t)kv_head_idx) * (int64_t)HEAD_DIM;
                const T* k_src = k_new + src_base;
                const T* v_src = v_new + src_base;
                float k_regs[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; ++j)
                    k_regs[j] = to_f32<T>(k_src[lane * VEC + j]);
                if (k_fmt == ArenaFormat::R16) {
                    int heads_per_group_w = n_q_head / n_kv_head;
                    if (heads_per_group_w < 1) heads_per_group_w = 1;
                    int q_head = kv_head_idx * heads_per_group_w;
                    int64_t q_base = ((int64_t)slot_idx * (int64_t)n_q_head + (int64_t)q_head) * (int64_t)HEAD_DIM;
                    float q_regs[VEC];
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j)
                        q_regs[j] = to_f32<Q_T>(q[q_base + lane * VEC + j]);
                    write_regs_to_r16<VEC>(k_arena, /*chunk_byte_offset=*/0, within, local_lane, k_regs, q_regs);
                } else if (k_esz > 0) {
                    int64_t eo = (int64_t)within * SUB_HEAD_DIM;
                    write_regs_to_arena<VEC>(k_arena, eo, local_lane, k_esz, k_fmt, k_regs);
                }
                float v_regs[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; ++j)
                    v_regs[j] = to_f32<T>(v_src[lane * VEC + j]);
                if (v_esz > 0) {
                    int64_t eo_v = (int64_t)within * SUB_HEAD_DIM;
                    write_regs_to_arena<VEC>(v_arena, eo_v, local_lane, v_esz, v_fmt, v_regs);
                }
            }
        }
        __syncthreads();
    }

    int heads_per_group = n_q_head / n_kv_head;
    if (heads_per_group <= 0) heads_per_group = 1;
    int head_idx = kv_head_idx * heads_per_group + warp;
    bool warp_active = (warp < heads_per_group) && (head_idx < n_q_head);

    int kv_len = (int)ws_rope + (int)ws_len + 1;
    if (kv_len <= 0) {
        float zero_reg[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) zero_reg[j] = 0.f;
        emit_result(head_idx, zero_reg, -1e38f, 0.f, warp_active);
        return;
    }
    int max_len = (int)n_slices * CHUNK_SIZE;
    if (kv_len > max_len) kv_len = max_len;

    // ─── Q load + RoPE ─────────────────────────────────────────────────
    float q_reg[VEC];
    if (warp_active) {
        const Q_T* q_ptr = q + ((int64_t)slot_idx * (int64_t)n_q_head + (int64_t)head_idx) * (int64_t)HEAD_DIM;
        #pragma unroll
        for (int j = 0; j < VEC; ++j)
            q_reg[j] = to_f32<Q_T>(q_ptr[lane * VEC + j]);
    } else {
        #pragma unroll
        for (int j = 0; j < VEC; ++j) q_reg[j] = 0.f;
    }
    {
        uint32_t q_rope_pos = (uint32_t)ws_rope + (uint32_t)ws_len;
        if constexpr (ROPE_INTERLEAVED && (VEC == 1 || VEC % 2 == 0)) {
            apply_rope_interleaved_f32<VEC, HEAD_DIM>(q_reg, lane, (int)q_rope_pos, rope_cs);
        } else {
            apply_rope_rotary_f32<VEC, HEAD_DIM>(q_reg, lane, (int)q_rope_pos, rope_cs);
        }
    }

    // Per-slice tiling (gap-aware). Each slice contributes ceil(eff_len/WARPS)
    // tiles, every tile entirely within one 32-token chunk; a global tile maps to
    // (slice, within_base = off + tile_in_slice*WARPS) by a forward scan. This is
    // what lets a sealed partial chunk's empty tail (the substrate-seal gap) be
    // skipped and the writer slice be reached at its true physical position
    // rather than aliased into the gap by a flat chunk_div(logical). Gapless
    // sequences (every slice full) tile identically to the old flat walk.
    auto slice_eff_len = [&](int s) -> int {
        const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
        int len = (int)slice_len(sl);
        int off = (int)slice_offset(sl);
        if (s == (int)write_slice_idx && len < CHUNK_SIZE && off + len < CHUNK_SIZE) len += 1;
        return len;
    };
    auto slice_tiles = [&](int s) -> int {
        return (slice_eff_len(s) + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    };
    // tile_idx -> (slice, within_base). The warp's token sits at within_base + warp.
    auto tile_to_slice = [&](int tile_idx, int& slice_out, int& within_base_out) {
        int base = 0, s = 0;
        while (s + 1 < (int)n_slices) {
            int st = slice_tiles(s);
            if (base + st <= tile_idx) {
                base += st;
                ++s;
            } else {
                break;
            }
        }
        slice_out = s;
        const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
        within_base_out = (int)slice_offset(sl) + (tile_idx - base) * WARPS_PER_BLOCK;
    };

    // Per-tile palette iterators (refresh on slice boundary).
    PalIter<VEC, HEAD_DIM> ki, vi;
    int kv_pal_slice_idx = -1;
    auto maybe_init_kv_iters_for_tile = [&](int tile_idx) {
        int tile_slice_idx, tile_within_base;
        tile_to_slice(tile_idx, tile_slice_idx, tile_within_base);
        (void)tile_within_base;
        if (tile_slice_idx != kv_pal_slice_idx && tile_slice_idx < (int)n_slices) {
            const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, tile_slice_idx, n_kv_head);
            const uint8_t* head_ptr = get_head<HEAD_DIM>(sl, kv_head_idx);
            ki.init(kvhead_k_pal_map<HEAD_DIM>(head_ptr), lane);
            vi.init(kvhead_v_pal_map<HEAD_DIM>(head_ptr), lane);
            kv_pal_slice_idx = tile_slice_idx;
        }
    };

    constexpr int SMEM_PAD = 8;
    static_assert((HEAD_DIM + SMEM_PAD) * sizeof(T) % 16 == 0,
                  "SMEM_PAD breaks 16-byte alignment");
    __shared__ alignas(128) T shared_k[NUM_STAGES][WARPS_PER_BLOCK][HEAD_DIM + SMEM_PAD];
    __shared__ alignas(128) T shared_v[NUM_STAGES][WARPS_PER_BLOCK][HEAD_DIM + SMEM_PAD];

    // INT8 K storage and per-token-per-palette scales for the INT8 dot path.
    // Shape: [stage][token=warp][dim contiguous], plus N_PALETTE scales per token.
    __shared__ alignas(128) int8_t shared_k_int8[NUM_STAGES][WARPS_PER_BLOCK][HEAD_DIM];
    __shared__ alignas(16)  float  shared_k_scale[NUM_STAGES][WARPS_PER_BLOCK][N_PALETTE];

    // INT8 V + per-token scale (single scale per V token; V is "value tokens"
    // contributing equally across head_dim). Used by the FP→INT8 V path.
    __shared__ alignas(128) int8_t shared_v_int8[NUM_STAGES][WARPS_PER_BLOCK][HEAD_DIM];
    __shared__ alignas(16)  float  shared_v_scale[NUM_STAGES][WARPS_PER_BLOCK];

    // V skip-dequant (Track A §1A): when the V arena is natively INT8
    // (Q8_0/Q4_0) for all palettes, V int8 is read straight through with no FP
    // round-trip into shared_v_int8 in PALETTE order, and the per-(dim,block)
    // scale lands here (one per dim — all 32 tokens of a chunk share it). The
    // PV gathers V via the `vi` palette iterator and applies the per-dim scale.
    // shared_v_readthrough[stage] flags the mode chosen for that tile's slice.
    __shared__ alignas(16) float shared_v_dim_scale[NUM_STAGES][HEAD_DIM];
    __shared__ int shared_v_readthrough[NUM_STAGES];

    // Tile-batched logits buffer for INT8 MMA / manual-dot paths. Each warp
    // owns one Q head, so the buffer must be PER-WARP — otherwise the 3+
    // active warps race-overwrite each others' logits and softmax sees the
    // wrong head's scores. Indexed [stage][q_warp][k_token].
    __shared__ alignas(16) float tile_logits[NUM_STAGES][WARPS_PER_BLOCK][WARPS_PER_BLOCK];

    constexpr int SUB_HEAD_DIM = HEAD_DIM / N_PALETTE;

    // The INT8 QK^T MMA (mma.m16n8k32) contracts exactly 32 K-columns per
    // palette, and its A/B fragment assembly (q_packed of VEC int8, b_frag[1]
    // at +16) is hardwired for SUB_HEAD_DIM == 32 (i.e. HEAD_DIM == 128, VEC==4).
    // For any other head dim (e.g. HEAD_DIM=64 → SUB_HEAD_DIM=16, VEC=2) the
    // m16n8k32 fragment straddles two palettes and reads past q_int8[VEC], which
    // structurally corrupts the logits. Fall back to the manual per-lane INT8
    // dot (correct for any VEC) whenever the palette isn't exactly 32 dims.
    constexpr bool USE_MMA_QK = (SUB_HEAD_DIM == 32);

    // ─── Per-palette Q quantization (warp-collective) ──────────────────
    // q_reg holds VEC=HEAD_DIM/32 dims per lane; lane t covers dims [t*VEC..t*VEC+VEC).
    // Each palette p covers dims [p*SUB_HEAD_DIM..(p+1)*SUB_HEAD_DIM) which spans
    // the lanes [p*8 .. p*8+7] when VEC=4. Max-abs is reduced inside that 8-lane
    // group via xor-1/2/4, leaving every member with the palette's max-abs.
    int8_t q_int8[VEC];
    uint32_t q_packed = 0;  // packed VEC=4 INT8 of this lane's Q dims (for MMA shuffles)
    float scale_Q[N_PALETTE];
    {  // Per-palette Q INT8 quantization.
        float my_max = 0.f;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float a = fabsf(q_reg[j]);
            if (a > my_max) my_max = a;
        }
        float pal_max = my_max;
        pal_max = fmaxf(pal_max, __shfl_xor_sync(0xffffffff, pal_max, 1));
        pal_max = fmaxf(pal_max, __shfl_xor_sync(0xffffffff, pal_max, 2));
        pal_max = fmaxf(pal_max, __shfl_xor_sync(0xffffffff, pal_max, 4));

        // Broadcast palette p's max from lane p*8 to all lanes, derive scale.
        #pragma unroll
        for (int p = 0; p < N_PALETTE; ++p) {
            float pmax = __shfl_sync(0xffffffff, pal_max, p * 8);
            float s = pmax / 127.f;
            if (s == 0.f) s = 1.f;
            scale_Q[p] = s;
        }
        // Quantize this lane's q_reg using its own palette's scale.
        int my_pal = lane / 8;
        float inv = 1.f / scale_Q[my_pal];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            float v = q_reg[j] * inv;
            v = fminf(fmaxf(v, -127.f), 127.f);
            q_int8[j] = (int8_t)__float2int_rn(v);
        }
        // Pack into a single uint32_t for MMA fragment assembly via shuffles.
        // Bytes: q_int8[0] is LSB, q_int8[3] is MSB (little-endian on NVIDIA).
        // Only assembled (and only valid) when the MMA path runs, which requires
        // VEC == 4 — reading q_int8[2]/[3] for VEC < 4 would be out of bounds.
        if constexpr (USE_MMA_QK) {
            q_packed = ((uint32_t)(uint8_t)q_int8[0])
                     | ((uint32_t)(uint8_t)q_int8[1] << 8)
                     | ((uint32_t)(uint8_t)q_int8[2] << 16)
                     | ((uint32_t)(uint8_t)q_int8[3] << 24);
        }
    }

    float m_i = -1e38f;
    float l_i = 0.f;
    float out_reg[VEC];
    #pragma unroll
    for (int j = 0; j < VEC; ++j) out_reg[j] = 0.f;

    int n_tiles = 0;
    for (int s = 0; s < (int)n_slices; ++s) n_tiles += slice_tiles(s);
    // Split-KV: this block processes the contiguous tile sub-range [tile_lo,
    // tile_hi). num_splits==1 → the whole [0, n_tiles). An empty range
    // (tile_lo >= n_tiles) runs no tiles and emits a null partial (m=-inf, l=0)
    // from the initial m_i/l_i/out_reg, which the combine ignores.
    const int tiles_per_split = (n_tiles + num_splits - 1) / num_splits;
    int tile_lo = split_idx * tiles_per_split;
    int tile_hi = tile_lo + tiles_per_split;
    if (tile_lo > n_tiles) tile_lo = n_tiles;
    if (tile_hi > n_tiles) tile_hi = n_tiles;
    constexpr int BLOCKS_PER_DIM = CHUNK_SIZE / 32;

    auto load_tile = [&](int tile_idx, int stage) {
        T* k_dst = shared_k[stage][warp];
        T* v_dst = shared_v[stage][warp];
        // All WARPS tokens of a tile live in one slice; the warp's token is at
        // within = within_base + warp, valid while it is below the slice's filled
        // count (slice_eff_len already folds in the writer's +1).
        int my_slice_idx, within_base;
        tile_to_slice(tile_idx, my_slice_idx, within_base);
        int within = within_base + warp;
        bool valid_k = my_slice_idx < (int)n_slices;
        if (valid_k) {
            const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, my_slice_idx, n_kv_head);
            int off = (int)slice_offset(sl);
            valid_k = within < off + slice_eff_len(my_slice_idx);
        }
        if (!valid_k) {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                k_dst[lane * VEC + j] = from_f32<T>(0.f);
                v_dst[lane * VEC + j] = from_f32<T>(0.f);
            }
            {
                int8_t* k_int8_dst = shared_k_int8[stage][warp];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    k_int8_dst[lane * VEC + j] = 0;
                }
                if (lane < N_PALETTE) {
                    shared_k_scale[stage][warp][lane] = 1.f;
                }
            }
            {
                int8_t* v_int8_dst = shared_v_int8[stage][warp];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) v_int8_dst[lane * VEC + j] = 0;
                if (lane == 0) shared_v_scale[stage][warp] = 1.f;
            }
            return;
        }
        constexpr int64_t sub_head_stride = (int64_t)SUB_HEAD_DIM * CHUNK_SIZE;
        const uint8_t* sl_ptr = get_slice<HEAD_DIM>(slices_ptr, my_slice_idx, n_kv_head);
        const uint8_t* head_ptr = get_head<HEAD_DIM>(sl_ptr, kv_head_idx);

        // V skip-dequant eligibility (Track A §1A): read V straight from the
        // arena (no FP round-trip) only when EVERY palette's V format is a
        // passthrough int8 family (Q8_0/Q4_0/Q5_0/Q2_0/Q3_0/Q4_KS/Q8_KS).
        // Mixed/asymmetric/FP formats keep the dequant→requant path. K never
        // skips — RoPE needs FP.
        bool v_readthrough = true;
        #pragma unroll
        for (int p = 0; p < N_PALETTE; ++p) {
            int vf = kvhead_v_fmt<HEAD_DIM>(head_ptr, p);
            if (!ArenaAccessor::is_int8_readthrough_format(vf)) v_readthrough = false;
        }
        if (lane == 0) shared_v_readthrough[stage] = v_readthrough ? 1 : 0;

        for (int p = 0; p < N_PALETTE; ++p) {
            uint64_t k_ptr_p = kvhead_k_ptr<HEAD_DIM>(head_ptr, p);
            uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(head_ptr, p);
            int k_fmt = kvhead_k_fmt<HEAD_DIM>(head_ptr, p);
            int v_fmt = kvhead_v_fmt<HEAD_DIM>(head_ptr, p);
            float k_scale_p = kvhead_k_scale<HEAD_DIM>(head_ptr, p);
            float v_scale_p = kvhead_v_scale<HEAD_DIM>(head_ptr, p);
            ArenaAccessor k_acc((const char*)(uintptr_t)k_ptr_p, k_fmt, sub_head_stride, sub_head_stride, BLOCKS_PER_DIM, 0);
            k_acc.template load_head_scaled<T, SUB_HEAD_DIM, USE_TC>(k_dst + p * SUB_HEAD_DIM, 0, 0, within, lane, k_scale_p);
            ArenaAccessor v_acc((const char*)(uintptr_t)v_ptr_p, v_fmt, sub_head_stride, sub_head_stride, BLOCKS_PER_DIM, 0);
            // V load. The per-tile gate `v_readthrough` (= is_int8_readthrough_format
            // over ALL palettes) selects the layout — read straight through (the
            // dispatcher hides the format switch, like load_head_scaled does) into
            // PALETTE-order int8 + per-dim block scale, or FP dequant for a
            // non-passthrough tile. Tile-uniform because the PV needs one layout per
            // tile. apply_rope_to_tile skips V quant on the read-through path; the PV
            // gathers via vi. K never skips (RoPE needs FP).
            if (v_readthrough) {
                v_acc.template load_head_int8_readthrough<SUB_HEAD_DIM>(
                    shared_v_int8[stage][warp] + p * SUB_HEAD_DIM,
                    shared_v_dim_scale[stage] + p * SUB_HEAD_DIM,
                    0, 0, within, lane, v_scale_p);
            } else {
                v_acc.template load_head_scaled<T, SUB_HEAD_DIM, USE_TC>(v_dst + p * SUB_HEAD_DIM, 0, 0, within, lane, v_scale_p);
            }
        }
    };

    auto apply_rope_to_tile = [&](int tile_idx, int stage) {
        int my_slice_idx, within_base;
        tile_to_slice(tile_idx, my_slice_idx, within_base);
        int within = within_base + warp;
        const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, my_slice_idx, n_kv_head);
        int off = (int)slice_offset(sl);
        if (my_slice_idx < (int)n_slices && within < off + slice_eff_len(my_slice_idx)) {
            {
                const int32_t rope_base = (int32_t)slice_rope(sl);
                const int32_t rope_pos  = rope_base + (within - off);
                T* k_dst = shared_k[stage][warp];
                float k_regs[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; ++j)
                    k_regs[j] = to_f32<T>(k_dst[ki[j]]);
                if constexpr (ROPE_INTERLEAVED && (VEC == 1 || VEC % 2 == 0)) {
                    apply_rope_interleaved_f32<VEC, HEAD_DIM>(k_regs, lane, rope_pos, rope_cs);
                } else {
                    apply_rope_rotary_f32<VEC, HEAD_DIM>(k_regs, lane, rope_pos, rope_cs);
                }
                __syncwarp();
                #pragma unroll
                for (int j = 0; j < VEC; ++j)
                    k_dst[lane * VEC + j] = from_f32<T>(k_regs[j]);

                // ─── INT8 K quantization (per-palette scale per token) ─
                // After RoPE: lane t holds K dims [t*VEC..t*VEC+VEC). Same
                // palette geometry as Q: lanes [p*8..p*8+7] cover palette p.
                {
                    float my_max = 0.f;
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j) {
                        float a = fabsf(k_regs[j]);
                        if (a > my_max) my_max = a;
                    }
                    float pal_max = my_max;
                    pal_max = fmaxf(pal_max, __shfl_xor_sync(0xffffffff, pal_max, 1));
                    pal_max = fmaxf(pal_max, __shfl_xor_sync(0xffffffff, pal_max, 2));
                    pal_max = fmaxf(pal_max, __shfl_xor_sync(0xffffffff, pal_max, 4));
                    int my_pal = lane / 8;
                    float my_scale = pal_max / 127.f;
                    if (my_scale == 0.f) my_scale = 1.f;
                    float inv = 1.f / my_scale;
                    if ((lane & 7) == 0) {
                        shared_k_scale[stage][warp][my_pal] = my_scale;
                    }
                    int8_t* k_int8_dst = shared_k_int8[stage][warp];
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j) {
                        float v = k_regs[j] * inv;
                        v = fminf(fmaxf(v, -127.f), 127.f);
                        k_int8_dst[lane * VEC + j] = (int8_t)__float2int_rn(v);
                    }
                }

                // V FP→INT8 path. SKIPPED entirely when the V arena was read
                // straight through as int8 in load_tile (§1A): shared_v_int8 is
                // already populated (palette order) with per-dim scales. The
                // branch is warp-uniform (flag set per tile/slice in load_tile).
                if (!shared_v_readthrough[stage]) {
                    T* v_dst = shared_v[stage][warp];
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j)
                        k_regs[j] = to_f32<T>(v_dst[vi[j]]);
                    __syncwarp();
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j)
                        v_dst[lane * VEC + j] = from_f32<T>(k_regs[j]);

                    // ─── INT8 V quantization (single per-token scale) ────────
                    // V is consumed in PV as B[token=K][dim=N]. We use one scale
                    // per token (max-abs across the 128 dims) — coarser than K's
                    // per-palette but PV is less sensitive than QK^T per design §3.3.
                    {
                        float my_max = 0.f;
                        #pragma unroll
                        for (int j = 0; j < VEC; ++j) {
                            float a = fabsf(k_regs[j]);
                            if (a > my_max) my_max = a;
                        }
                        float tok_max = my_max;
                        tok_max = fmaxf(tok_max, __shfl_xor_sync(0xffffffff, tok_max, 1));
                        tok_max = fmaxf(tok_max, __shfl_xor_sync(0xffffffff, tok_max, 2));
                        tok_max = fmaxf(tok_max, __shfl_xor_sync(0xffffffff, tok_max, 4));
                        tok_max = fmaxf(tok_max, __shfl_xor_sync(0xffffffff, tok_max, 8));
                        tok_max = fmaxf(tok_max, __shfl_xor_sync(0xffffffff, tok_max, 16));
                        float v_scale = tok_max / 127.f;
                        if (v_scale == 0.f) v_scale = 1.f;
                        if (lane == 0) shared_v_scale[stage][warp] = v_scale;
                        float inv = 1.f / v_scale;
                        int8_t* v_int8_dst = shared_v_int8[stage][warp];
                        #pragma unroll
                        for (int j = 0; j < VEC; ++j) {
                            float vf = k_regs[j] * inv;
                            vf = fminf(fmaxf(vf, -127.f), 127.f);
                            v_int8_dst[lane * VEC + j] = (int8_t)__float2int_rn(vf);
                        }
                    }
                }
            }
        }
    };

    // QSA: whether this slot restricts its read. Slot-uniform, so it is one
    // test hoisted out of the tile loop rather than a branch per key.
    const bool qsa_on = qsa_active(sel) && !qsa_row_dense(sel, slot_idx);

    auto process_tile = [&](int tile_idx, int stage) {
        int tile_slice, tile_within_base;
        tile_to_slice(tile_idx, tile_slice, tile_within_base);
        // tile_off/tile_bv frame the in-chunk validity window for the WARPS
        // tokens of this tile; slice_eff_len already folds in the writer's +1, so
        // a token is valid while within_base + t < tile_off + tile_bv.
        // tile_rope is the slice's base position, so token t's LOGICAL position
        // — what the selection is indexed by — is tile_rope + within − tile_off.
        uint32_t tile_off = 0;
        uint32_t tile_bv = (uint32_t)CHUNK_SIZE;
        int32_t tile_rope = 0;
        if (tile_slice < (int)n_slices) {
            const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, tile_slice, n_kv_head);
            tile_off = (uint32_t)slice_offset(sl);
            tile_bv = (uint32_t)slice_eff_len(tile_slice);
            tile_rope = (int32_t)slice_rope(sl);
        }
        // A token this slot does not select is masked exactly as an
        // out-of-range one: score −inf ⇒ weight 0. Evaluated once per token
        // here and read by both softmax phases below.
        auto tok_valid = [&](int t) {
            int actual_within = tile_within_base + t;
            bool ok = (tile_slice < (int)n_slices &&
                       actual_within < (int)(tile_off + tile_bv));
            if (ok && qsa_on) {
                ok = qsa_selects(sel, slot_idx, tile_rope + actual_within - (int)tile_off);
            }
            return ok;
        };
        // ── QK^T: precompute the INT8 logits, broadcast via tile_logits[].
        if constexpr (USE_MMA_QK) {
            if (warp_active) {
                float acc_lo = 0.f;
                float acc_hi = 0.f;
                #pragma unroll
                for (int p = 0; p < N_PALETTE; ++p) {
                    uint32_t a_frag[4];
                    int src0 = p * 8 + (lane & 3);
                    int src1 = p * 8 + 4 + (lane & 3);
                    a_frag[0] = __shfl_sync(0xffffffff, q_packed, src0);
                    a_frag[1] = 0;
                    a_frag[2] = __shfl_sync(0xffffffff, q_packed, src1);
                    a_frag[3] = 0;

                    uint32_t b_frag[2];
                    {
                        // PTX m16n8k32 .s8 col-major B layout:
                        //   lane t covers N-row = t/4 (0..7), K-col base = (t%4)*4 (0,4,8,12).
                        //   b[0]: row t/4, cols (t%4)*4..(t%4)*4+3
                        //   b[1]: row t/4, cols (t%4)*4+16..(t%4)*4+19
                        // shared_k_int8 is [stage][token=N-row][dim=K-col].
                        const int8_t* k_base_p = &shared_k_int8[stage][lane >> 2][p * SUB_HEAD_DIM + (lane & 3) * 4];
                        b_frag[0] = *reinterpret_cast<const uint32_t*>(k_base_p);
                        b_frag[1] = *reinterpret_cast<const uint32_t*>(k_base_p + 16);
                    }

                    int32_t c_p[4] = {0, 0, 0, 0};
                    mma_int8_m16n8k32(c_p, a_frag, b_frag, c_p);

                    if ((lane >> 2) == 0) {
                        int tok0 = (lane & 3) * 2;
                        int tok1 = tok0 + 1;
                        float s_q = scale_Q[p];
                        float s_k0 = shared_k_scale[stage][tok0][p];
                        float s_k1 = shared_k_scale[stage][tok1][p];
                        acc_lo += (float)c_p[0] * s_q * s_k0;
                        acc_hi += (float)c_p[1] * s_q * s_k1;
                    }
                }

                if ((lane >> 2) == 0 && (lane & 3) < 4) {
                    int t0 = (lane & 3) * 2;
                    tile_logits[stage][warp][t0]     = acc_lo;
                    tile_logits[stage][warp][t0 + 1] = acc_hi;
                }
                __syncwarp();

            }
        } else {
            // ── Manual per-lane INT8 dot — the production path for head dims
            // whose palette isn't 32 dims (e.g. HEAD_DIM=64), where the
            // m16n8k32 fragment layout doesn't apply.
            if (warp_active) {
                for (int t = 0; t < WARPS_PER_BLOCK; ++t) {
                    int my_pal = lane / 8;
                    float sQ = scale_Q[my_pal];
                    float sK = shared_k_scale[stage][t][my_pal];
                    int8_t* k_t = shared_k_int8[stage][t];
                    float dr = 0.f;
                    for (int j = 0; j < VEC; ++j) {
                        float qr = (float)q_int8[j] * sQ;
                        float kr = (float)k_t[lane * VEC + j] * sK;
                        dr = __fmaf_rn(qr, kr, dr);
                    }
                    dr = warp_reduce_sum(dr);
                    if (lane == 0) tile_logits[stage][warp][t] = dr;
                }
                __syncwarp();

            }
        }

        if (warp_active) {
            // ── Tile-batched (FlashAttention-style) softmax ────────────────
            // One running-max update per TILE, not per token. Phase 1 computes
            // the tile's per-token scores and the tile max; phase 2 rescales the
            // accumulator (carried from previous tiles) ONCE by alpha; phase 3
            // adds beta·V for every token relative to the tile max with no
            // per-token rescale. Mathematically identical to the per-token
            // online softmax (softmax is associative), but the rescale-free
            // accumulation is what lets the PV become a batched MMA (1C).
            // Scores stay in tile_logits[] (smem); each phase recomputes the
            // scaled score from there rather than holding an 8-wide per-lane
            // register array — frees ~8 registers/thread for occupancy. The
            // smem reads are warp-uniform broadcasts.
            float tile_max = -1e38f;
            #pragma unroll
            for (int t = 0; t < WARPS_PER_BLOCK; ++t) {
                float s = tok_valid(t) ? tile_logits[stage][warp][t] * softmax_scale : -1e38f;
                tile_max = fmaxf(tile_max, s);
            }

            // Phase 2: single accumulator rescale for the whole tile.
            float new_m = fmaxf(m_i, tile_max);
            float alpha = fast_exp::exp2<float, fast_exp::Softmax>(
                              make_float2(m_i - new_m, 0.f)).x;
            l_i *= alpha;
            #pragma unroll
            for (int j = 0; j < VEC; ++j) out_reg[j] *= alpha;

            // Phase 3: accumulate beta·V per token (no per-token rescale).
            // v_rt: V came straight from an int8 arena (palette order + per-dim
            // block scales, §1A) — gather via the vi palette iterator and scale
            // per dim. Otherwise V is in the FP→int8 path's logical layout with
            // one per-token scale.
            const bool v_rt = (shared_v_readthrough[stage] != 0);
            #pragma unroll
            for (int t = 0; t < WARPS_PER_BLOCK; ++t) {
                // Recompute the score from tile_logits[] (smem). s <= -1e37 marks
                // an invalid token; guard so an all-invalid tile (new_m==-1e38)
                // can't yield exp2(0)=1.
                float s = tok_valid(t) ? tile_logits[stage][warp][t] * softmax_scale : -1e38f;
                float beta = (s > -1e37f)
                    ? fast_exp::exp2<float, fast_exp::Softmax>(
                          make_float2(s - new_m, 0.f)).x
                    : 0.f;
                l_i += beta;

                {
                    float beta_abs = fabsf(beta);
                    float beta_scale = beta_abs / 127.f;
                    if (beta_scale == 0.f) beta_scale = 1.f;
                    int beta_q = (int)__float2int_rn(fminf(fmaxf(beta / beta_scale, -127.f), 127.f));
                    int8_t* v_int8_t = shared_v_int8[stage][t];
                    if (v_rt) {
                        // Read-through: gather palette-order V via vi, per-dim scale.
                        #pragma unroll
                        for (int j = 0; j < VEC; ++j) {
                            int src = vi[j];
                            int32_t prod = (int32_t)beta_q * (int32_t)v_int8_t[src];
                            float sc = beta_scale * shared_v_dim_scale[stage][src];
                            out_reg[j] = __fmaf_rn((float)prod, sc, out_reg[j]);
                        }
                    } else {
                        // FP→int8 path: logical layout, single per-token scale.
                        float combined_scale = beta_scale * shared_v_scale[stage][t];
                        #pragma unroll
                        for (int j = 0; j < VEC; ++j) {
                            int32_t prod = (int32_t)beta_q * (int32_t)v_int8_t[lane * VEC + j];
                            out_reg[j] = __fmaf_rn((float)prod, combined_scale, out_reg[j]);
                        }
                    }
                }
            }
            m_i = new_m;
        }
    };

    // Pipelined main loop (mirrors v2's structure), over this split's tile range.
    const int range = tile_hi - tile_lo;
    if constexpr (NUM_STAGES >= 2 && USE_TC) {
        int tiles_loaded = 0;
        if (range > 0) { load_tile(tile_lo + 0, 0); cp_async_commit<USE_TC>(); tiles_loaded = 1; }
        if (range > 1 && NUM_STAGES >= 2) { load_tile(tile_lo + 1, 1); cp_async_commit<USE_TC>(); tiles_loaded = 2; }
        if constexpr (NUM_STAGES >= 3) {
            if (range > 2) { load_tile(tile_lo + 2, 2); cp_async_commit<USE_TC>(); tiles_loaded = 3; }
        }
        if (tiles_loaded >= NUM_STAGES) {
            cp_async_wait<NUM_STAGES - 1, USE_TC>();
        } else if (tiles_loaded == 2) {
            cp_async_wait<1, USE_TC>();
        } else if (tiles_loaded == 1) {
            cp_async_wait<0, USE_TC>();
        }
        __syncthreads();
        if (range > 0) {
            maybe_init_kv_iters_for_tile(tile_lo + 0);
            apply_rope_to_tile(tile_lo + 0, 0);
        }
        int cur_stage = 0;
        for (int tile = tile_lo; tile < tile_hi; ++tile) {
            __syncthreads();
            process_tile(tile, cur_stage);
            __syncthreads();
            int prefetch_tile = tile + NUM_STAGES;
            if (prefetch_tile < tile_hi) {
                load_tile(prefetch_tile, cur_stage);
                cp_async_commit<USE_TC>();
            }
            int next_tile = tile + 1;
            if (next_tile < tile_hi) {
                cp_async_wait<NUM_STAGES - 1, USE_TC>();
                __syncthreads();
                maybe_init_kv_iters_for_tile(next_tile);
                apply_rope_to_tile(next_tile, (cur_stage + 1) % NUM_STAGES);
            }
            cur_stage = (cur_stage + 1) % NUM_STAGES;
        }
    } else {
        for (int tile = tile_lo; tile < tile_hi; ++tile) {
            load_tile(tile, 0);
            __syncthreads();
            maybe_init_kv_iters_for_tile(tile);
            apply_rope_to_tile(tile, 0);
            __syncthreads();
            process_tile(tile, 0);
            __syncthreads();
        }
    }

    emit_result(head_idx, out_reg, m_i, l_i, warp_active);
}

// Register target for the INT8 decode kernel. WARPS=8 (256 thr): 4 blocks/SM,
// which caps ptxas at 65536/(4*256)=64 registers → 67% theoretical occupancy
// (vs 50% at the v2 target of 3). WARPS=16 (512 thr) keeps 2.
template <int WARPS_PER_BLOCK>
constexpr int int8_decode_min_blocks() {
    return (WARPS_PER_BLOCK <= 8) ? 4 : 2;
}

template <typename Q_T, typename T, typename O,
          int HEAD_DIM, int WARPS_PER_BLOCK, bool ROPE_INTERLEAVED>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE,
                                   int8_decode_min_blocks<WARPS_PER_BLOCK>())
int8_decode_kernel(
    const Q_T* q,
    const uint8_t* headers_ptr,
    O* out,
    int num_active_slots,
    int n_q_head,
    int n_kv_head,
    float softmax_scale,
    const T* k_new,
    const T* v_new,
    const float* rope_cs,
    float* partial_acc,
    float* partial_ml,
    QsaSel sel
) {
    constexpr bool IS_HALF_TYPE = std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>;
    // The warp=head (wide) kernel runs only for heads_per_group > 8 (WARPS=16).
    // At HEAD_DIM=256 a single pipeline stage is ~27 KB but two stages is ~55 KB,
    // which overflows the 48 KiB static shared-memory cap — so HEAD_DIM=256 runs
    // single-stage here, without load/compute overlap. Every HEAD_DIM <= 128
    // instantiation keeps its original stage count unchanged.
    //
    // **This path is no longer exotic.** It used to say no target model had
    // hpg>8 at hd256; Qwen3.8-Flash-Next is one (24/2 heads at 256), so this is
    // the decode kernel its full-attention layers actually run. What the
    // overlap is worth was then MEASURED rather than assumed
    // (`test_engine_qsa_at_depth --features profile`, at the QSA-capped ~2051
    // read that is this model's steady state at any depth): 0.52 ms per
    // layer-call, so 12 attention layers are ~6 ms of a ~64 ms decode step —
    // about a tenth. Whatever a second stage recovers is bounded by that
    // tenth, which does not pay for putting this kernel on dynamic shared
    // memory (`cudaFuncSetAttribute`), a change every model sharing it would
    // wear. Re-measure before revisiting: a model with more attention layers,
    // or without QSA's cap on the read, moves the bound.
    constexpr int STAGES = (HEAD_DIM >= 256) ? 1 : (IS_HALF_TYPE ? 3 : 2);
    int8_decode_attn_impl<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, 32, STAGES, true, ROPE_INTERLEAVED>(
        q, headers_ptr, out, num_active_slots, n_q_head, n_kv_head, softmax_scale,
        k_new, v_new, rope_cs, partial_acc, partial_ml, sel);
}

// =============================================================================
// Block-level partial merge. Every warp of a block has walked its own token
// stripe and holds un-normalised flash state (ΣwV, m, l) for a subset of the
// group's heads; this folds the block's warps together per head and writes ONE
// partial per (slot, head, split), so the combine reduces over `num_splits`
// partials per row instead of `num_splits × WARPS_PER_BLOCK`. Head `h` is
// merged from the warps whose `head_lo <= h < head_hi`; a warp that never saw a
// token for it holds (m = -1e38, l = 0, ΣwV = 0) and contributes weight 0 (or,
// when every warp is empty, a null partial the combine drops the same way).
//
// The merge is the same natural-base log-sum-exp the combine kernel applies to
// the per-split partials: gm = max_w m_w, out = Σ_w ΣwV_w·e^(m_w-gm),
// L = Σ_w l_w·e^(m_w-gm). Hierarchical (warps here, splits in the combine) is
// the same sum in a different association.
//
// `s_merge` / `s_ml` are the block's scratch; the two `__syncthreads` per head
// make this a block-uniform call — every warp must reach it, including on the
// early-outs (no slices, no tokens), which is why the callers return through it
// rather than around it.
// =============================================================================
template <int HEAD_DIM, int WARPS_PER_BLOCK, int HPG, int WARP_HEADS>
__device__ __forceinline__ void stripe_block_merge_emit(
    const float (&out_reg)[WARP_HEADS][HEAD_DIM / WARP_SIZE],
    const float (&m_i)[WARP_HEADS],
    const float (&l_i)[WARP_HEADS],
    int head_lo,                 // this warp's first head (group-local)
    float (&s_merge)[WARPS_PER_BLOCK][HEAD_DIM],
    float (&s_ml)[WARPS_PER_BLOCK][2],
    float* __restrict__ partial_acc,
    float* __restrict__ partial_ml,
    int slot_idx, int kv_head_idx, int split_idx, int num_splits, int n_q_head,
    int warp, int lane
) {
    constexpr int VEC = HEAD_DIM / WARP_SIZE;
    constexpr int NT = WARPS_PER_BLOCK * WARP_SIZE;
    const int tid = warp * WARP_SIZE + lane;
    for (int h = 0; h < HPG; ++h) {
        // Stage this warp's state for head h — a warp that does not own h
        // stages a null so the merge below reads a uniform layout.
        const int hh = h - head_lo;
        const bool owns = (hh >= 0) && (hh < WARP_HEADS);
        float m = -1e38f, l = 0.f;
        #pragma unroll
        for (int k = 0; k < WARP_HEADS; ++k) {
            if (owns && k == hh) {
                m = m_i[k]; l = l_i[k];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) s_merge[warp][lane * VEC + j] = out_reg[k][j];
            }
        }
        if (!owns) {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) s_merge[warp][lane * VEC + j] = 0.f;
        }
        if (lane == 0) { s_ml[warp][0] = m; s_ml[warp][1] = l; }
        __syncthreads();
        const int qh = kv_head_idx * HPG + h;
        if (qh < n_q_head) {
            float gm = -1e38f;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) gm = fmaxf(gm, s_ml[w][0]);
            float wgt[WARPS_PER_BLOCK];
            float L = 0.f;
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                wgt[w] = expf(s_ml[w][0] - gm);
                L += s_ml[w][1] * wgt[w];
            }
            const int64_t base = ((int64_t)slot_idx * n_q_head + qh) * num_splits + split_idx;
            float* acc = partial_acc + base * HEAD_DIM;
            for (int d = tid; d < HEAD_DIM; d += NT) {
                float a = 0.f;
                #pragma unroll
                for (int w = 0; w < WARPS_PER_BLOCK; ++w) a += s_merge[w][d] * wgt[w];
                acc[d] = a;
            }
            if (tid == 0) { partial_ml[base * 2] = gm; partial_ml[base * 2 + 1] = L; }
        }
        __syncthreads();
    }
}

// Warp-pair barrier: the two warps of stripe pair `pair` (64 threads) rendezvous
// on named barrier `1 + pair` (0 is `__syncthreads`). `bar.sync` orders the
// participants' shared-memory writes before any participant's later reads, which
// is exactly the K-from-one-warp / V-from-the-other hand-off the stripe needs.
__device__ __forceinline__ void stripe_pair_sync(int pair) {
    asm volatile("bar.sync %0, %1;" :: "r"(1 + pair), "r"(2 * WARP_SIZE) : "memory");
}

// Per-warp constants of the stripe walk — everything the per-cell body reads
// but never writes. `k_stage` / `v_stage` are the pair's two double-buffered
// staging rows (`[2][HEAD_DIM]`), `shared_q` the block's RoPE'd queries.
template <typename T, int HEAD_DIM, int HPG>
struct StripeCellCtx {
    const float (*shared_q)[HEAD_DIM];   // [HPG][HEAD_DIM]
    T (*k_stage)[HEAD_DIM];              // [2][HEAD_DIM]
    T (*v_stage)[HEAD_DIM];              // [2][HEAD_DIM]
    const float* __restrict__ rope_cs;
    float softmax_scale;
    int kv_head_idx;
    int pair;
    int half;        // 0: this warp loads K; 1: this warp loads V
    int head_lo;     // first group-local head this warp computes
    int lane;
};

// Per-warp mutable walk state that is not a flash accumulator: the palette
// gather iterators, re-derived whenever the walk enters a new slice.
template <int VEC, int HEAD_DIM, int WARP_HEADS>
struct StripeWarpState {
    PalIter<VEC, HEAD_DIM> ki, vi;
    int cur_slice;
};

// One KV cell of the stripe walk: the pair stages K (even warp) and V (odd
// warp) for token `within` of slice `sl`, rendezvous, and each warp folds the
// token into the flash state of its `WARP_HEADS` heads. `cell` is 0-based
// within the pair's walk and picks the staging buffer, so cell j's loads land
// while the pair may still be reading cell j-1 from the other buffer; the
// barrier at cell j+1 orders every read of buffer j&1 before cell j+2 refills
// it. Both warps of a pair must call this for the same cell sequence.
template <typename T, int HEAD_DIM, bool ROPE_INTERLEAVED, int HPG, int WARP_HEADS>
__device__ __forceinline__ void stripe_process_cell(
    const StripeCellCtx<T, HEAD_DIM, HPG>& c,
    StripeWarpState<HEAD_DIM / WARP_SIZE, HEAD_DIM, WARP_HEADS>& st,
    float (&out_reg)[WARP_HEADS][HEAD_DIM / WARP_SIZE],
    float (&m_i)[WARP_HEADS],
    float (&l_i)[WARP_HEADS],
    int slice_idx, const uint8_t* sl, int within, int cell
) {
    constexpr int VEC = HEAD_DIM / WARP_SIZE;
    constexpr int N_PALETTE = 4;
    constexpr int SUB_HEAD_DIM = HEAD_DIM / N_PALETTE;
    constexpr int64_t sub_head_stride = (int64_t)SUB_HEAD_DIM * CHUNK_SIZE;
    constexpr int BLOCKS_PER_DIM = CHUNK_SIZE / 32;

    const uint8_t* head_ptr = get_head<HEAD_DIM>(sl, c.kv_head_idx);
    if (slice_idx != st.cur_slice) {
        st.ki.init(kvhead_k_pal_map<HEAD_DIM>(head_ptr), c.lane);
        st.vi.init(kvhead_v_pal_map<HEAD_DIM>(head_ptr), c.lane);
        st.cur_slice = slice_idx;
    }
    const int buf = cell & 1;
    T* k_stage = c.k_stage[buf];
    T* v_stage = c.v_stage[buf];
    if (c.half == 0) {
        #pragma unroll
        for (int p = 0; p < N_PALETTE; ++p) {
            uint64_t k_ptr_p = kvhead_k_ptr<HEAD_DIM>(head_ptr, p);
            if (k_ptr_p) {
                ArenaAccessor ka((const char*)(uintptr_t)k_ptr_p, kvhead_k_fmt<HEAD_DIM>(head_ptr, p),
                                 sub_head_stride, sub_head_stride, BLOCKS_PER_DIM, 0);
                ka.template load_head_scaled<T, SUB_HEAD_DIM, false>(
                    k_stage + p * SUB_HEAD_DIM, 0, 0, within, c.lane, kvhead_k_scale<HEAD_DIM>(head_ptr, p));
            }
        }
    } else {
        #pragma unroll
        for (int p = 0; p < N_PALETTE; ++p) {
            uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(head_ptr, p);
            if (v_ptr_p) {
                ArenaAccessor va((const char*)(uintptr_t)v_ptr_p, kvhead_v_fmt<HEAD_DIM>(head_ptr, p),
                                 sub_head_stride, sub_head_stride, BLOCKS_PER_DIM, 0);
                va.template load_head_scaled<T, SUB_HEAD_DIM, false>(
                    v_stage + p * SUB_HEAD_DIM, 0, 0, within, c.lane, kvhead_v_scale<HEAD_DIM>(head_ptr, p));
            }
        }
    }
    stripe_pair_sync(c.pair);

    // Two phases so K and V are never live together: the logits for every
    // head first (K in registers, V untouched), then V is gathered into the
    // registers K vacated and folded into the accumulators. Holding both rows
    // through the head loop costs VEC registers per lane for nothing.
    float logit[WARP_HEADS];
    {
        float k_regs[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) k_regs[j] = to_f32<T>(k_stage[st.ki[j]]);
        const int32_t rope_pos = (int32_t)slice_rope(sl) + (within - (int)slice_offset(sl));
        if constexpr (ROPE_INTERLEAVED && (VEC == 1 || VEC % 2 == 0))
            apply_rope_interleaved_f32<VEC, HEAD_DIM>(k_regs, c.lane, rope_pos, c.rope_cs);
        else
            apply_rope_rotary_f32<VEC, HEAD_DIM>(k_regs, c.lane, rope_pos, c.rope_cs);
        #pragma unroll
        for (int hh = 0; hh < WARP_HEADS; ++hh) {
            const int h = c.head_lo + hh;
            float dr = 0.f;
            if (h < HPG) {
                #pragma unroll
                for (int j = 0; j < VEC; ++j) dr = __fmaf_rn(c.shared_q[h][c.lane * VEC + j], k_regs[j], dr);
            }
            logit[hh] = warp_reduce_sum(dr) * c.softmax_scale;
        }
    }
    float v_regs[VEC];
    #pragma unroll
    for (int j = 0; j < VEC; ++j) v_regs[j] = to_f32<T>(v_stage[st.vi[j]]);
    #pragma unroll
    for (int hh = 0; hh < WARP_HEADS; ++hh) {
        const int h = c.head_lo + hh;
        if (h < HPG) {
            float new_m = fmaxf(m_i[hh], logit[hh]);
            float alpha = fast_exp::exp2<float, fast_exp::Softmax>(make_float2(m_i[hh] - new_m, 0.f)).x;
            float beta = fast_exp::exp2<float, fast_exp::Softmax>(make_float2(logit[hh] - new_m, 0.f)).x;
            l_i[hh] = l_i[hh] * alpha + beta;
            #pragma unroll
            for (int j = 0; j < VEC; ++j) out_reg[hh][j] = out_reg[hh][j] * alpha + beta * v_regs[j];
            m_i[hh] = new_m;
        }
    }
}

// =============================================================================
// WARP-STRIPE decode — every warp computes. Warps work in PAIRS: the two warps
// of a pair walk the same KV token stripe in lockstep, the even warp loading K
// and the odd warp loading V into a shared double-buffered staging row, and each
// computing the flash update for HALF of the group's heads. That keeps the
// per-warp flash state (heads × HEAD_DIM/32 accumulators) small enough to live
// in registers with the head loop fully unrolled — at HPG=8, HEAD_DIM=256 a
// single warp holding all eight heads needs 64 accumulators plus K/V/m/l and
// exceeds the 64-register budget of 4 blocks/SM, and the compiler puts the
// accumulator array on the stack (measured: a 336-byte frame, so every cell's
// update streamed through local memory). Halving the heads per warp puts it
// back in registers and halves each warp's load/dequant work as a side effect.
//
// Two walks share one per-cell body:
//   * dense — the row attends every visible token: tokens are enumerated per
//     slice (gap-aware) and striped across (split, pair);
//   * sparse — a QSA selection names the cells: the walk enumerates the
//     selection's ENTRIES (a run of ≤4 cells at one block), striped across
//     (split, pair), each lane resolving one entry's slice by binary search so
//     32 searches are in flight per warp, then the pair sweeps the resolved
//     cells by shuffle-broadcast. Trip count is the selection's size —
//     constant in depth — and nothing in the block scales with `n_slices`.
// Each block emits ONE partial per head via `stripe_block_merge_emit`.
// =============================================================================
template <typename Q_T, typename T, typename O,
          int HEAD_DIM, int WARPS_PER_BLOCK, bool ROPE_INTERLEAVED, int HPG>
__device__ __forceinline__ void int8_decode_stripe_impl(
    const Q_T* __restrict__ q,
    const uint8_t* __restrict__ headers_ptr,
    int num_active_slots,
    int n_q_head,
    int n_kv_head,
    float softmax_scale,
    const T* __restrict__ k_new,
    const T* __restrict__ v_new,
    const float* __restrict__ rope_cs,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_ml,
    QsaSel sel
) {
    constexpr int VEC = HEAD_DIM / WARP_SIZE;
    constexpr int N_PALETTE = 4;
    constexpr int SUB_HEAD_DIM = HEAD_DIM / N_PALETTE;
    static_assert(WARPS_PER_BLOCK % 2 == 0, "stripe warps work in pairs");
    static_assert(HPG <= WARPS_PER_BLOCK, "one warp per head loads shared_q");
    constexpr int PAIRS = WARPS_PER_BLOCK / 2;
    // Heads per warp: the even warp of a pair takes [0, WARP_HEADS), the odd
    // warp [WARP_HEADS, HPG). For odd HPG the odd warp's last slot is unused.
    constexpr int WARP_HEADS = (HPG + 1) / 2;

    int slot_idx = (int)blockIdx.x;
    int kv_head_idx = (int)blockIdx.y;
    int split_idx = (int)blockIdx.z;
    int num_splits = (int)gridDim.z;
    int tid = (int)threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (slot_idx >= num_active_slots || kv_head_idx >= n_kv_head) return;

    const int pair = warp >> 1;
    const int half = warp & 1;
    const int head_lo = half * WARP_HEADS;

    // Per-head flash-state for this warp's heads (un-normalized ΣwV, m, l).
    // Every index below is a compile-time constant after unrolling, so the
    // arrays stay in registers.
    float out_reg[WARP_HEADS][VEC];
    float m_i[WARP_HEADS], l_i[WARP_HEADS];
    #pragma unroll
    for (int h = 0; h < WARP_HEADS; ++h) {
        m_i[h] = -1e38f; l_i[h] = 0.f;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) out_reg[h][j] = 0.f;
    }

    __shared__ alignas(128) float s_merge[WARPS_PER_BLOCK][HEAD_DIM];
    __shared__ float s_ml[WARPS_PER_BLOCK][2];
    auto emit_block = [&]() {
        stripe_block_merge_emit<HEAD_DIM, WARPS_PER_BLOCK, HPG, WARP_HEADS>(
            out_reg, m_i, l_i, head_lo, s_merge, s_ml, partial_acc, partial_ml,
            slot_idx, kv_head_idx, split_idx, num_splits, n_q_head, warp, lane);
    };

    const SlotHeader& slot = get_slot_header(headers_ptr, slot_idx);
    // Positions end at the write slice (see `resolve_pos`): the chunks after
    // it are empty capacity whose `rope` nothing keeps current, and a
    // rope-ordered search that reached one would take it for the owner of
    // positions the writer holds.
    const uint32_t n_slices = min(slot.n_slices, slot.write_slice + 1u);
    const uint32_t write_slice_idx = slot.write_slice;
    const uint64_t slices_ptr = slot.slices_ptr;

    if (n_slices == 0) {
        emit_block();  // null partials
        return;
    }

    uint8_t* write_slice_ptr = get_slice_mut<HEAD_DIM>(slices_ptr, (int)write_slice_idx, n_kv_head);
    const uint16_t ws_offset = slice_offset(write_slice_ptr);
    const uint16_t ws_len = slice_len(write_slice_ptr);
    const uint32_t ws_rope = slice_rope(write_slice_ptr);

    // ─── Fused KV scatter (warp 0; idempotent across split/warp blocks) ──
    {
        const int within = (int)ws_offset + (int)ws_len;
        constexpr int LANES_PER_PAL = WARP_SIZE / N_PALETTE;
        if (warp == 0 && within < CHUNK_SIZE) {
            const uint8_t* head_ptr = get_head<HEAD_DIM>(write_slice_ptr, kv_head_idx);
            int pal = lane / LANES_PER_PAL;
            int local_lane = lane % LANES_PER_PAL;
            uint64_t k_ptr_p = kvhead_k_ptr<HEAD_DIM>(head_ptr, pal);
            uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(head_ptr, pal);
            int k_fmt = kvhead_k_fmt<HEAD_DIM>(head_ptr, pal);
            int v_fmt = kvhead_v_fmt<HEAD_DIM>(head_ptr, pal);
            if (k_ptr_p != 0) {
                char* k_arena = (char*)(uintptr_t)k_ptr_p;
                char* v_arena = (char*)(uintptr_t)v_ptr_p;
                int k_esz = ArenaFormat::float_elem_size(k_fmt);
                int v_esz = ArenaFormat::float_elem_size(v_fmt);
                int64_t src_base = ((int64_t)slot_idx * (int64_t)n_kv_head + (int64_t)kv_head_idx) * (int64_t)HEAD_DIM;
                const T* k_src = k_new + src_base;
                const T* v_src = v_new + src_base;
                float k_regs[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) k_regs[j] = to_f32<T>(k_src[lane * VEC + j]);
                if (k_fmt == ArenaFormat::R16) {
                    int hpg_w = n_q_head / n_kv_head; if (hpg_w < 1) hpg_w = 1;
                    int q_head = kv_head_idx * hpg_w;
                    int64_t q_base = ((int64_t)slot_idx * (int64_t)n_q_head + (int64_t)q_head) * (int64_t)HEAD_DIM;
                    float q_regs[VEC];
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j) q_regs[j] = to_f32<Q_T>(q[q_base + lane * VEC + j]);
                    write_regs_to_r16<VEC>(k_arena, 0, within, local_lane, k_regs, q_regs);
                } else if (k_esz > 0) {
                    int64_t eo = (int64_t)within * SUB_HEAD_DIM;
                    write_regs_to_arena<VEC>(k_arena, eo, local_lane, k_esz, k_fmt, k_regs);
                }
                float v_regs[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) v_regs[j] = to_f32<T>(v_src[lane * VEC + j]);
                if (v_esz > 0) {
                    int64_t eo_v = (int64_t)within * SUB_HEAD_DIM;
                    write_regs_to_arena<VEC>(v_arena, eo_v, local_lane, v_esz, v_fmt, v_regs);
                }
            }
        }
        __syncthreads();
    }

    int kv_len = (int)ws_rope + (int)ws_len + 1;
    if (kv_len <= 0) { emit_block(); return; }

    // Q for all heads (logical, RoPE'd) in SHARED smem — the query is
    // warp-independent, so one copy serves every warp and it stays out of
    // per-thread registers/stack. Warp h loads head h, so the prologue costs
    // one head's worth of loads per warp rather than HPG heads' worth on warp 0.
    __shared__ float shared_q[HPG][HEAD_DIM];
    if (warp < HPG) {
        const int h = warp;
        uint32_t q_rope_pos = (uint32_t)ws_rope + (uint32_t)ws_len;
        int qh = kv_head_idx * HPG + h;
        const Q_T* q_ptr = q + ((int64_t)slot_idx * n_q_head + qh) * (int64_t)HEAD_DIM;
        float qr[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) qr[j] = to_f32<Q_T>(q_ptr[lane * VEC + j]);
        if constexpr (ROPE_INTERLEAVED && (VEC == 1 || VEC % 2 == 0))
            apply_rope_interleaved_f32<VEC, HEAD_DIM>(qr, lane, (int)q_rope_pos, rope_cs);
        else
            apply_rope_rotary_f32<VEC, HEAD_DIM>(qr, lane, (int)q_rope_pos, rope_cs);
        #pragma unroll
        for (int j = 0; j < VEC; ++j) shared_q[h][lane * VEC + j] = qr[j];
    }
    __syncthreads();

    // Per-pair token K/V staging (FP, palette order before the ki/vi gather),
    // double-buffered: cell j lands in buffer j&1 while the pair still reads
    // cell j-1 from the other. The even warp fills sk, the odd warp sv; the
    // pair barrier publishes both before either warp gathers.
    __shared__ alignas(128) T sk[PAIRS][2][HEAD_DIM];
    __shared__ alignas(128) T sv[PAIRS][2][HEAD_DIM];

    // Per-slice token enumeration (gap-aware). Each slice contributes its
    // slice_eff_len valid tokens (+1 for the write slice's freshly-scattered
    // token); a global token index t maps to (slice, within = off + local) by a
    // forward scan. Iterating valid tokens per slice — rather than a flat
    // chunk_div(logical) that assumes 32 logical tokens per slice — skips a
    // sealed partial chunk's empty tail (the substrate-seal gap) and reaches the
    // writer slice at its true physical position instead of aliasing it into the
    // gap. For a gapless sequence every slice is full so this is identical to the
    // old flat walk.
    auto slice_eff_len = [&](int s) -> int {
        const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
        int len = (int)slice_len(sl);
        int off = (int)slice_offset(sl);
        if (s == (int)write_slice_idx && len < CHUNK_SIZE && off + len < CHUNK_SIZE) len += 1;
        return len;
    };
    // QSA: slot-uniform, hoisted out of the token walk.
    const bool qsa_on = qsa_active(sel) && !qsa_row_dense(sel, slot_idx);

    // ── Per-cell body, shared by both walks ────────────────────────────────
    // `cell` is 0-based within the pair's walk and picks the staging buffer.
    // Both warps of the pair call this for the same (slice, within) sequence.
    // A named function, not a lambda: a lambda called from two sites was left
    // out of line, which gave its by-reference captures (the flash
    // accumulators) an address and put them in local memory — a 352-byte
    // stack frame measured — so every cell's update streamed through it.
    StripeWarpState<VEC, HEAD_DIM, WARP_HEADS> st;
    st.cur_slice = -1;
    const StripeCellCtx<T, HEAD_DIM, HPG> ctx{
        shared_q, sk[pair], sv[pair], rope_cs, softmax_scale,
        kv_head_idx, pair, half, head_lo, lane };

    // Global stripe id: pair `pair` of split `split_idx`, out of `n_stripes`.
    const int stripe = split_idx * PAIRS + pair;
    const int n_stripes = num_splits * PAIRS;

    if (qsa_on) {
        // ── Sparse walk: enumerate the selection's ENTRIES ──────────────────
        //
        // A QSA row attends `top_k + ratio − 1` positions — 2051 for the
        // released checkpoint — however deep the cache is, so the walk's trip
        // count must be the selection's size and nothing in it may scale with
        // `n_slices`. Entries are striped round-robin over the stripes
        // (`stripe + n_stripes·i`), which balances them to within one entry.
        //
        // Each batch of up to 32 entries is resolved lane-parallel: lane `l`
        // takes one entry, binary-searches the slice holding its first cell
        // (rope ranges ascend and do not overlap, so `log2(n_slices)` probes —
        // 12 at 128K — and 32 such chains are in flight per warp instead of
        // one), then walks the entry's ≤4 consecutive cells with a forward
        // cursor and packs each as `(slice << 6) | within`, `CELL_NONE` for a
        // cell no slice holds. A selected position with no slice to hold it
        // would be a selection built against a different cache than the one
        // being read; skipping is the only safe answer — attending a slot
        // outside the slice would read another token's K/V.
        //
        // The pair then sweeps the batch's cells in entry order, each cell's
        // code shuffle-broadcast from the lane that resolved it, so the two
        // warps of the pair see an identical cell sequence (both resolve the
        // same batch — the search is cheap and duplicating it keeps the pair
        // free of any cross-warp hand-off beyond the K/V staging barrier).
        constexpr int MAX_CELLS = 1 << QSA_CELL_BITS;
        constexpr uint32_t CELL_NONE = 0xFFFFFFFFu;
        const uint32_t* sel_entries = sel.entries + (int64_t)slot_idx * sel.stride;
        const int sel_cnt = (int)sel.cnt[slot_idx];
        int cell_no = 0;
        for (int first = stripe; first < sel_cnt; first += n_stripes * WARP_SIZE) {
            // Lanes holding a live entry: `l < ceil((sel_cnt - first) / n_stripes)`.
            const int n_live = min(WARP_SIZE, (sel_cnt - first + n_stripes - 1) / n_stripes);
            uint32_t code[MAX_CELLS];
            #pragma unroll
            for (int c = 0; c < MAX_CELLS; ++c) code[c] = CELL_NONE;
            const int e = first + lane * n_stripes;
            if (lane < n_live) {
                const uint32_t ent = sel_entries[e];
                // Start and width through the page layout: a projected prefix
                // is pages whose last blocks are short (see `qsa_block_width_from`).
                const uint32_t blk = ent >> QSA_CELL_BITS;
                const int pos0 = qsa_block_start(sel, slot_idx, blk);
                const int cells = max(0, min((int)(ent & ((1u << QSA_CELL_BITS) - 1u)) + 1,
                                             qsa_block_width_from(sel, slot_idx, blk, pos0)));
                int s = 0;
                {
                    int lo_s = 0, hi_s = (int)n_slices - 1;
                    while (lo_s < hi_s) {
                        const int mid = (lo_s + hi_s + 1) >> 1;
                        const uint8_t* s_mid = get_slice<HEAD_DIM>(slices_ptr, mid, n_kv_head);
                        if ((int)slice_rope(s_mid) <= pos0) lo_s = mid; else hi_s = mid - 1;
                    }
                    s = lo_s;
                }
                const uint8_t* sp = get_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
                int s_rope = (int)slice_rope(sp);
                int s_len = slice_eff_len(s);
                int s_off = (int)slice_offset(sp);
                #pragma unroll
                for (int c = 0; c < MAX_CELLS; ++c) {
                    if (c < cells) {
                        const int pos = pos0 + c;
                        // Forward cursor: a run of consecutive positions can
                        // cross into the next slice(s).
                        while (s + 1 < (int)n_slices) {
                            const uint8_t* sn = get_slice<HEAD_DIM>(slices_ptr, s + 1, n_kv_head);
                            const int n_rope = (int)slice_rope(sn);
                            if (n_rope > pos) break;
                            ++s; sp = sn; s_rope = n_rope;
                            s_len = slice_eff_len(s);
                            s_off = (int)slice_offset(sp);
                        }
                        const int local = pos - s_rope;
                        if (local >= 0 && local < s_len) {
                            code[c] = ((uint32_t)s << 6) | (uint32_t)(s_off + local);
                        }
                    }
                }
            }
            // The cell sweep is NOT unrolled: the per-cell body inlines the
            // K/V arena loaders for four palettes across every arena format,
            // and four copies of it (measured: ~31K SASS instructions each)
            // left the register allocator spilling in the hot loop. One copy
            // per walk; the cell code is picked by a register select.
            for (int src = 0; src < n_live; ++src) {
                #pragma unroll 1
                for (int c = 0; c < MAX_CELLS; ++c) {
                    uint32_t mine = code[0];
                    #pragma unroll
                    for (int k = 1; k < MAX_CELLS; ++k) mine = (c == k) ? code[k] : mine;
                    const uint32_t cd = __shfl_sync(0xffffffffu, mine, src);
                    if (cd == CELL_NONE) continue;
                    const int slice_idx = (int)(cd >> 6);
                    const int within = (int)(cd & 63u);
                    const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, slice_idx, n_kv_head);
                    stripe_process_cell<T, HEAD_DIM, ROPE_INTERLEAVED, HPG, WARP_HEADS>(
                        ctx, st, out_reg, m_i, l_i, slice_idx, sl, within, cell_no++);
                }
            }
        }
    } else {
        // ── Dense walk: every visible token, striped over (split, pair) ─────
        //
        // The token total is block-uniform; striding the slices across the
        // block and reducing through shared memory is the same integer sum
        // (addition is associative, every term non-negative) for 1/blockDim of
        // the per-thread work. Only the dense walk needs it — the sparse walk
        // above never touches the total and stays O(selection) at any depth.
        __shared__ int s_total_tok;
        if (tid == 0) s_total_tok = 0;
        __syncthreads();
        {
            int part = 0;
            for (int s = tid; s < (int)n_slices; s += WARPS_PER_BLOCK * WARP_SIZE) {
                part += slice_eff_len(s);
            }
            if (part != 0) atomicAdd(&s_total_tok, part);
        }
        __syncthreads();
        const int total_tok = s_total_tok;

        int n_tiles = (total_tok + PAIRS - 1) / PAIRS;
        int tiles_per_split = (n_tiles + num_splits - 1) / num_splits;
        int tok_lo = (split_idx * tiles_per_split) * PAIRS;
        int tok_hi = (split_idx * tiles_per_split + tiles_per_split) * PAIRS;
        if (tok_hi > total_tok) tok_hi = total_tok;

        // Forward cursor: k is monotonic within a pair's strided iteration, so
        // the (slice, base) cursor only advances. `slice_eff_len` already
        // accounts for the writer's +1, so (k - scan_base) reaches the freshly
        // scattered token's slot.
        int scan_s = 0, scan_base = 0;
        int cell_no = 0;
        for (int k = tok_lo + pair; k < tok_hi; k += PAIRS) {
            while (scan_s + 1 < (int)n_slices) {
                int e = slice_eff_len(scan_s);
                if (scan_base + e <= k) { scan_base += e; ++scan_s; }
                else break;
            }
            const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, scan_s, n_kv_head);
            const int within = (int)slice_offset(sl) + (k - scan_base);
            stripe_process_cell<T, HEAD_DIM, ROPE_INTERLEAVED, HPG, WARP_HEADS>(
                ctx, st, out_reg, m_i, l_i, scan_s, sl, within, cell_no++);
        }
    }

    emit_block();
}

// Register target for the warp-stripe kernel: 3 blocks/SM (85 registers at
// 256 threads). The pair walk holds WARP_HEADS × VEC accumulators plus the
// K and V rows and both palette gathers in registers — at HPG=8, HEAD_DIM=256
// that is ~72 live values before addressing, so the 64-register cap of the
// 4-block target spills 1.3 KB per thread (measured) into the same local
// memory the pair split exists to avoid. The grid under a selection is
// ~1–2 blocks per SM anyway; 3 resident is not the limiter.
template <int WARPS_PER_BLOCK>
constexpr int int8_stripe_min_blocks() {
    return (WARPS_PER_BLOCK <= 8) ? 3 : 2;
}

template <typename Q_T, typename T, typename O,
          int HEAD_DIM, int WARPS_PER_BLOCK, bool ROPE_INTERLEAVED, int HPG>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE,
                                   int8_stripe_min_blocks<WARPS_PER_BLOCK>())
int8_decode_stripe_kernel(
    const Q_T* q,
    const uint8_t* headers_ptr,
    int num_active_slots,
    int n_q_head,
    int n_kv_head,
    float softmax_scale,
    const T* k_new,
    const T* v_new,
    const float* rope_cs,
    float* partial_acc,
    float* partial_ml,
    QsaSel sel
) {
    int8_decode_stripe_impl<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, ROPE_INTERLEAVED, HPG>(
        q, headers_ptr, num_active_slots, n_q_head, n_kv_head, softmax_scale,
        k_new, v_new, rope_cs, partial_acc, partial_ml, sel);
}

// =============================================================================
// BATCHED-M decode (1C final) — INT8 tensor-core MMA + read-through V.
// warp = tile-stripe (all warps compute). Per tile the warp runs an m16n8k32
// INT8 MMA over its 8 tokens (N=8) for all HPG query heads at once (M=HPG),
// 4 MMAs (one per 32-wide palette). C is extracted to scores_smem, then a
// per-head flash softmax + read-through INT8 V PV. Partials fold split*warp
// into the combine, as the stripe does. HEAD_DIM 128 only (the palette must
// span one MMA k-step); HEAD_DIM 256 takes int8_decode_tile_kernel.
// =============================================================================
template <typename Q_T, typename T, typename O,
          int HEAD_DIM, int WARPS_PER_BLOCK, bool ROPE_INTERLEAVED, int HPG>
__device__ __forceinline__ void int8_decode_bmma_impl(
    const Q_T* __restrict__ q,
    const uint8_t* __restrict__ headers_ptr,
    int num_active_slots,
    int n_q_head,
    int n_kv_head,
    float softmax_scale,
    const T* __restrict__ k_new,
    const T* __restrict__ v_new,
    const float* __restrict__ rope_cs,
    float* __restrict__ partial_acc,
    float* __restrict__ partial_ml,
    QsaSel sel
) {
    constexpr int VEC = HEAD_DIM / WARP_SIZE;
    constexpr int N_PALETTE = 4;
    constexpr int SUB_HEAD_DIM = HEAD_DIM / N_PALETTE;  // 32 for hd128
    static_assert(SUB_HEAD_DIM == 32, "batched-M MMA requires SUB_HEAD_DIM==32 (HEAD_DIM==128)");

    int slot_idx = (int)blockIdx.x;
    int kv_head_idx = (int)blockIdx.y;
    int split_idx = (int)blockIdx.z;
    int num_splits = (int)gridDim.z;
    int tid = (int)threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (slot_idx >= num_active_slots || kv_head_idx >= n_kv_head) return;

    constexpr int hpg = HPG;

    float out_reg[HPG][VEC];
    float m_i[HPG], l_i[HPG];
    #pragma unroll
    for (int h = 0; h < HPG; ++h) {
        m_i[h] = -1e38f; l_i[h] = 0.f;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) out_reg[h][j] = 0.f;
    }

    // One partial per block per head: the warps' flash states merge through
    // shared memory at the end (`stripe_block_merge_emit`), so the combine
    // reads `num_splits` partials per row, not `num_splits × warps`. Every
    // early-out returns through `emit_block` — the merge is block-wide.
    __shared__ alignas(128) float s_merge[WARPS_PER_BLOCK][HEAD_DIM];
    __shared__ float s_ml[WARPS_PER_BLOCK][2];
    auto emit_block = [&]() {
        stripe_block_merge_emit<HEAD_DIM, WARPS_PER_BLOCK, HPG, HPG>(
            out_reg, m_i, l_i, /*head_lo=*/0, s_merge, s_ml, partial_acc, partial_ml,
            slot_idx, kv_head_idx, split_idx, num_splits, n_q_head, warp, lane);
    };

    const SlotHeader& slot = get_slot_header(headers_ptr, slot_idx);
    // Positions end at the write slice (see `resolve_pos`): the chunks after
    // it are empty capacity whose `rope` nothing keeps current, and a
    // rope-ordered search that reached one would take it for the owner of
    // positions the writer holds.
    const uint32_t n_slices = min(slot.n_slices, slot.write_slice + 1u);
    const uint32_t write_slice_idx = slot.write_slice;
    const uint64_t slices_ptr = slot.slices_ptr;
    if (n_slices == 0) { emit_block(); return; }

    uint8_t* write_slice_ptr = get_slice_mut<HEAD_DIM>(slices_ptr, (int)write_slice_idx, n_kv_head);
    const uint16_t ws_offset = slice_offset(write_slice_ptr);
    const uint16_t ws_len = slice_len(write_slice_ptr);
    const uint32_t ws_rope = slice_rope(write_slice_ptr);

    // ─── New-token scatter (warp 0; idempotent) ──────────────────────────
    {
        const int within = (int)ws_offset + (int)ws_len;
        constexpr int LANES_PER_PAL = WARP_SIZE / N_PALETTE;
        if (warp == 0 && within < CHUNK_SIZE) {
            const uint8_t* head_ptr = get_head<HEAD_DIM>(write_slice_ptr, kv_head_idx);
            int pal = lane / LANES_PER_PAL;
            int local_lane = lane % LANES_PER_PAL;
            uint64_t k_ptr_p = kvhead_k_ptr<HEAD_DIM>(head_ptr, pal);
            uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(head_ptr, pal);
            int k_fmt = kvhead_k_fmt<HEAD_DIM>(head_ptr, pal);
            int v_fmt = kvhead_v_fmt<HEAD_DIM>(head_ptr, pal);
            if (k_ptr_p != 0) {
                char* k_arena = (char*)(uintptr_t)k_ptr_p;
                char* v_arena = (char*)(uintptr_t)v_ptr_p;
                int k_esz = ArenaFormat::float_elem_size(k_fmt);
                int v_esz = ArenaFormat::float_elem_size(v_fmt);
                int64_t src_base = ((int64_t)slot_idx * (int64_t)n_kv_head + (int64_t)kv_head_idx) * (int64_t)HEAD_DIM;
                const T* k_src = k_new + src_base;
                const T* v_src = v_new + src_base;
                float k_regs[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) k_regs[j] = to_f32<T>(k_src[lane * VEC + j]);
                if (k_fmt == ArenaFormat::R16) {
                    int q_head = kv_head_idx * hpg;
                    int64_t q_base = ((int64_t)slot_idx * (int64_t)n_q_head + (int64_t)q_head) * (int64_t)HEAD_DIM;
                    float q_regs[VEC];
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j) q_regs[j] = to_f32<Q_T>(q[q_base + lane * VEC + j]);
                    write_regs_to_r16<VEC>(k_arena, 0, within, local_lane, k_regs, q_regs);
                } else if (k_esz > 0) {
                    write_regs_to_arena<VEC>(k_arena, (int64_t)within * SUB_HEAD_DIM, local_lane, k_esz, k_fmt, k_regs);
                }
                float v_regs[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) v_regs[j] = to_f32<T>(v_src[lane * VEC + j]);
                if (v_esz > 0) write_regs_to_arena<VEC>(v_arena, (int64_t)within * SUB_HEAD_DIM, local_lane, v_esz, v_fmt, v_regs);
            }
        }
        __syncthreads();
    }

    constexpr int64_t sub_head_stride = (int64_t)SUB_HEAD_DIM * CHUNK_SIZE;
    constexpr int BLOCKS_PER_DIM = CHUNK_SIZE / 32;

    // ── Q staged as INT8 16x32 k-major per palette (rows 0..hpg-1 = heads). ──
    __shared__ alignas(128) int8_t shared_qa[N_PALETTE][16][SUB_HEAD_DIM];
    __shared__ float scaleQ[HPG][N_PALETTE];
    {
        // zero the pad rows (hpg..15) once
        for (int idx = tid; idx < N_PALETTE * 16 * SUB_HEAD_DIM; idx += WARPS_PER_BLOCK * WARP_SIZE) {
            int p = idx / (16 * SUB_HEAD_DIM);
            int rem = idx % (16 * SUB_HEAD_DIM);
            int r = rem / SUB_HEAD_DIM;
            int k = rem % SUB_HEAD_DIM;
            if (r >= hpg) shared_qa[p][r][k] = 0;
        }
        uint32_t q_rope_pos = (uint32_t)ws_rope + (uint32_t)ws_len;
        if (warp == 0) {
            #pragma unroll
            for (int h = 0; h < HPG; ++h) {
                int qh = kv_head_idx * hpg + h;
                const Q_T* q_ptr = q + ((int64_t)slot_idx * n_q_head + qh) * (int64_t)HEAD_DIM;
                float qr[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) qr[j] = to_f32<Q_T>(q_ptr[lane * VEC + j]);
                if constexpr (ROPE_INTERLEAVED && (VEC == 1 || VEC % 2 == 0))
                    apply_rope_interleaved_f32<VEC, HEAD_DIM>(qr, lane, (int)q_rope_pos, rope_cs);
                else
                    apply_rope_rotary_f32<VEC, HEAD_DIM>(qr, lane, (int)q_rope_pos, rope_cs);
                // per-palette quant (palette = lane/8, within-palette pos = (lane%8)*4+j)
                float my_max = 0.f;
                #pragma unroll
                for (int j = 0; j < VEC; ++j) my_max = fmaxf(my_max, fabsf(qr[j]));
                float pal_max = my_max;
                pal_max = fmaxf(pal_max, __shfl_xor_sync(0xffffffff, pal_max, 1));
                pal_max = fmaxf(pal_max, __shfl_xor_sync(0xffffffff, pal_max, 2));
                pal_max = fmaxf(pal_max, __shfl_xor_sync(0xffffffff, pal_max, 4));
                int my_pal = lane / 8;
                float sc = pal_max / 127.f;
                if (sc == 0.f) sc = 1.f;
                if ((lane & 7) == 0) scaleQ[h][my_pal] = sc;
                float inv = 1.f / sc;
                #pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    float v = fminf(fmaxf(qr[j] * inv, -127.f), 127.f);
                    shared_qa[my_pal][h][(lane % 8) * 4 + j] = (int8_t)__float2int_rn(v);
                }
            }
        }
    }
    __syncthreads();

    // ── Per-warp tile scratch ────────────────────────────────────────────
    __shared__ alignas(128) T      skt[2][WARPS_PER_BLOCK][HEAD_DIM];        // K load (palette order), cp.async double-buffered
    __shared__ alignas(128) int8_t shared_kb[WARPS_PER_BLOCK][8][HEAD_DIM];  // K int8 (logical = B-frag src)
    __shared__ alignas(16)  float  scaleK[WARPS_PER_BLOCK][8][N_PALETTE];
    __shared__ alignas(16)  float  scores_smem[WARPS_PER_BLOCK][HPG][8];

    // ── Per-slice tiling (gap-aware). Each slice contributes ceil(eff_len/8)
    // 8-token MMA tiles, eff_len being its filled count (+1 for the write
    // slice's freshly-scattered token). Iterating per slice — rather than by a
    // flat chunk_div(logical) that assumes 32 logical tokens per slice — is what
    // lets a sealed partial chunk's empty tail (the substrate-seal gap) be
    // skipped: its unfilled positions are never addressed, and the writer slice
    // that follows is reached at its true physical position rather than being
    // aliased into the gap. ──
    auto slice_eff_len = [&](int s) -> int {
        const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
        int len = (int)slice_len(sl);
        int off = (int)slice_offset(sl);
        if (s == (int)write_slice_idx && len < CHUNK_SIZE && off + len < CHUNK_SIZE) len += 1;
        return len;
    };
    auto slice_tile_count = [&](int s) -> int { return (slice_eff_len(s) + 7) / 8; };

    int total_tiles = 0;
    for (int s = 0; s < (int)n_slices; ++s) total_tiles += slice_tile_count(s);


    int tiles_per_split = (total_tiles + num_splits - 1) / num_splits;
    int tile_lo = split_idx * tiles_per_split;
    int tile_hi = tile_lo + tiles_per_split;
    if (tile_hi > total_tiles) tile_hi = total_tiles;

    PalIter<VEC, HEAD_DIM> ki, vi;
    int cur_slice = -1;
    // Map a global tile g -> (slice, tile-in-slice) with a forward scan. g is
    // monotonic within a warp's strided iteration, so the cursor only advances.
    int scan_s = 0, scan_base = 0;
    // QSA: slot-uniform, hoisted out of the tile walk.
    const bool qsa_on = qsa_active(sel) && !qsa_row_dense(sel, slot_idx);

    // warp-stripe: each warp takes every WARPS_PER_BLOCK-th tile of the split's
    // range (its own tile + smem buffers), so the 8 warps share the work rather
    // than redundantly recomputing the whole range.
    for (int tile = tile_lo + warp; tile < tile_hi; tile += WARPS_PER_BLOCK) {
        while (scan_s + 1 < (int)n_slices) {
            int t_here = slice_tile_count(scan_s);
            if (scan_base + t_here <= tile) { scan_base += t_here; ++scan_s; }
            else break;
        }
        int slice_idx = scan_s;
        int tile_in_slice = tile - scan_base;
        const bool slice_ok = true;  // we only iterate real slices now
        // All 8 tokens of a per-slice tile live in this one 32-token chunk, so
        // the slice / head_ptr / off / bv / ki / vi are shared — hoist them.
        const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, slice_idx, n_kv_head);
        const uint8_t* head_ptr = get_head<HEAD_DIM>(sl, kv_head_idx);
        uint32_t off = (uint32_t)slice_offset(sl);
        uint32_t bv = (uint32_t)slice_len(sl);
        if (slice_idx == (int)write_slice_idx && bv < CHUNK_SIZE && off + bv < CHUNK_SIZE) bv += 1;
        if (slice_idx != cur_slice) {
            ki.init(kvhead_k_pal_map<HEAD_DIM>(head_ptr), lane);
            vi.init(kvhead_v_pal_map<HEAD_DIM>(head_ptr), lane);
            cur_slice = slice_idx;
        }
        int32_t rope_base = (int32_t)slice_rope(sl);
        int within_base = (int)off + tile_in_slice * 8;

        int tok_within[8];
        bool tok_valid[8];
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int within = within_base + t;
            bool valid = (within < (int)(off + bv));
            // QSA: a token this slot does not select is dead exactly as an
            // out-of-range one — its score becomes −inf below.
            if (valid && qsa_on) {
                valid = qsa_selects(sel, slot_idx, rope_base + (within - (int)off));
            }
            // Pad lanes of the slice's last tile read a safe in-bounds slot and
            // are discarded (tok_valid=false) below.
            tok_within[t] = valid ? within : (int)off;
            tok_valid[t] = valid;
        }
        // ── stage the 8 tokens' K → shared_kb, cp.async double-buffered so each
        // token's load overlaps the previous token's gather/RoPE/quant. The
        // prefetch is unconditional when slice_ok (all 8 tokens share the chunk,
        // so every `within` is in-bounds); invalid tokens just zero shared_kb.
        if (slice_ok) {
            #pragma unroll
            for (int p = 0; p < N_PALETTE; ++p) {
                uint64_t k_ptr_p = kvhead_k_ptr<HEAD_DIM>(head_ptr, p);
                if (k_ptr_p) {
                    ArenaAccessor ka((const char*)(uintptr_t)k_ptr_p, kvhead_k_fmt<HEAD_DIM>(head_ptr, p), sub_head_stride, sub_head_stride, BLOCKS_PER_DIM, 0);
                    ka.template load_head_scaled<T, SUB_HEAD_DIM, true>(skt[0][warp] + p * SUB_HEAD_DIM, 0, 0, tok_within[0], lane, kvhead_k_scale<HEAD_DIM>(head_ptr, p));
                }
            }
            cp_async_commit<true>();
        }
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (slice_ok) {
                if (t + 1 < 8) {
                    #pragma unroll
                    for (int p = 0; p < N_PALETTE; ++p) {
                        uint64_t k_ptr_p = kvhead_k_ptr<HEAD_DIM>(head_ptr, p);
                        if (k_ptr_p) {
                            ArenaAccessor ka((const char*)(uintptr_t)k_ptr_p, kvhead_k_fmt<HEAD_DIM>(head_ptr, p), sub_head_stride, sub_head_stride, BLOCKS_PER_DIM, 0);
                            ka.template load_head_scaled<T, SUB_HEAD_DIM, true>(skt[(t + 1) & 1][warp] + p * SUB_HEAD_DIM, 0, 0, tok_within[t + 1], lane, kvhead_k_scale<HEAD_DIM>(head_ptr, p));
                        }
                    }
                    cp_async_commit<true>();
                    cp_async_wait<1, true>();
                } else {
                    cp_async_wait<0, true>();
                }
            }
            __syncwarp();
            if (!tok_valid[t]) {
                #pragma unroll
                for (int j = 0; j < VEC; ++j) shared_kb[warp][t][lane * VEC + j] = 0;
                if (lane < N_PALETTE) scaleK[warp][t][lane] = 1.f;
                continue;
            }
            float k_regs[VEC];
            #pragma unroll
            for (int j = 0; j < VEC; ++j) k_regs[j] = to_f32<T>(skt[t & 1][warp][ki[j]]);
            int32_t rope_pos = rope_base + (tok_within[t] - (int)off);
            if constexpr (ROPE_INTERLEAVED && (VEC == 1 || VEC % 2 == 0))
                apply_rope_interleaved_f32<VEC, HEAD_DIM>(k_regs, lane, rope_pos, rope_cs);
            else
                apply_rope_rotary_f32<VEC, HEAD_DIM>(k_regs, lane, rope_pos, rope_cs);
            float my_max = 0.f;
            #pragma unroll
            for (int j = 0; j < VEC; ++j) my_max = fmaxf(my_max, fabsf(k_regs[j]));
            float pal_max = my_max;
            pal_max = fmaxf(pal_max, __shfl_xor_sync(0xffffffff, pal_max, 1));
            pal_max = fmaxf(pal_max, __shfl_xor_sync(0xffffffff, pal_max, 2));
            pal_max = fmaxf(pal_max, __shfl_xor_sync(0xffffffff, pal_max, 4));
            int my_pal = lane / 8;
            float sc = pal_max / 127.f; if (sc == 0.f) sc = 1.f;
            if ((lane & 7) == 0) scaleK[warp][t][my_pal] = sc;
            float inv = 1.f / sc;
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                float v = fminf(fmaxf(k_regs[j] * inv, -127.f), 127.f);
                shared_kb[warp][t][lane * VEC + j] = (int8_t)__float2int_rn(v);
            }
        }

        // ── QK^T: M=HPG x N=8 INT8 MMA per palette, scaled-accumulate ──
        int my_m = lane >> 2;            // head this lane's C holds
        int tok0 = (lane & 3) * 2;
        int tok1 = tok0 + 1;
        float acc_lo = 0.f, acc_hi = 0.f;
        #pragma unroll
        for (int p = 0; p < N_PALETTE; ++p) {
            uint32_t a_frag[4];
            load_a_frag_m16k32(a_frag, &shared_qa[p][0][0], SUB_HEAD_DIM, lane);
            uint32_t b_frag[2];
            load_b_frag_n8k32(b_frag, &shared_kb[warp][0][p * SUB_HEAD_DIM], HEAD_DIM, lane);
            int32_t c[4] = {0, 0, 0, 0};
            mma_int8_m16n8k32(c, a_frag, b_frag, c);
            if (my_m < hpg) {
                float sq = scaleQ[my_m][p];
                acc_lo += (float)c[0] * sq * scaleK[warp][tok0][p];
                acc_hi += (float)c[1] * sq * scaleK[warp][tok1][p];
            }
        }
        if (my_m < hpg) {
            scores_smem[warp][my_m][tok0] = tok_valid[tok0] ? acc_lo : -1e38f;
            scores_smem[warp][my_m][tok1] = tok_valid[tok1] ? acc_hi : -1e38f;
        }
        __syncwarp();

        // ── softmax pass 1: per head, running-max + accumulator rescale ──
        float new_m[HPG];
        #pragma unroll
        for (int h = 0; h < HPG; ++h) {
            float tile_max = -1e38f;
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                float s = scores_smem[warp][h][t];
                tile_max = fmaxf(tile_max, (s > -1e37f) ? s * softmax_scale : -1e38f);
            }
            float nm = fmaxf(m_i[h], tile_max);
            float alpha = fast_exp::exp2<float, fast_exp::Softmax>(make_float2(m_i[h] - nm, 0.f)).x;
            l_i[h] *= alpha;
            #pragma unroll
            for (int j = 0; j < VEC; ++j) out_reg[h][j] *= alpha;
            new_m[h] = nm;
        }

        // ── PV pass 2: load each token's V once (cp.async double-buffered into
        // the reused skt ring) so its load overlaps the previous token's
        // accumulate, and add it across all heads — no per-tile V smem staging.
        // Prefetch is unconditional when slice_ok (in-bounds); invalid tokens are
        // skipped in the accumulate. ──
        if (slice_ok) {
            #pragma unroll
            for (int p = 0; p < N_PALETTE; ++p) {
                uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(head_ptr, p);
                if (v_ptr_p) {
                    ArenaAccessor va((const char*)(uintptr_t)v_ptr_p, kvhead_v_fmt<HEAD_DIM>(head_ptr, p), sub_head_stride, sub_head_stride, BLOCKS_PER_DIM, 0);
                    va.template load_head_scaled<T, SUB_HEAD_DIM, true>(skt[0][warp] + p * SUB_HEAD_DIM, 0, 0, tok_within[0], lane, kvhead_v_scale<HEAD_DIM>(head_ptr, p));
                }
            }
            cp_async_commit<true>();
        }
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (slice_ok) {
                if (t + 1 < 8) {
                    #pragma unroll
                    for (int p = 0; p < N_PALETTE; ++p) {
                        uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(head_ptr, p);
                        if (v_ptr_p) {
                            ArenaAccessor va((const char*)(uintptr_t)v_ptr_p, kvhead_v_fmt<HEAD_DIM>(head_ptr, p), sub_head_stride, sub_head_stride, BLOCKS_PER_DIM, 0);
                            va.template load_head_scaled<T, SUB_HEAD_DIM, true>(skt[(t + 1) & 1][warp] + p * SUB_HEAD_DIM, 0, 0, tok_within[t + 1], lane, kvhead_v_scale<HEAD_DIM>(head_ptr, p));
                        }
                    }
                    cp_async_commit<true>();
                    cp_async_wait<1, true>();
                } else {
                    cp_async_wait<0, true>();
                }
            }
            __syncwarp();
            if (!tok_valid[t]) continue;
            float v_regs[VEC];
            #pragma unroll
            for (int j = 0; j < VEC; ++j) v_regs[j] = to_f32<T>(skt[t & 1][warp][vi[j]]);
            #pragma unroll
            for (int h = 0; h < HPG; ++h) {
                float s = scores_smem[warp][h][t];
                if (!(s > -1e37f)) continue;
                float beta = fast_exp::exp2<float, fast_exp::Softmax>(make_float2(s * softmax_scale - new_m[h], 0.f)).x;
                l_i[h] += beta;
                #pragma unroll
                for (int j = 0; j < VEC; ++j) out_reg[h][j] = __fmaf_rn(beta, v_regs[j], out_reg[h][j]);
            }
            __syncwarp();
        }
        #pragma unroll
        for (int h = 0; h < HPG; ++h) m_i[h] = new_m[h];
    }

    emit_block();
}

template <typename Q_T, typename T, typename O,
          int HEAD_DIM, int WARPS_PER_BLOCK, bool ROPE_INTERLEAVED, int HPG>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE,
                                   int8_decode_min_blocks<WARPS_PER_BLOCK>())
int8_decode_bmma_kernel(
    const Q_T* q, const uint8_t* headers_ptr, int num_active_slots,
    int n_q_head, int n_kv_head, float softmax_scale,
    const T* k_new, const T* v_new, const float* rope_cs,
    float* partial_acc, float* partial_ml, QsaSel sel
) {
    int8_decode_bmma_impl<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, ROPE_INTERLEAVED, HPG>(
        q, headers_ptr, num_active_slots, n_q_head, n_kv_head, softmax_scale,
        k_new, v_new, rope_cs, partial_acc, partial_ml, sel);
}

// -----------------------------------------------------------------------------
// Split-KV combine: merge the num_splits per-split partial flash-states for each
// (slot, query-head) into the final normalized output. One block per output row
// (slot*n_q_head + qh); HEAD_DIM threads, each owning one output dim. The merge
// is the standard log-sum-exp in NATURAL base — the flash kernels accumulate
// with `fast_exp` e^x (the `exp2` there is the float2-vectorized form, not
// base-2), so the per-split maxima in `partial_ml` are natural-log magnitudes:
//   gm = max_s m_s;  out = (Σ_s ΣwV_s · e^(m_s-gm)) / (Σ_s l_s · e^(m_s-gm)).
// Null partials (m=-inf, l=0) contribute zero.
//
// The split factor is sized to fill the device, not to the depth, so at short
// context nearly every partial is null (one live split of 110 at a 31-token
// row). The block therefore reads every split's (m, l) once, HEAD_DIM splits
// per round across its threads, and compacts the LIVE splits into shared
// arrays in split order; the merge then walks only those. The merge's
// arithmetic — ascending split order, `acc += pa·w; L += l·w` — is the same
// whichever splits are live, so the bytes match the CPU byte oracle exactly.
//
// The launch also commits the step's write-slice length when
// `commit_headers` is non-null (thread 0 of each slot's head-0 block): the
// commit has to follow every split's read of the write slice, and this
// kernel is already the launch that does, so a separate commit launch would
// be one more fixed cost per step.
// -----------------------------------------------------------------------------
template <typename O, int HEAD_DIM>
__global__ void int8_decode_combine_kernel(
    O* __restrict__ out,
    const float* __restrict__ partial_acc,   // [row][split][HEAD_DIM]
    const float* __restrict__ partial_ml,    // [row][split][2]
    int num_rows,
    int num_splits,
    uint8_t* __restrict__ q8_out,            // non-null → emit q8a128 (B2; HEAD_DIM % 128 == 0)
    const O* __restrict__ gate,              // non-null → val ⊙ sigmoid(gate) before the emit.
                                             // A slot's n_q_head·HEAD_DIM gate values are
                                             // contiguous; consecutive slots are
                                             // gate_slot_stride elements apart, so the gate can
                                             // be a strided view of the fused [q|gate]
                                             // projection with no copy.
    int64_t gate_slot_stride,
    int row_heads,                           // n_q_head — decomposes `row` into (slot, head)
                                             // for the gate and the commit
    const uint8_t* __restrict__ commit_headers,  // non-null → commit each slot's write-slice
                                                 // length (slot headers of the decode launch)
    int n_kv_head
) {
    static_assert(HEAD_DIM % WARP_SIZE == 0, "combine block is HEAD_DIM threads, whole warps");
    constexpr int WARPS = HEAD_DIM / WARP_SIZE;
    constexpr int ROUNDS = (MAX_SPLITS + HEAD_DIM - 1) / HEAD_DIM;

    int row = (int)blockIdx.x;
    if (row >= num_rows) return;
    const int d = (int)threadIdx.x;
    const int warp = d >> 5;
    const int lane = d & 31;

    if (commit_headers != nullptr && d == 0 && (row % row_heads) == 0) {
        commit_decode_write_len<HEAD_DIM>(commit_headers, row / row_heads, n_kv_head);
    }

    const float* ml = partial_ml + (int64_t)row * num_splits * 2;
    const float* pa = partial_acc + (int64_t)row * num_splits * HEAD_DIM;

    __shared__ float s_red[WARPS];
    __shared__ int   s_cnt[WARPS];
    __shared__ float s_w[MAX_SPLITS];
    __shared__ float s_l[MAX_SPLITS];
    __shared__ int   s_idx[MAX_SPLITS];

    // Thread d owns splits d, d + HEAD_DIM, ... — one (m, l) pair each.
    float m_r[ROUNDS];
    float l_r[ROUNDS];
    float gm = -1e38f;
    #pragma unroll
    for (int k = 0; k < ROUNDS; ++k) {
        const int s = d + k * HEAD_DIM;
        m_r[k] = -1e38f;
        l_r[k] = 0.f;
        if (s < num_splits) {
            const float2 v = *reinterpret_cast<const float2*>(ml + (int64_t)s * 2);
            m_r[k] = v.x;
            l_r[k] = v.y;
            gm = fmaxf(gm, v.x);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        gm = fmaxf(gm, __shfl_xor_sync(0xffffffff, gm, off, 32));
    }
    if (lane == 0) s_red[warp] = gm;
    __syncthreads();
    #pragma unroll
    for (int w = 0; w < WARPS; ++w) gm = fmaxf(gm, s_red[w]);

    // Compact the live splits in ascending split order: a warp ballot ranks a
    // thread within its warp, the warp counts rank the warps, and `base`
    // carries the count across rounds. A null partial (no tokens) writes only
    // its (m, l); its accumulator slot in the uninitialised pool is never
    // touched, so it is dropped on l rather than read. A split that saw
    // tokens always has l > 0.
    int base = 0;
    #pragma unroll
    for (int k = 0; k < ROUNDS; ++k) {
        if (k * HEAD_DIM >= num_splits) break;   // block-uniform
        const int s = d + k * HEAD_DIM;
        const bool live = (s < num_splits) && (l_r[k] != 0.f);
        const uint32_t bal = __ballot_sync(0xffffffff, live);
        if (lane == 0) s_cnt[warp] = __popc(bal);
        __syncthreads();
        int pos = base;
        #pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            const int c = s_cnt[w];
            if (w < warp) pos += c;
            base += c;
        }
        pos += __popc(bal & ((1u << lane) - 1u));
        if (live) {
            // Natural base to match the flash kernels' e^x accumulation — a 2^Δ
            // weight here would under-shrink low-max partials (2^Δ > e^Δ for
            // Δ<0) and skew the merged softmax wherever per-split maxima differ.
            s_w[pos] = expf(m_r[k] - gm);
            s_l[pos] = l_r[k];
            s_idx[pos] = s;
        }
        __syncthreads();
    }
    const int n_live = base;

    float acc = 0.f, L = 0.f;
    #pragma unroll 4
    for (int i = 0; i < n_live; ++i) {
        const float w = s_w[i];
        acc += pa[(int64_t)s_idx[i] * HEAD_DIM + d] * w;
        L   += s_l[i] * w;
    }
    float inv = __fdividef(1.f, fmaxf(L, 1e-10f));
    float val = acc * inv;

    // The emit — the plain O store, or B2's fused q8a128 context with the
    // optional output gate — is `int8_decode_emit_row`, shared with the tile
    // kernel's in-kernel merge so both paths produce the same bytes.
    __shared__ float sh_amax[HEAD_DIM / 32];
    __shared__ float sh_sum[HEAD_DIM / 32];
    // Raw Σx, pinned. These bytes are an ATTENTION CONTEXT — a convex
    // combination of V rows, so its per-128 sums are bounded by the same
    // magnitudes the residual stream already carries and never approach f16's
    // 65504. The Rust side wraps them with `Q8a128Operand`'s default
    // `SumScale::Raw`, and the two must agree: a producer here that normalised
    // while the operand reported raw would hand the matmul a header it
    // reinterprets, which is a wrong number rather than an error. A model that
    // ever needs otherwise threads the flag from the wrap site, as
    // `quantize_acts_q8a128` and the fused norms do.
    int8_decode_emit_row<O, HEAD_DIM>(val, (int64_t)row, d, out, q8_out, gate,
                                      gate_slot_stride, row_heads, sh_amax, sh_sum,
                                      /*sum_norm=*/0);
}

// SM count (cached) — used to size the split-KV factor to fill the device.
inline int fused_attn_sm_count() {
    static int sm = 0;
    if (sm == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sm, cudaDevAttrMultiProcessorCount, dev);
        if (sm <= 0) sm = 1;
    }
    return sm;
}

// Grow-on-demand device scratch for split-KV partials. Persistent (never freed),
// reused across launches; allocation happens on the first split launch / on a
// grow, never in the steady-state timed path. Single-stream decode only (the
// pool is process-global, not per-stream).
inline void fused_attn_partial_pool(
    int64_t rows, int splits, int head_dim, float** acc_out, float** ml_out,
    cudaStream_t stream
) {
    static float* g_acc = nullptr;
    static float* g_ml  = nullptr;
    static int64_t g_cap_acc = 0;  // capacity in floats
    static int64_t g_cap_ml  = 0;
    int64_t need_acc = rows * splits * head_dim;
    int64_t need_ml  = rows * splits * 2;
    if (need_acc > g_cap_acc) {
        if (g_acc) {
            // Drain the stream before freeing: cudaFree is not stream-ordered,
            // and an earlier split launch on this stream may still be writing
            // the old pool. Growth is rare (a new high-water row count), so
            // the sync cost is amortized away.
            cudaStreamSynchronize(stream);
            cudaFree(g_acc);
        }
        if (cudaMalloc(&g_acc, (size_t)need_acc * sizeof(float)) != cudaSuccess) {
            g_acc = nullptr; g_cap_acc = 0; *acc_out = nullptr; *ml_out = nullptr; return;
        }
        g_cap_acc = need_acc;
    }
    if (need_ml > g_cap_ml) {
        if (g_ml) {
            cudaStreamSynchronize(stream);
            cudaFree(g_ml);
        }
        if (cudaMalloc(&g_ml, (size_t)need_ml * sizeof(float)) != cudaSuccess) {
            g_ml = nullptr; g_cap_ml = 0; *acc_out = nullptr; *ml_out = nullptr; return;
        }
        g_cap_ml = need_ml;
    }
    *acc_out = g_acc;
    *ml_out  = g_ml;
}

// Returns 0 on success, 1 when the split-KV partial pool could not be
// allocated for a launch that requires it (split accumulation, warp-stripe,
// or a q8 emit — the combine kernel is the only q8 emitter and the stripe
// kernels have no direct-write form). The caller must treat 1 as a failed
// wave: nothing was launched, `out`/`q8_out` hold no result.
template <typename Q_T, typename T, typename O, int HEAD_DIM>
int launch_int8_decode_attn(
    const Q_T* q,
    const uint8_t* headers_ptr,
    O* out,
    int num_active_slots,
    int n_q_head,
    int n_kv_head,
    float softmax_scale,
    const T* k_new,
    const T* v_new,
    const float* rope_cs,
    int rope_interleaved,
    cudaStream_t stream = nullptr,
    uint8_t* q8_out = nullptr,  // non-null → B2 fused q8a128 context (combine path, HEAD_DIM % 128 == 0)
    const O* gate = nullptr,    // non-null → combine applies sigmoid(gate) ⊙ val (requires q8_out)
    int64_t gate_slot_stride = 0, // elements between consecutive slots' gate rows
                                  // (n_q_head·HEAD_DIM when the gate is contiguous;
                                  // the fused [q|gate] projection's row width when
                                  // the gate is a strided view of it)
    QsaSel sel = {nullptr, nullptr, nullptr, nullptr, 0, 1} // QSA: one selection row per slot
) {
    int heads_per_group = (n_kv_head > 0) ? (n_q_head / n_kv_head) : 1;
    if (heads_per_group < 1) heads_per_group = 1;
    const bool use_wide = (HEAD_DIM >= 128) && (heads_per_group > 8);

    // ── Split-KV factor: fan each (slot, kv_head)'s KV-tile loop across multiple
    // blocks so the grid fills the SMs when batch*heads is a small grid. Target
    // ~2 waves at the register-bound ~3 blocks/SM; clamp to MAX_SPLITS. Empty
    // splits (short context) early-out cheaply; S=1 keeps the direct-write path.
    int base_blocks = num_active_slots * n_kv_head;
    int num_splits = 1;
    if (base_blocks > 0) {
        int target_blocks = fused_attn_sm_count() * 3 * 2;
        num_splits = (target_blocks + base_blocks - 1) / base_blocks;
    }
    // **Raised from 32.** The split factor is what fills the SMs when the grid
    // is otherwise `slots × kv_heads` — 2 for a single-sequence decode, which is
    // two blocks on a 110-SM card. The heuristic above asks for ~330 and the old
    // clamp handed back 32, i.e. a 64-block grid: measured by ncu at 5.3% DRAM
    // throughput and 17.8% memory throughput, so the kernel was neither
    // bandwidth- nor compute-bound but simply starved of blocks.
    //
    // Splits past the tile count do no work (their range is empty) and cost only
    // a null partial each, so this is a ceiling rather than a target — the
    // `tiles_per_split` arithmetic below still hands out at most one tile per
    // split once the work runs out.
    if (num_splits < 1) num_splits = 1;
    if (num_splits > MAX_SPLITS) num_splits = MAX_SPLITS;

    // ── Wide heads (HEAD_DIM 256) with a group of ≤ 16 query heads take the
    // INT8 tile kernel: one 32-token slice per tile on the tensor cores, the
    // group's heads as the MMA's M rows, 256 threads per block. It always
    // emits partials (one per split per head) and combines. Under a QSA
    // selection its work is the entry count, so the split factor is derived
    // from that — INT8_TILE_ENTRIES_PER_SPLIT entries per block — rather
    // than from the dense token-walk heuristic.
    if constexpr (HEAD_DIM == 256) {
        if (heads_per_group <= TILE_M_ROWS) {
            // Dense: at most one wave of the tile kernel's real residency
            // (TILE_MIN_BLOCKS per SM, the 128-register cap), the depth split
            // evenly across it. The generic heuristic above sizes for a
            // 3-blocks/SM kernel and hands this one 2.33 waves — the trailing
            // third of a wave ran at a third of the occupancy while the DRAM
            // it was walking sat 15% idle (ncu: SM Active 791K of 929K cycles
            // at 128K depth). The split count rounds DOWN: rounding up put
            // 224 blocks on a 220-block card at 8 slots × 2 heads, and the
            // four that did not fit ran as a second wave by themselves —
            // ncu "Waves Per SM 1.02", the kernel taking two block-times for
            // one wave of work.
            const int wave_blocks = fused_attn_sm_count() * TILE_MIN_BLOCKS;
            int splits = (base_blocks > 0) ? wave_blocks / base_blocks : 1;
            if (splits < 1) splits = 1;
            if (splits > MAX_SPLITS) splits = MAX_SPLITS;
            // **Do not oversubscribe the split past one wave to "bound each
            // block's depth" — it was tried and it is worse.** The reasoning is
            // seductive: at `base_blocks >= wave_blocks` the rounding above
            // hands back 1, every block then walks its slot's whole KV, and the
            // concurrent footprint grows with slot count. Forcing the grid to
            // ~4 waves instead (`splits <= 1` → oversubscribe) was measured on a
            // 110-SM card at ~660-token depth:
            //
            // |  slots | splits 1 (avg ms) | oversubscribed (avg ms) |
            // |--------|-------------------|-------------------------|
            // |     64 | 0.868             | 15.112  (splits 7) ✗    |
            // |    128 | 34.420            | 26.819  (splits 4) ~    |
            //
            // 17× worse at 64 slots for a 1.3× gain at 128, and aggregate decode
            // throughput did not move at either width. Whatever makes the wide
            // grid slow, per-block depth is not it.
            if (sel.entries != nullptr) {
                splits = ((int)sel.stride + INT8_TILE_ENTRIES_PER_SPLIT - 1)
                         / INT8_TILE_ENTRIES_PER_SPLIT;
                if (splits < 1) splits = 1;
                if (splits > MAX_SPLITS) splits = MAX_SPLITS;
            }
            float* pa = nullptr;
            float* pm = nullptr;
            fused_attn_partial_pool((int64_t)num_active_slots * n_q_head, splits,
                                    HEAD_DIM, &pa, &pm, stream);
            if (pa == nullptr || pm == nullptr) {
                return 1;
            }
            // An in-kernel split merge (a pairwise tree over the partial
            // pool, the last-arriving block of each pair merging) was
            // measured against this two-launch form: its serial tail — up
            // to seven levels of device fence, global atomic and dependent
            // loads on the critical path — cost a constant 36 µs at every
            // depth, seven times the launch it saved. The combine kernel
            // stays.
            dim3 grid(num_active_slots, n_kv_head, splits);
            dim3 block(TILE_THREADS);
            if (rope_interleaved) {
                int8_decode_tile_kernel<Q_T, T, HEAD_DIM, true><<<grid, block, 0, stream>>>(
                    q, headers_ptr, num_active_slots, n_q_head, n_kv_head,
                    softmax_scale, k_new, v_new, rope_cs, pa, pm, sel);
            } else {
                int8_decode_tile_kernel<Q_T, T, HEAD_DIM, false><<<grid, block, 0, stream>>>(
                    q, headers_ptr, num_active_slots, n_q_head, n_kv_head,
                    softmax_scale, k_new, v_new, rope_cs, pa, pm, sel);
            }
            const int num_rows = num_active_slots * n_q_head;
            const int64_t g_stride =
                (gate_slot_stride != 0) ? gate_slot_stride : (int64_t)n_q_head * HEAD_DIM;
            // The combine also commits the write-slice lengths.
            int8_decode_combine_kernel<O, HEAD_DIM><<<num_rows, HEAD_DIM, 0, stream>>>(
                out, pa, pm, num_rows, splits, q8_out, gate, g_stride, n_q_head,
                headers_ptr, n_kv_head);
            return 0;
        }
    }

    auto launch = [&](auto warps_const, auto rope_const) -> int {
        constexpr int WARPS_PER_BLOCK = decltype(warps_const)::value;
        constexpr bool ROPE_INTERLEAVED = decltype(rope_const)::value;

        // Warp-stripe (1C) when heads_per_group <= WARPS: every warp computes
        // (in pairs, each pair over its own KV stripe), the block merges its
        // warps' flash states and writes ONE partial per head, so it always
        // writes partials + combines. Every kernel emits `num_splits` partials
        // per row. hpg==8 (e.g. Qwen3-MoE, n_q/n_kv=32/4) is included: the
        // batched-M MMA path is gap-aware and the warp=head path is not, so
        // route it here.
        const bool use_stripe = (heads_per_group >= 1 && heads_per_group <= 8
                                 && heads_per_group <= WARPS_PER_BLOCK);
        // Under a QSA selection the stripe's work is the selection's entry
        // count, not the depth, so the split factor is derived from it: one
        // entry (≤4 cells) per pair-stripe. The dense heuristic above sizes
        // for a token walk and would fan a 513-entry row over 256 splits ×
        // 4 pairs — a thousand stripes for five hundred entries, each block
        // paying its Q-staging prologue and block merge for nothing.
        int splits = num_splits;
        if (use_stripe && sel.entries != nullptr) {
            constexpr int PAIRS = WARPS_PER_BLOCK / 2;
            const int stripes = (int)sel.stride;   // entries per row ≥ cnt[slot]
            splits = (stripes + PAIRS - 1) / PAIRS;
            if (splits < 1) splits = 1;
            if (splits > MAX_SPLITS) splits = MAX_SPLITS;
        }
        const int partials_per_row = splits;
        // One predicate decides the whole route: split accumulation, the stripe
        // kernels (which have no direct-write form), and a q8 emit (the combine
        // kernel is the only emitter, and `out` is null on that path) all
        // require the partial pool + combine. If the pool cannot be allocated,
        // launching anything would either write through a null `out` or leave
        // `q8_out` uninitialized while reporting success — so a required pool
        // that fails to allocate fails the launch loudly instead.
        const bool need_pool = use_stripe || (splits > 1) || (q8_out != nullptr);
        float* pa = nullptr;
        float* pm = nullptr;
        if (need_pool) {
            fused_attn_partial_pool((int64_t)num_active_slots * n_q_head, partials_per_row,
                                    HEAD_DIM, &pa, &pm, stream);
            if (pa == nullptr || pm == nullptr) {
                return 1;
            }
        }

        dim3 grid(num_active_slots, n_kv_head, splits);
        dim3 block(WARP_SIZE * WARPS_PER_BLOCK);

        // **The stripe/bmma family only exists below 16 warps.**
        //
        // `use_stripe` requires `heads_per_group <= 8`, and 16 warps is selected
        // only when `heads_per_group > 8` (`use_wide`, above) — so at 16 warps
        // the branch below is unreachable by construction. Without the
        // `if constexpr` it was still *instantiated*: eight `BMMA` plus eight
        // `STRIPE` kernels per (head dim × rope × dtype), compiled into the
        // archive and never launched. Gating the instantiation rather than the
        // launch is what removes them.
        bool launched_stripe = false;
        if constexpr (WARPS_PER_BLOCK <= 8) {
        if (use_stripe) {
            // HPG compile-time so the per-head flash-state arrays stay in registers.
            // hd128 uses the batched-M INT8 tensor-core MMA; other head dims (no
            // 32-wide palette) use the CUDA-core warp-stripe.
            #define BMMA_LAUNCH(H)                                                         \
                int8_decode_bmma_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK,              \
                                        ROPE_INTERLEAVED, H>                               \
                    <<<grid, block, 0, stream>>>(                                          \
                        q, headers_ptr, num_active_slots, n_q_head, n_kv_head,             \
                        softmax_scale, k_new, v_new, rope_cs, pa, pm, sel)
            #define STRIPE_LAUNCH(H)                                                       \
                int8_decode_stripe_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK,            \
                                          ROPE_INTERLEAVED, H>                             \
                    <<<grid, block, 0, stream>>>(                                          \
                        q, headers_ptr, num_active_slots, n_q_head, n_kv_head,             \
                        softmax_scale, k_new, v_new, rope_cs, pa, pm, sel)
            if constexpr (HEAD_DIM == 128 && WARPS_PER_BLOCK <= 8) {
                // batched-M's per-warp tile smem fits at WARPS<=8 (~29 KB);
                // WARPS=16 (the hpg>8 wide path) never reaches use_stripe, so it
                // would only be a compiled-never-run instantiation that blows the
                // 48 KB cap — route it to the stripe instead.
                switch (heads_per_group) {
                    case 1: BMMA_LAUNCH(1); break;
                    case 2: BMMA_LAUNCH(2); break;
                    case 3: BMMA_LAUNCH(3); break;
                    case 4: BMMA_LAUNCH(4); break;
                    case 5: BMMA_LAUNCH(5); break;
                    case 6: BMMA_LAUNCH(6); break;
                    case 7: BMMA_LAUNCH(7); break;
                    case 8: BMMA_LAUNCH(8); break;
                    default: break;
                }
            } else {
                switch (heads_per_group) {
                    case 1: STRIPE_LAUNCH(1); break;
                    case 2: STRIPE_LAUNCH(2); break;
                    case 3: STRIPE_LAUNCH(3); break;
                    case 4: STRIPE_LAUNCH(4); break;
                    case 5: STRIPE_LAUNCH(5); break;
                    case 6: STRIPE_LAUNCH(6); break;
                    case 7: STRIPE_LAUNCH(7); break;
                    case 8: STRIPE_LAUNCH(8); break;
                    default: break;
                }
            }
            #undef BMMA_LAUNCH
            #undef STRIPE_LAUNCH
            launched_stripe = true;
        }
        }
        if (!launched_stripe) {
            // Existing INT8-MMA kernel (warp=head). `pa` is non-null exactly
            // when the route goes through partials + combine (need_pool held
            // and the alloc succeeded — a failed alloc returned above); null
            // `pa` is the single-block direct write.
            int8_decode_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, ROPE_INTERLEAVED>
                <<<grid, block, 0, stream>>>(
                    q, headers_ptr, out, num_active_slots, n_q_head, n_kv_head,
                    softmax_scale, k_new, v_new, rope_cs, pa, pm, sel);
        }

        // The write-slice commit rides in the combine when there is one; the
        // direct-write route (no partials) commits with its own launch.
        if (pa != nullptr) {
            int num_rows = num_active_slots * n_q_head;
            const int64_t g_stride =
                (gate_slot_stride != 0) ? gate_slot_stride : (int64_t)n_q_head * HEAD_DIM;
            int8_decode_combine_kernel<O, HEAD_DIM><<<num_rows, HEAD_DIM, 0, stream>>>(
                out, pa, pm, num_rows, partials_per_row, q8_out, gate, g_stride, n_q_head,
                headers_ptr, n_kv_head);
        } else {
            constexpr int COMMIT_THREADS = 128;
            dim3 commit_grid((num_active_slots + COMMIT_THREADS - 1) / COMMIT_THREADS);
            commit_decode_write_len_kernel<HEAD_DIM><<<commit_grid, COMMIT_THREADS, 0, stream>>>(
                headers_ptr, num_active_slots, n_kv_head);
        }
        return 0;
    };

    // 4-way dispatch over (use_wide, rope_interleaved). use_wide selects
    // WARPS_PER_BLOCK=16 for heads_per_group > 8 (e.g. Llama-3 70B class), else 8.
    //
    // The `if constexpr` is the same argument as the stripe gate above, one level
    // out: `use_wide` carries `HEAD_DIM >= 128`, so below that head dim it is
    // always false and the entire 16-warp half of this dispatch — every kernel
    // `launch` instantiates — was compiled for a branch that cannot be taken.
    if constexpr (HEAD_DIM >= 128) {
        if (use_wide) {
            if (rope_interleaved) {
                return launch(std::integral_constant<int, 16>{}, std::true_type{});
            }
            return launch(std::integral_constant<int, 16>{}, std::false_type{});
        }
    }
    if (rope_interleaved) {
        return launch(std::integral_constant<int, 8>{}, std::true_type{});
    }
    return launch(std::integral_constant<int, 8>{}, std::false_type{});
}

} // namespace fused_attn
