#pragma once
// =============================================================================
// int8_decode_kernel.cuh — v2-API-compatible decode-attention kernel (Track A).
//
// Drop-in for v2's paged_decode_attn, computing the QK^T as an INT8 m16n8k32 MMA
// per palette (lane-collective INT8 dot fallback for head dims whose palette
// isn't 32-wide) and the PV in INT8, with a per-32-token tile-batched softmax
// and the §1A V read-through. Same slot-header / paged-arena interface as v2.
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
#include "../mma/mma_wrappers.cuh"

namespace fused_attn {

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
    float* __restrict__ partial_ml     // split-KV: [slot*n_q_head+qh][split][2] = (m, l)
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
    const uint32_t n_slices  = slot.n_slices;
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

    auto process_tile = [&](int tile_idx, int stage) {
        int tile_slice, tile_within_base;
        tile_to_slice(tile_idx, tile_slice, tile_within_base);
        // tile_off/tile_bv frame the in-chunk validity window for the WARPS
        // tokens of this tile; slice_eff_len already folds in the writer's +1, so
        // a token is valid while within_base + t < tile_off + tile_bv.
        uint32_t tile_off = 0;
        uint32_t tile_bv = (uint32_t)CHUNK_SIZE;
        if (tile_slice < (int)n_slices) {
            const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, tile_slice, n_kv_head);
            tile_off = (uint32_t)slice_offset(sl);
            tile_bv = (uint32_t)slice_eff_len(tile_slice);
        }
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
                int actual_within = tile_within_base + t;
                bool valid = (tile_slice < (int)n_slices &&
                              actual_within < (int)(tile_off + tile_bv));
                float s = valid ? tile_logits[stage][warp][t] * softmax_scale : -1e38f;
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
                int actual_within = tile_within_base + t;
                bool valid = (tile_slice < (int)n_slices &&
                              actual_within < (int)(tile_off + tile_bv));
                // Recompute the score from tile_logits[] (smem). s <= -1e37 marks
                // an invalid token; guard so an all-invalid tile (new_m==-1e38)
                // can't yield exp2(0)=1.
                float s = valid ? tile_logits[stage][warp][t] * softmax_scale : -1e38f;
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
    float* partial_ml
) {
    constexpr bool IS_HALF_TYPE = std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>;
    // The warp=head (wide) kernel runs only for heads_per_group > 8 (WARPS=16).
    // At HEAD_DIM=256 a single pipeline stage is ~27 KB but two stages is ~55 KB,
    // which overflows the 48 KiB static shared-memory cap — so HEAD_DIM=256 runs
    // single-stage here. This path is exotic (no target model has hpg>8 at hd256;
    // real hd256 models like Gemma have small GQA ratios and take the full-perf
    // stripe path), so the lost load/compute overlap is irrelevant. Every
    // HEAD_DIM <= 128 instantiation keeps its original stage count unchanged.
    constexpr int STAGES = (HEAD_DIM >= 256) ? 1 : (IS_HALF_TYPE ? 3 : 2);
    int8_decode_attn_impl<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, 32, STAGES, true, ROPE_INTERLEAVED>(
        q, headers_ptr, out, num_active_slots, n_q_head, n_kv_head, softmax_scale,
        k_new, v_new, rope_cs, partial_acc, partial_ml);
}

// =============================================================================
// WARP-STRIPE decode (1C) — every warp computes. Each warp walks its own KV
// token stripe for ALL heads in the group, one token at a time, accumulating
// per-head flash-state, then emits one partial per head. The combine folds the
// warp axis in for free: the partial index is (split * WARPS_PER_BLOCK + warp),
// so it reduces over num_splits*WARPS partials per (slot, head). Used when
// heads_per_group < WARPS_PER_BLOCK (the GQA case that idles 5/8 warps).
// FP path (v1); manual-INT8 dot + V read-through to follow (M3).
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
    float* __restrict__ partial_ml
) {
    constexpr int VEC = HEAD_DIM / WARP_SIZE;
    constexpr int N_PALETTE = 4;
    constexpr int SUB_HEAD_DIM = HEAD_DIM / N_PALETTE;
    constexpr int MAXH = HPG;  // heads-per-group is compile-time → arrays in registers

    int slot_idx = (int)blockIdx.x;
    int kv_head_idx = (int)blockIdx.y;
    int split_idx = (int)blockIdx.z;
    int num_splits = (int)gridDim.z;
    int tid = (int)threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;
    if (slot_idx >= num_active_slots || kv_head_idx >= n_kv_head) return;

    constexpr int hpg = HPG;
    int num_partials = num_splits * WARPS_PER_BLOCK;

    // Per-head flash-state for this warp's stripe (un-normalized ΣwV, m, l).
    float out_reg[MAXH][VEC];
    float m_i[MAXH], l_i[MAXH];
    #pragma unroll
    for (int h = 0; h < MAXH; ++h) {
        m_i[h] = -1e38f; l_i[h] = 0.f;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) out_reg[h][j] = 0.f;
    }

    auto emit_partial = [&](int qh_local) {
        int qh = kv_head_idx * hpg + qh_local;
        if (qh >= n_q_head) return;
        int64_t base = ((int64_t)slot_idx * n_q_head + qh) * num_partials
                     + (int64_t)split_idx * WARPS_PER_BLOCK + warp;
        float* acc = partial_acc + base * HEAD_DIM;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) acc[lane * VEC + j] = out_reg[qh_local][j];
        if (lane == 0) { partial_ml[base * 2] = m_i[qh_local]; partial_ml[base * 2 + 1] = l_i[qh_local]; }
    };

    const SlotHeader& slot = get_slot_header(headers_ptr, slot_idx);
    const uint32_t n_slices = slot.n_slices;
    const uint32_t write_slice_idx = slot.write_slice;
    const uint64_t slices_ptr = slot.slices_ptr;

    if (n_slices == 0) {
        for (int h = 0; h < hpg; ++h) emit_partial(h);  // null partials
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
    if (kv_len <= 0) { for (int h = 0; h < hpg; ++h) emit_partial(h); return; }
    int max_len = (int)n_slices * CHUNK_SIZE;
    if (kv_len > max_len) kv_len = max_len;

    // Q for all heads (logical, RoPE'd) in SHARED smem — the query is
    // warp-independent, so one copy serves every warp and it stays out of
    // per-thread registers/stack. Loaded once by warp 0.
    __shared__ float shared_q[HPG][HEAD_DIM];
    if (warp == 0) {
        uint32_t q_rope_pos = (uint32_t)ws_rope + (uint32_t)ws_len;
        #pragma unroll
        for (int h = 0; h < HPG; ++h) {
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
    }
    __syncthreads();

    // Per-warp token K/V scratch (FP, palette order before the ki/vi gather).
    __shared__ alignas(128) T sk[WARPS_PER_BLOCK][HEAD_DIM];
    __shared__ alignas(128) T sv[WARPS_PER_BLOCK][HEAD_DIM];

    constexpr int64_t sub_head_stride = (int64_t)SUB_HEAD_DIM * CHUNK_SIZE;
    constexpr int BLOCKS_PER_DIM = CHUNK_SIZE / 32;

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
    int total_tok = 0;
    for (int s = 0; s < (int)n_slices; ++s) total_tok += slice_eff_len(s);

    int n_tiles = (total_tok + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    int tiles_per_split = (n_tiles + num_splits - 1) / num_splits;
    int tok_lo = (split_idx * tiles_per_split) * WARPS_PER_BLOCK;
    int tok_hi = (split_idx * tiles_per_split + tiles_per_split) * WARPS_PER_BLOCK;
    if (tok_hi > total_tok) tok_hi = total_tok;

    PalIter<VEC, HEAD_DIM> ki, vi;
    int cur_slice = -1;
    // Forward cursor: t is monotonic within a warp's strided iteration, so the
    // (slice, base) cursor only advances.
    int scan_s = 0, scan_base = 0;

    for (int k = tok_lo + warp; k < tok_hi; k += WARPS_PER_BLOCK) {
        while (scan_s + 1 < (int)n_slices) {
            int e = slice_eff_len(scan_s);
            if (scan_base + e <= k) { scan_base += e; ++scan_s; }
            else break;
        }
        int slice_idx = scan_s;
        const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, slice_idx, n_kv_head);
        uint32_t off = (uint32_t)slice_offset(sl);
        // within = off + local; slice_eff_len already accounts for the writer's
        // +1, so (k - scan_base) reaches the freshly-scattered token's slot.
        int within = (int)off + (k - scan_base);
        const uint8_t* head_ptr = get_head<HEAD_DIM>(sl, kv_head_idx);

        if (slice_idx != cur_slice) {
            ki.init(kvhead_k_pal_map<HEAD_DIM>(head_ptr), lane);
            vi.init(kvhead_v_pal_map<HEAD_DIM>(head_ptr), lane);
            cur_slice = slice_idx;
        }

        #pragma unroll
        for (int p = 0; p < N_PALETTE; ++p) {
            uint64_t k_ptr_p = kvhead_k_ptr<HEAD_DIM>(head_ptr, p);
            float k_scale_p = kvhead_k_scale<HEAD_DIM>(head_ptr, p);
            int k_fmt = kvhead_k_fmt<HEAD_DIM>(head_ptr, p);
            if (k_ptr_p) {
                ArenaAccessor ka((const char*)(uintptr_t)k_ptr_p, k_fmt, sub_head_stride, sub_head_stride, BLOCKS_PER_DIM, 0);
                ka.template load_head_scaled<T, SUB_HEAD_DIM, false>(sk[warp] + p * SUB_HEAD_DIM, 0, 0, within, lane, k_scale_p);
            }
            uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(head_ptr, p);
            float v_scale_p = kvhead_v_scale<HEAD_DIM>(head_ptr, p);
            int v_fmt = kvhead_v_fmt<HEAD_DIM>(head_ptr, p);
            if (v_ptr_p) {
                ArenaAccessor va((const char*)(uintptr_t)v_ptr_p, v_fmt, sub_head_stride, sub_head_stride, BLOCKS_PER_DIM, 0);
                va.template load_head_scaled<T, SUB_HEAD_DIM, false>(sv[warp] + p * SUB_HEAD_DIM, 0, 0, within, lane, v_scale_p);
            }
        }
        __syncwarp();

        float k_regs[VEC], v_regs[VEC];
        #pragma unroll
        for (int j = 0; j < VEC; ++j) {
            k_regs[j] = to_f32<T>(sk[warp][ki[j]]);
            v_regs[j] = to_f32<T>(sv[warp][vi[j]]);
        }
        const int32_t rope_pos = (int32_t)slice_rope(sl) + (within - (int)off);
        if constexpr (ROPE_INTERLEAVED && (VEC == 1 || VEC % 2 == 0))
            apply_rope_interleaved_f32<VEC, HEAD_DIM>(k_regs, lane, rope_pos, rope_cs);
        else
            apply_rope_rotary_f32<VEC, HEAD_DIM>(k_regs, lane, rope_pos, rope_cs);

        #pragma unroll 1
        for (int h = 0; h < hpg; ++h) {
            float dr = 0.f;
            #pragma unroll
            for (int j = 0; j < VEC; ++j) dr = __fmaf_rn(shared_q[h][lane * VEC + j], k_regs[j], dr);
            dr = warp_reduce_sum(dr);
            float logit = dr * softmax_scale;
            float new_m = fmaxf(m_i[h], logit);
            float alpha = fast_exp::exp2<float, fast_exp::Softmax>(make_float2(m_i[h] - new_m, 0.f)).x;
            float beta = fast_exp::exp2<float, fast_exp::Softmax>(make_float2(logit - new_m, 0.f)).x;
            l_i[h] = l_i[h] * alpha + beta;
            #pragma unroll
            for (int j = 0; j < VEC; ++j) out_reg[h][j] = out_reg[h][j] * alpha + beta * v_regs[j];
            m_i[h] = new_m;
        }
    }

    for (int h = 0; h < hpg; ++h) emit_partial(h);
}

template <typename Q_T, typename T, typename O,
          int HEAD_DIM, int WARPS_PER_BLOCK, bool ROPE_INTERLEAVED, int HPG>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE,
                                   int8_decode_min_blocks<WARPS_PER_BLOCK>())
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
    float* partial_ml
) {
    int8_decode_stripe_impl<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, ROPE_INTERLEAVED, HPG>(
        q, headers_ptr, num_active_slots, n_q_head, n_kv_head, softmax_scale,
        k_new, v_new, rope_cs, partial_acc, partial_ml);
}

// =============================================================================
// BATCHED-M decode (1C final) — INT8 tensor-core MMA + read-through V.
// warp = tile-stripe (all warps compute). Per tile the warp runs an m16n8k32
// INT8 MMA over its 8 tokens (N=8) for all HPG query heads at once (M=HPG),
// 4 MMAs (one per 32-wide palette). C is extracted to scores_smem, then a
// per-head flash softmax + read-through INT8 V PV. Partials fold split*warp
// into the combine, as the stripe does. See docs/decode_kernel_batched_m.md.
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
    float* __restrict__ partial_ml
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
    int num_partials = num_splits * WARPS_PER_BLOCK;

    float out_reg[HPG][VEC];
    float m_i[HPG], l_i[HPG];
    #pragma unroll
    for (int h = 0; h < HPG; ++h) {
        m_i[h] = -1e38f; l_i[h] = 0.f;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) out_reg[h][j] = 0.f;
    }

    auto emit_partial = [&](int qh_local) {
        int qh = kv_head_idx * hpg + qh_local;
        if (qh >= n_q_head) return;
        int64_t base = ((int64_t)slot_idx * n_q_head + qh) * num_partials
                     + (int64_t)split_idx * WARPS_PER_BLOCK + warp;
        float* acc = partial_acc + base * HEAD_DIM;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) acc[lane * VEC + j] = out_reg[qh_local][j];
        if (lane == 0) { partial_ml[base * 2] = m_i[qh_local]; partial_ml[base * 2 + 1] = l_i[qh_local]; }
    };

    const SlotHeader& slot = get_slot_header(headers_ptr, slot_idx);
    const uint32_t n_slices = slot.n_slices;
    const uint32_t write_slice_idx = slot.write_slice;
    const uint64_t slices_ptr = slot.slices_ptr;
    if (n_slices == 0) { for (int h = 0; h < hpg; ++h) emit_partial(h); return; }

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

    for (int h = 0; h < hpg; ++h) emit_partial(h);
}

template <typename Q_T, typename T, typename O,
          int HEAD_DIM, int WARPS_PER_BLOCK, bool ROPE_INTERLEAVED, int HPG>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE,
                                   int8_decode_min_blocks<WARPS_PER_BLOCK>())
int8_decode_bmma_kernel(
    const Q_T* q, const uint8_t* headers_ptr, int num_active_slots,
    int n_q_head, int n_kv_head, float softmax_scale,
    const T* k_new, const T* v_new, const float* rope_cs,
    float* partial_acc, float* partial_ml
) {
    int8_decode_bmma_impl<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, ROPE_INTERLEAVED, HPG>(
        q, headers_ptr, num_active_slots, n_q_head, n_kv_head, softmax_scale,
        k_new, v_new, rope_cs, partial_acc, partial_ml);
}

// -----------------------------------------------------------------------------
// Split-KV combine: merge the num_splits per-split partial flash-states for each
// (slot, query-head) into the final normalized output. One block per output row
// (slot*n_q_head + qh); HEAD_DIM threads, each owning one output dim. The merge
// is the standard log-sum-exp (base-2, matching the decode kernel's exp2):
//   gm = max_s m_s;  out = (Σ_s ΣwV_s · 2^(m_s-gm)) / (Σ_s l_s · 2^(m_s-gm)).
// Null partials (m=-inf, l=0) contribute zero.
// -----------------------------------------------------------------------------
template <typename O, int HEAD_DIM>
__global__ void int8_decode_combine_kernel(
    O* __restrict__ out,
    const float* __restrict__ partial_acc,   // [row][split][HEAD_DIM]
    const float* __restrict__ partial_ml,    // [row][split][2]
    int num_rows,
    int num_splits,
    uint8_t* __restrict__ q8_out             // non-null → emit q8a128 (B2; HEAD_DIM==128 only)
) {
    int row = (int)blockIdx.x;
    if (row >= num_rows) return;
    int d = (int)threadIdx.x;
    if (d >= HEAD_DIM) return;

    const float* ml = partial_ml + (int64_t)row * num_splits * 2;
    const float* pa = partial_acc + (int64_t)row * num_splits * HEAD_DIM;

    float gm = -1e38f;
    for (int s = 0; s < num_splits; ++s) gm = fmaxf(gm, ml[s * 2]);

    float acc = 0.f, L = 0.f;
    for (int s = 0; s < num_splits; ++s) {
        float w = exp2f(ml[s * 2] - gm);
        acc += pa[(int64_t)s * HEAD_DIM + d] * w;
        L   += ml[s * 2 + 1] * w;
    }
    float inv = __fdividef(1.f, fmaxf(L, 1e-10f));
    float val = acc * inv;

    // B2: fused attention → q8a128 context emit. At HEAD_DIM==128 one block (one
    // query head, 128 threads) is exactly one q8a128 128-tile; the context row for
    // a token is n_q_head heads × 128 = hidden, so flat_tile = row (= token*n_q_head
    // + qh = token*tiles_per_row + qh). Block-reduce amax/Σx over the 128 dims, then
    // thread d writes its quant byte and thread 0 the per-128 {scale,sum}. The value
    // is rounded through O first to mirror the unfused FP store + re-quant.
    //
    // These bytes are MODE-AGNOSTIC: the q8a1024 flat-grouped layout is byte-identical
    // for the matmul's V (mode-1, Bm=16) and X (mode-2, Bm=32) variants — the mode only
    // changes how the matmul tiles the SAME bytes. So this kernel never decides V vs X.
    // That choice rides in the `Q8a128Operand.ytype`, derived from the token count M via
    // `q8a128_mode_for_m()` when the rust side wraps `q8_out` into the operand (M ≥ 64 →
    // X). Hard-coding a mode here would be wrong; it is carried in the DynamicTensor.
    if constexpr (HEAD_DIM == 128) {
        if (q8_out != nullptr) {
            const float vr = to_f32<O>(from_f32<O>(val));
            float amax = fabsf(vr);
            float s = vr;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                amax = fmaxf(amax, __shfl_xor_sync(0xffffffff, amax, off, 32));
                s += __shfl_xor_sync(0xffffffff, s, off, 32);
            }
            __shared__ float sh_amax[HEAD_DIM / 32];
            __shared__ float sh_sum[HEAD_DIM / 32];
            const int warp = d >> 5;
            const int lane = d & 31;
            if (lane == 0) { sh_amax[warp] = amax; sh_sum[warp] = s; }
            __syncthreads();
            if (warp == 0) {
                float a = (lane < HEAD_DIM / 32) ? sh_amax[lane] : 0.f;
                float ss = (lane < HEAD_DIM / 32) ? sh_sum[lane] : 0.f;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    a = fmaxf(a, __shfl_xor_sync(0xffffffff, a, off, 32));
                    ss += __shfl_xor_sync(0xffffffff, ss, off, 32);
                }
                if (lane == 0) { sh_amax[0] = a; sh_sum[0] = ss; }
            }
            __syncthreads();
            const float tile_amax = sh_amax[0];
            const float tile_sum = sh_sum[0];
            const float id = (tile_amax != 0.f) ? 127.f / tile_amax : 0.f;
            uint8_t* obytes = q8_out;
            const int64_t flat = row;
            obytes[q8a1024_qs_off(flat) + d] = (int8_t)__float2int_rn(vr * id);
            if (d == 0) {
                half2* ds = reinterpret_cast<half2*>(obytes + q8a1024_ds_off(flat));
                ds[0] = make_half2(__float2half_rn(tile_amax / 127.f), __float2half_rn(tile_sum));
            }
            return;
        }
    }
    out[(int64_t)row * HEAD_DIM + d] = from_f32<O>(val);
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

template <typename Q_T, typename T, typename O, int HEAD_DIM>
void launch_int8_decode_attn(
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
    uint8_t* q8_out = nullptr   // non-null → B2 fused q8a128 context (combine path, HEAD_DIM==128)
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
    constexpr int MAX_SPLITS = 32;
    if (num_splits < 1) num_splits = 1;
    if (num_splits > MAX_SPLITS) num_splits = MAX_SPLITS;

    auto launch = [&](auto warps_const, auto rope_const) {
        constexpr int WARPS_PER_BLOCK = decltype(warps_const)::value;
        constexpr bool ROPE_INTERLEAVED = decltype(rope_const)::value;

        // Warp-stripe (1C) when heads_per_group <= WARPS: every warp computes,
        // each over its own KV stripe for all heads, and the partial index folds
        // in the warp axis (split*WARPS + warp) — so it always writes partials +
        // combines. partials_per_row = num_splits*WARPS for stripe, else num_splits.
        // hpg==8 (e.g. Qwen3-MoE, n_q/n_kv=32/4) is included: the batched-M MMA
        // path is gap-aware and the warp=head path is not, so route it here.
        const bool use_stripe = (heads_per_group >= 1 && heads_per_group <= 8
                                 && heads_per_group <= WARPS_PER_BLOCK);
        const int partials_per_row = use_stripe ? (num_splits * WARPS_PER_BLOCK) : num_splits;
        const bool need_pool = use_stripe || (num_splits > 1);
        float* pa = nullptr;
        float* pm = nullptr;
        if (need_pool) {
            fused_attn_partial_pool((int64_t)num_active_slots * n_q_head, partials_per_row,
                                    HEAD_DIM, &pa, &pm, stream);
        }

        dim3 grid(num_active_slots, n_kv_head, num_splits);
        dim3 block(WARP_SIZE * WARPS_PER_BLOCK);
        if (use_stripe && pa != nullptr) {
            // HPG compile-time so the per-head flash-state arrays stay in registers.
            // hd128 uses the batched-M INT8 tensor-core MMA; other head dims (no
            // 32-wide palette) use the CUDA-core warp-stripe.
            #define BMMA_LAUNCH(H)                                                         \
                int8_decode_bmma_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK,              \
                                        ROPE_INTERLEAVED, H>                               \
                    <<<grid, block, 0, stream>>>(                                          \
                        q, headers_ptr, num_active_slots, n_q_head, n_kv_head,             \
                        softmax_scale, k_new, v_new, rope_cs, pa, pm)
            #define STRIPE_LAUNCH(H)                                                       \
                int8_decode_stripe_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK,            \
                                          ROPE_INTERLEAVED, H>                             \
                    <<<grid, block, 0, stream>>>(                                          \
                        q, headers_ptr, num_active_slots, n_q_head, n_kv_head,             \
                        softmax_scale, k_new, v_new, rope_cs, pa, pm)
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
        } else {
            // Existing INT8-MMA kernel (warp=head). Direct write if no pool
            // (single block) — also the fallback if the stripe pool alloc failed.
            float* kpa = (pa != nullptr && num_splits > 1) ? pa : nullptr;
            float* kpm = (pm != nullptr && num_splits > 1) ? pm : nullptr;
            int8_decode_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, ROPE_INTERLEAVED>
                <<<grid, block, 0, stream>>>(
                    q, headers_ptr, out, num_active_slots, n_q_head, n_kv_head,
                    softmax_scale, k_new, v_new, rope_cs, kpa, kpm);
        }

        if (pa != nullptr && (use_stripe || num_splits > 1)) {
            int num_rows = num_active_slots * n_q_head;
            int8_decode_combine_kernel<O, HEAD_DIM><<<num_rows, HEAD_DIM, 0, stream>>>(
                out, pa, pm, num_rows, partials_per_row, q8_out);
        }

        constexpr int COMMIT_THREADS = 128;
        dim3 commit_grid((num_active_slots + COMMIT_THREADS - 1) / COMMIT_THREADS);
        commit_decode_write_len_kernel<HEAD_DIM><<<commit_grid, COMMIT_THREADS, 0, stream>>>(
            headers_ptr, num_active_slots, n_kv_head);
    };

    // 4-way dispatch over (use_wide, rope_interleaved). use_wide selects
    // WARPS_PER_BLOCK=16 for heads_per_group > 8 (e.g. Llama-3 70B class), else 8.
    if (use_wide) {
        if (rope_interleaved) {
            launch(std::integral_constant<int, 16>{}, std::true_type{});
        } else {
            launch(std::integral_constant<int, 16>{}, std::false_type{});
        }
    } else {
        if (rope_interleaved) {
            launch(std::integral_constant<int, 8>{}, std::true_type{});
        } else {
            launch(std::integral_constant<int, 8>{}, std::false_type{});
        }
    }
}

} // namespace fused_attn
