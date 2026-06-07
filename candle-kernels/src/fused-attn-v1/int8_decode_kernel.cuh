#pragma once
// =============================================================================
// int8_decode_kernel.cuh — v2-API-compatible decode-attention kernel that we
// own and evolve incrementally toward the design's INT8 MMA path.
//
// ITERATION LADDER:
//   I0  Clone of v2's `paged_decode_attn_v2_impl` body verbatim.   <-- NOW
//   I1  Replace QK^T fmaf with INT8 MMA per palette.
//   I2  Replace PV fmaf with INT8 MMA per output tile.
//   I3  Add per-block activation INT8 quant for fused QKV (later phase).
//
// The body is structurally a copy of v2 so we can diff and modify it in place.
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
#include "../paged-decode/slot_types.cuh"
#include "../paged-decode/pal_iter.cuh"
// Pull in v2's local helpers (vec2_traits, load_vec2, cp_async_*, RoPE, scatter
// helpers). They're inline templates living in paged_decode_kernel.cuh.
#include "../paged-decode/paged_decode_kernel.cuh"
#include "mma_wrappers.cuh"

namespace fused_attn {

// INT8_QK / INT8_PV flags isolate the INT8 conversions:
//   INT8_QK=false, INT8_PV=false → pure FP fmaf (iter-1 baseline; matches v2)
//   INT8_QK=true,  INT8_PV=false → INT8 QK^T + FP PV
//   INT8_QK=false, INT8_PV=true  → FP QK^T + INT8 manual PV
//   INT8_QK=true,  INT8_PV=true  → fully INT8 (iter-2b/3 target)
//
// INT8_QK_USE_MMA selects MMA (true, default) vs naive lane-collective dot
// product (false) for the INT8 QK^T computation. The manual dot uses the
// SAME q_int8 / k_int8 / scale_Q / shared_k_scale data as the MMA, so any
// agreement between FP and manual-dot paths but disagreement with MMA
// localises the bug to the MMA fragment assembly or scale composition.
template <typename Q_T, typename T, typename O,
          int HEAD_DIM, int WARPS_PER_BLOCK,
          int TILE_K = 32, int NUM_STAGES = 2,
          bool USE_TC = false, bool ROPE_INTERLEAVED = false,
          bool INT8_QK = true, bool INT8_PV = true,
          bool INT8_QK_USE_MMA = true>
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
    const float* __restrict__ rope_cs
) {
    constexpr int VEC = HEAD_DIM / WARP_SIZE;
    static_assert(HEAD_DIM % WARP_SIZE == 0, "HEAD_DIM must be multiple of 32");
    static_assert(VEC <= 8, "HEAD_DIM must be <= 256");
    static_assert(NUM_STAGES >= 1 && NUM_STAGES <= 3, "NUM_STAGES must be 1-3");
    static_assert(CHUNK_SIZE % WARPS_PER_BLOCK == 0,
        "CHUNK_SIZE must be a multiple of WARPS_PER_BLOCK");

    int slot_idx = (int)blockIdx.x;
    int kv_head_idx = (int)blockIdx.y;
    int tid = (int)threadIdx.x;
    int warp = tid / WARP_SIZE;
    int lane = tid % WARP_SIZE;

    if (slot_idx >= num_active_slots || kv_head_idx >= n_kv_head) return;

    const SlotHeader& slot = get_slot_header(headers_ptr, slot_idx);
    const uint32_t n_slices  = slot.n_slices;
    const uint32_t write_slice_idx = slot.write_slice;
    const uint64_t slices_ptr = slot.slices_ptr;

    if (n_slices == 0) {
        int heads_per_group = n_q_head / n_kv_head;
        if (heads_per_group <= 0) heads_per_group = 1;
        int head_idx = kv_head_idx * heads_per_group + warp;
        bool warp_active = (warp < heads_per_group) && (head_idx < n_q_head);
        if (warp_active) {
            int64_t out_base = ((int64_t)slot_idx * (int64_t)n_q_head + (int64_t)head_idx) * (int64_t)HEAD_DIM;
            #pragma unroll
            for (int j = 0; j < VEC; ++j)
                out[out_base + lane * VEC + j] = from_f32<O>(0.f);
        }
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
        if (warp_active) {
            int64_t out_base = ((int64_t)slot_idx * (int64_t)n_q_head + (int64_t)head_idx) * (int64_t)HEAD_DIM;
            #pragma unroll
            for (int j = 0; j < VEC; ++j)
                out[out_base + lane * VEC + j] = from_f32<O>(0.f);
        }
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

    // Per-tile palette iterators (refresh on slice boundary).
    PalIter<VEC, HEAD_DIM> ki, vi;
    int kv_pal_slice_idx = -1;
    auto maybe_init_kv_iters_for_tile = [&](int tile_idx) {
        int tile_k_base = tile_idx * WARPS_PER_BLOCK;
        int tile_slice_idx = chunk_div(tile_k_base);
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
    // contributing equally across head_dim).
    __shared__ alignas(128) int8_t shared_v_int8[NUM_STAGES][WARPS_PER_BLOCK][HEAD_DIM];
    __shared__ alignas(16)  float  shared_v_scale[NUM_STAGES][WARPS_PER_BLOCK];

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
    constexpr bool USE_MMA_QK = INT8_QK_USE_MMA && (SUB_HEAD_DIM == 32);

    // ─── Per-palette Q quantization (warp-collective) ──────────────────
    // q_reg holds VEC=HEAD_DIM/32 dims per lane; lane t covers dims [t*VEC..t*VEC+VEC).
    // Each palette p covers dims [p*SUB_HEAD_DIM..(p+1)*SUB_HEAD_DIM) which spans
    // the lanes [p*8 .. p*8+7] when VEC=4. Max-abs is reduced inside that 8-lane
    // group via xor-1/2/4, leaving every member with the palette's max-abs.
    int8_t q_int8[VEC];
    uint32_t q_packed = 0;  // packed VEC=4 INT8 of this lane's Q dims (for MMA shuffles)
    float scale_Q[N_PALETTE];
    if constexpr (INT8_QK) {
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
    } else {
        #pragma unroll
        for (int p = 0; p < N_PALETTE; ++p) scale_Q[p] = 1.f;
        #pragma unroll
        for (int j = 0; j < VEC; ++j) q_int8[j] = 0;
    }

    float m_i = -1e38f;
    float l_i = 0.f;
    float out_reg[VEC];
    #pragma unroll
    for (int j = 0; j < VEC; ++j) out_reg[j] = 0.f;

    const int n_tiles = (kv_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    constexpr int BLOCKS_PER_DIM = CHUNK_SIZE / 32;

    auto load_tile = [&](int tile_idx, int stage) {
        int k_base = tile_idx * WARPS_PER_BLOCK;
        int k_idx = k_base + warp;
        bool valid_k = (k_idx < kv_len);
        T* k_dst = shared_k[stage][warp];
        T* v_dst = shared_v[stage][warp];
        int my_slice_idx = chunk_div(k_idx);
        int within = chunk_mod(k_idx);
        if (valid_k && my_slice_idx < (int)n_slices) {
            const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, my_slice_idx, n_kv_head);
            uint32_t bv  = (uint32_t)slice_len(sl);
            uint32_t off = (uint32_t)slice_offset(sl);
            if (my_slice_idx == (int)write_slice_idx &&
                bv < CHUNK_SIZE && off + bv < CHUNK_SIZE) {
                bv += 1;
            }
            valid_k = (within >= (int)off && within < (int)(off + bv));
        } else {
            valid_k = false;
        }
        if (!valid_k) {
            #pragma unroll
            for (int j = 0; j < VEC; ++j) {
                k_dst[lane * VEC + j] = from_f32<T>(0.f);
                v_dst[lane * VEC + j] = from_f32<T>(0.f);
            }
            if constexpr (INT8_QK) {
                int8_t* k_int8_dst = shared_k_int8[stage][warp];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    k_int8_dst[lane * VEC + j] = 0;
                }
                if (lane < N_PALETTE) {
                    shared_k_scale[stage][warp][lane] = 1.f;
                }
            }
            if constexpr (INT8_PV) {
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
            v_acc.template load_head_scaled<T, SUB_HEAD_DIM, USE_TC>(v_dst + p * SUB_HEAD_DIM, 0, 0, within, lane, v_scale_p);
        }
    };

    auto apply_rope_to_tile = [&](int tile_idx, int stage) {
        int k_base = tile_idx * WARPS_PER_BLOCK;
        int k_idx = k_base + warp;
        if (k_idx < kv_len) {
            int my_slice_idx = chunk_div(k_idx);
            int within = chunk_mod(k_idx);
            const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, my_slice_idx, n_kv_head);
            uint32_t bv  = (uint32_t)slice_len(sl);
            uint32_t off = (uint32_t)slice_offset(sl);
            if (my_slice_idx == (int)write_slice_idx &&
                bv < CHUNK_SIZE && off + bv < CHUNK_SIZE) {
                bv += 1;
            }
            if (within >= (int)off && within < (int)(off + bv)) {
                const int32_t rope_base = (int32_t)slice_rope(sl);
                const int32_t rope_pos  = rope_base + (within - (int)off);
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
                if constexpr (INT8_QK) {
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
                if constexpr (INT8_PV) {
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
    };

    auto process_tile = [&](int tile_idx, int stage) {
        int k_base = tile_idx * WARPS_PER_BLOCK;
        int tile_slice = chunk_div(k_base);
        int tile_within_base = chunk_mod(k_base);
        uint32_t tile_bv  = 0;
        uint32_t tile_off = 0;
        if (tile_slice < (int)n_slices) {
            const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, tile_slice, n_kv_head);
            tile_bv  = (uint32_t)slice_len(sl);
            tile_off = (uint32_t)slice_offset(sl);
            if (tile_slice == (int)write_slice_idx &&
                tile_bv < CHUNK_SIZE && tile_off + tile_bv < CHUNK_SIZE) {
                tile_bv += 1;
            }
        } else {
            tile_bv = (uint32_t)CHUNK_SIZE;
            tile_off = 0u;
        }
        // ── QK^T: precompute INT8 logits if INT8_QK=true, broadcast via tile_logits[].
        if constexpr (INT8_QK && USE_MMA_QK) {
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
        } else if constexpr (INT8_QK && !USE_MMA_QK) {
            // ── Manual per-lane INT8 dot — correct for any VEC / SUB_HEAD_DIM.
            // Used for the MMA bug-bisect harness AND as the production path for
            // head dims whose palette isn't 32 dims (e.g. HEAD_DIM=64).
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
            constexpr int TILE_UNROLL = (WARPS_PER_BLOCK <= 8) ? 4 : 2;
            #pragma unroll TILE_UNROLL
            for (int t = 0; t < WARPS_PER_BLOCK; ++t) {
                int actual_k = k_base + t;
                int actual_within = tile_within_base + t;
                float valid_mask = (actual_k < kv_len &&
                                    actual_within >= (int)tile_off &&
                                    actual_within < (int)(tile_off + tile_bv)) ? 1.f : 0.f;

                float dot;
                if constexpr (INT8_QK) {
                    dot = tile_logits[stage][warp][t];
                } else {
                    float d = 0.f;
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j)
                        d = __fmaf_rn(q_reg[j], to_f32<T>(shared_k[stage][t][lane * VEC + j]), d);
                    dot = warp_reduce_sum(d);
                }

                float score = dot * softmax_scale;
                float masked_score = (valid_mask > 0.f) ? score : -1e38f;
                float new_m = fmaxf(m_i, masked_score);
                float2 exp_ab = fast_exp::exp2<float, fast_exp::Softmax>(
                    make_float2(m_i - new_m, masked_score - new_m));
                float alpha = exp_ab.x;
                float beta = valid_mask * exp_ab.y;
                l_i = __fmaf_rn(l_i, alpha, beta);

                if constexpr (INT8_PV) {
                    float beta_abs = fabsf(beta);
                    float beta_scale = beta_abs / 127.f;
                    if (beta_scale == 0.f) beta_scale = 1.f;
                    int beta_q = (int)__float2int_rn(fminf(fmaxf(beta / beta_scale, -127.f), 127.f));
                    float v_scale_t = shared_v_scale[stage][t];
                    float combined_scale = beta_scale * v_scale_t;

                    int8_t* v_int8_t = shared_v_int8[stage][t];
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j) {
                        int32_t prod = (int32_t)beta_q * (int32_t)v_int8_t[lane * VEC + j];
                        out_reg[j] = __fmaf_rn(out_reg[j], alpha, (float)prod * combined_scale);
                    }
                } else if constexpr ((std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) && VEC >= 2) {
                    constexpr int VEC2 = VEC / 2;
                    #pragma unroll
                    for (int j = 0; j < VEC2; ++j) {
                        float2 vf = load_vec2<T>(&shared_v[stage][t][lane * VEC + j * 2]);
                        out_reg[j * 2]     = __fmaf_rn(out_reg[j * 2], alpha, beta * vf.x);
                        out_reg[j * 2 + 1] = __fmaf_rn(out_reg[j * 2 + 1], alpha, beta * vf.y);
                    }
                } else {
                    #pragma unroll
                    for (int j = 0; j < VEC; ++j) {
                        float v_val = to_f32<T>(shared_v[stage][t][lane * VEC + j]);
                        out_reg[j] = __fmaf_rn(out_reg[j], alpha, beta * v_val);
                    }
                }
                m_i = new_m;
            }
        }
    };

    // Pipelined main loop (mirrors v2's structure).
    if constexpr (NUM_STAGES >= 2 && USE_TC) {
        int tiles_loaded = 0;
        if (n_tiles > 0) { load_tile(0, 0); cp_async_commit<USE_TC>(); tiles_loaded = 1; }
        if (n_tiles > 1 && NUM_STAGES >= 2) { load_tile(1, 1); cp_async_commit<USE_TC>(); tiles_loaded = 2; }
        if constexpr (NUM_STAGES >= 3) {
            if (n_tiles > 2) { load_tile(2, 2); cp_async_commit<USE_TC>(); tiles_loaded = 3; }
        }
        if (tiles_loaded >= NUM_STAGES) {
            cp_async_wait<NUM_STAGES - 1, USE_TC>();
        } else if (tiles_loaded == 2) {
            cp_async_wait<1, USE_TC>();
        } else if (tiles_loaded == 1) {
            cp_async_wait<0, USE_TC>();
        }
        __syncthreads();
        if (n_tiles > 0) {
            maybe_init_kv_iters_for_tile(0);
            apply_rope_to_tile(0, 0);
        }
        int cur_stage = 0;
        for (int tile = 0; tile < n_tiles; ++tile) {
            __syncthreads();
            process_tile(tile, cur_stage);
            __syncthreads();
            int prefetch_tile = tile + NUM_STAGES;
            if (prefetch_tile < n_tiles) {
                load_tile(prefetch_tile, cur_stage);
                cp_async_commit<USE_TC>();
            }
            int next_tile = tile + 1;
            if (next_tile < n_tiles) {
                cp_async_wait<NUM_STAGES - 1, USE_TC>();
                __syncthreads();
                maybe_init_kv_iters_for_tile(next_tile);
                apply_rope_to_tile(next_tile, (cur_stage + 1) % NUM_STAGES);
            }
            cur_stage = (cur_stage + 1) % NUM_STAGES;
        }
    } else {
        for (int tile = 0; tile < n_tiles; ++tile) {
            load_tile(tile, 0);
            __syncthreads();
            maybe_init_kv_iters_for_tile(tile);
            apply_rope_to_tile(tile, 0);
            __syncthreads();
            process_tile(tile, 0);
            __syncthreads();
        }
    }

    if (warp_active) {
        float inv_l = __fdividef(1.f, fmaxf(l_i, 1e-10f));
        O* out_ptr = out + ((int64_t)slot_idx * (int64_t)n_q_head + (int64_t)head_idx) * (int64_t)HEAD_DIM;
        #pragma unroll
        for (int j = 0; j < VEC; ++j)
            out_ptr[lane * VEC + j] = from_f32<O>(out_reg[j] * inv_l);
    }
}

template <typename Q_T, typename T, typename O,
          int HEAD_DIM, int WARPS_PER_BLOCK, bool ROPE_INTERLEAVED,
          bool INT8_QK, bool INT8_PV, bool INT8_QK_USE_MMA>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE,
                                   v2_min_blocks_per_sm<WARPS_PER_BLOCK>())
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
    const float* rope_cs
) {
    constexpr bool IS_HALF_TYPE = std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>;
    constexpr int STAGES = IS_HALF_TYPE ? 3 : 2;
    int8_decode_attn_impl<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, 32, STAGES, true, ROPE_INTERLEAVED, INT8_QK, INT8_PV, INT8_QK_USE_MMA>(
        q, headers_ptr, out, num_active_slots, n_q_head, n_kv_head, softmax_scale,
        k_new, v_new, rope_cs);
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
    cudaStream_t stream = nullptr
) {
    int heads_per_group = (n_kv_head > 0) ? (n_q_head / n_kv_head) : 1;
    if (heads_per_group < 1) heads_per_group = 1;
    const bool use_wide = (HEAD_DIM >= 128) && (heads_per_group > 8);

    auto launch = [&](auto warps_const, auto rope_const, auto qk_const, auto pv_const, auto mma_const) {
        constexpr int WARPS_PER_BLOCK = decltype(warps_const)::value;
        constexpr bool ROPE_INTERLEAVED = decltype(rope_const)::value;
        constexpr bool INT8_QK = decltype(qk_const)::value;
        constexpr bool INT8_PV = decltype(pv_const)::value;
        constexpr bool INT8_QK_USE_MMA = decltype(mma_const)::value;
        dim3 grid(num_active_slots, n_kv_head);
        dim3 block(WARP_SIZE * WARPS_PER_BLOCK);
        int8_decode_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, ROPE_INTERLEAVED, INT8_QK, INT8_PV, INT8_QK_USE_MMA>
            <<<grid, block, 0, stream>>>(
                q, headers_ptr, out, num_active_slots, n_q_head, n_kv_head,
                softmax_scale, k_new, v_new, rope_cs);

        constexpr int COMMIT_THREADS = 128;
        dim3 commit_grid((num_active_slots + COMMIT_THREADS - 1) / COMMIT_THREADS);
        commit_decode_write_len_kernel<HEAD_DIM><<<commit_grid, COMMIT_THREADS, 0, stream>>>(
            headers_ptr, num_active_slots, n_kv_head);
    };

    // CANDLE_FUSED_ATTN_INT8 (bitmask, default 3 = full INT8 since the
    // gated test passes in that mode):
    //   bit 0 (1) = INT8 QK^T (else FP)
    //   bit 1 (2) = INT8 PV   (else FP)
    //   bit 2 (4) = use manual lane-collective dot for INT8 QK^T (else MMA)
    // Common values:
    //   0 → FP/FP                   (iter-1 baseline / fallback)
    //   1 → MMA QK + FP PV          (isolate INT8 QK^T)
    //   2 → FP QK + INT8 PV         (isolate INT8 PV)
    //   3 → MMA QK + INT8 PV        (default — full INT8)
    //   5 → manual QK + FP PV       (DEBUG)
    //   7 → manual QK + INT8 PV     (DEBUG)
    static int int8_mode = -1;
    if (int8_mode < 0) {
        const char* env = std::getenv("CANDLE_FUSED_ATTN_INT8");
        int8_mode = env ? std::atoi(env) : 3;
        if (int8_mode < 0 || int8_mode > 7) int8_mode = 3;
    }

    auto dispatch = [&](auto warps_c, auto rope_c) {
        const bool qk = (int8_mode & 1) != 0;
        const bool pv = (int8_mode & 2) != 0;
        const bool use_mma = (int8_mode & 4) == 0;
        if (!qk && !pv) {
            launch(warps_c, rope_c, std::false_type{}, std::false_type{}, std::true_type{});
        } else if (qk && !pv && use_mma) {
            launch(warps_c, rope_c, std::true_type{}, std::false_type{}, std::true_type{});
        } else if (!qk && pv) {
            launch(warps_c, rope_c, std::false_type{}, std::true_type{}, std::true_type{});
        } else if (qk && pv && use_mma) {
            launch(warps_c, rope_c, std::true_type{}, std::true_type{}, std::true_type{});
        } else if (qk && !pv && !use_mma) {
            launch(warps_c, rope_c, std::true_type{}, std::false_type{}, std::false_type{});
        } else if (qk && pv && !use_mma) {
            launch(warps_c, rope_c, std::true_type{}, std::true_type{}, std::false_type{});
        } else {
            launch(warps_c, rope_c, std::false_type{}, std::false_type{}, std::true_type{});
        }
    };

    // 4-way dispatch over (use_wide, rope_interleaved). use_wide selects
    // WARPS_PER_BLOCK=16 for heads_per_group > 8 (e.g. Llama-3 70B class).
    if (use_wide) {
        if (rope_interleaved) {
            dispatch(std::integral_constant<int, 16>{}, std::true_type{});
        } else {
            dispatch(std::integral_constant<int, 16>{}, std::false_type{});
        }
    } else {
        if (rope_interleaved) {
            dispatch(std::integral_constant<int, 8>{}, std::true_type{});
        } else {
            dispatch(std::integral_constant<int, 8>{}, std::false_type{});
        }
    }
}

} // namespace fused_attn
