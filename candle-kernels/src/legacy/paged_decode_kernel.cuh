#pragma once
// ============================================================================
// Paged Decode Attention V2 (legacy).
//   The original persistent-slot-buffer FP decode kernel. Superseded as the
//   default by the INT8 decode kernel; retained for head_dim=256 and as the
//   A/B regression reference (run_paged_decode_legacy_*).
// ============================================================================

#include "../paged-decode/decode_helpers.cuh"

template <typename Q_T, typename T, typename O, int HEAD_DIM, int WARPS_PER_BLOCK, int TILE_K = 32, int NUM_STAGES = 2, bool USE_TC = false, bool ROPE_INTERLEAVED = false>
__device__ __forceinline__ void paged_decode_attn_v2_impl(
    const Q_T* __restrict__ q,           // [num_active_slots, n_q_head, HEAD_DIM]
    const uint8_t* __restrict__ headers_ptr,  // [num_active_slots] × 16-byte SlotHeader
    O* __restrict__ out,                 // [num_active_slots, n_q_head, HEAD_DIM]
    int num_active_slots,
    int n_q_head,    // total Q heads; one warp per Q head within the CTA
    int n_kv_head,   // KV heads — one CTA column per KV head; n_q_head / n_kv_head = GQA group size
    float softmax_scale,
    const T* __restrict__ k_new,         // [num_active_slots, n_kv_head, HEAD_DIM]
    const T* __restrict__ v_new,         // [num_active_slots, n_kv_head, HEAD_DIM]
    const float* __restrict__ rope_cs    // cos/sin table [max_pos * HEAD_DIM]
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

    // =========================================================================
    // Read slot header for this batch index
    // =========================================================================
    const SlotHeader& slot = get_slot_header(headers_ptr, slot_idx);
    const uint32_t n_slices  = slot.n_slices;
    const uint32_t write_slice_idx = slot.write_slice;
    const uint64_t slices_ptr = slot.slices_ptr;

    if (n_slices == 0) {
        // No slices yet — write zeros and return
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

    // Mutable write slice pointer (used by scatter and by the post-attention len increment).
    uint8_t* write_slice_ptr = get_slice_mut<HEAD_DIM>(slices_ptr, (int)write_slice_idx, n_kv_head);
    const uint16_t ws_offset = slice_offset(write_slice_ptr);
    const uint16_t ws_len    = slice_len(write_slice_ptr);    // tokens in write slice BEFORE this step
    const uint32_t ws_rope   = slice_rope(write_slice_ptr);

    // Host contract: decode must rotate to a fresh write slice before the next
    // write position reaches the end of the chunk. The kernel relies on one
    // free slot to scatter the current token and expose it via the "+1"
    // self-attention rule below.
#ifndef NDEBUG
    if (warp == 0 && lane == 0) {
        assert((int)ws_offset >= 0 && (int)ws_offset <= CHUNK_SIZE);
        assert((int)ws_len >= 0 && (int)ws_len <= CHUNK_SIZE);
        assert((int)ws_offset + (int)ws_len < CHUNK_SIZE);
    }
#endif

    // =========================================================================
    // Fused KV scatter (warp 0 only)
    // =========================================================================
    {
        // Write position in the write slice chunk: physical position = ws_offset + ws_len
        const int within = (int)ws_offset + (int)ws_len;

        constexpr int LANES_PER_PAL = WARP_SIZE / N_PALETTE;  // = 8
        constexpr int SUB_HEAD_DIM  = HEAD_DIM / N_PALETTE;

        if (warp == 0 && within < CHUNK_SIZE) {
            const uint8_t* head_ptr = get_head<HEAD_DIM>(write_slice_ptr, kv_head_idx);

            // Identity palette routing: lane -> palette and local_lane.
            // Write chunks are always uniform-format (F16 or R16) with identity
            // pal_map, so no pal_map indirection is needed here.  Non-identity
            // pal_maps only appear on sealed/reconciled read-only chunks, which
            // are never scatter targets.
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
                    // R16: Q-capture scatter. k_ptr is chunk start, chunk_byte_offset=0.
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
                    // Float format: elem offset from start of chunk = within * SUB_HEAD_DIM
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
        __syncthreads();  // Ensure scatter visible before attention reads
    }

    // =========================================================================
    // GQA setup and kv_len
    // =========================================================================
    int heads_per_group = n_q_head / n_kv_head;
    if (heads_per_group <= 0) heads_per_group = 1;
    int head_idx = kv_head_idx * heads_per_group + warp;
    bool warp_active = (warp < heads_per_group) && (head_idx < n_q_head);

    // kv_len after scatter = all prior tokens + tokens in write slice before this step + 1 (new token)
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

    // Clamp kv_len to max_slices * CHUNK_SIZE
    int max_len = (int)n_slices * CHUNK_SIZE;
    if (kv_len > max_len) kv_len = max_len;

    // =========================================================================
    // Load Q and apply RoPE
    // =========================================================================
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
        // Q RoPE position = absolute position of the new token (0-indexed)
        uint32_t q_rope_pos = (uint32_t)ws_rope + (uint32_t)ws_len;  // = kv_len - 1
        if constexpr (ROPE_INTERLEAVED && (VEC == 1 || VEC % 2 == 0)) {
            apply_rope_interleaved_f32<VEC, HEAD_DIM>(q_reg, lane, (int)q_rope_pos, rope_cs);
        } else {
            apply_rope_rotary_f32<VEC, HEAD_DIM>(q_reg, lane, (int)q_rope_pos, rope_cs);
        }
    }

    // K and V palette routing can vary across read slices.  Under cum-token
    // addressing a single tile of `WARPS_PER_BLOCK` K positions may span
    // multiple slices when the slot contains partial-tail slices smaller
    // than the tile width, so we cannot cache palette iterators per tile.
    // Each warp resolves its own slice and initializes its own `ki`/`vi`
    // inside `apply_rope_to_tile`, with a per-warp cache to skip re-init
    // when the same slice gets reused across tiles.
    PalIter<VEC, HEAD_DIM> ki, vi;
    int kv_pal_slice_idx = -1;

    // Bank-conflict padding for the K/V staging rings. HD=256 with 16 warps and
    // a 3-stage half ring is 50688 B with pad=8 — 1536 B (exactly the padding)
    // over the 48 KB static-smem ceiling. Drop the pad for HD=256 only: it lands
    // at exactly 49152 B, and that head_dim isn't used by our models (Qwen3 /
    // Llama are 128), so its bank-conflict avoidance costs us nothing. The
    // smaller, actually-used head dims keep their padding.
    constexpr int SMEM_PAD = (HEAD_DIM >= 256) ? 0 : 8;
    static_assert((HEAD_DIM + SMEM_PAD) * sizeof(T) % 16 == 0,
                  "SMEM_PAD breaks 16-byte alignment required by cp.async.cg");
    __shared__ alignas(128) T shared_k[NUM_STAGES][WARPS_PER_BLOCK][HEAD_DIM + SMEM_PAD];
    __shared__ alignas(128) T shared_v[NUM_STAGES][WARPS_PER_BLOCK][HEAD_DIM + SMEM_PAD];

    constexpr int SUB_HEAD_DIM = HEAD_DIM / N_PALETTE;

    float m_i = -1e38f;
    float l_i = 0.f;
    float out_reg[VEC];
    #pragma unroll
    for (int j = 0; j < VEC; ++j) out_reg[j] = 0.f;

    const int n_tiles = (kv_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    constexpr int BLOCKS_PER_DIM = CHUNK_SIZE / 32;

    // =========================================================================
    // load_tile lambda — uses pre-resolved k_ptr[p] per slice
    // =========================================================================
    auto load_tile = [&](int tile_idx, int stage) {
        int k_base = tile_idx * WARPS_PER_BLOCK;
        int k_idx = k_base + warp;
        bool valid_k = (k_idx < kv_len);

        T* k_dst = shared_k[stage][warp];
        T* v_dst = shared_v[stage][warp];

        int my_slice_idx = 0, within = 0;
        if (valid_k) {
            resolve_pos(slot, k_idx, my_slice_idx, within);
        }

        // Validate against slice usage window
        if (valid_k && my_slice_idx < (int)n_slices) {
            const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, my_slice_idx, n_kv_head);
            uint32_t bv  = (uint32_t)slice_len(sl);
            uint32_t off = (uint32_t)slice_offset(sl);
            // The current step has already scattered k_new/v_new into the write slice,
            // but ws.len is only committed after the attention pass completes. Treat the
            // write slice as having one additional valid token for this invocation so
            // the current token can attend to itself.
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
            return;
        }

        constexpr int64_t sub_head_stride = (int64_t)SUB_HEAD_DIM * CHUNK_SIZE;

        const uint8_t* sl_ptr = get_slice<HEAD_DIM>(slices_ptr, my_slice_idx, n_kv_head);
        const uint8_t* head_ptr = get_head<HEAD_DIM>(sl_ptr, kv_head_idx);

        // Load all 4 palettes directly into shared_k/v in palette-contiguous order.
        // Scale is folded into load_head_scaled (applied in F32 before narrowing to T).
        // K and V are normalized into logical-dim order once per tile before
        // entering the hot accumulation loop.
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

    // =========================================================================
    // apply_rope_to_tile lambda — uses slice.rope and slice.offset
    // =========================================================================
    auto apply_rope_to_tile = [&](int tile_idx, int stage) {
        int k_base = tile_idx * WARPS_PER_BLOCK;
        int k_idx = k_base + warp;
        if (k_idx < kv_len) {
            int my_slice_idx, within;
            resolve_pos(slot, k_idx, my_slice_idx, within);
            const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, my_slice_idx, n_kv_head);
            uint32_t bv  = (uint32_t)slice_len(sl);
            uint32_t off = (uint32_t)slice_offset(sl);
            if (my_slice_idx == (int)write_slice_idx &&
                bv < CHUNK_SIZE && off + bv < CHUNK_SIZE) {
                bv += 1;
            }
            if (within >= (int)off && within < (int)(off + bv)) {
                // Refresh K/V palette iterators when this warp's slice
                // differs from the previously-cached one.  Each warp
                // caches its own `kv_pal_slice_idx`, so within a warp
                // adjacent same-slice tiles skip the re-init; across
                // tiles that straddle a slice boundary each warp
                // independently picks up the right palette map for
                // the slice it's actually reading.
                if (my_slice_idx != kv_pal_slice_idx) {
                    const uint8_t* head_ptr =
                        get_head<HEAD_DIM>(sl, kv_head_idx);
                    ki.init(kvhead_k_pal_map<HEAD_DIM>(head_ptr), lane);
                    vi.init(kvhead_v_pal_map<HEAD_DIM>(head_ptr), lane);
                    kv_pal_slice_idx = my_slice_idx;
                }
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

                // Normalize V into logical-dim order once per tile so the hot
                // accumulation loop below can use contiguous loads.
                T* v_dst = shared_v[stage][warp];
                #pragma unroll
                for (int j = 0; j < VEC; ++j)
                    k_regs[j] = to_f32<T>(v_dst[vi[j]]);
                __syncwarp();
                #pragma unroll
                for (int j = 0; j < VEC; ++j)
                    v_dst[lane * VEC + j] = from_f32<T>(k_regs[j]);
            }
        }
    };

    // =========================================================================
    // process_tile lambda — K·Q dot + softmax + V accumulation
    //
    // Under cum-token addressing a single tile of `WARPS_PER_BLOCK` K
    // positions may straddle two or more slices when the slot contains
    // partial-tail slices smaller than the tile width.  The per-tile
    // `tile_slice` / `tile_off` / `tile_bv` caching used by the old
    // positional-addressing kernel would tag every position in such a
    // tile with the first slice's offset/len, masking out the latter
    // positions even though they belong to a different (valid) slice.
    // To keep the kernel correct we now resolve each `t` independently
    // via `position_map` — one extra LUT load per position in the hot
    // loop, but it preserves the load_tile/apply_rope_to_tile data
    // that's already correctly per-warp.
    // =========================================================================
    auto process_tile = [&](int tile_idx, int stage) {
        int k_base = tile_idx * WARPS_PER_BLOCK;

        if (warp_active) {
            constexpr int TILE_UNROLL = (WARPS_PER_BLOCK <= 8) ? 4 : 2;
            #pragma unroll TILE_UNROLL
            for (int t = 0; t < WARPS_PER_BLOCK; ++t) {
                int actual_k = k_base + t;
                float valid_mask = 0.f;
                if (actual_k < kv_len) {
                    int t_slice, t_within;
                    resolve_pos(slot, actual_k, t_slice, t_within);
                    if (t_slice < (int)n_slices) {
                        const uint8_t* sl =
                            get_slice<HEAD_DIM>(slices_ptr, t_slice, n_kv_head);
                        uint32_t bv  = (uint32_t)slice_len(sl);
                        uint32_t off = (uint32_t)slice_offset(sl);
                        // Write-slice bump: scatter has placed the
                        // current step's K/V at (write_slice,
                        // off+bv) but ws.len isn't committed until
                        // the post-attention kernel.  Treat that one
                        // position as valid for this invocation so
                        // the current token attends to itself.
                        if (t_slice == (int)write_slice_idx &&
                            bv < CHUNK_SIZE && off + bv < CHUNK_SIZE) {
                            bv += 1;
                        }
                        if (t_within >= (int)off && t_within < (int)(off + bv)) {
                            valid_mask = 1.f;
                        }
                    }
                }

                // K dot: after apply_rope_to_tile(), shared_k is in logical dim
                // order, so Q and K can be consumed directly without remapping.
                float dot = 0.f;
                #pragma unroll
                for (int j = 0; j < VEC; ++j)
                    dot = __fmaf_rn(q_reg[j], to_f32<T>(shared_k[stage][t][lane * VEC + j]), dot);
                dot = warp_reduce_sum(dot);
                float score = dot * softmax_scale;
                float masked_score = (valid_mask > 0.f) ? score : -1e38f;
                float new_m = fmaxf(m_i, masked_score);
                // Compute both exp terms in one vectorized instruction.
                float2 exp_ab = fast_exp::exp2<float, fast_exp::Softmax>(
                    make_float2(m_i - new_m, masked_score - new_m));
                float alpha = exp_ab.x;
                float beta = valid_mask * exp_ab.y;
                l_i = __fmaf_rn(l_i, alpha, beta);

                // V accumulation: shared_v has already been normalized into
                // logical dim order for this tile, so out_reg[j] stays logical.
                if constexpr ((std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>) && VEC >= 2) {
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

    // =========================================================================
    // Multi-stage pipelined main loop (identical structure to v1)
    // =========================================================================
    if constexpr (NUM_STAGES >= 2 && USE_TC) {
        int tiles_loaded = 0;
        if (n_tiles > 0) {
            load_tile(0, 0);
            cp_async_commit<USE_TC>();
            tiles_loaded = 1;
        }
        if (n_tiles > 1 && NUM_STAGES >= 2) {
            load_tile(1, 1);
            cp_async_commit<USE_TC>();
            tiles_loaded = 2;
        }
        if constexpr (NUM_STAGES >= 3) {
            if (n_tiles > 2) {
                load_tile(2, 2);
                cp_async_commit<USE_TC>();
                tiles_loaded = 3;
            }
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
            // Palette iterators are refreshed per-warp inside
            // `apply_rope_to_tile` when this warp's slice differs
            // from the cached one — no per-tile init needed.
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
                apply_rope_to_tile(next_tile, (cur_stage + 1) % NUM_STAGES);
            }
            cur_stage = (cur_stage + 1) % NUM_STAGES;
        }
    } else {
        // Non-pipelined path: palette iterators are refreshed per
        // warp inside `apply_rope_to_tile` so cross-slice tiles
        // get the right palette map per position.
        for (int tile = 0; tile < n_tiles; ++tile) {
            load_tile(tile, 0);
            __syncthreads();
            apply_rope_to_tile(tile, 0);
            __syncthreads();
            process_tile(tile, 0);
            __syncthreads();
        }
    }

    // =========================================================================
    // Write output
    // =========================================================================
    if (warp_active) {
        float inv_l = __fdividef(1.f, fmaxf(l_i, 1e-10f));
        O* out_ptr = out + ((int64_t)slot_idx * (int64_t)n_q_head + (int64_t)head_idx) * (int64_t)HEAD_DIM;
        // out_reg is already in logical dim order.
        #pragma unroll
        for (int j = 0; j < VEC; ++j)
            out_ptr[lane * VEC + j] = from_f32<O>(out_reg[j] * inv_l);
    }

    // =========================================================================
    // ws.len commit happens in a tiny follow-up kernel on the same stream.
    // That preserves the GPU-resident decode fast path while ensuring this
    // attention kernel sees a stable per-launch snapshot of the write slice.
    // =========================================================================
}


template <int WARPS_PER_BLOCK>
constexpr int v2_min_blocks_per_sm() {
    return (WARPS_PER_BLOCK <= 8) ? 3 : 2;
}

// ============================================================================
// Per-head-dim kernel launchers (launched from v2 API files)
// ============================================================================

template <typename Q_T, typename T, typename O, int HEAD_DIM, int WARPS_PER_BLOCK, bool ROPE_INTERLEAVED>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * WARP_SIZE, v2_min_blocks_per_sm<WARPS_PER_BLOCK>())
paged_decode_v2_kernel(
    const Q_T* q,
    const uint8_t* headers_ptr,
    O* out,
    int num_active_slots,
    int n_q_head,    // total Q heads
    int n_kv_head,   // KV heads (grid columns); n_q_head / n_kv_head = GQA group size
    float softmax_scale,
    const T* k_new,
    const T* v_new,
    const float* rope_cs
) {
    // The v2 grid is (num_active_slots, n_kv_head).
    // Enable cp.async pipelining (legacy helper flag name USE_TC=true;
    // this path does not issue tensor-core MMA instructions).
    // Use 3 stages for half-precision types (26 KB smem @ HD=128, fits in 48 KB);
    // keep 2 stages for float32 (34 KB @ HD=128, would exceed 48 KB with 3 stages).
    constexpr bool IS_HALF_TYPE = std::is_same_v<T, __half> || std::is_same_v<T, __nv_bfloat16>;
    constexpr int STAGES = IS_HALF_TYPE ? 3 : 2;
    paged_decode_attn_v2_impl<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, 32, STAGES, true, ROPE_INTERLEAVED>(
        q, headers_ptr, out, num_active_slots, n_q_head, n_kv_head, softmax_scale,
        k_new, v_new, rope_cs
    );
}

// Launcher: dispatches grid and block dimensions then calls the kernel.
// WARPS_PER_BLOCK selection:
//   - HEAD_DIM <  128: always 8 warps.
//   - HEAD_DIM >= 128 and heads_per_group <= 8: use 8 warps so every warp does
//     attention work (with WARPS=16 and GQA ratio 8, half the warps are idle
//     during process_tile, wasting registers and occupancy). This is the
//     common case (Llama-3, Qwen3-MoE, Mistral, etc).
//   - HEAD_DIM >= 128 and heads_per_group >  8: use 16 warps for more tiles
//     per pass and better arithmetic intensity (rare configs only).
template <typename Q_T, typename T, typename O, int HEAD_DIM>
void launch_paged_decode_attn(
    const Q_T* q,
    const uint8_t* headers_ptr,
    O* out,
    int num_active_slots,
    int n_q_head,    // total Q heads
    int n_kv_head,   // KV heads (grid columns); n_q_head / n_kv_head = GQA group size
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

    auto launch = [&](auto warps_const, auto rope_const) {
        constexpr int WARPS_PER_BLOCK = decltype(warps_const)::value;
        constexpr bool ROPE_INTERLEAVED = decltype(rope_const)::value;
        dim3 grid(num_active_slots, n_kv_head);
        dim3 block(WARP_SIZE * WARPS_PER_BLOCK);
        paged_decode_v2_kernel<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, ROPE_INTERLEAVED><<<grid, block, 0, stream>>>(
            q, headers_ptr, out, num_active_slots, n_q_head, n_kv_head,
            softmax_scale, k_new, v_new, rope_cs
        );

        constexpr int COMMIT_THREADS = 128;
        dim3 commit_grid((num_active_slots + COMMIT_THREADS - 1) / COMMIT_THREADS);
        commit_decode_write_len_kernel<HEAD_DIM><<<commit_grid, COMMIT_THREADS, 0, stream>>>(
            headers_ptr, num_active_slots, n_kv_head
        );
    };

    if (use_wide) {
        if (rope_interleaved) launch(std::integral_constant<int, 16>{}, std::true_type{});
        else                  launch(std::integral_constant<int, 16>{}, std::false_type{});
    } else {
        if (rope_interleaved) launch(std::integral_constant<int, 8>{}, std::true_type{});
        else                  launch(std::integral_constant<int, 8>{}, std::false_type{});
    }
}
