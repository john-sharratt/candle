// The batched-M decode kernel on INT8 tensor cores with read-through V, and the
// per-token staging it uses.
//
// `int8_decode_bmma_impl` runs the batched-M decode with INT8 tensor-core MMA
// and read-through V. `bmma_stage_token` stages one token's K or V row into the
// destination in palette order.

// Stage one token's K (IS_K) or V row into `dst` (palette order), every
// palette through the accessor (cp.async for a same-type dtype palette).
// `within` is the token's slot in the chunk.
//
// A Q0_V palette decodes through the accessor too, whose read is the folded
// header's arithmetic (block_q0_v.cuh), so it is bit for bit what caching the
// palette's header per tile would give. That cache saved re-reading a 2-byte
// code per token and held four headers live across the token loop on every
// tile, Q0_V or not — in a loop at the register cap, which put them in local
// memory. A free function with explicit operands, not a lambda: a closure
// this size is outlined into a real call, its captures then living in local
// memory across the whole kernel body.
template <typename T, bool IS_K, int HEAD_DIM>
__device__ __forceinline__ void bmma_stage_token(T* dst, const uint8_t* head_ptr, int within, int lane)
{
    constexpr int SUB_HEAD_DIM = HEAD_DIM / N_PALETTE;
    constexpr int64_t sub_head_stride = (int64_t)SUB_HEAD_DIM * CHUNK_SIZE;
    constexpr int BLOCKS_PER_DIM = CHUNK_SIZE / 32;
    // A real loop: each palette's load sits behind its own format dispatch,
    // which its loads do not move above, so unrolling bought no overlap — only
    // four inlined copies of the accessor's every-format dispatch.
    #pragma unroll 1
    for (int p = 0; p < N_PALETTE; ++p) {
        const uint64_t ptr = IS_K ? kvhead_k_ptr<HEAD_DIM>(head_ptr, p) : kvhead_v_ptr<HEAD_DIM>(head_ptr, p);
        if (ptr == 0) continue;
        const int fmt = IS_K ? kvhead_k_fmt<HEAD_DIM>(head_ptr, p) : kvhead_v_fmt<HEAD_DIM>(head_ptr, p);
        const float scale = IS_K ? kvhead_k_scale<HEAD_DIM>(head_ptr, p) : kvhead_v_scale<HEAD_DIM>(head_ptr, p);
        ArenaAccessor acc((const char*)(uintptr_t)ptr, fmt, sub_head_stride, sub_head_stride, BLOCKS_PER_DIM, 0);
        acc.template load_head_scaled<T, SUB_HEAD_DIM, true, IS_K>(dst + p * SUB_HEAD_DIM, 0, 0, within, lane, scale);
    }
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
    const RopeRungs rungs,
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
    constexpr unsigned FULL_MASK = 0xffffffffu;

    float out_reg[HPG][VEC];
    #pragma unroll
    for (int h = 0; h < HPG; ++h) {
        #pragma unroll
        for (int j = 0; j < VEC; ++j) out_reg[h][j] = 0.f;
    }
    // Each head's flash state (running max `m`, running sum `l`) is
    // warp-uniform — every lane computes it from the same broadcast scores — so
    // it is held once, by lane `h`, and read with a shuffle where it is used,
    // instead of HPG copies in every lane. With the PV accumulators taking
    // HPG·VEC registers this is what keeps the tile loop inside the 4-block
    // register target without spilling. Same operations, same order.
    float m_lane = -1e38f, l_lane = 0.f;

    // One partial per block per head: the warps' flash states merge through
    // shared memory at the end (`stripe_block_merge_emit`), so the combine
    // reads `num_splits` partials per row, not `num_splits × warps`. Every
    // early-out returns through `emit_block` — the merge is block-wide.
    __shared__ alignas(128) float s_merge[WARPS_PER_BLOCK][HEAD_DIM];
    __shared__ float s_ml[WARPS_PER_BLOCK][2];
    auto emit_block = [&]() {
        float m_i[HPG], l_i[HPG];
        #pragma unroll
        for (int h = 0; h < HPG; ++h) {
            m_i[h] = __shfl_sync(FULL_MASK, m_lane, h);
            l_i[h] = __shfl_sync(FULL_MASK, l_lane, h);
        }
        stripe_block_merge_emit<HEAD_DIM, WARPS_PER_BLOCK, HPG, HPG>(
            out_reg, m_i, l_i, /*head_lo=*/0, s_merge, s_ml, partial_acc, partial_ml,
            slot_idx, kv_head_idx, split_idx, num_splits, n_q_head, warp, lane);
    };

    const SlotHeader& slot = get_slot_header(headers_ptr, slot_idx);
    const uint32_t n_slices = slot.n_slices;
    const uint32_t write_slice_idx = slot.write_slice;
    const uint64_t slices_ptr = slot.slices_ptr;
    const RopeView rope = rope_view(rungs, slot.rope_rung);
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
                    apply_rope_interleaved_f32<VEC, HEAD_DIM>(qr, lane, (int)q_rope_pos, rope.for_q());
                else
                    apply_rope_rotary_f32<VEC, HEAD_DIM>(qr, lane, (int)q_rope_pos, rope.for_q());
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

        // Token validity as one bitmask, and each token's position recomputed
        // where it is read (`TOK_WITHIN`) — eight positions and eight flags held
        // across the tile cost registers the PV accumulators need, and the
        // QK^T write indexed the flag array by a lane-dependent token, which put
        // it in local memory.
        uint32_t tok_mask = 0;
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            int within = within_base + t;
            bool valid = (within < (int)(off + bv));
            // QSA: a token this slot does not select is dead exactly as an
            // out-of-range one — its score becomes −inf below.
            if (valid && qsa_on) {
                valid = qsa_selects(sel, slot_idx, rope_base + (within - (int)off));
            }
            if (valid) tok_mask |= 1u << t;
        }
        // Pad lanes of the slice's last tile read a safe in-bounds slot and
        // are discarded (their mask bit is clear) below.
        #define TOK_VALID(t) (((tok_mask >> (t)) & 1u) != 0u)
        #define TOK_WITHIN(t) (TOK_VALID(t) ? within_base + (t) : (int)off)
        // ── stage the 8 tokens' K → shared_kb, cp.async double-buffered so each
        // token's load overlaps the previous token's gather/RoPE/quant. The
        // prefetch is unconditional when slice_ok (all 8 tokens share the chunk,
        // so every `within` is in-bounds); invalid tokens just zero shared_kb.
        //
        // A real loop over the 8 tokens, not unrolled: each iteration inlines
        // the accessor's per-format load dispatch for 4 palettes, and unrolling
        // pasted that 32 times here (and 32 more in the PV loop), stretching the
        // tile state's live ranges across all of it — which is what spilled at
        // the 64-register target. Every index below is runtime arithmetic
        // already (`t & 1`, the mask shift), and the cp.async double buffer is
        // what overlaps one token's load with the previous token's work.
        //
        // The loop starts one early: iteration −1 only issues token 0's load,
        // so the staging is ONE inlined dispatch rather than a prologue copy of
        // it beside the loop's. Every later iteration issues token t+1, then
        // waits for token t — the same commit/wait order as a prologue.
        #pragma unroll 1
        for (int t = -1; t < 8; ++t) {
            if (slice_ok && t + 1 < 8) {
                bmma_stage_token<T, true, HEAD_DIM>(skt[(t + 1) & 1][warp], head_ptr, TOK_WITHIN(t + 1), lane);
                cp_async_commit<true>();
            }
            if (t < 0) continue;
            if (slice_ok) {
                if (t + 1 < 8) cp_async_wait<1, true>();
                else           cp_async_wait<0, true>();
            }
            __syncwarp();
            if (!TOK_VALID(t)) {
                #pragma unroll
                for (int j = 0; j < VEC; ++j) shared_kb[warp][t][lane * VEC + j] = 0;
                if (lane < N_PALETTE) scaleK[warp][t][lane] = 1.f;
                continue;
            }
            float k_regs[VEC];
            #pragma unroll
            for (int j = 0; j < VEC; ++j) k_regs[j] = to_f32<T>(skt[t & 1][warp][ki[j]]);
            int32_t rope_pos = rope_base + (TOK_WITHIN(t) - (int)off);
            if constexpr (ROPE_INTERLEAVED && (VEC == 1 || VEC % 2 == 0))
                apply_rope_interleaved_f32<VEC, HEAD_DIM>(k_regs, lane, rope_pos, rope);
            else
                apply_rope_rotary_f32<VEC, HEAD_DIM>(k_regs, lane, rope_pos, rope);
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
            scores_smem[warp][my_m][tok0] = TOK_VALID(tok0) ? acc_lo : -1e38f;
            scores_smem[warp][my_m][tok1] = TOK_VALID(tok1) ? acc_hi : -1e38f;
        }
        __syncwarp();

        // ── softmax pass 1: per head, running-max + accumulator rescale. The
        // head's new max is installed in its owning lane here; pass 2 reads it
        // from there. ──
        #pragma unroll
        for (int h = 0; h < HPG; ++h) {
            float tile_max = -1e38f;
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                float s = scores_smem[warp][h][t];
                tile_max = fmaxf(tile_max, (s > -1e37f) ? s * softmax_scale : -1e38f);
            }
            const float mh = __shfl_sync(FULL_MASK, m_lane, h);
            float nm = fmaxf(mh, tile_max);
            float alpha = fast_exp::exp2<float, fast_exp::Softmax>(make_float2(mh - nm, 0.f)).x;
            // A select, not a lane-gated branch: the warp never diverges, so
            // nothing after this is issued per sub-group.
            const bool own = (lane == h);
            l_lane = own ? l_lane * alpha : l_lane;
            m_lane = own ? nm : m_lane;
            #pragma unroll
            for (int j = 0; j < VEC; ++j) out_reg[h][j] *= alpha;
        }

        // ── PV pass 2: load each token's V once (cp.async double-buffered into
        // the reused skt ring) so its load overlaps the previous token's
        // accumulate, and add it across all heads — no per-tile V smem staging.
        // Prefetch is unconditional when slice_ok (in-bounds); invalid tokens are
        // skipped in the accumulate. ──
        // A real loop starting one early, for the same reasons as the K stage
        // above.
        #pragma unroll 1
        for (int t = -1; t < 8; ++t) {
            if (slice_ok && t + 1 < 8) {
                bmma_stage_token<T, false, HEAD_DIM>(skt[(t + 1) & 1][warp], head_ptr, TOK_WITHIN(t + 1), lane);
                cp_async_commit<true>();
            }
            if (t < 0) continue;
            if (slice_ok) {
                if (t + 1 < 8) cp_async_wait<1, true>();
                else           cp_async_wait<0, true>();
            }
            __syncwarp();
            if (!TOK_VALID(t)) continue;
            float v_regs[VEC];
            #pragma unroll
            for (int j = 0; j < VEC; ++j) v_regs[j] = to_f32<T>(skt[t & 1][warp][vi[j]]);
            #pragma unroll
            for (int h = 0; h < HPG; ++h) {
                // Warp-uniform (a broadcast smem read), so the whole warp
                // skips together and the shuffle below stays converged.
                float s = scores_smem[warp][h][t];
                if (!(s > -1e37f)) continue;
                const float mh = __shfl_sync(FULL_MASK, m_lane, h);
                float beta = fast_exp::exp2<float, fast_exp::Softmax>(make_float2(s * softmax_scale - mh, 0.f)).x;
                l_lane = (lane == h) ? l_lane + beta : l_lane;
                #pragma unroll
                for (int j = 0; j < VEC; ++j) out_reg[h][j] = __fmaf_rn(beta, v_regs[j], out_reg[h][j]);
            }
            __syncwarp();
        }
        #undef TOK_WITHIN
        #undef TOK_VALID
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
    const T* k_new, const T* v_new, const RopeRungs rungs,
    float* partial_acc, float* partial_ml, QsaSel sel
) {
    int8_decode_bmma_impl<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, ROPE_INTERLEAVED, HPG>(
        q, headers_ptr, num_active_slots, n_q_head, n_kv_head, softmax_scale,
        k_new, v_new, rungs, partial_acc, partial_ml, sel);
}
