// The split-KV INT8 decode attention kernel with its split-count ceiling and
// register target.
//
// `int8_decode_attn_impl` computes QK^T with the m16n8k32 INT8 MMA when a
// palette spans exactly 32 entries. `MAX_SPLITS` ceils the split-KV factor, and
// `int8_decode_min_blocks` sets the register target for the kernel.

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
    const RopeRungs rungs,
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
    const uint32_t n_slices  = slot.n_slices;
    const uint32_t write_slice_idx = slot.write_slice;
    const uint64_t slices_ptr = slot.slices_ptr;
    // This slot's rung, from its own header: nothing rung-dependent is shared
    // with another slot's CTA.
    const RopeView rope = rope_view(rungs, slot.rope_rung);

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
            apply_rope_interleaved_f32<VEC, HEAD_DIM>(q_reg, lane, (int)q_rope_pos, rope.for_q());
        } else {
            apply_rope_rotary_f32<VEC, HEAD_DIM>(q_reg, lane, (int)q_rope_pos, rope.for_q());
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
        // passthrough int8 family whose scale is the same for every token of a
        // block. The warps of a tile stage different tokens but share one
        // per-dim scale row (`shared_v_dim_scale[stage]`), so a format whose
        // scale depends on the token — the sink-protected Q4_KS/Q8_KS, tokens
        // 0–3 on their own fine scale — would have the warps overwrite each
        // other's scales; it takes the dequant→requant path, as it does in the
        // tile kernel. Mixed/asymmetric/FP formats keep that path too. K never
        // skips — RoPE needs FP.
        bool v_readthrough = true;
        #pragma unroll
        for (int p = 0; p < N_PALETTE; ++p) {
            int vf = kvhead_v_fmt<HEAD_DIM>(head_ptr, p);
            if (!int8_elem::i8_tile_readthrough_format(vf)) v_readthrough = false;
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
            k_acc.template load_head_scaled<T, SUB_HEAD_DIM, USE_TC, true>(k_dst + p * SUB_HEAD_DIM, 0, 0, within, lane, k_scale_p);
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
                v_acc.template load_head_scaled<T, SUB_HEAD_DIM, USE_TC, false>(v_dst + p * SUB_HEAD_DIM, 0, 0, within, lane, v_scale_p);
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
                    apply_rope_interleaved_f32<VEC, HEAD_DIM>(k_regs, lane, rope_pos, rope);
                } else {
                    apply_rope_rotary_f32<VEC, HEAD_DIM>(k_regs, lane, rope_pos, rope);
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
                // One m16n8k32 scores 8 tokens (its N), so a tile of
                // WARPS_PER_BLOCK tokens takes WARPS_PER_BLOCK / 8 groups. The
                // tile is 16 tokens wide at 16 warps; scoring only the first 8
                // would leave the rest of tile_logits[] as whatever the block's
                // shared memory last held.
                static_assert(WARPS_PER_BLOCK % 8 == 0, "the MMA scores tokens 8 at a time");
                #pragma unroll
                for (int tg = 0; tg < WARPS_PER_BLOCK; tg += 8) {
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
                            const int8_t* k_base_p = &shared_k_int8[stage][tg + (lane >> 2)][p * SUB_HEAD_DIM + (lane & 3) * 4];
                            b_frag[0] = *reinterpret_cast<const uint32_t*>(k_base_p);
                            b_frag[1] = *reinterpret_cast<const uint32_t*>(k_base_p + 16);
                        }

                        int32_t c_p[4] = {0, 0, 0, 0};
                        mma_int8_m16n8k32(c_p, a_frag, b_frag, c_p);

                        if ((lane >> 2) == 0) {
                            int tok0 = tg + (lane & 3) * 2;
                            int tok1 = tok0 + 1;
                            float s_q = scale_Q[p];
                            float s_k0 = shared_k_scale[stage][tok0][p];
                            float s_k1 = shared_k_scale[stage][tok1][p];
                            acc_lo += (float)c_p[0] * s_q * s_k0;
                            acc_hi += (float)c_p[1] * s_q * s_k1;
                        }
                    }

                    if ((lane >> 2) == 0 && (lane & 3) < 4) {
                        int t0 = tg + (lane & 3) * 2;
                        tile_logits[stage][warp][t0]     = acc_lo;
                        tile_logits[stage][warp][t0 + 1] = acc_hi;
                    }
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
        // The prologue fills up to NUM_STAGES stages, tile i into stage i, in
        // order — a real loop, so `load_tile` (every palette's K and V across
        // every arena format) is one inlined copy here rather than one per
        // stage.
        int tiles_loaded = 0;
        #pragma unroll 1
        for (int i = 0; i < NUM_STAGES && i < range; ++i) {
            load_tile(tile_lo + i, i);
            cp_async_commit<USE_TC>();
            tiles_loaded = i + 1;
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
    const RopeRungs rungs,
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
        k_new, v_new, rungs, partial_acc, partial_ml, sel);
}
