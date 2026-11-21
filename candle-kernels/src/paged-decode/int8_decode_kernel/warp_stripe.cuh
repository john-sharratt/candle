// The warp-stripe decode kernel, in which every warp computes and warps share K
// and V staging in pairs.
//
// `int8_decode_stripe_impl` walks the KV cells with warps working in pairs, and
// `stripe_process_cell` stages one cell per pair: the even warp stages K and
// the odd warp stages V. `stripe_pair_sync` rendezvous each pair, and
// `stripe_block_merge_emit` merges the block's partial states.

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
    RopeView rope;                       // the slot's rung, for K
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
    // Real loops over the palettes: each palette's load sits behind its own
    // format dispatch, which its loads do not move above, so unrolling bought
    // no overlap — only four inlined copies of the accessor's every-format
    // dispatch per side.
    if (c.half == 0) {
        #pragma unroll 1
        for (int p = 0; p < N_PALETTE; ++p) {
            uint64_t k_ptr_p = kvhead_k_ptr<HEAD_DIM>(head_ptr, p);
            if (k_ptr_p) {
                ArenaAccessor ka((const char*)(uintptr_t)k_ptr_p, kvhead_k_fmt<HEAD_DIM>(head_ptr, p),
                                 sub_head_stride, sub_head_stride, BLOCKS_PER_DIM, 0);
                ka.template load_head_scaled<T, SUB_HEAD_DIM, false, true>(
                    k_stage + p * SUB_HEAD_DIM, 0, 0, within, c.lane, kvhead_k_scale<HEAD_DIM>(head_ptr, p));
            }
        }
    } else {
        #pragma unroll 1
        for (int p = 0; p < N_PALETTE; ++p) {
            uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(head_ptr, p);
            if (v_ptr_p) {
                ArenaAccessor va((const char*)(uintptr_t)v_ptr_p, kvhead_v_fmt<HEAD_DIM>(head_ptr, p),
                                 sub_head_stride, sub_head_stride, BLOCKS_PER_DIM, 0);
                va.template load_head_scaled<T, SUB_HEAD_DIM, false, false>(
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
            apply_rope_interleaved_f32<VEC, HEAD_DIM>(k_regs, c.lane, rope_pos, c.rope);
        else
            apply_rope_rotary_f32<VEC, HEAD_DIM>(k_regs, c.lane, rope_pos, c.rope);
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
    const RopeRungs rungs,
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
    const uint32_t n_slices = slot.n_slices;
    const uint32_t write_slice_idx = slot.write_slice;
    const uint64_t slices_ptr = slot.slices_ptr;
    const RopeView rope = rope_view(rungs, slot.rope_rung);

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
            apply_rope_interleaved_f32<VEC, HEAD_DIM>(qr, lane, (int)q_rope_pos, rope.for_q());
        else
            apply_rope_rotary_f32<VEC, HEAD_DIM>(qr, lane, (int)q_rope_pos, rope.for_q());
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
        shared_q, sk[pair], sv[pair], rope, softmax_scale,
        kv_head_idx, pair, half, head_lo, lane };

    // Global stripe id: pair `pair` of split `split_idx`, out of `n_stripes`.
    const int stripe = split_idx * PAIRS + pair;
    const int n_stripes = num_splits * PAIRS;

    // ── Sparse walk: enumerate the selection's ENTRIES ──────────────────────
    //
    // A QSA row attends `top_k + ratio − 1` positions — 2051 for the released
    // checkpoint — however deep the cache is, so the walk's trip count must be
    // the selection's size and nothing in it may scale with `n_slices`.
    // Entries are striped round-robin over the stripes (`stripe +
    // n_stripes·i`), which balances them to within one entry.
    //
    // Each batch of up to 32 entries is resolved lane-parallel: lane `l` takes
    // one entry, binary-searches the slice holding its first cell (rope ranges
    // ascend and do not overlap, so `log2(n_slices)` probes — 12 at 128K — and
    // 32 such chains are in flight per warp instead of one), then walks the
    // entry's ≤4 consecutive cells with a forward cursor and packs each as
    // `(slice << 6) | within`, `CELL_NONE` for a cell no slice holds. A
    // selected position with no slice to hold it would be a selection built
    // against a different cache than the one being read; skipping is the only
    // safe answer — attending a slot outside the slice would read another
    // token's K/V.
    //
    // The pair then sweeps the batch's cells in entry order, each cell's code
    // shuffle-broadcast from the lane that resolved it, so the two warps of
    // the pair see an identical cell sequence (both resolve the same batch —
    // the search is cheap and duplicating it keeps the pair free of any
    // cross-warp hand-off beyond the K/V staging barrier). The sweep is a real
    // loop, not unrolled, and the cell's code is picked by a register select:
    // four inlined copies of the per-cell body (measured ~31K SASS instructions
    // each) left the register allocator spilling in the hot loop.

    // ── Dense walk: every visible token, striped over (split, pair) ─────────
    //
    // The token total is block-uniform; striding the slices across the block
    // and reducing through shared memory is the same integer sum (addition is
    // associative, every term non-negative) for 1/blockDim of the per-thread
    // work. Only the dense walk needs it — the sparse walk never touches the
    // total and stays O(selection) at any depth. `qsa_on` is per slot, so
    // block-uniform, and the barriers below are taken by the whole block or
    // by none of it.
    int k = 0, tok_hi = 0;
    if (!qsa_on) {
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
        tok_hi = (split_idx * tiles_per_split + tiles_per_split) * PAIRS;
        if (tok_hi > total_tok) tok_hi = total_tok;
        k = tok_lo + pair;
    }
    // Forward cursor: k is monotonic within a pair's strided iteration, so the
    // (slice, base) cursor only advances. `slice_eff_len` already accounts for
    // the writer's +1, so (k - scan_base) reaches the freshly scattered
    // token's slot.
    int scan_s = 0, scan_base = 0;

    // ── Sparse walk state: the selection's entries (see the note above) ──────
    constexpr int MAX_CELLS = 1 << QSA_CELL_BITS;
    constexpr uint32_t CELL_NONE = 0xFFFFFFFFu;
    const uint32_t* sel_entries = qsa_on ? sel.entries + (int64_t)slot_idx * sel.stride : nullptr;
    const int sel_cnt = qsa_on ? (int)sel.cnt[slot_idx] : 0;
    int first = stripe;    // the next batch's first entry
    int n_live = 0;        // the current batch's resolved lanes
    int src = 0;           // the lane whose entry is being swept
    int cell = 0;          // the cell of that entry
    uint32_t code[MAX_CELLS];
    #pragma unroll
    for (int c = 0; c < MAX_CELLS; ++c) code[c] = CELL_NONE;

    // ── One walk, one per-cell body ─────────────────────────────────────────
    //
    // Each iteration takes the next cell from whichever walk this slot runs —
    // a warp-uniform branch — and processes it at a single call site. The
    // per-cell body inlines the K/V arena loaders for four palettes across
    // every arena format; a call site per walk was two copies of all of it.
    int cell_no = 0;
    for (;;) {
        int slice_idx;
        int within;
        if (qsa_on) {
            uint32_t cd = CELL_NONE;
            for (;;) {
                if (src >= n_live) {
                    if (first >= sel_cnt) break;
                    // Lanes holding a live entry: `l < ceil((sel_cnt - first) / n_stripes)`.
                    n_live = min(WARP_SIZE, (sel_cnt - first + n_stripes - 1) / n_stripes);
                    #pragma unroll
                    for (int c = 0; c < MAX_CELLS; ++c) code[c] = CELL_NONE;
                    const int e = first + lane * n_stripes;
                    if (lane < n_live) {
                        const uint32_t ent = sel_entries[e];
                        // Start and width through the page layout: a projected
                        // prefix is pages whose last blocks are short (see
                        // `qsa_block_width_from`).
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
                                // Forward cursor: a run of consecutive positions
                                // can cross into the next slice(s).
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
                    first += n_stripes * WARP_SIZE;
                    src = 0;
                    cell = 0;
                }
                uint32_t mine = code[0];
                #pragma unroll
                for (int c = 1; c < MAX_CELLS; ++c) mine = (cell == c) ? code[c] : mine;
                const uint32_t got = __shfl_sync(0xffffffffu, mine, src);
                if (++cell == MAX_CELLS) { cell = 0; ++src; }
                if (got != CELL_NONE) { cd = got; break; }
            }
            if (cd == CELL_NONE) break;
            slice_idx = (int)(cd >> 6);
            within = (int)(cd & 63u);
        } else {
            if (k >= tok_hi) break;
            while (scan_s + 1 < (int)n_slices) {
                int e = slice_eff_len(scan_s);
                if (scan_base + e <= k) { scan_base += e; ++scan_s; }
                else break;
            }
            slice_idx = scan_s;
            within = (int)slice_offset(get_slice<HEAD_DIM>(slices_ptr, scan_s, n_kv_head)) + (k - scan_base);
            k += PAIRS;
        }
        const uint8_t* sl = get_slice<HEAD_DIM>(slices_ptr, slice_idx, n_kv_head);
        stripe_process_cell<T, HEAD_DIM, ROPE_INTERLEAVED, HPG, WARP_HEADS>(
            ctx, st, out_reg, m_i, l_i, slice_idx, sl, within, cell_no++);
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
    const RopeRungs rungs,
    float* partial_acc,
    float* partial_ml,
    QsaSel sel
) {
    int8_decode_stripe_impl<Q_T, T, O, HEAD_DIM, WARPS_PER_BLOCK, ROPE_INTERLEAVED, HPG>(
        q, headers_ptr, num_active_slots, n_q_head, n_kv_head, softmax_scale,
        k_new, v_new, rungs, partial_acc, partial_ml, sel);
}
