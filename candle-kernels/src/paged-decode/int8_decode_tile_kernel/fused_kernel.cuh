// The INT8 tile kernel that runs the softmax warps and V warps over each
// compacted 32-token tile.
//
// Holds the `__launch_bounds__` kernel of the `fused_attn` namespace, which
// runs the softmax warps and the V warps over each 32-token INT8 tile.
// It is one 85 KB function, so it stays whole.

// ============================================================================
// The kernel
// ============================================================================
template <typename Q_T, typename T, int HEAD_DIM, bool ROPE_INTERLEAVED>
__global__ void __launch_bounds__(TILE_THREADS, TILE_MIN_BLOCKS)
int8_decode_tile_kernel(
    const Q_T* __restrict__ q,                 // [slots, n_q_head, HD], unrotated
    const uint8_t* __restrict__ headers_ptr,
    int num_active_slots,
    int n_q_head,
    int n_kv_head,
    float softmax_scale,
    const T* __restrict__ k_new,               // [slots, n_kv_head, HD], unrotated
    const T* __restrict__ v_new,
    const RopeRungs rungs,
    float* __restrict__ partial_acc,           // [slot·n_q_head + qh][split][HD]
    float* __restrict__ partial_ml,            // [slot·n_q_head + qh][split][2]
    QsaSel sel
) {
    static_assert(HEAD_DIM % 64 == 0 && HEAD_DIM >= 64 && HEAD_DIM <= 256,
                  "int8 tile decode: HEAD_DIM must be a multiple of 64 in [64, 256]");
    constexpr int N_WIN = HEAD_DIM / 32;        // QK k-step windows (also dims/lane)
    constexpr int SUB = HEAD_DIM / N_PALETTE;   // palette band width
    constexpr int VEC = HEAD_DIM / WARP_SIZE;   // new-token scatter dims per lane
    constexpr int PV_H = HEAD_DIM / 8 / TILE_WARPS; // output n-slices per warp
    constexpr int V_WPW = (N_WIN + TILE_V_WARPS - 1) / TILE_V_WARPS; // V windows per V warp
    constexpr int PAL_U32 = HEAD_DIM / 16;      // u32 words per palette map (16 dims each)
    static_assert(N_PALETTE == 4, "rank-table byte packs the palette into 2 bits");
    static_assert(SUB >= 16 && SUB <= 64, "rank needs 6 bits");
    static_assert(TILE_SM_WARPS * 8 == TILE_TOK && PV_H >= 1,
                  "the softmax warps split QK by n-slice; 8 warps split PV by dim");
    static_assert(TILE_GROUPS == TILE_WARPS && PAL_U32 <= 16,
                  "the map-word role is warp = group, lane = (side, word)");
    static_assert(TILE_GROUPS * 2 * N_PALETTE <= TILE_THREADS, "one stager per (group, side, palette)");
    static_assert(HEAD_DIM == 8 * WARP_SIZE,
                  "K decode: warp = group, lane = quad — a lane's 8 dims × 32 lanes cover the head");
    static_assert(HEAD_DIM / TILE_WARPS == 32 && TILE_TOK == 4 * 8,
                  "V decode: warp = 32-dim window, lane = (token quarter, quad)");
    constexpr int EPS = INT8_TILE_ENTRIES_PER_SPLIT;
    static_assert(EPS <= WARP_SIZE, "one lane resolves one entry");
    constexpr int MAX_GROUPS = EPS * GROUP_TOK;
    constexpr int N_STAGERS = TILE_GROUPS * 2 * N_PALETTE;

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int slot_idx = (int)blockIdx.x;
    const int kv_head_idx = (int)blockIdx.y;
    const int split_idx = (int)blockIdx.z;
    const int num_splits = (int)gridDim.z;
    if (slot_idx >= num_active_slots || kv_head_idx >= n_kv_head) return;
    int hpg = n_q_head / n_kv_head;
    if (hpg < 1) hpg = 1;
    if (hpg > TILE_M_ROWS) return;              // the launcher never routes this
    const int first_q_head = kv_head_idx * hpg;

    // ------------------------------------------------------------------
    // Shared memory: one arena holding the per-tile decode→compute handoff
    // (int8 K/V tiles + scales, the softmax's row posts, the per-group
    // palette rank tables) and the int8 Q slab, staged once in the
    // prologue and re-read as MMA fragments every tile so the registers it
    // would pin are free during the decode. The four MMA slabs are
    // unpadded and XOR-swizzled (`sw_off`): K and Q rows are HEAD_DIM
    // bytes, the transposed V and the P rows TILE_TOK bytes.
    // ------------------------------------------------------------------
    constexpr int TBL_BYTES = TILE_SLOTS * 2 * HEAD_DIM;

    constexpr int ALIGN16 = 15;
    constexpr int OFF_K8 = 0;
    constexpr int OFF_KS = (OFF_K8 + TILE_TOK * HEAD_DIM + ALIGN16) & ~ALIGN16;
    constexpr int OFF_V8T = (OFF_KS + TILE_TOK * N_WIN * 2 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_VS = (OFF_V8T + HEAD_DIM * TILE_TOK + ALIGN16) & ~ALIGN16;
    constexpr int OFF_P8 = (OFF_VS + HEAD_DIM * 2 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_RMAX = (OFF_P8 + TILE_M_ROWS * TILE_TOK + ALIGN16) & ~ALIGN16;
    constexpr int OFF_LSUM = (OFF_RMAX + TILE_M_ROWS * TILE_SM_WARPS * 4 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_ALPHA = (OFF_LSUM + TILE_M_ROWS * TILE_SM_WARPS * 4 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_TBL = (OFF_ALPHA + TILE_M_ROWS * 4 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_INV = (OFF_TBL + TBL_BYTES + ALIGN16) & ~ALIGN16;
    constexpr int OFF_VABS = (OFF_INV + TBL_BYTES + ALIGN16) & ~ALIGN16;
    constexpr int OFF_KSTG = (OFF_VABS + 2 * HEAD_DIM * 4 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_Q8 = (OFF_KSTG + TILE_TOK * HEAD_DIM + ALIGN16) & ~ALIGN16;
    constexpr int ARENA_BYTES = OFF_Q8 + TILE_M_ROWS * HEAD_DIM;
    static_assert(ARENA_BYTES + 4 * 1024 <= 64 * 1024, "two blocks' arenas must fit the smallest SM");
    // The K block path stages a group's dims (HEAD_DIM × 8 B = two 4-row
    // spans) across its own K rows and the K staging slab rows of the
    // same index; nothing else writes either during the K phase.
    static_assert(TILE_GROUPS * 4 * HEAD_DIM == TILE_TOK * HEAD_DIM, "K staging fits the two slabs");

    __shared__ __align__(16) uint8_t s_arena[ARENA_BYTES];
    __shared__ __half s_q_scale[TILE_M_ROWS][N_WIN];
    __shared__ TileExt s_ext[2];
    __shared__ TileGroups s_tg[2];
    // A chunk's tile-0 map words, staged by warp 7 for every thread; from
    // tile 1 on each thread carries its own word in a register.
    __shared__ uint32_t s_map[TILE_SLOTS][2][16];       // [group slot][side][word]
    __shared__ uint32_t s_grp[MAX_GROUPS];
    __shared__ GroupSrc s_grp_src[MAX_GROUPS];
    __shared__ int s_n_tiles;
    // The window's live columns accumulated across a multi-pass tile's
    // passes (zero at every tile's start; a single-pass tile reads it as 0).
    __shared__ uint32_t s_colmask;
    // Dense: the slice holding each of this split's windows' base positions
    // (0xffff before the sequence starts), built once in the prologue so a
    // tile's group resolves from one shared read and one independent header
    // load rather than a walk of dependent loads. A split wider than the
    // table walks from the cursor instead.
    constexpr int WIN_TABLE = 256;
    __shared__ uint16_t s_win_slice[WIN_TABLE];
    // A window's first slice header, fetched two tiles ahead by one thread
    // with a 16-byte cp.async: it is the dense walk's first load, which
    // every thread otherwise waits on at the top of the tile. Slot (t & 1)
    // is filled at the top of tile t for tile t + 2 and read at the top of
    // tile t + 1, behind tile t's mid barrier — the fill and the reads never
    // share a slot. `slice` is the header's slice, -1 for none.
    __shared__ __align__(16) TokenSliceHdr s_hdr_pre[2];
    __shared__ int s_hdr_pre_slice[2];
    // Whether group g's side map is a band map (the identity within each
    // palette): the V warps take the vector path only when every live
    // group's V map is.
    __shared__ uint8_t s_band[TILE_SLOTS][2];
    // Whether this tile's V went into the slab by the Q0_V read-through, so
    // the PV adds each dim's centroid · Σp (the centroids' bits are in the
    // tile's absmax buffer). Written by the V warps before the tile's last
    // barrier, read after it.
    __shared__ int s_v_affine;

    int8_t* s_q8 = reinterpret_cast<int8_t*>(s_arena + OFF_Q8);
    int8_t* s_k8 = reinterpret_cast<int8_t*>(s_arena + OFF_K8);
    auto s_k_scale = reinterpret_cast<__half(*)[N_WIN]>(s_arena + OFF_KS);
    int8_t* s_v8t = reinterpret_cast<int8_t*>(s_arena + OFF_V8T);
    auto s_v_scale = reinterpret_cast<__half*>(s_arena + OFF_VS);
    int8_t* s_p8 = reinterpret_cast<int8_t*>(s_arena + OFF_P8);
    auto s_rmax = reinterpret_cast<float(*)[TILE_SM_WARPS]>(s_arena + OFF_RMAX);
    auto s_lsum = reinterpret_cast<float(*)[TILE_SM_WARPS]>(s_arena + OFF_LSUM);
    auto s_alpha = reinterpret_cast<float*>(s_arena + OFF_ALPHA);
    // Rank byte (palette << 6 | rank) of dim d under group g's side map:
    // s_tbl[(g·2 + side)·HEAD_DIM + d].
    uint8_t* s_tbl = s_arena + OFF_TBL;
    // Its inverse — the dim of slot (palette p, rank r):
    // s_inv[(g·2 + side)·HEAD_DIM + p·SUB + r]. A permutation, since the
    // seal gives every palette exactly SUB dims.
    uint8_t* s_inv = s_arena + OFF_INV;
    // Per-dim V absmax of the tile as float bits, an atomicMax target; two
    // buffers alternate by tile so the next tile's zeroing does not race
    // this tile's readers.
    int* s_vabs = reinterpret_cast<int*>(s_arena + OFF_VABS);

    const int g = lane >> 2;
    const int n0 = (lane & 3) * 2;

    // A split with nothing to attend emits (m, l) only.
    auto emit_null = [&]() {
        if (tid < hpg) {
            const int64_t base =
                ((int64_t)slot_idx * n_q_head + first_q_head + tid) * num_splits + split_idx;
            partial_ml[base * 2] = -1e38f;
            partial_ml[base * 2 + 1] = 0.f;
        }
    };

    // Three independent loads open the block — the slot header, the row's
    // selection count, and this split's first chunk of entries — in that
    // order, so all three are in flight together: the header is the
    // longest chain (every slice lookup hangs off it) and nothing below
    // branches until it lands.
    constexpr int SCATTER_WARP = TILE_WARPS - 2;
    constexpr int STAGE_WARP = TILE_WARPS - 1;
    const SlotHeader& slot = get_slot_header(headers_ptr, slot_idx);
    const int n_slices = (int)slot.n_slices;
    const int write_slice_idx = (int)slot.write_slice;
    const uint64_t slices_ptr = slot.slices_ptr;
    // The slot's own rung: its table for K and the unit step, its m² for Q.
    const RopeView rope = rope_view(rungs, slot.rope_rung);

    const bool qsa_on = qsa_active(sel) && !qsa_row_dense(sel, slot_idx);
    const int sel_cnt = qsa_on ? (int)sel.cnt[slot_idx] : 0;
    const uint32_t* sel_entries = sel.entries + (int64_t)slot_idx * sel.stride;
    int w_lo = 0, w_hi = 0, e_lo = 0, e_hi = 0;
    uint32_t ent0 = 0u;
    bool has0 = false;
    if (qsa_on) {
        const int per = (sel_cnt + num_splits - 1) / num_splits;
        e_lo = split_idx * per;
        e_hi = min(sel_cnt, e_lo + per);
        has0 = warp == STAGE_WARP && e_lo + lane < min(e_hi, e_lo + EPS);
        if (has0) ent0 = sel_entries[e_lo + lane];
    }

    if (n_slices == 0) { emit_null(); return; }
    if (n_slices >= (1 << 23)) __trap();   // the group descriptor's slice field

    uint8_t* write_slice_ptr = get_slice_mut<HEAD_DIM>(slices_ptr, write_slice_idx, n_kv_head);
    const int ws_offset = (int)slice_offset(write_slice_ptr);
    const int ws_len = (int)slice_len(write_slice_ptr);
    const int ws_rope = (int)slice_rope(write_slice_ptr);

    // ------------------------------------------------------------------
    // This split's share. Dense: a contiguous slice range. Sparse: a
    // contiguous run of the row's selection entries, walked EPS entries at
    // a time — warp 7 resolves each chunk to groups. The launcher sizes the
    // split count so one chunk is the common case; a row longer than the
    // split cap allows takes several. Both are block-uniform. Split 0 runs
    // the scatter below even with no share (the token must land whichever
    // split reads it) before emitting null.
    // ------------------------------------------------------------------
    // Dense: the split's share is a contiguous range of LOGICAL WINDOWS —
    // 32-position spans of the token sequence, the new token included — not
    // of slices. `kv_len` counts every position the query attends, and a
    // window's tokens are gathered from whichever slices hold them, so the
    // partition (and everything the softmax derives from it) is the same for
    // every chunk layout of the same tokens.
    const int kv_len = ws_rope + ws_len + 1;
    const int n_windows = (kv_len + TILE_TOK - 1) / TILE_TOK;
    if (!qsa_on) {
        const int per = (n_windows + num_splits - 1) / num_splits;
        w_lo = split_idx * per;
        w_hi = min(n_windows, w_lo + per);
    }
    const bool has_share = qsa_on ? (e_lo < e_hi) : (w_lo < w_hi);

    // ------------------------------------------------------------------
    // Fused new-token scatter (warp 6): the step's K/V land in the write
    // slice. Split 0 always writes them — the token persists whichever
    // splits read it this step — and so does a split whose share reads
    // the write slice, so its own read (behind the prologue barrier) sees
    // the token. Dense: the split holding write_slice_idx. Sparse: the
    // entries ascend by block, so the split whose first and last entries
    // bracket the new token's block is the one that can hold it. Every
    // other split skips the scatter: with the split count sized to the
    // card, a short row is mostly null splits, and a redundant copy from
    // each of them is a same-line store storm from every SM that the one
    // real block's reads then queue behind.
    // ------------------------------------------------------------------
    {
        const int within = ws_offset + ws_len;
        constexpr int LANES_PER_PAL = WARP_SIZE / N_PALETTE;
        if (warp == SCATTER_WARP && within < CHUNK_SIZE) {
            bool covers = split_idx == 0;
            if (qsa_on) {
                if (e_lo < e_hi) {
                    const uint32_t qb = qsa_block_of(sel, slot_idx, ws_rope + ws_len);
                    covers |= (sel_entries[e_lo] >> QSA_CELL_BITS) <= qb &&
                              qb <= (sel_entries[e_hi - 1] >> QSA_CELL_BITS);
                }
            } else {
                // The window holding the new token — its position is the
                // sequence's last.
                const int wn = (kv_len - 1) / TILE_TOK;
                covers |= w_lo <= wn && wn < w_hi;
            }
            if (covers) {
                // The new rows and the head record do not depend on each
                // other: all of it goes out in one round behind the slice
                // header, the format bytes beside the arena pointers and
                // the Q row beside K (the R16 format stores both), rather
                // than each behind the last one's landing.
                const int64_t src_base =
                    ((int64_t)slot_idx * n_kv_head + kv_head_idx) * (int64_t)HEAD_DIM;
                const int64_t q_base =
                    ((int64_t)slot_idx * n_q_head + first_q_head) * (int64_t)HEAD_DIM;
                // Each lane's VEC contiguous elements are one 16-byte row
                // segment: one vector load per row, unpacked in registers.
                static_assert(VEC * sizeof(T) == sizeof(uint4) && VEC * sizeof(Q_T) == sizeof(uint4),
                              "a lane's scatter segment is one 16-byte load");
                const uint4 k_raw = *reinterpret_cast<const uint4*>(k_new + src_base + lane * VEC);
                const uint4 v_raw = *reinterpret_cast<const uint4*>(v_new + src_base + lane * VEC);
                const uint4 q_raw = *reinterpret_cast<const uint4*>(q + q_base + lane * VEC);
                const T* k_el = reinterpret_cast<const T*>(&k_raw);
                const T* v_el = reinterpret_cast<const T*>(&v_raw);
                const Q_T* q_el = reinterpret_cast<const Q_T*>(&q_raw);
                float k_regs[VEC], v_regs[VEC], q_regs[VEC];
                #pragma unroll
                for (int j = 0; j < VEC; ++j) {
                    k_regs[j] = to_f32<T>(k_el[j]);
                    v_regs[j] = to_f32<T>(v_el[j]);
                    q_regs[j] = to_f32<Q_T>(q_el[j]);
                }
                const uint8_t* head_ptr = get_head<HEAD_DIM>(write_slice_ptr, kv_head_idx);
                const int pal = lane / LANES_PER_PAL;
                const int local_lane = lane % LANES_PER_PAL;
                const uint64_t k_ptr_p = kvhead_k_ptr<HEAD_DIM>(head_ptr, pal);
                const uint64_t v_ptr_p = kvhead_v_ptr<HEAD_DIM>(head_ptr, pal);
                const int k_fmt = kvhead_k_fmt<HEAD_DIM>(head_ptr, pal);
                const int v_fmt = kvhead_v_fmt<HEAD_DIM>(head_ptr, pal);
                if (k_ptr_p != 0) {
                    char* k_arena = (char*)(uintptr_t)k_ptr_p;
                    char* v_arena = (char*)(uintptr_t)v_ptr_p;
                    const int k_esz = ArenaFormat::float_elem_size(k_fmt);
                    const int v_esz = ArenaFormat::float_elem_size(v_fmt);
                    if (k_fmt == ArenaFormat::R16) {
                        write_regs_to_r16<VEC>(k_arena, 0, within, local_lane, k_regs, q_regs);
                    } else if (k_esz > 0) {
                        write_regs_to_arena<VEC>(k_arena, (int64_t)within * SUB, local_lane,
                                                 k_esz, k_fmt, k_regs);
                    }
                    if (v_esz > 0) {
                        write_regs_to_arena<VEC>(v_arena, (int64_t)within * SUB, local_lane,
                                                 v_esz, v_fmt, v_regs);
                    }
                }
            }
        }
    }

    if (!has_share) { emit_null(); return; }
    int n_tiles = qsa_on ? 0 : (w_hi - w_lo);   // sparse: set from the resolve

    // ------------------------------------------------------------------
    // Tile staging roles.
    //   map word   — every thread: group = warp, side = lane >> 4,
    //                word = lane & 15 (16 dims of the side's palette map).
    //   descriptor — lanes < 8 of every warp: group = warp,
    //                side = (lane >> 2) & 1, palette = lane & 3.
    // Both roles put a thread on the group of its own warp, so a tile's
    // group is resolved ONCE per thread (`group_info`, the dense window
    // walk or the sparse resolve's table) and feeds both its map word and
    // its descriptor. The resolved group carries the slice's KvHead pointer
    // and rope base, so a stager reaches the head record in one global
    // round. A tile past the chunk's last is all dead groups.
    // ------------------------------------------------------------------
    const int st_g = warp, st_side = (lane >> 2) & 1, st_p = lane & 3;
    const bool stager = lane < 2 * N_PALETTE;
    static_assert(TILE_WARPS * 2 * N_PALETTE == N_STAGERS, "a warp's first 8 lanes stage its group");
    const int mw_side = lane >> 4, mw_word = lane & 15;
    struct GroupInfo { uint32_t desc; GroupSrc src; int col0; };
    // The dense walk's slice cursor: the slice holding the base of the last
    // window this thread resolved, seated once by the gallop at the split's
    // first window. A tile asks for its own window and the next one's, and a
    // multi-pass tile asks for its own again after the lookahead, so the
    // cursor is a hint the walk corrects in either direction — by the
    // handful of slices two neighbouring windows can differ by.
    int cur_slice = -1;
    // Group `gi` of tile `t`.
    //
    // Dense: tile t is window [wb, wb + 32) of positions, and its groups are
    // the aligned physical quads of the slices holding those positions, in
    // position order — a whole window inside one slice is 8 full quads exactly
    // as before; a window a hole falls in has a partial quad on each side of
    // it. `gi` past the last quad is dead. The walk is register arithmetic
    // except where it steps to the next slice (one header load).
    //
    // Sparse: the resolve's packed groups, 8 per tile, at column 4·gi.
    // `ahead`: the call is the tile loop's own look-ahead (tile t + 1 from
    // the top of tile t), whose first slice header was prefetched into
    // `s_hdr_pre[t & 1]`; any other caller loads the header itself.
    auto group_info = [&](int t, int gi, bool ahead) -> GroupInfo {
        GroupInfo g = {0u, {0u, 0}, 0};
        if (t >= n_tiles) return g;
        if (qsa_on) {
            g.desc = s_grp[t * TILE_GROUPS + gi];
            g.src = s_grp_src[t * TILE_GROUPS + gi];
            g.col0 = gi * GROUP_TOK;
            return g;
        }
        const int wb = (w_lo + t) * TILE_TOK;
        const int we = min(wb + TILE_TOK, kv_len);
        // The slice's rows holding window positions: [row_lo, row_hi).
        auto rows_of = [&](const TokenSliceHdr& hh, int slice, int& row_lo, int& row_hi) {
            const int off = hh.offset();
            const int rb = hh.rope_base();
            const int eff = tile_slice_eff_len(hh, slice, write_slice_idx);
            row_lo = off + max(wb - rb, 0);
            row_hi = off + min(eff, we - rb);
        };
        // The window's first slice: the table's, resolved in the prologue.
        // Past the table, seat on the cursor's slice (the gallop's, at the
        // split's first window) and step to the slice whose tokens reach the
        // window: back while the slice begins past it, then forward past any
        // that ends before it (the cursor's, when the window moved on) or that
        // the window starts past the end of (an empty writer behind a partial
        // chunk shares its base).
        int s;
        TokenSliceHdr h;
        if (t < WIN_TABLE) {
            const int ws = (int)s_win_slice[t];
            if (ws == 0xffff) return g;
            s = ws;
            if (ahead && s == s_hdr_pre_slice[t & 1]) h = s_hdr_pre[t & 1];
            else h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
        } else {
            s = cur_slice < 0
                ? tile_slice_holding<HEAD_DIM>(slices_ptr, n_slices, n_kv_head, wb)
                : cur_slice;
            h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
            while (s > 0 && h.rope_base() > wb) {
                --s;
                h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
            }
        }
        int row_lo, row_hi;
        rows_of(h, s, row_lo, row_hi);
        while (row_lo >= row_hi) {
            if (++s >= n_slices) return g;
            h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
            if (h.rope_base() >= we) return g;
            rows_of(h, s, row_lo, row_hi);
        }
        cur_slice = s;
        // Quad gi: skip whole slices' worth of quads, then index into the
        // slice that holds it — arithmetic per slice crossed, not per quad.
        int q, rem = gi;
        for (;;) {
            const int here = ((row_hi - 1) >> 2) - (row_lo >> 2) + 1;
            if (rem < here) {
                q = (row_lo & ~3) + rem * GROUP_TOK;
                break;
            }
            rem -= here;
            do {
                if (++s >= n_slices) return g;
                h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
                if (h.rope_base() >= we) return g;
                rows_of(h, s, row_lo, row_hi);
            } while (row_lo >= row_hi);
        }
        uint32_t mask = 0u;
        #pragma unroll
        for (int j = 0; j < GROUP_TOK; ++j)
            if (q + j >= row_lo && q + j < row_hi) mask |= 1u << j;
        g.desc = grp_pack(s, q, mask);
        g.src.kvheads = h.kvheads_ptr();
        g.src.rope_base = h.rope_base() - h.offset();
        g.col0 = (g.src.rope_base + q) - wb;
        return g;
    };
    // How many quads window `t` spans — the tile's pass count is this over
    // TILE_GROUPS. Counted in one pass over the window's slices, each
    // contributing the aligned quads its rows in the window touch: one
    // header load per slice, arithmetic otherwise. (Asking `group_info` for
    // groups until a dead one is the same answer at nine walks' cost, every
    // one of them a chain of dependent loads on a single thread — measured as
    // a per-launch tax that doubled a one-window decode.)
    auto window_groups = [&](int t, bool ahead) -> int {
        if (qsa_on || t >= n_tiles) return 0;
        const int wb = (w_lo + t) * TILE_TOK;
        const int we = min(wb + TILE_TOK, kv_len);
        int s;
        TokenSliceHdr h;
        if (t < WIN_TABLE) {
            const int ws = (int)s_win_slice[t];
            if (ws == 0xffff) return 0;
            s = ws;
            if (ahead && s == s_hdr_pre_slice[t & 1]) h = s_hdr_pre[t & 1];
            else h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
        } else {
            s = cur_slice < 0
                ? tile_slice_holding<HEAD_DIM>(slices_ptr, n_slices, n_kv_head, wb)
                : cur_slice;
            h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
            while (s > 0 && h.rope_base() > wb) {
                --s;
                h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
            }
        }
        int n = 0;
        for (;;) {
            const int off = h.offset();
            const int rb = h.rope_base();
            const int eff = tile_slice_eff_len(h, s, write_slice_idx);
            const int row_lo = off + max(wb - rb, 0);
            const int row_hi = off + min(eff, we - rb);
            if (row_lo < row_hi) n += ((row_hi - 1) >> 2) - (row_lo >> 2) + 1;
            // A slice reaching the window's end holds the rest of it: no
            // probe of the next slice's header (a dependent load on warp 0's
            // critical path, every tile, for a packed layout).
            if (rb + eff >= we) break;
            if (++s >= n_slices) break;
            h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
            if (h.rope_base() >= we) break;
        }
        return n;
    };
    // One map word of a resolved group: its `side` map, 16-dim word `w`
    // (0 for a dead group or a word past the map).
    auto map_word_of = [&](const GroupInfo& g, int side, int w) -> uint32_t {
        if (grp_mask(g.desc) == 0u || w >= PAL_U32) return 0u;
        const uint8_t* hd = get_head_at<HEAD_DIM>(g.src.kvheads, kv_head_idx);
        const uint8_t* pm = side ? kvhead_v_pal_map<HEAD_DIM>(hd) : kvhead_k_pal_map<HEAD_DIM>(hd);
        return reinterpret_cast<const uint32_t*>(pm)[w];
    };
    auto load_map_word = [&](int t, int gi, int side, int w) -> uint32_t {
        return map_word_of(group_info(t, gi, false), side, w);
    };
    // One descriptor item of a resolved group: (side, palette p).
    auto desc_of = [&](const GroupInfo& g, int side, int p, NextTile& nx) {
        nx.desc = g.desc; nx.rope0 = 0; nx.col0 = 0; nx.ptr = 0; nx.inv = 1.f; nx.fmt = 0;
        if (grp_mask(g.desc) != 0u) {
            const uint8_t* hd = get_head_at<HEAD_DIM>(g.src.kvheads, kv_head_idx);
            nx.ptr = side ? kvhead_v_ptr<HEAD_DIM>(hd, p) : kvhead_k_ptr<HEAD_DIM>(hd, p);
            nx.fmt = side ? kvhead_v_fmt<HEAD_DIM>(hd, p) : kvhead_k_fmt<HEAD_DIM>(hd, p);
            nx.inv = __frcp_rn(side ? kvhead_v_scale<HEAD_DIM>(hd, p) : kvhead_k_scale<HEAD_DIM>(hd, p));
            nx.rope0 = g.src.rope_base + grp_within(g.desc);
            nx.col0 = g.col0;
        }
    };
    auto load_desc = [&](int t, int gi, int side, int p, NextTile& nx) {
        desc_of(group_info(t, gi, false), side, p, nx);
    };
    auto commit_desc = [&](int buf, int gi, int side, int p, const NextTile& nx) {
        s_ext[buf].gbase[gi][side][p] = (const char*)(uintptr_t)nx.ptr;
        s_ext[buf].inv[gi][side][p] = nx.inv;
        s_ext[buf].fmt[gi][side][p] = (uint8_t)nx.fmt;
        if (side == 0 && p == 0) {
            s_tg[buf].desc[gi] = nx.desc;
            s_tg[buf].rope0[gi] = nx.rope0;
            s_tg[buf].col0[gi] = (int8_t)nx.col0;
            if (gi == 0) s_tg[buf].n_groups = nx.ngroups;
        }
    };
    // The tile's pass count comes from its window's quad count: a sparse
    // tile is the resolve's fixed 8 groups, a dense one is walked.
    auto tile_groups = [&](int t, bool ahead) -> int {
        return qsa_on ? TILE_GROUPS : window_groups(t, ahead);
    };
    // Issue tile t's staging loads into registers under the tile-loop
    // roles. Nothing waits on them here: the descriptors are consumed by
    // `stage_prefetch` and `stage_commit`, the map word by the next tile's
    // table build.
    auto stage_issue = [&](int t, NextTile& nx, bool ahead) {
        const GroupInfo g = group_info(t, warp, ahead);
        nx.map = map_word_of(g, mw_side, mw_word);
        nx.desc = 0u; nx.rope0 = 0; nx.col0 = 0; nx.ngroups = 0; nx.ptr = 0; nx.inv = 1.f; nx.fmt = 0;
        if (stager) desc_of(g, st_side, st_p, nx);
        if (tid == 0) nx.ngroups = (t < n_tiles) ? tile_groups(t, ahead) : 0;
    };
    // L2 prefetch of the staged group's span: a dtype palette's 4 tokens
    // (SUB elements each, contiguous), a quant palette's whole block run
    // (every dim's block holds the 4 tokens' bytes).
    auto prefetch_span = [&](const NextTile& nx) {
        if (nx.ptr != 0 && grp_mask(nx.desc) != 0u) {
            const int es = ArenaFormat::float_elem_size(nx.fmt);
            uint64_t lo, hi;
            if (es == 0) {
                lo = nx.ptr;
                hi = lo + (uint64_t)SUB * ArenaAccessor::get_quant_block_bytes(nx.fmt);
            } else {
                lo = nx.ptr + (uint64_t)grp_within(nx.desc) * SUB * es;
                hi = lo + (uint64_t)GROUP_TOK * SUB * es;
            }
            for (uint64_t a = lo & ~(uint64_t)127; a < hi; a += 128)
                asm volatile("prefetch.global.L2 [%0];" :: "l"(a));
        }
    };
    auto stage_prefetch = [&](const NextTile& nx) {
        if (stager) prefetch_span(nx);
    };
    auto stage_commit = [&](int buf, const NextTile& nx) {
        if (stager) commit_desc(buf, st_g, st_side, st_p, nx);
    };
    // Whether the next tile's K lower half is staged through the slab, decided
    // a tile ahead from the descriptors in registers: a live group whose K is
    // one narrow dtype across the four palettes under a band map — the
    // conditions of the vector path, the slab's reader. The decision is the
    // warp's own (it decodes its own group, and the block path — the slab's
    // other user — scribbles only in its own group's span) and rides in a
    // register to the next tile's table phase, where the copies are issued;
    // it replaces the block-wide vote the decode used to take from shared
    // state after the table barrier. Returns the element size, 0 for none.
    //
    // The copies themselves are NOT issued here. Issuing them at this point —
    // a tile ahead, with the V decode, softmax and PV to land under — was
    // measured: fewer instructions, but the stager region's stalls doubled and
    // the tile loop lost 6–17 % at 32K and above, so they issue where they
    // always did.
    auto stage_k_ahead = [&](const NextTile& nx) -> int {
        if constexpr (ROPE_INTERLEAVED) return 0;
        static_assert(N_PALETTE == 4, "stager lanes 0..3 hold side 0's four palettes");
        constexpr int WORDS_PER_BAND = SUB / 16;
        const uint32_t band = 0x55555555u * (uint32_t)(mw_word / WORDS_PER_BAND);
        const bool band0 =
            __all_sync(0xffffffffu, mw_side != 0 || nx.map == 0u || nx.map == band);
        const uint32_t desc = __shfl_sync(0xffffffffu, nx.desc, 0);
        const int f0 = __shfl_sync(0xffffffffu, nx.fmt, 0);
        const bool fmt_ok = __all_sync(0xffffffffu, lane >= N_PALETTE || nx.fmt == f0);
        if (!band0 || !fmt_ok || grp_mask(desc) == 0u
            || !i8_is_narrow_dtype_format((uint32_t)f0))
            return 0;
        return ArenaFormat::float_elem_size(f0);
    };
    // The ninth quad of tile `t` — slot TILE_GROUPS — staged by warp 0 alone
    // into descriptor buffer `buf`, when the window spans exactly TILE_SLOTS
    // quads (`ngroups`, block-uniform, from the tile's staging): its eight
    // descriptor items on the stager lanes, its 32 map words one per lane
    // into s_map for the table build at the tile's top. Otherwise the slot
    // is committed dead. The loads are exposed on warp 0, a softmax warp,
    // between its K decode and the K barrier it waits at anyway.
    auto stage_slot8 = [&](int t, int buf, int ngroups, bool ahead) {
        if (ngroups == TILE_SLOTS) {
            const GroupInfo g = group_info(t, TILE_GROUPS, ahead);
            if (stager) {
                NextTile w;
                desc_of(g, st_side, st_p, w);
                w.ngroups = ngroups;
                commit_desc(buf, TILE_GROUPS, st_side, st_p, w);
            }
            s_map[TILE_GROUPS][mw_side][mw_word] = map_word_of(g, mw_side, mw_word);
        } else if (lane == 0) {
            s_tg[buf].desc[TILE_GROUPS] = 0u;
        }
    };
    // A resolved sparse chunk's tile 0, staged by the resolving warp alone
    // between chunks: the 64 descriptor items two per lane, the 256 map
    // words eight per lane, all loads in flight before the first shared
    // store, and the spans prefetched to L2 as soon as the descriptors
    // land. (The split's first tile is staged by the whole block under the
    // tile loop's roles, in the prologue.)
    auto stage_tile0 = [&]() {
        // Every group of tile 0, resolved up front per lane: the ten items a
        // lane stages (two descriptors, eight map words) span all eight
        // groups, and resolving each behind its own item put ten record-load
        // chains in series. Resolved first (ten walks of one window, its
        // header hot in L1 after the first), the ten record loads issue
        // together. The map words' groups are the unrolled index (registers);
        // the descriptors' two are lane-dependent, so they walk directly
        // rather than index the array.
        GroupInfo g0[TILE_GROUPS];
        #pragma unroll
        for (int gi = 0; gi < TILE_GROUPS; ++gi) g0[gi] = group_info(0, gi, false);
        NextTile w[2];
        uint32_t mws[8];
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int i = lane + 32 * k;
            desc_of(group_info(0, i >> 3, false), (i >> 2) & 1, i & 3, w[k]);
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int i = lane + 32 * k;
            mws[k] = map_word_of(g0[k], (i >> 4) & 1, i & 15);
        }
        #pragma unroll
        for (int k = 0; k < 2; ++k) prefetch_span(w[k]);
        // Item 0 — lane 0's first — is the one `commit_desc` takes the pass
        // count from.
        if (lane == 0) w[0].ngroups = tile_groups(0, false);
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int i = lane + 32 * k;
            commit_desc(0, i >> 3, (i >> 2) & 1, i & 3, w[k]);
        }
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const int i = lane + 32 * k;
            s_map[i >> 5][(i >> 4) & 1][i & 15] = mws[k];
        }
        // A sparse tile is the resolve's fixed TILE_GROUPS groups: no ninth.
        if (lane == 0) s_tg[0].desc[TILE_GROUPS] = 0u;
    };
    // Rank table of (group = warp, side, word) from the tile's committed
    // map word. A palette holds exactly SUB dims (a rank is 6 bits), so
    // the four palette counts of a word pack a byte each and one 16-lane
    // prefix scan of the packed word ranks all four palettes at once.
    //
    // Two maps have the identity table — rank byte d for dim d — and the
    // warp votes past the scan for them: the band partition (palette
    // d / SUB, the map of every float chunk and every chunk sealed under
    // a format override; word w reads 0x55555555 × (w / WORDS_PER_BAND))
    // and the all-zero map of a dead group. A word-wise mixture of the
    // two would give palette 0 more than SUB dims, so a warp whose words
    // all match one or the other holds one of the two maps.
    //
    // The inverse table (dim of slot) is built alongside: the identity is
    // its own inverse, and the scan path scatters each dim's index to its
    // rank byte.
    auto build_tables = [&](int g, uint32_t mw) {
        constexpr int WORDS_PER_BAND = SUB / 16;
        static_assert(SUB % 16 == 0, "a palette band is whole map words");
        const uint32_t band = 0x55555555u * (uint32_t)(mw_word / WORDS_PER_BAND);
        const int tbl_off = (g * 2 + mw_side) * HEAD_DIM;
        if (__all_sync(0xffffffffu, mw == 0u || mw == band)) {
            if (mw_word < PAL_U32) {
                const uint32_t w0 = 0x03020100u + 0x10101010u * (uint32_t)mw_word;
                const uint4 ident = make_uint4(w0, w0 + 0x04040404u, w0 + 0x08080808u, w0 + 0x0c0c0c0cu);
                *(uint4*)&s_tbl[tbl_off + 16 * mw_word] = ident;
                *(uint4*)&s_inv[tbl_off + 16 * mw_word] = ident;
                if (mw_word == 0) s_band[g][mw_side] = 1;
            }
            return;
        }
        if (mw_word == 0) s_band[g][mw_side] = 0;
        // Byte p of `cnt` counts the word's dims in palette p; the
        // exclusive prefix over the side's 16 words is the palette's rank
        // base at this word.
        uint32_t cnt = 0u;
        #pragma unroll
        for (int p = 0; p < N_PALETTE; ++p) {
            const uint32_t x = mw ^ (0x55555555u * (uint32_t)p);
            cnt += (uint32_t)__popc(~(x | (x >> 1)) & 0x55555555u) << (8 * p);
        }
        uint32_t incl = cnt;
        #pragma unroll
        for (int d = 1; d < 16; d <<= 1) {
            const uint32_t v = __shfl_up_sync(0xffffffffu, incl, d, 16);
            if (mw_word >= d) incl += v;
        }
        const uint32_t excl = incl - cnt;
        if (mw_word < PAL_U32) {
            // Dim i's rank is its palette's base plus the lower dims of the
            // word in the same palette: the palette broadcast to every 2-bit
            // field, XOR-matched against the word under a below-i mask.
            uint32_t words[4] = { 0u, 0u, 0u, 0u };
            #pragma unroll
            for (int i = 0; i < 16; ++i) {
                const uint32_t p = (mw >> (2 * i)) & 3u;
                const uint32_t x = mw ^ (0x55555555u * p);
                const uint32_t below = 0x55555555u & ((1u << (2 * i)) - 1u);
                const uint32_t local = (uint32_t)__popc(~(x | (x >> 1)) & below);
                const uint32_t base = __byte_perm(excl, 0u, 0x4440u | p);
                const uint32_t rb = (p << 6) + base + local;
                constexpr uint32_t INSERT[4] = { 0x3214u, 0x3240u, 0x3410u, 0x4210u };
                words[i >> 2] = __byte_perm(words[i >> 2], rb, INSERT[i & 3]);
                s_inv[tbl_off + (int)(rb & 0xffu)] = (uint8_t)(16 * mw_word + i);
            }
            *(uint4*)&s_tbl[tbl_off + 16 * mw_word] =
                make_uint4(words[0], words[1], words[2], words[3]);
        }
    };
    // ------------------------------------------------------------------
    // Prologue, two chains side by side. Warps 0–6: rotate and quantise
    // the Q rows into the Q slab (a warp per row, a lane per dim of
    // each 32-dim window; the window absmax is a warp reduce). Warp 7:
    // resolve the split's first selection chunk and stage tile 0. One
    // barrier publishes the scatter, s_q8/s_q_scale, s_grp/s_n_tiles and
    // tile 0's descriptors together.
    // ------------------------------------------------------------------
    int chunk_lo = e_lo;   // first unresolved sparse entry
    // Both V absmax buffers start clear (atomic accumulators, read before
    // written); each tile clears the other buffer for the tile after it.
    for (int i = tid; i < 2 * HEAD_DIM; i += TILE_THREADS) s_vabs[i] = 0;
    if (warp < STAGE_WARP) {
        const int q_pos = ws_rope + ws_len;
        // The window scale carries the softmax scale and the change of
        // base, so a dequantised score is the log2-domain logit outright.
        const float q_scale_mul = softmax_scale * TILE_LOG2E / 127.f;
        for (int r = warp; r < TILE_M_ROWS; r += STAGE_WARP) {
            // A pad row (r >= hpg) is all zeros with zero scales: written
            // directly, no rotation and no absmax reduce.
            if (r >= hpg) {
                #pragma unroll
                for (int w = 0; w < N_WIN; ++w) s_q8[sw_off<HEAD_DIM>(r, lane + 32 * w)] = 0;
                if (lane < N_WIN) s_q_scale[r][lane] = __float2half(0.f);
                continue;
            }
            float x[N_WIN];
            const Q_T* qrow = q + ((int64_t)slot_idx * n_q_head + first_q_head + r) * HEAD_DIM;
            #pragma unroll
            for (int w = 0; w < N_WIN; ++w) x[w] = to_f32<Q_T>(qrow[lane + 32 * w]);
            i8_apply_rope<HEAD_DIM, N_WIN>(x, q_pos, lane, ROPE_INTERLEAVED ? 1 : 0, rope.for_q());
            #pragma unroll
            for (int w = 0; w < N_WIN; ++w) {
                float a = fabsf(x[w]);
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1)
                    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
                const float inv = (a > 0.f) ? 127.f / a : 0.f;
                s_q8[sw_off<HEAD_DIM>(r, lane + 32 * w)] = i8_quant(x[w], inv);
                if (lane == 0) s_q_scale[r][w] = __float2half(a * q_scale_mul);
            }
        }
    } else {
        if (qsa_on) {
            tile_resolve_entries<HEAD_DIM>(ent0, has0, sel, slot_idx, slices_ptr,
                                           n_slices, write_slice_idx, n_kv_head, lane,
                                           s_grp, s_grp_src, &s_n_tiles);
            __syncwarp();
            n_tiles = s_n_tiles;
            if (lane == 0) {
                s_hdr_pre_slice[0] = -1;
                s_hdr_pre_slice[1] = -1;
            }
        } else {
            // The window table: a lane per slice, 32 slices a round from the
            // slice holding the split's first position (one gallop, warp-wide
            // on the same addresses), each lane claiming the windows whose
            // base lies in its slice's positions [rb, rb + eff). An empty
            // slice claims none; a window before the sequence keeps 0xffff.
            // The round that reaches the split's last position is the last.
            for (int i = lane; i < WIN_TABLE; i += WARP_SIZE) s_win_slice[i] = 0xffff;
            __syncwarp();
            const int pos_lo = w_lo * TILE_TOK;
            const int pos_hi = min(w_hi * TILE_TOK, kv_len);
            const int s0 = tile_slice_holding<HEAD_DIM>(slices_ptr, n_slices, n_kv_head, pos_lo);
            for (int sb = s0; sb < n_slices; sb += WARP_SIZE) {
                const int s = sb + lane;
                const bool has = s < n_slices;
                int rb = 0, eff = 0;
                if (has) {
                    const TokenSliceHdr h = load_token_slice<HEAD_DIM>(slices_ptr, s, n_kv_head);
                    rb = h.rope_base();
                    eff = tile_slice_eff_len(h, s, write_slice_idx);
                }
                if (has && eff > 0) {
                    const int wf = max((rb + TILE_TOK - 1) / TILE_TOK, w_lo);
                    const int wl = min((rb + eff - 1) / TILE_TOK, min(w_hi, w_lo + WIN_TABLE) - 1);
                    for (int w = wf; w <= wl; ++w) s_win_slice[w - w_lo] = (uint16_t)s;
                }
                if (__any_sync(0xffffffffu, has && rb + eff >= pos_hi)) break;
            }
            __syncwarp();
            // Tile 1's first header, for tile 0's look-ahead; tile 0's own
            // is staged by the whole block below without one.
            if (lane == 0) {
                s_hdr_pre_slice[0] = -1;
                const int s1 = (n_tiles > 1) ? (int)s_win_slice[1] : 0xffff;
                s_hdr_pre_slice[1] = (s1 != 0xffff) ? s1 : -1;
                if (s1 != 0xffff) s_hdr_pre[1] = load_token_slice<HEAD_DIM>(slices_ptr, s1, n_kv_head);
            }
        }
    }
    if (qsa_on) chunk_lo = min(e_hi, chunk_lo + EPS);
    __syncthreads();
    if (qsa_on) n_tiles = s_n_tiles;
    // Tile 0 is staged under the tile loop's own roles — one descriptor or
    // map word per thread, every chain in flight at once across the block —
    // rather than by one warp staging ten items a lane in series while the
    // other seven wait at the barrier. (Measured at 8K, three tiles a block:
    // that serial stage was a tenth of the kernel's stall samples.)
    int k_stg_esz = 0;   // K lower half staged for the next tile: element size, 0 for none
    {
        NextTile nx0;
        stage_issue(0, nx0, false);
        stage_prefetch(nx0);
        stage_commit(0, nx0);
        s_map[warp][mw_side][mw_word] = nx0.map;
        k_stg_esz = stage_k_ahead(nx0);
        if (warp == 0) stage_slot8(0, 0, __shfl_sync(0xffffffffu, nx0.ngroups, 0), false);
    }
    __syncthreads();

    const int ns = warp & 3;        // QK n-slice (tokens ns*8 .. ns*8+7) on the softmax warps
    uint32_t mw = 0u;               // this thread's map word of the tile being decoded

    // Per-warp output state: every warp accumulates its own PV_H output
    // slices at dim_base and carries the 16-row running sums (rebuilt each
    // tile from the softmax warps' posts); the running max is live only on
    // the softmax warps, which are the ones that read it.
    const int dim_base = warp * (HEAD_DIM / TILE_WARPS);
    float o_acc[PV_H][4];
    #pragma unroll
    for (int s = 0; s < PV_H; ++s)
        #pragma unroll
        for (int i = 0; i < 4; ++i) o_acc[s][i] = 0.f;
    float m_run[2] = { -INFINITY, -INFINITY };
    float l_run[2] = { 0.f, 0.f };

    // ==================================================================
    // Tile loop. The outer loop re-runs it per resolved sparse chunk; a
    // dense split is one pass.
    // ==================================================================
    int tile_no = 0;   // tiles done across chunks; selects the V absmax buffer
    for (;;) {
    for (int t = 0; t < n_tiles; ++t, ++tile_no) {
        const int cur = t & 1, nxt = cur ^ 1;
        const TileGroups& tg = s_tg[cur];
        const TileExt& ext = s_ext[cur];
        // A window of more quads than one pass holds — a hole off the 4-grid
        // splits a quad in two — is taken in passes that share this tile's
        // softmax and V scale (the pre-sweep below). Block-uniform: the count
        // was committed before the previous tile's last barrier.
        const int n_pass = (tg.n_groups <= TILE_SLOTS)
                               ? 1
                               : (tg.n_groups + TILE_GROUPS - 1) / TILE_GROUPS;

        // -------------------- STAGE --------------------
        // The next tile's loads go out first; this tile's tables are built
        // from its map word — this thread's own word of s_map, left there by
        // the prologue's staging for tile 0 and by the previous tile's commit
        // for every later one — only for a group whose word differs from the
        // previous tile's (warp-uniform vote; a chunk's tile 0 always builds).
        NextTile nx;
        {
            const uint32_t mw_cur = s_map[warp][mw_side][mw_word];
            stage_issue(t + 1, nx, true);
            if (t == 0 || __any_sync(0xffffffffu, mw_cur != mw)) build_tables(warp, mw_cur);
            mw = mw_cur;
            // The ninth quad's tables, by warp 0 from the words its staging
            // left in s_map, whenever the slot is live (block-uniform: the
            // descriptor was committed before the previous tile's last
            // barrier, or in the prologue).
            if (warp == 0 && grp_mask(tg.desc[TILE_GROUPS]) != 0u)
                build_tables(TILE_GROUPS, s_map[TILE_GROUPS][mw_side][mw_word]);
            if (tid == 0) {
                s_colmask = 0u;
                // Two tiles ahead: the slice header tile t + 2's walk starts
                // from, into the slot tile t + 1's look-ahead reads. Waited
                // for before this tile's mid barrier, which publishes it.
                int ps = -1;
                if (!qsa_on && t + 2 < n_tiles && t + 2 < WIN_TABLE) {
                    const int ws = (int)s_win_slice[t + 2];
                    if (ws != 0xffff) ps = ws;
                }
                s_hdr_pre_slice[t & 1] = ps;
                if (ps >= 0) {
                    tile_cp_async_16(&s_hdr_pre[t & 1],
                                     get_slice<HEAD_DIM>(slices_ptr, ps, n_kv_head));
                    cp_async_commit<true>();
                }
            }
        }
        __syncthreads(); // publishes the rank tables
        // Whether the ninth slot is live this tile (block-uniform: committed
        // before the previous tile's last barrier, or in the prologue). The
        // V side and warp 0's second K round key on it, so an eight-quad
        // tile runs code with no trace of the slot.
        const bool has8 = grp_mask(tg.desc[TILE_GROUPS]) != 0u;

        // The next tile's staging, committed: its descriptors to the other
        // buffer and its map word to this thread's s_map word — nothing reads
        // either until the next tile's first barrier publishes them, and
        // every warp is past this tile's — and its spans warmed in L2 while
        // this tile decodes. The loads went out at the tile's top and the
        // table build has run under them. Committed here rather than after
        // the K decode, nothing of the staged tile rides through the decode:
        // a `NextTile` is nine words a thread, which the decode's register
        // peak spilled to local memory and reloaded every tile. What the tile
        // still needs of it is one word each — the next tile's K staging
        // decision and its quad count. The prefetch and the commit run on the
        // stager lanes alone, in loops of per-lane length; the warp
        // reconverges before the decode's warp-wide votes.
        stage_prefetch(nx);
        stage_commit(nxt, nx);
        s_map[warp][mw_side][mw_word] = nx.map;
        __syncwarp();
        const int k_stg_next = (t + 1 < n_tiles) ? stage_k_ahead(nx) : 0;
        const int ngroups_next = __shfl_sync(0xffffffffu, nx.ngroups, 0);

        // K lower-half staging: on a single-pass tile whose group's K is a
        // narrow dtype under a band map (`k_stg_esz`, decided a tile ahead
        // by `stage_k_ahead` from the descriptors in registers), the warp
        // copies its group's four rows of dims [0, HEAD_DIM/2) into its span
        // of the slab now with cp.async, and the decode reads quad A from
        // them once the rope rows and quad B's global load are in flight.
        // Half the K bytes' latency then sits under the tile's staging and
        // table work rather than at the head of the decode.
        uint8_t* k_raw = nullptr;
        if (k_stg_esz > 0 && n_pass == 1) {
            k_raw = s_arena + OFF_KSTG + warp * (GROUP_TOK * HEAD_DIM);
            const int within = grp_within(tg.desc[warp]);
            const int row = (HEAD_DIM / 2) * k_stg_esz;   // a token's lower half, bytes
            const int ch = row / 16;                      // its 16-byte chunks
            #pragma unroll
            for (int r = 0; r < 2; ++r) {
                const int q = lane + WARP_SIZE * r;
                if (q < GROUP_TOK * ch) {
                    const int j = q / ch, cq = q % ch;
                    const int p = cq / (ch / 2);          // palette 0 or 1
                    const int off = (cq % (ch / 2)) * 16;
                    const char* src = ext.gbase[warp][0][p]
                                      + (int64_t)(within + j) * SUB * k_stg_esz + off;
                    tile_cp_async_16(k_raw + j * row + cq * 16, src);
                }
            }
            cp_async_commit<true>();
        }

        // ------------------------------------------------------------------
        // K decode, and the multi-pass window. A single-pass tile (every
        // window without a hole off the 4-grid inside it) is one trip of
        // sweep 1 with its groups already staged: the K decode and nothing
        // else. A multi-pass window (n_pass > 1) does not fit one set of
        // group slots, so its passes are published here one at a time, and
        // the tile's regular body runs the last. What makes the passes ONE
        // tile numerically is that they share the softmax (the K rows of
        // every pass sit in their own columns of the K slab when the body's
        // QK runs, and the column mask accumulates in s_colmask) and the V
        // scale (every pass's per-dim absmax folds into the tile's absmax
        // buffer before any pass's V is quantised, so the body's block path
        // quantises the last pass against the window's maximum and sweep 2
        // quantises the earlier ones against the same). The V side takes
        // the block path throughout — its absmax is an atomic fold into
        // that buffer, which is what lets it span passes.
        //
        // Slow, exposed and serial by design: a pass costs a synchronous
        // stage, and the earlier passes' V rows are decoded twice (absmax,
        // then quantise). A window off the 4-grid — every window after a
        // section sealed mid-quad, until the next boundary — spans nine
        // quads and fits one pass through slot TILE_GROUPS; only a window
        // with two holes inside it (more than TILE_SLOTS quads) comes here.
        // ------------------------------------------------------------------
        int* vabs_mp = s_vabs + (tile_no & 1) * HEAD_DIM;
        // Publish pass p's groups into this tile's descriptor buffer and
        // tables, and fold its columns into the tile's mask.
        auto publish = [&](int p) {
            const GroupInfo g = group_info(t, p * TILE_GROUPS + warp, false);
            if (stager) {
                NextTile w;
                desc_of(g, st_side, st_p, w);
                w.ngroups = tg.n_groups;
                commit_desc(cur, st_g, st_side, st_p, w);
                if (st_side == 0 && st_p == 0)
                    atomicOr(&s_colmask, grp_cols(grp_mask(w.desc), w.col0));
            }
            const uint32_t word = map_word_of(g, mw_side, mw_word);
            __syncthreads();   // the previous pass's readers are done with the tables
            build_tables(warp, word);
            mw = word;
            __syncthreads();
        };
        auto v_live = [&]() -> uint32_t {
            const bool l = lane < TILE_SLOTS && grp_mask(tg.desc[lane]) != 0u;
            return __ballot_sync(0xffffffffu, l);
        };
        // Sweep 1: every pass's K rows and V absmax. The K decode's one site.
        //
        // The ninth quad (slot TILE_GROUPS, live only on a single-pass
        // nine-quad window) is warp 0's second round of the same loop, through
        // its own staging span (warp 0's is span 0), from the group's spans —
        // a dead slot returns at once. A loop rather than a second call: the
        // K decode is the vector and block paths over every format, and a
        // second call site was a second inlined copy of all of it.
        //
        // Sweep 2 (a multi-pass tile only): the earlier passes' V rows,
        // quantised against the window's absmax. The last pass is the body's:
        // its groups are published and go out as the tile's.
        //
        // Both sweeps are one loop, `i` running sweep 1's passes and then
        // sweep 2's, so the V block decode both take has one site — two
        // sites were two inlined copies of it over every format.
        const int k_rounds = (warp == 0 && has8) ? 2 : 1;
        const int n_iter = (n_pass > 1) ? 2 * n_pass : 1;
        for (int i = 0; i < n_iter; ++i) {
            const bool sweep2 = i >= n_pass;
            const int p = sweep2 ? i - n_pass : i;
            if (n_pass > 1) publish(p);
            if (sweep2 && p + 1 == n_pass) break;
            if (!sweep2) {
                #pragma unroll 1
                for (int r = 0; r < (n_pass == 1 ? k_rounds : 1); ++r) {
                    tile_k_decode<HEAD_DIM, ROPE_INTERLEAVED>(ext, tg, s_tbl, s_inv, s_arena + OFF_KSTG,
                                                              warp, s_k8, s_k_scale, rope,
                                                              r == 0 ? warp : TILE_GROUPS, lane,
                                                              r == 0 ? k_raw : nullptr);
                }
            }
            if (n_pass > 1) {
                if (warp >= TILE_SM_WARPS) {
                    // Sweep 1 folds the absmax and drops the words; sweep 2
                    // decodes again and quantises. The two ranks are a real
                    // loop (this path is a tile wider than a pass, the rare
                    // case).
                    const int vw = warp - TILE_SM_WARPS;
                    const uint32_t live_g = v_live();
                    #pragma unroll 1
                    for (int s = 0; s < 2; ++s) {
                        uint32_t w[TILE_SLOTS][GROUP_TOK / 2];
                        tile_v_block_decode<HEAD_DIM, TILE_GROUPS>(ext, tg, s_inv, vabs_mp, live_g, vw,
                                                                   lane + WARP_SIZE * s, w);
                        if (sweep2)
                            tile_v_block_quantise<HEAD_DIM, TILE_GROUPS>(w, tg, s_inv, vabs_mp, s_v8t, vw,
                                                                         lane + WARP_SIZE * s);
                    }
                }
                __syncthreads();
            }
        }

        // The header prefetch has landed by now; wait so the mid barrier
        // publishes it (the K staging's copies were waited for in the
        // decode, by the warp that reads them).
        if (tid == 0) cp_async_wait<0, true>();
        __syncwarp();

        // The group's K rows are stored: the V warps mark them published
        // without waiting (the softmax warps wait below, after their own
        // staging work, so a slow group's decode overlaps the staging).
        if (warp >= TILE_SM_WARPS)
            asm volatile("bar.arrive %0, %1;" :: "n"(TILE_K_BARRIER), "n"(TILE_THREADS) : "memory");

        // Reconverge before the split: a warp that arrives here diverged
        // stays diverged through the whole V side, which has no full-warp
        // barrier before the tile's last one, and issues every V instruction
        // once per fragment (measured: 16 active lanes per V-side
        // instruction, the phase at twice its instruction count, every
        // shuffle through the collective slow path). The softmax warps
        // reconverge at their K barrier below.
        __syncwarp();
        // The next tile's K staging, decided at the commit; this tile's own
        // was read by its K staging above.
        k_stg_esz = k_stg_next;
        // The next tile's ninth quad, if it has one, into the buffer the
        // commit filled (tid 0 held the next tile's quad count).
        if (warp == 0) stage_slot8(t + 1, nxt, ngroups_next, true);

        // The tile's middle splits the block two ways: warps 0..3 run QK
        // and the softmax, warps 4..7 read and quantise V at the same
        // time, and the two halves meet at the tile's last barrier. Neither
        // half holds anything of the other's work, so each runs at its own
        // register footprint, and V's load latency is paid under the MMAs
        // and exp2s of the other half rather than on anyone's critical
        // path.
        //
        // QK: warp ns owns the tile's columns ns·8..+7 — N_WIN m16n8k32
        // MMAs over the whole head, the Q A-fragments and both rows'
        // window scales re-read from the Q slab each window (one ldmatrix
        // per window per operand). The accumulator lands in the lane's own
        // softmax cells: s[row][i] is (row g + 8·row, column ns·8 + n0 +
        // i), so no score leaves the warp and no barrier sits between QK
        // and the softmax.
        //
        // Softmax: a masked column is -inf. The row max crosses the four
        // warps through s_rmax under the softmax warps' own named barrier;
        // each warp then exponentiates only its two columns per row, packs
        // them into the int8 P slab, and posts its partial row sum. Warp 0
        // posts the row's rescale factor for the PV warps. The running max
        // is clamped to -1e30 — below any logit, above -inf — so a masked
        // column's exp2(-inf − m) is exactly 0 and a not-yet-started row's
        // exp2(m_run − m_new) likewise, with no guard on either; the
        // null-split test is then "no row sum was ever added" (a live
        // column contributes ≥ 1).
        if (warp < TILE_SM_WARPS) {
            // Wait for every group's K rows (the V warps arrived above).
            asm volatile("bar.sync %0, %1;" :: "n"(TILE_K_BARRIER), "n"(TILE_THREADS) : "memory");
            float s[2][2] = { { 0.f, 0.f }, { 0.f, 0.f } };
            #pragma unroll
            for (int wa = 0; wa < N_WIN; ++wa) {
                uint32_t q_frag[4], b[2];
                ldmatrix_x4_b16(q_frag, sw_a_frag_addr<HEAD_DIM>(s_q8, 32 * wa, lane));
                ldmatrix_x2_b16(b, sw_b_frag_addr<HEAD_DIM>(s_k8, ns * 8, 32 * wa, lane));
                const float qs0 = __half2float(s_q_scale[g][wa]);
                const float qs1 = __half2float(s_q_scale[g + 8][wa]);
                const float ks0 = __half2float(s_k_scale[ns * 8 + n0][wa]);
                const float ks1 = __half2float(s_k_scale[ns * 8 + n0 + 1][wa]);
                int32_t c_i[4] = { 0, 0, 0, 0 };
                int32_t d_i[4];
                mma_int8_m16n8k32(d_i, q_frag, b, c_i);
                s[0][0] += (float)d_i[0] * qs0 * ks0;
                s[0][1] += (float)d_i[1] * qs0 * ks1;
                s[1][0] += (float)d_i[2] * qs1 * ks0;
                s[1][1] += (float)d_i[3] * qs1 * ks1;
            }
            // The window's live columns. A multi-pass tile's earlier passes
            // left theirs in s_colmask; this pass's groups add their own.
            uint32_t mask = s_colmask;
            #pragma unroll
            for (int gi = 0; gi < TILE_SLOTS; ++gi)
                mask |= grp_cols(grp_mask(tg.desc[gi]), (int)tg.col0[gi]);
            const int c = ns * 8 + n0;
            const bool live0 = (mask >> c) & 1u;
            const bool live1 = (mask >> (c + 1)) & 1u;
            #pragma unroll
            for (int row = 0; row < 2; ++row) {
                const int r = g + row * 8;
                s[row][0] = live0 ? s[row][0] : -INFINITY;
                s[row][1] = live1 ? s[row][1] : -INFINITY;
                float mt = fmaxf(s[row][0], s[row][1]);
                mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 1));
                mt = fmaxf(mt, __shfl_xor_sync(0xffffffffu, mt, 2));
                if ((lane & 3) == 0) s_rmax[r][ns] = mt;
            }
            asm volatile("bar.sync %0, %1;" :: "n"(TILE_SM_BARRIER), "n"(TILE_SM_WARPS * WARP_SIZE) : "memory");
            #pragma unroll
            for (int row = 0; row < 2; ++row) {
                const int r = g + row * 8;
                const float4 rm = *(const float4*)&s_rmax[r][0];
                const float m_tile = fmaxf(fmaxf(rm.x, rm.y), fmaxf(rm.z, rm.w));
                const float m_new = fmaxf(fmaxf(m_run[row], m_tile), -1e30f);
                const float alpha = exp2f(m_run[row] - m_new);
                const float p0 = exp2f(s[row][0] - m_new);
                const float p1 = exp2f(s[row][1] - m_new);
                float ls = p0 + p1;
                ls += __shfl_xor_sync(0xffffffffu, ls, 1);
                ls += __shfl_xor_sync(0xffffffffu, ls, 2);
                *(uint16_t*)(s_p8 + sw_off<TILE_TOK>(r, c)) = (uint16_t)i8_pack2(p0, p1, 127.f);
                if ((lane & 3) == 0) {
                    s_lsum[r][ns] = ls;
                    if (warp == 0) s_alpha[r] = alpha;
                }
                m_run[row] = m_new;
            }
        } else {
            // V: warp vw = warp − 4 owns the head's 32-dim windows
            // vw·V_WPW .. +V_WPW−1 (two of the eight at HEAD_DIM 256; a
            // window past N_WIN is a whole warp's no-op). Within a window,
            // lane = (token quarter, quad): the lane holds dims win·32 +
            // 4·(lane&7) .. +3 of groups 2·(lane>>3) and +1 — the same
            // quad across two groups, so a window's eight loads go out
            // under one dispatch.
            //
            // Quantise: the PV MMA takes one scale per dim for all 32
            // columns, and the groups come from different arena blocks
            // with different block scales, so each dim's absmax over the
            // tile is the lane's 8 tokens joined across the four
            // quarter-lanes (xor 8, 16); every lane then packs its quad
            // for each of its two groups as one word of the transposed
            // slab, and the quarter-0 lanes write the four dims' scales as
            // one vector. A dead group's quad reads as zeros and masked
            // tokens inside a live group quantise to 0. Either way P = 0
            // for them. The reduction is warp-wide, so it runs after the
            // per-lane decode paths have reconverged.
            const int vw = warp - TILE_SM_WARPS;
            const int vq = lane >> 3;
            const int vg0 = 2 * vq;
            const int vgi[2] = { vg0, vg0 + 1 };
            const auto win_d0 = [&](int h) { return (vw * V_WPW + h) * 32 + (lane & 7) * 4; };
            // Every live group of the tile at its slot's columns 4·g — the
            // packed layout, and every window without a hole inside it. One
            // vote per tile (lane g judges group g) picks `v_quantise`'s
            // store path below.
            bool lane_slotted = true;
            if (lane < TILE_SLOTS) {
                const uint32_t lm = grp_mask(tg.desc[lane]);
                lane_slotted = lm == 0u || (lm == 15u && (int)tg.col0[lane] == lane * GROUP_TOK);
            }
            const bool slotted = __all_sync(0xffffffffu, lane_slotted);
            __syncwarp();
            // `v8` is this lane-octet's token of the ninth quad (zero when the
            // slot is dead or the token masked, so an eight-quad window's
            // arithmetic is untouched); `tok8` says it is live and stores.
            // `ninth` (a std::bool_constant) selects the nine-slot body: an
            // eight-quad tile's instantiation never touches v8 or tok8.
            const auto v_quantise = [&](auto ninth, const float (&v)[2][4][GROUP_TOK],
                                        const float (&v8)[4], bool tok8, int vd0) {
                constexpr bool NINTH = decltype(ninth)::value;
                float scale[4], inv[4];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    float a = 0.f;
                    #pragma unroll
                    for (int gg = 0; gg < 2; ++gg)
                        #pragma unroll
                        for (int j = 0; j < GROUP_TOK; ++j) a = fmaxf(a, fabsf(v[gg][k][j]));
                    if constexpr (NINTH) a = fmaxf(a, fabsf(v8[k]));
                    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 8));
                    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 16));
                    scale[k] = a / 127.f;
                    inv[k] = (scale[k] > 0.f) ? 1.f / scale[k] : 0.f;
                }
                // A group's four tokens go to its own columns. A whole,
                // 4-aligned group is one word per dim row — one swizzled
                // address, the four rows at immediate offsets (vd0 is a
                // multiple of 4, so they share a swizzle key). A group that is
                // partial or off the 4-grid — a quad at either side of a hole,
                // or at the window's edge — stores its live tokens a byte at a
                // time: its word would cover a neighbour's columns. A dead
                // group has no columns and stores nothing.
                //
                // The eight words are quantised ONCE, before any branch —
                // the paths below differ only in where they store. (Written
                // per branch, NVCC materialised the quantisation twice on the
                // taken path.)
                uint32_t w[2][4];
                #pragma unroll
                for (int gg = 0; gg < 2; ++gg)
                    #pragma unroll
                    for (int k = 0; k < 4; ++k) w[gg][k] = i8_pack4(v[gg][k], inv[k]);
                // The ninth quad's token: a byte per dim row at its own
                // column (its quad straddles the window's edge, so it is
                // never a slotted word). The same code as the packed bytes.
                if constexpr (NINTH) {
                    if (tok8) {
                        const int c8 = (int)tg.col0[TILE_GROUPS] + vq;
                        #pragma unroll
                        for (int k = 0; k < 4; ++k)
                            s_v8t[sw_off<TILE_TOK>(vd0 + k, c8)] = i8_quant(v8[k], inv[k]);
                    }
                }
                // `slotted` (a vote per tile, below) says every live group of
                // the tile sits at its slot's columns 4·g — the packed layout,
                // and every window without a hole inside it. That is the
                // straight-line path: one address per lane and eight words,
                // no branch per group; a dead group's words land in columns
                // the mask retires, as they always have.
                if (slotted) {
                    int8_t* vrow = s_v8t + sw_off<TILE_TOK>(vd0, vg0 * GROUP_TOK);
                    #pragma unroll
                    for (int gg = 0; gg < 2; ++gg)
                        #pragma unroll
                        for (int k = 0; k < 4; ++k)
                            *(uint32_t*)(vrow + k * TILE_TOK + gg * GROUP_TOK) = w[gg][k];
                } else {
                    #pragma unroll
                    for (int gg = 0; gg < 2; ++gg) {
                        const int g = vg0 + gg;
                        const uint32_t lv = grp_mask(tg.desc[g]);
                        if (lv == 0u) continue;
                        const int c0 = (int)tg.col0[g];
                        if (lv == 15u && (c0 & 3) == 0) {
                            int8_t* vrow = s_v8t + sw_off<TILE_TOK>(vd0, c0);
                            #pragma unroll
                            for (int k = 0; k < 4; ++k)
                                *(uint32_t*)(vrow + k * TILE_TOK) = w[gg][k];
                        } else {
                            // The live tokens' bytes out of the packed word
                            // (token j is byte j): its word would cover a
                            // neighbour's columns.
                            #pragma unroll
                            for (int j = 0; j < GROUP_TOK; ++j) {
                                if (!((lv >> j) & 1u)) continue;
                                #pragma unroll
                                for (int k = 0; k < 4; ++k)
                                    s_v8t[sw_off<TILE_TOK>(vd0 + k, c0 + j)] =
                                        (int8_t)((w[gg][k] >> (8 * j)) & 0xffu);
                            }
                        }
                    }
                }
                if (vq == 0) {
                    const __half2 s01 = __floats2half2_rn(scale[0], scale[1]);
                    const __half2 s23 = __floats2half2_rn(scale[2], scale[3]);
                    *(uint2*)&s_v_scale[vd0] = make_uint2(
                        *reinterpret_cast<const uint32_t*>(&s01),
                        *reinterpret_cast<const uint32_t*>(&s23));
                }
            };
            // The tile's V absmax buffer; the other one, last read by the
            // previous tile's quantise, is cleared here for the next tile.
            const int vt = tid - TILE_SM_WARPS * WARP_SIZE;
            int* vabs = s_vabs + (tile_no & 1) * HEAD_DIM;
            int* vabs_next = s_vabs + ((tile_no & 1) ^ 1) * HEAD_DIM;
            #pragma unroll
            for (int d = vt; d < HEAD_DIM; d += TILE_V_WARPS * WARP_SIZE) vabs_next[d] = 0;
            // The path is decided for the whole V side at once, from the
            // tile's live groups: every live group's V map a band map and
            // every palette a narrow dtype, and the windows take the
            // vector path — a vote per warp could split the head between
            // the two paths' different partitions of it (windows against
            // slots) and write a dim twice or never. Lane g judges group g
            // and the warp ballots, so the decision is a dozen instructions
            // rather than eight groups' worth per lane.
            static_assert(TILE_GROUPS <= WARP_SIZE, "a lane per group");
            // The Q0_V read-through (`tile_v_q0v_readthrough`) is judged in
            // the same vote: every live group Q0_V on all four V palettes and
            // in the first live group's slice, so one block serves each dim.
            const int g_first = __ffs(__ballot_sync(0xffffffffu,
                                                    lane < TILE_SLOTS && grp_mask(tg.desc[lane]) != 0u)) - 1;
            bool lane_live = false, lane_vec = true, lane_q0v = true;
            if (lane < TILE_SLOTS) {
                lane_live = grp_mask(tg.desc[lane]) != 0u;
                const uint32_t f4 = *reinterpret_cast<const uint32_t*>(&ext.fmt[lane][1][0]);
                bool narrow = s_band[lane][1] != 0;
                #pragma unroll
                for (int p = 0; p < N_PALETTE; ++p)
                    narrow = narrow && i8_is_narrow_dtype_format((f4 >> (8 * p)) & 0xffu);
                lane_vec = !lane_live || narrow;
                lane_q0v = !lane_live
                           || (f4 == (uint32_t)ArenaFormat::Q0_V * 0x01010101u
                               && grp_slice(tg.desc[lane]) == grp_slice(tg.desc[g_first]));
            }
            const uint32_t live_g = __ballot_sync(0xffffffffu, lane_live);
            // A multi-pass tile takes the block path whatever its formats:
            // its V scale is the window's absmax, folded across the passes
            // through the absmax buffer, and only the block path scales
            // through that buffer. (The vote runs first — it is collective.)
            const bool vec_all = __all_sync(0xffffffffu, lane_vec);
            const bool vec = vec_all && n_pass == 1;
            const bool q0v = __all_sync(0xffffffffu, lane_q0v) && live_g != 0u && n_pass == 1;
            if (vt == 0) s_v_affine = q0v ? 1 : 0;
            __syncwarp();
            if (q0v) {
                tile_v_q0v_readthrough<HEAD_DIM>(ext, tg, s_inv, vabs, s_v_scale, g_first, vw, lane, s_v8t);
            } else if (vec) {
                // The windows are software-pipelined one deep: window
                // h+1's loads go out once window h's words are converted
                // and before its quantise, so at most one window's words
                // and one window's floats are live at once while the next
                // window's latency runs under the quantise.
                QuadInfo q[V_WPW][2];
                #pragma unroll
                for (int h = 0; h < V_WPW; ++h) {
                    if (vw * V_WPW + h < N_WIN) {
                        #pragma unroll
                        for (int i = 0; i < 2; ++i)
                            q[h][i] = tile_quad_info<HEAD_DIM>(ext, tg, s_tbl, 1, vgi[i], win_d0(h));
                    }
                }
                QuadRaw<2> raw;
                if (vw * V_WPW < N_WIN) tile_quads_vec_load<HEAD_DIM>(ext, q[0], 1, vgi, raw);
                // Two instantiations of the window loop under the tile's
                // block-uniform ninth-slot flag: a nine-quad tile reads this
                // lane-octet's token of the ninth quad (token vq of slot
                // TILE_GROUPS) each window; an eight-quad tile runs the loop
                // with no trace of it. Within each, the ninth-token array is
                // written on every path (under a runtime predicate it would
                // go to local memory for the whole kernel — 1.5 KB of spills
                // at the gate).
                if (has8) {
                    const bool tok8 = ((grp_mask(tg.desc[TILE_GROUPS]) >> vq) & 1u) != 0u;
                    #pragma unroll
                    for (int h = 0; h < V_WPW; ++h) {
                        if (vw * V_WPW + h < N_WIN) {
                            float v8[4];
                            tile_quad_vec_token<HEAD_DIM, 2>(ext, tg, s_tbl, 1, TILE_GROUPS,
                                                             win_d0(h), vq, v8);
                            float v[2][4][GROUP_TOK];
                            tile_quads_vec_cvt(raw, 1, v);
                            if (h + 1 < V_WPW && vw * V_WPW + h + 1 < N_WIN)
                                tile_quads_vec_load<HEAD_DIM>(ext, q[h + 1], 1, vgi, raw);
                            v_quantise(std::true_type{}, v, v8, tok8, win_d0(h));
                        }
                    }
                } else {
                    #pragma unroll
                    for (int h = 0; h < V_WPW; ++h) {
                        if (vw * V_WPW + h < N_WIN) {
                            float v[2][4][GROUP_TOK];
                            tile_quads_vec_cvt(raw, 1, v);
                            if (h + 1 < V_WPW && vw * V_WPW + h + 1 < N_WIN)
                                tile_quads_vec_load<HEAD_DIM>(ext, q[h + 1], 1, vgi, raw);
                            const float v8[4] = { 0.f, 0.f, 0.f, 0.f };
                            v_quantise(std::false_type{}, v, v8, false, win_d0(h));
                        }
                    }
                }
            } else {
                // Block path (`tile_v_block_path`): the nine-slot or the
                // eight-slot instantiation under the tile's uniform flag.
                static_assert(TILE_V_WARPS == N_PALETTE, "a V warp per palette");
                static_assert(HEAD_DIM / N_PALETTE == 2 * WARP_SIZE, "two ranks per lane");
                if (has8) tile_v_block_path<HEAD_DIM, TILE_SLOTS>(ext, tg, s_inv, vabs, live_g, vw, lane, s_v8t);
                else      tile_v_block_path<HEAD_DIM, TILE_GROUPS>(ext, tg, s_inv, vabs, live_g, vw, lane, s_v8t);
                #pragma unroll
                // `a / 127`, as the vector path stores it — not `a · (1/127)`,
                // which rounds differently and would give a multi-pass tile a
                // different PV scale from its single-pass equivalent.
                for (int d = vt; d < HEAD_DIM; d += TILE_V_WARPS * WARP_SIZE)
                    s_v_scale[d] = __float2half(__int_as_float(vabs[d]) / 127.f);
            }
        }
        // Publishes the P and V slabs (and the next tile's descriptors,
        // committed above). Nothing after it touches a slab the next tile
        // writes before its own first barrier, so this is the tile's last
        // barrier.
        __syncthreads();

        // PV: one m16n8k32 per output-dim slice (k = the tile's 32 tokens).
        // The P A-fragment comes from the slab (one ldmatrix), the row
        // rescale factors and the four partial row sums from the softmax
        // warps' posts.
        uint32_t pa[4];
        ldmatrix_x4_b16(pa, sw_a_frag_addr<TILE_TOK>(s_p8, 0, lane));
        float alpha[2];
        float psum[2];   // the tile's Σp per row
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            const int r = g + row * 8;
            alpha[row] = s_alpha[r];
            const float4 l4 = *(const float4*)&s_lsum[r][0];
            psum[row] = (l4.x + l4.y) + (l4.z + l4.w);
            l_run[row] = l_run[row] * alpha[row] + psum[row];
        }
        // A Q0_V read-through tile's V is S·b + C per dim: the MMA carries
        // S·b, and C · Σp is added here.
        const bool v_affine = s_v_affine != 0;
        const int* v_cent = s_vabs + (tile_no & 1) * HEAD_DIM;
        #pragma unroll
        for (int sl8 = 0; sl8 < PV_H; ++sl8) {
            const int dim = dim_base + sl8 * 8;
            uint32_t vb[2];
            ldmatrix_x2_b16(vb, sw_b_frag_addr<TILE_TOK>(s_v8t, dim, 0, lane));
            const float vs0 = __half2float(s_v_scale[dim + n0]) * (1.f / 127.f);
            const float vs1 = __half2float(s_v_scale[dim + n0 + 1]) * (1.f / 127.f);
            int32_t c_i[4] = { 0, 0, 0, 0 };
            int32_t d_i[4];
            mma_int8_m16n8k32(d_i, pa, vb, c_i);
            float t0 = (float)d_i[0] * vs0, t1 = (float)d_i[1] * vs1;
            float t2 = (float)d_i[2] * vs0, t3 = (float)d_i[3] * vs1;
            if (v_affine) {
                const float c0 = __int_as_float(v_cent[dim + n0]);
                const float c1 = __int_as_float(v_cent[dim + n0 + 1]);
                t0 += c0 * psum[0];
                t1 += c1 * psum[0];
                t2 += c0 * psum[1];
                t3 += c1 * psum[1];
            }
            o_acc[sl8][0] = o_acc[sl8][0] * alpha[0] + t0;
            o_acc[sl8][1] = o_acc[sl8][1] * alpha[0] + t1;
            o_acc[sl8][2] = o_acc[sl8][2] * alpha[1] + t2;
            o_acc[sl8][3] = o_acc[sl8][3] * alpha[1] + t3;
        }
    }
    if (!qsa_on || chunk_lo >= e_hi) break;
    // Next sparse chunk: warp 7 resolves it and stages its tile 0, exposed;
    // one barrier publishes both. The last tile's post-softmax barrier
    // orders every thread's reads of s_grp and the descriptor buffers
    // ahead of the rewrite (PV reads only the P and V slabs).
    {
        const int chunk_hi = min(e_hi, chunk_lo + EPS);
        if (warp == STAGE_WARP) {
            const bool has = chunk_lo + lane < chunk_hi;
            const uint32_t ent = has ? sel_entries[chunk_lo + lane] : 0u;
            tile_resolve_entries<HEAD_DIM>(ent, has, sel, slot_idx, slices_ptr,
                                           n_slices, write_slice_idx, n_kv_head, lane,
                                           s_grp, s_grp_src, &s_n_tiles);
            __syncwarp();
            n_tiles = s_n_tiles;
            stage_tile0();
        }
        chunk_lo = chunk_hi;
    }
    __syncthreads();
    n_tiles = s_n_tiles;
    }
    // No tile carried a live token (every selected cell fell outside the
    // slot, or the dense range was all sealed holes): this split is null.
    // Every warp reads the same row sums, so the test is block-uniform.
    if (l_run[0] == 0.f) { emit_null(); return; }

    // ------------------------------------------------------------------
    // Epilogue: this split's un-normalised (ΣpV, m, l) per live row. The
    // softmax warps hold m (log2 units), every warp l; warp 0's quad
    // leaders write both, m back in natural-log units for the combine.
    // ------------------------------------------------------------------
    #pragma unroll
    for (int row = 0; row < 2; ++row) {
        const int r = g + row * 8;
        if (r >= hpg) continue;
        const int64_t base =
            ((int64_t)slot_idx * n_q_head + first_q_head + r) * num_splits + split_idx;
        float* acc = partial_acc + base * HEAD_DIM;
        #pragma unroll
        for (int sl8 = 0; sl8 < PV_H; ++sl8) {
            const int dim = dim_base + sl8 * 8 + n0;
            *(float2*)&acc[dim] = make_float2(o_acc[sl8][row * 2], o_acc[sl8][row * 2 + 1]);
        }
        if (warp == 0 && (lane & 3) == 0) {
            partial_ml[base * 2] = m_run[row] * TILE_LN2;
            partial_ml[base * 2 + 1] = l_run[row];
        }
    }
}
