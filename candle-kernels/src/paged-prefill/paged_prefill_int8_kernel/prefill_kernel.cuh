// The INT8 prefix-attention prefill kernel.
//
// Holds the `__launch_bounds__` kernel that runs causal prefill attention over
// a palette-quantized paged KV prefix, with INT8 m16n8k32 tensor-core MMA for
// both Q·Kᵀ and P·V.
// It is one 44 KB function, so it stays whole.

// ============================================================================
// The kernel
// ============================================================================

// minBlocks pins the register budget: 4 blocks × 256 threads at 64
// regs/thread through HEAD_DIM 128 (the max-occupancy configuration, 67%
// theoretical; 6 blocks would need ≤42 regs, unreachable past o_acc's 32),
// 2 blocks × 128 regs at 256. The Q-fragment drain, the union smem arena,
// and the deliberately register-lean staging (recompute lambdas,
// smem-resident Q scales, two-pass V requant, serialized palette loops)
// are what make the budget close with only a small residual spill.
//
// `PRESTAGED` is the staging policy: false reads every column from its
// source; true takes a staged sequence's four-aligned column quads from
// `stage` (see the header).
template <typename QT, int HEAD_DIM, bool PRESTAGED>
__global__ void __launch_bounds__(I8_THREADS, i8_min_blocks(HEAD_DIM))
paged_prefill_int8_kernel(
    const QT* __restrict__ q,          // [total_q, n_head, HD] packed, unrotated
    const QT* __restrict__ k_packed,   // [total_q, n_kv_head, HD] packed, unrotated
    const QT* __restrict__ v_packed,   // [total_q, n_kv_head, HD]
    const uint8_t* __restrict__ headers_ptr,
    const uint32_t* __restrict__ cu_seqlens_q,
    const uint32_t* __restrict__ q_lens,
    const uint32_t* __restrict__ kv_lens,
    QT* __restrict__ out,              // [total_q, n_head, HD]
    int batch_size,
    int n_head,
    int n_kv_head,
    float softmax_scale,
    const RopeRungs rungs,
    int rope_interleaved,               // 0 = half-split pairing, 1 = interleaved (LLaMA)
    // Split-KV: grid.z = batch_size × num_splits. Shard s of a sequence
    // processes tiles (sealed AND fresh — one shared ordinal space) with
    // ordinal ≡ s (mod num_splits). num_splits == 1 stores O directly;
    // otherwise each shard emits
    // an un-normalized (ΣpV, m, l) partial into `partials`
    // [total_q·n_head rows][num_splits][HEAD_DIM + 2] and the combine kernel
    // merges them (base-e log-sum-exp).
    int num_splits,
    float* __restrict__ partials,
    QsaSel sel,
    const PrefillKvStage stage,
    const PrefillCut cut
) {
    static_assert(HEAD_DIM % 64 == 0 && HEAD_DIM >= 64 && HEAD_DIM <= 256,
                  "int8 prefill: HEAD_DIM must be a multiple of 64 in [64, 256]");
    constexpr int N_WIN = HEAD_DIM / 32;       // QK k-step windows (also dims/lane)
    constexpr int PV_SLICES = HEAD_DIM / 8;    // PV n-slices (output dims per mma)
    constexpr int SUB = HEAD_DIM / N_PALETTE;  // palette band width
    // A lane's rank bytes pack (palette, rank) as p<<6 | rank.
    static_assert(N_PALETTE == 4, "rank byte packs the palette into 2 bits");
    static_assert(SUB <= 64, "rank needs 6 bits");

    constexpr int DIM_SPLIT = i8_dim_split(HEAD_DIM);
    constexpr int I8_ROW_TILES = i8_row_tiles(HEAD_DIM);
    constexpr int I8_M_ROWS = i8_m_rows(HEAD_DIM);
    static_assert(I8_ROW_TILES * DIM_SPLIT == I8_WARPS,
                  "warps must partition exactly into (row-tile, dim-part)");

    const int tid = (int)threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int row_tile = warp / DIM_SPLIT; // which m16 row-tile this warp serves
    const int dim_part = warp % DIM_SPLIT; // which output-dim slice it accumulates
    const int batch_idx = (int)blockIdx.z / num_splits;
    const int split_idx = (int)blockIdx.z % num_splits;
    const int kv_head_idx = (int)blockIdx.y;
    if (batch_idx >= batch_size || kv_head_idx >= n_kv_head) return;

    const SlotHeader& slot_hdr = get_slot_header(headers_ptr, batch_idx);
    // The sequence's own rung: a block never spans two sequences, so nothing
    // rung-dependent here is shared with another sequence's rows.
    const RopeView rope = rope_view(rungs, slot_hdr.rope_rung);

    const int q_start = (int)cu_seqlens_q[batch_idx];
    const int q_len = (int)q_lens[batch_idx];
    const int kv_len = (int)kv_lens[batch_idx];
    int prefix_len = kv_len - q_len;
    if (prefix_len < 0) prefix_len = 0;

    int hpg = n_head / n_kv_head;
    if (hpg <= 0) hpg = 1;
    if (hpg > I8_M_ROWS) return; // unsupported (production hpg = 8)
    const int block_m_tok = I8_M_ROWS / hpg; // tokens covered per block
    const int rows_used = block_m_tok * hpg; // ≤ I8_M_ROWS; rows beyond are idle
    const int t0 = (cut.blk_lo + (int)blockIdx.x) * block_m_tok;
    if (t0 >= q_len) return;
    const int first_q_head = kv_head_idx * hpg;

    // The block's causal horizon: no row it serves attends at or past this
    // position. A dense row's walk would otherwise propose every position up
    // to `kv_len` — in a prompt prefilled from zero, four tiles in five of a
    // dense block's walk — each staged, multiplied and then masked to nothing.
    // A tile is skipped only when its FIRST block starts past the horizon: a
    // tile straddling it is kept whole, because the V requant's per-dim scale
    // is taken over every column of the tile, masked or not, and cutting the
    // tile short would move it.
    const int blk_horizon = min(kv_len, prefix_len + min(t0 + block_m_tok, q_len));

    // Whether this block's sequence was pre-staged, and where its rows start
    // in each head's run of the stage. Block-uniform.
    //
    // A staged block attends this launch's key window: tiles whose first block
    // starts in `[stage.chunk_lo, stage.chunk_hi)`, so it stops at the window's
    // end exactly as at its horizon (`tile_stop`). It has nothing to do in a
    // window that starts at or past its horizon — an earlier window finalised
    // it. An unstaged block reads every column from its source and runs whole
    // in its row group's first window.
    bool staged = false;
    int64_t stage_base = 0;
    int tile_stop = blk_horizon;
    if constexpr (PRESTAGED) {
        staged = q_len >= stage.min_q_len;
        if (staged) {
            if (blk_horizon <= stage.chunk_lo) return;
            // The last window reaches every block's horizon; a block that
            // would carry past it would never be finalised.
            if (cut.last_chunk && blk_horizon > stage.chunk_hi) __trap();
            tile_stop = min(blk_horizon, stage.chunk_hi);
            stage_base = kv_stage_seq_base(q_lens, kv_lens, batch_idx, stage);
        } else if (stage.chunk_lo > 0) {
            return;
        }
    }

    // ------------------------------------------------------------------
    // Shared memory: ONE union arena, sized for 4 blocks/SM (the 25.6 KB
    // per-block budget at the 102 KB SM limit — the max-occupancy target).
    //
    // The arena has two overlays with disjoint lifetimes:
    //   PROLOGUE overlay (block start only): s_q8 + s_q_scale. Q is staged
    //     here once, then DRAINED TO REGISTERS (the q8-matmul trick at
    //     block scope: Q is constant across the tile loop, and a warp only
    //     ever reads its own 16 rows as N_WIN ldmatrix A-fragments = 16
    //     registers + scales). After the drain barrier the whole region is
    //     dead and the tile overlay reuses its bytes.
    //   TILE overlay (per tile): the staging→compute handoff slabs
    //     (s_k8/s_v8t + scales) and the fresh∪p8 scratch. The slabs span
    //     the staging barrier so they cannot union among themselves, but
    //     all of them may alias the dead prologue.
    //
    // +16-byte row pads on the MMA slabs (the q8-matmul KI8_STRIDE
    // convention): a multiple of 16 keeps every row address ldmatrix-legal,
    // while NOT being a multiple of 128 rotates the 8 tile rows across
    // banks. Scale tables are FP16 (max-abs magnitudes; ~0.05% error is
    // noise under int8's 0.4%).
    // ------------------------------------------------------------------
    constexpr int Q8_LD = HEAD_DIM + 16;
    constexpr int V8T_LD = I8_TILE_TOK + 16;
    constexpr int P8_BYTES = I8_WARPS * 16 * V8T_LD;
    // FP16 V stash: the tile's 32 columns in natural dim order, read back
    // per dim by the requant pass.
    constexpr int FRESH_BYTES = I8_TILE_TOK * HEAD_DIM * 2;
    constexpr int SCRATCH_BYTES = (P8_BYTES > FRESH_BYTES) ? P8_BYTES : FRESH_BYTES;

    constexpr int ALIGN16 = 15;
    // Tile overlay offsets (all 16-aligned).
    constexpr int OFF_K8 = 0;
    constexpr int OFF_KS = (OFF_K8 + I8_TILE_TOK * Q8_LD + ALIGN16) & ~ALIGN16;
    constexpr int OFF_V8T = (OFF_KS + I8_TILE_TOK * N_WIN * 2 + ALIGN16) & ~ALIGN16;
    constexpr int OFF_VS = (OFF_V8T + HEAD_DIM * V8T_LD + ALIGN16) & ~ALIGN16;
    constexpr int OFF_SCR = (OFF_VS + HEAD_DIM * 2 + ALIGN16) & ~ALIGN16;
    constexpr int TILE_BYTES = OFF_SCR + SCRATCH_BYTES;
    // Prologue overlay (s_q8 only — the Q scales are RESIDENT, below).
    constexpr int PRO_BYTES = I8_M_ROWS * Q8_LD;
    constexpr int ARENA_BYTES = (TILE_BYTES > PRO_BYTES) ? TILE_BYTES : PRO_BYTES;
    // The resident statics beside the arena: Q scales, per-warp slice
    // bindings (palette metadata, rank tables and the map words they were
    // ranked under), per-warp column origins.
    constexpr int RESIDENT_BYTES = I8_M_ROWS * N_WIN * 2
                                 + I8_WARPS * (int)sizeof(WarpPalette<HEAD_DIM>)
                                 + I8_WARPS * 4;
    static_assert(ARENA_BYTES + RESIDENT_BYTES <= i8_smem_budget(HEAD_DIM),
                  "arena + residents must fit the target-residency smem budget");

    __shared__ __align__(16) uint8_t s_arena[ARENA_BYTES];
    // Q scales stay RESIDENT in smem (512 B of the 4-block headroom): the
    // QK fixup reads them as broadcasts, and NOT draining them to
    // registers hands ptxas 4 regs/thread of slack at the 64-reg cap —
    // measured spill traffic was ~25% of global sector volume.
    __shared__ __half s_q_scale[I8_M_ROWS][N_WIN];
    // Per-WARP binding of the slice the warp's current column lives in —
    // palette metadata and rank tables (`WarpPalette`). A warp rebinds it
    // only when its column moves to another slice.
    __shared__ WarpPalette<HEAD_DIM> s_wpal[I8_WARPS];
    // Logical kv position of each warp's first column (column 4w + c sits
    // at s_qpos[w] + c); a warp with no block in the tile publishes a
    // position past every horizon so its columns mask out.
    __shared__ int s_qpos[I8_WARPS];
    constexpr int DEAD_QPOS = 0x7fff0000;

    // Prologue view (dead after the Q drain barrier).
    auto s_q8 = reinterpret_cast<int8_t(*)[Q8_LD]>(s_arena);
    // Tile views (alias the prologue bytes — valid only after the drain).
    auto s_k8 = reinterpret_cast<int8_t(*)[Q8_LD]>(s_arena + OFF_K8);
    auto s_k_scale = reinterpret_cast<__half(*)[N_WIN]>(s_arena + OFF_KS);
    auto s_v8t = reinterpret_cast<int8_t(*)[V8T_LD]>(s_arena + OFF_V8T);
    auto s_v_scale = reinterpret_cast<__half*>(s_arena + OFF_VS);

    // Scratch tenant views (inside the tile overlay). Temporally disjoint:
    //   s_fresh — FP16 V stash for the per-dim requant, STAGING only.
    //   s_p8    — per-warp quantized P tiles, COMPUTE phase only.
    auto s_fresh = reinterpret_cast<__half(*)[HEAD_DIM]>(s_arena + OFF_SCR);
    auto s_p8 = reinterpret_cast<int8_t(*)[16][V8T_LD]>(s_arena + OFF_SCR);

    // ------------------------------------------------------------------
    // Row → (token, head) mapping and per-thread fragment rows.
    // Thread's fragment rows within its warp tile: g = lane>>2 and g+8.
    // ------------------------------------------------------------------
    const int g = lane >> 2;
    const int n0 = (lane & 3) * 2;
    // Row-derived values (token, head, liveness, causal horizon) are
    // recomputed on demand: at the 64-register 4-blocks/SM cap, holding
    // them in arrays is ~10 across-loop registers — guaranteed hot spills.
    auto row_of = [&](int i) { return row_tile * 16 + g + i * 8; };
    auto row_tok = [&](int i) { return t0 + row_of(i) / hpg; };
    auto row_head = [&](int i) {
        int r = row_of(i);
        return first_q_head + (r - (r / hpg) * hpg);
    };
    auto row_live = [&](int i) {
        return (row_of(i) < rows_used) && (row_tok(i) < q_len);
    };

    // ------------------------------------------------------------------
    // Q staging: load, RoPE, per-window int8 quantize (natural dim order).
    // Lane holds dims {lane + 32w : w in 0..N_WIN}; RoPE pair (d, d+HALF)
    // is (w, w + N_WIN/2) — in-thread for HEAD_DIM % 64 == 0.
    // ------------------------------------------------------------------
    //
    // A warp's rows are straight-line code, every row's reads in flight at
    // once: a row past the block's run reads row 0 of the launch (always
    // there) and is zeroed, so no branch stands between one row's reads and
    // the next's. A zero row rotates and quantises to zero codes and a zero
    // scale, exactly what the row would hold unrotated.
    static_assert(I8_M_ROWS % I8_WARPS == 0, "every warp stages the same number of Q rows");
    #pragma unroll
    for (int ri = 0; ri < I8_M_ROWS / I8_WARPS; ++ri) {
        const int r = warp + ri * I8_WARPS;
        int tl = r / hpg;
        int tok = t0 + tl;
        int head = first_q_head + (r - tl * hpg);
        float x[N_WIN];
        bool live = (r < rows_used) && (tok < q_len);
        const QT* qrow = q + (live ? ((int64_t)(q_start + tok) * n_head + head) * HEAD_DIM : 0);
        #pragma unroll
        for (int w = 0; w < N_WIN; ++w) {
            const float v = qt_to_f32<QT>(qrow[lane + 32 * w]);
            x[w] = live ? v : 0.f;
        }
        i8_rope_row<N_WIN>(x, prefix_len + tok, lane, rope_interleaved, rope.for_q());
        #pragma unroll
        for (int w = 0; w < N_WIN; ++w) {
            float a = fabsf(x[w]);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1)
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
            float scale = a / 127.f;
            float inv = (scale > 0.f) ? 1.f / scale : 0.f;
            s_q8[r][lane + 32 * w] = i8_quant(x[w], inv);
            if (lane == 0) s_q_scale[r][w] = __float2half(scale);
        }
    }

    // ------------------------------------------------------------------
    // Arena write pre-pass (split 0 of the row group's first key window only):
    // seal this block's fresh tokens into the writer chunks (unrotated K +
    // Q-capture, straight from the packed inputs — identical semantics to the
    // FP16 kernel's writeback, hoisted out of the tile loop). Writer chunks
    // use the identity palette.
    // ------------------------------------------------------------------
    bool writes_arena = split_idx == 0;
    if constexpr (PRESTAGED) writes_arena = writes_arena && stage.chunk_lo == 0;
    if (writes_arena) {
        const int tok_end = min(t0 + block_m_tok, q_len);
        for (int tok = t0; tok < tok_end; ++tok) {
            int w_slice, w_in_blk;
            resolve_pos(slot_hdr, prefix_len + tok, w_slice, w_in_blk);
            const uint8_t* w_sl = get_slice<HEAD_DIM>(slot_hdr.slices_ptr, w_slice, n_kv_head);
            const uint8_t* w_head = get_head<HEAD_DIM>(w_sl, kv_head_idx);
            const QT* k_row = k_packed + ((int64_t)(q_start + tok) * n_kv_head + kv_head_idx) * HEAD_DIM;
            const QT* v_row = v_packed + ((int64_t)(q_start + tok) * n_kv_head + kv_head_idx) * HEAD_DIM;
            const QT* q_row = q + ((int64_t)(q_start + tok) * n_head + first_q_head) * HEAD_DIM;
            for (int d = tid * 8; d < HEAD_DIM; d += I8_THREADS * 8) {
                int p = d / SUB;
                int local_d = d - p * SUB;
                store_kv_chunk_arena<QT, QT, SUB>(
                    (char*)kvhead_k_ptr<HEAD_DIM>(w_head, p),
                    (char*)kvhead_v_ptr<HEAD_DIM>(w_head, p),
                    &k_row[d], &v_row[d], &q_row[d],
                    kvhead_k_fmt<HEAD_DIM>(w_head, p),
                    kvhead_v_fmt<HEAD_DIM>(w_head, p),
                    0, 0, w_in_blk, local_d, 0, 0);
            }
        }
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Drain Q to registers (block-scope q8-matmul trick): each warp's QK
    // A-operand is its own 16 rows — N_WIN ldmatrix fragments + the two
    // fragment rows' per-window scales. Q never changes across the tile
    // loop, so after this barrier the prologue smem region is dead and
    // the tile overlay owns its bytes.
    // ------------------------------------------------------------------
    uint32_t q_frag[N_WIN][4];
    #pragma unroll
    for (int w = 0; w < N_WIN; ++w) {
        load_a_frag_m16k32_ldmatrix(q_frag[w], &s_q8[row_tile * 16][32 * w], Q8_LD, lane);
    }
    __syncthreads();

    // ------------------------------------------------------------------
    // Per-warp softmax + output state (registers). PV_H slices = this
    // warp's share of the output dims; they start at
    // dim_part * (HEAD_DIM / DIM_SPLIT).
    // ------------------------------------------------------------------
    constexpr int PV_H = PV_SLICES / DIM_SPLIT;
    const int dim_base = dim_part * (HEAD_DIM / DIM_SPLIT);
    float o_acc[PV_H][4];
    #pragma unroll
    for (int s = 0; s < PV_H; ++s)
        #pragma unroll
        for (int i = 0; i < 4; ++i) o_acc[s][i] = 0.f;
    float m_run[2] = { -INFINITY, -INFINITY };
    float l_run[2] = { 0.f, 0.f };

    // ------------------------------------------------------------------
    // Block walk. The block's query tokens between them select a bounded
    // set of `QB`-position selection blocks whatever the depth; each tile
    // packs the next NB = 32 / QB of them in ascending order, so the tile
    // loop's trip count is bounded by the rows' combined budget. A launch
    // without a selection walks 32-position blocks, one per tile — the
    // dense causal read. The walk is warp-private register state; every
    // warp derives the same block sequence, so the tile's composition
    // stays block-uniform without a handoff (see qsa_walk.cuh).
    //
    // Warp w stages columns 4w..4w+3: the cells `my_to..my_to+3` of the
    // tile's block `my_blk` (one warp per block at QB 4, all eight on the
    // single block at QB 32).
    // ------------------------------------------------------------------
    // A lane slot per 32 query tokens of the block's run: the widest run is
    // I8_M_ROWS tokens (hpg 1).
    constexpr int WALK_ROWS = (I8_M_ROWS + 31) / 32;
    using Walk = QsaWalk<WALK_ROWS>;
    static_assert(I8_M_ROWS <= Walk::MAX_ROWS,
                  "the walk binds one query row per lane slot");
    Walk walk;
    if (qsa_active(sel)) {
        walk.init(sel, q_start + t0, min(block_m_tok, q_len - t0), lane);
    } else {
        walk.init_dense(min(block_m_tok, q_len - t0), I8_TILE_TOK, lane);
    }
    const int QB = walk.ratio;
    const int NB = I8_TILE_TOK / QB;
    const int my_blk = (warp * I8_COLS_PER_WARP) / QB;
    const int my_to = warp * I8_COLS_PER_WARP - my_blk * QB;

    // Per-warp slice binding for sealed columns: the slice whose palette
    // metadata and rank tables sit in s_wpal[warp] (-1 until the first sealed
    // column; the rank tables are rebuilt on the first bind and thereafter
    // only on a map change).
    int bound_slice = -1;

    // ==================================================================
    // Split-KV ownership. A launch of sparse rows gives each split a
    // contiguous RANGE of positions — split s takes the blocks starting in
    // [share_start(s), share_start(s+1)), the s-th equal share of the first
    // row's entry list — and the split seeks straight to it and walks only
    // that. The walk is a serial chain of dependent loads per selected block,
    // so a split that walked the whole selection to keep every
    // `num_splits`-th tile paid the full walk however many splits shared the
    // work: a 5-row verify window at a 2,048-position budget ran ~300 µs per
    // launch, flat across 32 splits. A block belongs to the split its start
    // falls in, so no block is staged twice or skipped across a boundary.
    //
    // A dense row steps from the walk's bound through every position, so a
    // step can straddle a range boundary; a launch with any dense row keeps
    // the round-robin over tiles instead.
    //
    // A run with NO sparse row steps every tile across the 32 positions from
    // where the previous tile stopped, so its t-th tile is the 32 positions
    // from 32·t whatever came before it. Its shards seek straight to their own
    // tiles (`dealt_seek`) instead of walking past every tile dealt to the
    // others — a 294K-token prefix is 9K tiles, of which a shard keeps
    // 1/num_splits. Each shard still owns and packs exactly the tiles the
    // round-robin deals it.
    // ==================================================================
    const bool range_split = num_splits > 1 && !__any_sync(0xffffffffu, walk.dense != 0u);
    const bool dealt_seek = !range_split && num_splits > 1 && walk.all_dense();
    int bound = 0;               // next block start the walk may return
    int split_end = QSA_WALK_END; // first position past this split's range
    if (range_split) {
        bound = walk.share_start(split_idx, num_splits);
        split_end = walk.share_start(split_idx + 1, num_splits);
        walk.seek(bound);
    }

    // ==================================================================
    // Tile loop: each tile is the next NB selected blocks, packed.
    // ==================================================================
    // Visited-tile ordinal, for split-KV round-robin; under `dealt_seek`, the
    // ordinal of this shard's next tile.
    int tile_ord = dealt_seek ? split_idx : 0;

    // A staged block past its first key window resumes where the previous
    // window's launch left it: the softmax state, exact, and the walk's bound
    // — the bound the stopping tile's walk began from, so the first tile here
    // is packed from the same blocks it would have been. A sparse walk seeks
    // its cursors to that bound, which is where consuming every entry below
    // it one step at a time would have left them; a dense walk steps from the
    // bound whatever came before. Only live rows carry: a dead row's slot
    // would alias another sequence's rows.
    if constexpr (PRESTAGED) {
        if (stage.chunk_lo > 0) {
            int64_t tok_base;
            int blk_base;
            i8_carry_base(q_lens, kv_lens, batch_idx, stage.min_q_len, cut, block_m_tok,
                          tok_base, blk_base);
            const uint2 at = cut.resume[(int64_t)(blk_base + (int)blockIdx.x) * n_kv_head + kv_head_idx];
            bound = (int)at.x;
            tile_ord = (int)at.y;
            if (qsa_active(sel)) walk.seek(bound);
            constexpr int REC = HEAD_DIM + 2; // [o[HD], m, l]
            const int group_t0 = cut.blk_lo * block_m_tok;
            #pragma unroll
            for (int row = 0; row < 2; ++row) {
                if (!row_live(row)) continue;
                const float* rec = cut.carry
                    + ((tok_base + row_tok(row) - group_t0) * n_head + row_head(row)) * REC;
                #pragma unroll
                for (int s = 0; s < PV_H; ++s) {
                    const int dim = dim_base + s * 8 + n0;
                    o_acc[s][row * 2] = rec[dim];
                    o_acc[s][row * 2 + 1] = rec[dim + 1];
                }
                m_run[row] = rec[HEAD_DIM];
                l_run[row] = rec[HEAD_DIM + 1];
            }
        }
    }

    for (;;) {
        if (dealt_seek) bound = tile_ord * I8_TILE_TOK;
        // ---- WALK (every warp, identical): the tile's blocks ----
        // rm[h]: lane's row h's selected-cell bits over the tile's 32
        // columns (block b's cells at bits b·QB ..). my_q: the start of
        // this warp's block, or END when the tile ends before it.
        uint32_t rm[WALK_ROWS];
        #pragma unroll
        for (int h = 0; h < WALK_ROWS; ++h) rm[h] = 0u;
        int my_q = QSA_WALK_END;
        int n_blk = 0;
        for (int b = 0; b < NB; ++b) {
            uint32_t mk[WALK_ROWS];
            int step_end;
            const int q = walk.next(bound, mk, step_end);
            // A tile stops at the horizon or the key window's end by its FIRST
            // block only, and leaves `bound` where this step began.
            if (q >= kv_len || q >= split_end || (b == 0 && q >= tile_stop)) break;
            #pragma unroll
            for (int h = 0; h < WALK_ROWS; ++h) rm[h] |= mk[h] << (b * QB);
            if (b == my_blk) my_q = q;
            n_blk = b + 1;
            // The step's own end: through a page layout a block can be shorter
            // than `QB`, and the next block starts where this one ends.
            bound = step_end;
        }
        if (n_blk == 0) break;
        // Round-robin tiles across shards. The skip is block-uniform
        // (every thread computes identical walk state), so the staging
        // barriers below stay convergent.
        const bool mine = range_split || (tile_ord % num_splits) == split_idx;
        tile_ord += dealt_seek ? num_splits : 1;
        if (!mine) continue;

        // -------------------- STAGE (all warps) --------------------
        // Warp w stages its four columns into s_k8 / s_k_scale and stashes
        // V (natural dims, FP16) in s_fresh; the per-dim V requant below
        // runs over the whole tile once every column is in. A column past
        // the tile's blocks (or past kv_len inside a partial last block) is
        // zero and masked with P == 0 in the compute phase; a dead warp's
        // columns start at kv_len, so all four are.
        static_assert(I8_COLS_PER_WARP == 4, "a warp's columns are one token quad");
        const int pos0 = (my_q == QSA_WALK_END) ? kv_len : my_q + my_to;
        const int j0 = warp * I8_COLS_PER_WARP;
        bool from_stage = false;
        // Warp-uniform. The window's ends are multiples of 32, so an aligned
        // quad that starts inside the window lies wholly inside it.
        if constexpr (PRESTAGED)
            from_stage = staged && (pos0 & 3) == 0 && pos0 >= stage.chunk_lo
                      && pos0 < stage.chunk_hi;
        if (from_stage) {
            // The staged quad at pos0 — what `i8_stage_quad` below would
            // write for these four positions, copied in 16-byte pieces. A
            // column past kv_len was never staged and is zero, as the decode
            // would have made it.
            constexpr int K_CHUNKS = HEAD_DIM / 16;     // 16-byte pieces of a K code row
            constexpr int V_CHUNKS = HEAD_DIM * 2 / 16; // … of an FP16 V row
            constexpr int S_BYTES = N_WIN * 2;          // a K row's window scales
            const int live = kv_len - pos0;             // columns inside the sequence
            const int64_t row0 =
                (int64_t)kv_head_idx * stage.positions + stage_base + (pos0 - stage.chunk_lo);
            const int8_t* kc = kv_stage_k<HEAD_DIM>(stage, n_kv_head) + row0 * HEAD_DIM;
            const __half* ks = kv_stage_k_scale<HEAD_DIM>(stage, n_kv_head) + row0 * N_WIN;
            const __half* vv = kv_stage_v<HEAD_DIM>(stage, n_kv_head) + row0 * HEAD_DIM;
            constexpr int K_PIECES = I8_COLS_PER_WARP * K_CHUNKS;
            constexpr int V_PIECES = I8_COLS_PER_WARP * V_CHUNKS;
            #pragma unroll
            for (int i = 0; i < (K_PIECES + 31) / 32; ++i) {
                const int c = lane + 32 * i;
                if (c < K_PIECES) {
                    const int tt = c / K_CHUNKS;
                    const int off = (c - tt * K_CHUNKS) * 16;
                    int8_t* dst = &s_k8[j0 + tt][off];
                    if (tt < live) i8_cp_async16(dst, kc + (int64_t)tt * HEAD_DIM + off);
                    else *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            #pragma unroll
            for (int i = 0; i < (V_PIECES + 31) / 32; ++i) {
                const int c = lane + 32 * i;
                if (c < V_PIECES) {
                    const int tt = c / V_CHUNKS;
                    const int off = (c - tt * V_CHUNKS) * 8; // halves
                    __half* dst = &s_fresh[j0 + tt][off];
                    if (tt < live) i8_cp_async16(dst, vv + (int64_t)tt * HEAD_DIM + off);
                    else *reinterpret_cast<uint4*>(dst) = make_uint4(0u, 0u, 0u, 0u);
                }
            }
            if (lane < I8_COLS_PER_WARP) {
                const int tt = lane;
                __half* dst = &s_k_scale[j0 + tt][0];
                if (tt < live) {
                    kv_stage_cp_async<S_BYTES>(dst, ks + (int64_t)tt * N_WIN);
                } else {
                    #pragma unroll
                    for (int w = 0; w < N_WIN; w += 2)
                        *reinterpret_cast<uint32_t*>(dst + w) = 0u;
                }
            }
        } else {
            i8_stage_quad<QT, HEAD_DIM>(
                s_wpal[warp], bound_slice, slot_hdr, n_kv_head, kv_head_idx,
                k_packed, v_packed, q_start, prefix_len, kv_len, pos0, lane,
                rope_interleaved, rope,
                [&](int tt, int w, int8_t code, float scale) {
                    s_k8[j0 + tt][lane + 32 * w] = code;
                    if (lane == 0) s_k_scale[j0 + tt][w] = __float2half(scale);
                },
                [&](int tt, int w, float v) {
                    s_fresh[j0 + tt][lane + 32 * w] = __float2half(v);
                });
        }
        if (lane == 0) s_qpos[warp] = (my_q == QSA_WALK_END) ? DEAD_QPOS : my_q + my_to;
        if constexpr (PRESTAGED) {
            // cp.async groups are per thread: each drains its own copies
            // before the barrier publishes them (a bare __syncthreads does
            // not fence cp.async).
            i8_cp_commit();
            i8_cp_wait0();
        }
        __syncthreads();
        // V: per natural dim, max-abs over the tile's 32 columns, then
        // requant into the V^T slab (four columns per aligned store).
        for (int rr = tid; rr < HEAD_DIM; rr += I8_THREADS) {
            float a = 0.f;
            #pragma unroll
            for (int j = 0; j < I8_TILE_TOK; ++j)
                a = fmaxf(a, fabsf(__half2float(s_fresh[j][rr])));
            float scale = a / 127.f;
            float inv = (scale > 0.f) ? 1.f / scale : 0.f;
            for (int j4 = 0; j4 < I8_TILE_TOK; j4 += 4) {
                uint32_t pack = 0;
                #pragma unroll
                for (int jj = 0; jj < 4; ++jj)
                    pack |= (uint32_t)(uint8_t)i8_quant(
                                __half2float(s_fresh[j4 + jj][rr]), inv)
                            << (8 * jj);
                *(uint32_t*)&s_v8t[rr][j4] = pack;
            }
            s_v_scale[rr] = __float2half(scale);
        }
        __syncthreads();

        // -------------------- COMPUTE (per warp) --------------------
        // QK: 4 column slices × N_WIN window k-steps, FP32 fixup per window.
        //
        // Software-pipelined (the q8-matmul pattern): iteration k+1's
        // fragment + scale smem loads are ISSUED before iteration k's MMA,
        // so their smem-scoreboard latency drains under the tensor-core op
        // and the FP32 fixup instead of stalling the next issue — the
        // profiler's top stall at 33% occupancy. Fully unrolled: the
        // cur/next rotation is register renaming, not copies.
        float sc[4][4]; // [n-slice][fragment c-index]
        #pragma unroll
        for (int s = 0; s < 4; ++s)
            #pragma unroll
            for (int i = 0; i < 4; ++i) sc[s][i] = 0.f;

        {
            // Q fragments come from the drained registers (q_frag); the Q
            // scales broadcast from resident smem in the fixup. Only the
            // K-side B fragments + scales pipeline through smem.
            uint32_t b_cur[2], b_nxt[2];
            float ks_cur[2], ks_nxt[2];

            load_b_frag_n8k32_ldmatrix(b_cur, &s_k8[0][0], Q8_LD, lane);
            ks_cur[0] = __half2float(s_k_scale[n0][0]);
            ks_cur[1] = __half2float(s_k_scale[n0 + 1][0]);

            #pragma unroll
            for (int w = 0; w < N_WIN; ++w) {
                #pragma unroll
                for (int s = 0; s < 4; ++s) {
                    // Issue iteration k+1's loads first.
                    int it = w * 4 + s;
                    if (it + 1 < N_WIN * 4) {
                        int wn = (it + 1) >> 2;
                        int sn = (it + 1) & 3;
                        load_b_frag_n8k32_ldmatrix(b_nxt, &s_k8[sn * 8][32 * wn], Q8_LD, lane);
                        ks_nxt[0] = __half2float(s_k_scale[sn * 8 + n0][wn]);
                        ks_nxt[1] = __half2float(s_k_scale[sn * 8 + n0 + 1][wn]);
                    }
                    // MMA + fixup on iteration k while the loads fly.
                    int32_t c_i[4] = {0, 0, 0, 0};
                    int32_t d_i[4];
                    mma_int8_m16n8k32(d_i, q_frag[w], b_cur, c_i);
                    float2 qs;
                    qs.x = __half2float(s_q_scale[row_of(0)][w]);
                    qs.y = __half2float(s_q_scale[row_of(1)][w]);
                    sc[s][0] += (float)d_i[0] * qs.x * ks_cur[0];
                    sc[s][1] += (float)d_i[1] * qs.x * ks_cur[1];
                    sc[s][2] += (float)d_i[2] * qs.y * ks_cur[0];
                    sc[s][3] += (float)d_i[3] * qs.y * ks_cur[1];
                    // Rotate (renamed away under full unroll).
                    #pragma unroll
                    for (int r = 0; r < 2; ++r) {
                        b_cur[r] = b_nxt[r];
                        ks_cur[r] = ks_nxt[r];
                    }
                }
            }
        }

        // Mask + scale. Column j's logical kv position is the staging
        // warp's block start plus its cell (s_qpos[j / 4] + j % 4; a dead
        // warp's DEAD_QPOS fails the horizon). The row's selected cells are
        // the walk's bit mask, fetched from the lane that owns the row
        // (row r of the block's run lives on lane r & 31, half r >> 5) —
        // every column a row attends is a bit test, no per-key search. The
        // causal horizons are per-tile transients (dead after this loop).
        int horizon[2];
        uint32_t rmask[2];
        #pragma unroll
        for (int i = 0; i < 2; ++i) {
            int h = prefix_len + row_tok(i) + 1;
            if (h > kv_len) h = kv_len;
            horizon[i] = row_live(i) ? h : 0;
            const int r = row_tok(i) - t0;
            uint32_t m = __shfl_sync(0xffffffffu, rm[0], r & 31);
            #pragma unroll
            for (int half = 1; half < WALK_ROWS; ++half) {
                const uint32_t mh = __shfl_sync(0xffffffffu, rm[half], r & 31);
                if ((r >> 5) == half) m = mh;
            }
            rmask[i] = row_live(i) ? m : 0u;
        }
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            int ja = s * 8 + n0;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                int j = ja + (i & 1);
                int row = i >> 1;
                const int pos = s_qpos[j / I8_COLS_PER_WARP] + (j % I8_COLS_PER_WARP);
                const bool ok = (pos < horizon[row]) && ((rmask[row] >> j) & 1u);
                sc[s][i] = ok ? sc[s][i] * softmax_scale : -INFINITY;
            }
        }

        // Online softmax per fragment row.
        float m_new[2], alpha[2];
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            float m_tile = -INFINITY;
            #pragma unroll
            for (int s = 0; s < 4; ++s) {
                m_tile = fmaxf(m_tile, sc[s][row * 2]);
                m_tile = fmaxf(m_tile, sc[s][row * 2 + 1]);
            }
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffffu, m_tile, 1));
            m_tile = fmaxf(m_tile, __shfl_xor_sync(0xffffffffu, m_tile, 2));
            m_new[row] = fmaxf(m_run[row], m_tile);
            alpha[row] = (m_run[row] == -INFINITY) ? 0.f : __expf(m_run[row] - m_new[row]);
        }

        float l_add[2] = { 0.f, 0.f };
        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            int ja = s * 8 + n0;
            #pragma unroll
            for (int row = 0; row < 2; ++row) {
                float p0 = (sc[s][row * 2] == -INFINITY || m_new[row] == -INFINITY)
                               ? 0.f
                               : __expf(sc[s][row * 2] - m_new[row]);
                float p1 = (sc[s][row * 2 + 1] == -INFINITY || m_new[row] == -INFINITY)
                               ? 0.f
                               : __expf(sc[s][row * 2 + 1] - m_new[row]);
                l_add[row] += p0;
                l_add[row] += p1;
                // ja is even: the byte pair stores as one aligned u16.
                uint16_t pk = (uint16_t)(uint8_t)(int8_t)rintf(p0 * 127.f) |
                              ((uint16_t)(uint8_t)(int8_t)rintf(p1 * 127.f) << 8);
                *(uint16_t*)&s_p8[warp][g + row * 8][ja] = pk;
            }
        }
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            float ls = l_add[row];
            ls += __shfl_xor_sync(0xffffffffu, ls, 1);
            ls += __shfl_xor_sync(0xffffffffu, ls, 2);
            l_run[row] = l_run[row] * alpha[row] + ls;
            m_run[row] = m_new[row];
        }
        __syncwarp();

        // PV: one m16n8k32 per output-dim slice (k = the tile's 32 tokens),
        // software-pipelined like QK: slice s+1's V fragment + scales issue
        // before slice s's MMA.
        {
            uint32_t pa[4];
            load_a_frag_m16k32_ldmatrix(pa, &s_p8[warp][0][0], V8T_LD, lane);
            uint32_t vb_cur[2], vb_nxt[2];
            float vs_cur[2], vs_nxt[2];
            load_b_frag_n8k32_ldmatrix(vb_cur, &s_v8t[dim_base][0], V8T_LD, lane);
            vs_cur[0] = __half2float(s_v_scale[dim_base + n0]) * (1.f / 127.f);
            vs_cur[1] = __half2float(s_v_scale[dim_base + n0 + 1]) * (1.f / 127.f);
            #pragma unroll
            for (int s = 0; s < PV_H; ++s) {
                if (s + 1 < PV_H) {
                    load_b_frag_n8k32_ldmatrix(vb_nxt, &s_v8t[dim_base + (s + 1) * 8][0], V8T_LD, lane);
                    vs_nxt[0] = __half2float(s_v_scale[dim_base + (s + 1) * 8 + n0]) * (1.f / 127.f);
                    vs_nxt[1] = __half2float(s_v_scale[dim_base + (s + 1) * 8 + n0 + 1]) * (1.f / 127.f);
                }
                int32_t c_i[4] = {0, 0, 0, 0};
                int32_t d_i[4];
                mma_int8_m16n8k32(d_i, pa, vb_cur, c_i);
                o_acc[s][0] = o_acc[s][0] * alpha[0] + (float)d_i[0] * vs_cur[0];
                o_acc[s][1] = o_acc[s][1] * alpha[0] + (float)d_i[1] * vs_cur[1];
                o_acc[s][2] = o_acc[s][2] * alpha[1] + (float)d_i[2] * vs_cur[0];
                o_acc[s][3] = o_acc[s][3] * alpha[1] + (float)d_i[3] * vs_cur[1];
                vb_cur[0] = vb_nxt[0];
                vb_cur[1] = vb_nxt[1];
                vs_cur[0] = vs_nxt[0];
                vs_cur[1] = vs_nxt[1];
            }
        }

        __syncthreads(); // staging buffers are reused next iteration
    }

    // ------------------------------------------------------------------
    // A staged block whose horizon lies past this key window carries: its
    // un-normalized state and the bound its walk stopped at go to the carry
    // for the next window's launch, and nothing is written to `out`. The
    // records are [o[HD], m, l] like the split partials; every warp of a
    // row-tile group holds identical m/l (duplicated QK), so the dim_part==0
    // warp's quad leader writes them.
    // ------------------------------------------------------------------
    if constexpr (PRESTAGED) {
        if (staged && min(kv_len, prefix_len + min(t0 + block_m_tok, q_len)) > stage.chunk_hi) {
            int64_t tok_base;
            int blk_base;
            i8_carry_base(q_lens, kv_lens, batch_idx, stage.min_q_len, cut, block_m_tok,
                          tok_base, blk_base);
            if (tid == 0)
                cut.resume[(int64_t)(blk_base + (int)blockIdx.x) * n_kv_head + kv_head_idx] =
                    make_uint2((uint32_t)bound, (uint32_t)tile_ord);
            constexpr int REC = HEAD_DIM + 2;
            const int group_t0 = cut.blk_lo * block_m_tok;
            #pragma unroll
            for (int row = 0; row < 2; ++row) {
                if (!row_live(row)) continue;
                float* rec = cut.carry
                    + ((tok_base + row_tok(row) - group_t0) * n_head + row_head(row)) * REC;
                #pragma unroll
                for (int s = 0; s < PV_H; ++s) {
                    const int dim = dim_base + s * 8 + n0;
                    rec[dim] = o_acc[s][row * 2];
                    rec[dim + 1] = o_acc[s][row * 2 + 1];
                }
                if (dim_part == 0 && (lane & 3) == 0) {
                    rec[HEAD_DIM] = m_run[row];
                    rec[HEAD_DIM + 1] = l_run[row];
                }
            }
            return;
        }
    }

    // ------------------------------------------------------------------
    // Epilogue. num_splits == 1: normalize and store O directly (natural
    // dims — no permute needed). Otherwise: emit the un-normalized
    // (ΣpV, m, l) partial for this shard; the combine kernel merges.
    // ------------------------------------------------------------------
    if (num_splits == 1) {
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            if (!row_live(row) || l_run[row] <= 0.f) continue;
            float inv_l = 1.f / l_run[row];
            QT* orow = out + ((int64_t)(q_start + row_tok(row)) * n_head + row_head(row)) * HEAD_DIM;
            #pragma unroll
            for (int s = 0; s < PV_H; ++s) {
                int dim = dim_base + s * 8 + n0;
                orow[dim] = qt_from_f32<QT>(o_acc[s][row * 2] * inv_l);
                orow[dim + 1] = qt_from_f32<QT>(o_acc[s][row * 2 + 1] * inv_l);
            }
        }
    } else {
        constexpr int REC = HEAD_DIM + 2; // [o[HD], m, l]
        #pragma unroll
        for (int row = 0; row < 2; ++row) {
            if (!row_live(row)) continue; // dead rows alias other seqs' rows
            int64_t row_id = (int64_t)(q_start + row_tok(row)) * n_head + row_head(row);
            float* rec = partials + (row_id * num_splits + split_idx) * REC;
            #pragma unroll
            for (int s = 0; s < PV_H; ++s) {
                int dim = dim_base + s * 8 + n0;
                rec[dim] = o_acc[s][row * 2];
                rec[dim + 1] = o_acc[s][row * 2 + 1];
            }
            // Softmax state: every warp of a row-tile group holds identical
            // m/l (duplicated QK); the dim_part==0 warp's lane 0 writes it.
            if (dim_part == 0 && (lane & 3) == 0) {
                rec[HEAD_DIM] = m_run[row];
                rec[HEAD_DIM + 1] = l_run[row];
            }
        }
    }
}
