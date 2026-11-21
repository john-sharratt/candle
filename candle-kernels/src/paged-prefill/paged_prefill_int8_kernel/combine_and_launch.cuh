// The split-KV combine kernel and the prefill launchers.
//
// Holds `paged_prefill_int8_combine_kernel`, which merges the `num_splits`
// partial outputs of each row, and `launch_paged_prefill_int8`, which runs the
// grid in row groups and key chunks.
// Also holds `launch_paged_prefill_kv_stage_only`, the pre-staging pass alone,
// with its own launch check.

// ============================================================================
// Split-KV combine: one block per output row, merging `num_splits` partials
// with base-e log-sum-exp. Empty shards carry (m = -inf, l = 0) and vanish.
// ============================================================================

// The most shards a launch splits into. A short window over a long prefix —
// a speculative verify of a few rows — runs as few as 4 unsplit blocks, and
// with the walk range-split each shard's latency is its share of the
// selection, so the launch fills every resident slot: 2 blocks/SM at
// HEAD_DIM 256 is 220 slots on 110 SMs, 55 shards of 4 blocks.
constexpr int I8_MAX_SPLITS = 64;

template <typename QT, int HEAD_DIM>
__global__ void paged_prefill_int8_combine_kernel(
    const float* __restrict__ partials,
    QT* __restrict__ out,
    int num_splits,
    int64_t total_rows
) {
    constexpr int REC = HEAD_DIM + 2;
    const int64_t row_id = blockIdx.x;
    if (row_id >= total_rows) return;
    const int d = (int)threadIdx.x;

    __shared__ float s_gm;
    __shared__ float s_scale[I8_MAX_SPLITS]; // exp(m_s - gm) per split
    __shared__ float s_inv_l;

    const float* base = partials + row_id * num_splits * REC;
    // The shards' (m, l) fold on warp 0, two shards per lane: every load is
    // issued at once and the max and the rescaled sum are shuffle trees, so a
    // wide split costs one round trip rather than a serial chain per shard.
    static_assert(I8_MAX_SPLITS <= 64, "the fold gives each lane two shards");
    if (d < 32) {
        const int s0 = d, s1 = d + 32;
        const float m0 = s0 < num_splits ? base[s0 * REC + HEAD_DIM] : -INFINITY;
        const float m1 = s1 < num_splits ? base[s1 * REC + HEAD_DIM] : -INFINITY;
        const float l0 = s0 < num_splits ? base[s0 * REC + HEAD_DIM + 1] : 0.f;
        const float l1 = s1 < num_splits ? base[s1 * REC + HEAD_DIM + 1] : 0.f;
        float gm = fmaxf(m0, m1);
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            gm = fmaxf(gm, __shfl_xor_sync(0xffffffffu, gm, off));
        const float sc0 = (m0 == -INFINITY) ? 0.f : __expf(m0 - gm);
        const float sc1 = (m1 == -INFINITY) ? 0.f : __expf(m1 - gm);
        if (s0 < num_splits) s_scale[s0] = sc0;
        if (s1 < num_splits) s_scale[s1] = sc1;
        float l_tot = l0 * sc0 + l1 * sc1;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            l_tot += __shfl_xor_sync(0xffffffffu, l_tot, off);
        if (d == 0) {
            s_gm = gm;
            s_inv_l = (l_tot > 0.f) ? 1.f / l_tot : 0.f;
        }
    }
    __syncthreads();
    if (s_gm == -INFINITY) return; // row never attended anything

    float acc = 0.f;
    // Unrolled so a batch of shards' loads is in flight before their adds.
    #pragma unroll 8
    for (int s = 0; s < num_splits; ++s) {
        acc += base[s * REC + d] * s_scale[s];
    }
    out[row_id * HEAD_DIM + d] = qt_from_f32<QT>(acc * s_inv_l);
}

// ============================================================================
// Launcher
// ============================================================================

template <typename QT, int HEAD_DIM>
inline void launch_paged_prefill_int8(
    const void* q_ptr,
    const void* k_ptr,
    const void* v_ptr,
    const uint8_t* headers_ptr,
    const uint32_t* cu_seqlens_q,
    const uint32_t* q_lens,
    const uint32_t* kv_lens,
    void* o_ptr,
    int32_t total_q,
    int32_t batch_size,
    int32_t n_head,
    int32_t n_kv_head,
    int32_t max_q_len,
    float softmax_scale,
    const RopeRungs rungs,
    int32_t rope_interleaved,
    cudaStream_t stream,
    QsaSel sel,
    // The pre-staged K/V (`kv_stage.cuh`): `stage.buf == nullptr` stages
    // nothing and every column is read from its source; otherwise every
    // sequence with `q_len >= stage.min_q_len` is staged into `stage.buf`
    // (`stage_bytes` long) ahead of each launch, one key window at a time,
    // `stage_max_kv` the deepest. The window fields are set per launch here.
    const PrefillKvStage stage,
    int64_t stage_bytes,
    int32_t stage_max_kv,
    // The cut (see the header): `n_groups` row groups of `group_blocks`
    // grid-x blocks, group g running `group_chunks[g]` key windows of
    // `chunk_positions` (a HOST array); `carry` / `resume` as `PrefillCut`
    // describes them, unused when no group runs more than one window.
    const uint32_t* group_chunks,
    int32_t n_groups,
    int32_t group_blocks,
    int32_t chunk_positions,
    float* carry,
    uint2* resume
) {
    // The tile packs 32 / ratio selection blocks, so the ratio must divide
    // the tile. A selection built at any other ratio is a host bug, not a
    // runtime condition — refuse loudly rather than attend the wrong keys.
    if (sel.entries != nullptr && !i8_ratio_supported(sel.ratio)) {
        fprintf(stderr, "PAGED PREFILL INT8: unsupported QSA ratio %d (need 4, 8, 16 or 32)\n",
                sel.ratio);
        abort();
    }
    int hpg = (n_kv_head > 0) ? n_head / n_kv_head : 1;
    if (hpg <= 0) hpg = 1;
    int block_m_tok = i8_m_rows(HEAD_DIM) / hpg;
    if (block_m_tok <= 0) block_m_tok = 1;
    uint32_t grid_x = (uint32_t)((max_q_len + block_m_tok - 1) / block_m_tok);
    if (grid_x == 0) grid_x = 1;

    // The cut is the host's plan (`PrefillKvStageLayout`); one that does not
    // tile this grid, chunks a launch with no stage, or carries with nowhere
    // to carry to is a host bug — refuse it rather than leave rows unwritten.
    bool chunked = false;
    if (group_chunks != nullptr && group_blocks > 0) {
        for (int g = 0; g < n_groups; ++g) chunked = chunked || group_chunks[g] > 1u;
    }
    if (group_chunks == nullptr || group_blocks <= 0
        || (int64_t)n_groups * group_blocks < (int64_t)grid_x
        || (int64_t)(n_groups - 1) * group_blocks >= (int64_t)grid_x
        || chunk_positions <= 0 || chunk_positions % I8_TILE_TOK != 0
        || (chunked && (stage.buf == nullptr || carry == nullptr || resume == nullptr))) {
        fprintf(stderr,
                "PAGED PREFILL INT8: a cut of %d groups × %d blocks (chunks %d positions, "
                "chunked %d, stage %p, carry %p, resume %p) for a grid of %u blocks\n",
                n_groups, group_blocks, chunk_positions, (int)chunked, (void*)stage.buf,
                (void*)carry, (void*)resume, grid_x);
        abort();
    }

    // Split-KV factor: fan the tile walk across grid.z shards up to this
    // head dim's residency limit. The short-q/long-prefix regime otherwise
    // runs the whole prefix walk in grid_x × n_kv_head × batch blocks — as
    // few as 4 on the production shape — leaving the GPU idle.
    static int s_sm_count = 0;
    if (s_sm_count == 0) {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&s_sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (s_sm_count <= 0) s_sm_count = 64;
    }
    int base_blocks = (int)grid_x * n_kv_head * (int)batch_size;
    // Split only when the unsplit grid leaves SMs idle — a grid that
    // already covers the SMs loses more to partial-emit + combine traffic
    // than it gains from sharding the walk (measured: q256/prefix2k
    // regressed 2.8 → 3.2 ms when split unconditionally). Fill toward the
    // residency limit but NEVER past it — floor, not ceil: one block over
    // the slot count starts a second wave and near-doubles the makespan
    // (measured: 320 blocks on 304 slots ran 1.69 → 2.36 ms).
    int num_splits = 1;
    if (base_blocks < s_sm_count) {
        num_splits = (i8_min_blocks(HEAD_DIM) * s_sm_count) / base_blocks;
        if (num_splits < 1) num_splits = 1;
        if (num_splits > I8_MAX_SPLITS) num_splits = I8_MAX_SPLITS;
    }
    // Decided over the WHOLE grid, so a row group of an unchunked launch keeps
    // the split the uncut launch would have had — and with it the same bits.
    // A launch that carries runs unsplit: the carry holds one softmax state
    // per row, and a split's shards each hold a partial one that only the
    // combine can merge.
    if (chunked) num_splits = 1;

    // The partial pool, held for one scratch epoch (same idiom as the decode
    // split-KV pool: single-stream, taken with `grow_scratch`, which is safe
    // while the thread records a graph). A pool the scratch cannot give runs
    // the launch unsplit below.
    float* partials = nullptr;
    if (num_splits > 1) {
        static void* s_pool = nullptr;
        static size_t s_pool_bytes = 0;
        static unsigned long long s_pool_epoch = 0;
        size_t need = (size_t)total_q * (size_t)n_head * (size_t)num_splits * (HEAD_DIM + 2)
                      * sizeof(float);
        if (grow_scratch(&s_pool, &s_pool_bytes, &s_pool_epoch, need) == cudaSuccess) {
            partials = (float*)s_pool;
        }
        if (partials == nullptr) num_splits = 1; // OOM fallback: direct store
    }

    dim3 block(I8_THREADS, 1, 1);

    // Clear any error left sticky on this thread by a PRIOR launch so the
    // post-launch check below reflects only this kernel — otherwise a stale
    // error is misattributed here (printed against this grid) even though this
    // launch config is valid and its output correct.
    (void)cudaGetLastError();

    // Row groups outer, key windows inner, all on one stream: a window's
    // pre-staging overwrites the stage only after the previous window's
    // attention has read it, and a window's launch reads the carry the
    // previous one left. Each group's blocks are a slice of the whole grid.
    dim3 grid(grid_x, (uint32_t)n_kv_head, (uint32_t)(batch_size * num_splits));
    for (int g = 0; g < n_groups; ++g) {
        const int blk_lo = g * group_blocks;
        grid.x = (uint32_t)min(group_blocks, (int)grid_x - blk_lo);
        const int chunks = (int)group_chunks[g];
        for (int k = 0; k < chunks; ++k) {
            const PrefillCut cut{blk_lo, group_blocks, chunk_positions, (int)(k + 1 == chunks),
                                 carry, resume};
            PrefillKvStage window = stage;
            window.chunk_lo = k * chunk_positions;
            window.chunk_hi = (k + 1) * chunk_positions;
            window.tok_lo = blk_lo * block_m_tok;
            if (stage.buf != nullptr) {
                launch_paged_prefill_kv_prestage<QT, HEAD_DIM>(
                    k_ptr, v_ptr, headers_ptr, cu_seqlens_q, q_lens, kv_lens,
                    batch_size, n_kv_head, stage_max_kv, rungs, rope_interleaved,
                    window, stage_bytes, stream);
                paged_prefill_int8_kernel<QT, HEAD_DIM, true><<<grid, block, 0, stream>>>(
                    (const QT*)q_ptr, (const QT*)k_ptr, (const QT*)v_ptr,
                    headers_ptr, cu_seqlens_q, q_lens, kv_lens,
                    (QT*)o_ptr, (int)batch_size, (int)n_head, (int)n_kv_head,
                    softmax_scale, rungs, (int)rope_interleaved,
                    num_splits, partials, sel, window, cut);
            } else {
                paged_prefill_int8_kernel<QT, HEAD_DIM, false><<<grid, block, 0, stream>>>(
                    (const QT*)q_ptr, (const QT*)k_ptr, (const QT*)v_ptr,
                    headers_ptr, cu_seqlens_q, q_lens, kv_lens,
                    (QT*)o_ptr, (int)batch_size, (int)n_head, (int)n_kv_head,
                    softmax_scale, rungs, (int)rope_interleaved,
                    num_splits, partials, sel, window, cut);
            }
        }
    }

    // One combine over every row: the groups' partial rows are disjoint.
    if (num_splits > 1) {
        int64_t total_rows = (int64_t)total_q * n_head;
        dim3 cgrid((uint32_t)total_rows);
        dim3 cblock(HEAD_DIM, 1, 1);
        paged_prefill_int8_combine_kernel<QT, HEAD_DIM><<<cgrid, cblock, 0, stream>>>(
            partials, (QT*)o_ptr, num_splits, total_rows);
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr,
                "PAGED PREFILL INT8 KERNEL LAUNCH FAILED: %s (grid=%d,%d,%d hd=%d splits=%d "
                "staged=%d)\n",
                cudaGetErrorString(err), grid.x, grid.y, grid.z, HEAD_DIM, num_splits,
                (int)(stage.buf != nullptr));
    }
}

/// The pre-staging pass alone, with its own launch check — what a test reads
/// the stage planes back from.
template <typename QT, int HEAD_DIM>
inline void launch_paged_prefill_kv_stage_only(
    const void* k_ptr,
    const void* v_ptr,
    const uint8_t* headers_ptr,
    const uint32_t* cu_seqlens_q,
    const uint32_t* q_lens,
    const uint32_t* kv_lens,
    int32_t batch_size,
    int32_t n_kv_head,
    const RopeRungs rungs,
    int32_t rope_interleaved,
    cudaStream_t stream,
    const PrefillKvStage stage,
    int64_t stage_bytes,
    int32_t stage_max_kv
) {
    (void)cudaGetLastError();
    launch_paged_prefill_kv_prestage<QT, HEAD_DIM>(
        k_ptr, v_ptr, headers_ptr, cu_seqlens_q, q_lens, kv_lens,
        batch_size, n_kv_head, stage_max_kv, rungs, rope_interleaved,
        stage, stage_bytes, stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "PAGED PREFILL KV STAGE LAUNCH FAILED: %s (hd=%d max_kv=%d)\n",
                cudaGetErrorString(err), HEAD_DIM, stage_max_kv);
    }
}
