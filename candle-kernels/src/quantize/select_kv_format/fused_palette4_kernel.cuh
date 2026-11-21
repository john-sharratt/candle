// The fused selection kernel that assigns each block a palette slot and format.
//
// The fused selection kernel, one CUDA block per (chunk, head). It detects
// attention-sink tokens, sorts blocks by amax, and lets each of the four
// palette slots claim the most aggressive format whose normalised error stays
// within threshold for at least HB/4 of the unclaimed blocks. The outputs are
// the per-slot tags and scales, and the per-block palette map that the
// attention kernel dispatches on.

// =============================================================================
// FUSED SELECTION + PALETTE4 KERNEL
// =============================================================================
//
// Pass 2 of the selection pipeline. One CUDA block per (chunk, head):
//
//     grid  = (total_heads, 1, 1)             // total_heads = chunks · n_kv_head
//     block = (FUSED_THREADS_PER_BLOCK, 1, 1) // 4 warps × 32 lanes = 128
//
// Each block reads HB blocks of K and V, performs sink detection,
// sorts by amax, runs four BPE-ascending searches (one per palette
// slot, twice for K and V), and writes 4 (fmt, scale) palette slots
// plus a per-block slot index.
//
// HB is the template parameter (blocks_per_head = head_dim, 128 or 256);
// the runtime `blocks_per_head` argument MUST equal HB — the kernel
// returns early if that contract is violated. Thread count stays at 128:
// at HB = 256 every per-entry phase strides so each thread owns 2 entries.
//
// Algorithm summary
// -----------------
// Phases (all comments inside the kernel are tagged "Phase N: …"):
//
//   1.    Parallel load + per-block amax + per-block q-relevance.
//   2.5.  Per-token attention-sink detection. Compute the chunk's mean
//         Q vector; per-token sink score = q_mean · K_token / √d;
//         z-score against chunk-local μ/σ; sink_weight[t] = max(0, tanh(z)).
//         Hoist the V-side threshold² out of the search loop.
//   2.    Bitonic sort HB (amax, idx) pairs descending by amax.
//   3.    Per-block K threshold from q-relevance z-score.
//   4+5.  Iterative slot search — see process_side lambda.
//
// Slot evolution (Q = HB/4, the per-slot quota: 32 at HB=128, 64 at HB=256)
// -------------------------------------------------------------------------
// idx_sorted is amax-descending after Phase 2. Each slot iteration
// compacts the alive bitmask, claims Q entries (passing-first then
// fill), and tombstones them in the bitmask for the next slot to skip.
//
//   slot 0:  ████████████████████████████████████████████████████████████  (live = 4Q)
//                                                                ┘ claims Q (highest amax)
//
//   slot 1:  ████████████████████████████████████████████████              (live = 3Q)
//                                                            ┘ claims Q
//
//   slot 2:  ████████████████████████████████                              (live = 2Q)
//                                            ┘ claims Q
//
//   slot 3:  ████████████████                                              (live =  Q)
//                            ┘ claims Q (lowest amax)
//
//   Slot 0 ends up with the most conservative format (highest BPE) and
//   slot 3 with the most aggressive (lowest BPE) — the BPE-ascending
//   search exits at the first format where Q blocks pass, and
//   high-amax blocks need more bits to stay within threshold.
//
// Why MSE on V and top-4 mean on K
// --------------------------------
// V's contribution to attention output is Σ_t a_t · v_t — an
// attention-weighted sum whose error budget is L2 (sum-squared), so
// mean-squared error is the structurally correct choice. Top-4 mean
// would over-penalise outlier elements that don't move the L2 of the
// actual computation V participates in.
//
// K errors enter the softmax via the per-element products q·k. Outliers
// in K dominate the score perturbation, but a single-lane spike often
// gets washed out by softmax — using max alone over-rejects benign
// candidates. The mean of the four largest weighted errors is a
// stable proxy that tracks the worst few errors without being all-or-
// nothing on a single lane.
//
// Sink protection (V-side)
// ------------------------
// Sink tokens carry disproportionate attention mass and dominate output
// error if their V is quantised loosely. The V threshold is interpolated
// between `v_threshold_lo` (lenient, non-sink) and `v_threshold_hi`
// (strict, peak-sink) using the chunk-wide max sink_weight:
//
//     v_thr_eff = v_threshold_lo + max_sink · (v_threshold_hi − v_threshold_lo)
//     v_thr_sq  = v_thr_eff²
//
// One strong sink token in the chunk forces stricter quality on every
// V block of the chunk. Conservative, but correct: any block could end
// up being attended to jointly with the sink during decode, so its
// error budget needs to account for that.
//
// FUSED_WARP_SIZE / FUSED_THREADS_PER_BLOCK are defined near the top of
// the file alongside SELECT_FMT_* so the templated search/claim helpers
// can reach them.

// Shared memory layout — sizes at HB=128 / HB=256 (AW = HB/64 mask words)
// -------------------------------------------------------------------------------
//
//   smem_kv       [HB × 32]    f16    8,192 / 16,384 B  K then V values (aliased);
//                                              f16 halves the prior f32 buffer.
//   amax_k        [HB]         f32      512 / 1,024 B  K block amax, sorted desc
//   amax_v        [HB]         f32      512 / 1,024 B  V block amax, sorted desc
//   kidx, vidx    [HB]         u16    2×256 / 2×512 B  sorted block IDs (read-only after sort)
//   qrel_k        [HB]         f16      256 /   512 B  per-block q-relevance
//   kthresh       [HB]         f16      256 /   512 B  per-block K threshold
//   q_mean        [HB]         f32      512 / 1,024 B  per-head_dim mean Q for Phase 2.5
//                                              (head_dim = HB: one block per dim);
//                                              then amax_compact, the live amaxes
//                                              in sort order, for each slot
//   sink_score    [32]         f32      128 B  per-token raw Q·K (2.5)
//   sink_weight   [32]         f32      128 B  per-token weight ∈ [0,1] (2.5)
//   warp_f32      [4][32]      f32      512 B  per-warp round-trip scratch
//   warp_quant    [4][36]      u8       144 B  per-warp quantized block scratch
//   slot_tags     [4]          i32       16 B  winning fmt per slot
//   slot_scales   [4]          f32       16 B  winning outer scale per slot
//   k_alive,v_alive [AW]       u64     32 / 64 B  K/V alive bitmask (1 bit/block)
//   idx_compact   [HB]         u16      256 /   512 B  compacted live entries per slot
//   warp_pass     [4×6×AW]     u64      384 /   768 B  pass-mask words   [warp][si][w]
//   warp_amax_err [4×6]        f32       96 B  max passing error        [warp][si]
//   s_best_*, s_fallback_*              ~60 /   ~92 B  search winner / fallback state
//   s_seg_pop     [HB/32]      i32     16 / 32 B  compaction scratch (one
//                                              entry per 32-position sort segment)
//   s_slot_{amax,p95,p80,mean,p25}       ~20 B  cross-warp scalars
//   s_band_ptr    [2][4]       ptr       64 B  } per-band source views (one per
//   s_band_fmt    [2][4]       i32       32 B  } (side, palette) gid) — resolved
//   s_band_outer  [2][4]       f32       32 B  } up front, read by Phase 1 and
//                                              } V reload. Keeps source ptrs/fmts
//                                              } out of regs across process_side.
//
// Totals: ~12.6 KB at HB=128 (8 blocks/SM → ~100.6 KB) and ~23.8 KB at
// HB=256 (4 blocks/SM → ~95 KB), both inside the ~102.4 KB MaxShared
// budget on Ada. launch_bounds(128, FusedOccupancy<HB>) caps registers;
// search_scales_for_fmt is __noinline__ so its frame is separate,
// preventing the inlined loop body from inflating the kernel's combined
// register count.
//
// Tombstoning lives in `k_alive`/`v_alive` (no -1 sentinel in idx arrays);
// per-slot `idx_compact` rebuild lets the search inner loop walk only
// live entries without an alive_get probe per iteration.
//
// Pass-mask packing
// -----------------
// Each block has an ID b ∈ [0, HB). A passing block sets bit (b & 63) of
// word b >> 6 in an HB-bit mask carried as AW u64 words. The mask is set
// DURING the search; when a (fmt, scale) wins, its mask becomes the new
// s_best_pass. The claim phase reads this mask and skips the round-trip —
// search already determined which blocks pass at the winning combo.

template <int HB>
__global__ __launch_bounds__(FUSED_THREADS_PER_BLOCK, FusedOccupancy<HB>::min_blocks_per_sm) void select_kv_format_palette4_paged(
    const int64_t* __restrict__ per_head_table_raw,
    const int64_t* __restrict__ head_gids,
    const float*   __restrict__ q_relevance_median,   // [total_heads]
    const float*   __restrict__ q_relevance_spread,   // [total_heads]
    const float*   __restrict__ k_head_amax_in,       // [total_heads]
    const float*   __restrict__ v_head_amax_in,       // [total_heads]
    const float*   __restrict__ k_head_p95_in,        // [total_heads] 95th pct of |K|; caps the head scale
    const float*   __restrict__ v_head_p95_in,        // [total_heads] 95th pct of |V|; caps the head scale
    const int*     __restrict__ k_candidates,
    const int*     __restrict__ v_candidates,
    int num_k_candidates,
    int num_v_candidates,
    float k_threshold_hi,
    float k_threshold_lo,
    float v_threshold_hi,   // strict — applied at peak sink (w_max=1)
    float v_threshold_lo,   // lenient — applied at non-sink (w_max=0). Convention: lo > hi numerically.
    int total_heads,
    int blocks_per_head,    // must equal the template parameter HB
    int n_kv_head,
    int arena_chunks,
    // Per-chunk valid token range, packed (offset << 8) | len with
    // len in [1, 32]. Every load reads a slot outside [offset, offset+len)
    // as ZERO (`lane_in_window`), so a dead slot contributes nothing to
    // amax or the error sums whatever bytes it holds; the range also fixes
    // the COUNT-normalized metrics (V's mean-over-32 MSE, K's top-4 mean)
    // and the sink statistics, which would otherwise be diluted by the zero
    // lanes.
    const int* __restrict__ valid_ranges,          // [n_chunks]
    int*   __restrict__ k_palette_tags,            // [total_heads * 4]
    int*   __restrict__ v_palette_tags,            // [total_heads * 4]
    float* __restrict__ k_palette_scale,           // [total_heads * 4] outer scale per slot
    float* __restrict__ v_palette_scale,           // [total_heads * 4] outer scale per slot
    int*   __restrict__ k_palette_map,             // [total_heads * HB]
    int*   __restrict__ v_palette_map,             // [total_heads * HB]
    int*   __restrict__ k_effective_block_tags,    // [total_heads * HB]
    int*   __restrict__ v_effective_block_tags,    // [total_heads * HB]
    int*   __restrict__ k_head_tags,               // [total_heads]
    int*   __restrict__ v_head_tags,               // [total_heads]
    float* __restrict__ q_relevance_out            // [total_heads * HB] or nullptr
) {
    // Derived layout constants: AW u64 words carry the HB-bit alive/pass
    // masks; NSEG 32-position sort segments cover the HB sorted entries.
    constexpr int AW   = HB / 64;
    constexpr int NSEG = HB / FUSED_WARP_SIZE;
    // Single aliased buffer (f16): K data lives here through Phase 2.5 + K
    // process_side, then V data is reloaded before V process_side.  f16 halves
    // the 16 KB of the prior f32 buffer; the remaining precision is sufficient
    // for both the sink-score dot product and the roundtrip-error comparison.
    // Combined with MaxShared this is the binding smem constraint for 5 blocks/SM.
    __shared__ __half   smem_kv    [HB * FUSED_WARP_SIZE];
    // q_vals_half removed: q_mean is now computed inline during Phase 1 via
    // warp reduce, saving 8 KB and eliminating the Phase 2.5(a) readback loop.
    __shared__ float    amax_k     [HB];
    __shared__ float    amax_v     [HB];
    // Block ids fit in 9 bits (0..HB-1) — use uint16_t to halve smem footprint
    // and free us from the prior int-tombstone (-1) sentinel; tombstoning now
    // lives in the alive bitmasks below.
    __shared__ uint16_t kidx       [HB];
    __shared__ uint16_t vidx       [HB];
    // f16 halves the footprint vs f32; precision is adequate for threshold comparisons.
    __shared__ __half   qrel_k     [HB];
    __shared__ __half   kthresh    [HB];
    __shared__ float    q_mean     [HB];   // Phase 2.5 — mean Q per head_dim slot (head_dim = HB)
    __shared__ float    sink_score [FUSED_WARP_SIZE];     // Phase 2.5 — raw per-token Q·K alignment
    __shared__ float    sink_weight[FUSED_WARP_SIZE];     // Phase 2.5 — per-token weight ∈ [0, 1]
    // Per-warp round-trip scratch — each warp owns one row, so all 4 warps
    // can run independent (fmt, scale, block) round-trips in parallel
    // without contending for buffer space.
    __shared__ float    warp_f32   [FUSED_WARPS_PER_BLOCK][FUSED_WARP_SIZE];
    __shared__ uint8_t  warp_quant [FUSED_WARPS_PER_BLOCK][MAX_QUANT_BLOCK_BYTES];
    __shared__ int      slot_tags  [4];
    __shared__ float    slot_scales[4];
    // HB-bit alive mask per side (AW u64 words); bit b set iff K/V block b is
    // unclaimed. Replaces the prior idx_sorted[i] = -1 tombstoning. Updated
    // atomically by `alive_clear` since multiple lanes can target distinct
    // bits during the parallel second-pass fill.
    __shared__ uint64_t k_alive[AW];
    __shared__ uint64_t v_alive[AW];
    // HB-bit flatness mask per side; bit b set iff K/V block b is flat to within
    // one INT8 step of the head scale (`block_var_is_flat`). Written once, in the
    // pass that first loads the side's values, and read by the Q0 search.
    __shared__ uint64_t k_flat[AW];
    __shared__ uint64_t v_flat[AW];
    // Compacted live-index scratch, rebuilt at the top of each slot from
    // `kidx`/`vidx` + the alive bitmask. The search inner loop walks
    // idx_compact[0..live_count) instead of idx_sorted with a per-block
    // alive probe, so the hot path becomes branch-free over the iteration
    // count. Reused across K and V because the two process_side calls run
    // sequentially.
    __shared__ uint16_t idx_compact[HB];

    // Cross-warp search aggregation — one row per (warp, si) pair. Every scale
    // is accumulated in the same pass over the blocks without intermediate
    // __syncthreads; tid 0 reduces all rows in one pass after a single
    // __syncthreads at the end of the search. Pass counts are the merged
    // masks' populations.
    // Layout: [warp_id * NUM_SCALE_CANDIDATES + si] (× AW words for warp_pass).
    __shared__ uint64_t warp_pass    [FUSED_WARPS_PER_BLOCK * NUM_SCALE_CANDIDATES * AW];
    __shared__ float    warp_amax_err[FUSED_WARPS_PER_BLOCK * NUM_SCALE_CANDIDATES];
    __shared__ int      warp_nonflat [FUSED_WARPS_PER_BLOCK];

    // Cross-thread best/fallback state for the current slot's search.
    // Lives in shared because the search is now multi-warp: every warp
    // reads `s_search_done` to decide whether to advance to the next
    // candidate, and tid 0 updates the best/fallback fields after the
    // cross-warp reduction.
    __shared__ int      s_best_fmt;
    __shared__ float    s_best_scale;
    __shared__ float    s_best_err;
    __shared__ uint64_t s_best_pass[AW];
    __shared__ int      s_search_done;
    __shared__ int      s_fallback_fmt;
    __shared__ float    s_fallback_scale;
    __shared__ float    s_fallback_err;
    __shared__ uint64_t s_fallback_pass[AW];
    // Early rejection of hopeless scales (`search_scales_for_fmt`): failure
    // counts per scale (the last entry counts against every scale), which
    // scales some warp stopped measuring, and whether that left the fallback
    // unknown. The search re-zeroes the first two on exit.
    __shared__ int      s_fail[NUM_SCALE_CANDIDATES + 1];
    __shared__ int      s_dropped;
    __shared__ int      s_fallback_incomplete;

    // Per-slot scratch shared with all threads: the live population of each
    // 32-position sort segment (segments = warps at HB=128; each warp owns
    // NSEG/4 segments at HB=256), and the slot-level amax statistics (max, p95,
    // p80, mean, p25) of the unclaimed set, computed by tid 0 from the
    // compacted amaxes and broadcast to all threads.
    __shared__ int      s_seg_pop[NSEG];
    __shared__ float    s_slot_amax;   // max amax of unclaimed set
    __shared__ float    s_slot_p95;    // 95th-percentile amax (5% of blocks exceed this)
    __shared__ float    s_slot_p80;    // 80th-percentile amax (20% of blocks exceed this)
    __shared__ float    s_slot_mean;   // mean amax of unclaimed set
    __shared__ float    s_slot_p25;    // 25th-percentile amax (75% of blocks exceed this)

    // Per-band source views, resolved through each band's own gid (bands of a
    // head may live in different arenas, at non-contiguous chunk indices, and
    // in different formats). Living in smem keeps the source pointers/formats
    // out of the register frame across Phase 2.5 / Phase 2 / Phase 3 and both
    // process_side calls — the same occupancy argument as the former V-reload
    // spill slots, which these arrays replace. Index [0] = K, [1] = V.
    __shared__ const char* s_band_ptr  [2][N_PALETTE];
    __shared__ int         s_band_fmt  [2][N_PALETTE];
    __shared__ float       s_band_outer[2][N_PALETTE];

    const int head_id = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid / FUSED_WARP_SIZE;
    const int lane    = tid % FUSED_WARP_SIZE;

    if (head_id >= total_heads || blocks_per_head != HB) return;

    const int chunk_id = head_id / n_kv_head;
    const int head_idx = head_id % n_kv_head;

    // Valid token window of this chunk (see the parameter comment).
    const TokenWindow win = unpack_token_window(__ldg(&valid_ranges[chunk_id]));
    const int valid_lo  = win.lo;
    const int valid_hi  = win.hi;
    const int valid_len = valid_hi - valid_lo;
    // Count corrections for the fixed-count metric normalizations:
    // V pass_metric divides the error sum by 32 lanes; K's by top-4.
    const float v_valid_corr = 32.0f / (float)valid_len;
    const float k_valid_corr = 4.0f / (float)min(4, valid_len);

    // Resolve every band's source view through its own gid; one thread per
    // (side, band). The barrier publishes the views for Phase 1 and V reload.
    if (tid < 2 * N_PALETTE) {
        const int side = tid >> 2; // 0 = K, 1 = V
        const int p    = tid & (N_PALETTE - 1);
        resolve_band_source(per_head_table_raw, head_gids, chunk_id, head_idx,
                            n_kv_head, arena_chunks, p, side != 0,
                            &s_band_ptr[side][p], &s_band_fmt[side][p],
                            &s_band_outer[side][p]);
    }
    if (tid < AW) {
        k_flat[tid] = 0ULL;
        v_flat[tid] = 0ULL;
    }
    __syncthreads();
    // Block → band mapping: block b covers dim b of the head; each band holds
    // FUSED_BAND_BLOCKS consecutive dims' blocks in its own arena slot
    // (32 at HB=128, 64 at HB=256).
    constexpr int FUSED_BAND_BLOCKS = HB / N_PALETTE;

    const float q_med    = __ldg(&q_relevance_median[head_id]);
    const float q_spread = __ldg(&q_relevance_spread[head_id]);

    {
    // One INT8 step of each side's head scale: a block whose variance is below
    // its square is flat (`block_var_is_flat`). Scoped to Phase 1, the only
    // place the flatness masks are written.
    const float k_flat_step = head_scale(__ldg(&k_head_amax_in[head_id]), __ldg(&k_head_p95_in[head_id])) * (1.0f / FLAT_BLOCK_STEPS);
    const float v_flat_step = head_scale(__ldg(&v_head_amax_in[head_id]), __ldg(&v_head_p95_in[head_id])) * (1.0f / FLAT_BLOCK_STEPS);
    // ── Phase 1: Load all HB blocks; compute per-block amax and q-relevance ──
    // Multi-warp stride: each of the 4 warps owns HB/4 blocks (warp_id, +4, +8 …).
    // Lanes within a warp cooperate to load one block's 32 elements; warp-
    // local reductions (`__shfl_xor_sync`) compute amax / q-relevance per
    // block. No cross-warp dependencies in this phase — the closing
    // __syncthreads makes Phase 2.5/2 see consistent smem state.
    //
    // A warp loads LOAD_GROUP of its blocks before it processes any of them, so
    // their global loads are in flight together rather than one latency apiece.
    // A group spans LOAD_GROUP × 4 consecutive block ids, which stays inside one
    // band (FUSED_BAND_BLOCKS is 32 or 64), so the group shares one source view.
    constexpr int LOAD_GROUP = 4;
    static_assert(FUSED_BAND_BLOCKS % (LOAD_GROUP * FUSED_WARPS_PER_BLOCK) == 0,
                  "a load group must not straddle a band");
    for (int g = warp_id; g < HB; g += FUSED_WARPS_PER_BLOCK * LOAD_GROUP) {
    float k_in[LOAD_GROUP], q_in[LOAD_GROUP], v_in[LOAD_GROUP];
    load_band_group<LOAD_GROUP, true>(s_band_ptr, s_band_fmt, s_band_outer,
                                      g, FUSED_WARPS_PER_BLOCK, FUSED_BAND_BLOCKS, lane,
                                      k_in, q_in, v_in);
    #pragma unroll
    for (int j = 0; j < LOAD_GROUP; j++) {
        const int blk = g + j * FUSED_WARPS_PER_BLOCK;
        float k_val = k_in[j], q_val = q_in[j], v_val = v_in[j];
        if (!lane_in_window(lane, win)) {
            k_val = 0.0f;
            q_val = 0.0f;
            v_val = 0.0f;
        }

        smem_kv   [blk * 32 + lane] = __float2half(k_val);   // K stored as f16; V discarded (reloaded before V process_side)

        float k_abs = fabsf(k_val);
        float v_abs = fabsf(v_val);
        // First and second moments of both sides ride along in the amax
        // reduction (dead lanes are zero, so the sums are over the valid
        // tokens): they decide each block's flatness below at no extra latency.
        float k_s1 = k_val, k_s2 = k_val * k_val;
        float v_s1 = v_val, v_s2 = v_val * v_val;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            k_abs = fmaxf(k_abs, __shfl_xor_sync(0xffffffff, k_abs, off));
            v_abs = fmaxf(v_abs, __shfl_xor_sync(0xffffffff, v_abs, off));
            k_s1 += __shfl_xor_sync(0xffffffff, k_s1, off);
            k_s2 += __shfl_xor_sync(0xffffffff, k_s2, off);
            v_s1 += __shfl_xor_sync(0xffffffff, v_s1, off);
            v_s2 += __shfl_xor_sync(0xffffffff, v_s2, off);
        }
        if (lane == 0) {
            const float n_inv = 1.0f / (float)valid_len;
            const float k_mean = k_s1 * n_inv;
            const float v_mean = v_s1 * n_inv;
            if (block_var_is_flat(k_s2 * n_inv - k_mean * k_mean, k_flat_step))
                atomicOr((unsigned long long*)&k_flat[blk >> 6], 1ULL << (blk & 63));
            if (block_var_is_flat(v_s2 * n_inv - v_mean * v_mean, v_flat_step))
                atomicOr((unsigned long long*)&v_flat[blk >> 6], 1ULL << (blk & 63));
        }

        // Inline q_mean: all 32 lanes hold q_val for one position in this block,
        // so a warp sum gives the mean without a staging buffer.  Saves 8 KB vs.
        // storing to q_vals_half and reading back in Phase 2.5(a).
        float q_mean_sum = q_val;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            q_mean_sum += __shfl_xor_sync(0xffffffff, q_mean_sum, off);
        if (lane == 0) q_mean[blk] = q_mean_sum * (1.0f / 32.0f);

        float qr = 1.0f;
        const int has_q = __any_sync(0xffffffff, q_val != 0.0f);
        if (has_q) {
            qr = block_relevance(k_val, q_val);
            qr = __shfl_sync(0xffffffff, qr, 0);
        }

        if (lane == 0) {
            // Per-block hash jitter (~6e-8) added to the amax values to
            // break sort ties on tied amax. Partial-tail chunks read as
            // zero outside their token window, producing many near-equal
            // small amax that — under the bitonic sort's tie behaviour —
            // drift toward near-monotonic block-index order; the claim
            // phase then assigns long contiguous dim ranges to a single
            // palette (tail-chunk clustering). The jitter sits well
            // below any real activation amax (≥ 1e-2 typical), so the
            // format-search and threshold paths on real data are
            // unaffected. Must match the Rust mirror's
            // `amax_tie_jitter` in `cpu_selection.rs` byte-for-byte.
            const unsigned int j_h = ((unsigned int)blk * 2654435761u) ^ 0x9e3779b9u;
            const float j_v = __int_as_float((int)((j_h & 0x007fffffu) | 0x33800000u));
            amax_k[blk] = k_abs + j_v;
            amax_v[blk] = v_abs + j_v;
            qrel_k[blk] = __float2half(qr);
            kidx  [blk] = (uint16_t)blk;
            vidx  [blk] = (uint16_t)blk;
        }

        if (lane == 0 && q_relevance_out != nullptr) {
            q_relevance_out[head_id * HB + blk] = qr;
        }
    }
    }

    // Initialise alive masks; the __syncthreads below makes them visible to
    // all threads. (V-reload source views live in the s_band_* arrays.)
    if (tid == 0) {
        // Initialize alive masks: all HB bits set (AW full u64 words — HB is
        // a multiple of 64, so there are no partial-word tail bits).
        #pragma unroll
        for (int w = 0; w < AW; w++) {
            k_alive[w] = ~0ULL;
            v_alive[w] = ~0ULL;
        }
    }
    }  // ── end Phase 1 scope ──
    __syncthreads();

    // ── Phase 2.5: per-token attention-sink detection ─────────────────────────
    // Compute Q-K alignment-based sink score per token, then statistically
    // detect sinks via a chunk-local z-score with tanh weighting.
    //
    // Mechanism: tokens whose K vector aligns with the average Q direction
    // of the chunk receive more attention from typical queries. This is a
    // pre-softmax proxy for attention received, capturing both K-norm
    // anomalies (registration sinks at sequence start) and K-direction
    // anomalies (mid-sequence emergent sinks documented in KVSink).
    //
    // Output: sink_weight[t] ∈ [0,1] per token. Above-average tokens get
    // positive weight (saturating at z≈2); below-average tokens get 0.
    // Parameter-free: detection threshold, sharpness, and floor are all
    // derived from chunk-local statistics via tanh(z-score).
    //
    // Used by V-side selection to lerp the per-block error threshold between
    // v_threshold_lo (lenient) and v_threshold_hi (strict) using the maximum
    // sink_weight among the 32 tokens in the block — one strong sink in a
    // block forces tighter quality on the entire block.

    // q_mean is populated per-block during Phase 1 (inline warp reduce).
    // Phase 2.5(a) smem readback loop removed; the existing __syncthreads
    // above already makes q_mean visible to Phase 2.5(b).

    // (b) Per-token sink_score: dot product of q_mean[0..HB-1] with K[blk, token].
    //     Each lane computes one token's score; all 4 warps participate.
    //     warp_f32 is idle here (only used inside search_scales_for_fmt) so
    //     we reuse it as partial-sum scratch without extra smem cost.
    //     Each warp accumulates HB/FUSED_WARPS_PER_BLOCK consecutive blocks
    //     (= head dims); warp 0 then sums the 4 partials and continues with
    //     stats + weight.
    {
        float score = 0.0f;
        const int blk_begin = warp_id * (HB / FUSED_WARPS_PER_BLOCK);
        const int blk_end   = blk_begin + (HB / FUSED_WARPS_PER_BLOCK);
        #pragma unroll 4
        for (int blk = blk_begin; blk < blk_end; blk++) {
            score += q_mean[blk] * __half2float(smem_kv[blk * 32 + lane]);
        }
        warp_f32[warp_id][lane] = score;
    }
    __syncthreads();
    // (c)+(d) μ/σ statistics and sink_weight: warp-wide reductions, warp 0 only.
    if (warp_id == 0) {
        // Sum the 4 partial dot products into the final per-token score.
        float score = warp_f32[0][lane] + warp_f32[1][lane]
                    + warp_f32[2][lane] + warp_f32[3][lane];
        sink_score[lane] = score * rsqrtf((float)HB);
        __syncwarp();

        // (c) Chunk-local statistics: mean and std of sink_score across the
        //     VALID tokens (dead lanes of a partial chunk have zero K and
        //     would drag mu toward zero, granting every real token spurious
        //     sink weight). Warp reductions; mu and sigma broadcast.
        const bool  in_window = lane_in_window(lane, win);
        const float s = sink_score[lane];
        float ssum = in_window ? s : 0.0f;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            ssum += __shfl_xor_sync(0xffffffff, ssum, off, 32);
        const float mu = ssum / (float)valid_len;

        const float dev = in_window ? (s - mu) : 0.0f;
        float dev2 = dev * dev;
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            dev2 += __shfl_xor_sync(0xffffffff, dev2, off, 32);
        const float sigma = sqrtf(dev2 / (float)valid_len);

        // (d) Per-token sink_weight via z-score → tanh, clamped at zero.
        //     Below-average tokens (z < 0) get 0 (no protection).
        //     Above-average tokens get tanh(z) ∈ (0, 1), saturating at z≈2.
        //     Dead lanes get no weight — a zero K vector is not a sink.
        const float safe_sigma = fmaxf(sigma, 1.0e-8f);
        const float z = (s - mu) / safe_sigma;
        sink_weight[lane] = in_window ? fmaxf(0.0f, tanhf(z)) : 0.0f;
    }
    __syncthreads();

    // ── Phase 2.5 (post): Hoist V-side threshold² out of the search loop ───
    // `v_thr_sq` is the constant V-side `thr_to_compare` used by every block
    // round-trip in process_side. It depends only on `sink_weight` (now
    // finalised) and the two threshold knobs — i.e. it's invariant across
    // (slot, fmt, scale, block). Computing it once here saves ~15K redundant
    // warp-max reductions per head in `compute_pass_metric`'s V-side branch.
    //
    // All 4 warps redundantly compute the same value from the shared
    // `sink_weight[0..31]` — cheaper than a smem broadcast for a single
    // warp-max + scalar arithmetic.
    float v_w_max_local = sink_weight[lane];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v_w_max_local = fmaxf(v_w_max_local, __shfl_xor_sync(0xffffffff, v_w_max_local, off, 32));
    const float v_thr_eff = v_threshold_lo + v_w_max_local * (v_threshold_hi - v_threshold_lo);
    const float v_thr_sq  = v_thr_eff * v_thr_eff;

    // ── Phase 2: Sort K and V blocks separately by their respective amax desc ──
    // Warp 0 sorts K, warp 1 sorts V — runs concurrently on different smem
    // arrays. The bitonic network is warp-local (lane-owned 4 elements
    // each, intra-warp `__shfl_xor` substitute for syncthreads); warps 2/3
    // idle through the sort.
    if (warp_id == 0) {
        bitonic_sort_amax_desc<HB>(amax_k, kidx, lane);
    } else if (warp_id == 1) {
        bitonic_sort_amax_desc<HB>(amax_v, vidx, lane);
    }
    __syncthreads();

    // ── Phase 3: Precompute per-block K error thresholds (all 128 threads stride the HB blocks) ──
    for (int b = tid; b < HB; b += FUSED_THREADS_PER_BLOCK) {
        kthresh[b] = __float2half((q_spread > 1.0e-8f)
            ? k_threshold_scaled(k_threshold_lo, k_threshold_hi, __half2float(qrel_k[b]), q_med, q_spread)
            : sqrtf(k_threshold_lo * k_threshold_hi));
    }

    const float k_head_amax_val = head_scale(__ldg(&k_head_amax_in[head_id]), __ldg(&k_head_p95_in[head_id]));
    // v_head_amax_val loaded lazily just before V process_side so it is not live
    // across the K process_side register frame.

    // Defensive: pre-zero palette_map so unclaimed blocks (shouldn't occur under normal
    // operation) always have a valid slot index rather than uninitialized device memory.
    for (int b = tid; b < HB; b += FUSED_THREADS_PER_BLOCK) {
        k_palette_map[head_id * HB + b] = 0;
        v_palette_map[head_id * HB + b] = 0;
    }
    __syncthreads();

    // ── Phase 4+5: Iterative slot search (process_side lambda) ──────────
    //
    // Runs once for K and once for V. For each of 4 slots:
    //
    //   (a) Compact alive entries from idx_sorted into idx_compact, with
    //       their amaxes into amax_compact. Each warp ballots its 32-position
    //       segment(s) of the alive bitmask (one segment per warp at HB=128,
    //       two at HB=256); per-segment prefix-popcount packs ranks, and every
    //       thread adds its segment's offset from the cross-segment prefix
    //       sum. tid 0 then reads the slot stats (amax, p95, p80, mean, p25)
    //       off amax_compact and broadcasts them via shared memory; these feed
    //       preferred_range to produce the six outer-scale candidates for the
    //       format search.
    //
    //   (b) Reset best/fallback state. `s_best_err = FLT_MAX`,
    //       `s_fallback_err = FLT_MAX` (so the first measurement wins).
    //
    //   (c) Search: scan candidates BPE-ascending (most aggressive
    //       first). For each (fmt, scale), `search_scales_for_fmt`
    //       counts how many live blocks pass and accumulates their
    //       pass mask. Once any (fmt, scale) hits the FUSED_BAND_BLOCKS
    //       quota, `s_search_done` is set and the outer ci-loop breaks.
    //       If no candidate hits the quota, the lowest-max-error fallback
    //       wins (with its partial pass mask).
    //
    //   (d) Claim phase (warp 0): walk idx_compact, claim blocks whose
    //       pass-mask bit is set (no round-trip), up to FUSED_BAND_BLOCKS.
    //       Then a second-pass fill sweeps in remaining alive blocks in sort
    //       order until the quota is claimed.
    //       Both phases tombstone via `alive_clear`, so the next slot's
    //       compaction skips them.
    //
    // What it modifies (per side):
    //   - `k_alive`/`v_alive` — tombstones cleared as blocks are claimed
    //   - `slot_tags[s]`,
    //     `slot_scales[s]`    — overwritten per slot
    //   - `out_pal_map`,
    //     `out_eff_tags`      — written per (head, block)
    //   - `out_pal_tags`,
    //     `out_pal_scale`,
    //     `out_head_tag`      — written per head at the end
    //
    // What it reads (must persist across K → V invocations):
    //   - `smem_data`         — read-only (smem_k for K, smem_v for V)
    //   - `kthresh`           — read-only, K side only
    //   - `sink_weight`       — read-only via the hoisted `v_thr_sq`
    //
    // Tombstoning lives in the alive mask (one bit per block, AW u64
    // words); idx_sorted is read-only after the bitonic sort. Live count
    // is a word-wise `__popcll` sum; "find first alive in sort order"
    // uses a 32-lane chunked __ballot_sync scan instead of a serial
    // walk.
    //
    // `q_mean` is dead once Phase 2.5 has its sink scores, so the slot loop
    // reuses it for the live amaxes in sort order.
    float* const amax_compact = q_mean;
    auto process_side = [&](
        const __half* __restrict__ smem_data,  // flat [HB * FUSED_WARP_SIZE] f16 values
        float*    __restrict__ amax_sorted, // [HB] desc-sorted amax, read-only
        const uint16_t* __restrict__ idx_sorted,  // [HB] original indices (read-only after sort)
        uint64_t* alive,                    // → __shared__ alive bitmask [AW]
        const uint64_t* flat,               // → __shared__ flatness bitmask [AW]
        const int* cands,
        int    num_cands,
        float  head_amax,
        bool   is_k,
        int*   __restrict__ out_pal_tags,
        float* __restrict__ out_pal_scale,
        int*   __restrict__ out_pal_map,
        int*   __restrict__ out_eff_tags,
        int*   __restrict__ out_head_tag
    ) {
        // Hoisted per-side constants used by every search/claim metric eval.
        // `inv_head_amax` replaces the prior `err / head_amax` divide; on
        // the V side the squared form is used against pre-squared `v_thr_sq`.
        // The valid-count corrections are folded in here: partial chunks'
        // dead lanes are zero and contribute no error, so multiplying by
        // 4/min(4,len) (K top-4 mean) and 32/len (V mean-over-32 MSE)
        // restores per-VALID-token normalization without threading the
        // count through the metric plumbing.
        const float inv_head_amax    = (1.0f / head_amax) * k_valid_corr;
        const float inv_head_amax_sq =
            (1.0f / head_amax) * (1.0f / head_amax) * v_valid_corr;

        // 32-position sort segments each warp owns during compaction
        // (1 at HB=128 — the segment IS the warp's position chunk — 2 at 256).
        constexpr int SEGS_PER_WARP = NSEG / FUSED_WARPS_PER_BLOCK;

        for (int s = 0; s < 4; s++) {
            // Snapshot the alive mask so search can read consistent state
            // without re-loading from shared on every block. (alive_clear
            // writes happen in the claim phase, after search completes.)
            uint64_t alive_snap[AW];
            #pragma unroll
            for (int w = 0; w < AW; w++) alive_snap[w] = alive[w];
            // Live blocks a flat-only format may be offered (see the search).
            int flat_live = 0;
            #pragma unroll
            for (int w = 0; w < AW; w++) flat_live += __popcll(alive_snap[w] & flat[w]);

            // ── Compact alive entries into idx_compact, all 4 warps in parallel ──
            // Each warp owns SEGS_PER_WARP 32-position segments (segment g
            // covers positions [g*32, g*32+32)). Within a segment: ballot the
            // alive mask, prefix-popcount gives each lane a packed rank inside
            // the segment. Cross-segment scan combines per-segment counts into
            // a global offset; each segment's first-alive position is also
            // recorded for the first-alive lookup used by tid 0's stat walk.
            int  seg_b   [SEGS_PER_WARP];
            bool seg_live[SEGS_PER_WARP];
            int  seg_rank[SEGS_PER_WARP];
            #pragma unroll
            for (int c = 0; c < SEGS_PER_WARP; c++) {
                const int  seg       = c * FUSED_WARPS_PER_BLOCK + warp_id;
                const int  pos       = seg * FUSED_WARP_SIZE + lane;
                const int  b_at_pos  = idx_sorted[pos];
                const bool is_alive  = alive_get<AW>(alive_snap, b_at_pos);
                const unsigned bal   = __ballot_sync(0xffffffff, is_alive);
                seg_b   [c] = b_at_pos;
                seg_live[c] = is_alive;
                seg_rank[c] = __popc(bal & ((1u << lane) - 1u));
                if (lane == 0) s_seg_pop[seg] = __popc(bal);
            }
            __syncthreads();

            // Every thread derives its segments' write offsets and the live count
            // from the NSEG segment populations (an exclusive prefix sum over a
            // handful of words, cheaper than a barrier round trip through one
            // thread), then writes its live entries in sort order: the block id
            // to idx_compact and its amax to amax_compact.
            int live_count = 0;
            int seg_off[SEGS_PER_WARP];
            #pragma unroll
            for (int g = 0; g < NSEG; g++) {
                #pragma unroll
                for (int c = 0; c < SEGS_PER_WARP; c++)
                    if (g == c * FUSED_WARPS_PER_BLOCK + warp_id) seg_off[c] = live_count;
                live_count += s_seg_pop[g];
            }
            #pragma unroll
            for (int c = 0; c < SEGS_PER_WARP; c++) {
                if (seg_live[c]) {
                    const int pos  = (c * FUSED_WARPS_PER_BLOCK + warp_id) * FUSED_WARP_SIZE + lane;
                    const int rank = seg_off[c] + seg_rank[c];
                    idx_compact [rank] = (uint16_t)seg_b[c];
                    amax_compact[rank] = amax_sorted[pos];
                }
            }
            __syncthreads();

            // tid 0 computes the slot stats (amax, p95, p80, mean, p25) of the
            // live set from its amaxes in sort order. amax_sorted is descending,
            // so rank r holds the (r+1)-th largest; each percentile is one read,
            // and the mean is a sum in sort order.
            if (tid == 0) {
                const int lc = live_count;
                float sum_amax = 0.0f;
                for (int r = 0; r < lc; r++) sum_amax += amax_compact[r];
                const float slot_amax_raw = (lc > 0) ? amax_compact[0] : 0.0f;
                const int p95_tgt = max(1, (lc + 19) / 20);  // ceil(5% of lc)
                const int p80_tgt = max(1, (lc + 4)  /  5);  // ceil(20% of lc)
                const int p25_tgt = max(1, (3 * lc)  /  4);  // floor(75% of lc)
                const float p95_val = (lc > 0) ? amax_compact[p95_tgt - 1] : 0.0f;
                const float p80_val = (lc > 0) ? amax_compact[p80_tgt - 1] : 0.0f;
                const float p25_val = (lc > 0) ? amax_compact[p25_tgt - 1] : 0.0f;
                s_slot_amax = fmaxf(slot_amax_raw, 1.0e-8f);
                s_slot_p95  = fmaxf(p95_val,       1.0e-8f);
                s_slot_p80  = fmaxf(p80_val,       1.0e-8f);
                s_slot_mean = (lc > 0) ? fmaxf(sum_amax / lc, 1.0e-8f) : 1.0e-8f;
                s_slot_p25  = fmaxf(p25_val,       1.0e-8f);
            }
            __syncthreads();

            const float slot_amax  = s_slot_amax;
            const float safe_p95   = s_slot_p95;
            const float safe_p80   = s_slot_p80;
            const float slot_mean  = s_slot_mean;
            const float safe_p25   = s_slot_p25;

            if (live_count == 0) {
                if (tid == 0) {
                    slot_tags  [s] = cands[num_cands - 1];
                    // Unclaimed slot: write identity scale (1.0). Encode multiplies and
                    // decode divides — both are no-ops at 1.0, so an unreferenced slot
                    // can't corrupt arena bytes or produce NaN/Inf at decode.
                    slot_scales[s] = 1.0f;
                }
                __syncthreads();
                continue;
            }

            // ── Reset best-of-slot search state ──────────────────────────
            // Initial best: the highest-BPE candidate at scale 1.0 with
            // err = FLT_MAX, so the first real measurement always wins.
            // Initial fallback: same seed; err = FLT_MAX guarantees the first
            // (fmt, scale) measurement wins. Fallback now tracks the candidate
            // with the lowest max error across all alive blocks (passing and
            // failing) — see `search_scales_for_fmt`.
            if (tid == 0) {
                s_best_fmt         = cands[num_cands - 1];
                s_best_scale       = 1.0f;
                s_best_err         = FLT_MAX;
                s_search_done      = 0;
                s_fallback_fmt     = cands[num_cands - 1];
                s_fallback_scale   = 1.0f;
                s_fallback_err     = FLT_MAX;
                #pragma unroll
                for (int w = 0; w < AW; w++) {
                    s_best_pass[w]     = 0;
                    s_fallback_pass[w] = 0;
                }
                #pragma unroll
                for (int si = 0; si <= NUM_SCALE_CANDIDATES; si++) s_fail[si] = 0;
                s_dropped             = 0;
                s_fallback_incomplete = 0;
            }
            __syncthreads();

            // ── Search: BPE-ascending format × NUM_SCALE_CANDIDATES scales ──
            // Per-candidate dispatch: `with_select_fmt` resolves the runtime
            // fmt to a compile-time FmtTag, and the side rides along as a
            // runtime argument. Inside `search_scales_for_fmt`, both quant and
            // dequant paths fully inline — no per-block fmt branching, no
            // __noinline__ slow-path call for cold formats.
            //
            // The 4 warps stride through live_count blocks in parallel; the
            // helper does a cross-warp reduction (per scale) and updates
            // shared best/fallback state. `s_search_done` is written by tid
            // 0 after a __syncthreads, so all threads see it consistently
            // at the top of the next ci iteration — break is convergent
            // across the block.
            //
            // The first search drops hopeless scales. If no candidate filled the
            // slot and a dropped scale was a fallback candidate, the fallback is
            // unknown, and the search runs again measuring every block: no
            // candidate fills the slot that time either, and the fallback it
            // ranks is the full search's.
            for (int attempt = 0; attempt < 2; attempt++) {
                const bool drop_hopeless = (attempt == 0);
                for (int ci = 0; ci < num_cands; ci++) {
                    if (s_search_done) break;
                    const int fmt = cands[ci];
                    // One instantiation per format, with the side passed in. This
                    // used to branch on `is_k` here and instantiate the whole search
                    // twice per format — see `search_scales_for_fmt`.
                    with_select_fmt(fmt, [&](auto tag) {
                        constexpr int FMT = decltype(tag)::value;
                        // A flat-only format passes only flat blocks. With fewer
                        // flat live blocks than the slot quota it cannot fill the
                        // slot, and with a non-flat block live it is no fallback
                        // either, so its search would change nothing: skip it.
                        if constexpr (flat_only<FMT>::value) {
                            if (flat_live < FUSED_BAND_BLOCKS) return;
                        }
                        search_scales_for_fmt<FMT, HB>(
                            is_k,
                            slot_amax, safe_p95, safe_p80, slot_mean, safe_p25,
                            smem_data, idx_compact, live_count,
                            tid, warp_id, lane,
                            warp_f32[warp_id], warp_quant[warp_id], kthresh,
                            inv_head_amax, inv_head_amax_sq, v_thr_sq,
                            flat, warp_nonflat,
                            warp_pass, warp_amax_err,
                            &s_best_fmt, &s_best_scale, &s_best_err,
                            s_best_pass, &s_search_done,
                            &s_fallback_fmt, &s_fallback_scale, &s_fallback_err,
                            s_fallback_pass,
                            drop_hopeless, s_fail, &s_dropped, &s_fallback_incomplete);
                    });
                }
                if (s_search_done || !s_fallback_incomplete) break;
                __syncthreads();
                if (tid == 0) {
                    s_fallback_fmt        = cands[num_cands - 1];
                    s_fallback_scale      = 1.0f;
                    s_fallback_err        = FLT_MAX;
                    #pragma unroll
                    for (int w = 0; w < AW; w++) s_fallback_pass[w] = 0;
                    s_fallback_incomplete = 0;
                }
                __syncthreads();
            }

            // No candidate hit the slot quota — fall back to the (fmt, scale)
            // with the lowest max error across all alive blocks (passing and
            // failing), with its best scale and its (partial) pass mask so
            // the claim phase still claims any blocks that did pass at the
            // fallback.
            if (tid == 0 && !s_search_done) {
                s_best_fmt   = s_fallback_fmt;
                s_best_scale = s_fallback_scale;
                #pragma unroll
                for (int w = 0; w < AW; w++) s_best_pass[w] = s_fallback_pass[w];
            }
            __syncthreads();

            const int   best_fmt   = s_best_fmt;
            const float best_scale = s_best_scale;
            // The winning pass mask stays in s_best_pass (shared); the claim
            // phase reads it directly — it is not overwritten until the next
            // slot's reset, which happens after another __syncthreads.

            // `best_scale` is the exact `outer` value that won the search.
            // Encoder must multiply by it; decoder must divide by it. This holds
            // uniformly across quant and float formats — the float-format encode
            // path must apply the same scale or the round-trip won't match.
            if (tid == 0) {
                slot_tags  [s] = best_fmt;
                slot_scales[s] = best_scale;
            }
            __syncthreads();

            // ── Claim, warp 0 ──────────────────────────────────────────────
            // Claim ORDER matters — slot s+1 inherits the next-largest amax
            // blocks — so both passes walk idx_compact from the top in 32-entry
            // chunks. Within a chunk, lanes ballot their predicate and a
            // prefix popcount ranks them, so up to `still_need` lanes commit in
            // parallel while preserving sort order. Parallelising across warps
            // would race the chunk pointer and reorder the claim.
            //
            // Pass 1 claims the blocks whose bit is set in the cached pass mask:
            // the search already measured them at the winning (fmt, scale), so
            // there is no round trip here. Pass 2 fills the rest of the quota
            // with the remaining live blocks, highest amax first, under the same
            // `best_fmt`.
            if (warp_id == 0) {
                int claimed = 0;
                for (int chunk = 0; chunk < live_count && claimed < FUSED_BAND_BLOCKS; chunk += FUSED_WARP_SIZE) {
                    const int  i       = chunk + lane;
                    const int  b       = (i < live_count) ? (int)idx_compact[i] : -1;
                    const bool passes  = (b >= 0) && (((s_best_pass[b >> 6] >> (b & 63)) & 1ULL) != 0ULL);
                    const unsigned bal = __ballot_sync(0xffffffff, passes);
                    const int still_need = FUSED_BAND_BLOCKS - claimed;
                    const int rank       = __popc(bal & ((1u << lane) - 1u));
                    if (passes && rank < still_need) {
                        out_pal_map [head_id * HB + b] = s;
                        out_eff_tags[head_id * HB + b] = best_fmt;
                        alive_clear(alive, b);
                    }
                    claimed += min(__popc(bal), still_need);
                }
                // Pass 2 reads the alive bits pass 1 cleared.
                __syncwarp();

                for (int chunk = 0; chunk < live_count && claimed < FUSED_BAND_BLOCKS; chunk += FUSED_WARP_SIZE) {
                    const int  i        = chunk + lane;
                    const int  b        = (i < live_count) ? (int)idx_compact[i] : -1;
                    const bool live     = (b >= 0) && alive_get<AW>(alive, b);
                    const unsigned bal2 = __ballot_sync(0xffffffff, live);
                    if (bal2 == 0) continue;

                    const int chunk_alive = __popc(bal2);
                    const int still_need  = FUSED_BAND_BLOCKS - claimed;
                    const int rank2       = __popc(bal2 & ((1u << lane) - 1u));
                    if (live && rank2 < still_need) {
                        out_pal_map [head_id * HB + b] = s;
                        out_eff_tags[head_id * HB + b] = best_fmt;
                        alive_clear(alive, b);
                    }
                    claimed += min(chunk_alive, still_need);
                    __syncwarp();
                }
            }
            __syncthreads();
        }  // for s in 0..4

        // Write palette tags and scale indices + head_tag in parallel: lanes
        // 0..3 of warp 0 each handle one slot. format_table_index_cuda runs
        // once per lane; warp-max reduce picks the worst (highest
        // table-index) slot.
        if (warp_id == 0) {
            if (lane < 4) {
                out_pal_tags [head_id * 4 + lane] = slot_tags  [lane];
                out_pal_scale[head_id * 4 + lane] = slot_scales[lane];
            }

            const int my_tag = (lane < 4) ? slot_tags[lane] : 0;
            const int my_ti  = (lane < 4) ? format_table_index_cuda(my_tag) : -1;
            int worst_ti  = my_ti;
            int worst_tag = my_tag;
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                const int other_ti  = __shfl_xor_sync(0xffffffff, worst_ti,  off, 32);
                const int other_tag = __shfl_xor_sync(0xffffffff, worst_tag, off, 32);
                if (other_ti > worst_ti) { worst_ti = other_ti; worst_tag = other_tag; }
            }
            if (lane == 0) {
                out_head_tag[head_id] = worst_tag;
            }
        }
        __syncthreads();
    };  // end process_side

    process_side(
        smem_kv, amax_k, kidx,
        k_alive, k_flat,
        k_candidates, num_k_candidates,
        k_head_amax_val, true,
        k_palette_tags, k_palette_scale,
        k_palette_map, k_effective_block_tags, k_head_tags
    );

    // Reload V values into smem_kv.  K data is consumed; overwriting is safe.
    // Per-band source views come from the s_band_* arrays resolved up front,
    // so nothing V-related was live across the K process_side frame. The
    // iterations are independent, so unrolling lets a warp's loads overlap.
    {
        #pragma unroll 4
        for (int blk = warp_id; blk < HB; blk += FUSED_WARPS_PER_BLOCK) {
            const int p   = blk / FUSED_BAND_BLOCKS;
            const int bib = blk - p * FUSED_BAND_BLOCKS;
            const int   v_src_fmt_r    = s_band_fmt[1][p];
            const char* v_chunk_data_r = s_band_ptr[1][p];
            float v_val;
            if (ArenaFormat::is_quantized(v_src_fmt_r)) {
                const char* v_blk = v_chunk_data_r + (int64_t)bib * quant_block_bytes(v_src_fmt_r);
                v_val = dequant_element_inline<float, false>(v_blk, lane, v_src_fmt_r, s_band_outer[1][p]);
            } else {
                v_val = load_as_float(v_chunk_data_r, float_band_elem(bib, lane, FUSED_BAND_BLOCKS), arena_fmt_to_dtype_code(v_src_fmt_r));
            }
            if (!lane_in_window(lane, win)) {
                v_val = 0.0f;
            }
            smem_kv[blk * 32 + lane] = __float2half(v_val);
        }
    }
    __syncthreads();

    // Lazy-load v_head_amax_val just before V process_side so it is not live
    // across the K process_side register frame (saved 1 register there).
    const float v_head_amax_val = head_scale(__ldg(&v_head_amax_in[head_id]), __ldg(&v_head_p95_in[head_id]));
    process_side(
        smem_kv, amax_v, vidx,
        v_alive, v_flat,
        v_candidates, num_v_candidates,
        v_head_amax_val, false,
        v_palette_tags, v_palette_scale,
        v_palette_map, v_effective_block_tags, v_head_tags
    );
}
