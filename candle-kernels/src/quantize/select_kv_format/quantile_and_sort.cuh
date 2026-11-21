// The quantile pass that summarises each (chunk, head), the format table index
// and the bitonic sort by amax.
//
// The quantile kernel, which summarises each (chunk, head) distribution with
// head amax, p95 amax and a 64-bin histogram of the q-relevance signal. The
// format-table index ranks formats by bits per element, and the bitonic sort
// orders a head's HB blocks by amax descending in one warp.

// =============================================================================
// PER-(CHUNK, HEAD) QUANTILE KERNEL
// =============================================================================
// Pass 1 of the selection pipeline. One CUDA block per (chunk, head),
// QREL_WARPS_PER_BLOCK warps. For each (chunk, head), emits:
//
//   k_head_amax_out [head_id]   max |K| across the 128 blocks
//   v_head_amax_out [head_id]   max |V| across the 128 blocks
//   k_head_p95_out  [head_id]   95th percentile of |K|, head-wide
//   v_head_p95_out  [head_id]   95th percentile of |V|, head-wide
//   q_relevance_median_out      median of per-block q-relevance
//   q_relevance_spread_out      IQR-scaled spread (≈ 2·1.4427 · IQR),
//                               used as the effective sigma in the
//                               z-scoring of q-relevance for K-side
//                               threshold scaling
//
// All quantiles are approximated via 64-bin histograms over the head's
// 4096 elements (128 blocks × 32 lanes). Bin width is amax / 63, so
// resolution is roughly 1.6% of amax — adequate for the downstream
// threshold/scale scaling, much cheaper than a true sort.
//
// Three histogram passes, run sequentially because they reuse the same
// shared memory:
//
//   (1)  K |x| histogram          →  k_head_p95_out
//   (2)  V |x| histogram          →  v_head_p95_out
//   (3)  q-relevance histogram (only when at least one block has Q),
//        bounded by [head_min, head_max] of the per-block q-relevance
//        scalars                  →  q1, median, q3  →  spread
//
// q-relevance is computed only for blocks whose source K is R16 (the
// format that carries Q); for other source formats q_val is uniformly
// 0, has_q is false, and only the amax + p95 statistics are emitted.
// A degenerate-distribution guard (sample_count ≤ 1 or no spread)
// returns median = spread = 0, which the selection kernel treats as
// "skip the z-scaling and use the geometric-mean threshold directly".

#define QREL_WARPS_PER_BLOCK SELECT_WARPS_PER_BLOCK
#define QREL_QUANTILE_THREADS (QREL_WARPS_PER_BLOCK * WARP_SIZE)
#define QREL_HIST_BINS 64
// Blocks per head the pass holds per-block state for: the largest head_dim the
// selection supports (pass 2 dispatches HB ∈ {128, 256}).
#define QREL_MAX_BLOCKS 256
// Blocks a warp loads before processing any (`load_band_group`). A group spans
// QREL_LOAD_GROUP × QREL_WARPS_PER_BLOCK = 16 consecutive blocks, inside the
// 32- or 64-block band of a supported head.
#define QREL_LOAD_GROUP 4

__global__ __launch_bounds__(QREL_QUANTILE_THREADS, 8) void approximate_q_relevance_quantiles(
    const int64_t* __restrict__ per_head_table_raw,
    const int64_t* __restrict__ head_gids,
    float* __restrict__ q_relevance_median_out,
    float* __restrict__ q_relevance_spread_out,
    float* __restrict__ k_head_amax_out,  // [total_heads] max(|K|) per (chunk, head)
    float* __restrict__ v_head_amax_out,  // [total_heads] max(|V|) per (chunk, head)
    float* __restrict__ k_head_p95_out,   // [total_heads] 95th pct of |K| per (chunk, head)
    float* __restrict__ v_head_p95_out,   // [total_heads] 95th pct of |V| per (chunk, head)
    int blocks_per_head,
    int total_heads,
    int n_kv_head,
    int arena_chunks,
    const int* __restrict__ valid_ranges  // [n_chunks] packed (offset << 8) | len
) {
    const int head_id       = blockIdx.x;
    const int tid           = threadIdx.x;
    const int warp_in_block = tid / WARP_SIZE;
    const int lane          = tid % WARP_SIZE;
    // Only the head widths pass 2 dispatches are supported: no wider than the
    // per-block state holds, with bands a whole number of load groups.
    if (head_id >= total_heads || blocks_per_head > QREL_MAX_BLOCKS
        || blocks_per_head % (N_PALETTE * QREL_WARPS_PER_BLOCK * QREL_LOAD_GROUP) != 0) return;

    const int chunk_idx = head_id / n_kv_head;
    const int head_idx  = head_id % n_kv_head;
    const TokenWindow win = unpack_token_window(__ldg(&valid_ranges[chunk_idx]));

    // Per-band source views, resolved through each band's own gid. Held in
    // shared memory so the walk loops stay register-neutral under the
    // launch_bounds occupancy cap. Index [0] = K, [1] = V.
    __shared__ const char* s_band_ptr  [2][N_PALETTE];
    __shared__ int         s_band_fmt  [2][N_PALETTE];
    __shared__ float       s_band_outer[2][N_PALETTE];
    if (tid < 2 * N_PALETTE) {
        const int side = tid >> 2; // 0 = K, 1 = V
        const int p    = tid & (N_PALETTE - 1);
        resolve_band_source(per_head_table_raw, head_gids, chunk_idx, head_idx,
                            n_kv_head, arena_chunks, p, side != 0,
                            &s_band_ptr[side][p], &s_band_fmt[side][p],
                            &s_band_outer[side][p]);
    }
    __syncthreads();
    // Block walk → band mapping: block b covers dim b of the head, and each
    // band holds sub_head_dim consecutive dims' blocks in its own slot.
    const int band_blocks = blocks_per_head / N_PALETTE;

    __shared__ float warp_min[QREL_WARPS_PER_BLOCK];
    __shared__ float warp_max[QREL_WARPS_PER_BLOCK];
    __shared__ int   warp_count[QREL_WARPS_PER_BLOCK];
    __shared__ float warp_k_amax[QREL_WARPS_PER_BLOCK];
    __shared__ float warp_v_amax[QREL_WARPS_PER_BLOCK];
    // hist[0..QREL_HIST_BINS)            — K absolute-value histogram
    // hist[QREL_HIST_BINS..2*QREL_HIST_BINS) — V absolute-value histogram
    // Both are populated in a single data pass (4 passes → 3).
    __shared__ int   hist[2 * QREL_HIST_BINS];
    __shared__ float head_min;
    __shared__ float head_max;
    __shared__ int   sample_count;
    __shared__ float smem_head_k_amax;
    __shared__ float smem_head_v_amax;
    // Each block's q-relevance, and whether it has one (its K source carries Q),
    // as the first walk computes them; the q-relevance histogram is built from
    // these rather than from a third read of the head's K and Q.
    __shared__ float   block_qrel  [QREL_MAX_BLOCKS];
    __shared__ uint8_t block_has_q [QREL_MAX_BLOCKS];

    float local_min    = 0.0f;
    float local_max    = 0.0f;
    int   local_count  = 0;
    float local_k_amax = 0.0f;
    float local_v_amax = 0.0f;

    // Each warp loads QREL_LOAD_GROUP of its blocks before processing any of them
    // (`load_band_group`), in the order the one-at-a-time walk took them.
    for (int g = warp_in_block; g < blocks_per_head; g += QREL_WARPS_PER_BLOCK * QREL_LOAD_GROUP) {
    float k_in[QREL_LOAD_GROUP], q_in[QREL_LOAD_GROUP], v_in[QREL_LOAD_GROUP];
    load_band_group<QREL_LOAD_GROUP, true>(s_band_ptr, s_band_fmt, s_band_outer,
                                           g, QREL_WARPS_PER_BLOCK, band_blocks, lane,
                                           k_in, q_in, v_in);
    #pragma unroll
    for (int j = 0; j < QREL_LOAD_GROUP; j++) {
        const int block_in_head = g + j * QREL_WARPS_PER_BLOCK;
        float k_val = k_in[j], q_val = q_in[j], v_val = v_in[j];
        if (!lane_in_window(lane, win)) {
            k_val = 0.0f;
            q_val = 0.0f;
            v_val = 0.0f;
        }

        // All 32 lanes participate in the warp reduce; only lane 0 accumulates.
        float k_block_amax = select_warp_max(fabsf(k_val));
        float v_block_amax = select_warp_max(fabsf(v_val));
        if (lane == 0) {
            local_k_amax = fmaxf(local_k_amax, k_block_amax);
            local_v_amax = fmaxf(local_v_amax, v_block_amax);
        }

        const int has_q = __any_sync(0xffffffff, q_val != 0.0f);
        if (lane == 0) block_has_q[block_in_head] = has_q ? 1 : 0;
        if (has_q) {
            float q_relevance = block_relevance(k_val, q_val);
            q_relevance = __shfl_sync(0xffffffff, q_relevance, 0, 32);
            if (lane == 0) {
                block_qrel[block_in_head] = q_relevance;
                if (local_count == 0) {
                    local_min = q_relevance;
                    local_max = q_relevance;
                } else {
                    local_min = fminf(local_min, q_relevance);
                    local_max = fmaxf(local_max, q_relevance);
                }
                local_count += 1;
            }
        }
    }
    }

    if (lane == 0) {
        warp_min[warp_in_block]    = local_min;
        warp_max[warp_in_block]    = local_max;
        warp_count[warp_in_block]  = local_count;
        warp_k_amax[warp_in_block] = local_k_amax;
        warp_v_amax[warp_in_block] = local_v_amax;
    }
    __syncthreads();

    if (tid == 0) {
        // Accumulate head amax from all warps
        float head_k_amax = warp_k_amax[0];
        float head_v_amax = warp_v_amax[0];
        head_min     = warp_min[0];
        head_max     = warp_max[0];
        sample_count = warp_count[0];
        for (int w = 1; w < QREL_WARPS_PER_BLOCK; ++w) {
            head_k_amax = fmaxf(head_k_amax, warp_k_amax[w]);
            head_v_amax = fmaxf(head_v_amax, warp_v_amax[w]);
            if (warp_count[w] <= 0) continue;
            if (sample_count <= 0) {
                head_min     = warp_min[w];
                head_max     = warp_max[w];
                sample_count = warp_count[w];
                continue;
            }
            head_min      = fminf(head_min, warp_min[w]);
            head_max      = fmaxf(head_max, warp_max[w]);
            sample_count += warp_count[w];
        }

        k_head_amax_out[head_id] = head_k_amax;
        v_head_amax_out[head_id] = head_v_amax;
        smem_head_k_amax = head_k_amax;
        smem_head_v_amax = head_v_amax;

        if (sample_count <= 1 || !(head_max > head_min) || head_max <= 0.0f) {
            q_relevance_median_out[head_id] = 0.0f;
            q_relevance_spread_out[head_id] = 0.0f;
        }
    }
    __syncthreads();

    // === K + V absolute-value histograms (combined pass) → 95th percentiles ===
    // hist[0..QREL_HIST_BINS)            accumulates K |value| frequencies.
    // hist[QREL_HIST_BINS..2*QREL_HIST_BINS) accumulates V |value| frequencies.
    // One data pass replaces the prior two separate passes (4 passes → 3).
    for (int i = tid; i < 2 * QREL_HIST_BINS; i += QREL_QUANTILE_THREADS) hist[i] = 0;
    __syncthreads();
    {
        const float k_amax_safe     = (smem_head_k_amax > 1.0e-8f) ? smem_head_k_amax : 1.0f;
        const float v_amax_safe     = (smem_head_v_amax > 1.0e-8f) ? smem_head_v_amax : 1.0f;
        // |x| bins are spaced by sqrt(|x| / amax), not linearly: a linear bin
        // is amax/63 wide, so when a sink puts the amax 100x above the bulk the
        // whole bulk lands in bin 0 and the percentile cannot be told from 0.
        // Square-root spacing resolves |x| / amax down to ~2.5e-4.
        const float k_inv_amax = __fdiv_rn(1.0f, k_amax_safe);
        const float v_inv_amax = __fdiv_rn(1.0f, v_amax_safe);
        for (int g = warp_in_block; g < blocks_per_head; g += QREL_WARPS_PER_BLOCK * QREL_LOAD_GROUP) {
            float k_in[QREL_LOAD_GROUP], q_in[QREL_LOAD_GROUP], v_in[QREL_LOAD_GROUP];
            load_band_group<QREL_LOAD_GROUP, false>(s_band_ptr, s_band_fmt, s_band_outer,
                                                    g, QREL_WARPS_PER_BLOCK, band_blocks, lane,
                                                    k_in, q_in, v_in);
            // Dead lanes of a partial chunk are not elements of the head: they
            // would count as zeros and drag the percentile toward 0.
            if (lane_in_window(lane, win)) {
                #pragma unroll
                for (int j = 0; j < QREL_LOAD_GROUP; j++) {
                    int k_bin = (int)floorf(__fmul_rn(sqrtf(__fmul_rn(fabsf(k_in[j]), k_inv_amax)), (float)(QREL_HIST_BINS - 1)));
                    k_bin = max(0, min(QREL_HIST_BINS - 1, k_bin));
                    atomicAdd(&hist[k_bin], 1);
                    int v_bin = (int)floorf(__fmul_rn(sqrtf(__fmul_rn(fabsf(v_in[j]), v_inv_amax)), (float)(QREL_HIST_BINS - 1)));
                    v_bin = max(0, min(QREL_HIST_BINS - 1, v_bin));
                    atomicAdd(&hist[QREL_HIST_BINS + v_bin], 1);
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            const int total_elems = blocks_per_head * (win.hi - win.lo);
            const int target_p95  = (int)floorf(0.95f * (float)(total_elems - 1));
            // The percentile is the UPPER edge of the bin it falls in, amax *
            // ((bin + 1) / (BINS - 1))^2, so it never understates |x|'s p95 and
            // the cap it feeds never tightens the head scale below what the
            // data supports. The top bin's edge is clamped to the amax.
            {
                int accum = 0, p95_bin = QREL_HIST_BINS - 1;
                for (int b = 0; b < QREL_HIST_BINS; ++b) {
                    accum += hist[b];
                    if (accum > target_p95) { p95_bin = b; break; }
                }
                const float edge = __fdiv_rn((float)(p95_bin + 1), (float)(QREL_HIST_BINS - 1));
                k_head_p95_out[head_id] = fminf(k_amax_safe, __fmul_rn(k_amax_safe, __fmul_rn(edge, edge)));
            }
            {
                int accum = 0, p95_bin = QREL_HIST_BINS - 1;
                for (int b = 0; b < QREL_HIST_BINS; ++b) {
                    accum += hist[QREL_HIST_BINS + b];
                    if (accum > target_p95) { p95_bin = b; break; }
                }
                const float edge = __fdiv_rn((float)(p95_bin + 1), (float)(QREL_HIST_BINS - 1));
                v_head_p95_out[head_id] = fminf(v_amax_safe, __fmul_rn(v_amax_safe, __fmul_rn(edge, edge)));
            }
        }
    }
    __syncthreads();

    if (sample_count <= 1 || !(head_max > head_min) || head_max <= 0.0f) {
        return;
    }

    for (int i = tid; i < QREL_HIST_BINS; i += QREL_QUANTILE_THREADS) {
        hist[i] = 0;
    }
    __syncthreads();

    // The q-relevance histogram, over the per-block values the first walk kept:
    // one thread per block, no reload of the head's K and Q.
    const float inv_range = __fdiv_rn((float)(QREL_HIST_BINS - 1), head_max - head_min);
    for (int b = tid; b < blocks_per_head; b += QREL_QUANTILE_THREADS) {
        if (block_has_q[b]) {
            int bin = (int)floorf(__fmul_rn(block_qrel[b] - head_min, inv_range));
            bin = max(0, min(QREL_HIST_BINS - 1, bin));
            atomicAdd(&hist[bin], 1);
        }
    }
    __syncthreads();

    if (tid == 0) {
        const int target_q1     = (int)floorf(0.25f * (float)(sample_count - 1));
        const int target_median = (int)floorf(0.50f * (float)(sample_count - 1));
        const int target_q3     = (int)floorf(0.75f * (float)(sample_count - 1));

        int accum = 0;
        int q1_bin = 0, median_bin = 0, q3_bin = QREL_HIST_BINS - 1;
        bool found_q1 = false, found_median = false, found_q3 = false;
        for (int b = 0; b < QREL_HIST_BINS; ++b) {
            accum += hist[b];
            if (!found_q1 && accum > target_q1)         { q1_bin = b;     found_q1 = true; }
            if (!found_median && accum > target_median) { median_bin = b; found_median = true; }
            if (!found_q3 && accum > target_q3)         { q3_bin = b;     found_q3 = true; break; }
        }

        const float scale = __fdiv_rn(head_max - head_min, (float)(QREL_HIST_BINS - 1));
        const float q1_v     = head_min + __fmul_rn((float)q1_bin, scale);
        const float median_v = head_min + __fmul_rn((float)median_bin, scale);
        const float q3_v     = head_min + __fmul_rn((float)q3_bin, scale);

        const float iqr = fmaxf(q3_v - q1_v, 1.0e-8f);
        const float z_scale = 2.0f;
        const float spread = __fmul_rn(iqr, __fmul_rn(z_scale, 1.4426950408889634f));

        q_relevance_median_out[head_id] = median_v;
        q_relevance_spread_out[head_id] = spread;
    }
}

// =============================================================================
// FORMAT TABLE INDEX  —  global BPE ranking
// =============================================================================
// Maps every supported format to a position in a global BPE-ascending
// ranking. Q0 (1 bit, most aggressive) is index 0; BF16 (16 bits, no
// quantization) is index 22.
//
// Used where we need to pick the "most conservative" tag from a set
// without referring to a candidate ordering — specifically:
//
//   - the head_tag computation in the main kernel (worst tag across
//     the 4 slots)
//   - `reduce_head_stats_format` (worst tag across all blocks of a head)
//
// Distinct from `format_bpe_x4`, which returns the actual BPE × 4 (used
// for BPE-ascending search ordering inside a candidate set). The table
// index includes F16/BF16 — useful for the head_tag write-out, where
// any slot's tag could be a float format.
//
// Must be defined before the fused palette4 kernel which uses it
// inside the process_side lambda.
__device__ __forceinline__ int format_table_index_cuda(int fmt) {
    switch (fmt) {
        case SELECT_FMT_Q0:         return 0;
        case SELECT_FMT_Q0_X:       return 1;
        case SELECT_FMT_Q0_V:       return 2;
        case SELECT_FMT_Q0_M2:      return 3;
        case SELECT_FMT_Q1_S:       return 4;
        case SELECT_FMT_Q1_A:       return 5;
        case SELECT_FMT_Q0_M4:      return 6;
        case SELECT_FMT_Q2_S:       return 7;
        case SELECT_FMT_Q2_0:       return 8;
        case SELECT_FMT_Q2_A:       return 9;
        case SELECT_FMT_Q2_1:       return 10;
        case SELECT_FMT_Q3_0:       return 11;
        case SELECT_FMT_Q3_1:       return 12;
        case SELECT_FMT_Q4_0:       return 13;
        case SELECT_FMT_Q4_1:       return 14;
        case SELECT_FMT_Q4_KS:      return 15;
        case SELECT_FMT_Q5_0:       return 16;
        case SELECT_FMT_Q5_1:       return 17;
        case SELECT_FMT_Q8_0:       return 18;
        case SELECT_FMT_Q8_1:       return 19;
        case SELECT_FMT_Q8_KS:      return 20;
        case SELECT_FMT_F16:        return 21;
        case SELECT_FMT_BF16:       return 22;
        default:                    return 20;
    }
}

// =============================================================================
// BITONIC SORT  —  HB entries by amax desc, single warp
// =============================================================================
// Sort phase of the main kernel. After Phase 1 each block has an
// (amax, idx) pair; this routine reorders both arrays so that
// idx_sorted[0] is the block with the largest amax,
// idx_sorted[HB-1] the smallest. Sort positions are deterministic given
// input order.
//
// One warp owns the entire sort: 32 lanes × HB/32 elements per lane
// (4 at HB = 128, 8 at HB = 256). The main kernel uses two of these in
// parallel — warp 0 sorts K's (amax, idx) pair while warp 1 sorts V's;
// warp-local ownership (plus `__syncwarp()` between steps) is what lets
// the two sorts run concurrently. Warps 2 and 3 sit idle until the
// post-sort `__syncthreads()`. The bitonic network is ~O(log² HB) stages
// of O(1) work plus a `__syncwarp()`, so sort is negligible in the
// kernel's overall time.
//
// The compare-exchange network (and therefore the sorted output, ties
// included) is a pure function of HB — per-lane element ownership only
// distributes the disjoint exchanges of each step across lanes.

template <int HB>
__device__ __forceinline__ void bitonic_sort_amax_desc(
    float*    __restrict__ s_amax,
    uint16_t* __restrict__ s_idx,
    int lane
) {
    // Each lane owns HB/32 contiguous elements (32 lanes cover HB total).
    // Standard bitonic sort over HB elements with want_desc ordering.
    constexpr int ELEMS_PER_LANE = HB / FUSED_WARP_SIZE;
    for (int k = 2; k <= HB; k <<= 1) {
        for (int j = k >> 1; j >= 1; j >>= 1) {
            #pragma unroll
            for (int li = 0; li < ELEMS_PER_LANE; li++) {
                const int i = lane * ELEMS_PER_LANE + li;
                const int p = i ^ j;
                if (p > i) {
                    const bool want_desc = ((i & k) == 0);
                    const bool swap = want_desc
                        ? (s_amax[i] < s_amax[p])
                        : (s_amax[i] > s_amax[p]);
                    if (swap) {
                        float    tmp_a = s_amax[i]; s_amax[i] = s_amax[p]; s_amax[p] = tmp_a;
                        uint16_t tmp_i = s_idx [i]; s_idx [i] = s_idx [p]; s_idx [p] = tmp_i;
                    }
                }
            }
            __syncwarp();
        }
    }
}
