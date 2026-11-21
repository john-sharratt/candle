// The threshold-sweep winner kernels and the legacy per-chunk and per-head
// format reductions.
//
// The winner selection and summarisation kernels used by the threshold sweep,
// and the per-chunk, per-head and per-head-stats format reductions that the
// older path used. Each has a host launcher beside it.

// =============================================================================
// WINNER SELECTION KERNEL  (threshold-sweep utility)
// =============================================================================
//
// Consumes pre-computed K and V error surfaces (output of
// `sample_quant_errors_kv_paged`) and selects, for every (chunk, head,
// dim) cell and every threshold in `k_thresholds` / `v_thresholds`, the
// most aggressive candidate whose error is ≤ the threshold.
//
// Doing selection on the GPU instead of after a download replaces the
// large float error array with a small u8 winner array — often the
// difference between a multi-MB transfer and a few hundred KB:
//
//   error array  : [n_chunks × head_dim × n_quant × n_kv_head × 2 × 4 B]
//   winner array : [n_thresholds × n_cells × 2 × 1 B]  (n_quant × 4 smaller)
//
// Grid: (ceil(n_cells / SELECT_WINNER_THREADS), 1, 1)
//   n_cells = n_chunks × n_kv_head × head_dim
//   Each thread handles one (chunk, head, dim) cell.
//
// Layout — must match `batch_select_and_summarize` on the host:
//
//   k_winners[t * n_cells + cell_id]
//   cell_id = (chunk * n_kv_head + head) * head_dim + dim  =  bh * n_dim + d
//   error[((chunk * head_dim + dim) * n_quant + q) * n_kv_head + head]

#define SELECT_WINNER_THREADS 256

extern "C" __global__ void select_winners_kv_paged(
    const float* __restrict__ k_errors,       // [n_chunks × head_dim × n_quant × n_kv_head]
    const float* __restrict__ v_errors,       // same layout
    const float* __restrict__ k_thresholds,   // [n_k_thresholds]
    const float* __restrict__ v_thresholds,   // [n_v_thresholds]
    uint8_t* __restrict__ k_winners,          // [n_k_thresholds × n_cells]
    uint8_t* __restrict__ v_winners,          // [n_v_thresholds × n_cells]
    int n_k_thresholds,
    int n_v_thresholds,
    int n_cells,       // = n_chunks × n_kv_head × head_dim
    int n_quant,
    int n_kv_head,
    int head_dim
) {
    const int cell_id = blockIdx.x * SELECT_WINNER_THREADS + threadIdx.x;
    if (cell_id >= n_cells) return;

    // Decompose cell_id using winner layout:
    //   cell_id = (chunk * n_kv_head + head) * head_dim + dim
    const int dim   = cell_id % head_dim;
    const int bh    = cell_id / head_dim;   // = chunk * n_kv_head + head
    const int head  = bh % n_kv_head;
    const int chunk = bh / n_kv_head;

    // Base index into the error surface for this (chunk, dim, head), varying over q:
    //   error_base + q * error_stride  =  ((chunk * head_dim + dim) * n_quant + q) * n_kv_head + head
    const int error_base   = ((chunk * head_dim + dim) * n_quant) * n_kv_head + head;
    const int error_stride = n_kv_head;

    // Cache all candidate errors in registers (n_quant ≤ 32).
    float k_e[32];
    float v_e[32];
    #pragma unroll 1
    for (int q = 0; q < n_quant; q++) {
        k_e[q] = __ldg(&k_errors[error_base + q * error_stride]);
        v_e[q] = __ldg(&v_errors[error_base + q * error_stride]);
    }

    // K: find first candidate whose error ≤ threshold for each threshold level.
    for (int t = 0; t < n_k_thresholds; t++) {
        const float thr = __ldg(&k_thresholds[t]);
        int winner = n_quant - 1;
        #pragma unroll 1
        for (int q = 0; q < n_quant; q++) {
            if (k_e[q] <= thr) { winner = q; break; }
        }
        k_winners[t * n_cells + cell_id] = (uint8_t)winner;
    }

    // V: same.
    for (int t = 0; t < n_v_thresholds; t++) {
        const float thr = __ldg(&v_thresholds[t]);
        int winner = n_quant - 1;
        #pragma unroll 1
        for (int q = 0; q < n_quant; q++) {
            if (v_e[q] <= thr) { winner = q; break; }
        }
        v_winners[t * n_cells + cell_id] = (uint8_t)winner;
    }
}

extern "C" void run_select_winners_kv_paged(
    const float* k_errors,
    const float* v_errors,
    const float* k_thresholds,
    const float* v_thresholds,
    uint8_t* k_winners,
    uint8_t* v_winners,
    int n_k_thresholds,
    int n_v_thresholds,
    int n_cells,
    int n_quant,
    int n_kv_head,
    int head_dim,
    void* stream
) {
    if (n_cells == 0) return;
    const int grid = (n_cells + SELECT_WINNER_THREADS - 1) / SELECT_WINNER_THREADS;
    select_winners_kv_paged<<<grid, SELECT_WINNER_THREADS, 0, (cudaStream_t)stream>>>(
        k_errors, v_errors,
        k_thresholds, v_thresholds,
        k_winners, v_winners,
        n_k_thresholds, n_v_thresholds,
        n_cells, n_quant, n_kv_head, head_dim
    );
}

// =============================================================================
// WINNER SUMMARIZATION KERNEL  (threshold-sweep utility)
// =============================================================================
//
// Reduces the full [n_thresholds × n_cells] u8 winner array (output of
// `select_winners_kv_paged`) into [n_thresholds × 3] f32 bit-count
// accumulators, without a round-trip to host. Used by the offline
// threshold sweep to compare three compression strategies head-to-head:
//
//   ideal_bits = Σ_{bh,d} cand_bpe[winners[t,bh,d]] × chunk_size
//                — per-block selection (best possible without grouping)
//
//   head_bits  = Σ_{bh}   cand_bpe[max_d winners[t,bh,d]] × n_dim × chunk_size
//                — per-head selection (one format per head, matches the
//                  worst-case block — no palette grouping)
//
//   pal4_bits  = Σ_{bh}   ( palette4_pass(bh) + palette overhead )
//                — palette4 grouping (the production strategy)
//
// Per-cell winners are histogrammed (32-element register array, since
// n_quant ≤ 32) and the histogram filled into 4 slots from low to
// high to estimate per-block format assignments under palette4.
//
// Grid:  (ceil(n_bh / SUMMARIZE_WINNER_THREADS), n_thresholds)
// Block: (SUMMARIZE_WINNER_THREADS, 1)
//
//   out[t * 3 + 0] = ideal_bits[t]
//   out[t * 3 + 1] = head_bits[t]
//   out[t * 3 + 2] = pal4_bits[t]
//
// The three output accumulators must be zero-initialised by the caller
// before launch (atomicAdd targets).

#define SUMMARIZE_WINNER_THREADS 128

extern "C" __global__ void summarize_winners_side_paged(
    const uint8_t* __restrict__ winners,    // [n_thresholds × n_cells], cell=bh*n_dim+d
    const float*   __restrict__ cand_bpe,   // [n_quant]  bits-per-element per candidate
    float*                      out,        // [n_thresholds × 3]  (atomicAdd target)
    int n_thresholds,
    int n_cells,                            // = n_bh × n_dim
    int n_bh,
    int n_dim,
    int n_quant,
    int chunk_size,
    float pal_overhead                      // (n_dim * 2 + 4 * 8) bits
) {
    const int bh = blockIdx.x * SUMMARIZE_WINNER_THREADS + threadIdx.x;
    const int t  = blockIdx.y;
    if (bh >= n_bh || t >= n_thresholds) return;

    const uint8_t* slice = winners + (size_t)t * n_cells + bh * n_dim;

    // ── Phase 1: freq + head_worst + ideal_bits ───────────────────────────
    uint8_t freq[32];
    #pragma unroll 1
    for (int q = 0; q < n_quant; q++) freq[q] = 0;
    int head_worst = 0;
    float ideal_bits = 0.0f;

    #pragma unroll 1
    for (int d = 0; d < n_dim; d++) {
        int w = (int)slice[d];
        if (w >= n_quant) w = n_quant - 1;
        freq[w]++;
        if (w > head_worst) head_worst = w;
        ideal_bits += cand_bpe[w] * (float)chunk_size;
    }

    float head_bits = cand_bpe[head_worst] * (float)(n_dim * chunk_size);

    // ── Phase 2: fixed 4-slot palette fill from lowest required formats ──
    const int base_slot = n_dim / 4;
    const int extra_slot = n_dim % 4;
    float pal4_bits = pal_overhead;
    int seen = 0;
    int q = 0;
    int filled = 0;

    #pragma unroll 1
    for (int p = 0; p < 4; p++) {
        const int slot_size = base_slot + (p < extra_slot ? 1 : 0);
        if (slot_size <= 0) continue;
        filled += slot_size;
        while (q + 1 < n_quant && seen + (int)freq[q] < filled) {
            seen += (int)freq[q];
            q += 1;
        }
        pal4_bits += cand_bpe[q] * (float)(slot_size * chunk_size);
    }

    // ── Accumulate into per-threshold output ────────────────────────────
    float* t_out = out + t * 3;
    atomicAdd(&t_out[0], ideal_bits);
    atomicAdd(&t_out[1], head_bits);
    atomicAdd(&t_out[2], pal4_bits);
}

extern "C" void run_summarize_winners_side_paged(
    const uint8_t* winners,     // [n_thresholds × n_cells]
    const float*   cand_bpe,    // [n_quant]
    float*         out,         // [n_thresholds × 3] — must be zero-initialised by caller
    int n_thresholds,
    int n_cells,
    int n_bh,
    int n_dim,
    int n_quant,
    int chunk_size,
    float pal_overhead,
    void* stream
) {
    if (n_bh == 0 || n_thresholds == 0) return;
    dim3 grid((n_bh + SUMMARIZE_WINNER_THREADS - 1) / SUMMARIZE_WINNER_THREADS,
              n_thresholds);
    summarize_winners_side_paged<<<grid, SUMMARIZE_WINNER_THREADS, 0, (cudaStream_t)stream>>>(
        winners, cand_bpe, out,
        n_thresholds, n_cells, n_bh, n_dim, n_quant, chunk_size, pal_overhead
    );
}

// =============================================================================
// PER-CHUNK FORMAT REDUCTION  (legacy path)
// =============================================================================
//
// Reduces a per-block format-tag array into one tag per chunk. Used by
// the older non-palette path that quantizes whole chunks at a single
// format rather than grouping into 4 palette slots. The fused selection
// kernel above does its own (different) reduction into `head_tag`, so
// this kernel is not on the production path — kept for the legacy code
// that still emits per-block tags.
//
// Reduction policy: most conservative wins. The chunk's format is the
// highest-fidelity (largest candidate index) tag found among any of
// its blocks — downgrading any block below its required precision
// would violate the threshold for that block. F16 is treated as a
// sentinel "no quantization": if any block is F16, the whole chunk is.
//
// Format ordering is by candidate index: candidates[0] is most
// aggressive (lowest BPE), candidates[N−1] is highest fidelity
// (highest BPE).

__global__ void reduce_chunk_format(
    const int* __restrict__ block_tags,   // [num_chunks * blocks_per_chunk]
    int* __restrict__ chunk_tags,         // [num_chunks] output
    const int* __restrict__ candidates,   // [num_candidates] ordered low→high fidelity (ascending BPE)
    int num_candidates,
    int blocks_per_chunk,
    int num_chunks
) {
    const int chunk_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (chunk_idx >= num_chunks) return;

    const int base = chunk_idx * blocks_per_chunk;

    // Find the most conservative (highest candidate index) block in this chunk.
    // Start at -1 (no candidate seen yet).
    int max_cidx = -1;
    bool any_f16 = false;

    for (int i = 0; i < blocks_per_chunk; i++) {
        int t = block_tags[base + i];

        if (t == SELECT_FMT_F16) {
            // F16 sentinel — most conservative, entire chunk stays float
            any_f16 = true;
            break;
        }

        // Map tag to candidate index
        for (int c = 0; c < num_candidates; c++) {
            if (candidates[c] == t) {
                if (c > max_cidx) max_cidx = c;
                break;
            }
        }
    }

    if (any_f16) {
        chunk_tags[chunk_idx] = SELECT_FMT_F16;
    } else if (max_cidx >= 0) {
        chunk_tags[chunk_idx] = candidates[max_cidx];
    } else {
        // No blocks matched any candidate — fall back to F16
        chunk_tags[chunk_idx] = SELECT_FMT_F16;
    }
}

extern "C" void run_reduce_chunk_format(
    const int* k_block_tags,
    const int* v_block_tags,
    int* k_chunk_tags,
    int* v_chunk_tags,
    const int* k_candidates,
    const int* v_candidates,
    int num_k_candidates,
    int num_v_candidates,
    int blocks_per_chunk,
    int num_chunks,
    void* stream
) {
    if (num_chunks == 0) return;
    const int threads = 256;
    const int grid = (num_chunks + threads - 1) / threads;
    reduce_chunk_format<<<grid, threads, 0, (cudaStream_t)stream>>>(
        k_block_tags, k_chunk_tags, k_candidates, num_k_candidates,
        blocks_per_chunk, num_chunks);
    reduce_chunk_format<<<grid, threads, 0, (cudaStream_t)stream>>>(
        v_block_tags, v_chunk_tags, v_candidates, num_v_candidates,
        blocks_per_chunk, num_chunks);
}

// =============================================================================
// PER-HEAD FORMAT REDUCTION  (legacy path)
// =============================================================================
//
// Same worst-case-wins policy as `reduce_chunk_format`, but at the head
// level. Output is one tag per (chunk, head) — one entry per CUDA block
// of the per-block selection kernel.
//
// blocks_per_head = head_dim · chunk_size / 32  (e.g. 128·32/32 = 128).
// Block tags are laid out as `[chunk][head][blocks_per_head]` in the
// input. Each thread handles one (chunk, head) pair and scans the
// `blocks_per_head`-sized stripe at that offset.

__global__ void reduce_head_format(
    const int* __restrict__ block_tags,   // [num_chunks * n_kv_head * blocks_per_head]
    int* __restrict__ head_tags,          // [num_chunks * n_kv_head] output
    const int* __restrict__ candidates,   // [num_candidates] ordered low→high fidelity (ascending BPE)
    int num_candidates,
    int blocks_per_head,
    int n_kv_head,
    int num_chunks
) {
    // One thread per (chunk, head) pair
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_chunks * n_kv_head;
    if (idx >= total) return;

    const int chunk_idx = idx / n_kv_head;
    const int head_idx = idx % n_kv_head;

    // Block tags layout: [chunk_idx][head_idx][blocks_per_head]
    const int blocks_per_chunk = n_kv_head * blocks_per_head;
    const int base = chunk_idx * blocks_per_chunk + head_idx * blocks_per_head;

    // Find the most conservative (highest candidate index) block for this head.
    int max_cidx = -1;
    bool any_f16 = false;

    for (int i = 0; i < blocks_per_head; i++) {
        int t = block_tags[base + i];

        if (t == SELECT_FMT_F16) {
            any_f16 = true;
            break;
        }

        for (int c = 0; c < num_candidates; c++) {
            if (candidates[c] == t) {
                if (c > max_cidx) max_cidx = c;
                break;
            }
        }
    }

    if (any_f16) {
        head_tags[idx] = SELECT_FMT_F16;
    } else if (max_cidx >= 0) {
        head_tags[idx] = candidates[max_cidx];
    } else {
        head_tags[idx] = SELECT_FMT_F16;
    }
}

extern "C" void run_reduce_head_format(
    const int* k_block_tags,
    const int* v_block_tags,
    int* k_head_tags,
    int* v_head_tags,
    const int* k_candidates,
    const int* v_candidates,
    int num_k_candidates,
    int num_v_candidates,
    int blocks_per_head,
    int n_kv_head,
    int num_chunks,
    void* stream
) {
    if (num_chunks == 0) return;
    const int total = num_chunks * n_kv_head;
    const int threads = 256;
    const int grid = (total + threads - 1) / threads;
    reduce_head_format<<<grid, threads, 0, (cudaStream_t)stream>>>(
        k_block_tags, k_head_tags, k_candidates, num_k_candidates,
        blocks_per_head, n_kv_head, num_chunks);
    reduce_head_format<<<grid, threads, 0, (cudaStream_t)stream>>>(
        v_block_tags, v_head_tags, v_candidates, num_v_candidates,
        blocks_per_head, n_kv_head, num_chunks);
}

// =============================================================================
// PER-HEAD STATS-FORMAT REDUCTION
// =============================================================================
//
// Variant of `reduce_head_format` that uses `format_table_index_cuda`
// (rather than the candidate-index lookup) to rank tags. The table
// index orders all formats globally by BPE — including F16/BF16 — so
// this reduction works without needing a candidate set. Used by stats
// and observability code that needs to summarise heads regardless of
// which candidate set the selection ran with.
//
// Also writes `effective_block_tags`: every block in the head gets the
// head's worst-case tag, so downstream stats code sees the effective
// per-block format under a head-uniform encoding policy. This is the
// metric the threshold-sweep tooling uses to compare grouping
// strategies.

__global__ void reduce_head_stats_format(
    const int* __restrict__ block_tags,
    int* __restrict__ head_tags,
    int* __restrict__ effective_block_tags,
    int blocks_per_head,
    int n_kv_head,
    int num_chunks
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_chunks * n_kv_head;
    if (idx >= total) return;

    const int chunk_idx = idx / n_kv_head;
    const int head_idx = idx % n_kv_head;
    const int blocks_per_chunk = n_kv_head * blocks_per_head;
    const int base = chunk_idx * blocks_per_chunk + head_idx * blocks_per_head;

    int worst_tag = SELECT_FMT_F16;
    int worst_ti = -1;
    for (int i = 0; i < blocks_per_head; ++i) {
        const int tag = block_tags[base + i];
        const int ti = format_table_index_cuda(tag);
        if (ti > worst_ti) {
            worst_ti = ti;
            worst_tag = tag;
        }
    }
    head_tags[idx] = worst_tag;
    for (int i = 0; i < blocks_per_head; ++i) {
        effective_block_tags[base + i] = worst_tag;
    }
}

extern "C" void run_reduce_head_stats_format(
    const int* k_block_tags,
    const int* v_block_tags,
    int* k_head_tags,
    int* v_head_tags,
    int* k_effective_block_tags,
    int* v_effective_block_tags,
    int blocks_per_head,
    int n_kv_head,
    int num_chunks,
    void* stream
) {
    if (num_chunks == 0) return;
    const int total = num_chunks * n_kv_head;
    const int threads = 256;
    const int grid = (total + threads - 1) / threads;
    reduce_head_stats_format<<<grid, threads, 0, (cudaStream_t)stream>>>(
        k_block_tags, k_head_tags, k_effective_block_tags,
        blocks_per_head, n_kv_head, num_chunks);
    reduce_head_stats_format<<<grid, threads, 0, (cudaStream_t)stream>>>(
        v_block_tags, v_head_tags, v_effective_block_tags,
        blocks_per_head, n_kv_head, num_chunks);
}
