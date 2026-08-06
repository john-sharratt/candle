// =============================================================================
// GPU MoE EXPERT BUCKETIZE
// =============================================================================
// Replaces the CPU counting-sort in the grouped expert compute path: the
// per-layer routing indices no longer round-trip GPU→CPU→GPU. One launch turns
// `moe_route`'s top-k index tensor into every table the downstream GPU pipeline
// consumes — the expert-grouped assignment lists (gather), the tile tables
// (grouped GEMM), and the token-major segment tables (deterministic scatter) —
// entirely on the device.
//
// The kernel is a SINGLE thread block of 128 threads and uses NO atomics, so
// every output is bit-deterministic:
//   * phase 1 — thread `e` owns expert `e` and counts its assignments by
//     scanning the whole top-k list (an id ≥ n_experts is the router's
//     "no expert" sentinel for an empty slot and is skipped);
//   * phase 2 — thread 0 prefix-scans the 128 counts into bucket offsets,
//     accumulates the per-expert tile counts, and writes the device header;
//   * phase 3 — thread `e` re-scans the list in ascending assignment order and
//     writes its own bucket sequentially: the grouping is STABLE in
//     (token, slot) order, matching the CPU sort exactly;
//   * phase 4 — each thread emits its expert's GEMM tiles (≤ tile_w tokens
//     per tile); the tail up to the launch bound is padded with `b_cnt = 0`
//     tiles the grouped kernel skips, so the HOST needs no data-dependent
//     value for the GEMM grid — it launches at the `n_tokens × k` bound;
//   * phase 5 — a chunked block scan over the valid flags builds the
//     token-major compaction: `perm` (expert-grouped row of each valid
//     assignment), `reordered_weight_ids`, and `token_starts`, the exact
//     inputs of `deterministic_scatter_*`.
//
// `tile_expert` carries RAW expert ids: the grouped GEMM's `weight_ptrs` is the
// full per-layer resident pointer table (all experts VRAM-resident), so no
// active-list compaction or pointer gather is needed anywhere.
//
// Padding conventions consumed downstream:
//   tok_ids / weight_ids  : 0xFFFFFFFF  (gather skips the row)
//   tile_b_cnt            : 0           (grouped GEMM early-outs the block)
//   perm / rw_ids         : 0           (never referenced — token_starts
//                                        segments only cover valid rows)
// =============================================================================

#include <stdint.h>

// 256 threads so thread e can own expert e for up to 256 experts (DeepSeek-V4 has 256
// routed experts; Qwen3-MoE has 128). Phase-1/3/4 are one-thread-per-expert; phase-4b/5
// are BUCKETIZE_THREADS grid-stride, so both scale together.
#define BUCKETIZE_THREADS 256
#define MAX_EXPERTS 256
#define MAX_TOPK 32
#define INVALID_ROW 0xFFFFFFFFu

extern "C" __global__ void moe_bucketize_kernel(
    const uint32_t* __restrict__ topk_ids, // [n_tokens * k] row-major
    const int n_tokens,
    const int k,
    const int n_experts, // ≤ MAX_EXPERTS; id ≥ n_experts = sentinel
    const int tile_w,    // grouped-GEMM tile width (tokens per tile)
    uint32_t* __restrict__ tok_ids,      // [a_ub] expert-grouped token ids
    uint32_t* __restrict__ weight_ids,   // [a_ub] expert-grouped widx (= i)
    int32_t* __restrict__ tile_expert,   // [a_ub] RAW expert id per tile
    int32_t* __restrict__ tile_b_start,  // [a_ub]
    int32_t* __restrict__ tile_b_cnt,    // [a_ub]
    uint32_t* __restrict__ perm,         // [a_ub] token-major → grouped row
    uint32_t* __restrict__ rw_ids,       // [a_ub] token-major widx
    int32_t* __restrict__ token_starts,  // [n_tokens + 1]
    int32_t* __restrict__ header,        // [4]: n_active, total_valid, num_tiles
    uint32_t* __restrict__ inv,          // [a_ub] scratch: i → grouped row
    int32_t* __restrict__ scan)          // [a_ub] scratch: exclusive valid scan
{
    const int tid = (int)threadIdx.x;
    const int a_ub = n_tokens * k;

    __shared__ int32_t sh_counts[MAX_EXPERTS];
    __shared__ int32_t sh_offsets[MAX_EXPERTS + 1];
    __shared__ int32_t sh_tile_pref[MAX_EXPERTS + 1];
    __shared__ int32_t sh_scan[BUCKETIZE_THREADS + 1];
    __shared__ int32_t sh_header[3]; // n_active, total_valid, num_tiles

    // ── Phase 1: per-expert counts (thread e owns expert e) ──
    int my_count = 0;
    if (tid < n_experts) {
        for (int i = 0; i < a_ub; i++) {
            if (topk_ids[i] == (uint32_t)tid) {
                my_count++;
            }
        }
    }
    if (tid < MAX_EXPERTS) {
        sh_counts[tid] = my_count;
    }
    __syncthreads();

    // ── Phase 2: offsets + tile prefix + header (thread 0, ≤128 iterations) ──
    if (tid == 0) {
        int32_t off = 0;
        int32_t tiles = 0;
        int32_t active = 0;
        for (int e = 0; e < n_experts; e++) {
            sh_offsets[e] = off;
            sh_tile_pref[e] = tiles;
            const int32_t c = sh_counts[e];
            off += c;
            tiles += (c + tile_w - 1) / tile_w;
            if (c > 0) {
                active++;
            }
        }
        sh_offsets[n_experts] = off;
        sh_tile_pref[n_experts] = tiles;
        sh_header[0] = active;
        sh_header[1] = off;   // total_valid
        sh_header[2] = tiles; // num_tiles
        header[0] = active;
        header[1] = off;
        header[2] = tiles;
        header[3] = 0;
    }
    __syncthreads();

    const int32_t total_valid = sh_header[1];
    const int32_t num_tiles = sh_header[2];

    // ── Phase 3: stable bucket write (thread e re-scans in ascending i) ──
    if (tid < n_experts && my_count > 0) {
        int32_t cursor = sh_offsets[tid];
        for (int i = 0; i < a_ub; i++) {
            if (topk_ids[i] == (uint32_t)tid) {
                tok_ids[cursor] = (uint32_t)(i / k);
                weight_ids[cursor] = (uint32_t)i;
                inv[i] = (uint32_t)cursor;
                cursor++;
            }
        }
    }

    // ── Phase 4: tile tables + padding ──
    if (tid < n_experts && my_count > 0) {
        const int32_t base = sh_tile_pref[tid];
        const int32_t start = sh_offsets[tid];
        const int32_t n_my_tiles = (my_count + tile_w - 1) / tile_w;
        for (int t = 0; t < n_my_tiles; t++) {
            tile_expert[base + t] = tid;
            tile_b_start[base + t] = start + t * tile_w;
            const int32_t rem = my_count - t * tile_w;
            tile_b_cnt[base + t] = rem < tile_w ? rem : tile_w;
        }
    }
    for (int t = num_tiles + tid; t < a_ub; t += BUCKETIZE_THREADS) {
        tile_expert[t] = 0;
        tile_b_start[t] = 0;
        tile_b_cnt[t] = 0;
    }
    for (int i = total_valid + tid; i < a_ub; i += BUCKETIZE_THREADS) {
        tok_ids[i] = INVALID_ROW;
        weight_ids[i] = INVALID_ROW;
    }
    __syncthreads();

    // ── Phase 5: token-major compaction (chunked exclusive scan of valid) ──
    // 5a: per-thread chunk sums.
    const int chunk = (a_ub + BUCKETIZE_THREADS - 1) / BUCKETIZE_THREADS;
    const int c_lo = tid * chunk;
    const int c_hi = c_lo + chunk < a_ub ? c_lo + chunk : a_ub;
    int32_t local = 0;
    for (int i = c_lo; i < c_hi; i++) {
        if (topk_ids[i] < (uint32_t)n_experts) {
            local++;
        }
    }
    sh_scan[tid] = local;
    __syncthreads();
    // 5b: exclusive scan of the 128 chunk sums (thread 0).
    if (tid == 0) {
        int32_t run = 0;
        for (int t = 0; t < BUCKETIZE_THREADS; t++) {
            const int32_t c = sh_scan[t];
            sh_scan[t] = run;
            run += c;
        }
        sh_scan[BUCKETIZE_THREADS] = run;
    }
    __syncthreads();
    // 5c: chunk re-sweep → the full exclusive scan.
    int32_t run = sh_scan[tid];
    for (int i = c_lo; i < c_hi; i++) {
        scan[i] = run;
        if (topk_ids[i] < (uint32_t)n_experts) {
            run++;
        }
    }
    __syncthreads();
    // 5d: per-token compaction + segment boundaries + padding. Within a token
    // the (perm, rw_ids) pairs are ordered by ASCENDING expert-grouped row —
    // the scatter accumulates each token's contributions sequentially in perm
    // order, and this matches the CPU path's `sort_by_key((token_id, row))`
    // exactly, so the float-summation order (and therefore every output bit)
    // is identical to the CPU-built tables. k is small (≤ MAX_TOPK), so each
    // token sorts its pairs with an in-register insertion sort — deterministic,
    // one thread per token.
    for (int t = tid; t < n_tokens; t += BUCKETIZE_THREADS) {
        uint32_t rows[MAX_TOPK];
        uint32_t wids[MAX_TOPK];
        int n_valid = 0;
        for (int s = 0; s < k; s++) {
            const int i = t * k + s;
            if (topk_ids[i] < (uint32_t)n_experts) {
                const uint32_t r = inv[i];
                // Insertion sort by grouped row, ascending.
                int p = n_valid;
                while (p > 0 && rows[p - 1] > r) {
                    rows[p] = rows[p - 1];
                    wids[p] = wids[p - 1];
                    p--;
                }
                rows[p] = r;
                wids[p] = (uint32_t)i;
                n_valid++;
            }
        }
        const int32_t base = scan[t * k];
        for (int s = 0; s < n_valid; s++) {
            perm[base + s] = rows[s];
            rw_ids[base + s] = wids[s];
        }
    }
    for (int t = tid; t <= n_tokens; t += BUCKETIZE_THREADS) {
        token_starts[t] = t < n_tokens ? scan[t * k] : total_valid;
    }
    for (int j = total_valid + tid; j < a_ub; j += BUCKETIZE_THREADS) {
        perm[j] = 0;
        rw_ids[j] = 0;
    }
}

// Single-block launch: the kernel is a fixed 128-thread cooperative sort over
// ≤ 128 experts. `stream` is the caller's compute stream, so the outputs are
// ordered after `moe_route`'s writes with no host synchronisation.
extern "C" void run_moe_bucketize(
    const void* topk_ids,
    int32_t n_tokens,
    int32_t k,
    int32_t n_experts,
    int32_t tile_w,
    void* tok_ids,
    void* weight_ids,
    void* tile_expert,
    void* tile_b_start,
    void* tile_b_cnt,
    void* perm,
    void* rw_ids,
    void* token_starts,
    void* header,
    void* inv,
    void* scan,
    void* stream)
{
    if (n_tokens <= 0 || k <= 0 || k > MAX_TOPK || n_experts <= 0 ||
        n_experts > MAX_EXPERTS || tile_w <= 0) {
        return;
    }
    moe_bucketize_kernel<<<1, BUCKETIZE_THREADS, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)topk_ids, n_tokens, k, n_experts, tile_w,
        (uint32_t*)tok_ids, (uint32_t*)weight_ids, (int32_t*)tile_expert,
        (int32_t*)tile_b_start, (int32_t*)tile_b_cnt, (uint32_t*)perm,
        (uint32_t*)rw_ids, (int32_t*)token_starts, (int32_t*)header,
        (uint32_t*)inv, (int32_t*)scan);
}
