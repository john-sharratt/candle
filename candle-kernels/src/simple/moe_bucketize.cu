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
// The kernel is a SINGLE thread block of BUCKETIZE_THREADS (256) threads, and
// serves up to MAX_EXPERTS (512) experts — MORE experts than it has threads, so
// every per-expert phase is a grid-stride loop rather than one-thread-per-expert.
// That distinction is the whole of what raising the bound cost: at 256 the two
// forms coincide, and at 512 the one-thread-per-expert form would have left the
// upper half of the experts with no offsets, no scatter bases and no tiles —
// silently, because their assignments would simply land at stale positions.
//
// Every output is bit-deterministic:
//   * phase 1 — a grid-stride per-expert histogram: each assignment is read
//     ONCE and bumped into its expert's shared bin with an atomicAdd (an id
//     ≥ n_experts is the router's "no expert" sentinel and is skipped). The
//     sums are order-independent, so the counts — and every table derived from
//     them — are identical to a serial scan, at O(a_ub) instead of
//     O(n_experts × a_ub) work;
//   * phase 2 — thread 0 prefix-scans the counts into bucket offsets,
//     accumulates the per-expert tile counts, and writes the device header;
//   * phase 3 — a chunked STABLE counting-sort scatter: the list is split into
//     NCHUNK contiguous chunks, each chunk counts its assignments per expert, a
//     per-expert exclusive prefix across chunks gives each chunk its write base,
//     and each chunk scatters in ascending i. Chunks ordered + within-chunk
//     ascending ⇒ each bucket is STABLE (ascending i), matching the CPU sort
//     exactly, in O(a_ub) work (no per-expert full rescan);
//   * phase 4 — each thread emits its expert's GEMM tiles (≤ tile_w tokens
//     per tile); the tail up to the launch bound is padded with `b_cnt = 0`
//     tiles the grouped kernel skips, so the HOST needs no data-dependent
//     value for the GEMM grid — it launches at the `n_tokens × k` bound;
//   * phase 5 — a chunked block scan over the valid flags builds the
//     token-major compaction: `perm` (expert-grouped row of each valid
//     assignment), `reordered_weight_ids`, and `token_starts`, the exact
//     inputs of `deterministic_scatter_*`.
//
// `tile_expert` carries RAW expert ids into the layer's row of the LIVE expert
// table (`expert_lre::live_table`), whose entry for an expert is its address in
// VRAM, its address in pinned host memory (inside one of the two `pinned`
// ranges: the warm tier and the pad), or 0 — cold, still on the NVMe pack.
// Pinned and cold experts are REMOTE: the grouped GEMM's worker blocks copy them
// into VRAM and compute them. The GEMMs read the SNAPSHOT this kernel takes of
// the routed experts' entries (phase 1b), not the live row.
//
// Two orders, deliberately different:
//   * the ROW layout (tok_ids, weight_ids, perm, rw_ids, token_starts) is
//     ascending expert id, always — it decides the scatter's summation order,
//     so it must be a function of the routing alone;
//   * the TILE order puts remote experts' tiles first — pinned, then cold —
//     then the VRAM experts', each class in ascending id. The grouped GEMM's
//     blocks are independent and every output row has one writer, so tile
//     order changes when a row is computed, never its bits.
// The REMOTE LIST (`remote`, `[n_experts][4]`) names each routed remote expert
// in that order as `{expert, first_tile, n_tiles, cold}`; `header[3]` counts
// them and `header[4]` counts their tiles. `counters[0..3]` — the work counters
// of the layer's gate, up and down launches — are zeroed here.
//
// PROMOTION: each remote expert, in list order, takes the next free VRAM slot
// image from the promotion ring while it has one (`remote_dst[i]`, else 0),
// and the ring's log records `summary_seq << 32 | row << 16 | expert` for it.
// The expert's mark (`promo_marks[row][expert]`) is set with it and cleared by
// the host when the promotion lands; a marked expert is not given a second
// slot by a later invocation that finds it still remote. The grouped GEMMs'
// workers write every slice they copy into that slot too, so once the layer is
// done the expert is whole in VRAM and the host points its entry there.
//
// The ROUTING SUMMARY (`summary`, optional) is what the host learns about the
// layer: `count | pinned << 29 | cold << 30 | decode << 31` per expert, where
// `decode` marks an expert some assignment of a token `< decode_tokens` routed
// to, then `summary_seq` in the word after them. The buffer is mapped host
// memory: the host polls the sequence word and reads the counts in place, with
// no copy and no event.
//
// Padding conventions consumed downstream:
//   tok_ids / weight_ids  : 0xFFFFFFFF  (gather skips the row)
//   tile_b_cnt            : 0           (grouped GEMM early-outs the block)
//   perm / rw_ids         : 0           (never referenced — token_starts
//                                        segments only cover valid rows)
// =============================================================================

#include <stdint.h>

// 256 threads; every per-expert phase strides over `n_experts`, so the expert
// count is independent of the block width. Phase 3 is one-thread-per-chunk.
//
// MAX_EXPERTS 512 is Qwen3.8-Flash-Next's width (Qwen3-MoE has 128, Qwen3.5 has
// 256). The static shared-memory cost is `sh_cc` (32 KB, fixed) plus three
// int32 arrays over the expert axis — 512·4·3 ≈ 6 KB — plus ~1 KB of scan and
// header, ≈ 39 KB against the 48 KiB static cap. Raising the bound again means
// dynamic shared memory, not a constant change.
#define BUCKETIZE_THREADS 256
#define MAX_EXPERTS 512
#define MAX_TOPK 32
#define INVALID_ROW 0xFFFFFFFFu
// Phase-3 chunk-table budget (ints). NCHUNK = SH_CC_INTS / n_experts chunks.
#define SH_CC_INTS 8192
#define SH_CC_MAX_CHUNK 128
// Where an expert's gate entry points.
#define CLS_VRAM 0
#define CLS_PINNED 1
#define CLS_COLD 2

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
    // [5]: n_active, total_valid, num_tiles, remote experts, remote tiles
    int32_t* __restrict__ header,
    uint32_t* __restrict__ inv,          // [a_ub] scratch: i → grouped row
    int32_t* __restrict__ scan,          // [a_ub] scratch: exclusive valid scan
    // This layer's row of the live gate table; null = every expert in VRAM
    // (no remote experts — plain ascending order). The up and down rows are
    // `table_plane` and `2 · table_plane` entries after it.
    const uint64_t* gate_row,
    const long long table_plane,
    // [3][n_experts] snapshot of the routed experts' entries (gate, up, down),
    // the layer's GEMM weight tables; required when `gate_row` is set.
    uint64_t* __restrict__ snap,
    // The two pinned host ranges `[lo, hi)` a remote entry lies in.
    const uint64_t pinned0_lo, const uint64_t pinned0_hi,
    const uint64_t pinned1_lo, const uint64_t pinned1_hi,
    const int decode_tokens,             // tokens [0, decode_tokens) are decode rows
    uint32_t* __restrict__ summary,      // [n_experts + 1] routing summary, or null
    const uint32_t summary_seq,          // stored at summary[n_experts] last
    int32_t* __restrict__ remote,        // [n_experts][4] remote list, or null
    int32_t* __restrict__ counters,      // [3] launch work counters, or null
    // The promotion ring (mapped host memory), or null: `promo_slots[cap]`
    // VRAM slot images the host has freed, `promo_log[cap]` what each was
    // given to (`summary_seq << 32 | row << 16 | expert` — the sequence word
    // says which invocation must finish before the slot is whole), and the two
    // counters — `tail` the host's, `head` this kernel's.
    const uint64_t* promo_slots,
    uint64_t* promo_log,
    uint32_t* promo_head,
    const uint32_t* promo_tail,
    const uint32_t promo_cap,
    // `[rows][n_experts]` marks (mapped): non-zero while an expert's promotion
    // is in flight — set here when a slot is given, cleared by the host when
    // it lands — so a later invocation does not promote it a second time.
    uint32_t* promo_marks,
    const int32_t row,                   // this layer's row, for the log
    uint64_t* __restrict__ remote_dst)   // [n_experts] promotion slot per remote expert, or null
{
    const int tid = (int)threadIdx.x;
    const int a_ub = n_tokens * k;

    __shared__ int32_t sh_counts[MAX_EXPERTS];
    __shared__ int32_t sh_offsets[MAX_EXPERTS + 1];
    __shared__ int32_t sh_tile_pref[MAX_EXPERTS + 1];
    __shared__ int32_t sh_scan[BUCKETIZE_THREADS + 1];
    __shared__ int32_t sh_header[3]; // n_active, total_valid, num_tiles
    // Per-expert: where the gate entry points (CLS_VRAM / CLS_PINNED / CLS_COLD),
    // and whether a decode row routed here.
    __shared__ uint8_t sh_cls[MAX_EXPERTS];
    __shared__ uint8_t sh_dec[MAX_EXPERTS];
    // Per routed remote expert: a promotion of it is already in flight.
    __shared__ uint8_t sh_marked[MAX_EXPERTS];
    // Per-chunk per-expert scratch for the phase-3 stable scatter, flat
    // `[NCHUNK][n_experts]` with a runtime `n_experts` stride. 8192 ints (32 KB)
    // gives NCHUNK = 8192/n_experts chunks — 64 at 128 experts, 32 at 256.
    __shared__ int32_t sh_cc[SH_CC_INTS];

    // ── Phase 1: per-expert histogram (grid-stride, shared-memory atomics) ──
    // Each assignment is read ONCE and atomically bumped into its expert's
    // shared bin — O(a_ub) work instead of the O(n_experts × a_ub) scan where
    // every thread re-read the whole list. Summation is order-independent, so
    // the counts (and therefore every downstream table) are identical to the
    // serial scan: the change is bit-exact, just far faster for large prefill
    // `a_ub` (the single-block kernel's dominant cost).
    //
    for (int e = tid; e < n_experts; e += BUCKETIZE_THREADS) {
        sh_counts[e] = 0;
        sh_dec[e] = 0;
        sh_cls[e] = CLS_VRAM;
        sh_marked[e] = 0;
    }
    __syncthreads();
    for (int i = tid; i < a_ub; i += BUCKETIZE_THREADS) {
        const uint32_t e = topk_ids[i];
        if (e < (uint32_t)n_experts) {
            atomicAdd(&sh_counts[e], 1);
            // Every writer stores the same 1, so the race is benign.
            if (i / k < decode_tokens) {
                sh_dec[e] = 1;
            }
        }
    }
    __syncthreads();
    // ── Phase 1b: classify every routed expert, and snapshot its entries ──
    // The live entries are read once per routed expert, volatile: the host
    // rewrites them while kernels run. What was read is copied into `snap`
    // (`[3][n_experts]`: gate, up, down), and the layer's GEMMs read their
    // weights from the snapshot, never from the live table — so an entry the
    // host retargets after this point cannot reach a block that was ordered by
    // the old value. (Every address read stays valid until the host has seen a
    // later layer's summary word: `expert_lre`'s reclaim rule.) A cold expert —
    // any of its three entries 0 — is the one exception: its workers wait on
    // the live gate entry and read the live up / down entries, which the host
    // publishes before it and never clears while this layer can read them.
    if (gate_row != nullptr) {
        for (int e = tid; e < n_experts; e += BUCKETIZE_THREADS) {
            if (sh_counts[e] == 0) {
                continue;
            }
            const volatile uint64_t* g = (const volatile uint64_t*)gate_row;
            const uint64_t pg = g[e];
            const uint64_t pu = g[table_plane + e];
            const uint64_t pd = g[2 * table_plane + e];
            uint8_t cls = CLS_VRAM;
            if (pg == 0ull || pu == 0ull || pd == 0ull) {
                cls = CLS_COLD;
            } else if ((pg >= pinned0_lo && pg < pinned0_hi) || (pg >= pinned1_lo && pg < pinned1_hi)) {
                cls = CLS_PINNED;
            }
            sh_cls[e] = cls;
            snap[e] = pg;
            snap[n_experts + e] = pu;
            snap[2 * n_experts + e] = pd;
            if (promo_slots != nullptr && cls != CLS_VRAM) {
                sh_marked[e] =
                    ((const volatile uint32_t*)promo_marks)[(size_t)row * n_experts + e] != 0u;
            }
        }
        __syncthreads();
    }
    if (summary != nullptr) {
        for (int e = tid; e < n_experts; e += BUCKETIZE_THREADS) {
            summary[e] = (uint32_t)sh_counts[e]
                       | ((uint32_t)(sh_cls[e] == CLS_PINNED) << 29)
                       | ((uint32_t)(sh_cls[e] == CLS_COLD) << 30)
                       | ((uint32_t)sh_dec[e] << 31);
        }
    }

    // ── Phase 2: offsets + tile prefix + remote list + header (thread 0) ──
    // Offsets ascend by expert id (the row layout); the tile prefix runs over
    // the pinned experts, then the cold, then the VRAM ones (the tile order),
    // and every routed remote expert is listed as it is placed — and given the
    // next promotion slot while the ring has one.
    if (tid == 0) {
        uint32_t head = 0;
        uint32_t tail = 0;
        if (promo_slots != nullptr) {
            head = *(volatile const uint32_t*)promo_head;
            tail = *(volatile const uint32_t*)promo_tail;
            // The slots the host published before its tail store.
            __threadfence_system();
        }
        int32_t off = 0;
        int32_t active = 0;
        for (int e = 0; e < n_experts; e++) {
            sh_offsets[e] = off;
            const int32_t c = sh_counts[e];
            off += c;
            if (c > 0) {
                active++;
            }
        }
        int32_t tiles = 0;
        int32_t n_remote = 0;
        const uint8_t order[3] = {CLS_PINNED, CLS_COLD, CLS_VRAM};
        int32_t remote_tiles = 0;
        for (int pass = 0; pass < 3; pass++) {
            for (int e = 0; e < n_experts; e++) {
                if (sh_cls[e] != order[pass]) {
                    continue;
                }
                const int32_t n = (sh_counts[e] + tile_w - 1) / tile_w;
                sh_tile_pref[e] = tiles;
                if (order[pass] != CLS_VRAM && n > 0 && remote != nullptr) {
                    remote[4 * n_remote + 0] = e;
                    remote[4 * n_remote + 1] = tiles;
                    remote[4 * n_remote + 2] = n;
                    remote[4 * n_remote + 3] = order[pass] == CLS_COLD;
                    if (remote_dst != nullptr) {
                        uint64_t dst = 0ull;
                        if (promo_slots != nullptr && head != tail && !sh_marked[e]) {
                            const uint32_t i = head % promo_cap;
                            dst = ((const volatile uint64_t*)promo_slots)[i];
                            ((volatile uint64_t*)promo_log)[i] =
                                ((uint64_t)summary_seq << 32) | ((uint64_t)row << 16) |
                                (uint64_t)e;
                            ((volatile uint32_t*)promo_marks)[(size_t)row * n_experts + e] =
                                summary_seq;
                            head++;
                        }
                        remote_dst[n_remote] = dst;
                    }
                    n_remote++;
                }
                tiles += n;
            }
            if (order[pass] == CLS_COLD) {
                remote_tiles = tiles;
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
        header[3] = n_remote;
        header[4] = remote_tiles;
        if (counters != nullptr) {
            counters[0] = 0;
            counters[1] = 0;
            counters[2] = 0;
        }
        if (promo_slots != nullptr) {
            // The log entries before the counter that covers them.
            __threadfence_system();
            *(volatile uint32_t*)promo_head = head;
        }
    }
    __syncthreads();
    // The sequence word is the host's signal: it polls this word, so it lands
    // only after every summary word, the promotion log and its head are
    // visible system-wide.
    if (summary != nullptr && tid == 0) {
        __threadfence_system();
        *(volatile uint32_t*)&summary[n_experts] = summary_seq;
    }

    const int32_t total_valid = sh_header[1];
    const int32_t num_tiles = sh_header[2];

    // ── Phase 3: chunked STABLE counting-sort scatter (O(a_ub) work) ──
    // Split the list into NCHUNK contiguous chunks. Pass 1: each chunk-thread
    // counts its chunk's assignments per expert into `sh_cc[chunk][e]`. A
    // per-expert exclusive prefix across chunks (seeded at `sh_offsets[e]`) turns
    // those counts into each chunk's write base for each expert. Pass 2: each
    // chunk re-scans in ASCENDING i and writes every assignment at `base[e]++`.
    // Chunks are ordered and within a chunk i is ascending, so each expert's
    // bucket comes out STABLE (ascending i) — bit-identical to the CPU sort —
    // and no expert ever rescans the whole list.
    const int NCHUNK =
        (SH_CC_INTS / n_experts < SH_CC_MAX_CHUNK) ? (SH_CC_INTS / n_experts) : SH_CC_MAX_CHUNK;
    const int chunk_len = (a_ub + NCHUNK - 1) / NCHUNK;
    for (int idx = tid; idx < NCHUNK * n_experts; idx += BUCKETIZE_THREADS) {
        sh_cc[idx] = 0;
    }
    __syncthreads();
    // Pass 1: per-chunk per-expert counts.
    if (tid < NCHUNK) {
        const int lo = tid * chunk_len;
        const int hi = (lo + chunk_len < a_ub) ? (lo + chunk_len) : a_ub;
        int32_t* cc = &sh_cc[tid * n_experts];
        for (int i = lo; i < hi; i++) {
            const uint32_t e = topk_ids[i];
            if (e < (uint32_t)n_experts) {
                cc[e]++;
            }
        }
    }
    __syncthreads();
    // Per-expert exclusive prefix across chunks: sh_cc[c][e] becomes chunk c's
    // write base for expert e. A thread owns an expert and sweeps the NCHUNK
    // counts; with more experts than threads it takes several, striding.
    for (int e = tid; e < n_experts; e += BUCKETIZE_THREADS) {
        int32_t run = sh_offsets[e];
        for (int c = 0; c < NCHUNK; c++) {
            const int idx = c * n_experts + e;
            const int32_t v = sh_cc[idx];
            sh_cc[idx] = run;
            run += v;
        }
    }
    __syncthreads();
    // Pass 2: stable scatter using each chunk's per-expert running base.
    if (tid < NCHUNK) {
        const int lo = tid * chunk_len;
        const int hi = (lo + chunk_len < a_ub) ? (lo + chunk_len) : a_ub;
        int32_t* base = &sh_cc[tid * n_experts];
        for (int i = lo; i < hi; i++) {
            const uint32_t e = topk_ids[i];
            if (e < (uint32_t)n_experts) {
                const int32_t pos = base[e]++;
                tok_ids[pos] = (uint32_t)(i / k);
                weight_ids[pos] = (uint32_t)i;
                inv[i] = (uint32_t)pos;
            }
        }
    }
    __syncthreads();

    // ── Phase 4: tile tables + padding ──
    for (int e = tid; e < n_experts; e += BUCKETIZE_THREADS) {
        const int32_t count = sh_counts[e];
        if (count <= 0) {
            continue;
        }
        const int32_t base = sh_tile_pref[e];
        const int32_t start = sh_offsets[e];
        const int32_t n_my_tiles = (count + tile_w - 1) / tile_w;
        for (int t = 0; t < n_my_tiles; t++) {
            tile_expert[base + t] = e;
            tile_b_start[base + t] = start + t * tile_w;
            const int32_t rem = count - t * tile_w;
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

// Single-block launch: one BUCKETIZE_THREADS-wide block over ≤ MAX_EXPERTS
// experts. `stream` is the caller's compute stream, so the outputs are ordered
// after the router's writes with no host synchronisation.
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
    const void* gate_row,
    int64_t table_plane,
    void* snap,
    uint64_t pinned0_lo,
    uint64_t pinned0_hi,
    uint64_t pinned1_lo,
    uint64_t pinned1_hi,
    int32_t decode_tokens,
    void* summary,
    uint32_t summary_seq,
    void* remote,
    void* counters,
    const void* promo_slots,
    void* promo_log,
    void* promo_head,
    const void* promo_tail,
    uint32_t promo_cap,
    void* promo_marks,
    int32_t row,
    void* remote_dst,
    void* stream)
{
    if (n_tokens <= 0 || k <= 0 || k > MAX_TOPK || n_experts <= 0 ||
        n_experts > MAX_EXPERTS || tile_w <= 0 || (gate_row != nullptr && snap == nullptr) ||
        (promo_slots != nullptr && (promo_cap == 0 || promo_log == nullptr ||
                                    promo_head == nullptr || promo_tail == nullptr ||
                                    promo_marks == nullptr || remote_dst == nullptr))) {
        return;
    }
    moe_bucketize_kernel<<<1, BUCKETIZE_THREADS, 0, (cudaStream_t)stream>>>(
        (const uint32_t*)topk_ids, n_tokens, k, n_experts, tile_w,
        (uint32_t*)tok_ids, (uint32_t*)weight_ids, (int32_t*)tile_expert,
        (int32_t*)tile_b_start, (int32_t*)tile_b_cnt, (uint32_t*)perm,
        (uint32_t*)rw_ids, (int32_t*)token_starts, (int32_t*)header,
        (uint32_t*)inv, (int32_t*)scan, (const uint64_t*)gate_row,
        (long long)table_plane, (uint64_t*)snap, pinned0_lo, pinned0_hi, pinned1_lo, pinned1_hi,
        decode_tokens, (uint32_t*)summary, summary_seq,
        (int32_t*)remote, (int32_t*)counters,
        (const uint64_t*)promo_slots, (uint64_t*)promo_log, (uint32_t*)promo_head,
        (const uint32_t*)promo_tail, promo_cap, (uint32_t*)promo_marks, row,
        (uint64_t*)remote_dst);
}
