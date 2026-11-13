// =============================================================================
// GAP CLAIM: destinations for the gap's quantize, chosen on the device
// =============================================================================
// `docs/kv_quantize_in_the_gap.md` §3.2–§3.4.
//
// The format selection leaves each (job, head)'s palette tags, outer scales and
// per-channel palette slots in device memory. The host has leased, per size
// class, a stack of free slot addresses. `gap_claim_kernel` turns the one into
// destinations from the other without the host seeing either, in one launch per
// policy group:
//
//   * every band's index in its class — the count of bands of that class before
//     it in table order, bands ordered job, head, palette, K before V,
//     continuing from the count earlier policy groups of the same gap took
//     (`taken`, left advanced past this group). Deterministic: a prefix count,
//     not a race on an atomic cursor.
//   * each job deferred when any of its bands' indices falls past its class's
//     lease (or its code is one the host's table does not cover, which also
//     raises `error_word`);
//   * every head's destination KvHead record — packed palette maps, record
//     format codes, outer scales and slot addresses (the sink for a deferred
//     job) — written into the convert's buffer and into the mapped summary,
//     with each band's lease index and each job's deferred flag.
//
// The prefix count is a single-pass scan with decoupled look-back over the
// grid: a block takes a tile of whole jobs (up to `TILE_HEADS` heads, one
// counting thread each), counts its bands per class, publishes the tile's
// per-class totals, packs the tile's palette maps with all its threads (the bulk
// of the bytes, and independent of the count) while its predecessors resolve,
// then looks back for its exclusive prefix, sixteen predecessors a load per
// class. Sixteen class counters ride two to a 32-bit word through the warp and
// block scans.
//
// `gap_complete_kernel` (one thread), after the convert, writes the completion
// word the host polls.
//
// Tile k is block k. Blocks of a 1-D grid are dispatched in index order, so a
// tile's predecessors are resident (or finished) whenever it looks back — the
// assumption CUB's single-pass scan makes too.
//
// Status (`claim_status_words`, zeroed once when allocated, then used by this
// kernel alone): a per-(tile, class) word, `epoch | flag | count`. The host
// passes a new epoch with every launch on a status buffer, so no launch memsets
// anything: a word an earlier launch left carries an earlier epoch and reads as
// not ready.
//
// A tile's rows — the summary row, whose first `hd/2 + 104` bytes are the
// convert record — are assembled in shared memory and leave in two contiguous
// spans of 8-byte stores, coalesced across the block; the summary is normally
// mapped host memory, where whole-line writes are what the link carries well.
//
// The KvHead layout written here is the one `build_kvhead_bytes_raw`
// (candle-core `quantized/cuda.rs`) writes on the host:
//   [k pal map hd/4][v pal map hd/4][k ptr ×4][v ptr ×4][k fmt ×4][v fmt ×4]
//   [k scale ×4][v scale ×4]   — hd/2 + 104 bytes.
// =============================================================================

#include <cuda_runtime.h>
#include <stdint.h>

namespace kv_gap_claim {

constexpr int BANDS_PER_HEAD = 8;
constexpr int THREADS = 256;
// Heads per tile, one counting thread each; every thread packs maps.
constexpr int TILE_HEADS = 64;
constexpr int COUNT_WARPS = TILE_HEADS / 32;
// At least `SizeClass::COUNT`; the host refuses a ladder longer than this.
constexpr int MAX_CLASS = 16;
// Class counters two to a word.
constexpr int PAIRS = MAX_CLASS / 2;
// Look-back lanes per class: the block covers every class at once.
constexpr int LOOK_LANES = THREADS / MAX_CLASS;
static_assert(LOOK_LANES == 16, "a class's look-back lanes are one half-warp");
// Select codes are below this; the host's code tables are this long.
constexpr int N_CODES = 64;
// A tile holds whole jobs, so a job's heads never straddle two blocks.
constexpr int MAX_KV_HEAD = 64;
constexpr uint32_t FLAG_AGGREGATE = 1u << 30;
constexpr uint32_t FLAG_PREFIX = 2u << 30;
constexpr uint32_t COUNT_MASK = (1u << 30) - 1;
constexpr uint32_t FULL = 0xffffffffu;
constexpr uint32_t NO_CLASS = 0xffu;

struct Smem {
    uint32_t excl[PAIRS][TILE_HEADS];  // each head's exclusive in-tile counts
    uint32_t job_deferred[TILE_HEADS];
    uint64_t band_class[TILE_HEADS];  // byte b: band b's class, NO_CLASS when uncovered
    uint32_t band_rank[TILE_HEADS];   // nibble b: band b's rank among its head's own
    uint32_t warp_total[COUNT_WARPS][PAIRS];
    uint32_t block_total[PAIRS];
    uint32_t prefix[MAX_CLASS];  // the tile's exclusive prefix per class
    uint32_t lease[MAX_CLASS];
    uint32_t stack_off[MAX_CLASS];
    uint8_t fmt[N_CODES];
    int8_t cls[N_CODES];
};

__device__ __forceinline__ int lane_of(const int4& v, int i) {
    return i == 0 ? v.x : i == 1 ? v.y : i == 2 ? v.z : v.w;
}

// A channel's palette slot, clamped into 0..3 as the host readback of the
// selection clamps it, so the two paths pack the same byte.
__device__ __forceinline__ uint32_t slot(int x) { return (uint32_t)min(max(x, 0), 3); }

__device__ __forceinline__ uint32_t pack4(const int4& v) {
    return slot(v.x) | (slot(v.y) << 2) | (slot(v.z) << 4) | (slot(v.w) << 6);
}

__device__ __forceinline__ uint64_t load_status(const uint64_t* p) {
    return *(const volatile uint64_t*)p;
}

__device__ __forceinline__ void store_status(uint64_t* p, uint32_t epoch, uint32_t word) {
    *(volatile uint64_t*)p = ((uint64_t)epoch << 32) | word;
}

// Five blocks an SM: at hd 128 a block's staged rows and counters are ~16 KB of
// shared memory, which already holds an sm_86 SM to five, so the register cap
// is set to match (48 registers, no spills) rather than to a sixth block that
// would not fit.
template <int HD>
__global__ void __launch_bounds__(THREADS, 5) gap_claim_kernel(
    const int32_t* __restrict__ k_tags,
    const int32_t* __restrict__ v_tags,
    const float* __restrict__ k_scale,
    const float* __restrict__ v_scale,
    const int32_t* __restrict__ k_map,
    const int32_t* __restrict__ v_map,
    int32_t k_override,
    int32_t v_override,
    const int8_t* __restrict__ code_class,
    const uint8_t* __restrict__ code_fmt,
    const uint32_t* __restrict__ lease_len,
    const uint64_t* __restrict__ stack,
    const uint32_t* __restrict__ stack_off,
    uint32_t* __restrict__ taken,
    uint64_t sink,
    uint8_t* __restrict__ dst_records,
    uint8_t* __restrict__ summary_heads,
    uint32_t* __restrict__ summary_flags,
    uint32_t* __restrict__ error_word,
    uint64_t* __restrict__ status,
    uint32_t epoch,
    int32_t n_jobs,
    int32_t n_kv_head,
    int32_t n_class)
{
    constexpr int REC = HD / 2 + 104;
    constexpr int SROW = REC + BANDS_PER_HEAD * 4;
    constexpr int MAP_WORDS = HD / 16;  // u32 words of one side's packed map
    constexpr int HEAD_WORDS = 2 * MAP_WORDS;
    constexpr int MAP_ITERS = TILE_HEADS * HEAD_WORDS / THREADS;
    static_assert(TILE_HEADS * HEAD_WORDS % THREADS == 0, "map words split evenly");
    static_assert(REC % 8 == 0 && SROW % 8 == 0, "rows are whole 8-byte words");

    __shared__ Smem sm;
    __shared__ uint64_t rows[TILE_HEADS * SROW / 8];
    uint8_t* row_bytes = (uint8_t*)rows;
    const int t = threadIdx.x;
    const int lane = t & 31;
    const int warp = t >> 5;
    const uint32_t tile = blockIdx.x;

    if (t < N_CODES) {
        const int c = code_class[t];
        sm.cls[t] = (c >= 0 && c < n_class) ? (int8_t)c : (int8_t)-1;
        sm.fmt[t] = code_fmt[t];
    }
    if (t < n_class) {
        sm.lease[t] = lease_len[t];
        sm.stack_off[t] = stack_off[t];
    }
    if (t < TILE_HEADS) sm.job_deferred[t] = 0;
    __syncthreads();

    const int jobs_per_tile = TILE_HEADS / n_kv_head;
    const int heads_per_tile = jobs_per_tile * n_kv_head;
    const int head0 = (int)tile * heads_per_tile;
    const int tile_heads = min(heads_per_tile, n_jobs * n_kv_head - head0);
    const int head = head0 + t;
    const bool counter = t < TILE_HEADS;
    const bool live = t < tile_heads;

    // ── This head's eight bands: class, rank among its own, record format ──
    int4 kc = make_int4(k_override, k_override, k_override, k_override);
    int4 vc = make_int4(v_override, v_override, v_override, v_override);
    if (live && k_override < 0) kc = __ldg((const int4*)k_tags + head);
    if (live && v_override < 0) vc = __ldg((const int4*)v_tags + head);
    uint64_t local = 0;    // nibble c: this thread's bands of class c so far
    uint64_t classes = 0;  // byte b: band b's class, NO_CLASS when uncovered
    uint32_t ranks = 0;    // nibble b: band b's rank among this thread's own
    uint32_t k_fmt = 0, v_fmt = 0;
    bool bad = false;
    #pragma unroll
    for (int b = 0; b < BANDS_PER_HEAD; ++b) {
        const int p = b >> 1;
        const int code = (b & 1) ? lane_of(vc, p) : lane_of(kc, p);
        const bool known = code >= 0 && code < N_CODES;
        const int c = known ? sm.cls[code] : -1;
        const uint32_t f = known ? sm.fmt[code] : 0u;
        if (b & 1) v_fmt |= f << (8 * p); else k_fmt |= f << (8 * p);
        if (c < 0) {
            bad |= live;
            classes |= (uint64_t)NO_CLASS << (8 * b);
        } else if (live) {
            ranks |= (uint32_t)((local >> (4 * c)) & 0xf) << (4 * b);
            local += 1ull << (4 * c);
            classes |= (uint64_t)c << (8 * b);
        }
    }
    // Parked until the indices are known, so nothing per band stays live in
    // registers through the map packing and the look-back.
    if (live) {
        rows[t * (SROW / 8) + (HD / 2 + 64) / 8] = (uint64_t)k_fmt | ((uint64_t)v_fmt << 32);
        sm.band_class[t] = classes;
        sm.band_rank[t] = ranks;
        if (bad) {
            // A code the host's table does not cover: the host validated every
            // candidate before launch, so this is a broken table. The job
            // defers. A plain store: every writer stores the same value, and
            // the word may be mapped host memory, where a system-scope atomic
            // is not available everywhere.
            atomicOr(&sm.job_deferred[t / n_kv_head], 1u);
            *(volatile uint32_t*)error_word = 1u;
        }
    }

    // ── In-block exclusive counts: warp scan, then across warps ──
    uint32_t cnt[PAIRS];
    #pragma unroll
    for (int r = 0; r < PAIRS; ++r) {
        cnt[r] = (uint32_t)((local >> (8 * r)) & 0xf) | ((uint32_t)((local >> (8 * r + 4)) & 0xf) << 16);
    }
    if (counter) {  // warp-uniform: TILE_HEADS is a multiple of 32
        #pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            #pragma unroll
            for (int r = 0; r < PAIRS; ++r) {
                const uint32_t y = __shfl_up_sync(FULL, cnt[r], d);
                if (lane >= d) cnt[r] += y;
            }
        }
        if (lane == 31) {
            #pragma unroll
            for (int r = 0; r < PAIRS; ++r) sm.warp_total[warp][r] = cnt[r];
        }
        #pragma unroll
        for (int r = 0; r < PAIRS; ++r) {
            const uint32_t y = __shfl_up_sync(FULL, cnt[r], 1);
            cnt[r] = lane == 0 ? 0u : y;
        }
    }
    __syncthreads();
    if (t < PAIRS) {
        uint32_t run = 0;
        #pragma unroll
        for (int w = 0; w < COUNT_WARPS; ++w) {
            const uint32_t x = sm.warp_total[w][t];
            sm.warp_total[w][t] = run;
            run += x;
        }
        sm.block_total[t] = run;
    }
    __syncthreads();
    if (counter) {
        #pragma unroll
        for (int r = 0; r < PAIRS; ++r) sm.excl[r][t] = cnt[r] + sm.warp_total[warp][r];
    }

    // ── Publish the tile's totals, so successors can look past it ──
    if (t < n_class) {
        const uint32_t agg = (sm.block_total[t >> 1] >> (16 * (t & 1))) & 0xffffu;
        if (tile == 0) {
            const uint32_t pre = taken[t];
            sm.prefix[t] = pre;
            store_status(status + t, epoch, FLAG_PREFIX | (pre + agg));
            if (gridDim.x == 1) taken[t] = pre + agg;
        } else {
            store_status(status + (size_t)tile * MAX_CLASS + t, epoch, FLAG_AGGREGATE | agg);
        }
    }

    // ── Packed palette maps, while predecessors resolve ──
    // Each word is sixteen channels' slots: four 16-byte loads of the
    // selection's per-channel map, consecutive threads on consecutive words.
    const int words = tile_heads * HEAD_WORDS;
    #pragma unroll
    for (int it = 0; it < MAP_ITERS; ++it) {
        const int i = t + it * THREADS;
        if (i < words) {
            const int hl = i / HEAD_WORDS;
            const int side = (i % HEAD_WORDS) / MAP_WORDS;
            const int w = i % MAP_WORDS;
            const bool identity = side ? v_override >= 0 : k_override >= 0;
            uint32_t word;
            if (identity) {
                // Sixteen channels all in one quarter of the head.
                word = (uint32_t)((64 * w) / HD) * 0x55555555u;
            } else {
                const int4* src =
                    (const int4*)((side ? v_map : k_map) + (size_t)(head0 + hl) * HD + 16 * w);
                const int4 a = __ldcs(src), b = __ldcs(src + 1), c = __ldcs(src + 2), d = __ldcs(src + 3);
                word = pack4(a) | (pack4(b) << 8) | (pack4(c) << 16) | (pack4(d) << 24);
            }
            *(uint32_t*)(row_bytes + hl * SROW + side * (HD / 4) + 4 * w) = word;
        }
    }

    // ── Look back for the tile's exclusive prefix ──
    // Sixteen lanes per class, each reading a different predecessor: a window
    // of sixteen tiles costs one load, and the nearest tile already holding its
    // inclusive prefix ends the walk. Tile 0 always holds one.
    {
        const int c = t / LOOK_LANES;
        const int j = t % LOOK_LANES;
        const uint32_t seg = 0xffffu << (lane & 16);
        if (tile > 0 && c < n_class) {  // uniform across the class's sixteen lanes
            uint32_t run = 0;
            for (int base = (int)tile - 1;; base -= LOOK_LANES) {
                const int k = base - j;
                uint32_t lo = FLAG_PREFIX;  // past tile 0: nothing more to add
                if (k >= 0) {
                    uint64_t s;
                    do {
                        s = load_status(status + (size_t)k * MAX_CLASS + c);
                    } while ((uint32_t)(s >> 32) != epoch || ((uint32_t)s & ~COUNT_MASK) == 0);
                    lo = (uint32_t)s;
                }
                const uint32_t prefix_lanes =
                    (__ballot_sync(seg, (lo & ~COUNT_MASK) == FLAG_PREFIX) >> (lane & 16)) & 0xffffu;
                // Lanes up to and including the nearest prefix (all of them
                // when the window holds none) add their counts.
                const int stop = prefix_lanes ? __ffs(prefix_lanes) - 1 : LOOK_LANES - 1;
                uint32_t add = j <= stop ? (lo & COUNT_MASK) : 0u;
                #pragma unroll
                for (int o = LOOK_LANES / 2; o > 0; o >>= 1) add += __shfl_xor_sync(seg, add, o, LOOK_LANES);
                run += add;
                if (prefix_lanes) break;
            }
            if (j == 0) {
                sm.prefix[c] = run;
                const uint32_t agg = (sm.block_total[c >> 1] >> (16 * (c & 1))) & 0xffffu;
                store_status(status + (size_t)tile * MAX_CLASS + c, epoch, FLAG_PREFIX | (run + agg));
                if (tile == gridDim.x - 1) taken[c] = run + agg;
            }
        }
    }
    __syncthreads();

    // ── Indices and deferral ──
    uint32_t idx[BANDS_PER_HEAD];
    const uint64_t parked_class = live ? sm.band_class[t] : ~0ull;
    const uint32_t parked_rank = live ? sm.band_rank[t] : 0u;
    bool deferred_here = false;
    #pragma unroll
    for (int b = 0; b < BANDS_PER_HEAD; ++b) {
        const uint32_t c = (uint32_t)(parked_class >> (8 * b)) & 0xffu;
        if (c == NO_CLASS) {
            idx[b] = 0xffffffffu;
        } else {
            const uint32_t e = (sm.excl[c >> 1][t] >> (16 * (c & 1))) & 0xffffu;
            idx[b] = sm.prefix[c] + e + ((parked_rank >> (4 * b)) & 0xf);
            deferred_here |= idx[b] >= sm.lease[c];
        }
    }
    const int job_local = t / n_kv_head;
    if (live && deferred_here) atomicOr(&sm.job_deferred[job_local], 1u);
    __syncthreads();

    // ── This head's slot addresses, formats, scales and lease indices ──
    if (live) {
        const bool deferred = sm.job_deferred[job_local] != 0;
        uint64_t* r = rows + t * (SROW / 8);
        #pragma unroll
        for (int b = 0; b < BANDS_PER_HEAD; ++b) {
            const uint32_t c = (uint32_t)(parked_class >> (8 * b)) & 0xffu;
            r[(HD / 2) / 8 + (b & 1) * 4 + (b >> 1)] =
                (deferred || c == NO_CLASS) ? sink : __ldg(stack + sm.stack_off[c] + idx[b]);
        }
        float4 ks = make_float4(1.f, 1.f, 1.f, 1.f), vs = ks;
        if (k_override < 0) ks = __ldg((const float4*)k_scale + head);
        if (v_override < 0) vs = __ldg((const float4*)v_scale + head);
        r[(HD / 2 + 72) / 8 + 0] = (uint64_t)__float_as_uint(ks.x) | ((uint64_t)__float_as_uint(ks.y) << 32);
        r[(HD / 2 + 72) / 8 + 1] = (uint64_t)__float_as_uint(ks.z) | ((uint64_t)__float_as_uint(ks.w) << 32);
        r[(HD / 2 + 72) / 8 + 2] = (uint64_t)__float_as_uint(vs.x) | ((uint64_t)__float_as_uint(vs.y) << 32);
        r[(HD / 2 + 72) / 8 + 3] = (uint64_t)__float_as_uint(vs.z) | ((uint64_t)__float_as_uint(vs.w) << 32);
        #pragma unroll
        for (int q = 0; q < 4; ++q) {
            r[REC / 8 + q] = (uint64_t)idx[2 * q] | ((uint64_t)idx[2 * q + 1] << 32);
        }
        if (t % n_kv_head == 0) {
            summary_flags[head / n_kv_head] = deferred ? 1u : 0u;
        }
    }
    __syncthreads();

    // ── The tile's rows out: two contiguous spans, consecutive threads on
    // consecutive words ──
    uint64_t* summary_out = (uint64_t*)(summary_heads + (size_t)head0 * SROW);
    for (int i = t; i < tile_heads * (SROW / 8); i += THREADS) summary_out[i] = rows[i];
    uint64_t* dst_out = (uint64_t*)(dst_records + (size_t)head0 * REC);
    for (int i = t; i < tile_heads * (REC / 8); i += THREADS) {
        const int hl = i / (REC / 8);
        dst_out[i] = rows[hl * (SROW / 8) + i % (REC / 8)];
    }
}

__global__ void gap_complete_kernel(uint32_t* word, uint32_t seq) {
    // Everything enqueued before this on the stream has completed; the fence
    // orders it ahead of the word for the host that polls it.
    __threadfence_system();
    *(volatile uint32_t*)word = seq;
}

}  // namespace kv_gap_claim

extern "C" int run_kv_gap_claim(
    const int32_t* k_tags,
    const int32_t* v_tags,
    const float* k_scale,
    const float* v_scale,
    const int32_t* k_map,
    const int32_t* v_map,
    int32_t k_override,
    int32_t v_override,
    const int8_t* code_class,
    const uint8_t* code_fmt,
    const uint32_t* lease_len,
    const uint64_t* stack,
    const uint32_t* stack_off,
    uint32_t* taken,
    uint64_t sink,
    uint8_t* dst_records,
    uint8_t* summary_heads,
    uint32_t* summary_flags,
    uint32_t* error_word,
    uint64_t* status,
    uint32_t epoch,
    int32_t n_jobs,
    int32_t n_kv_head,
    int32_t head_dim,
    int32_t n_class,
    void* stream)
{
    using namespace kv_gap_claim;
    if (n_jobs <= 0 || n_kv_head <= 0 || n_kv_head > MAX_KV_HEAD || n_class <= 0 ||
        n_class > MAX_CLASS) {
        return -1;
    }
    const int jobs_per_tile = TILE_HEADS / n_kv_head;
    const unsigned grid = (unsigned)((n_jobs + jobs_per_tile - 1) / jobs_per_tile);
    cudaStream_t s = (cudaStream_t)stream;
#define KV_GAP_CLAIM_ARGS                                                                          \
    k_tags, v_tags, k_scale, v_scale, k_map, v_map, k_override, v_override, code_class, code_fmt, \
        lease_len, stack, stack_off, taken, sink, dst_records, summary_heads, summary_flags,      \
        error_word, status, epoch, n_jobs, n_kv_head, n_class
    switch (head_dim) {
        case 64: gap_claim_kernel<64><<<grid, THREADS, 0, s>>>(KV_GAP_CLAIM_ARGS); break;
        case 128: gap_claim_kernel<128><<<grid, THREADS, 0, s>>>(KV_GAP_CLAIM_ARGS); break;
        case 256: gap_claim_kernel<256><<<grid, THREADS, 0, s>>>(KV_GAP_CLAIM_ARGS); break;
        case 512: gap_claim_kernel<512><<<grid, THREADS, 0, s>>>(KV_GAP_CLAIM_ARGS); break;
        default: return -1;
    }
#undef KV_GAP_CLAIM_ARGS
    return (int)cudaGetLastError();
}

extern "C" void run_kv_gap_complete(uint32_t* word, uint32_t seq, void* stream) {
    kv_gap_claim::gap_complete_kernel<<<1, 1, 0, (cudaStream_t)stream>>>(word, seq);
}
