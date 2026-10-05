// =============================================================================
// qsa_topk — QSA block selection: per-query top-k over indexer scores
// =============================================================================
//
// The device half of `models::qwen4exp::qsa_select::selection_entries`. Given
// one row of indexer scores per query — `s(t,b) = Σ_h ReLU(⟨q_th, k̄_b⟩)` over
// the complete index blocks below the query's tail — it emits that query's
// selection as the packed ascending entry list the attention kernels read
// (`../qsa_select.cuh`).
//
// WHAT MAKES THIS EXACT
// ---------------------
// The reference ranks cells by `(score desc, cell asc)`, so equal scores must
// resolve to the LOWER block. Scores are sums of ReLUs and therefore never
// negative, which makes the IEEE-754 bit pattern of a float order-isomorphic
// to its value — so one 64-bit key
//
//     key(b) = bits(score) << 32 | (0xFFFFFFFF − b)
//
// is a total order that is exactly `(score desc, block asc)` when compared as
// an unsigned integer. Distinct blocks give distinct keys, so there are no
// ties to resolve by any other rule, and the selection is reproducible. No
// real key is zero (`b < 0xFFFFFFFF`), so zero pads a key list harmlessly.
//
// THE SELECTION, WITHOUT SORTING ANYTHING BUT THE ANSWER
// ------------------------------------------------------
// A depth-L sequence has ~L/ratio candidate blocks and we need the best ~512
// of them, so sorting the row would be the wrong shape by three orders of
// magnitude. Instead each block streams its candidates through a shared
// buffer against a running threshold:
//
//   - The buffer's first `count` slots hold every key still in the running,
//     in no order. A chunk appends only keys above the threshold.
//   - When the buffer could not absorb another chunk, a TRIM selects the top
//     `keep` of it and compacts them to the front, and the smallest of them
//     becomes the threshold: a later key that does not beat it cannot be in
//     the top `keep`, because `keep` keys above it are already known. After
//     the first few thousand blocks the threshold is high enough that trims
//     become rare.
//
// The trim is an MSB radix select over the keys, held in registers: a
// histogram of the next byte of every key still matching the chosen prefix
// names the byte the `keep`-th key has, and the walk stops at the first byte
// whose bin holds exactly the keys still needed. Nothing is sorted. A ranking
// needs only two facts — which keys are in the top `keep`, and which of them
// is last, the one the budget's partial cut lands in — and the trim yields
// both. The chosen blocks are sorted once, by block, when the row is written.
//
// `keep ≤ CAP − THREADS` is what bounds the append (a chunk adds at most
// THREADS keys and is checked immediately after), and the host refuses a
// `top_k`/`ratio` pair that would break it rather than silently truncating.
//
// STRATIFIED SELECTION (`docs/qsa_stratified_selection.md`)
// ---------------------------------------------------------
// The candidates may be cut into windows of `window_blocks` blocks walking
// forward from block 0. Each window streams its own POOL through the buffer
// above, from scratch, and spends the whole budget on it:
//
//   - its own blocks,
//   - the system prompt's leading blocks outside it,
//   - as candidates, the recent span's blocks outside both.
//
// A FORCED recent span is instead left out of every pool and attended whole.
// Every window's choices are gathered in one shared entry buffer, sorted
// ascending, and — when a block was chosen by more than one window — reduced to
// its widest cut. Packed entries sort by block and then by cell count, so a
// block's widest cut is the LAST of its run. The forced span and the query's
// tail sit above every pool, so they follow the reduced entries in order and
// are written without being sorted at all.
//
// One window (`window_blocks == 0`, or wider than the row) makes the pool every
// candidate: the prompt and recent spans fall inside it, nothing can repeat,
// and the kernel does exactly the work of the checkpoint's selection.
//
// SPLITTING A ROW ACROSS BLOCKS
// -----------------------------
// One block per row is the right shape for a prefill tile — a thousand rows
// fill the device — and the wrong one for decode, where a handful of rows
// leaves almost every SM idle while each block walks ~70K candidates in
// sequence. So a narrow launch splits each window's own blocks into `parts`
// contiguous slices, one block each (`qsa_topk_segment_kernel`), and a second
// pass merges a row's slices (`qsa_topk_merge_kernel`).
//
// That is exact, not approximate: a key in the window's top `keep` is in the
// top `keep` of whichever slice holds it, because nothing outside the slice
// can outrank it inside the slice. So the merge ranks the union of the slices'
// survivors and finds the same top `keep` the single pass would have.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <atomic>
#include <math.h>
#include <stdint.h>

#include "../qsa_select.cuh"

namespace qsa_topk {

constexpr int THREADS = 256;
constexpr int CAP = 1024;          // shared candidate buffer, u64 keys
// The append bound: a chunk adds at most THREADS keys into the free space
// below the survivors, and the trim below fires as soon as the free space
// could not absorb another chunk.
constexpr int MAX_KEEP = CAP - THREADS;
// The union's sort buffer ceiling, in u32 entries. With the survivor buffer it
// is the kernel's dynamic shared memory: 8 KiB + 64 KiB, inside the opt-in
// limit of every architecture the archive carries.
constexpr int MAX_ENTRIES = 16384;
// The most blocks a split launch may run — the scratch the host provides holds
// this many slices of MAX_KEEP keys.
constexpr int SPLIT_BLOCKS = 512;

__device__ __forceinline__ uint32_t key_block(unsigned long long key) {
    return 0xFFFFFFFFu - (uint32_t)(key & 0xFFFFFFFFull);
}

__device__ __forceinline__ unsigned long long score_key(float score, int b) {
    return ((unsigned long long)__float_as_uint(score) << 32)
        | (unsigned long long)(0xFFFFFFFFu - (uint32_t)b);
}

__device__ __forceinline__ int span_len(int lo, int hi) {
    return hi > lo ? hi - lo : 0;
}

__device__ __forceinline__ int overlap(int a, int b, int c, int d) {
    return span_len(a > c ? a : c, b < d ? b : d);
}

// Bitonic sort of `n` (a power of two) shared values, ascending.
template <typename T>
__device__ void bitonic_ascending(T* buf, int n, int tid) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < n; i += THREADS) {
                int ixj = i ^ j;
                if (ixj > i) {
                    bool up = ((i & k) == 0);
                    T a = buf[i];
                    T b = buf[ixj];
                    if ((a > b) == up) {
                        buf[i] = b;
                        buf[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// One query row's geometry: its budget and the spans every window shares.
struct Row {
    int cand;       // candidate blocks below the tail
    int n_tail;     // cells of its own block the query has
    int full;       // whole blocks the budget buys
    int rem;        // cells of the block the budget runs out inside
    int keep;       // blocks a window keeps: `full`, plus the partial one
    int prompt;     // prompt blocks, clipped to the row
    int recent_lo;  // first block of the recent span
    int forced_lo;  // first block of the forced span (`cand` when none)
    int win;        // blocks per window
    int n_win;      // windows
};

// `false` for a dense row — every visible cell attended, nothing to select.
__device__ __forceinline__ bool row_geometry(
    int row,
    const uint32_t* __restrict__ n_cand,
    const uint32_t* __restrict__ qpos,
    const uint32_t* __restrict__ tail_len,
    const uint32_t* __restrict__ prompt_blocks,
    int ratio,
    int top_k,
    int window_blocks,
    int recent_blocks,
    int recent_forced,
    Row& g
) {
    const int width = top_k + ratio - 1;
    if ((int)qpos[row] + 1 <= width) return false;
    // **The tail comes from the host, and the block index it lands in is
    // `n_cand[row]` — not `visible / ratio`.**
    //
    // Those agree only while every block covers `ratio` consecutive positions.
    // A sequence whose prefix arrived as separately sealed pieces has a short
    // block at each boundary, so a position no longer divides into its block;
    // what does not change is that if `C` blocks sit wholly below the query, the
    // query is in block `C`. So the identity is used and the arithmetic is not,
    // and the one quantity that cannot be recovered here — how many cells of its
    // own block the query has — is passed in.
    g.n_tail = (int)tail_len[row];
    g.cand = (int)n_cand[row];
    const int budget = width - g.n_tail;
    g.full = budget / ratio;
    g.rem = budget - g.full * ratio;
    g.keep = g.full + (g.rem > 0 ? 1 : 0);
    // The append bound the buffer arithmetic rests on. The host refuses a
    // `top_k`/`ratio` pair that exceeds it (`qsa_topk::MAX_KEEP` is mirrored
    // there), so this clamp cannot fire; it is here so that a precondition
    // broken upstream degrades the selection instead of writing past `buf`.
    if (g.keep > MAX_KEEP) g.keep = MAX_KEEP;
    const int p = (int)prompt_blocks[row];
    g.prompt = p < g.cand ? p : g.cand;
    g.recent_lo = g.cand - (recent_blocks < g.cand ? recent_blocks : g.cand);
    g.forced_lo = recent_forced ? g.recent_lo : g.cand;
    g.win = (window_blocks > 0 && window_blocks < g.cand) ? window_blocks : g.cand;
    g.n_win = g.win > 0 ? (g.cand + g.win - 1) / g.win : 0;
    return true;
}

// Window `w`'s pool, as three disjoint ranges: R1 the window less the forced
// span, R2 the prompt less the window and the forced span, and — as candidates
// — R3 the recent span less the window and the prompt. The forced span is the
// top of the row, so clipping each range's end at `forced_lo` removes it.
struct Window {
    int lo, hi;  // the window
    int r1_hi;   // R1 = [lo, r1_hi)
    int r2_hi;   // R2 = [0, r2_hi) ∖ [lo, hi)
    bool r3;     // R3 = [recent_lo, cand) ∖ [lo, hi) ∖ [0, prompt)
};

__device__ __forceinline__ Window window_of(const Row& g, int w, int recent_forced) {
    Window v;
    v.lo = w * g.win;
    v.hi = v.lo + g.win < g.cand ? v.lo + g.win : g.cand;
    v.r1_hi = v.hi < g.forced_lo ? v.hi : g.forced_lo;
    v.r2_hi = g.prompt < g.forced_lo ? g.prompt : g.forced_lo;
    v.r3 = !recent_forced;
    return v;
}

// |R2| + |R3| — the part of a window's pool outside its own blocks.
__device__ __forceinline__ int shared_pool(const Row& g, const Window& v) {
    int n = v.r2_hi - overlap(0, v.r2_hi, v.lo, v.hi);
    if (v.r3) {
        // |R ∖ W ∖ P| = |R| − |R∩W| − |R∩P| + |R∩W∩P|, where R = [recent_lo, cand)
        // and R∩W∩P is the overlap of R∩W = [max(recent_lo, lo), hi) with P.
        const int rw_lo = g.recent_lo > v.lo ? g.recent_lo : v.lo;
        n += span_len(g.recent_lo, g.cand) - overlap(g.recent_lo, g.cand, v.lo, v.hi)
            - overlap(g.recent_lo, g.cand, 0, g.prompt) + overlap(rw_lo, v.hi, 0, g.prompt);
    }
    return n;
}

constexpr int PER_THREAD = CAP / THREADS;
constexpr int BINS = 256;
constexpr int WARP = 32;

// The survivor buffer and the radix select's state, in shared memory.
struct Survivors {
    unsigned long long* buf;  // CAP keys; [0, count) live, in no order
    int* count;
    unsigned long long* thr;  // a key must exceed this to enter
    int* hist;                // BINS counters
    int* pick;                // the select's answer: digit, keys above it, its bin
};

__device__ __forceinline__ void survivors_reset(const Survivors& s, int tid) {
    if (tid == 0) {
        *s.count = 0;
        *s.thr = 0ull;
    }
    __syncthreads();
}

// Reduce the buffer's `n` keys to its top `keep`, compacted to the front, and
// set the threshold to the smallest of them. `n ≥ keep`. Every thread calls
// this, so its barriers are uniform.
__device__ void survivors_select(const Survivors& s, int n, int keep, int tid) {
    unsigned long long k[PER_THREAD];
    bool live[PER_THREAD];
#pragma unroll
    for (int j = 0; j < PER_THREAD; ++j) {
        const int i = tid + j * THREADS;
        live[j] = i < n;
        k[j] = live[j] ? s.buf[i] : 0ull;
    }

    // The top `keep` are the keys whose masked bits are at least `prefix`.
    unsigned long long prefix = 0ull, mask = 0ull;
    int need = keep;
    if (n > keep) {
        for (int shift = 56; shift >= 0; shift -= 8) {
            s.hist[tid] = 0;
            __syncthreads();
#pragma unroll
            for (int j = 0; j < PER_THREAD; ++j) {
                if (live[j] && (k[j] & mask) == prefix) {
                    atomicAdd(&s.hist[(int)((k[j] >> shift) & 0xFFull)], 1);
                }
            }
            __syncthreads();
            if (tid < WARP) {
                // Lane l holds bins [8l, 8l + 8); `above` counts the keys in
                // every higher lane's bins.
                int bins[BINS / WARP];
                int t = 0;
#pragma unroll
                for (int b = 0; b < BINS / WARP; ++b) {
                    bins[b] = s.hist[tid * (BINS / WARP) + b];
                    t += bins[b];
                }
                int incl = t;
#pragma unroll
                for (int off = 1; off < WARP; off <<= 1) {
                    const int v = __shfl_down_sync(0xFFFFFFFFu, incl, off);
                    if (tid + off < WARP) incl += v;
                }
                int above = incl - t;
                if (above < need && need <= incl) {
                    for (int b = BINS / WARP - 1; b >= 0; --b) {
                        if (above + bins[b] >= need) {
                            s.pick[0] = tid * (BINS / WARP) + b;
                            s.pick[1] = above;
                            s.pick[2] = bins[b];
                            break;
                        }
                        above += bins[b];
                    }
                }
            }
            __syncthreads();
            prefix |= (unsigned long long)s.pick[0] << shift;
            mask |= 0xFFull << shift;
            need -= s.pick[1];
            const bool whole_bin = s.pick[2] == need;
            __syncthreads();
            if (whole_bin) break;
        }
    }

    if (tid == 0) {
        *s.count = 0;
        *s.thr = ~0ull;
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < PER_THREAD; ++j) {
        if (live[j] && (k[j] & mask) >= prefix) {
            s.buf[atomicAdd(s.count, 1)] = k[j];
            atomicMin(s.thr, k[j]);
        }
    }
    __syncthreads();
}

// Offer one chunk's keys (each thread's `key`, `0` for none) to the buffer,
// then trim it if another chunk might not fit. Every thread calls this the
// same number of times, so the barriers inside are uniform.
//
// `n` is each thread's own copy of the buffer's count, advanced by the
// barrier's own tally of the chunk. The trim decision has to be the same in
// every thread, and a count read back from shared memory is not: a thread
// that reads it late can see the next chunk's appends from a thread that ran
// ahead, decide differently, and leave the block split across two barriers.
__device__ __forceinline__ void survivors_offer(
    const Survivors& s,
    unsigned long long key,
    int keep,
    int& n,
    int tid
) {
    const bool append = key > *s.thr;
    if (append) {
        const int slot = atomicAdd(s.count, 1);
        s.buf[slot] = key;
    }
    n += __syncthreads_count(append);
    if (n > CAP - THREADS) {
        survivors_select(s, n, keep, tid);
        n = keep;
    }
}

// Settle the buffer to exactly its top `keep` once the pool is streamed.
__device__ __forceinline__ void survivors_finish(const Survivors& s, int keep, int n, int tid) {
    survivors_select(s, n, keep, tid);
}

// Stream the scores of `[lo, hi)` that `skip` does not exclude.
template <typename Skip>
__device__ void offer_scores(
    const Survivors& s,
    const float* __restrict__ srow,
    int lo,
    int hi,
    Skip skip,
    int keep,
    int& n,
    int tid
) {
    for (int base = lo; base < hi; base += THREADS) {
        const int b = base + tid;
        const unsigned long long key = (b < hi && !skip(b)) ? score_key(srow[b], b) : 0ull;
        survivors_offer(s, key, keep, n, tid);
    }
}

// Stream window `v`'s pool: the slice `[own_lo, own_hi)` of its own blocks,
// then R2 and R3 when `shared` says this pass ranks them. Returns the
// buffer's count.
__device__ int offer_pool(
    const Survivors& s,
    const float* __restrict__ srow,
    const Row& g,
    const Window& v,
    int own_lo,
    int own_hi,
    bool shared,
    int keep,
    int tid
) {
    int n = 0;
    offer_scores(s, srow, own_lo, own_hi, [](int) { return false; }, keep, n, tid);
    if (!shared) return n;
    const int lo = v.lo, hi = v.hi, prompt = g.prompt;
    offer_scores(
        s, srow, 0, v.r2_hi, [lo, hi](int b) { return b >= lo && b < hi; }, keep, n, tid);
    if (v.r3) {
        offer_scores(
            s, srow, g.recent_lo, g.cand,
            [lo, hi, prompt](int b) { return (b >= lo && b < hi) || b < prompt; }, keep, n,
            tid);
    }
    return n;
}

// Append a settled buffer's `keep` keys as entries: whole blocks, except that
// when the budget runs out inside a block — `keep` is one past the whole
// blocks it buys — the last-ranked key, the threshold, contributes only its
// lowest cells.
__device__ void emit_window(
    const Survivors& s,
    const Row& g,
    int keep,
    int ratio,
    uint32_t* ent,
    int* n_ent,
    int tid
) {
    const int base_e = *n_ent;
    const bool part = g.rem > 0 && keep > g.full;
    const unsigned long long last = *s.thr;
    for (int r = tid; r < keep; r += THREADS) {
        const unsigned long long key = s.buf[r];
        const int cells = (part && key == last) ? g.rem : ratio;
        ent[base_e + r] = (key_block(key) << 2) | (uint32_t)(cells - 1);
    }
    __syncthreads();
    if (tid == 0) *n_ent = base_e + keep;
    __syncthreads();
}

// Sort the gathered window entries, keep each block's widest cut, and write
// them, the forced span and the tail as the row's selection. `scan` is THREADS
// ints of scratch.
__device__ void finish_row(
    const Row& g,
    int ratio,
    uint32_t* ent,
    int n_ent,
    int* scan,
    uint32_t* __restrict__ out,
    uint32_t* __restrict__ cnt_out,
    int tid
) {
    int sort_n = 1;
    while (sort_n < n_ent) sort_n <<= 1;
    for (int i = n_ent + tid; i < sort_n; i += THREADS) ent[i] = 0xFFFFFFFFu;
    __syncthreads();
    bitonic_ascending(ent, sort_n, tid);

    // Each thread owns a contiguous run of the sorted entries and keeps those
    // that end their block's run — the widest cut. A block-wide scan of the
    // per-thread counts places every kept entry.
    const int per = (n_ent + THREADS - 1) / THREADS;
    const int a = tid * per;
    const int b = a + per < n_ent ? a + per : n_ent;
    int mine = 0;
    for (int i = a; i < b; ++i) {
        if (i + 1 == n_ent || (ent[i] >> 2) != (ent[i + 1] >> 2)) mine += 1;
    }
    scan[tid] = mine;
    __syncthreads();
    for (int off = 1; off < THREADS; off <<= 1) {
        const int v = tid >= off ? scan[tid - off] : 0;
        __syncthreads();
        scan[tid] += v;
        __syncthreads();
    }
    int at = scan[tid] - mine;
    for (int i = a; i < b; ++i) {
        if (i + 1 == n_ent || (ent[i] >> 2) != (ent[i + 1] >> 2)) out[at++] = ent[i];
    }
    const int kept = scan[THREADS - 1];

    const int n_forced = g.cand - g.forced_lo;
    for (int i = tid; i < n_forced; i += THREADS) {
        out[kept + i] = ((uint32_t)(g.forced_lo + i) << 2) | (uint32_t)(ratio - 1);
    }
    if (tid == 0) {
        int n = kept + n_forced;
        if (g.n_tail > 0) {
            out[n] = ((uint32_t)g.cand << 2) | (uint32_t)(g.n_tail - 1);
            n += 1;
        }
        *cnt_out = (uint32_t)n;
    }
}

// The shared-memory layout every kernel here uses: CAP u64 survivors, then the
// entry buffer.
struct Smem {
    Survivors s;
    uint32_t* ent;
};

__device__ __forceinline__ Smem smem_layout(
    unsigned long long* dyn,
    int* count,
    unsigned long long* thr,
    int* hist,
    int* pick
) {
    Smem m;
    m.s.buf = dyn;
    m.s.count = count;
    m.s.thr = thr;
    m.s.hist = hist;
    m.s.pick = pick;
    m.ent = (uint32_t*)(dyn + CAP);
    return m;
}

// One block per query row — the whole selection in one pass.
//
//   scores  [n_rows, score_stride] f32 — row r's block scores, valid on
//                                        [0, n_cand[r]); the rest is ignored
//   n_cand  [n_rows] u32 — candidate blocks (the query's tail_start / ratio)
//   qpos    [n_rows] u32 — the query's ABSOLUTE position
//   tail    [n_rows] u32 — cells of its OWN block the query has, `1..=ratio`.
//                          Not derived here: a block's width is a property of
//                          the page it belongs to.
//   prompt  [n_rows] u32 — blocks wholly inside the row's system prompt
//   entries [n_rows, entry_stride] u32 — output, ascending by block
//   cnt     [n_rows] u32 — entries written, or QSA_DENSE_ROW
//
// Dynamic shared memory: CAP u64 survivors, then `ent_cap` u32 entries.
__global__ void __launch_bounds__(THREADS) qsa_topk_entries_kernel(
    const float* __restrict__ scores,
    int score_stride,
    const uint32_t* __restrict__ n_cand,
    const uint32_t* __restrict__ qpos,
    const uint32_t* __restrict__ tail_len,
    const uint32_t* __restrict__ prompt_blocks,
    uint32_t* __restrict__ entries,
    int entry_stride,
    uint32_t* __restrict__ cnt,
    int ratio,
    int top_k,
    int window_blocks,
    int recent_blocks,
    int recent_forced,
    int n_rows
) {
    const int row = (int)blockIdx.x;
    if (row >= n_rows) return;
    const int tid = (int)threadIdx.x;
    Row g;
    if (!row_geometry(
            row, n_cand, qpos, tail_len, prompt_blocks, ratio, top_k, window_blocks,
            recent_blocks, recent_forced, g)) {
        // Every visible cell is attended — the identity, and the reason a
        // shallow context needs no indexer at all.
        if (tid == 0) cnt[row] = QSA_DENSE_ROW;
        return;
    }

    extern __shared__ unsigned long long smem_qsa_topk[];
    __shared__ int s_count;
    __shared__ int s_n_ent;
    __shared__ unsigned long long s_thr;
    __shared__ int s_hist[BINS];
    __shared__ int s_pick[3];
    const Smem m = smem_layout(smem_qsa_topk, &s_count, &s_thr, s_hist, s_pick);
    if (tid == 0) s_n_ent = 0;

    const float* srow = scores + (size_t)row * (size_t)score_stride;
    for (int w = 0; w < g.n_win; ++w) {
        const Window v = window_of(g, w, recent_forced);
        const int pool = span_len(v.lo, v.r1_hi) + shared_pool(g, v);
        const int keep = g.keep < pool ? g.keep : pool;
        if (keep <= 0) continue;
        survivors_reset(m.s, tid);
        const int n = offer_pool(m.s, srow, g, v, v.lo, v.r1_hi, true, keep, tid);
        survivors_finish(m.s, keep, n, tid);
        emit_window(m.s, g, keep, ratio, m.ent, &s_n_ent, tid);
    }
    __syncthreads();
    finish_row(
        g, ratio, m.ent, s_n_ent, (int*)m.s.buf,
        entries + (size_t)row * (size_t)entry_stride, cnt + row, tid);
}

// The split launch's first pass: block (row, w·parts + p) ranks slice `p` of
// window `w`'s own blocks — plus, for slice 0, the prompt and recent spans the
// window shares — and writes its top `g.keep` keys, in no order and
// zero-padded, to `keys[(row · segs + seg) · MAX_KEEP ..]`.
__global__ void __launch_bounds__(THREADS) qsa_topk_segment_kernel(
    const float* __restrict__ scores,
    int score_stride,
    const uint32_t* __restrict__ n_cand,
    const uint32_t* __restrict__ qpos,
    const uint32_t* __restrict__ tail_len,
    const uint32_t* __restrict__ prompt_blocks,
    unsigned long long* __restrict__ keys,
    int ratio,
    int top_k,
    int window_blocks,
    int recent_blocks,
    int recent_forced,
    int parts,
    int n_rows
) {
    const int row = (int)blockIdx.x;
    const int seg = (int)blockIdx.y;
    if (row >= n_rows) return;
    const int tid = (int)threadIdx.x;
    Row g;
    if (!row_geometry(
            row, n_cand, qpos, tail_len, prompt_blocks, ratio, top_k, window_blocks,
            recent_blocks, recent_forced, g)) {
        return;
    }
    const int w = seg / parts;
    const int p = seg - w * parts;
    if (w >= g.n_win) return;

    __shared__ unsigned long long buf[CAP];
    __shared__ int s_count;
    __shared__ unsigned long long s_thr;
    __shared__ int s_hist[BINS];
    __shared__ int s_pick[3];
    const Survivors s = {buf, &s_count, &s_thr, s_hist, s_pick};

    const Window v = window_of(g, w, recent_forced);
    const int own = span_len(v.lo, v.r1_hi);
    const int own_lo = v.lo + (int)((long long)own * p / parts);
    const int own_hi = v.lo + (int)((long long)own * (p + 1) / parts);
    const bool shared = p == 0;
    const int pool = (own_hi - own_lo) + (shared ? shared_pool(g, v) : 0);
    const int keep = g.keep < pool ? g.keep : pool;

    unsigned long long* out = keys + ((size_t)row * gridDim.y + (size_t)seg) * MAX_KEEP;
    if (keep > 0) {
        survivors_reset(s, tid);
        const int n = offer_pool(
            s, scores + (size_t)row * (size_t)score_stride, g, v, own_lo, own_hi, shared, keep,
            tid);
        survivors_finish(s, keep, n, tid);
    }
    for (int r = tid; r < g.keep; r += THREADS) out[r] = r < keep ? buf[r] : 0ull;
}

// The split launch's second pass: one block per row ranks each window's
// slices' survivors, then gathers and writes the row's selection as the
// single-pass kernel does.
__global__ void __launch_bounds__(THREADS) qsa_topk_merge_kernel(
    const unsigned long long* __restrict__ keys,
    int segs,
    const uint32_t* __restrict__ n_cand,
    const uint32_t* __restrict__ qpos,
    const uint32_t* __restrict__ tail_len,
    const uint32_t* __restrict__ prompt_blocks,
    uint32_t* __restrict__ entries,
    int entry_stride,
    uint32_t* __restrict__ cnt,
    int ratio,
    int top_k,
    int window_blocks,
    int recent_blocks,
    int recent_forced,
    int parts,
    int n_rows
) {
    const int row = (int)blockIdx.x;
    if (row >= n_rows) return;
    const int tid = (int)threadIdx.x;
    Row g;
    if (!row_geometry(
            row, n_cand, qpos, tail_len, prompt_blocks, ratio, top_k, window_blocks,
            recent_blocks, recent_forced, g)) {
        if (tid == 0) cnt[row] = QSA_DENSE_ROW;
        return;
    }

    extern __shared__ unsigned long long smem_qsa_topk[];
    __shared__ int s_count;
    __shared__ int s_n_ent;
    __shared__ unsigned long long s_thr;
    __shared__ int s_hist[BINS];
    __shared__ int s_pick[3];
    const Smem m = smem_layout(smem_qsa_topk, &s_count, &s_thr, s_hist, s_pick);
    if (tid == 0) s_n_ent = 0;

    for (int w = 0; w < g.n_win; ++w) {
        const Window v = window_of(g, w, recent_forced);
        const int pool = span_len(v.lo, v.r1_hi) + shared_pool(g, v);
        const int keep = g.keep < pool ? g.keep : pool;
        if (keep <= 0) continue;
        survivors_reset(m.s, tid);
        // The window's slices are adjacent, each `g.keep` keys long.
        const unsigned long long* wk = keys + ((size_t)row * segs + (size_t)w * parts) * MAX_KEEP;
        const int total = parts * g.keep;
        int n = 0;
        for (int base = 0; base < total; base += THREADS) {
            const int e = base + tid;
            unsigned long long key = 0ull;
            if (e < total) {
                const int slice = e / g.keep;
                key = wk[(size_t)slice * MAX_KEEP + (e - slice * g.keep)];
            }
            survivors_offer(m.s, key, keep, n, tid);
        }
        survivors_finish(m.s, keep, n, tid);
        emit_window(m.s, g, keep, ratio, m.ent, &s_n_ent, tid);
    }
    __syncthreads();
    finish_row(
        g, ratio, m.ent, s_n_ent, (int*)m.s.buf,
        entries + (size_t)row * (size_t)entry_stride, cnt + row, tid);
}

constexpr int MAX_DEVICES = 64;

// The current device's SM count — a split launch aims for two blocks per SM.
// Cached per device: the attribute never changes, and any host thread may ask.
int sm_count() {
    static std::atomic<int> cached[MAX_DEVICES];
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= MAX_DEVICES) return SPLIT_BLOCKS;
    int sms = cached[dev].load(std::memory_order_relaxed);
    if (sms == 0) {
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        if (sms <= 0) sms = 1;
        cached[dev].store(sms, std::memory_order_relaxed);
    }
    return sms;
}

// Opt `kernel` in to `smem` bytes of dynamic shared memory on the current
// device, which a kernel must do past the default 48 KiB. The largest size
// already granted is remembered per device, so a launch pays the driver call
// only when it needs more than any launch before it.
template <typename Kernel>
void opt_in_smem(Kernel kernel, std::atomic<int>* granted, size_t smem) {
    if (smem <= 48u * 1024u) return;
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= MAX_DEVICES) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        return;
    }
    if (granted[dev].load(std::memory_order_relaxed) >= (int)smem) return;
    if (cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem)
        == cudaSuccess) {
        granted[dev].store((int)smem, std::memory_order_relaxed);
    }
}

std::atomic<int> entries_granted[MAX_DEVICES];
std::atomic<int> merge_granted[MAX_DEVICES];

} // namespace qsa_topk

// How many slices each window of a launch is split into, or `0` to run the
// single pass. Only a launch too narrow to fill the device splits: its windows
// then run side by side, one block each at least, and each window splits
// further only as far as keeps the two passes balanced — a slice ranks
// `win / parts` blocks and the merge `parts · keep` keys, which meet near
// `parts = √(win / keep)`.
extern "C" int32_t qsa_topk_split_parts(
    int32_t n_rows,
    int32_t cand_max,
    int32_t window_blocks,
    int32_t top_k,
    int32_t ratio
) {
    if (n_rows <= 0 || cand_max <= 0 || ratio <= 0) return 0;
    const int target = 2 * qsa_topk::sm_count() < qsa_topk::SPLIT_BLOCKS
        ? 2 * qsa_topk::sm_count()
        : qsa_topk::SPLIT_BLOCKS;
    if (n_rows * 2 > target) return 0;
    const int win = (window_blocks > 0 && window_blocks < cand_max) ? window_blocks : cand_max;
    const long long launched = (long long)n_rows * ((cand_max + win - 1) / win);
    if (launched > qsa_topk::SPLIT_BLOCKS) return 0;
    const int keep = (top_k + ratio - 2) / ratio + 1;
    const int balanced = (int)sqrtf((float)win / (float)keep);
    int parts = (int)(target / launched);
    if (parts > balanced) parts = balanced;
    if (parts < 1) parts = 1;
    // One window in one slice is the single pass with a merge behind it.
    return (launched == n_rows && parts == 1) ? 0 : parts;
}

extern "C" void run_qsa_topk_entries(
    const float* scores,
    int32_t score_stride,
    const uint32_t* n_cand,
    const uint32_t* qpos,
    const uint32_t* tail_len,
    const uint32_t* prompt_blocks,
    uint32_t* entries,
    int32_t entry_stride,
    uint32_t* cnt,
    int32_t ratio,
    int32_t top_k,
    int32_t window_blocks,
    int32_t recent_blocks,
    int32_t recent_forced,
    int32_t ent_cap,
    int32_t cand_max,
    void* split_keys,
    int32_t n_rows,
    void* stream
) {
    if (n_rows <= 0) return;
    const size_t smem = (size_t)qsa_topk::CAP * sizeof(unsigned long long)
        + (size_t)ent_cap * sizeof(uint32_t);
    // Without scratch the launch runs the single pass, whatever its width.
    const int parts = split_keys == nullptr
        ? 0
        : qsa_topk_split_parts(n_rows, cand_max, window_blocks, top_k, ratio);
    const cudaStream_t s = (cudaStream_t)stream;
    if (parts == 0) {
        qsa_topk::opt_in_smem(qsa_topk::qsa_topk_entries_kernel, qsa_topk::entries_granted, smem);
        qsa_topk::qsa_topk_entries_kernel<<<(unsigned)n_rows, qsa_topk::THREADS, smem, s>>>(
            scores, score_stride, n_cand, qpos, tail_len, prompt_blocks, entries, entry_stride,
            cnt, ratio, top_k, window_blocks, recent_blocks, recent_forced, n_rows);
        return;
    }
    const int win = (window_blocks > 0 && window_blocks < cand_max) ? window_blocks : cand_max;
    const int segs = ((cand_max + win - 1) / win) * parts;
    unsigned long long* keys = (unsigned long long*)split_keys;
    qsa_topk::qsa_topk_segment_kernel<<<dim3((unsigned)n_rows, (unsigned)segs),
                                       qsa_topk::THREADS, 0, s>>>(
        scores, score_stride, n_cand, qpos, tail_len, prompt_blocks, keys, ratio, top_k,
        window_blocks, recent_blocks, recent_forced, parts, n_rows);
    qsa_topk::opt_in_smem(qsa_topk::qsa_topk_merge_kernel, qsa_topk::merge_granted, smem);
    qsa_topk::qsa_topk_merge_kernel<<<(unsigned)n_rows, qsa_topk::THREADS, smem, s>>>(
        keys, segs, n_cand, qpos, tail_len, prompt_blocks, entries, entry_stride, cnt, ratio,
        top_k, window_blocks, recent_blocks, recent_forced, parts, n_rows);
}
