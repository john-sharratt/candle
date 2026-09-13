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
// ties to resolve by any other rule, and the selection is reproducible.
//
// THE SELECTION, WITHOUT MATERIALIZING A SORT
// -------------------------------------------
// A depth-L sequence has ~L/ratio candidate blocks and we need the best ~512
// of them, so sorting the row would be the wrong shape by three orders of
// magnitude. Instead each block streams its candidates through a shared
// buffer against a running threshold:
//
//   - The buffer's TOP `keep` slots hold the survivors, sorted ascending; the
//     bottom is free space whose stale keys are all below the threshold and
//     therefore can never re-enter the top (their blocks were already beaten,
//     and no two blocks share a key).
//   - A chunk appends only keys above the threshold. When the free space drops
//     below one chunk's worth, one bitonic sort re-establishes the invariant
//     and raises the threshold. After the first few thousand blocks the
//     threshold is high enough that sorts become rare.
//
// `keep ≤ CAP − THREADS` is what bounds the append (a chunk adds at most
// THREADS keys and is checked immediately after), and the host refuses a
// `top_k`/`ratio` pair that would break it rather than silently truncating.
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "../qsa_select.cuh"

namespace qsa_topk {

constexpr int THREADS = 256;
constexpr int CAP = 1024;          // shared candidate buffer, u64 keys
constexpr int ENT_CAP = 1024;      // shared entry buffer for the ascending sort
// The append bound: a chunk adds at most THREADS keys into the free space
// below the survivors, and the trim below fires as soon as the free space
// could not absorb another chunk.
constexpr int MAX_KEEP = CAP - THREADS;

__device__ __forceinline__ uint32_t key_block(unsigned long long key) {
    return 0xFFFFFFFFu - (uint32_t)(key & 0xFFFFFFFFull);
}

// Bitonic sort of `n` (a power of two) shared values, ascending.
template <typename T>
__device__ void bitonic_ascending(T* buf, int n, int tid, int nthreads) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = tid; i < n; i += nthreads) {
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

// One block per query row.
//
//   scores  [n_rows, score_stride] f32 — row r's block scores, valid on
//                                        [0, n_cand[r]); the rest is ignored
//   n_cand  [n_rows] u32 — candidate blocks (the query's tail_start / ratio)
//   qpos    [n_rows] u32 — the query's ABSOLUTE position
//   entries [n_rows, entry_stride] u32 — output, ascending by block
//   cnt     [n_rows] u32 — entries written, or QSA_DENSE_ROW
__global__ void __launch_bounds__(THREADS) qsa_topk_entries_kernel(
    const float* __restrict__ scores,
    int score_stride,
    const uint32_t* __restrict__ n_cand,
    const uint32_t* __restrict__ qpos,
    // [n_rows] — cells of its OWN block the query has, `1..=ratio`. Not derived
    // here: a block's width is a property of the page it belongs to.
    const uint32_t* __restrict__ tail_len,
    uint32_t* __restrict__ entries,
    int entry_stride,
    uint32_t* __restrict__ cnt,
    int ratio,
    int top_k,
    int n_rows
) {
    const int row = (int)blockIdx.x;
    if (row >= n_rows) return;
    const int tid = (int)threadIdx.x;

    const int width = top_k + ratio - 1;
    const int visible = (int)qpos[row] + 1;
    if (visible <= width) {
        // Every visible cell is attended — the identity, and the reason a
        // shallow context needs no indexer at all.
        if (tid == 0) cnt[row] = QSA_DENSE_ROW;
        return;
    }

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
    const int n_tail = (int)tail_len[row];
    const int budget = width - n_tail;
    const int cand = (int)n_cand[row];
    const int full = budget / ratio;
    const int rem = budget - full * ratio;
    int keep = full + (rem > 0 ? 1 : 0);
    if (keep > cand) keep = cand;
    // The append bound the buffer arithmetic rests on. The host refuses a
    // `top_k`/`ratio` pair that exceeds it (`qsa_topk::MAX_KEEP` is mirrored
    // there), so this clamp cannot fire; it is here so that a precondition
    // broken upstream degrades the selection instead of writing past `buf`.
    if (keep > MAX_KEEP) keep = MAX_KEEP;

    __shared__ unsigned long long buf[CAP];
    __shared__ uint32_t ent[ENT_CAP];
    __shared__ int s_count;
    __shared__ unsigned long long s_thr;

    for (int i = tid; i < CAP; i += THREADS) buf[i] = 0ull;
    if (tid == 0) {
        s_count = 0;
        s_thr = 0ull;
    }
    __syncthreads();

    const float* srow = scores + (size_t)row * (size_t)score_stride;
    const int free_slots = CAP - keep;

    for (int base = 0; base < cand; base += THREADS) {
        const int b = base + tid;
        if (b < cand) {
            const uint32_t bits = __float_as_uint(srow[b]);
            const unsigned long long key =
                ((unsigned long long)bits << 32) | (unsigned long long)(0xFFFFFFFFu - (uint32_t)b);
            if (key > s_thr) {
                const int slot = atomicAdd(&s_count, 1);
                buf[slot] = key;
            }
        }
        __syncthreads();
        if (s_count > free_slots - THREADS) {
            bitonic_ascending(buf, CAP, tid, THREADS);
            if (tid == 0) {
                s_thr = buf[CAP - keep];
                s_count = 0;
            }
            __syncthreads();
        }
    }
    bitonic_ascending(buf, CAP, tid, THREADS);

    // Rank r (0 = best) is buf[CAP − 1 − r]. Whole blocks up to the budget,
    // then the block the budget runs out inside contributes its lowest cells,
    // then the query's own tail.
    int n_ent = 0;
    for (int r = tid; r < ENT_CAP; r += THREADS) ent[r] = 0xFFFFFFFFu;
    __syncthreads();
    const int n_whole = full < keep ? full : keep;
    for (int r = tid; r < n_whole; r += THREADS) {
        ent[r] = ((key_block(buf[CAP - 1 - r]) << 2) | (uint32_t)(ratio - 1));
    }
    n_ent = n_whole;
    if (rem > 0 && keep > full) {
        if (tid == 0) {
            ent[n_whole] = (key_block(buf[CAP - 1 - full]) << 2) | (uint32_t)(rem - 1);
        }
        n_ent += 1;
    }
    if (n_tail > 0) {
        if (tid == 0) {
            ent[n_ent] = ((uint32_t)cand << 2) | (uint32_t)(n_tail - 1);
        }
        n_ent += 1;
    }
    __syncthreads();
    bitonic_ascending(ent, ENT_CAP, tid, THREADS);
    uint32_t* out = entries + (size_t)row * (size_t)entry_stride;
    for (int i = tid; i < n_ent; i += THREADS) out[i] = ent[i];
    if (tid == 0) cnt[row] = (uint32_t)n_ent;
}

} // namespace qsa_topk

extern "C" void run_qsa_topk_entries(
    const float* scores,
    int32_t score_stride,
    const uint32_t* n_cand,
    const uint32_t* qpos,
    const uint32_t* tail_len,
    uint32_t* entries,
    int32_t entry_stride,
    uint32_t* cnt,
    int32_t ratio,
    int32_t top_k,
    int32_t n_rows,
    void* stream
) {
    if (n_rows <= 0) return;
    qsa_topk::qsa_topk_entries_kernel<<<(unsigned)n_rows, qsa_topk::THREADS, 0,
                                       (cudaStream_t)stream>>>(
        scores, score_stride, n_cand, qpos, tail_len, entries, entry_stride, cnt, ratio, top_k,
        n_rows);
}
