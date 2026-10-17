#pragma once
// Gated DeltaNet — the fold: apply the first `m` rows of a span's timeline to
// its state and conv tail, in place (docs/deltanet_timeline.md §6.3).
//
// A short span writes its contribution to the state as timeline rows
// (delta_net_timeline.cuh) and leaves S and the tail as it found them. Once
// the step knows how many of the span's rows it keeps — every row for a span
// nobody is verifying, the accepted prefix of a speculative block — the fold
// lands them:
//
//   S[i][j] ← S[i][j]·e^{G[m−1]} + Σ_{t<m} v_new[t][i] · (k[t][j]·e^{G[m−1]−G[t]})
//   tail    ← the last K−1 of [tail | x[0] .. x[m−1]]
//
// which is the three-kernel state pass's phase 3 over a span of those m rows,
// operation for operation: the decayed key one rounded multiply, the rank-1
// sum one rounded FMA per row with t ascending, the store one rounded FMA
// (dn_decayed_key, dn_rank1_acc, dn_state_update — the helpers that pass and
// the decode step use). Every row quantity is prefix-stable, so this is the
// state the scan writes for those m rows, bit for bit. Which thread computes
// an element does not enter its arithmetic, so the thread map is the fold's
// own.
//
// One launch for every (entry, V head, d_v tile): an entry is one sequence's
// layer, and a step folds every one it has in a single grid. The table is one
// upload:
//   table [4, n] i64 — conv tail, state, timeline (pointers), then the rows
//                      to fold, 1..=DNT_ROWS
//
// S: the fold reads and writes every element once and does ~2m FMAs on it, so
// it runs at the memory system's rate and is built for bytes in flight. A
// lane owns four consecutive columns and a warp four consecutive rows of the
// block's 32-row tile: every S, k and v access is one 128-bit load, a warp's
// S row a contiguous 512 B. The tile is loaded before the timeline is read,
// so its latency is paid behind the row loop rather than after it.
//
// Tail: every block of an entry shares its channels, a whole channel per
// thread. The fold writes the tail in place, and for m < K−1 the new element
// jj is the old element jj + m — so a thread reads its channel's whole old
// tail into registers before it writes any of it, and picks each new element
// with a select over static indices: an array indexed by `m` would live in
// local memory. A per-element spread (the short-span kernel's tail writer,
// which writes a separate buffer) would race.

#include "delta_net_prefill_kernel.cuh"
#include "delta_net_timeline.cuh"

#define DNF_THREADS 256
// Rows of the tile a warp owns, and columns a lane owns.
#define DNF_ROWS 4
#define DNF_COLS 4

namespace delta_net {

static_assert(DNF_THREADS / 32 * DNF_ROWS == DNP_TV, "eight warps of four rows tile the 32-row tile");
static_assert(32 * DNF_COLS == DNP_DIM, "a warp's lanes cover the 128 columns");

static __global__ void __launch_bounds__(DNF_THREADS) delta_net_fold_f32_kernel(
        const long long* __restrict__ table, // [4, n]
        int n,
        int n_v_heads,
        int n_k_heads,
        int channels,
        int kwidth) {
    const int z = (int)blockIdx.z;
    float* __restrict__ tail = reinterpret_cast<float*>(table[z]);
    float* __restrict__ state = reinterpret_cast<float*>(table[n + z]);
    const DnTimeline tl =
        dn_timeline(reinterpret_cast<float*>(table[2 * n + z]), n_v_heads, n_k_heads, channels);
    const long long m_rows = table[3 * n + z];
    // The launcher admits the table by its rows; a count outside 1..=DNT_ROWS
    // is a table that disagrees with what was written, and stops the kernel
    // rather than reading rows nobody wrote.
    if (m_rows < 1 || m_rows > DNT_ROWS) __trap();
    const int m = (int)m_rows;

    const int h = blockIdx.x;
    const int kh = h % n_k_heads;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int row0 = (int)blockIdx.y * DNP_TV + (tid >> 5) * DNF_ROWS;
    const int col0 = lane * DNF_COLS;

    // ---- S ----------------------------------------------------------------
    // The tile first: four 128-bit loads in flight before anything waits.
    float4* __restrict__ s4 = reinterpret_cast<float4*>(state + ((size_t)h * DNP_DIM + row0) * DNP_DIM + col0);
    float4 s[DNF_ROWS];
    #pragma unroll
    for (int r = 0; r < DNF_ROWS; ++r) s[r] = s4[r * (DNP_DIM / 4)];

    const float gl = tl.g[(size_t)(m - 1) * n_v_heads + h];
    const float dec = expf(gl);
    float4 acc[DNF_ROWS];
    #pragma unroll
    for (int r = 0; r < DNF_ROWS; ++r) acc[r] = make_float4(0.f, 0.f, 0.f, 0.f);
    for (int t = 0; t < m; ++t) {
        const float e = expf(gl - tl.g[(size_t)t * n_v_heads + h]); // ≤ 1: G decreases
        const float4 k = *reinterpret_cast<const float4*>(
            tl.k + ((size_t)t * n_k_heads + kh) * DNP_DIM + col0);
        const float4 v = *reinterpret_cast<const float4*>(
            tl.v + ((size_t)t * n_v_heads + h) * DNP_DIM + row0);
        float4 kg;
        kg.x = dn_decayed_key(k.x, e);
        kg.y = dn_decayed_key(k.y, e);
        kg.z = dn_decayed_key(k.z, e);
        kg.w = dn_decayed_key(k.w, e);
        const float vr[DNF_ROWS] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int r = 0; r < DNF_ROWS; ++r) {
            acc[r].x = dn_rank1_acc(vr[r], kg.x, acc[r].x);
            acc[r].y = dn_rank1_acc(vr[r], kg.y, acc[r].y);
            acc[r].z = dn_rank1_acc(vr[r], kg.z, acc[r].z);
            acc[r].w = dn_rank1_acc(vr[r], kg.w, acc[r].w);
        }
    }
    #pragma unroll
    for (int r = 0; r < DNF_ROWS; ++r) {
        float4 o;
        o.x = dn_state_update(s[r].x, dec, acc[r].x);
        o.y = dn_state_update(s[r].y, dec, acc[r].y);
        o.z = dn_state_update(s[r].z, dec, acc[r].z);
        o.w = dn_state_update(s[r].w, dec, acc[r].w);
        s4[r * (DNP_DIM / 4)] = o;
    }

    // ---- the conv tail ------------------------------------------------------
    const int tcols = kwidth - 1;
    const int nblk = (int)(gridDim.x * gridDim.y);
    const int b = (int)(blockIdx.y * gridDim.x + blockIdx.x);
    for (int c = b * DNF_THREADS + tid; c < channels; c += nblk * DNF_THREADS) {
        float* trow = tail + (size_t)c * tcols;
        float old[DNC_KMAX - 1];
        #pragma unroll
        for (int q = 0; q < DNC_KMAX - 1; ++q) old[q] = q < tcols ? trow[q] : 0.f;
        #pragma unroll
        for (int jj = 0; jj < DNC_KMAX - 1; ++jj) {
            if (jj < tcols) {
                // Element jj of [tail | x[0..m)]'s last K−1: x row m − K + 1 + jj
                // when that is a row, else the old tail's element jj + m.
                const int idx = m - tcols + jj;
                float nv;
                if (idx >= 0) {
                    nv = tl.x[(size_t)idx * channels + c];
                } else {
                    nv = 0.f;
                    #pragma unroll
                    for (int q = 0; q < DNC_KMAX - 1; ++q) {
                        if (q == jj + m) nv = old[q];
                    }
                }
                trow[jj] = nv;
            }
        }
    }
}

// 0 when launched, 1 when the shape is refused (nothing written): no entries,
// more than the grid's z extent, a geometry the update's channel map does not
// cover, or a conv window past the tail's register budget.
static inline int launch_fold_f32(
        const long long* table,
        int n,
        int n_v_heads,
        int n_k_heads,
        int channels,
        int kwidth,
        cudaStream_t stream) {
    if (n <= 0 || n > 65535) return 1;
    if (n_v_heads <= 0 || n_k_heads <= 0 || n_v_heads % n_k_heads != 0) return 1;
    if (channels != (2 * n_k_heads + n_v_heads) * DNP_DIM) return 1;
    if (kwidth < 2 || kwidth > DNC_KMAX) return 1;
    dim3 grid(n_v_heads, DNP_DIM / DNP_TV, n);
    delta_net_fold_f32_kernel<<<grid, DNF_THREADS, 0, stream>>>(
        table, n, n_v_heads, n_k_heads, channels, kwidth);
    return 0;
}

} // namespace delta_net
