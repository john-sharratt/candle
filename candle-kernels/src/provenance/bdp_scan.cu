// Batched Binary Directional Provenance (BDP) flat scan.
//
// The CPU reference is `candle_conversation::provenance::score_packed`: for each
// (query token, layer-group) it scans the whole gallery computing per-case best
// sign-agreement (popcount of XNOR over the group's words) plus the agreement
// sum/sumsq, then emits one `z * margin` vote for the leading case. The needle
// gate + per-case tally run on the host, identical to the CPU path — so this
// kernel only produces the per-(query-token, group) `(leading_case, vote)` pairs.
//
// This design is the empirically-fastest of several tried (thread-per-query-token
// with a shared gallery and scalar accumulators, and a broadcast-read variant,
// both measured ~3x slower): a **query-token tile per block** with the tile loop
// **fully unrolled**. The unroll is what makes it fast — it issues `BDP_TQ`
// independent popcount chains per gallery token (high ILP), keeping the popcount
// pipelines saturated even at the low (~12%) occupancy the resulting 200-register
// footprint implies. That is the Volkov "high-ILP, low-occupancy" regime; forcing
// higher occupancy (register caps / less unroll) measured *slower*.
//
// Layout tricks:
//  * **Group-major** gallery `[group][token][gw]` → consecutive threads read
//    consecutive words (coalesced), not 192-byte-strided.
//  * Vectorized `ulonglong4` gallery loads (the locked 8-word group = two loads).
//  * The full unroll keeps the `lsum`/`lsumsq` accumulators register-resident
//    (constant-indexed); a partial unroll would push them to local memory.
//
// Batching: `probe_words` concatenates every request's probe tokens, one launch
// per wave. Integer reductions (u32 max, u64 sums) are exact/order-independent;
// the final float math is fast-math f32 (~ULP off the CPU — irrelevant for a
// ranking; validated approximately + identical argmax).

#include <cstdint>
#include <cuda_runtime.h>

// Locked folded-signature group width (4 heads x 128 bits = 8 u64).
#define BDP_MAX_GW 8
// Query tokens per block tile — the gallery-reuse / ILP factor.
#define BDP_TQ 8

extern "C" __global__ void bdp_scan_kernel(
    const unsigned long long *__restrict__ gallery_words, // GROUP-major: [g][token][gw]
    const unsigned int *__restrict__ gallery_case,        // n_tokens
    const unsigned long long *__restrict__ probe_words,   // token-major: n_probe_tokens * wpt
    int n_tokens,
    int n_probe_tokens,
    int n_groups,
    int gw,  // words per layer-group (<= BDP_MAX_GW)
    int wpt, // words per token (n_groups * gw)
    int n_cases,
    int *__restrict__ out_case,   // n_probe_tokens * n_groups
    float *__restrict__ out_vote) // n_probe_tokens * n_groups
{
    const int tile = blockIdx.x / n_groups;
    const int g = blockIdx.x % n_groups;
    const int q0 = tile * BDP_TQ;
    if (q0 >= n_probe_tokens) {
        return;
    }
    int tq = n_probe_tokens - q0;
    if (tq > BDP_TQ) {
        tq = BDP_TQ;
    }

    // Dynamic shared: per-query-token per-case best agreement — [BDP_TQ][n_cases].
    extern __shared__ unsigned int s_case_max[];
    __shared__ unsigned long long s_sum[BDP_TQ];
    __shared__ unsigned long long s_sumsq[BDP_TQ];
    __shared__ unsigned long long qg[BDP_TQ * BDP_MAX_GW];

    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    for (int i = tid; i < tq * n_cases; i += nthreads) {
        s_case_max[i] = 0u;
    }
    for (int i = tid; i < tq; i += nthreads) {
        s_sum[i] = 0ull;
        s_sumsq[i] = 0ull;
    }
    // The tile's query-token group words (probe stays token-major — tiny).
    for (int idx = tid; idx < tq * gw; idx += nthreads) {
        const int i = idx / gw;
        const int k = idx % gw;
        qg[i * gw + k] = probe_words[(size_t)(q0 + i) * wpt + (size_t)g * gw + k];
    }
    __syncthreads();

    const unsigned long long *g_base = gallery_words + (size_t)g * n_tokens * gw;
    unsigned long long lsum[BDP_TQ];
    unsigned long long lsumsq[BDP_TQ];
    for (int i = 0; i < tq; i++) {
        lsum[i] = 0ull;
        lsumsq[i] = 0ull;
    }

    for (int j = tid; j < n_tokens; j += nthreads) {
        // Load the gallery token's group words ONCE, reuse across the tile. The
        // locked 8-word group is two vectorized 32-byte (ulonglong4) loads.
        unsigned long long tw[BDP_MAX_GW];
        const unsigned long long *tok = g_base + (size_t)j * gw;
        if (gw == BDP_MAX_GW) {
            const ulonglong4 *t4 = reinterpret_cast<const ulonglong4 *>(tok);
            const ulonglong4 a = t4[0];
            const ulonglong4 b = t4[1];
            tw[0] = a.x; tw[1] = a.y; tw[2] = a.z; tw[3] = a.w;
            tw[4] = b.x; tw[5] = b.y; tw[6] = b.z; tw[7] = b.w;
        } else {
#pragma unroll
            for (int k = 0; k < BDP_MAX_GW; k++) {
                if (k < gw) {
                    tw[k] = tok[k];
                }
            }
        }
        const unsigned int c = gallery_case[j];
        // Full unroll over the compile-time tile: keeps `lsum`/`lsumsq`
        // register-resident and issues BDP_TQ independent popcount chains (ILP).
#pragma unroll
        for (int i = 0; i < BDP_TQ; i++) {
            if (i >= tq) {
                continue;
            }
            const unsigned long long *qi = qg + (size_t)i * gw;
            unsigned int ag = 0u;
#pragma unroll
            for (int k = 0; k < BDP_MAX_GW; k++) {
                if (k < gw) {
                    ag += __popcll(~(qi[k] ^ tw[k]));
                }
            }
            atomicMax(&s_case_max[(size_t)i * n_cases + c], ag);
            lsum[i] += ag;
            lsumsq[i] += (unsigned long long)ag * (unsigned long long)ag;
        }
    }
    for (int i = 0; i < tq; i++) {
        atomicAdd(&s_sum[i], lsum[i]);
        atomicAdd(&s_sumsq[i], lsumsq[i]);
    }
    __syncthreads();

    // One thread per query token in the tile: leader/runner-up → z*margin vote.
    for (int i = tid; i < tq; i += nthreads) {
        const unsigned int *cmax = s_case_max + (size_t)i * n_cases;
        unsigned int top1 = 0u, top2 = 0u;
        int top1c = -1;
        for (int c = 0; c < n_cases; c++) {
            const unsigned int m = cmax[c];
            if (m > top1) {
                top2 = top1;
                top1 = m;
                top1c = c;
            } else if (m > top2) {
                top2 = m;
            }
        }
        const int out_idx = (q0 + i) * n_groups + g;
        if (top1c < 0) {
            out_case[out_idx] = -1;
            out_vote[out_idx] = 0.0f;
        } else {
            const float n_gal = (float)n_tokens;
            const float mean = (float)s_sum[i] / n_gal;
            float var = (float)s_sumsq[i] / n_gal - mean * mean;
            if (var < 1e-6f) {
                var = 1e-6f;
            }
            float z = ((float)top1 - mean) / sqrtf(var);
            if (z < 0.0f) {
                z = 0.0f;
            }
            const float margin = (float)(top1 - top2);
            out_case[out_idx] = top1c;
            out_vote[out_idx] = z * margin;
        }
    }
}

// Host launcher: one block per (query-token tile, group), `threads` per block,
// dynamic shared = BDP_TQ * n_cases * sizeof(u32). Launched on the caller's stream.
extern "C" void run_batched_bdp_scan(
    const unsigned long long *gallery_words,
    const unsigned int *gallery_case,
    const unsigned long long *probe_words,
    int n_tokens,
    int n_probe_tokens,
    int n_groups,
    int gw,
    int wpt,
    int n_cases,
    int *out_case,
    float *out_vote,
    void *stream)
{
    if (n_probe_tokens <= 0 || n_tokens <= 0 || n_groups <= 0) {
        return;
    }
    const int n_tiles = (n_probe_tokens + BDP_TQ - 1) / BDP_TQ;
    const int blocks = n_tiles * n_groups;
    const int threads = 128;
    const size_t shmem = (size_t)BDP_TQ * n_cases * sizeof(unsigned int);
    cudaStream_t s = (cudaStream_t)stream;
    bdp_scan_kernel<<<blocks, threads, shmem, s>>>(
        gallery_words, gallery_case, probe_words,
        n_tokens, n_probe_tokens, n_groups, gw, wpt, n_cases,
        out_case, out_vote);
}
