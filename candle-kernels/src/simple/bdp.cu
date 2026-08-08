// =============================================================================
// bdp.cu — sign-pack + BDP recall agreement for the compressed-corpus
// two-stage selection (BDP recall in the Indexer's sign space → Indexer
// float precision on the shortlist).
//
// Both kernels are tiny, launch-bound utilities that keep the corpus-selection
// recall stage fully on-device (the decode hot path allows exactly one host
// readback — the sampler).
// =============================================================================

#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Pack the sign bits of `n` rows of `dim` f32 values into ceil(dim/32) u32
// words per row (bit d of word w = sign(x[32w+d]) >= 0). One thread per row
// word: grid-stride over n * words.
extern "C" __global__ void sign_pack_kernel(
    const float* __restrict__ x, // [n, dim]
    uint32_t* __restrict__ out,  // [n, words]
    int n,
    int dim
) {
    int words = (dim + 31) / 32;
    int total = n * words;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += gridDim.x * blockDim.x) {
        int row = i / words;
        int w = i % words;
        uint32_t bits = 0;
        int base = w * 32;
        int lim = dim - base;
        if (lim > 32) lim = 32;
        const float* xr = x + (int64_t)row * dim + base;
        for (int b = 0; b < lim; ++b) {
            if (xr[b] >= 0.f) bits |= (1u << b);
        }
        out[(int64_t)row * words + w] = bits;
    }
}

// BDP recall agreement: counts[g] = Σ_h Σ_w popcount(~(q_signs[h][w] ^
// signs[g][w])) — total sign agreement between every query head and entry g.
// One thread per entry.
extern "C" __global__ void bdp_recall_kernel(
    const uint32_t* __restrict__ q_signs, // [n_heads, words]
    const uint32_t* __restrict__ signs,   // [g, words]
    uint32_t* __restrict__ counts,        // [g]
    int n_heads,
    int g,
    int words,
    int dim // true bit width (tail bits above dim never disagree-count)
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= g) return;
    const uint32_t* sg = signs + (int64_t)idx * words;
    uint32_t total = 0;
    int tail_pad = words * 32 - dim; // invalid high bits in the last word
    for (int h = 0; h < n_heads; ++h) {
        const uint32_t* qh = q_signs + (int64_t)h * words;
        for (int w = 0; w < words; ++w) {
            uint32_t agree = ~(qh[w] ^ sg[w]);
            if (w == words - 1 && tail_pad > 0) {
                agree &= 0xFFFFFFFFu >> tail_pad;
            }
            total += __popc(agree);
        }
    }
    counts[idx] = total;
}

// ─── Exact device top-M over bounded u32 keys ───────────────────────────────
// The recall shortlist must be selected over an UNBOUNDED entry count (the
// whole corpus), where a single-block bitonic argsort cannot go. Agreement
// counts are bounded (≤ n_heads · dim), so an exact histogram-threshold
// select works in three stream-ordered launches with no host round-trip:
// histogram → suffix-scan for the threshold → compact (ties fill arbitrary —
// any M-superset is a valid recall shortlist; the float rescore re-ranks it).

extern "C" __global__ void topm_hist_kernel(
    const uint32_t* __restrict__ counts, uint32_t* __restrict__ hist, int g, int bins
) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < g;
         i += gridDim.x * blockDim.x) {
        uint32_t c = counts[i];
        if ((int)c >= bins) c = bins - 1;
        atomicAdd(&hist[c], 1u);
    }
}

// meta[0] = threshold bin, meta[1] = #entries strictly above threshold,
// meta[2] = tie cursor (zeroed). Single block; bins ≤ a few thousand.
extern "C" __global__ void topm_threshold_kernel(
    const uint32_t* __restrict__ hist, uint32_t* __restrict__ meta, int bins, int m
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t cum = 0;
    for (int b = bins - 1; b >= 0; --b) {
        uint32_t h = hist[b];
        if (cum + h >= (uint32_t)m || b == 0) {
            meta[0] = (uint32_t)b;
            meta[1] = cum; // strictly above the threshold bin
            return;
        }
        cum += h;
    }
}

extern "C" __global__ void topm_compact_kernel(
    const uint32_t* __restrict__ counts,
    uint32_t* __restrict__ meta,          // [thr, n_above, tie_cursor, above_cursor]
    uint32_t* __restrict__ out_ids,       // [m]
    int g,
    int m
) {
    uint32_t thr = meta[0];
    uint32_t n_above = meta[1];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < g;
         i += gridDim.x * blockDim.x) {
        uint32_t c = counts[i];
        if (c > thr) {
            uint32_t slot = atomicAdd(&meta[3], 1u);
            if ((int)slot < m) out_ids[slot] = (uint32_t)i;
        } else if (c == thr) {
            uint32_t t = atomicAdd(&meta[2], 1u);
            uint32_t slot = n_above + t;
            if ((int)slot < m) out_ids[slot] = (uint32_t)i;
        }
    }
}

extern "C" int32_t run_topm_select(
    const uint32_t* counts,
    uint32_t* hist,    // [bins] scratch (zeroed here)
    uint32_t* meta,    // [4] scratch (zeroed here)
    uint32_t* out_ids, // [m]
    int32_t g,
    int32_t m,
    int32_t bins,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (g <= 0 || m <= 0) return 0;
    cudaMemsetAsync(hist, 0, (size_t)bins * sizeof(uint32_t), stream);
    cudaMemsetAsync(meta, 0, 4 * sizeof(uint32_t), stream);
    int threads = 256;
    int blocks = (g + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;
    topm_hist_kernel<<<blocks, threads, 0, stream>>>(counts, hist, g, bins);
    topm_threshold_kernel<<<1, 32, 0, stream>>>(hist, meta, bins, m);
    topm_compact_kernel<<<blocks, threads, 0, stream>>>(counts, meta, out_ids, g, m);
    return (int32_t)cudaGetLastError();
}

extern "C" int32_t run_sign_pack(
    const float* x, uint32_t* out, int32_t n, int32_t dim, void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    int words = (dim + 31) / 32;
    int total = n * words;
    if (total <= 0) return 0;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    if (blocks > 4096) blocks = 4096;
    sign_pack_kernel<<<blocks, threads, 0, stream>>>(x, out, n, dim);
    return (int32_t)cudaGetLastError();
}

extern "C" int32_t run_bdp_recall(
    const uint32_t* q_signs,
    const uint32_t* signs,
    uint32_t* counts,
    int32_t n_heads,
    int32_t g,
    int32_t words,
    int32_t dim,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (g <= 0) return 0;
    int threads = 256;
    int blocks = (g + threads - 1) / threads;
    bdp_recall_kernel<<<blocks, threads, 0, stream>>>(
        q_signs, signs, counts, n_heads, g, words, dim);
    return (int32_t)cudaGetLastError();
}
