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

// ─── Batched selection across concurrent decode sessions ────────────────────
// Decode runs the two-stage selection ONCE PER SESSION per CSA layer (≤64
// sessions/wave), each over its own gallery. The per-session launch loop is
// dominated by `topm_select` — three tiny launches plus a single-warp serial
// scan over `bins` histogram bins, all repeated per session. These batched
// kernels fold the whole wave into ONE launch per stage (session = a grid
// dimension), with byte-identical per-session math. Session `s` owns the
// contiguous ranges `counts[off[s] .. off[s]+cnt[s])` (its bdp counts),
// `signs[off[s] ..)` (its packed entries) and writes `out_ids[s*max_m ..)`
// (session-relative ids 0..cnt[s]); its histogram/meta scratch are
// `hist[s*bins ..)` / `meta[s*4 ..)`.

// Batched BDP recall: one thread per (session, entry). The session's query
// heads are staged in shared memory once per block (n_heads·words u32 ≤ 1 KB).
extern "C" __global__ void bdp_recall_batched_kernel(
    const uint32_t* __restrict__ q_signs,  // [n_sess*n_heads, words]
    const uint64_t* __restrict__ sign_ptrs, // [n_sess] base ptr to session s's [cnt[s], words]
    const uint32_t* __restrict__ off,      // [n_sess] OUTPUT base per session (into counts)
    const uint32_t* __restrict__ cnt,      // [n_sess] entry count per session
    uint32_t* __restrict__ counts,         // [total_g]
    int n_heads,
    int words,
    int dim
) {
    extern __shared__ uint32_t qh_shared[]; // [n_heads*words], 16B-aligned (dyn smem)
    int s = blockIdx.y;
    int g = (int)cnt[s];
    int base = (int)off[s]; // output offset into the concatenated `counts`
    // Each session's packed signs are read IN PLACE from its own resident buffer
    // (no O(Σlen·words) concatenation that grows with context depth); the entry index
    // is session-relative (`e`), while `counts` stays concatenated at `base + e`.
    const uint32_t* signs = (const uint32_t*)sign_ptrs[s];
    const uint32_t* qbase = q_signs + (int64_t)s * n_heads * words;
    // Stage this session's query heads once (coalesced over the block).
    for (int i = threadIdx.x; i < n_heads * words; i += blockDim.x) {
        qh_shared[i] = qbase[i];
    }
    __syncthreads();
    int tail_pad = words * 32 - dim;
    uint32_t last_mask = tail_pad > 0 ? (0xFFFFFFFFu >> tail_pad) : 0xFFFFFFFFu;

    // Fast path for the single-latent Indexer width (`index_head_dim = 128` ⇒
    // `words = 4`): read the query head and the entry signs as 128-bit `uint4`
    // (one L1/smem transaction each instead of four), and keep the entry's signs
    // in registers across the 64-head reduction. This is the L1/TEX-bound inner
    // loop, so quartering the transaction count is the win.
    if (words == 4) {
        const uint4* qh4 = reinterpret_cast<const uint4*>(qh_shared);
        const uint4* sg4 = reinterpret_cast<const uint4*>(signs);
        for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < g;
             e += gridDim.x * blockDim.x) {
            uint4 sg = sg4[e];
            uint32_t total = 0;
            for (int h = 0; h < n_heads; ++h) {
                uint4 q = qh4[h];
                total += __popc(~(q.x ^ sg.x)) + __popc(~(q.y ^ sg.y)) +
                         __popc(~(q.z ^ sg.z)) + __popc((~(q.w ^ sg.w)) & last_mask);
            }
            counts[(int64_t)base + e] = total;
        }
        return;
    }

    // Generic width: hoist the entry's signs to registers (up to 32 words),
    // removing the redundant per-head re-reads of the same row.
    for (int e = blockIdx.x * blockDim.x + threadIdx.x; e < g;
         e += gridDim.x * blockDim.x) {
        const uint32_t* sg = signs + (int64_t)e * words;
        uint32_t sgr[32];
        for (int w = 0; w < words && w < 32; ++w) {
            sgr[w] = sg[w];
        }
        uint32_t total = 0;
        for (int h = 0; h < n_heads; ++h) {
            const uint32_t* qh = qh_shared + h * words;
            for (int w = 0; w < words; ++w) {
                uint32_t agree = ~(qh[w] ^ (w < 32 ? sgr[w] : sg[w]));
                if (w == words - 1) {
                    agree &= last_mask;
                }
                total += __popc(agree);
            }
        }
        counts[(int64_t)base + e] = total;
    }
}

extern "C" __global__ void topm_hist_batched_kernel(
    const uint32_t* __restrict__ counts,
    const uint32_t* __restrict__ off,
    const uint32_t* __restrict__ cnt,
    uint32_t* __restrict__ hist, // [n_sess*bins]
    int bins
) {
    int s = blockIdx.y;
    int g = (int)cnt[s];
    const uint32_t* cs = counts + off[s];
    uint32_t* hs = hist + (int64_t)s * bins;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < g;
         i += gridDim.x * blockDim.x) {
        uint32_t c = cs[i];
        if ((int)c >= bins) c = bins - 1;
        atomicAdd(&hs[c], 1u);
    }
}

// One block per session; the WHOLE block cooperatively finds the threshold bin
// (`thr` = the highest bin b such that S(b) = Σ_{c≥b} hist[c] ≥ m, i.e. the m-th
// largest agreement count). The per-session single-warp serial scan over all
// `bins` this replaced was the dominant Stage-1 cost. Here each thread sums a
// contiguous descending CHUNK of the histogram, a block-wide inclusive scan
// locates the one chunk that straddles the m-th element, and only that chunk's
// owner serially scans its `ceil(bins/blockDim)` bins — turning an O(bins)
// serial scan into O(bins/blockDim + log blockDim).
extern "C" __global__ void topm_threshold_batched_kernel(
    const uint32_t* __restrict__ hist,
    const uint32_t* __restrict__ cnt,
    uint32_t* __restrict__ meta, // [n_sess*4] -> [thr, n_above, tie_cursor=0, above_cursor=0]
    int bins,
    int max_m
) {
    extern __shared__ uint32_t csum[]; // [blockDim.x] chunk sums (from the top)
    int s = blockIdx.x;
    int m = (int)cnt[s];
    if (max_m < m) m = max_m;
    const uint32_t* hs = hist + (int64_t)s * bins;
    uint32_t* ms = meta + (int64_t)s * 4;
    int nt = blockDim.x;
    int tid = threadIdx.x;
    int per = (bins + nt - 1) / nt;

    // Chunk `tid` covers the `tid`-th block of `per` bins counting DOWN from the
    // top bin (`bins-1`): bins [top-per+1 .. top], top = bins-1 - tid*per.
    int top = bins - 1 - tid * per;
    uint32_t local = 0;
    for (int j = 0; j < per; ++j) {
        int b = top - j;
        if (b >= 0) {
            local += hs[b];
        }
    }
    csum[tid] = local;
    __syncthreads();

    // Inclusive scan (Hillis-Steele) so csum[tid] = Σ_{c≤tid} chunk_c = count of
    // entries whose agreement lands in chunks 0..tid (the top (tid+1)·per bins).
    for (int d = 1; d < nt; d <<= 1) {
        uint32_t add = (tid >= d) ? csum[tid - d] : 0u;
        __syncthreads();
        csum[tid] += add;
        __syncthreads();
    }

    // The crossing chunk is the first tid whose inclusive sum reaches m (its
    // exclusive prefix is < m). If the total never reaches m (all entries fit in
    // the shortlist) the last chunk owns the scan and it bottoms out at b==0.
    uint32_t incl = csum[tid];
    uint32_t excl = incl - local;
    bool mine = (excl < (uint32_t)m && incl >= (uint32_t)m) ||
                (tid == nt - 1 && csum[nt - 1] < (uint32_t)m);
    if (mine) {
        uint32_t cc = excl;
        for (int j = 0; j < per; ++j) {
            int b = top - j;
            if (b < 0) {
                break;
            }
            uint32_t h = hs[b];
            if (cc + h >= (uint32_t)m || b == 0) {
                ms[0] = (uint32_t)b;
                ms[1] = cc;
                return;
            }
            cc += h;
        }
        // Only reached if this chunk didn't include b==0 and never crossed m;
        // by construction the crossing chunk always does, so this is a guard.
        ms[0] = 0;
        ms[1] = cc;
    }
}

extern "C" __global__ void topm_compact_batched_kernel(
    const uint32_t* __restrict__ counts,
    const uint32_t* __restrict__ off,
    const uint32_t* __restrict__ cnt,
    uint32_t* __restrict__ meta,    // [n_sess*4]
    uint32_t* __restrict__ out_ids, // [n_sess*max_m] session-relative
    int max_m
) {
    int s = blockIdx.y;
    int g = (int)cnt[s];
    int m = max_m < g ? max_m : g;
    const uint32_t* cs = counts + off[s];
    uint32_t* ms = meta + (int64_t)s * 4;
    uint32_t* out = out_ids + (int64_t)s * max_m;
    uint32_t thr = ms[0];
    uint32_t n_above = ms[1];
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < g;
         i += gridDim.x * blockDim.x) {
        uint32_t c = cs[i];
        if (c > thr) {
            uint32_t slot = atomicAdd(&ms[3], 1u);
            if ((int)slot < m) out[slot] = (uint32_t)i;
        } else if (c == thr) {
            uint32_t t = atomicAdd(&ms[2], 1u);
            uint32_t slot = n_above + t;
            if ((int)slot < m) out[slot] = (uint32_t)i;
        }
    }
}

extern "C" int32_t run_bdp_recall_batched(
    const uint32_t* q_signs,
    const uint64_t* sign_ptrs,
    const uint32_t* off,
    const uint32_t* cnt,
    uint32_t* counts,
    int32_t n_sess,
    int32_t n_heads,
    int32_t max_g,
    int32_t words,
    int32_t dim,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (n_sess <= 0 || max_g <= 0) return 0;
    int threads = 256;
    int bx = (max_g + threads - 1) / threads;
    if (bx > 256) bx = 256;
    dim3 blocks(bx, n_sess);
    size_t smem = (size_t)n_heads * words * sizeof(uint32_t);
    bdp_recall_batched_kernel<<<blocks, threads, smem, stream>>>(
        q_signs, sign_ptrs, off, cnt, counts, n_heads, words, dim);
    return (int32_t)cudaGetLastError();
}

extern "C" int32_t run_topm_select_batched(
    const uint32_t* counts,
    const uint32_t* off,
    const uint32_t* cnt,
    uint32_t* hist,    // [n_sess*bins] scratch (zeroed here)
    uint32_t* meta,    // [n_sess*4] scratch (zeroed here)
    uint32_t* out_ids, // [n_sess*max_m]
    int32_t n_sess,
    int32_t max_g,
    int32_t max_m,
    int32_t bins,
    void* stream_ptr
) {
    cudaStream_t stream = (cudaStream_t)stream_ptr;
    if (n_sess <= 0 || max_g <= 0 || max_m <= 0) return 0;
    cudaMemsetAsync(hist, 0, (size_t)n_sess * bins * sizeof(uint32_t), stream);
    cudaMemsetAsync(meta, 0, (size_t)n_sess * 4 * sizeof(uint32_t), stream);
    int threads = 256;
    int bx = (max_g + threads - 1) / threads;
    if (bx > 256) bx = 256;
    dim3 blocks(bx, n_sess);
    topm_hist_batched_kernel<<<blocks, threads, 0, stream>>>(
        counts, off, cnt, hist, bins);
    // One block per session, 256 threads cooperatively scanning the histogram
    // (chunk sums + block scan) — the shared buffer holds one u32 per thread.
    int thr_threads = 256;
    topm_threshold_batched_kernel<<<n_sess, thr_threads,
        (size_t)thr_threads * sizeof(uint32_t), stream>>>(
        hist, cnt, meta, bins, max_m);
    topm_compact_batched_kernel<<<blocks, threads, 0, stream>>>(
        counts, off, cnt, meta, out_ids, max_m);
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
