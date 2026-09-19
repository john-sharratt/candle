// =============================================================================
// FUSED MoE GATHER / WEIGHTED-SCATTER-ADD KERNELS
// =============================================================================
// These kernels replace the multi-op gather and scatter sequences in the
// grouped expert compute path.  Each collapses 2-6 separate kernel launches
// + intermediate allocations into a single batched kernel launch.
//
// Gather: out[i, j] = xs[token_ids[i], j]
//   Replaces: Tensor::new(token_ids) + xs.index_select
//
// Weighted scatter:
//   ys[token_ids[i], j] = Σ_i weights_flat[weight_ids[i]] * src[i, j]
//   Replaces: Tensor::new(weight_ids) + index_select + reshape + to_dtype
//             + broadcast_mul + index_add  (6 ops → 1 kernel)
//
// Both are batched across ALL experts in a single call (not per-expert).
// =============================================================================

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>

// =============================================================================
// FUSED GATHER KERNEL
// =============================================================================
// Grid: (total_rows, ceil(hidden_dim / BLOCK_SIZE))
// Each thread handles one element: out[row, col] = xs[token_ids[row], col]

template<typename T>
__device__ void moe_gather_impl(
    T*             __restrict__ out,
    const T*       __restrict__ xs,
    const uint32_t* __restrict__ token_ids,
    const size_t total_rows,
    const size_t hidden_dim
) {
    const unsigned int row = blockIdx.x;
    if (row >= total_rows) return;

    const uint32_t src_row = token_ids[row];
    // 0xFFFFFFFF marks a padding row (device-built tables launched at an upper
    // bound, see moe_bucketize.cu). Zero it rather than skip: downstream never
    // consumes the row's VALUES (grouped-GEMM padding tiles and scatter
    // segments both stop at the valid count), but a deterministic zero keeps
    // THIS stacked buffer initialized. Scope: the gather output only — the
    // grouped GEMM's own padding output rows stay unwritten (its padding tiles
    // exit before computing), so whole-pipeline byte-stability is not implied.
    if (src_row == 0xFFFFFFFFu) {
        T* dst = out + (size_t)row * hidden_dim;
        for (unsigned int col = blockIdx.y * blockDim.x + threadIdx.x;
             col < hidden_dim;
             col += blockDim.x * gridDim.y) {
            dst[col] = T(0);
        }
        return;
    }
    const T* src = xs + (size_t)src_row * hidden_dim;
    T* dst = out + (size_t)row * hidden_dim;

    for (unsigned int col = blockIdx.y * blockDim.x + threadIdx.x;
         col < hidden_dim;
         col += blockDim.x * gridDim.y) {
        dst[col] = src[col];
    }
}

extern "C" __global__ void moe_gather_bf16(
    __nv_bfloat16* out, const __nv_bfloat16* xs,
    const uint32_t* token_ids,
    size_t total_rows, size_t hidden_dim
) {
    moe_gather_impl(out, xs, token_ids, total_rows, hidden_dim);
}

extern "C" __global__ void moe_gather_f16(
    __half* out, const __half* xs,
    const uint32_t* token_ids,
    size_t total_rows, size_t hidden_dim
) {
    moe_gather_impl(out, xs, token_ids, total_rows, hidden_dim);
}

extern "C" __global__ void moe_gather_f32(
    float* out, const float* xs,
    const uint32_t* token_ids,
    size_t total_rows, size_t hidden_dim
) {
    moe_gather_impl(out, xs, token_ids, total_rows, hidden_dim);
}

// B3: TILE gather for pre-quantized q8a128 activations, so the experts consume the
// already-quantized FFN input directly — no gather-then-quantize.
//
// Tile-granular rather than row-granular because a row is not a byte range in general. The
// flat layout packs eight 128-element tiles to a 1152-byte super-block (qs de-interleaved from
// the per-tile ds slots — blocks.cuh), so a row is whole super-blocks only when hidden % 1024
// == 0; at 2560 a row is 20 tiles and straddles blocks. Output tile (r, t) is source tile
// (token_ids[r], t), copied quants and scale slot alike — which at hidden % 1024 == 0 is
// exactly the byte-row copy, and at every other multiple of 128 is the only correct one.
//
// One warp per output tile, grid-strided: the 32 lanes move the 128 quants as one int32 each
// (a tile's qs run is 128-byte aligned), lane 0 the 16-byte ds slot. A padding row
// (0xFFFFFFFF, see moe_gather_impl) is written as zeros, scale included.
#include "../blocks.cuh"

extern "C" __global__ void moe_gather_q8a128_tiles(
    uint8_t* __restrict__ out, const uint8_t* __restrict__ xs,
    const uint32_t* __restrict__ token_ids,
    size_t total_rows, size_t tiles_per_row
) {
    const int64_t total_tiles = (int64_t)total_rows * (int64_t)tiles_per_row;
    const int64_t total_warps = ((int64_t)gridDim.x * blockDim.x) >> 5;
    const int lane = threadIdx.x & 31;
    for (int64_t tile = ((int64_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
         tile < total_tiles; tile += total_warps) {
        const int64_t r = tile / (int64_t)tiles_per_row;
        const int64_t t = tile - r * (int64_t)tiles_per_row;
        const uint32_t src_row = token_ids[r];
        int32_t* dq = reinterpret_cast<int32_t*>(out + q8a1024_qs_off(tile));
        uint4* dds = reinterpret_cast<uint4*>(out + q8a1024_ds_off(tile));
        if (src_row == 0xFFFFFFFFu) {
            dq[lane] = 0;
            if (lane == 0) *dds = make_uint4(0u, 0u, 0u, 0u);
            continue;
        }
        const int64_t src = (int64_t)src_row * (int64_t)tiles_per_row + t;
        dq[lane] = reinterpret_cast<const int32_t*>(xs + q8a1024_qs_off(src))[lane];
        if (lane == 0) *dds = *reinterpret_cast<const uint4*>(xs + q8a1024_ds_off(src));
    }
}

// =============================================================================
// DETERMINISTIC SCATTER (no atomicAdd — one block per output token)
// =============================================================================
// Grid: (num_tokens, ceil(hidden_dim / BLOCK_SIZE))
//
// down_out is in expert-major order. perm[i] maps token-major index i to
// the corresponding row in down_out, so no CPU-side reorder pass is needed.
// The kernel gathers down_out[perm[idx]] directly, eliminating a separate
// index_select + contiguous pass that would otherwise reorder MBs of data.
//
// Since each output slot (ys[t]) is written by exactly ONE block (blockIdx.x==t),
// no atomicAdd is needed. Sequential F32 accumulation is fully deterministic.
//
// token_starts[t]   = start index in token-major space for token t
// token_starts[t+1] = end index (exclusive) for token t
// This is a prefix sum of per-token expert counts (variable k supported).
//
// **The kernel DEFINES `ys`; it does not accumulate into it.** The grid is
// (num_tokens, ceil(hidden/BLOCK)) and the column loop strides the whole row, so
// every (token, column) of the target is stored exactly once — which means the
// target must be allocated UNINITIALISED (hot-path invariant 6), never zeroed.
//
// It used to seed the reduction from `ys` itself, so every caller paid a memset
// over the combine target and this kernel paid a full read of it, per MoE layer
// per forward, to add a value that was always zero. That was load-bearing only
// for a two-call shape (hits, then misses) that no longer exists: the callers
// merged into one canonically-ordered pass because residency-dependent grouping
// made decode non-deterministic, and the seed outlived the reason for it.
//
// A token with no contributions (start == end) stores 0, which is what the
// zeroed buffer held — so the degenerate row is defined here rather than by the
// absent memset. The one case that is NOT covered is a caller skipping the
// launch entirely when nothing is routed; such a caller must zero its own
// target, and both do.

// **`down_out` is ALWAYS F32, whatever `ys` is.** The grouped int8 GEMM that
// produces it emits F32, and this kernel already accumulates in float — so it
// reads the producer's type directly and narrows once, at the store, into the
// type the residual stream wants. The alternative, converting the GEMM's whole
// output to `ys`'s type before the call, is a full-tensor pass per layer per
// forward to hand this loop a value it would immediately widen back to float
// (hot-path invariant 1: the kernel emits the final type).
//
// It also loses nothing numerically: the accumulation was always float, and the
// narrowing now happens after the sum rather than before it.

__device__ __forceinline__ void scatter_store(float* p, float v)         { *p = v; }
__device__ __forceinline__ void scatter_store(__half* p, float v)        { *p = __float2half(v); }
__device__ __forceinline__ void scatter_store(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }

template<typename YS>
__device__ __forceinline__ void deterministic_scatter_impl(
    YS* __restrict__ ys,
    const float* __restrict__ down_out,
    const uint32_t* __restrict__ perm,
    const float* __restrict__ weights_flat,
    const uint32_t* __restrict__ reordered_weight_ids,
    const int* __restrict__ token_starts,
    int num_tokens, int hidden
) {
    const int t = (int)blockIdx.x;
    if (t >= num_tokens) return;
    const int start = token_starts[t];
    const int end   = token_starts[t + 1];
    YS* dst = ys + (size_t)t * hidden;
    for (int col = (int)(blockIdx.y * blockDim.x + threadIdx.x);
         col < hidden;
         col += (int)(blockDim.x * gridDim.y)) {
        float sum = 0.f;
        for (int idx = start; idx < end; idx++) {
            float w = weights_flat[reordered_weight_ids[idx]];
            sum += w * down_out[(size_t)perm[idx] * hidden + col];
        }
        scatter_store(&dst[col], sum);
    }
}

extern "C" __global__ void deterministic_scatter_bf16(
    __nv_bfloat16* __restrict__ ys,
    const float* __restrict__ down_out,
    const uint32_t* __restrict__ perm,
    const float* __restrict__ weights_flat,
    const uint32_t* __restrict__ reordered_weight_ids,
    const int* __restrict__ token_starts,
    int num_tokens, int hidden
) {
    deterministic_scatter_impl(ys, down_out, perm, weights_flat,
                               reordered_weight_ids, token_starts, num_tokens, hidden);
}

extern "C" __global__ void deterministic_scatter_f16(
    __half* __restrict__ ys,
    const float* __restrict__ down_out,
    const uint32_t* __restrict__ perm,
    const float* __restrict__ weights_flat,
    const uint32_t* __restrict__ reordered_weight_ids,
    const int* __restrict__ token_starts,
    int num_tokens, int hidden
) {
    deterministic_scatter_impl(ys, down_out, perm, weights_flat,
                               reordered_weight_ids, token_starts, num_tokens, hidden);
}

extern "C" __global__ void deterministic_scatter_f32(
    float* __restrict__ ys,
    const float* __restrict__ down_out,
    const uint32_t* __restrict__ perm,
    const float* __restrict__ weights_flat,
    const uint32_t* __restrict__ reordered_weight_ids,
    const int* __restrict__ token_starts,
    int num_tokens, int hidden
) {
    deterministic_scatter_impl(ys, down_out, perm, weights_flat,
                               reordered_weight_ids, token_starts, num_tokens, hidden);
}

// =============================================================================
// FUSED MoE ROUTE (softmax → top-k select → renormalize, in one launch)
// =============================================================================
// Replaces the routing op chain `softmax_last_dim → to_f32 → sort_last_dim →
// narrow(top-k) → (renorm) → flatten` (≈6 kernel launches over a tiny
// [num_tokens, n_experts] tensor) with a single kernel.
//
// Two facts collapse the work to one pass over the experts:
//   * softmax is monotonic, so the top-k of softmax(logits) == top-k of logits
//     (the full sort over softmax outputs is wasted) — we select on raw logits;
//   * with `norm_topk`, renormalizing the selected softmax weights cancels the
//     global denominator exactly:
//         renorm(softmax_all(l)[topk]) = exp(l_i) / Σ_{j∈topk} exp(l_j)
//     i.e. a softmax over just the k selected logits — the 128-wide softmax
//     denominator is never needed. Without `norm_topk`, the weights are the
//     plain full softmax values, so we accumulate the global Σexp in the same
//     pass.
//
// **One warp owns one token.** The 32 lanes cooperatively load the experts (lane `l` holds
// experts `l, l+32, l+64, …`, ≤ MOE_ROUTE_MAX_SLOTS of them) — coalesced and read **once** into
// registers — then warp-shuffle reductions do max / Σexp / top-k. This hides memory latency
// (32 in-flight loads instead of one serial thread) and never re-reads global memory. Output is
// top-k **indices** (u32) and **weights** (f32) in descending-logit order, matching the sort path
// (`sort_last_dim(descending)` → `narrow(0, k)`); ties resolve to the lowest expert index.
#define MOE_ROUTE_MAX_K 16
// Experts-per-lane is a TEMPLATE parameter (`SLOTS`), not this constant: 8
// slots serve up to 32·8 = 256 experts (the qwen3/3.5 lineage and DeepSeek),
// 16 serve 512 (qwen4exp). Per instantiation, so the narrow variant's
// register pressure is untouched by the wide one's existence.
#define MOE_ROUTE_MAX_SLOTS 8

template<typename T>
__device__ __forceinline__ float moe_route_to_f32(T x);
template<> __device__ __forceinline__ float moe_route_to_f32<float>(float x) { return x; }
template<> __device__ __forceinline__ float moe_route_to_f32<__half>(__half x) { return __half2float(x); }
template<> __device__ __forceinline__ float moe_route_to_f32<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }

template<typename T, int SLOTS>
__device__ void moe_route_impl(
    const T*       __restrict__ logits,      // [num_tokens, n_experts]
    uint32_t*      __restrict__ out_idx,     // [num_tokens, k]
    float*         __restrict__ out_weights, // [num_tokens, k]
    int num_tokens, int n_experts, int k, int norm_topk
) {
    const unsigned FULL = 0xffffffffu;
    const int lane  = (int)(threadIdx.x & 31);
    const int token = (int)((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    if (token >= num_tokens) return;  // whole warp shares `token`, so it exits together
    const T* row = logits + (size_t)token * (size_t)n_experts;

    // Coalesced single read of this lane's experts into registers (−inf pads the tail).
    float v[SLOTS];
    #pragma unroll
    for (int j = 0; j < SLOTS; ++j) {
        int e = lane + 32 * j;
        v[j] = (e < n_experts) ? moe_route_to_f32<T>(row[e]) : -INFINITY;
    }

    // Global max (warp reduction over the per-lane local max).
    float gmax = -INFINITY;
    #pragma unroll
    for (int j = 0; j < SLOTS; ++j) gmax = fmaxf(gmax, v[j]);
    for (int off = 16; off > 0; off >>= 1) gmax = fmaxf(gmax, __shfl_xor_sync(FULL, gmax, off));

    // Full Σ exp(l − gmax) — only needed for the un-renormalized softmax weights.
    float z_all = 0.f;
    if (!norm_topk) {
        float ls = 0.f;
        #pragma unroll
        for (int j = 0; j < SLOTS; ++j) {
            if (v[j] > -INFINITY) ls += __expf(v[j] - gmax);
        }
        for (int off = 16; off > 0; off >>= 1) ls += __shfl_xor_sync(FULL, ls, off);
        z_all = ls;
    }

    // Top-k: k rounds of warp argmax (max value, lowest index on tie). The winning lane masks
    // its own slot to −inf so the next round skips it. Results accumulate on lane 0.
    float sel_w[MOE_ROUTE_MAX_K];
    int   sel_i[MOE_ROUTE_MAX_K];
    float z_top = 0.f;
    for (int p = 0; p < k; ++p) {
        float bv = -INFINITY;
        int   bi = n_experts;          // sentinel > any valid index
        #pragma unroll
        for (int j = 0; j < SLOTS; ++j) {
            int e = lane + 32 * j;
            if (e < n_experts && v[j] > bv) { bv = v[j]; bi = e; }
        }
        for (int off = 16; off > 0; off >>= 1) {
            float obv = __shfl_xor_sync(FULL, bv, off);
            int   obi = __shfl_xor_sync(FULL, bi, off);
            if (obv > bv || (obv == bv && obi < bi)) { bv = obv; bi = obi; }
        }
        float ev = __expf(bv - gmax);
        z_top += ev;                   // identical on every lane (all share bv)
        if (lane == 0) { sel_w[p] = ev; sel_i[p] = bi; }
        // Owner masks the winner — but only when a winner was found. `bi` seeds
        // at the `n_experts` sentinel, and when no lane holds a finite candidate
        // it survives to here, where `v[bi >> 5]` is `v[n_experts >> 5]`. At
        // `n_experts == 32 * SLOTS` — 512 on the x512 instantiation, 256 on the
        // narrow one — that is exactly `v[SLOTS]`, one past the end of the
        // per-lane array. The store lands on whatever the compiler placed after
        // it, silently, on the degenerate-routing path the output clamp below is
        // already written to survive. Below that width the index is in bounds but
        // still masks a slot no round asked for, so the guard is on the sentinel
        // rather than on the width.
        if (bi < n_experts && lane == (bi & 31)) v[bi >> 5] = -INFINITY;
    }

    if (lane == 0) {
        const float denom = norm_topk ? z_top : z_all;
        const float inv = denom > 0.f ? (1.f / denom) : 0.f;
        uint32_t* oi = out_idx + (size_t)token * (size_t)k;
        float*    ow = out_weights + (size_t)token * (size_t)k;
        for (int p = 0; p < k; ++p) {
            // `bi` seeds at `n_experts` as a "not found" sentinel. It survives to
            // `sel_i[p]` only when this slot had no finite candidate — a
            // degenerate token whose remaining logits are all -inf/NaN (NaN loses
            // every `>` compare), or fewer finite experts than k. Emit a valid
            // index with zero weight so the downstream gather/scatter and the
            // expert-paging pipeline never index out of bounds on the sentinel.
            if (sel_i[p] >= n_experts) {
                oi[p] = 0u;
                ow[p] = 0.f;
            } else {
                oi[p] = (uint32_t)sel_i[p];
                ow[p] = sel_w[p] * inv;
            }
        }
    }
}

extern "C" __global__ void moe_route_f32(
    const float* logits, uint32_t* out_idx, float* out_weights,
    int num_tokens, int n_experts, int k, int norm_topk
) {
    moe_route_impl<float, 8>(logits, out_idx, out_weights, num_tokens, n_experts, k, norm_topk);
}

extern "C" __global__ void moe_route_f16(
    const __half* logits, uint32_t* out_idx, float* out_weights,
    int num_tokens, int n_experts, int k, int norm_topk
) {
    moe_route_impl<__half, 8>(logits, out_idx, out_weights, num_tokens, n_experts, k, norm_topk);
}

extern "C" __global__ void moe_route_bf16(
    const __nv_bfloat16* logits, uint32_t* out_idx, float* out_weights,
    int num_tokens, int n_experts, int k, int norm_topk
) {
    moe_route_impl<__nv_bfloat16, 8>(logits, out_idx, out_weights, num_tokens, n_experts, k, norm_topk);
}

// The 512-expert instantiations (16 slots per lane) — qwen4exp's router.
extern "C" __global__ void moe_route_f32_x512(
    const float* logits, uint32_t* out_idx, float* out_weights,
    int num_tokens, int n_experts, int k, int norm_topk
) {
    moe_route_impl<float, 16>(logits, out_idx, out_weights, num_tokens, n_experts, k, norm_topk);
}

extern "C" __global__ void moe_route_f16_x512(
    const __half* logits, uint32_t* out_idx, float* out_weights,
    int num_tokens, int n_experts, int k, int norm_topk
) {
    moe_route_impl<__half, 16>(logits, out_idx, out_weights, num_tokens, n_experts, k, norm_topk);
}

extern "C" __global__ void moe_route_bf16_x512(
    const __nv_bfloat16* logits, uint32_t* out_idx, float* out_weights,
    int num_tokens, int n_experts, int k, int norm_topk
) {
    moe_route_impl<__nv_bfloat16, 16>(logits, out_idx, out_weights, num_tokens, n_experts, k, norm_topk);
}
